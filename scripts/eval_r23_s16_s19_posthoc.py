"""CPU post-lock S16+S19 evaluation for R15 (A0) vs R19 A1 (IC-only reject) + 6-point admission gate."""

from __future__ import annotations

import json
import pathlib
import socket
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    _prf1,
    compute_iou_xyxy,
    evaluate_composite_board,
    load_coco_gts_for_single_board,
)
from pcb_vlm.evaluation.crop_verifier_r18 import extract_square_context_crop  # noqa: E402

R23_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration23-20261125")
LOCK_FILE = R23_ART / "eval" / "r23_s16_s19_pre_gt_lock.json"
PROTO_FILE = R23_ART / "provenance" / "r23_s16_s19_provenance_and_protocol_frozen.json"
COCOS = {
    "s16": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-heldout-evaluation-20261116\coco\s16_front_coco.json"),
    "s19": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-heldout-evaluation-20261116\coco\s19_front_coco.json"),
}
NON_IC_FIVE = ("capacitor", "resistor", "inductor", "diode", "transistor")


def _sig_five(preds: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Tuple[float, ...], float]]:
    return [
        (str(p["label"]).lower(), tuple(float(v) for v in p["bbox_xyxy"]), float(p["score"]))
        for p in preds
        if str(p.get("label") or "").lower() in NON_IC_FIVE
    ]


def _matched_ic_tp_ids(preds: Sequence[Mapping[str, Any]], gts: Sequence[Mapping[str, Any]]) -> List[str]:
    ic_preds = sorted(
        [p for p in preds if str(p.get("label") or "").lower() == "ic"],
        key=lambda p: (-float(p.get("score", 0.0)), str(p.get("prediction_id") or "")),
    )
    ic_gts = [g for g in gts if str(g.get("label") or "").lower() == "ic"]
    used_gt = set()
    tp_ids: List[str] = []
    for p in ic_preds:
        bi, bh = -1, 0.0
        for gi, g in enumerate(ic_gts):
            if gi in used_gt:
                continue
            iou = compute_iou_xyxy(p["bbox_xyxy"], g["bbox_xyxy"])
            if iou > bh:
                bh, bi = iou, gi
        if bi >= 0 and bh >= 0.50:
            used_gt.add(bi)
            tp_ids.append(str(p.get("prediction_id") or ""))
    return tp_ids


def _max_iou_any_gt(box: Sequence[float], gts: Sequence[Mapping[str, Any]]) -> Tuple[str, float]:
    bi, bh = "BG", 0.0
    for g in gts:
        iou = compute_iou_xyxy(box, g["bbox_xyxy"])
        if iou > bh:
            bh, bi = iou, str(g["label"])
    return (bi if bh >= 0.50 else "BG"), round(bh, 4)


def _aggregate(evals: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tp = sum(e["class_aware"]["tp"] for e in evals)
    fp = sum(e["class_aware"]["fp"] for e in evals)
    fn = sum(e["class_aware"]["fn"] for e in evals)
    overall = _prf1(tp, fp, fn)
    per_class = {}
    for cls in ("capacitor", "resistor", "inductor", "transistor", "diode", "ic"):
        ctp = sum(e["per_class"][cls]["tp"] for e in evals)
        cfp = sum(e["per_class"][cls]["fp"] for e in evals)
        cfn = sum(e["per_class"][cls]["fn"] for e in evals)
        per_class[cls] = {"tp": ctp, "fp": cfp, "fn": cfn, **_prf1(ctp, cfp, cfn)}
    return {"tp": tp, "fp": fp, "fn": fn, **overall, "per_class": per_class}


def _render_dropped(cases: Sequence[Mapping[str, Any]], out_path: pathlib.Path) -> None:
    cols = 4
    sample = list(cases[:12])
    rows = max(1, (len(sample) + cols - 1) // cols)
    cell_w, cell_h = 240, 265
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 28, 34))
    draw = ImageDraw.Draw(canvas)
    for idx, c in enumerate(sample):
        r, col = divmod(idx, cols)
        x0, y0 = col * cell_w, r * cell_h
        with Image.open(c["tif_path"]) as im:
            crop = extract_square_context_crop(im, c["bbox_xyxy"], target_size=224, context_scale=1.60)
        cd = ImageDraw.Draw(crop)
        color = (80, 220, 120) if c["gt_match_iou50"] != "ic" else (240, 90, 90)
        cd.rectangle([42, 42, 182, 182], outline=color, width=2)
        canvas.paste(crop, (x0 + 8, y0 + 36))
        draw.text((x0 + 8, y0 + 4), f"{c['board']} DROP IC r15={c['r15_score']:.2f}", fill=(240, 240, 240))
        draw.text((x0 + 8, y0 + 18), f"GT={c['gt_match_iou50']} maxIoU={c['max_iou_any_gt']:.2f}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    s = socket.socket()
    s.settimeout(1.0)
    rc = s.connect_ex(("127.0.0.1", 8827))
    s.close()
    assert rc != 0, "Port 8827 must be closed before post-GT evaluation!"

    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    lock_mismatches = sum(
        1 for it in lock["locked_files"]
        if not pathlib.Path(it["path"]).is_file() or file_sha256(pathlib.Path(it["path"])).lower() != it["sha256"].lower()
    )
    assert lock_mismatches == 0, f"Pre-GT lock mismatches: {lock_mismatches}"

    proto = json.loads(PROTO_FILE.read_text(encoding="utf-8"))
    proto_diffs = sum(
        1 for it in proto["frozen_inputs"]
        if not pathlib.Path(it["path"]).is_file() or file_sha256(pathlib.Path(it["path"])).lower() != it["sha256"].lower()
    )
    assert proto_diffs == 0, f"Protocol diffs: {proto_diffs}"

    per_board: Dict[str, Any] = {}
    dropped_all: List[Dict[str, Any]] = []
    evals_a0, evals_a1 = [], []

    for bkey in ("s16", "s19"):
        binfo = lock["boards"][bkey]
        bdir = pathlib.Path(binfo["artifacts_dir"])
        gts, _ = load_coco_gts_for_single_board(COCOS[bkey], image_id=None)
        roi = binfo["roi_bbox_half_open"]
        a0 = json.loads((bdir / "r19_a0_predictions.json").read_text(encoding="utf-8"))
        a1 = json.loads((bdir / "r19_a1_predictions.json").read_text(encoding="utf-8"))
        pool_005 = json.loads((bdir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        audit = json.loads((bdir / "r19_postproc_audit.json").read_text(encoding="utf-8"))

        five_id = (_sig_five(a0) == _sig_five(a1))
        assert five_id

        ev_a0 = evaluate_composite_board(gts=gts, predictions_025=a0, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
        ev_a1 = evaluate_composite_board(gts=gts, predictions_025=a1, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
        evals_a0.append(ev_a0)
        evals_a1.append(ev_a1)

        tp_ids_a0 = _matched_ic_tp_ids(a0, gts)
        tp_ids_a1 = _matched_ic_tp_ids(a1, gts)

        dropped_b = []
        for d in audit["a1_dropped_ic_boxes"]:
            gt_lbl, max_iou = _max_iou_any_gt(d["bbox_xyxy"], gts)
            rec = {
                "board": bkey,
                "prediction_id": d["prediction_id"],
                "r15_score": d["r15_score"],
                "top1_class": d["top1_class"],
                "top1_prob": d["top1_prob"],
                "margin": d["margin"],
                "gt_match_iou50": gt_lbl,
                "max_iou_any_gt": max_iou,
                "bbox_xyxy": d["bbox_xyxy"],
                "tif_path": binfo["tif_path"],
                "probs": d["probs"],
            }
            dropped_b.append(rec)
            dropped_all.append(rec)

        per_board[bkey] = {
            "gt_total": len(gts),
            "other_5_classes_100pct_identical": five_id,
            "a0_ic_tp_ids": tp_ids_a0,
            "a1_ic_tp_ids": tp_ids_a1,
            "ic_tp_ids_preserved": (tp_ids_a0 == tp_ids_a1),
            "dropped_ic_count": len(dropped_b),
            "A0": ev_a0,
            "A1": ev_a1,
        }

    comb_a0 = _aggregate(evals_a0)
    comb_a1 = _aggregate(evals_a1)
    ic_fp_a0 = comb_a0["per_class"]["ic"]["fp"]
    ic_fp_a1 = comb_a1["per_class"]["ic"]["fp"]
    ic_fp_red = ic_fp_a0 - ic_fp_a1
    ic_fp_red_pct = round(100.0 * ic_fp_red / max(1, ic_fp_a0), 2)

    checks = {
        "both_boards_r15_ic_tp_ge_1": all(per_board[b]["A0"]["per_class"]["ic"]["tp"] >= 1 for b in ("s16", "s19")),
        "both_boards_a1_preserves_100pct_ic_tp_and_ids": all(per_board[b]["ic_tp_ids_preserved"] for b in ("s16", "s19")),
        "both_boards_a1_total_tp_ge_a0_and_f1_ge_a0": all(
            per_board[b]["A1"]["class_aware"]["tp"] >= per_board[b]["A0"]["class_aware"]["tp"]
            and per_board[b]["A1"]["class_aware"]["f1_pct"] >= per_board[b]["A0"]["class_aware"]["f1_pct"]
            for b in ("s16", "s19")
        ),
        "combined_ic_fp_reduction_ge_2_and_ge_20pct": (ic_fp_red >= 2 and ic_fp_red_pct >= 20.0),
        "both_boards_other_5_classes_100pct_identical": all(per_board[b]["other_5_classes_100pct_identical"] for b in ("s16", "s19")),
        "zero_lock_and_protocol_diffs_and_gpu_le_400s": (lock_mismatches == 0 and proto_diffs == 0 and lock["gpu_wall_seconds_total"] <= 400.0),
    }
    gate_passed = all(checks.values())

    (R23_ART / "eval" / "r23_s16_s19_dropped_ic_audit.json").write_text(
        json.dumps(dropped_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if dropped_all:
        _render_dropped(dropped_all, R23_ART / "eval" / "r23_s16_s19_dropped_ic_cases.png")

    report = {
        "schema_version": "r23_s16_s19_eval_report_v1",
        "port_8827_open": False,
        "pre_gt_lock_mismatches": lock_mismatches,
        "frozen_protocol_diffs": proto_diffs,
        "gpu_wall_seconds_total": lock["gpu_wall_seconds_total"],
        "combined": {
            "A0": comb_a0,
            "A1": comb_a1,
            "f1_gain_pp": round(comb_a1["f1_pct"] - comb_a0["f1_pct"], 2),
            "ic_fp_reduction": ic_fp_red,
            "ic_fp_reduction_pct": ic_fp_red_pct,
        },
        "per_board": per_board,
        "r23_admission_gate": {"passed": gate_passed, "checks": checks},
    }
    (R23_ART / "eval" / "r23_s16_s19_eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "s16_A0_f1": per_board["s16"]["A0"]["class_aware"]["f1_pct"],
        "s16_A1_f1": per_board["s16"]["A1"]["class_aware"]["f1_pct"],
        "s16_ic_tp": f"{per_board['s16']['A0']['per_class']['ic']['tp']}->{per_board['s16']['A1']['per_class']['ic']['tp']}",
        "s16_ic_fp": f"{per_board['s16']['A0']['per_class']['ic']['fp']}->{per_board['s16']['A1']['per_class']['ic']['fp']}",
        "s19_A0_f1": per_board["s19"]["A0"]["class_aware"]["f1_pct"],
        "s19_A1_f1": per_board["s19"]["A1"]["class_aware"]["f1_pct"],
        "s19_ic_tp": f"{per_board['s19']['A0']['per_class']['ic']['tp']}->{per_board['s19']['A1']['per_class']['ic']['tp']}",
        "s19_ic_fp": f"{per_board['s19']['A0']['per_class']['ic']['fp']}->{per_board['s19']['A1']['per_class']['ic']['fp']}",
        "comb_f1": f"{comb_a0['f1_pct']}->{comb_a1['f1_pct']} (+{round(comb_a1['f1_pct']-comb_a0['f1_pct'],2)} pp)",
        "comb_ic_fp": f"{ic_fp_a0}->{ic_fp_a1} (-{ic_fp_red}, -{ic_fp_red_pct}%)",
        "gate_passed": gate_passed,
        "checks": checks,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
