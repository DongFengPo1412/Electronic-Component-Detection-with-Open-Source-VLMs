"""CPU post-lock S10 evaluation for R15 (A0) vs R19 A1 (IC-only reject) + 6-point acceptance gate."""

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
    compute_iou_xyxy,
    evaluate_composite_board,
    load_coco_gts_for_single_board,
)
from pcb_vlm.evaluation.crop_verifier_r18 import extract_square_context_crop  # noqa: E402

R22_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration22-20261125")
LOCK_FILE = R22_ART / "eval" / "r22_s10_pre_gt_lock.json"
PROTO_FILE = R22_ART / "provenance" / "r22_s10_provenance_and_protocol_frozen.json"
S10_COCO_GT = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s10_front_official_coco.json")
NON_IC_FIVE_CLASSES = ("capacitor", "resistor", "inductor", "diode", "transistor")


def _sig_five(preds: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Tuple[float, ...], float]]:
    return [
        (str(p["label"]).lower(), tuple(float(v) for v in p["bbox_xyxy"]), float(p["score"]))
        for p in preds
        if str(p.get("label") or "").lower() in NON_IC_FIVE_CLASSES
    ]


def _best_gt(box: Sequence[float], gts: Sequence[Mapping[str, Any]]) -> Tuple[str, float]:
    bi, bh = "BG", 0.0
    for g in gts:
        iou = compute_iou_xyxy(box, g["bbox_xyxy"])
        if iou > bh:
            bh, bi = iou, str(g["label"])
    return (bi if bh >= 0.50 else "BG"), round(bh, 4)


def _render_dropped(cases: Sequence[Mapping[str, Any]], out_path: pathlib.Path) -> None:
    cols = max(1, min(4, len(cases)))
    rows = max(1, (len(cases) + cols - 1) // cols)
    cell_w, cell_h = 240, 265
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 28, 34))
    draw = ImageDraw.Draw(canvas)
    for idx, c in enumerate(cases):
        r, col = divmod(idx, cols)
        x0, y0 = col * cell_w, r * cell_h
        with Image.open(c["tif_path"]) as im:
            crop = extract_square_context_crop(im, c["bbox_xyxy"], target_size=224, context_scale=1.60)
        cd = ImageDraw.Draw(crop)
        color = (80, 220, 120) if c["gt_match"] != "ic" else (240, 90, 90)
        cd.rectangle([42, 42, 182, 182], outline=color, width=2)
        canvas.paste(crop, (x0 + 8, y0 + 36))
        draw.text((x0 + 8, y0 + 4), f"s10 DROP IC r15={c['r15_score']:.2f}", fill=(240, 240, 240))
        draw.text((x0 + 8, y0 + 18), f"GT={c['gt_match']} p_other={c['top1_prob']:.2f}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    s = socket.socket()
    s.settimeout(1.0)
    rc = s.connect_ex(("127.0.0.1", 8826))
    s.close()
    assert rc != 0, "Port 8826 must be closed before post-GT evaluation!"

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

    gts, _ = load_coco_gts_for_single_board(S10_COCO_GT, image_id=None)
    s10_dir = R22_ART / "live_demo" / "s10"
    a0 = json.loads((s10_dir / "r19_a0_predictions.json").read_text(encoding="utf-8"))
    a1 = json.loads((s10_dir / "r19_a1_predictions.json").read_text(encoding="utf-8"))
    pool_005 = json.loads((s10_dir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
    audit = json.loads((s10_dir / "r19_postproc_audit.json").read_text(encoding="utf-8"))
    roi = lock["s10"]["roi_bbox_half_open"]

    five_identical = (_sig_five(a0) == _sig_five(a1))
    assert five_identical

    ev_a0 = evaluate_composite_board(gts=gts, predictions_025=a0, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
    ev_a1 = evaluate_composite_board(gts=gts, predictions_025=a1, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)

    dropped_audit = []
    for d in audit["a1_dropped_ic_boxes"]:
        gt_lbl, gt_iou = _best_gt(d["bbox_xyxy"], gts)
        dropped_audit.append({**d, "gt_match": gt_lbl, "gt_iou": gt_iou, "tif_path": lock["s10"]["tif_path"]})

    (R22_ART / "eval" / "r22_s10_dropped_ic_audit.json").write_text(
        json.dumps(dropped_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if dropped_audit:
        _render_dropped(dropped_audit, R22_ART / "eval" / "r22_s10_dropped_ic_cases.png")

    ic_tp_a0 = ev_a0["per_class"]["ic"]["tp"]
    ic_fp_a0 = ev_a0["per_class"]["ic"]["fp"]
    ic_tp_a1 = ev_a1["per_class"]["ic"]["tp"]
    ic_fp_a1 = ev_a1["per_class"]["ic"]["fp"]
    ic_fp_red = ic_fp_a0 - ic_fp_a1
    ic_fp_red_pct = round(100.0 * ic_fp_red / max(1, ic_fp_a0), 2)
    f1_a0 = ev_a0["class_aware"]["f1_pct"]
    f1_a1 = ev_a1["class_aware"]["f1_pct"]
    f1_gain_pp = round(f1_a1 - f1_a0, 2)

    checks = {
        "s10_r15_ic_tp_ge_1": ic_tp_a0 >= 1,
        "s10_a1_ic_tp_preserved": ic_tp_a1 == ic_tp_a0,
        "s10_ic_fp_reduction_ge_2_and_ge_20pct": (ic_fp_red >= 2 and ic_fp_red_pct >= 20.0),
        "s10_f1_gain_ge_2_0_pp": f1_gain_pp >= 2.0,
        "other_5_classes_100pct_identical": five_identical,
        "zero_lock_and_protocol_diffs": (lock_mismatches == 0 and proto_diffs == 0),
    }
    gate_passed = all(checks.values())

    report = {
        "schema_version": "r22_s10_eval_report_v1",
        "port_8826_open": False,
        "pre_gt_lock_mismatches": lock_mismatches,
        "frozen_protocol_diffs": proto_diffs,
        "gt_total": len(gts),
        "A0_r15": ev_a0,
        "A1_r19": ev_a1,
        "comparison": {
            "f1_a0_pct": f1_a0,
            "f1_a1_pct": f1_a1,
            "f1_gain_pp": f1_gain_pp,
            "ic_tp_a0": ic_tp_a0,
            "ic_tp_a1": ic_tp_a1,
            "ic_fp_a0": ic_fp_a0,
            "ic_fp_a1": ic_fp_a1,
            "ic_fp_reduction": ic_fp_red,
            "ic_fp_reduction_pct": ic_fp_red_pct,
            "ic_dropped_count": len(dropped_audit),
        },
        "s10_a1_acceptance_gate": {
            "passed": gate_passed,
            "checks": checks,
        },
    }
    (R22_ART / "eval" / "r22_s10_eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "f1_a0": f1_a0, "f1_a1": f1_a1, "f1_gain_pp": f1_gain_pp,
        "tp_fp_fn_a0": [ev_a0["class_aware"]["tp"], ev_a0["class_aware"]["fp"], ev_a0["class_aware"]["fn"]],
        "tp_fp_fn_a1": [ev_a1["class_aware"]["tp"], ev_a1["class_aware"]["fp"], ev_a1["class_aware"]["fn"]],
        "ic_tp": f"{ic_tp_a0}->{ic_tp_a1}",
        "ic_fp": f"{ic_fp_a0}->{ic_fp_a1} (-{ic_fp_red}, -{ic_fp_red_pct}%)",
        "gate_passed": gate_passed,
        "checks": checks,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
