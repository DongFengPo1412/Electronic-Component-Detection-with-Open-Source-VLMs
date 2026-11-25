"""Pure CPU post-GT paired evaluation, auxiliary 0.05-0.25 D/Q diagnostic & visual gallery for Round 18."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import evaluate as repo_eval  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    _prf1,
    box_in_composite_region,
    evaluate_composite_board,
    load_coco_gts_for_single_board,
)
from pcb_vlm.evaluation.crop_verifier_r18 import (  # noqa: E402
    R18_TARGET_CLASSES,
    R18_UNTOUCHED_CLASSES,
    apply_r18_verifier_to_predictions,
    load_r18_verifier_checkpoint,
)

DEFAULT_LOCK_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\eval\r18_pre_gt_lock.json")
DEFAULT_INVENTORY = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json")
DEFAULT_CAL_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\calibration\r18_val_abstention_calibration.json")
DEFAULT_VERIFIER_WEIGHTS = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\weights\r18_resnet18_crop_verifier_best.pt")
DEFAULT_OUTPUT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\eval\r18_paired_eval_report.json")
DEFAULT_GALLERY = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\eval\r18_verifier_before_after_cases.png")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    ia = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if ia <= 0.0:
        return 0.0
    aa = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return ia / max(1e-9, aa + ab - ia)


def _render_cases_gallery(cases: Sequence[Mapping[str, Any]], out_path: pathlib.Path) -> None:
    cols, cw, ch, pad = 4, 300, 320, 12
    rows = max(1, (len(cases) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad) + pad), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    cache: Dict[str, Image.Image] = {}
    try:
        for idx, c in enumerate(cases[:12]):
            r, col_i = idx // cols, idx % cols
            ox, oy = pad + col_i * (cw + pad), pad + r * (ch + pad)
            tp = str(c["tif_path"])
            if tp not in cache:
                cache[tp] = Image.open(tp).convert("RGB")
            im = cache[tp]
            w, h = im.size
            x1, y1, x2, y2 = c["bbox_xyxy"]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            half = max(32.0, max(x2 - x1, y2 - y1) * 1.15)
            rx1, ry1 = max(0, int(cx - half)), max(0, int(cy - half))
            rx2, ry2 = min(w, int(cx + half)), min(h, int(cy + half))
            crop = im.crop((rx1, ry1, rx2, ry2))
            sc = min(cw / max(1, crop.width), (ch - 36) / max(1, crop.height))
            res = crop.resize((max(1, int(crop.width * sc)), max(1, int(crop.height * sc))), Image.Resampling.BILINEAR)
            d2 = ImageDraw.Draw(res)
            bx1, by1 = (x1 - rx1) * sc, (y1 - ry1) * sc
            bx2, by2 = (x2 - rx1) * sc, (y2 - ry1) * sc
            color = (34, 197, 94) if c["decision"] == "relabel" else ((248, 113, 113) if c["decision"] == "reject_other" else (250, 204, 21))
            d2.rectangle([bx1, by1, bx2, by2], outline=color, width=3)
            draw.text((ox, oy), f"{c['board']} {c['orig']}->{c['new']} ({c['decision']})", fill=(241, 245, 249), font=font)
            draw.text((ox, oy + 14), f"GT={c['gt_match']} p={c['top1_prob']:.2f} r15={c['r15_score']:.2f}", fill=(148, 163, 184), font=font)
            canvas.paste(res, (ox, oy + 32))
    finally:
        for im in cache.values():
            im.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate locked R18 predictions against COCO GT (CPU only).")
    parser.add_argument("--lock-file", type=pathlib.Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--inventory", type=pathlib.Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--cal-file", type=pathlib.Path, default=DEFAULT_CAL_FILE)
    parser.add_argument("--verifier-weights", type=pathlib.Path, default=DEFAULT_VERIFIER_WEIGHTS)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gallery", type=pathlib.Path, default=DEFAULT_GALLERY)
    args = parser.parse_args(argv)

    lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
    mismatches: List[Dict[str, str]] = []
    for lf in lock["locked_files"]:
        lp = pathlib.Path(str(lf["path"]))
        if not lp.is_file() or _sha256_file(lp) != str(lf["sha256"]).lower():
            mismatches.append({"role": lf["role"], "path": str(lp)})
    if mismatches:
        raise RuntimeError(f"Pre-GT lock verification failed: {mismatches}")

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    coco_by_bkey = {
        b["physical_board"]: pathlib.Path(str(b["coco_path"]))
        for b in inv["boards"]
        if b.get("coco_path") and b["physical_board"] in ("s20", "s4", "s1")
    }
    cal = json.loads(args.cal_file.read_text(encoding="utf-8"))["selected"]
    cv_model, _, cv_sha = load_r18_verifier_checkpoint(args.verifier_weights, device=torch.device("cpu"))

    per_board: Dict[str, Any] = {}
    aux_lowconf_study: Dict[str, Any] = {}
    gallery_cases: List[Dict[str, Any]] = []
    non_target_invariance_ok = True

    for bkey, binfo in lock["boards"].items():
        gts, _ = load_coco_gts_for_single_board(coco_by_bkey[bkey], image_id=None)
        a_dir = pathlib.Path(str(binfo["artifacts_dir"]))
        r15_preds = json.loads((a_dir / "yolo_official_025_frozen_r15.json").read_text(encoding="utf-8"))
        r18_preds = json.loads((a_dir / "yolo_r18_verified_025.json").read_text(encoding="utf-8"))
        pool_005 = json.loads((a_dir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        decisions = json.loads((a_dir / "r18_crop_verifier_decisions.json").read_text(encoding="utf-8"))
        roi = binfo["roi_bbox_half_open"]

        # Verify capacitor/resistor/inductor bit-for-bit invariance
        r15_nt = [(p["label"], p["score"], p["bbox_xyxy"]) for p in r15_preds if p["label"] in R18_UNTOUCHED_CLASSES]
        r18_nt = [(p["label"], p["score"], p["bbox_xyxy"]) for p in r18_preds if p["label"] in R18_UNTOUCHED_CLASSES]
        if r15_nt != r18_nt:
            non_target_invariance_ok = False

        ev_r15 = evaluate_composite_board(gts=gts, predictions_025=r15_preds, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
        ev_r18 = evaluate_composite_board(gts=gts, predictions_025=r18_preds, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)

        # Separate auxiliary 0.05-0.25 D/Q pool study (NEVER merged into official 0.25)
        low_only = [p for p in pool_005 if float(p.get("score", 0.0)) < 0.25 and p.get("label") in R18_TARGET_CLASSES]
        aux_out = apply_r18_verifier_to_predictions(
            image_path=pathlib.Path(str(binfo["tif_path"])),
            predictions=low_only,
            model=cv_model,
            device=torch.device("cpu"),
            weights_sha256=cv_sha,
            t_relabel=float(cal["t_relabel"]),
            m_relabel=float(cal["m_relabel"]),
            t_reject_other=float(cal["t_reject_other"]),
            m_reject_other=float(cal["m_reject_other"]),
        )
        dq_gts = [g for g in gts if g["label"] in ("diode", "transistor")]
        aux_dq_preds = [p for p in aux_out["verified_predictions"] if p["label"] in ("diode", "transistor")]
        aux_tp = 0
        for cls in ("diode", "transistor"):
            cg = [{"bbox": g["bbox_xyxy"], "category": cls} for g in dq_gts if g["label"] == cls]
            cp = [{"bbox": p["bbox_xyxy"], "category": cls, "score": float(p.get("score", 0.0))} for p in aux_dq_preds if p["label"] == cls]
            aux_tp += repo_eval.match_category_predictions_one_to_one(cg, cp, iou_threshold=0.50)["tp"]
        aux_lowconf_study[bkey] = {
            "low_conf_target_candidates_005_to_025": len(low_only),
            "aux_verified_dq_preds": len(aux_dq_preds),
            "aux_dq_strict_tp": aux_tp,
            "note": "Auxiliary diagnostic only; NEVER merged into official 0.25 output.",
        }

        # Collect gallery cases
        for d in decisions:
            if d["decision"] in ("relabel", "reject_other"):
                matched_gt = [g["label"] for g in gts if _iou(d["bbox_xyxy"], g["bbox_xyxy"]) >= 0.50]
                gallery_cases.append({
                    "board": bkey,
                    "tif_path": binfo["tif_path"],
                    "bbox_xyxy": d["bbox_xyxy"],
                    "orig": d["orig_r15_label"],
                    "new": d["verified_label"] or "DROP",
                    "decision": d["decision"],
                    "top1_prob": d["top1_prob"],
                    "r15_score": d["r15_score"],
                    "gt_match": matched_gt[0] if matched_gt else "BG",
                })

        q_evals: List[Dict[str, Any]] = []
        for qrec in binfo.get("r18_queries") or []:
            q_doc = json.loads(pathlib.Path(str(qrec["query_file"])).read_text(encoding="utf-8"))
            tset = set(q_doc["query_spec"]["target_classes"])
            rmode = str(q_doc["query_spec"]["region_mode"])
            orig_wh = q_doc["orig_wh"]
            scoped_gts = [g for g in gts if g["label"] in tset and box_in_composite_region(g["bbox_xyxy"], orig_wh=orig_wh, region_mode=rmode)[0]]
            tp = fp = fn = 0
            for cls in sorted(tset):
                cg = [{"bbox": g["bbox_xyxy"], "category": cls} for g in scoped_gts if g["label"] == cls]
                cp = [{"bbox": p["bbox_xyxy"], "category": cls, "score": float(p.get("score", 0.0))} for p in q_doc["predictions"] if p["label"] == cls]
                m = repo_eval.match_category_predictions_one_to_one(cg, cp, iou_threshold=0.50)
                tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
            q_evals.append({
                "raw_query": q_doc["query_spec"]["raw_query"],
                "gt_count": len(scoped_gts),
                "predicted_count": int(q_doc["predicted_count"]),
                **_prf1(tp, fp, fn),
            })

        per_board[bkey] = {
            "gt_total": len(gts),
            "r15_frozen": ev_r15,
            "r18_verified": ev_r18,
            "verifier_decisions_summary": {
                "calls": len(decisions),
                "relabel": sum(1 for d in decisions if d["decision"] == "relabel"),
                "reject_other": sum(1 for d in decisions if d["decision"] == "reject_other"),
                "confirm_orig": sum(1 for d in decisions if d["decision"] == "confirm_orig"),
                "abstain_keep_orig": sum(1 for d in decisions if d["decision"] == "abstain_keep_orig"),
            },
            "r18_query_evaluations": q_evals,
        }

    if gallery_cases:
        _render_cases_gallery(gallery_cases, args.gallery)

    def _combine(key: str) -> Dict[str, Any]:
        bkeys = ("s20", "s4", "s1")
        tp = sum(int(per_board[b][key]["class_aware"]["tp"]) for b in bkeys)
        fp = sum(int(per_board[b][key]["class_aware"]["fp"]) for b in bkeys)
        fn = sum(int(per_board[b][key]["class_aware"]["fn"]) for b in bkeys)
        per_cls = {}
        for cls in ("capacitor", "resistor", "inductor", "transistor", "diode", "ic"):
            ctp = sum(int(per_board[b][key]["per_class"][cls]["tp"]) for b in bkeys)
            cfp = sum(int(per_board[b][key]["per_class"][cls]["fp"]) for b in bkeys)
            cfn = sum(int(per_board[b][key]["per_class"][cls]["fn"]) for b in bkeys)
            per_cls[cls] = _prf1(ctp, cfp, cfn)
        return {**_prf1(tp, fp, fn), "per_class": per_cls}

    c_r15 = _combine("r15_frozen")
    c_r18 = _combine("r18_verified")
    r15_dq = c_r15["per_class"]["diode"]["tp"] + c_r15["per_class"]["transistor"]["tp"]
    r18_dq = c_r18["per_class"]["diode"]["tp"] + c_r18["per_class"]["transistor"]["tp"]
    s1_absent_dq_fp = per_board["s1"]["r18_verified"]["per_class"]["diode"]["fp"] + per_board["s1"]["r18_verified"]["per_class"]["transistor"]["fp"]

    gate_checks = {
        "dq_tp_gain_ge_3": (r18_dq - r15_dq) >= 3,
        "combined_f1_non_decreasing": c_r18["f1_pct"] >= c_r15["f1_pct"],
        "ic_fp_non_increasing_and_tp_drop_le_1": (
            c_r18["per_class"]["ic"]["fp"] <= c_r15["per_class"]["ic"]["fp"]
            and c_r18["per_class"]["ic"]["tp"] >= (c_r15["per_class"]["ic"]["tp"] - 1)
        ),
        "cap_res_ind_bit_identical": non_target_invariance_ok,
        "s1_absent_dq_fp_ok": s1_absent_dq_fp <= 2 and per_board["s1"]["r18_verified"]["per_class"]["diode"]["fp"] <= 1 and per_board["s1"]["r18_verified"]["per_class"]["transistor"]["fp"] <= 1,
    }
    gate_passed = all(gate_checks.values())

    report = {
        "schema_version": "r18_paired_eval_report_v1",
        "pre_gt_lock_mismatches": 0,
        "combined_s20_s4_s1": {
            "r15_frozen": c_r15,
            "r18_verified": c_r18,
            "delta_f1_pp": round(c_r18["f1_pct"] - c_r15["f1_pct"], 2),
            "dq_tp_r15": r15_dq,
            "dq_tp_r18": r18_dq,
            "dq_tp_gain": r18_dq - r15_dq,
        },
        "r18_promotion_gate": {"passed": gate_passed, "checks": gate_checks},
        "auxiliary_005_to_025_study": aux_lowconf_study,
        "per_board": per_board,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "r15_f1": c_r15["f1_pct"],
        "r18_f1": c_r18["f1_pct"],
        "dq_tp": f"{r15_dq}->{r18_dq}",
        "ic_fp": f"{c_r15['per_class']['ic']['fp']}->{c_r18['per_class']['ic']['fp']}",
        "gate_passed": gate_passed,
        "checks": gate_checks,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
