"""CPU post-lock evaluation of Round 21 448px calibration (coverage split, 4-arm metrics, early-stop gate)."""

from __future__ import annotations

import csv
import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.data.fics_dslr import FICS_RAW_TO_CANONICAL, file_sha256, parse_component_location  # noqa: E402
from pcb_vlm.evaluation.composite_detector import _prf1, compute_iou_xyxy, evaluate_composite_board  # noqa: E402
from pcb_vlm.evaluation.crop_verifier_r18 import extract_square_context_crop  # noqa: E402
from pcb_vlm.evaluation.round21_multiscale_merge import compute_one_to_one_coverage_split  # noqa: E402

R18_MANIFEST = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration18-20261125\crop_dataset_r18\crop_manifest.json")
R21_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration21-20261125")
LOCK_FILE = R21_ART / "calibration" / "r21_cal_pre_gt_lock.json"
FROZEN_PLAN = R21_ART / "frozen_plan_r21_pre_cal.json"


def _read_csv_gts(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    raw_lower = {k.lower(): v for k, v in FICS_RAW_TO_CANONICAL.items()}
    gts: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rt = str(row.get("component_type") or "").strip()
            canon = FICS_RAW_TO_CANONICAL.get(rt) or raw_lower.get(rt.lower())
            if not canon:
                continue
            _, box, _ = parse_component_location(row.get("component_location") or "")
            gts.append({"label": canon, "bbox_xyxy": [float(v) for v in box]})
    return gts


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
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **overall,
        "dq_tp": per_class["diode"]["tp"] + per_class["transistor"]["tp"],
        "dq_fp": per_class["diode"]["fp"] + per_class["transistor"]["fp"],
        "ic_tp": per_class["ic"]["tp"],
        "ic_fp": per_class["ic"]["fp"],
        "per_class": per_class,
    }


def _best_gt(box: Sequence[float], gts: Sequence[Mapping[str, Any]]) -> Tuple[str, float]:
    bi, bh = "BG", 0.0
    for g in gts:
        iou = compute_iou_xyxy(box, g["bbox_xyxy"])
        if iou > bh:
            bh, bi = iou, str(g["label"])
    return (bi if bh >= 0.50 else "BG"), round(bh, 4)


def _render_cases(cases: Sequence[Mapping[str, Any]], out_path: pathlib.Path) -> None:
    cols = 4
    rows = max(1, (len(cases) + cols - 1) // cols)
    cell_w, cell_h = 240, 265
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 28, 34))
    draw = ImageDraw.Draw(canvas)
    for idx, c in enumerate(cases[: cols * rows]):
        r, col = divmod(idx, cols)
        x0, y0 = col * cell_w, r * cell_h
        with Image.open(c["tif_path"]) as im:
            crop = extract_square_context_crop(im, c["bbox_xyxy"], target_size=224, context_scale=1.60)
        cd = ImageDraw.Draw(crop)
        color = (80, 220, 120) if c["status"] == "TP" else (255, 190, 60) if c["status"] == "NEW_COV" else (240, 90, 90)
        cd.rectangle([42, 42, 182, 182], outline=color, width=2)
        canvas.paste(crop, (x0 + 8, y0 + 36))
        draw.text((x0 + 8, y0 + 4), c["title1"], fill=(240, 240, 240))
        draw.text((x0 + 8, y0 + 18), c["title2"], fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    lock_mismatches = sum(
        1 for it in lock["locked_files"]
        if not pathlib.Path(it["path"]).is_file() or file_sha256(pathlib.Path(it["path"])).lower() != it["sha256"].lower()
    )
    assert lock_mismatches == 0, f"Pre-GT lock mismatches: {lock_mismatches}"

    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    plan_diffs = sum(
        1 for it in plan["frozen_inputs"]
        if not pathlib.Path(it["path"]).is_file() or file_sha256(pathlib.Path(it["path"])).lower() != it["sha256"].lower()
    )
    assert plan_diffs == 0, f"Frozen plan diffs: {plan_diffs}"

    prop_idx = json.loads(R18_MANIFEST.read_text(encoding="utf-8"))["proposal_cache_index"]
    per_board: Dict[str, Any] = {}
    arm_evals: Dict[str, List[Dict[str, Any]]] = {"A0": [], "A1": [], "A0_plus_448": [], "A1_plus_448": []}
    visual_cases: List[Dict[str, Any]] = []

    total_new_cov_448_005 = 0
    total_new_cov_448_025 = 0

    for stem in ("s14_front1", "s15_front1", "s3_front"):
        binfo = lock["boards"][stem]
        art_dir = pathlib.Path(binfo["artifacts_dir"])
        gts = _read_csv_gts(pathlib.Path(str(prop_idx[stem]["csv_path"])))
        roi = binfo["roi_640"]

        pool_640_005 = json.loads((art_dir / "yolo_640_low_conf_pool_005.json").read_text(encoding="utf-8"))
        pool_448_005 = json.loads((art_dir / "yolo_448_low_conf_pool_005.json").read_text(encoding="utf-8"))
        preds_448_025 = json.loads((art_dir / "yolo_448_official_025.json").read_text(encoding="utf-8"))

        a0 = json.loads((art_dir / "arm_a0_predictions.json").read_text(encoding="utf-8"))
        a1 = json.loads((art_dir / "arm_a1_predictions.json").read_text(encoding="utf-8"))
        a0_448 = json.loads((art_dir / "arm_a0_plus_448_predictions.json").read_text(encoding="utf-8"))
        a1_448 = json.loads((art_dir / "arm_a1_plus_448_predictions.json").read_text(encoding="utf-8"))
        audit = json.loads((art_dir / "r21_448_merge_audit.json").read_text(encoding="utf-8"))

        cov = compute_one_to_one_coverage_split(
            gts=gts,
            pool_640_005=pool_640_005,
            pool_448_005=pool_448_005,
            preds_448_025=preds_448_025,
        )
        total_new_cov_448_005 += cov["summary"]["newly_covered_only_by_448_pool005"]
        total_new_cov_448_025 += cov["summary"]["newly_covered_only_by_448_off025"]

        ev_map = {
            "A0": evaluate_composite_board(gts=gts, predictions_025=a0, low_conf_pool_005=pool_640_005, roi_bbox_half_open=roi),
            "A1": evaluate_composite_board(gts=gts, predictions_025=a1, low_conf_pool_005=pool_640_005, roi_bbox_half_open=roi),
            "A0_plus_448": evaluate_composite_board(gts=gts, predictions_025=a0_448, low_conf_pool_005=pool_640_005, roi_bbox_half_open=roi),
            "A1_plus_448": evaluate_composite_board(gts=gts, predictions_025=a1_448, low_conf_pool_005=pool_640_005, roi_bbox_half_open=roi),
        }
        for ak, ev in ev_map.items():
            arm_evals[ak].append(ev)

        for acc in audit["accepted_448_boxes"]:
            gt_lbl, gt_iou = _best_gt(acc["bbox_xyxy"], gts)
            is_tp = (gt_lbl == acc["label"])
            visual_cases.append({
                "status": "TP" if is_tp else "FP",
                "tif_path": binfo["tif_path"],
                "bbox_xyxy": acc["bbox_xyxy"],
                "title1": f"{stem} 448 {acc['label']} s={acc['score']:.2f}",
                "title2": f"{'TP' if is_tp else 'FP'} GT={gt_lbl} ({gt_iou})",
            })
        for rec in cov["gt_records"]:
            if rec["coverage_status"] == "newly_covered_only_by_448_pool005":
                m448 = rec["match_448_005"]
                visual_cases.append({
                    "status": "NEW_COV",
                    "tif_path": binfo["tif_path"],
                    "bbox_xyxy": rec["bbox_xyxy"],
                    "title1": f"{stem} NEW_448 GT={rec['gt_label']}",
                    "title2": f"p={m448['pred_label']}({m448['pred_score']:.2f})",
                })

        per_board[stem] = {
            "gt_total": len(gts),
            "coverage_split": cov["summary"],
            "merge_summary": audit["merge_summary"],
            "arms": {
                ak: {
                    "tp": ev["class_aware"]["tp"],
                    "fp": ev["class_aware"]["fp"],
                    "fn": ev["class_aware"]["fn"],
                    "f1_pct": ev["class_aware"]["f1_pct"],
                    "diode": ev["per_class"]["diode"],
                    "transistor": ev["per_class"]["transistor"],
                    "ic": ev["per_class"]["ic"],
                }
                for ak, ev in ev_map.items()
            },
        }

    agg = {ak: _aggregate(evs) for ak, evs in arm_evals.items()}
    a1_agg, a1_448_agg = agg["A1"], agg["A1_plus_448"]
    dq_tp_gain = a1_448_agg["dq_tp"] - a1_agg["dq_tp"]
    dq_fp_added = a1_448_agg["dq_fp"] - a1_agg["dq_fp"]
    orig_tp_preserved = all(
        per_board[s]["arms"]["A1_plus_448"]["tp"] >= per_board[s]["arms"]["A1"]["tp"]
        for s in ("s14_front1", "s15_front1", "s3_front")
    )

    gate_checks = {
        "cal_newly_covered_only_by_448_ge_2": total_new_cov_448_005 >= 2,
        "cal_formal_dq_tp_gain_ge_1": dq_tp_gain >= 1,
        "cal_formal_dq_fp_added_le_tp_gain": dq_fp_added <= dq_tp_gain,
        "cal_orig_tp_preserved": orig_tp_preserved,
    }
    cal_gate_passed = all(gate_checks.values())

    _render_cases(visual_cases, R21_ART / "calibration" / "r21_cal_448_cases.png")

    eval_report = {
        "schema_version": "r21_cal_eval_report_v1",
        "pre_gt_lock_mismatches": lock_mismatches,
        "frozen_plan_diffs": plan_diffs,
        "total_newly_covered_only_by_448_pool005": total_new_cov_448_005,
        "total_newly_covered_only_by_448_off025": total_new_cov_448_025,
        "combined_cal": agg,
        "per_board": per_board,
    }
    (R21_ART / "calibration" / "r21_cal_eval_report.json").write_text(
        json.dumps(eval_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    gate_doc = {
        "schema_version": "r21_cal_gate_decision_v1",
        "cal_gate_passed": cal_gate_passed,
        "proceed_to_dev_eval": cal_gate_passed,
        "dq_tp_gain_a1_plus_448_vs_a1": dq_tp_gain,
        "dq_fp_added_a1_plus_448_vs_a1": dq_fp_added,
        "checks": gate_checks,
    }
    (R21_ART / "calibration" / "r21_cal_gate_decision.json").write_text(
        json.dumps(gate_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "new_cov_448_005": total_new_cov_448_005,
        "new_cov_448_025": total_new_cov_448_025,
        "A0": {"f1": agg["A0"]["f1_pct"], "dq_tp": agg["A0"]["dq_tp"], "dq_fp": agg["A0"]["dq_fp"]},
        "A1": {"f1": a1_agg["f1_pct"], "dq_tp": a1_agg["dq_tp"], "dq_fp": a1_agg["dq_fp"]},
        "A0_plus_448": {"f1": agg["A0_plus_448"]["f1_pct"], "dq_tp": agg["A0_plus_448"]["dq_tp"], "dq_fp": agg["A0_plus_448"]["dq_fp"]},
        "A1_plus_448": {"f1": a1_448_agg["f1_pct"], "dq_tp": a1_448_agg["dq_tp"], "dq_fp": a1_448_agg["dq_fp"]},
        "cal_gate_passed": cal_gate_passed,
        "checks": gate_checks,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
