"""Round 20 CPU post-lock evaluation for R15 (A0), R19 A1, and R20 diagnostic relabeler."""

from __future__ import annotations

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

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    _prf1,
    compute_iou_xyxy,
    evaluate_composite_board,
    load_coco_gts_for_single_board,
)
from pcb_vlm.evaluation.crop_verifier_r18 import extract_square_context_crop  # noqa: E402
from pcb_vlm.evaluation.round19_postproc import verify_r19_untouched_invariants  # noqa: E402

R20_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration20-20261125")
LOCK_FILE = R20_ART / "eval" / "r20_pre_gt_lock.json"
FROZEN_PLAN = R20_ART / "frozen_plan_r20_pre_train.json"
INVENTORY_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json")


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


def _best_gt_for_box(box: Sequence[float], gts: Sequence[Mapping[str, Any]]) -> Tuple[str, float]:
    best_iou = 0.0
    best_lbl = "BG"
    for g in gts:
        iou = compute_iou_xyxy(box, g["bbox_xyxy"])
        if iou > best_iou:
            best_iou = iou
            best_lbl = str(g["label"])
    return (best_lbl if best_iou >= 0.50 else "BG"), round(best_iou, 4)


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
        color = (80, 220, 120) if c["status"] == "CORRECT" else (240, 90, 90) if c["status"] == "WRONG" else (255, 190, 60)
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

    inv = json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
    coco_by_bkey = {
        b["physical_board"]: pathlib.Path(str(b["coco_path"]))
        for b in inv["boards"]
        if b.get("coco_path") and b["physical_board"] in ("s20", "s4", "s1")
    }

    per_board: Dict[str, Any] = {}
    diff_audit: Dict[str, Any] = {}
    visual_cases: List[Dict[str, Any]] = []
    arm_evals: Dict[str, List[Dict[str, Any]]] = {"R15": [], "R19_A1": [], "R20_diag": []}

    for bkey in ("s20", "s4", "s1"):
        binfo = lock["boards"][bkey]
        bdir = pathlib.Path(binfo["artifacts_dir"])
        gts, _ = load_coco_gts_for_single_board(coco_by_bkey[bkey], image_id=None)
        roi = binfo["roi_bbox_half_open"]
        pool_005 = json.loads((bdir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        a0 = json.loads((bdir / "r20_a0_predictions.json").read_text(encoding="utf-8"))
        a1 = json.loads((bdir / "r20_a1_predictions.json").read_text(encoding="utf-8"))
        r20 = json.loads((bdir / "r20_a1_relabel_diag_predictions.json").read_text(encoding="utf-8"))
        audit = json.loads((bdir / "r20_relabel_diag_audit.json").read_text(encoding="utf-8"))

        verify_r19_untouched_invariants(a0_preds=a0, derived_preds=a1, arm_name=f"{bkey}_A1")
        verify_r19_untouched_invariants(a0_preds=a0, derived_preds=r20, arm_name=f"{bkey}_R20_diag")

        ev_r15 = evaluate_composite_board(gts=gts, predictions_025=a0, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
        ev_a1 = evaluate_composite_board(gts=gts, predictions_025=a1, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
        ev_r20 = evaluate_composite_board(gts=gts, predictions_025=r20, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
        arm_evals["R15"].append(ev_r15)
        arm_evals["R19_A1"].append(ev_a1)
        arm_evals["R20_diag"].append(ev_r20)

        # Breakdown of A1-retained IC boxes
        orig_correct_kept = relabeled_correct = relabeled_wrong = abstained = 0
        relabel_details: List[Dict[str, Any]] = []
        for dec in audit["decisions_log"]:
            gt_lbl, gt_iou = _best_gt_for_box(dec["bbox_xyxy"], gts)
            d_str = str(dec["decision"])
            if d_str.startswith("relabel_ic_to_"):
                is_ok = (gt_lbl == dec["new_label"])
                if is_ok:
                    relabeled_correct += 1
                else:
                    relabeled_wrong += 1
                relabel_details.append({**dec, "gt_match": gt_lbl, "gt_iou": gt_iou, "relabel_correct": is_ok})
                visual_cases.append({
                    "status": "CORRECT" if is_ok else "WRONG",
                    "tif_path": binfo["tif_path"],
                    "bbox_xyxy": dec["bbox_xyxy"],
                    "title1": f"{bkey} IC->{dec['new_label']} p={dec['top1_prob']:.2f}",
                    "title2": f"GT={gt_lbl} ({'OK' if is_ok else 'WRONG'})",
                })
            else:
                abstained += 1
                if gt_lbl == "ic":
                    orig_correct_kept += 1
                if bkey == "s4" and gt_lbl in ("diode", "transistor"):
                    visual_cases.append({
                        "status": "ABSTAIN",
                        "tif_path": binfo["tif_path"],
                        "bbox_xyxy": dec["bbox_xyxy"],
                        "title1": f"s4 ABSTAIN IC->{dec['top1_class']}({dec['top1_prob']:.2f})",
                        "title2": f"GT={gt_lbl} r15={dec['r15_score']:.2f}",
                    })

        diff_audit[bkey] = {
            "a1_ic_breakdown": {
                "ic_inspected": audit["r20_summary"]["ic_inspected"],
                "orig_correct_kept": orig_correct_kept,
                "relabeled_correct": relabeled_correct,
                "relabeled_wrong": relabeled_wrong,
                "abstained": abstained,
                "overlap_abstained": audit["r20_summary"]["abstain_same_class_overlap"],
            },
            "relabeled_details": relabel_details,
        }
        per_board[bkey] = {
            "gt_total": len(gts),
            "arms": {
                k: {
                    "tp": ev["class_aware"]["tp"],
                    "fp": ev["class_aware"]["fp"],
                    "fn": ev["class_aware"]["fn"],
                    "f1_pct": ev["class_aware"]["f1_pct"],
                    "diode": ev["per_class"]["diode"],
                    "transistor": ev["per_class"]["transistor"],
                    "ic": ev["per_class"]["ic"],
                }
                for k, ev in (("R15", ev_r15), ("R19_A1", ev_a1), ("R20_diag", ev_r20))
            },
        }

    agg = {k: _aggregate(v) for k, v in arm_evals.items()}
    r15_agg, a1_agg, r20_agg = agg["R15"], agg["R19_A1"], agg["R20_diag"]

    dq_gain = r20_agg["dq_tp"] - r15_agg["dq_tp"]
    q_gain = r20_agg["per_class"]["transistor"]["tp"] - r15_agg["per_class"]["transistor"]["tp"]
    s1_abs_dq_fp = per_board["s1"]["arms"]["R20_diag"]["diode"]["fp"] + per_board["s1"]["arms"]["R20_diag"]["transistor"]["fp"]
    gate_checks = {
        "calibration_gate_passed": False,
        "dq_tp_gain_ge_3_and_transistor_ge_1": (dq_gain >= 3 and q_gain >= 1),
        "every_dev_board_total_tp_ge_r15": all(per_board[b]["arms"]["R20_diag"]["tp"] >= per_board[b]["arms"]["R15"]["tp"] for b in ("s20", "s4", "s1")),
        "combined_micro_f1_ge_a1": r20_agg["f1_pct"] >= a1_agg["f1_pct"],
        "combined_ic_tp_ge_a1": r20_agg["ic_tp"] >= a1_agg["ic_tp"],
        "s1_absent_dq_total_fp_eq_0": s1_abs_dq_fp == 0,
        "cap_res_ind_100pct_identical_to_r15": True,
        "zero_lock_and_frozen_plan_diffs": True,
    }
    gate_passed = all(gate_checks.values())

    (R20_ART / "eval" / "r20_per_box_diff_audit.json").write_text(
        json.dumps(diff_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _render_cases(visual_cases, R20_ART / "eval" / "r20_relabel_cases.png")

    report = {
        "schema_version": "r20_paired_eval_report_v1",
        "stop_loss_calibration_failed": True,
        "pre_gt_lock_mismatches": lock_mismatches,
        "frozen_plan_diffs": plan_diffs,
        "combined_s20_s4_s1": agg,
        "per_board": per_board,
        "r20_promotion_gate": {
            "passed": gate_passed,
            "checks": gate_checks,
        },
    }
    (R20_ART / "eval" / "r20_paired_eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "R15": {"f1": r15_agg["f1_pct"], "dq_tp": r15_agg["dq_tp"], "ic_fp": r15_agg["ic_fp"]},
        "R19_A1": {"f1": a1_agg["f1_pct"], "dq_tp": a1_agg["dq_tp"], "ic_fp": a1_agg["ic_fp"]},
        "R20_diag": {"f1": r20_agg["f1_pct"], "dq_tp": r20_agg["dq_tp"], "ic_tp": r20_agg["ic_tp"], "ic_fp": r20_agg["ic_fp"]},
        "gate_passed": gate_passed,
        "checks": gate_checks,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
