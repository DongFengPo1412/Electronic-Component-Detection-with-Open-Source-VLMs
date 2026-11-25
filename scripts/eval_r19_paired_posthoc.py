"""Round 19 CPU-only post-lock GT evaluation for all 4 arms (A0, A1, A2, A3) + raw YOLO low-conf comparison."""

from __future__ import annotations

import csv
import hashlib
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
from pcb_vlm.evaluation.round19_postproc import (  # noqa: E402
    select_r19_a2_low_conf_candidates,
    verify_r19_untouched_invariants,
)

R19_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration19-20261125")
LOCK_FILE = R19_ART / "eval" / "r19_pre_gt_lock.json"
FROZEN_PLAN = R19_ART / "frozen_plan_r19_pre_cal.json"
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


def _geometry_coverage(gts: Sequence[Mapping[str, Any]], preds_025: Sequence[Mapping[str, Any]], pool_005: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    dq_gts = [g for g in gts if g["label"] in ("diode", "transistor")]
    low_only = [p for p in pool_005 if float(p.get("score", 0.0)) < 0.25]

    def _cov(pool: Sequence[Mapping[str, Any]], class_aware: bool) -> Dict[str, int]:
        d_c = q_c = 0
        for g in dq_gts:
            for p in pool:
                if class_aware and str(p.get("label") or "").lower() != g["label"]:
                    continue
                if compute_iou_xyxy(g["bbox_xyxy"], p["bbox_xyxy"]) >= 0.50:
                    if g["label"] == "diode":
                        d_c += 1
                    else:
                        q_c += 1
                    break
        return {"diode": d_c, "transistor": q_c, "dq_total": d_c + q_c}

    return {
        "gt_counts": {
            "diode": sum(1 for g in dq_gts if g["label"] == "diode"),
            "transistor": sum(1 for g in dq_gts if g["label"] == "transistor"),
            "dq_total": len(dq_gts),
        },
        "class_agnostic_iou50": {
            "official_025": _cov(preds_025, class_aware=False),
            "low_005_to_025": _cov(low_only, class_aware=False),
            "combined_ge_005": _cov(pool_005, class_aware=False),
        },
        "class_aware_iou50": {
            "official_025": _cov(preds_025, class_aware=True),
            "low_005_to_025": _cov(low_only, class_aware=True),
            "combined_ge_005": _cov(pool_005, class_aware=True),
        },
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


def _render_case_comparison(cases: Sequence[Mapping[str, Any]], out_path: pathlib.Path) -> None:
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
        color = (80, 220, 120) if c["kind"].startswith("RESCUE_TP") else (255, 190, 60) if c["kind"].startswith("DROP_IC") else (240, 90, 90)
        cd.rectangle([42, 42, 182, 182], outline=color, width=2)
        canvas.paste(crop, (x0 + 8, y0 + 36))
        draw.text((x0 + 8, y0 + 4), c["title1"], fill=(240, 240, 240))
        draw.text((x0 + 8, y0 + 18), c["title2"], fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    # 1. Port closed & lock/frozen-plan verification
    s = socket.socket()
    s.settimeout(1.0)
    rc = s.connect_ex(("127.0.0.1", 8823))
    s.close()
    assert rc != 0, "Port 8823 must be closed before post-hoc eval!"

    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    lock_mismatches = 0
    for item in lock["locked_files"]:
        p = pathlib.Path(item["path"])
        if not p.is_file() or file_sha256(p).lower() != item["sha256"].lower():
            lock_mismatches += 1
    assert lock_mismatches == 0, f"Pre-GT lock mismatches: {lock_mismatches}"

    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    plan_diffs = 0
    for item in plan["frozen_inputs"]:
        p = pathlib.Path(item["path"])
        if not p.is_file() or file_sha256(p).lower() != item["sha256"].lower():
            plan_diffs += 1
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

    arm_evals: Dict[str, List[Dict[str, Any]]] = {"A0": [], "A1": [], "A2": [], "A3": [], "A2_yolo_only_s05": []}

    for bkey in ("s20", "s4", "s1"):
        binfo = lock["boards"][bkey]
        bdir = pathlib.Path(binfo["artifacts_dir"])
        gts, _ = load_coco_gts_for_single_board(coco_by_bkey[bkey], image_id=None)
        roi = binfo["roi_bbox_half_open"]
        pool_005 = json.loads((bdir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        a0 = json.loads((bdir / "r19_a0_predictions.json").read_text(encoding="utf-8"))
        a1 = json.loads((bdir / "r19_a1_predictions.json").read_text(encoding="utf-8"))
        a2 = json.loads((bdir / "r19_a2_predictions.json").read_text(encoding="utf-8"))
        a3 = json.loads((bdir / "r19_a3_predictions.json").read_text(encoding="utf-8"))
        audit = json.loads((bdir / "r19_postproc_audit.json").read_text(encoding="utf-8"))
        yolo_summary = json.loads((bdir / "yolo_run_summary.json").read_text(encoding="utf-8"))

        # Verify untouched classes identical across A1, A2, A3
        verify_r19_untouched_invariants(a0_preds=a0, derived_preds=a1, arm_name=f"{bkey}_A1")
        verify_r19_untouched_invariants(a0_preds=a0, derived_preds=a2, arm_name=f"{bkey}_A2")
        verify_r19_untouched_invariants(a0_preds=a0, derived_preds=a3, arm_name=f"{bkey}_A3")

        # Raw yolo_only_s05 baseline on the exact same locked pools
        y05_res = select_r19_a2_low_conf_candidates(
            official_preds_025=a0,
            low_conf_pool_005=pool_005,
            rule_spec={"rule_id": "yolo_only_s05", "rule_family": "yolo_only", "score_min": 0.05},
        )

        ev_map = {
            "A0": evaluate_composite_board(gts=gts, predictions_025=a0, low_conf_pool_005=pool_005, roi_bbox_half_open=roi),
            "A1": evaluate_composite_board(gts=gts, predictions_025=a1, low_conf_pool_005=pool_005, roi_bbox_half_open=roi),
            "A2": evaluate_composite_board(gts=gts, predictions_025=a2, low_conf_pool_005=pool_005, roi_bbox_half_open=roi),
            "A3": evaluate_composite_board(gts=gts, predictions_025=a3, low_conf_pool_005=pool_005, roi_bbox_half_open=roi),
            "A2_yolo_only_s05": evaluate_composite_board(gts=gts, predictions_025=y05_res["a2_predictions"], low_conf_pool_005=pool_005, roi_bbox_half_open=roi),
        }
        for arm_k, ev in ev_map.items():
            arm_evals[arm_k].append(ev)

        # Standalone S4 localized-but-misclassified D/Q record
        s4_localized_miscls = []
        if bkey == "s4":
            for g in gts:
                if g["label"] not in ("diode", "transistor"):
                    continue
                for p in a0:
                    iou = compute_iou_xyxy(g["bbox_xyxy"], p["bbox_xyxy"])
                    if iou >= 0.50 and str(p["label"]).lower() != g["label"]:
                        s4_localized_miscls.append({
                            "gt_label": g["label"],
                            "r15_pred_id": p.get("prediction_id"),
                            "r15_label": p["label"],
                            "r15_score": round(float(p["score"]), 4),
                            "iou": round(iou, 4),
                        })

        # Per-box diff audit
        dropped_ic_audit = []
        for d in audit["a1_dropped_ic_boxes"]:
            gt_lbl, gt_iou = _best_gt_for_box(d["bbox_xyxy"], gts)
            dropped_ic_audit.append({**d, "gt_match": gt_lbl, "gt_iou": gt_iou})
            if len(visual_cases) < 6:
                visual_cases.append({
                    "kind": "DROP_IC",
                    "tif_path": binfo["tif_path"],
                    "bbox_xyxy": d["bbox_xyxy"],
                    "title1": f"{bkey} DROP IC r15={d['r15_score']:.2f}",
                    "title2": f"GT={gt_lbl} p_other={d['top1_prob']:.2f}",
                })

        accepted_low_audit = []
        for lb in audit["a2_accepted_low_boxes"]:
            gt_lbl, gt_iou = _best_gt_for_box(lb["bbox_xyxy"], gts)
            is_tp = (gt_lbl == lb["label"])
            accepted_low_audit.append({
                "prediction_id": lb.get("prediction_id"),
                "label": lb["label"],
                "score": lb["score"],
                "verifier_top1_prob": lb.get("verifier_top1_prob"),
                "bbox_xyxy": lb["bbox_xyxy"],
                "gt_match": gt_lbl,
                "gt_iou": gt_iou,
                "is_strict_tp": is_tp,
            })
            visual_cases.append({
                "kind": "RESCUE_TP" if is_tp else "LOW_FP",
                "tif_path": binfo["tif_path"],
                "bbox_xyxy": lb["bbox_xyxy"],
                "title1": f"{bkey} LOW {lb['label']} s={lb['score']:.2f}",
                "title2": f"{'TP' if is_tp else 'FP'} GT={gt_lbl} p={lb.get('verifier_top1_prob', 0):.2f}",
            })

        diff_audit[bkey] = {
            "a1_dropped_ic_boxes": dropped_ic_audit,
            "a2_accepted_low_boxes": accepted_low_audit,
            "a2_dedup_suppressed_boxes": audit["a2_dedup_suppressed_boxes"],
            "s4_localized_misclassified_dq": s4_localized_miscls if bkey == "s4" else None,
        }

        per_board[bkey] = {
            "gt_total": len(gts),
            "official_025_count": len(a0),
            "low_conf_pool_005_count": len(pool_005),
            "wall_seconds": binfo["client_poll_wall_seconds"],
            "geometry_coverage": _geometry_coverage(gts, a0, pool_005),
            "arms": {
                arm_k: {
                    "tp": ev["class_aware"]["tp"],
                    "fp": ev["class_aware"]["fp"],
                    "fn": ev["class_aware"]["fn"],
                    "f1_pct": ev["class_aware"]["f1_pct"],
                    "diode": ev["per_class"]["diode"],
                    "transistor": ev["per_class"]["transistor"],
                    "ic": ev["per_class"]["ic"],
                }
                for arm_k, ev in ev_map.items()
            },
        }

    agg = {arm_k: _aggregate(evs) for arm_k, evs in arm_evals.items()}

    # 7-point Round 19 promotion gate evaluated on A3 (combined candidate)
    a0_agg, a3_agg = agg["A0"], agg["A3"]
    dq_tp_gain = a3_agg["dq_tp"] - a0_agg["dq_tp"]
    s1_absent_dq_fp = per_board["s1"]["arms"]["A3"]["diode"]["fp"] + per_board["s1"]["arms"]["A3"]["transistor"]["fp"]
    every_board_tp_ok = all(
        per_board[b]["arms"]["A3"]["tp"] >= per_board[b]["arms"]["A0"]["tp"]
        for b in ("s20", "s4", "s1")
    )
    gate_checks = {
        "dq_tp_gain_ge_3": dq_tp_gain >= 3,
        "combined_micro_f1_ge_66_32": a3_agg["f1_pct"] >= 66.32,
        "every_dev_board_total_tp_ge_r15": every_board_tp_ok,
        "combined_ic_fp_le_72_and_tp_ge_10": (a3_agg["ic_fp"] <= 72 and a3_agg["ic_tp"] >= 10),
        "s1_absent_dq_total_fp_eq_0": s1_absent_dq_fp == 0,
        "cap_res_ind_100pct_identical_to_a0": True,
        "zero_lock_and_frozen_plan_diffs": True,
    }
    gate_passed = all(gate_checks.values())

    (R19_ART / "eval" / "r19_per_box_diff_audit.json").write_text(
        json.dumps(diff_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _render_case_comparison(visual_cases, R19_ART / "eval" / "r19_case_comparison.png")

    report = {
        "schema_version": "r19_paired_eval_report_v1",
        "port_8823_open": False,
        "pre_gt_lock_mismatches": lock_mismatches,
        "frozen_plan_diffs": plan_diffs,
        "combined_s20_s4_s1": agg,
        "per_board": per_board,
        "r19_promotion_gate": {
            "passed": gate_passed,
            "dq_tp_gain_a3_vs_a0": dq_tp_gain,
            "dq_tp_a2_yolo_only_s05_vs_a0": agg["A2_yolo_only_s05"]["dq_tp"] - a0_agg["dq_tp"],
            "checks": gate_checks,
        },
    }
    (R19_ART / "eval" / "r19_paired_eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "A0_f1": a0_agg["f1_pct"], "A0_dq_tp": a0_agg["dq_tp"], "A0_ic_fp": a0_agg["ic_fp"],
        "A1_f1": agg["A1"]["f1_pct"], "A1_dq_tp": agg["A1"]["dq_tp"], "A1_ic_fp": agg["A1"]["ic_fp"],
        "A2_f1": agg["A2"]["f1_pct"], "A2_dq_tp": agg["A2"]["dq_tp"], "A2_dq_fp": agg["A2"]["dq_fp"],
        "A2_yolo05_dq_tp": agg["A2_yolo_only_s05"]["dq_tp"], "A2_yolo05_dq_fp": agg["A2_yolo_only_s05"]["dq_fp"],
        "A3_f1": a3_agg["f1_pct"], "A3_dq_tp": a3_agg["dq_tp"], "A3_dq_fp": a3_agg["dq_fp"], "A3_ic_fp": a3_agg["ic_fp"],
        "s1_absent_dq_fp_a3": s1_absent_dq_fp,
        "gate_passed": gate_passed,
        "checks": gate_checks,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
