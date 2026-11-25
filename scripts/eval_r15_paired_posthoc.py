"""Pure CPU post-GT evaluator for Round 15 locked web predictions (`r15_pre_gt_lock.json`).

Strict separation:
- Contains ZERO web API / GPU calls and NEVER modifies any file listed in `r15_pre_gt_lock.json`.
- Verifies EVERY locked prediction/raw file SHA-256 before computing metrics (0 mismatches required).
- Evaluates BOTH `wacv_yolo11n` and `fics_yolo11n_r15` across `S20` (111 GT), `S4` (52 GT), and `S1` (85 GT),
  including per-class, per-size-bin, class-agnostic IoU 0.25/0.50 (0.05 & 0.25), error decomposition,
  ROI miss audit, R14 reproducibility cross-check, pre-registered R15 Dev Promotion Gate, and language/absent queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

DEFAULT_LOCK_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\eval\r15_pre_gt_lock.json")
DEFAULT_INVENTORY = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json")
DEFAULT_OUTPUT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\eval\r15_paired_eval_report.json")
DEFAULT_R14_CODEX_AUDIT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-codex-audit-20261123\round14_independent_recomputation.json")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate locked R15 composite predictions against COCO GT (CPU only).")
    parser.add_argument("--lock-file", type=pathlib.Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--inventory", type=pathlib.Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
    # 1) Verify ZERO tampering / 0 SHA-256 mismatches across all pre-GT locked files
    mismatches: List[Dict[str, str]] = []
    for lf in lock["locked_files"]:
        lp = pathlib.Path(str(lf["path"]))
        if not lp.is_file():
            mismatches.append({"role": lf["role"], "path": str(lp), "reason": "missing"})
            continue
        actual_sha = _sha256_file(lp)
        if actual_sha != str(lf["sha256"]).lower():
            mismatches.append({"role": lf["role"], "path": str(lp), "expected": lf["sha256"], "actual": actual_sha})
    if mismatches:
        raise RuntimeError(f"Pre-GT lock SHA-256 verification failed: {mismatches}")

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    coco_by_bkey = {
        b["physical_board"]: pathlib.Path(str(b["coco_path"]))
        for b in inv["boards"]
        if b.get("coco_path") and b["physical_board"] in ("s20", "s4", "s1")
    }

    per_board: Dict[str, Any] = {}
    for bkey, binfo in lock["boards"].items():
        gts, _ = load_coco_gts_for_single_board(coco_by_bkey[bkey], image_id=None)
        arms_eval: Dict[str, Any] = {}
        for dp in ("wacv_yolo11n", "fics_yolo11n_r15"):
            a_dir = pathlib.Path(str(binfo["arms"][dp]["artifacts_dir"]))
            preds_025 = json.loads((a_dir / "yolo_official_025.json").read_text(encoding="utf-8"))
            pool_005 = json.loads((a_dir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
            summary = json.loads((a_dir / "yolo_run_summary.json").read_text(encoding="utf-8"))
            ev = evaluate_composite_board(
                gts=gts,
                predictions_025=preds_025,
                low_conf_pool_005=pool_005,
                roi_bbox_half_open=binfo["roi_bbox_half_open"],
            )
            arms_eval[dp] = {
                "job_id": binfo["arms"][dp]["job_id"],
                "weights_sha256": summary.get("weights_sha256"),
                "non_fics_extra_counts_025": summary.get("counts", {}).get("non_fics_class_counts_025", {}),
                "timing_and_vram": summary.get("timing_and_vram"),
                "client_poll_wall_seconds": binfo["arms"][dp]["client_poll_wall_seconds"],
                **ev,
            }

        # Evaluate queries run on fics_yolo11n_r15
        q_evals: List[Dict[str, Any]] = []
        for qrec in binfo.get("fics_queries") or []:
            q_doc = json.loads(pathlib.Path(str(qrec["query_file"])).read_text(encoding="utf-8"))
            target_set = set(q_doc["query_spec"]["target_classes"])
            reg_mode = str(q_doc["query_spec"]["region_mode"])
            orig_wh = q_doc["orig_wh"]
            scoped_gts = [
                g for g in gts
                if g["label"] in target_set
                and box_in_composite_region(g["bbox_xyxy"], orig_wh=orig_wh, region_mode=reg_mode)[0]
            ]
            # Per-class 1-to-1 matching scoped to query targets & spatial half
            tot_tp = tot_fp = tot_fn = 0
            for cls in sorted(target_set):
                cg = [{"bbox": g["bbox_xyxy"], "category": cls} for g in scoped_gts if g["label"] == cls]
                cp = [{"bbox": p["bbox_xyxy"], "category": cls, "score": float(p.get("score", 0.0))} for p in q_doc["predictions"] if p["label"] == cls]
                m = repo_eval.match_category_predictions_one_to_one(cg, cp, iou_threshold=0.50)
                tot_tp += m["tp"]; tot_fp += m["fp"]; tot_fn += m["fn"]
            qm = _prf1(tot_tp, tot_fp, tot_fn)
            if len(scoped_gts) == 0:
                qm["recall_display"] = "N/A (GT=0)"
                qm["f1_display"] = "N/A (GT=0)"
            q_evals.append({
                "query_id": q_doc["query_id"],
                "raw_query": q_doc["query_spec"]["raw_query"],
                "target_classes": sorted(target_set),
                "region_mode": reg_mode,
                "execution_mechanism": q_doc.get("query_execution_mechanism"),
                "model_participation": q_doc.get("model_participation"),
                "gt_count": len(scoped_gts),
                "predicted_count": int(q_doc["predicted_count"]),
                "count_abs_error": abs(int(q_doc["predicted_count"]) - len(scoped_gts)),
                **qm,
            })

        per_board[bkey] = {
            "gt_total": len(gts),
            "arms": arms_eval,
            "fics_query_evaluations": q_evals,
        }

    # Combine S20 + S4 for R15 Dev Promotion Gate
    def _combine(board_keys: Sequence[str], arm_key: str) -> Dict[str, Any]:
        tp = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["tp"]) for b in board_keys)
        fp = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["fp"]) for b in board_keys)
        fn = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["fn"]) for b in board_keys)
        gt_tot = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["gt_total"]) for b in board_keys)
        pred_tot = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["pred_total"]) for b in board_keys)
        ca025 = sum(int(per_board[b]["arms"][arm_key]["class_agnostic_coverage"]["official_025_iou50"]["matched_gt"]) for b in board_keys)
        ca005 = sum(int(per_board[b]["arms"][arm_key]["class_agnostic_coverage"]["pool_005_iou50"]["matched_gt"]) for b in board_keys)
        res = _prf1(tp, fp, fn)
        res.update({
            "gt_total": gt_tot,
            "pred_total": pred_tot,
            "count_abs_error": abs(pred_tot - gt_tot),
            "ca_025_iou50_matched": ca025,
            "ca_025_iou50_recall_pct": round((ca025 / gt_tot) * 100.0, 2) if gt_tot else 0.0,
            "ca_005_iou50_matched": ca005,
            "ca_005_iou50_recall_pct": round((ca005 / gt_tot) * 100.0, 2) if gt_tot else 0.0,
        })
        return res

    s20_s4_combined = {
        "wacv_yolo11n": _combine(("s20", "s4"), "wacv_yolo11n"),
        "fics_yolo11n_r15": _combine(("s20", "s4"), "fics_yolo11n_r15"),
    }
    s20_s4_s1_combined = {
        "wacv_yolo11n": _combine(("s20", "s4", "s1"), "wacv_yolo11n"),
        "fics_yolo11n_r15": _combine(("s20", "s4", "s1"), "fics_yolo11n_r15"),
    }

    wacv_f1_pct = s20_s4_combined["wacv_yolo11n"]["f1_pct"]
    fics_f1_pct = s20_s4_combined["fics_yolo11n_r15"]["f1_pct"]
    s20_fics_tp = int(per_board["s20"]["arms"]["fics_yolo11n_r15"]["class_aware"]["tp"])
    s4_fics_tp = int(per_board["s4"]["arms"]["fics_yolo11n_r15"]["class_aware"]["tp"])

    gate_passed = (fics_f1_pct >= 37.57) and (s20_fics_tp >= 29) and (s4_fics_tp >= 28)

    # Check exact R14 baseline reproducibility
    wacv_s20 = per_board["s20"]["arms"]["wacv_yolo11n"]["class_aware"]
    wacv_s4 = per_board["s4"]["arms"]["wacv_yolo11n"]["class_aware"]
    wacv_s1 = per_board["s1"]["arms"]["wacv_yolo11n"]["class_aware"]
    r14_repro = {
        "s20_matches_r14_29_87_82": (wacv_s20["tp"], wacv_s20["fp"], wacv_s20["fn"]) == (29, 87, 82),
        "s4_matches_r14_28_43_24": (wacv_s4["tp"], wacv_s4["fp"], wacv_s4["fn"]) == (28, 43, 24),
        "s1_matches_r14_codex_37_80_48": (wacv_s1["tp"], wacv_s1["fp"], wacv_s1["fn"]) == (37, 80, 48),
        "observed": {
            "s20": [wacv_s20["tp"], wacv_s20["fp"], wacv_s20["fn"]],
            "s4": [wacv_s4["tp"], wacv_s4["fp"], wacv_s4["fn"]],
            "s1": [wacv_s1["tp"], wacv_s1["fp"], wacv_s1["fn"]],
        },
    }

    report = {
        "schema_version": "r15_paired_eval_report_v1",
        "pre_gt_lock_verified": True,
        "pre_gt_lock_mismatches": mismatches,
        "locked_files_count": len(lock["locked_files"]),
        "web_inference_total_wall_seconds": lock["web_inference_total_wall_seconds"],
        "r14_wacv_reproducibility_check": r14_repro,
        "s20_s4_combined_micro": s20_s4_combined,
        "s20_s4_s1_combined_micro": s20_s4_s1_combined,
        "r15_dev_promotion_gate": {
            "rule": "S20+S4 combined_micro_f1 >= 37.57% (+5.0 pp over 32.57%) AND S20 TP >= 29 AND S4 TP >= 28",
            "fics_s20_s4_micro_f1_pct": fics_f1_pct,
            "wacv_s20_s4_micro_f1_pct": wacv_f1_pct,
            "gain_pp": round(fics_f1_pct - wacv_f1_pct, 2),
            "s20_fics_tp": s20_fics_tp,
            "s4_fics_tp": s4_fics_tp,
            "passed": gate_passed,
            "recommended_web_default_before_s31": "keep_wacv_default_until_codex_s31_verified" if gate_passed else "keep_wacv_default",
        },
        "per_board": per_board,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "r14_wacv_reproducibility_check": r14_repro,
        "s20_s4_combined_micro": s20_s4_combined,
        "r15_dev_promotion_gate": report["r15_dev_promotion_gate"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
