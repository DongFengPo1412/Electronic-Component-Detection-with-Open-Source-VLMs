"""Round 19 CPU-only four-arm calibration, multi-view audit, and geometry upper-bound report.

Strict invariants:
- Pure CPU (`device="cpu"`), zero GPU calls.
- Reads ONLY `s14_*`, `s15_*`, `s2_*`, `s3_*` from R18's `crop_manifest.json` proposal cache.
- NEVER reads or touches `s20`, `s4`, `s1`, `s12`, `s24`, or any other dev/blind board.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image
import torch

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.data.fics_dslr import FICS_RAW_TO_CANONICAL, file_sha256, parse_component_location  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    _prf1,
    compute_iou_xyxy,
    evaluate_composite_board,
)
from pcb_vlm.evaluation.crop_verifier_r18 import load_r18_verifier_checkpoint  # noqa: E402
from pcb_vlm.evaluation.round19_postproc import (  # noqa: E402
    R19_A1_M_REJECT_OTHER,
    R19_A1_T_REJECT_OTHER,
    apply_r19_four_arms,
    compute_r19_crop_probs_for_boxes,
)

R18_MANIFEST = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration18-20261125\crop_dataset_r18\crop_manifest.json")
R18_VERIFIER_WEIGHTS = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\weights\r18_resnet18_crop_verifier_best.pt")
R18_VERIFIER_SHA = "c27119eac5b98d6f8e56a61b280da94ce30bbed4443bb0cfe57c380d46379ca3"

R19_CACHE = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration19-20261125")
R19_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration19-20261125")
FROZEN_PLAN_PATH = R19_ART / "frozen_plan_r19_pre_cal.json"

ALLOWED_VAL_PREFIXES = ("s14_", "s15_", "s2_", "s3_")
CANONICAL_SIDES = ("s14_front1", "s14_back1", "s15_front1", "s15_back1")
TWIN_SIDES = ("s14_front2", "s14_back2", "s15_front2", "s15_back2")
LEGACY_SIDES = ("s2_front", "s2_back", "s3_front")


def _read_csv_gts_with_designators(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    raw_lower = {k.lower(): v for k, v in FICS_RAW_TO_CANONICAL.items()}
    gts: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rt = str(row.get("component_type") or "").strip()
            canon = FICS_RAW_TO_CANONICAL.get(rt) or raw_lower.get(rt.lower())
            if not canon:
                continue
            _, box, _ = parse_component_location(row.get("component_location") or "")
            des = str(row.get("component_designator") or row.get("designator") or row.get("id") or "").strip()
            gts.append({
                "label": canon,
                "raw_type": rt,
                "designator": des,
                "bbox_xyxy": [float(v) for v in box],
            })
    return gts


def _geometry_coverage_stats(
    gts: Sequence[Mapping[str, Any]],
    preds_025: Sequence[Mapping[str, Any]],
    pool_005: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    dq_gts = [g for g in gts if g["label"] in ("diode", "transistor")]
    low_only = [p for p in pool_005 if float(p.get("score", 0.0)) < 0.25]

    def _cov(pool: Sequence[Mapping[str, Any]], class_aware: bool) -> Dict[str, int]:
        d_cov = 0
        q_cov = 0
        for g in dq_gts:
            hit = False
            for p in pool:
                if class_aware and str(p.get("label") or "").lower() != g["label"]:
                    continue
                if compute_iou_xyxy(g["bbox_xyxy"], p["bbox_xyxy"]) >= 0.50:
                    hit = True
                    break
            if hit:
                if g["label"] == "diode":
                    d_cov += 1
                else:
                    q_cov += 1
        return {"diode": d_cov, "transistor": q_cov, "dq_total": d_cov + q_cov}

    d_gt = sum(1 for g in dq_gts if g["label"] == "diode")
    q_gt = sum(1 for g in dq_gts if g["label"] == "transistor")
    return {
        "gt_counts": {"diode": d_gt, "transistor": q_gt, "dq_total": d_gt + q_gt},
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


def _aggregate_eval(eval_list: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tp = sum(e["class_aware"]["tp"] for e in eval_list)
    fp = sum(e["class_aware"]["fp"] for e in eval_list)
    fn = sum(e["class_aware"]["fn"] for e in eval_list)
    overall = _prf1(tp, fp, fn)

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in ("capacitor", "resistor", "inductor", "transistor", "diode", "ic"):
        ctp = sum(e["per_class"][cls]["tp"] for e in eval_list)
        cfp = sum(e["per_class"][cls]["fp"] for e in eval_list)
        cfn = sum(e["per_class"][cls]["fn"] for e in eval_list)
        per_class[cls] = {"tp": ctp, "fp": cfp, "fn": cfn, **_prf1(ctp, cfp, cfn)}

    d_tp = per_class["diode"]["tp"]
    q_tp = per_class["transistor"]["tp"]
    d_fp = per_class["diode"]["fp"]
    q_fp = per_class["transistor"]["fp"]
    d_fn = per_class["diode"]["fn"]
    q_fn = per_class["transistor"]["fn"]
    dq_prf1 = _prf1(d_tp + q_tp, d_fp + q_fp, d_fn + q_fn)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **overall,
        "dq": {"tp": d_tp + q_tp, "fp": d_fp + q_fp, "fn": d_fn + q_fn, **dq_prf1},
        "ic": {"tp": per_class["ic"]["tp"], "fp": per_class["ic"]["fp"], "fn": per_class["ic"]["fn"]},
        "per_class": per_class,
    }


def main() -> None:
    t0 = time.perf_counter()
    frozen_plan = json.loads(FROZEN_PLAN_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(R18_MANIFEST.read_text(encoding="utf-8"))
    prop_idx = manifest["proposal_cache_index"]

    val_entries: Dict[str, Dict[str, Any]] = {}
    for k, v in sorted(prop_idx.items()):
        if v["role"] in ("val", "r15_frozen_val"):
            assert any(k.startswith(p) for p in ALLOWED_VAL_PREFIXES), f"Unexpected val board {k}"
            val_entries[k] = v

    assert set(CANONICAL_SIDES).issubset(val_entries.keys())
    assert set(TWIN_SIDES).issubset(val_entries.keys())
    assert set(LEGACY_SIDES).issubset(val_entries.keys())

    # 1. Load data + multi-view physical-board audit
    board_data: Dict[str, Dict[str, Any]] = {}
    multi_view_views: Dict[str, Any] = {}
    for stem, entry in val_entries.items():
        csv_path = pathlib.Path(str(entry["csv_path"]))
        tif_path = pathlib.Path(str(entry["tif_path"]))
        preds_025_path = pathlib.Path(str(entry["preds_025_path"]))
        prop_dir = preds_025_path.parent
        pool_005_path = prop_dir / "yolo_low_conf_pool_005.json"
        summary_path = prop_dir / "yolo_run_summary.json"

        gts = _read_csv_gts_with_designators(csv_path)
        preds_025 = json.loads(preds_025_path.read_text(encoding="utf-8"))
        pool_005 = json.loads(pool_005_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        roi = summary["geometry"]["roi_bbox_half_open"]

        cls_counts = dict(collections.Counter(g["label"] for g in gts))
        board_data[stem] = {
            "stem": stem,
            "physical_board": str(entry.get("physical_board") or stem.split("_")[0]),
            "role": entry["role"],
            "tif_path": tif_path,
            "csv_path": csv_path,
            "gts": gts,
            "preds_025": preds_025,
            "pool_005": pool_005,
            "roi": roi,
        }
        multi_view_views[stem] = {
            "physical_board": board_data[stem]["physical_board"],
            "role": entry["role"],
            "gt_total": len(gts),
            "gt_by_class": cls_counts,
            "preds_025_count": len(preds_025),
            "low_conf_005_to_025_count": sum(1 for p in pool_005 if float(p.get("score", 0.0)) < 0.25),
        }

    # Check view1 vs view2 duplication on s14 and s15
    pair_audits: List[Dict[str, Any]] = []
    for base in ("s14_front", "s14_back", "s15_front", "s15_back"):
        v1 = f"{base}1"
        v2 = f"{base}2"
        g1 = board_data[v1]["gts"]
        g2 = board_data[v2]["gts"]
        c1 = collections.Counter(g["label"] for g in g1)
        c2 = collections.Counter(g["label"] for g in g2)
        d1 = collections.Counter((g["label"], g["designator"]) for g in g1 if g["designator"])
        d2 = collections.Counter((g["label"], g["designator"]) for g in g2 if g["designator"])
        pair_audits.append({
            "physical_side": base,
            "view1": v1,
            "view2": v2,
            "view1_gt_total": len(g1),
            "view2_gt_total": len(g2),
            "view1_dq": {"diode": c1["diode"], "transistor": c1["transistor"]},
            "view2_dq": {"diode": c2["diode"], "transistor": c2["transistor"]},
            "class_counts_identical": dict(c1) == dict(c2),
            "designator_multiset_identical": (d1 == d2) if (d1 or d2) else None,
            "note": "Same physical PCB side imaged under two camera views; canonical set uses view1 only to prevent double-counting.",
        })

    multi_view_audit = {
        "schema_version": "r19_s14_s15_multi_view_audit_v1",
        "physical_boards_count": 2,
        "physical_sides_count": 4,
        "total_views_count": 8,
        "canonical_physical_sides": list(CANONICAL_SIDES),
        "twin_views": list(TWIN_SIDES),
        "legacy_val_sides": list(LEGACY_SIDES),
        "side_pair_duplication_audit": pair_audits,
        "per_view_summary": multi_view_views,
    }
    (R19_ART / "calibration" / "r19_s14_s15_multi_view_audit.json").write_text(
        json.dumps(multi_view_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 2. Precompute CPU ResNet-18 probabilities once for all needed target crops
    probs_cache_path = R19_CACHE / "cpu_calibration_cache" / "val_crop_probs_cpu.json"
    if probs_cache_path.is_file():
        cached_probs = json.loads(probs_cache_path.read_text(encoding="utf-8"))
        probs_by_stem = cached_probs["probs_by_stem"]
        print("[CPU_CAL] Loaded cached CPU crop probs:", probs_cache_path, flush=True)
    else:
        cpu_dev = torch.device("cpu")
        model, _, actual_sha = load_r18_verifier_checkpoint(
            R18_VERIFIER_WEIGHTS, expected_sha256=R18_VERIFIER_SHA, device=cpu_dev
        )
        assert actual_sha == R18_VERIFIER_SHA
        probs_by_stem = {}
        total_crops = 0
        for stem, bd in board_data.items():
            target_boxes = [
                p for p in bd["preds_025"]
                if str(p.get("label") or "").strip().lower() == "ic"
            ]
            target_boxes.extend(
                p for p in bd["pool_005"]
                if float(p.get("score", 0.0)) < 0.25
                and str(p.get("label") or "").strip().lower() in ("diode", "transistor", "ic")
            )
            probs_map = compute_r19_crop_probs_for_boxes(
                image_path=bd["tif_path"], boxes=target_boxes, model=model, device=cpu_dev, batch_size=32
            )
            probs_by_stem[stem] = probs_map
            total_crops += len(probs_map)
            print(f"[CPU_CAL] Scored {len(probs_map)} crops on CPU for {stem}", flush=True)
        probs_cache_path.write_text(
            json.dumps({"device": "cpu", "total_unique_crops": total_crops, "probs_by_stem": probs_by_stem}, indent=2),
            encoding="utf-8",
        )

    # 3. Geometry coverage upper-bound report
    geom_by_view = {}
    for stem, bd in board_data.items():
        geom_by_view[stem] = _geometry_coverage_stats(bd["gts"], bd["preds_025"], bd["pool_005"])

    def _sum_geom(stems: Sequence[str]) -> Dict[str, Any]:
        out = {
            "gt_counts": {"diode": 0, "transistor": 0, "dq_total": 0},
            "class_agnostic_iou50": {
                "official_025": {"diode": 0, "transistor": 0, "dq_total": 0},
                "low_005_to_025": {"diode": 0, "transistor": 0, "dq_total": 0},
                "combined_ge_005": {"diode": 0, "transistor": 0, "dq_total": 0},
            },
            "class_aware_iou50": {
                "official_025": {"diode": 0, "transistor": 0, "dq_total": 0},
                "low_005_to_025": {"diode": 0, "transistor": 0, "dq_total": 0},
                "combined_ge_005": {"diode": 0, "transistor": 0, "dq_total": 0},
            },
        }
        for s in stems:
            g = geom_by_view[s]
            for k in ("diode", "transistor", "dq_total"):
                out["gt_counts"][k] += g["gt_counts"][k]
                for mode in ("class_agnostic_iou50", "class_aware_iou50"):
                    for pool_k in ("official_025", "low_005_to_025", "combined_ge_005"):
                        out[mode][pool_k][k] += g[mode][pool_k][k]
        return out

    geom_report = {
        "schema_version": "r19_geometry_coverage_report_v1",
        "s14_s15_canonical_4_sides": _sum_geom(CANONICAL_SIDES),
        "s14_s15_all_8_views": _sum_geom(CANONICAL_SIDES + TWIN_SIDES),
        "s2_s3_legacy_val": _sum_geom(LEGACY_SIDES),
        "per_view": geom_by_view,
    }
    (R19_ART / "calibration" / "r19_geometry_coverage_report.json").write_text(
        json.dumps(geom_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 4. Evaluate A0, A1, and all 21 A2/A3 configs on CPU
    grid = frozen_plan["a2_candidate_grid"]

    # First compute A0 and A1 once across all 11 views
    a0_evals: Dict[str, Any] = {}
    a1_evals: Dict[str, Any] = {}
    dummy_rule = grid[0]
    for stem, bd in board_data.items():
        res = apply_r19_four_arms(
            official_preds_025=bd["preds_025"],
            low_conf_pool_005=bd["pool_005"],
            a2_rule_spec=dummy_rule,
            probs_by_pred_id=probs_by_stem[stem],
            t_reject_other=R19_A1_T_REJECT_OTHER,
            m_reject_other=R19_A1_M_REJECT_OTHER,
        )
        a0_evals[stem] = evaluate_composite_board(
            gts=bd["gts"], predictions_025=res["a0_predictions"], low_conf_pool_005=bd["pool_005"], roi_bbox_half_open=bd["roi"]
        )
        a1_evals[stem] = evaluate_composite_board(
            gts=bd["gts"], predictions_025=res["a1_predictions"], low_conf_pool_005=bd["pool_005"], roi_bbox_half_open=bd["roi"]
        )

    a0_canon = _aggregate_eval([a0_evals[s] for s in CANONICAL_SIDES])
    a0_twin = _aggregate_eval([a0_evals[s] for s in TWIN_SIDES])
    a0_all8 = _aggregate_eval([a0_evals[s] for s in CANONICAL_SIDES + TWIN_SIDES])
    a0_s2s3 = _aggregate_eval([a0_evals[s] for s in LEGACY_SIDES])

    a1_canon = _aggregate_eval([a1_evals[s] for s in CANONICAL_SIDES])
    a1_all8 = _aggregate_eval([a1_evals[s] for s in CANONICAL_SIDES + TWIN_SIDES])
    a1_s2s3 = _aggregate_eval([a1_evals[s] for s in LEGACY_SIDES])

    # Evaluate each of the 21 A2 rules
    grid_records: List[Dict[str, Any]] = []
    for rule in grid:
        a2_evals: Dict[str, Any] = {}
        a3_evals: Dict[str, Any] = {}
        dedup_canon = 0
        accepted_canon = 0
        for stem, bd in board_data.items():
            res = apply_r19_four_arms(
                official_preds_025=bd["preds_025"],
                low_conf_pool_005=bd["pool_005"],
                a2_rule_spec=rule,
                probs_by_pred_id=probs_by_stem[stem],
                t_reject_other=R19_A1_T_REJECT_OTHER,
                m_reject_other=R19_A1_M_REJECT_OTHER,
            )
            a2_evals[stem] = evaluate_composite_board(
                gts=bd["gts"], predictions_025=res["a2_predictions"], low_conf_pool_005=bd["pool_005"], roi_bbox_half_open=bd["roi"]
            )
            a3_evals[stem] = evaluate_composite_board(
                gts=bd["gts"], predictions_025=res["a3_predictions"], low_conf_pool_005=bd["pool_005"], roi_bbox_half_open=bd["roi"]
            )
            if stem in CANONICAL_SIDES:
                dedup_canon += len(res["a2_dedup_suppressed_boxes"])
                accepted_canon += len(res["a2_accepted_low_boxes"])

        a2_canon = _aggregate_eval([a2_evals[s] for s in CANONICAL_SIDES])
        a2_twin = _aggregate_eval([a2_evals[s] for s in TWIN_SIDES])
        a2_all8 = _aggregate_eval([a2_evals[s] for s in CANONICAL_SIDES + TWIN_SIDES])
        a2_s14 = _aggregate_eval([a2_evals[s] for s in ("s14_front1", "s14_front2", "s14_back1", "s14_back2")])
        a2_s15 = _aggregate_eval([a2_evals[s] for s in ("s15_front1", "s15_front2", "s15_back1", "s15_back2")])
        a2_s2s3 = _aggregate_eval([a2_evals[s] for s in LEGACY_SIDES])

        a3_canon = _aggregate_eval([a3_evals[s] for s in CANONICAL_SIDES])
        a3_all8 = _aggregate_eval([a3_evals[s] for s in CANONICAL_SIDES + TWIN_SIDES])
        a3_s2s3 = _aggregate_eval([a3_evals[s] for s in LEGACY_SIDES])

        s14_s15_dq_tp_gain = a2_all8["dq"]["tp"] - a0_all8["dq"]["tp"]
        s14_s15_dq_fp_added = a2_all8["dq"]["fp"] - a0_all8["dq"]["fp"]
        s14_s15_net_gain = s14_s15_dq_tp_gain - s14_s15_dq_fp_added

        s2s3_dq_tp_gain = a2_s2s3["dq"]["tp"] - a0_s2s3["dq"]["tp"]
        s2s3_dq_fp_added = a2_s2s3["dq"]["fp"] - a0_s2s3["dq"]["fp"]
        s2s3_added_absent_dq_fp = 0
        for s in LEGACY_SIDES:
            for cls in ("diode", "transistor"):
                if board_data[s]["gts"] and sum(1 for g in board_data[s]["gts"] if g["label"] == cls) == 0:
                    s2s3_added_absent_dq_fp += (a2_evals[s]["per_class"][cls]["fp"] - a0_evals[s]["per_class"][cls]["fp"])

        dq_tp_preserved = (a2_all8["dq"]["tp"] >= a0_all8["dq"]["tp"]) and (a2_s2s3["dq"]["tp"] >= a0_s2s3["dq"]["tp"])
        qualifies = (
            dq_tp_preserved
            and s14_s15_dq_tp_gain >= 1
            and s2s3_dq_fp_added <= 1
            and s14_s15_dq_fp_added <= 3 * s14_s15_dq_tp_gain
        )
        fail_reasons: List[str] = []
        if not dq_tp_preserved:
            fail_reasons.append("dq_tp_not_preserved")
        if s14_s15_dq_tp_gain < 1:
            fail_reasons.append(f"s14_s15_dq_tp_gain={s14_s15_dq_tp_gain}<1")
        if s2s3_dq_fp_added > 1:
            fail_reasons.append(f"s2s3_dq_fp_added={s2s3_dq_fp_added}>1")
        if s14_s15_dq_tp_gain >= 1 and s14_s15_dq_fp_added > 3 * s14_s15_dq_tp_gain:
            fail_reasons.append(f"s14_s15_dq_fp_added={s14_s15_dq_fp_added}>3*gain")

        grid_records.append({
            "rule": rule,
            "qualifies": qualifies,
            "fail_reasons": fail_reasons,
            "s14_s15_physical_boards": {
                "s14_dq_tp": a2_s14["dq"]["tp"],
                "s14_dq_fp": a2_s14["dq"]["fp"],
                "s15_dq_tp": a2_s15["dq"]["tp"],
                "s15_dq_fp": a2_s15["dq"]["fp"],
                "a2_f1_pct": a2_all8["f1_pct"],
                "a3_f1_pct": a3_all8["f1_pct"],
                "a2_dq_tp": a2_all8["dq"]["tp"],
                "a2_dq_fp": a2_all8["dq"]["fp"],
                "a2_dq_f1_pct": a2_all8["dq"]["f1_pct"],
                "dq_tp_gain": s14_s15_dq_tp_gain,
                "dq_fp_added": s14_s15_dq_fp_added,
                "net_dq_gain": s14_s15_net_gain,
            },
            "s14_s15_canonical": {
                "a2_f1_pct": a2_canon["f1_pct"],
                "a3_f1_pct": a3_canon["f1_pct"],
                "a2_dq_tp": a2_canon["dq"]["tp"],
                "a2_dq_fp": a2_canon["dq"]["fp"],
                "accepted_low_count": accepted_canon,
                "dedup_suppressed_count": dedup_canon,
            },
            "s14_s15_twin": {
                "a2_dq_tp": a2_twin["dq"]["tp"],
                "a2_dq_fp": a2_twin["dq"]["fp"],
                "dq_tp_gain": a2_twin["dq"]["tp"] - a0_twin["dq"]["tp"],
            },
            "s2_s3_legacy_val": {
                "a2_f1_pct": a2_s2s3["f1_pct"],
                "a3_f1_pct": a3_s2s3["f1_pct"],
                "a2_dq_tp": a2_s2s3["dq"]["tp"],
                "a2_dq_fp": a2_s2s3["dq"]["fp"],
                "dq_tp_gain": s2s3_dq_tp_gain,
                "dq_fp_added": s2s3_dq_fp_added,
                "added_absent_dq_fp": s2s3_added_absent_dq_fp,
            },
        })

    def _rank_key(rec: Mapping[str, Any]) -> Tuple[Any, ...]:
        pb = rec["s14_s15_physical_boards"]
        s23 = rec["s2_s3_legacy_val"]
        rule = rec["rule"]
        return (
            pb["net_dq_gain"],
            pb["a2_dq_f1_pct"],
            pb["dq_tp_gain"],
            -s23["dq_fp_added"],
            float(rule.get("t_confirm", 0.0)),
            1 if rule["rule_family"] == "cls_same_class" else 0,
            float(rule["score_min"]),
        )

    qualifying = [r for r in grid_records if r["qualifies"]]
    qualifying.sort(key=_rank_key, reverse=True)

    selected_record = None
    selection_reason = ""
    if not qualifying:
        stop_loss_triggered = True
        selection_reason = "No A2 candidate met all qualification criteria; stop-loss triggered (A0/A1 only)."
    else:
        stop_loss_triggered = False
        yolo_q = [r for r in qualifying if r["rule"]["rule_family"] == "yolo_only"]
        cls_q = [r for r in qualifying if r["rule"]["rule_family"] != "yolo_only"]
        if not yolo_q:
            selected_record = cls_q[0]
            selection_reason = "Selected top qualifying classification rule (no qualifying yolo_only baseline)."
        elif not cls_q:
            selected_record = yolo_q[0]
            selection_reason = "Selected top qualifying yolo_only rule (no qualifying classification rule)."
        else:
            best_y = yolo_q[0]
            best_c = cls_q[0]
            c_net = best_c["s14_s15_physical_boards"]["net_dq_gain"]
            y_net = best_y["s14_s15_physical_boards"]["net_dq_gain"]
            c_fp = best_c["s14_s15_physical_boards"]["dq_fp_added"]
            y_fp = best_y["s14_s15_physical_boards"]["dq_fp_added"]
            c_s23_fp = best_c["s2_s3_legacy_val"]["dq_fp_added"]
            y_s23_fp = best_y["s2_s3_legacy_val"]["dq_fp_added"]
            if (c_net > y_net) or (c_net == y_net and c_fp < y_fp and c_s23_fp <= y_s23_fp):
                selected_record = best_c
                selection_reason = f"Classification rule {best_c['rule']['rule_id']} strictly outperformed simpler yolo_only {best_y['rule']['rule_id']}."
            else:
                selected_record = best_y
                selection_reason = f"Simplicity rule selected {best_y['rule']['rule_id']} because classification rules did not strictly improve net D/Q gain over yolo_only."

    cal_report = {
        "schema_version": "r19_cpu_four_arm_calibration_v1",
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "gpu_calls": 0,
        "baselines": {
            "a0_s14_s15_canonical": a0_canon,
            "a1_s14_s15_canonical": a1_canon,
            "a0_s14_s15_all8": a0_all8,
            "a1_s14_s15_all8": a1_all8,
            "a0_s2_s3_legacy": a0_s2s3,
            "a1_s2_s3_legacy": a1_s2s3,
        },
        "a2_grid_results": grid_records,
        "qualifying_count": len(qualifying),
        "stop_loss_triggered": stop_loss_triggered,
        "selection_reason": selection_reason,
        "selected_record": selected_record,
    }
    cal_path = R19_ART / "calibration" / "r19_cpu_four_arm_calibration.json"
    cal_path.write_text(json.dumps(cal_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    selected_recipe = {
        "schema_version": "r19_selected_recipe_frozen_v1",
        "calibration_report_path": str(cal_path),
        "calibration_report_sha256": file_sha256(cal_path),
        "a1_rule": {"t_reject_other": R19_A1_T_REJECT_OTHER, "m_reject_other": R19_A1_M_REJECT_OTHER},
        "a2_qualified": not stop_loss_triggered,
        "stop_loss_triggered": stop_loss_triggered,
        "selection_reason": selection_reason,
        "selected_a2_rule": selected_record["rule"] if selected_record else None,
        "active_web_preset": "fics_yolo11n_r15_r19_a3_combined" if not stop_loss_triggered else "fics_yolo11n_r15_r19_a1_ic_reject",
    }
    (R19_ART / "calibration" / "r19_selected_recipe_frozen.json").write_text(
        json.dumps(selected_recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "wall_seconds": cal_report["wall_seconds"],
        "a0_canon_f1": a0_canon["f1_pct"],
        "a1_canon_f1": a1_canon["f1_pct"],
        "a0_canon_ic_fp": a0_canon["ic"]["fp"],
        "a1_canon_ic_fp": a1_canon["ic"]["fp"],
        "a0_canon_dq_tp": a0_canon["dq"]["tp"],
        "a0_s2s3_f1": a0_s2s3["f1_pct"],
        "a1_s2s3_f1": a1_s2s3["f1_pct"],
        "qualifying_count": len(qualifying),
        "stop_loss_triggered": stop_loss_triggered,
        "selected_rule": selected_recipe["selected_a2_rule"],
        "active_web_preset": selected_recipe["active_web_preset"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
