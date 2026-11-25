"""Unit & HTTP integration tests for Round 19 post-processing (A1 IC-reject & A2/A3 low-conf D/Q)."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any, Dict, List

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    DEFAULT_COMPOSITE_DETECTOR_PRESET,
    SUPPORTED_COMPOSITE_DETECTOR_PRESETS,
    execute_composite_query,
)
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402
from pcb_vlm.evaluation.round19_postproc import (  # noqa: E402
    apply_r19_a1_ic_reject_only,
    apply_r19_four_arms,
    select_r19_a2_low_conf_candidates,
)


def _sample_official_025() -> List[Dict[str, Any]]:
    return [
        {"prediction_id": "p_cap", "label": "capacitor", "score": 0.91, "bbox_xyxy": [10.0, 10.0, 30.0, 30.0]},
        {"prediction_id": "p_res", "label": "resistor", "score": 0.88, "bbox_xyxy": [50.0, 10.0, 70.0, 30.0]},
        {"prediction_id": "p_ind", "label": "inductor", "score": 0.77, "bbox_xyxy": [90.0, 10.0, 120.0, 40.0]},
        {"prediction_id": "p_ic_drop", "label": "ic", "score": 0.62, "bbox_xyxy": [200.0, 200.0, 260.0, 260.0]},
        {"prediction_id": "p_ic_keep", "label": "ic", "score": 0.55, "bbox_xyxy": [300.0, 200.0, 360.0, 260.0]},
        {"prediction_id": "p_q_a0", "label": "transistor", "score": 0.41, "bbox_xyxy": [400.0, 200.0, 440.0, 240.0]},
    ]


def _sample_probs() -> Dict[str, Dict[str, float]]:
    return {
        # top1='other' 0.85, top2='ic' 0.10 => margin 0.75 => drop
        "p_ic_drop": {"ic": 0.10, "diode": 0.03, "transistor": 0.02, "other": 0.85},
        # top1='transistor' 0.92 => in A1 cross-class relabel is DISABLED, so p_ic_keep stays 'ic'!
        "p_ic_keep": {"ic": 0.05, "diode": 0.02, "transistor": 0.92, "other": 0.01},
        # low-conf diode confirmed
        "low_d_ok": {"ic": 0.05, "diode": 0.80, "transistor": 0.05, "other": 0.10},
        # low-conf transistor abstained (top1=other, below reject or mismatch)
        "low_q_abstain": {"ic": 0.40, "diode": 0.10, "transistor": 0.35, "other": 0.15},
        # low-conf IC with high transistor prob
        "low_ic_to_q": {"ic": 0.05, "diode": 0.03, "transistor": 0.90, "other": 0.02},
        # low-conf diode overlapping A0 p_q_a0
        "low_d_dup_a0": {"ic": 0.05, "diode": 0.88, "transistor": 0.02, "other": 0.05},
    }


def test_r19_a1_ic_only_reject_and_no_relabel() -> None:
    preds = _sample_official_025()
    res = apply_r19_a1_ic_reject_only(official_preds_025=preds, probs_by_pred_id=_sample_probs())
    kept_ids = [p["prediction_id"] for p in res["predictions"]]
    assert kept_ids == ["p_cap", "p_res", "p_ind", "p_ic_keep", "p_q_a0"]
    # p_ic_keep MUST NOT be relabeled to transistor in A1
    ic_keep = next(p for p in res["predictions"] if p["prediction_id"] == "p_ic_keep")
    assert ic_keep["label"] == "ic"
    assert len(res["dropped_ic_boxes"]) == 1
    assert res["dropped_ic_boxes"][0]["prediction_id"] == "p_ic_drop"


def test_r19_a2_rules_abstention_and_a0_dedup() -> None:
    preds = _sample_official_025()
    low_pool = [
        {"prediction_id": "low_d_ok", "label": "diode", "score": 0.18, "bbox_xyxy": [500.0, 500.0, 530.0, 530.0]},
        {"prediction_id": "low_q_abstain", "label": "transistor", "score": 0.16, "bbox_xyxy": [600.0, 500.0, 630.0, 530.0]},
        {"prediction_id": "low_ic_to_q", "label": "ic", "score": 0.14, "bbox_xyxy": [700.0, 500.0, 740.0, 540.0]},
        {"prediction_id": "low_d_dup_a0", "label": "diode", "score": 0.22, "bbox_xyxy": [400.0, 200.0, 440.0, 240.0]},
        {"prediction_id": "low_cap_ignored", "label": "capacitor", "score": 0.23, "bbox_xyxy": [800.0, 500.0, 820.0, 520.0]},
    ]
    probs = _sample_probs()

    # 1. yolo_only keeps low_d_ok + low_q_abstain; suppresses low_d_dup_a0 against A0; ignores low_ic_to_q & low_cap_ignored
    y_res = select_r19_a2_low_conf_candidates(
        official_preds_025=preds,
        low_conf_pool_005=low_pool,
        rule_spec={"rule_id": "yolo_only_s10", "rule_family": "yolo_only", "score_min": 0.10},
    )
    assert [b["prediction_id"] for b in y_res["accepted_low_boxes"]] == ["low_d_ok", "low_q_abstain"]
    assert len(y_res["dedup_suppressed_boxes"]) == 1
    assert y_res["dedup_suppressed_boxes"][0]["conflict_reason"] == "overlap_a0_cross_class"

    # 2. cls_same_class: abstained low_q_abstain MUST NOT be added
    c_res = select_r19_a2_low_conf_candidates(
        official_preds_025=preds,
        low_conf_pool_005=low_pool,
        rule_spec={"rule_id": "cls_same_s10", "rule_family": "cls_same_class", "score_min": 0.10, "t_confirm": 0.65, "m_confirm": 0.15},
        probs_by_pred_id=probs,
    )
    assert [b["prediction_id"] for b in c_res["accepted_low_boxes"]] == ["low_d_ok"]

    # 3. cls_same_or_ic_to_dq: also rescues low_ic_to_q as transistor
    c2_res = select_r19_a2_low_conf_candidates(
        official_preds_025=preds,
        low_conf_pool_005=low_pool,
        rule_spec={"rule_id": "cls_ic_s10", "rule_family": "cls_same_or_ic_to_dq", "score_min": 0.10, "t_confirm": 0.65, "m_confirm": 0.15},
        probs_by_pred_id=probs,
    )
    assert [(b["prediction_id"], b["label"]) for b in c2_res["accepted_low_boxes"]] == [
        ("low_d_ok", "diode"),
        ("low_ic_to_q", "transistor"),
    ]


def test_r19_four_arms_preserve_untouched_and_provenance(tmp_path: pathlib.Path) -> None:
    preds = _sample_official_025()
    low_pool = [
        {"prediction_id": "low_d_ok", "label": "diode", "score": 0.18, "bbox_xyxy": [500.0, 500.0, 530.0, 530.0]},
    ]
    arms = apply_r19_four_arms(
        official_preds_025=preds,
        low_conf_pool_005=low_pool,
        a2_rule_spec={"rule_id": "cls_same_s10", "rule_family": "cls_same_class", "score_min": 0.10, "t_confirm": 0.65, "m_confirm": 0.15},
        probs_by_pred_id=_sample_probs(),
    )
    assert [p["prediction_id"] for p in arms["a3_predictions"]] == ["p_cap", "p_res", "p_ind", "p_ic_keep", "p_q_a0", "low_d_ok"]

    q = execute_composite_query(
        query_text="只找二极管并计数",
        predictions=arms["a3_predictions"],
        orig_wh=(1000, 1000),
        base_job_mode="COMPOSITE_YOLO",
        detector_preset="fics_yolo11n_r15_r19_a3_combined",
    )
    assert q["predicted_count"] == 1
    assert q["low_conf_supplement_count"] == 1
    assert q["box_source_breakdown"] == {"r19_low_conf_dq_cls": 1}
    assert q["gpu_called_for_query"] is False


def test_r19_presets_registered_and_default_unchanged(tmp_path: pathlib.Path) -> None:
    assert DEFAULT_COMPOSITE_DETECTOR_PRESET == "fics_yolo11n_r15"
    assert "fics_yolo11n_r15_r19_a1_ic_reject" in SUPPORTED_COMPOSITE_DETECTOR_PRESETS
    assert "fics_yolo11n_r15_r19_a3_combined" in SUPPORTED_COMPOSITE_DETECTOR_PRESETS

    svc = LiveDemoService(workspace_root=tmp_path / "ws")
    cfg = svc.get_ui_defaults()
    assert cfg["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
