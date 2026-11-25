"""Unit & config tests for Round 21 448px weak-class supplement merge."""

from __future__ import annotations

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
from pcb_vlm.evaluation.round21_multiscale_merge import (  # noqa: E402
    compute_one_to_one_coverage_split,
    merge_r21_448_dq_supplement,
)


def test_r21_448_merge_invariants_and_conflict_suppression() -> None:
    a0 = [
        {"prediction_id": "a0_cap", "label": "capacitor", "score": 0.90, "bbox_xyxy": [10.0, 10.0, 30.0, 30.0]},
        {"prediction_id": "a0_res", "label": "resistor", "score": 0.85, "bbox_xyxy": [50.0, 10.0, 70.0, 30.0]},
        {"prediction_id": "a0_ind", "label": "inductor", "score": 0.72, "bbox_xyxy": [90.0, 10.0, 120.0, 40.0]},
        {"prediction_id": "a0_ic", "label": "ic", "score": 0.80, "bbox_xyxy": [200.0, 200.0, 250.0, 250.0]},
    ]
    a1 = [p for p in a0]
    preds_448 = [
        # Valid new diode >= 0.25
        {"prediction_id": "p448_d_ok", "label": "diode", "score": 0.44, "bbox_xyxy": [400.0, 400.0, 430.0, 430.0], "tile_id": 17},
        # Low conf < 0.25 -> ignored
        {"prediction_id": "p448_d_low", "label": "diode", "score": 0.21, "bbox_xyxy": [500.0, 400.0, 530.0, 430.0], "tile_id": 18},
        # Non-D/Q class -> ignored
        {"prediction_id": "p448_cap_ign", "label": "capacitor", "score": 0.95, "bbox_xyxy": [600.0, 400.0, 630.0, 430.0], "tile_id": 19},
        # Overlaps A0 IC with IoU=1.0 -> suppressed as cross-class conflict
        {"prediction_id": "p448_q_conf", "label": "transistor", "score": 0.65, "bbox_xyxy": [200.0, 200.0, 250.0, 250.0], "tile_id": 20},
    ]
    merged = merge_r21_448_dq_supplement(a0_preds_640=a0, a1_preds_640=a1, preds_448_025=preds_448)
    assert merged["summary"]["accepted_448_count"] == 1
    assert merged["summary"]["conflict_suppressed_count"] == 1
    assert merged["conflict_suppressed"][0]["conflict_reason"] == "overlap_a0_cross_class"

    acc = merged["accepted_448_boxes"][0]
    assert acc["box_source"] == "r21_448px_dq_supplement"
    assert acc["is_multiscale_448_supplement"] is True
    assert acc["scale_tile_size"] == 448
    assert acc["source_tile_id"] == 17

    q = execute_composite_query(
        predictions=merged["a1_plus_448"],
        query_text="只找二极管并计数",
        orig_wh=(1000, 1000),
        base_job_mode="COMPOSITE_YOLO",
        detector_preset="fics_yolo11n_r15_r21_a1_plus_448",
    )
    assert q["predicted_count"] == 1
    assert q["predictions"][0]["is_multiscale_448_supplement"] is True
    assert q["predictions"][0]["source_tile_id"] == 17


def test_r21_one_to_one_coverage_split() -> None:
    gts = [
        {"label": "diode", "bbox_xyxy": [10.0, 10.0, 30.0, 30.0]},
        {"label": "transistor", "bbox_xyxy": [100.0, 100.0, 130.0, 130.0]},
        {"label": "diode", "bbox_xyxy": [200.0, 200.0, 230.0, 230.0]},
    ]
    pool_640 = [{"prediction_id": "p640", "label": "ic", "score": 0.12, "bbox_xyxy": [10.0, 10.0, 30.0, 30.0]}]
    pool_448 = [{"prediction_id": "p448", "label": "transistor", "score": 0.30, "bbox_xyxy": [100.0, 100.0, 130.0, 130.0]}]
    cov = compute_one_to_one_coverage_split(
        gts=gts,
        pool_640_005=pool_640,
        pool_448_005=pool_448,
        preds_448_025=pool_448,
    )
    assert cov["summary"]["covered_by_640_low_pool"] == 1
    assert cov["summary"]["newly_covered_only_by_448_pool005"] == 1
    assert cov["summary"]["newly_covered_only_by_448_off025"] == 1
    assert cov["summary"]["missed_by_both"] == 1


def test_r21_preset_registered_and_default_unchanged(tmp_path: pathlib.Path) -> None:
    assert DEFAULT_COMPOSITE_DETECTOR_PRESET == "fics_yolo11n_r15"
    assert "fics_yolo11n_r15_r21_a1_plus_448" in SUPPORTED_COMPOSITE_DETECTOR_PRESETS
    svc = LiveDemoService(workspace_root=tmp_path / "ws")
    cfg = svc.get_ui_defaults()
    assert cfg["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
