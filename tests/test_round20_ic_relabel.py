"""Unit & config tests for Round 20 targeted IC relabeler (`round20_ic_relabel.py`)."""

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
from pcb_vlm.evaluation.round20_ic_relabel import apply_r20_targeted_ic_relabel  # noqa: E402


def _sample_a1_preds() -> List[Dict[str, Any]]:
    return [
        {"prediction_id": "p_cap", "label": "capacitor", "score": 0.90, "bbox_xyxy": [10.0, 10.0, 30.0, 30.0]},
        {"prediction_id": "p_res", "label": "resistor", "score": 0.85, "bbox_xyxy": [50.0, 10.0, 70.0, 30.0]},
        {"prediction_id": "p_ind", "label": "inductor", "score": 0.72, "bbox_xyxy": [90.0, 10.0, 120.0, 40.0]},
        {"prediction_id": "p_q_exist", "label": "transistor", "score": 0.60, "bbox_xyxy": [200.0, 200.0, 240.0, 240.0]},
        {"prediction_id": "p_ic_to_q", "label": "ic", "score": 0.88, "bbox_xyxy": [300.0, 200.0, 350.0, 250.0]},
        {"prediction_id": "p_ic_dup_q", "label": "ic", "score": 0.79, "bbox_xyxy": [200.0, 200.0, 240.0, 240.0]},
        {"prediction_id": "p_ic_other_stay", "label": "ic", "score": 0.50, "bbox_xyxy": [500.0, 200.0, 550.0, 250.0]},
    ]


def test_r20_relabel_invariants_overlap_abstain_and_query_provenance() -> None:
    preds = _sample_a1_preds()
    probs = {
        "p_ic_to_q": {"ic": 0.05, "diode": 0.02, "transistor": 0.91, "other": 0.02},
        # Overlaps p_q_exist with IoU=1.0 => MUST abstain and keep 'ic'
        "p_ic_dup_q": {"ic": 0.02, "diode": 0.01, "transistor": 0.96, "other": 0.01},
        # top1='other' => NEVER relabels to 'other', keeps 'ic'
        "p_ic_other_stay": {"ic": 0.05, "diode": 0.02, "transistor": 0.03, "other": 0.90},
    }
    res = apply_r20_targeted_ic_relabel(
        a1_predictions=preds,
        probs_by_pred_id=probs,
        t_relabel=0.80,
        m_relabel=0.25,
    )
    out = res["predictions"]
    assert len(out) == len(preds)
    assert [p["prediction_id"] for p in out] == [p["prediction_id"] for p in preds]

    by_id = {p["prediction_id"]: p for p in out}
    assert by_id["p_ic_to_q"]["label"] == "transistor"
    assert by_id["p_ic_to_q"]["score"] == 0.88
    assert by_id["p_ic_to_q"]["bbox_xyxy"] == [300.0, 200.0, 350.0, 250.0]
    assert by_id["p_ic_to_q"]["box_source"] == "r20_ic_targeted_relabel"
    assert by_id["p_ic_to_q"]["r20_relabel_decision"] == "relabel_ic_to_transistor"

    assert by_id["p_ic_dup_q"]["label"] == "ic"
    assert len(res["overlap_abstained"]) == 1
    assert res["overlap_abstained"][0]["conflict_with_prediction_id"] == "p_q_exist"

    assert by_id["p_ic_other_stay"]["label"] == "ic"

    q = execute_composite_query(
        predictions=out,
        query_text="只找晶体管并计数",
        orig_wh=(1000, 1000),
        base_job_mode="COMPOSITE_YOLO",
        detector_preset="fics_yolo11n_r15_r20_a1_ic_relabel",
    )
    assert q["predicted_count"] == 2
    assert q["box_source_breakdown"] == {"r15_official_025": 1, "r20_ic_targeted_relabel": 1}


def test_r20_preset_registered_and_default_unchanged(tmp_path: pathlib.Path) -> None:
    assert DEFAULT_COMPOSITE_DETECTOR_PRESET == "fics_yolo11n_r15"
    assert "fics_yolo11n_r15_r20_a1_ic_relabel" in SUPPORTED_COMPOSITE_DETECTOR_PRESETS
    svc = LiveDemoService(workspace_root=tmp_path / "ws")
    cfg = svc.get_ui_defaults()
    assert cfg["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
