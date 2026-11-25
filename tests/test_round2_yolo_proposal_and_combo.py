"""Unit tests for Round 2 YOLO11n class-agnostic proposal locator and YOLO+Qwen3 classifier policy."""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.candidate_verifier import apply_verifier_filter_policy  # noqa: E402
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    _match_preds_with_gt_class_agnostic,
    validate_model_and_mode,
)
from pcb_vlm.inference.yolo_proposal_runner import class_agnostic_nms_with_dedup_trace  # noqa: E402


def test_class_agnostic_nms_tie_breaking_and_dedup_trace() -> None:
    proposals = [
        {
            "pre_nms_id": "b_raw0002",
            "tile_id": 5,
            "call_id": "yolo_T05",
            "tile_box_index": 0,
            "score": 0.80,
            "bbox_xyxy": [100.0, 100.0, 200.0, 200.0],
            "local_bbox_xyxy": [10.0, 10.0, 110.0, 110.0],
        },
        {
            "pre_nms_id": "b_raw0001",
            "tile_id": 4,
            "call_id": "yolo_T04",
            "tile_box_index": 1,
            "score": 0.80,
            "bbox_xyxy": [102.0, 102.0, 200.0, 200.0],
            "local_bbox_xyxy": [50.0, 50.0, 148.0, 148.0],
        },
        {
            "pre_nms_id": "b_raw0000",
            "tile_id": 4,
            "call_id": "yolo_T04",
            "tile_box_index": 0,
            "score": 0.80,
            "bbox_xyxy": [400.0, 400.0, 460.0, 460.0],
            "local_bbox_xyxy": [300.0, 300.0, 360.0, 360.0],
        },
    ]
    kept = class_agnostic_nms_with_dedup_trace(proposals, iou_threshold=0.50, board_id="s_test")
    assert len(kept) == 2
    # Equal score 0.80: tile_id=4, tile_box_index=0 comes first; then tile_id=4, tile_box_index=1 suppresses tile_id=5
    assert kept[0]["pre_nms_id"] == "b_raw0000"
    assert kept[0]["suppressed_count"] == 0
    assert kept[1]["pre_nms_id"] == "b_raw0001"
    assert kept[1]["suppressed_pre_nms_ids"] == ["b_raw0002"]
    assert kept[1]["label"] == "proposal"


def test_yolo_proposal_assign_qwen3_label_policy() -> None:
    preds = [
        {
            "prediction_id": "p0",
            "board_id": "s20_front",
            "orig_pred_label": "proposal",
            "orig_pred_bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
            "score": 0.91,
            "tile_id": 1,
            "call_id": "yolo_T01",
            "crop_valid": True,
            "unique_call_id": "u1",
        },
        {
            "prediction_id": "p1",
            "board_id": "s20_front",
            "orig_pred_label": "proposal",
            "orig_pred_bbox_xyxy": [60.0, 60.0, 90.0, 90.0],
            "score": 0.72,
            "tile_id": 2,
            "call_id": "yolo_T02",
            "crop_valid": True,
            "unique_call_id": "u2",
        },
        {
            "prediction_id": "p2",
            "board_id": "s20_front",
            "orig_pred_label": "proposal",
            "orig_pred_bbox_xyxy": [120.0, 120.0, 160.0, 160.0],
            "score": 0.55,
            "tile_id": 3,
            "call_id": "yolo_T03",
            "crop_valid": True,
            "unique_call_id": "u3",
        },
    ]
    ver_records = {
        "u1": {"ok": True, "strict_json_ok": True, "strict_label": "diode"},
        "u2": {"ok": True, "strict_json_ok": True, "strict_label": "other"},
        "u3": {"ok": True, "strict_json_ok": False, "strict_label": None},
    }
    res = apply_verifier_filter_policy(
        predictions=preds,
        verification_records_by_call_id=ver_records,
        policy="YOLO_PROPOSAL_ASSIGN_QWEN3_LABEL",
    )
    assert res["technical_complete"] is True
    assert res["counts"]["retained_count"] == 1
    assert res["counts"]["rejected_as_other"] == 1
    assert res["counts"]["format_failure_filtered"] == 1
    kept = res["kept_predictions_by_board"]["s20_front"]
    assert len(kept) == 1
    assert kept[0]["label"] == "diode"
    assert kept[0]["bbox_xyxy"] == [10.0, 10.0, 50.0, 50.0]
    assert kept[0]["score"] == 0.91


def test_validate_experimental_modes_and_class_agnostic_eval() -> None:
    assert validate_model_and_mode("yolo11n", "YOLO_T_LOC") == ("yolo11n", "YOLO_T_LOC")
    assert validate_model_and_mode("yolo11n_qwen3", "YOLO_T_QWEN3_CLS") == ("yolo11n_qwen3", "YOLO_T_QWEN3_CLS")
    gts = [
        {"id": 1, "label": "capacitor", "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
        {"id": 2, "label": "diode", "bbox_xyxy": [100.0, 100.0, 140.0, 140.0]},
    ]
    preds = [
        {"prediction_id": "p0", "label": "proposal", "bbox_xyxy": [11.0, 11.0, 49.0, 49.0], "score": 0.88},
    ]
    m = _match_preds_with_gt_class_agnostic(gts, preds, iou_thresh=0.50)
    assert m["metrics"]["tp"] == 1
    assert m["metrics"]["fp"] == 0
    assert m["metrics"]["fn"] == 1
    assert m["metrics"]["eval_protocol"] == "class_agnostic_proposal_localization_not_final_detection_tp"
