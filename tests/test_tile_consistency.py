"""Unit tests for Candidate A offline overlapping-tile consistency filtering."""

from __future__ import annotations

import copy
import inspect
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import FICS_CANONICAL_CATEGORIES  # noqa: E402
from pcb_vlm.evaluation.tile_consistency import (  # noqa: E402
    compute_opportunity_tiles,
    evaluate_tile_consistency_vs_baseline,
    filter_board_raw_predictions_by_tile_consistency,
)
from pcb_vlm.geometry.tiling import generate_tiles  # noqa: E402


def _make_t_record(
    *,
    board_id: str,
    tile_id: int,
    offset_x: int,
    offset_y: int,
    width: int,
    height: int,
    objects: list[dict],
    roi_offset_x: int = 100,
    roi_offset_y: int = 200,
    ok: bool = True,
) -> dict:
    payload = [
        {"bbox_2d": o["norm_bbox"], "label": o["label"]}
        for o in objects
    ]
    return {
        "call_id": f"qwen3_formal_{board_id}_T{tile_id:02d}",
        "board_id": board_id,
        "mode": "T",
        "tile_id": tile_id,
        "roi_offset_x": roi_offset_x,
        "roi_offset_y": roi_offset_y,
        "tile_offset_x": offset_x,
        "tile_offset_y": offset_y,
        "orig_input_width": width,
        "orig_input_height": height,
        "max_new_tokens": 512,
        "coordinate_protocol": "xyxy_norm1000",
        "raw_output": json.dumps(payload),
        "generated_tokens": 30,
        "generated_token_ids": [151645],
        "ok": ok,
    }


def _norm_from_tile_xyxy(box_global: list[float], tile_x: int, tile_y: int, roi_x: int = 100, roi_y: int = 200) -> list[int]:
    lx1 = box_global[0] - roi_x - tile_x
    ly1 = box_global[1] - roi_y - tile_y
    lx2 = box_global[2] - roi_x - tile_x
    ly2 = box_global[3] - roi_y - tile_y
    return [
        int(round(lx1 / 1024.0 * 1000.0)),
        int(round(ly1 / 1024.0 * 1000.0)),
        int(round(lx2 / 1024.0 * 1000.0)),
        int(round(ly2 / 1024.0 * 1000.0)),
    ]


def test_independent_tile_voting_and_same_tile_dedup():
    # Two overlapping tiles covering ROI 1843x1024 (x=0..1024 and x=819..1843, overlap x=819..1024)
    # In global coords (roi_offset=(100, 200)): overlap strip is x=[919, 1124], y=[200, 1224]
    tiles = generate_tiles(1843, 1024, tile_size=1024, overlap_ratio=0.20)
    assert len(tiles) == 2

    supported_box = [940.0, 400.0, 1080.0, 540.0]  # inside both tile 0 and tile 1 -> opp=2
    dup_single_tile_box = [950.0, 700.0, 1090.0, 820.0]  # inside both tiles -> opp=2, predicted twice ONLY in tile 0

    rec0 = _make_t_record(
        board_id="b1",
        tile_id=0,
        offset_x=0,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[
            {"label": "capacitor", "norm_bbox": _norm_from_tile_xyxy(supported_box, 0, 0)},
            {"label": "resistor", "norm_bbox": _norm_from_tile_xyxy(dup_single_tile_box, 0, 0)},
            {"label": "resistor", "norm_bbox": _norm_from_tile_xyxy([952.0, 702.0, 1092.0, 822.0], 0, 0)},
        ],
    )
    rec1 = _make_t_record(
        board_id="b1",
        tile_id=1,
        offset_x=819,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[
            {"label": "capacitor", "norm_bbox": _norm_from_tile_xyxy(supported_box, 819, 0)},
        ],
    )

    out = filter_board_raw_predictions_by_tile_consistency(
        [rec0, rec1],
        board_id="b1",
        img_width=2200,
        img_height=1500,
        expected_roi_size=(1843, 1024),
    )
    assert out["usable"] is True
    assert out["group_count"] == 2
    by_lbl = {g["label"]: g for g in out["groups"]}
    # Supported across 2 distinct tiles -> kept
    assert by_lbl["capacitor"]["opportunity_count"] == 2
    assert by_lbl["capacitor"]["support_count"] == 2
    assert by_lbl["capacitor"]["filtered_out"] is False
    # Same-tile duplicates count as support_count=1 -> filtered in 2-opp overlap
    assert by_lbl["resistor"]["member_count"] == 2
    assert by_lbl["resistor"]["opportunity_count"] == 2
    assert by_lbl["resistor"]["support_count"] == 1
    assert by_lbl["resistor"]["filtered_out"] is True
    kept_labels = [p["label"] for p in out["filtered_post_nms_predictions"]]
    assert kept_labels == ["capacitor"]


def test_single_coverage_and_oversized_boxes_preserved():
    # Single-coverage box only inside tile 0 (x in [200, 400] < 919); opp=1 -> must keep
    # Straddling box spanning x=[800, 1200] across overlap boundary (neither tile 0 [100,1124] alone nor tile 1 [919,1943] alone contains both edges? Wait: tile 0 ends at 1124 so x2=1200 is outside tile 0, and x1=800 < 919 is outside tile 1 -> opp=0 -> must keep!)
    single_cov = [200.0, 300.0, 400.0, 450.0]
    straddle_big = [800.0, 300.0, 1090.0, 500.0]  # Wait: straddle_big inside tile 0 [100,1124] only (opp=1)! Let's test true opp=0 spanning outside both:
    oversized_or_straddle = [800.0, 200.0, 1124.0, 1224.0]  # reaches tile0 right edge cleanly; let's exceed tile1 left and exceed tile0 right via clipping or test compute_opportunity_tiles directly too.
    # Better: 3-tile row ROI w=2662 -> tile0=[100..1124], tile1=[919..1943], tile2=[1738..2762]
    # A box spanning [850..1200] in global is in tile1 only? Wait: 850<919, 1200>1124! Not contained in ANY tile -> opp=0!
    tiles_by_id = {
        0: {"bbox": [100.0, 200.0, 1124.0, 1224.0]},
        1: {"bbox": [919.0, 200.0, 1943.0, 1224.0]},
    }
    assert compute_opportunity_tiles([850.0, 300.0, 1200.0, 500.0], tiles_by_id) == []

    # Test end-to-end single-coverage opp=1 preservation and straddle box via multi-prediction in 2 tiles
    rec0 = _make_t_record(
        board_id="b2",
        tile_id=0,
        offset_x=0,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[
            {"label": "ic", "norm_bbox": _norm_from_tile_xyxy(single_cov, 0, 0)},
            # Box right on edge of tile 0 extending to x=1124 (1000 in norm) is NOT inside tile 1 if x1=850
            {"label": "inductor", "norm_bbox": [732, 100, 1000, 400]},
        ],
    )
    rec1 = _make_t_record(
        board_id="b2",
        tile_id=1,
        offset_x=819,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[],
    )
    out = filter_board_raw_predictions_by_tile_consistency(
        [rec0, rec1],
        board_id="b2",
        img_width=2200,
        img_height=1500,
        expected_roi_size=(1843, 1024),
    )
    assert out["usable"] is True
    assert out["filtered_group_count"] == 0
    kept_labels = sorted(p["label"] for p in out["filtered_post_nms_predictions"])
    assert kept_labels == ["ic", "inductor"]


def test_cross_class_independence_and_deterministic_order_invariance():
    box_a = [950.0, 400.0, 1070.0, 520.0]
    rec0 = _make_t_record(
        board_id="b3",
        tile_id=0,
        offset_x=0,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[
            {"label": "resistor", "norm_bbox": _norm_from_tile_xyxy(box_a, 0, 0)},
            {"label": "capacitor", "norm_bbox": _norm_from_tile_xyxy([250.0, 250.0, 400.0, 400.0], 0, 0)},
        ],
    )
    rec1 = _make_t_record(
        board_id="b3",
        tile_id=1,
        offset_x=819,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[
            # Different class at identical coords must NOT support resistor in tile 0
            {"label": "diode", "norm_bbox": _norm_from_tile_xyxy(box_a, 819, 0)},
        ],
    )
    out1 = filter_board_raw_predictions_by_tile_consistency(
        [rec0, rec1],
        board_id="b3",
        img_width=2200,
        img_height=1500,
        expected_roi_size=(1843, 1024),
    )
    # Reverse both record order and internal object list order
    rec0_rev = copy.deepcopy(rec0)
    rec0_rev["raw_output"] = json.dumps(list(reversed(json.loads(rec0["raw_output"]))))
    out2 = filter_board_raw_predictions_by_tile_consistency(
        [rec1, rec0_rev],
        board_id="b3",
        img_width=2200,
        img_height=1500,
        expected_roi_size=(1843, 1024),
    )
    assert out1["usable"] and out2["usable"]
    # Neither resistor nor diode at overlap got 2 same-class votes -> both filtered; only interior capacitor kept
    assert [p["label"] for p in out1["filtered_post_nms_predictions"]] == ["capacitor"]
    assert out1["filtered_post_nms_predictions"] == out2["filtered_post_nms_predictions"]
    assert [(g["label"], g["rep_box_xyxy"], g["support_count"], g["opportunity_count"], g["filtered_out"]) for g in out1["groups"]] == [
        (g["label"], g["rep_box_xyxy"], g["support_count"], g["opportunity_count"], g["filtered_out"]) for g in out2["groups"]
    ]


def test_missing_tile_provenance_refused():
    rec0 = _make_t_record(
        board_id="b4",
        tile_id=0,
        offset_x=0,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[{"label": "ic", "norm_bbox": [100, 100, 300, 300]}],
    )
    rec_missing_id = copy.deepcopy(rec0)
    rec_missing_id["tile_id"] = None
    out_none = filter_board_raw_predictions_by_tile_consistency(
        [rec_missing_id],
        board_id="b4",
        img_width=2200,
        img_height=1500,
    )
    assert out_none["usable"] is False
    assert "missing_or_invalid_tile_id" in str(out_none["unusable_reason"])

    # Incomplete grid against expected_roi_size
    out_incomplete = filter_board_raw_predictions_by_tile_consistency(
        [rec0],
        board_id="b4",
        img_width=2200,
        img_height=1500,
        expected_roi_size=(1843, 1024),
    )
    assert out_incomplete["usable"] is False
    assert "tile_count_mismatch" in str(out_incomplete["unusable_reason"])


def test_gt_isolation_contract():
    sig = inspect.signature(filter_board_raw_predictions_by_tile_consistency)
    param_names = set(sig.parameters.keys())
    assert "board_coco" not in param_names
    assert "annotations" not in param_names
    assert "combined_coco" not in param_names

    box_keep = [250.0, 300.0, 400.0, 450.0]
    box_drop = [950.0, 400.0, 1080.0, 530.0]
    rec0 = _make_t_record(
        board_id="b5",
        tile_id=0,
        offset_x=0,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[
            {"label": "ic", "norm_bbox": _norm_from_tile_xyxy(box_keep, 0, 0)},
            {"label": "resistor", "norm_bbox": _norm_from_tile_xyxy(box_drop, 0, 0)},
        ],
    )
    rec1 = _make_t_record(
        board_id="b5",
        tile_id=1,
        offset_x=819,
        offset_y=0,
        width=1024,
        height=1024,
        objects=[],
    )

    coco_a = {
        "images": [{"id": 1, "board_id": "b5", "width": 2200, "height": 1500, "file_name": "b5.png"}],
        "categories": list(FICS_CANONICAL_CATEGORIES),
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 6, "bbox": [250.0, 300.0, 150.0, 150.0], "area": 22500.0, "iscrowd": 0},
        ],
    }
    coco_b = copy.deepcopy(coco_a)
    coco_b["annotations"].append(
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [950.0, 400.0, 130.0, 130.0], "area": 16900.0, "iscrowd": 0}
    )

    # Identical prediction filtering decision regardless of GT annotations
    filt_only = filter_board_raw_predictions_by_tile_consistency(
        [rec0, rec1],
        board_id="b5",
        img_width=2200,
        img_height=1500,
        expected_roi_size=(1843, 1024),
    )
    rep_a = evaluate_tile_consistency_vs_baseline(
        board_records_by_board={"b5": [rec0, rec1]},
        combined_coco=coco_a,
        roi_sizes_by_board={"b5": (1843, 1024)},
    )
    rep_b = evaluate_tile_consistency_vs_baseline(
        board_records_by_board={"b5": [rec0, rec1]},
        combined_coco=coco_b,
        roi_sizes_by_board={"b5": (1843, 1024)},
    )
    assert len(filt_only["filtered_post_nms_predictions"]) == 1
    # In coco_a, box_drop was FP -> removed_fp == 1, lost_tp == 0
    assert rep_a["per_board"]["b5"]["delta"]["removed_fp"] == 1
    assert rep_a["per_board"]["b5"]["delta"]["lost_tp"] == 0
    # In coco_b, exact same filter dropped box_drop even though it happened to be GT -> lost_tp == 1
    assert rep_b["per_board"]["b5"]["delta"]["lost_tp"] == 1
    assert rep_b["per_board"]["b5"]["candidate_a"]["predictions"] == rep_a["per_board"]["b5"]["candidate_a"]["predictions"]
