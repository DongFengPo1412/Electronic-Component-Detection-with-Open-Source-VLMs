"""Coordinate Protocol and Model Format Verification Script.

Rigorously verifies against empirical Qwen2-VL model outputs:
1. Canonical x-first axis order: (xmin, ymin), (xmax, ymax) in [0, 1000].
2. Direct validation against user's real 800x400 probe results:
   - Probe 1 (Top-Right): output "(652,95),(916,375)" -> IoU = 0.8608 with GT [520, 40, 740, 140].
   - Probe 2 (Floats): output "[0.6, 0.23, 0.8, 0.45]" -> correctly parsed without zero-box failure.
3. Explicit parsing status verification (success, no_objects_detected, parse_failed).
4. DeepSeek-VL-1.3B dialogue baseline yields 0 bounding boxes.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pcb_vlm.geometry.box_ops import (
    qwen_norm_to_xyxy,
    xyxy_to_qwen_norm,
    compute_iou,
)
from pcb_vlm.models.qwen_adapter import QwenAdapter
from pcb_vlm.models.deepseek_adapter import DeepSeekAdapter


def run_coordinate_protocol_audit():
    print("=" * 70)
    print("AUDIT: HISTORICAL TEXT REPLAY (legacy=True; NOT fresh model inference)")
    print("=" * 70)

    adapter = object.__new__(QwenAdapter)

    # 1. Real Probe 1: Top-Right Asymmetric Rectangle (800x400)
    img_w, img_h = 800, 400
    gt_box_1 = [520.0, 40.0, 740.0, 140.0]
    probe_output_1 = "<|box_start|>(652,95),(916,375)<|box_end|>"
    print(f"\n[Probe 1 - Top-Right Rectangle] Geometry: {img_w}x{img_h}, GT: {gt_box_1}")
    print(f"Model Real Output: '{probe_output_1}'")

    parsed_1, status_1, detail_1, trunc_1 = adapter.parse_grounding_output_with_status(probe_output_1, img_w, img_h, legacy=True)
    assert status_1 == "success", f"Probe 1 parse failed: {status_1}, {detail_1}"
    assert len(parsed_1) == 1
    assert trunc_1 is False, "Probe 1 should not be truncated"
    decoded_box_1 = parsed_1[0]["bbox_xyxy"]
    iou_x_first = compute_iou(decoded_box_1, gt_box_1)
    print(f"Decoded Box (x-first): {decoded_box_1} | IoU: {iou_x_first:.4f}")
    assert iou_x_first > 0.85, f"Expected IoU > 0.85, got {iou_x_first}"
    print("  -> PASS: Canonical x-first order correctly aligns with real model output (IoU > 0.85).\n")

    # 2. Real Probe 2: Normalized Float Array [0.6, 0.23, 0.8, 0.45]
    probe_output_2 = "[0.6, 0.23, 0.8, 0.45]"
    print(f"[Probe 2 - Normalized Floats] Model Real Output: '{probe_output_2}'")
    parsed_2, status_2, detail_2, trunc_2 = adapter.parse_grounding_output_with_status(probe_output_2, img_w, img_h, legacy=True)
    assert status_2 == "success", f"Probe 2 parse failed: {status_2}, {detail_2}"
    assert len(parsed_2) == 1
    assert trunc_2 is False, "Probe 2 should not be truncated"
    decoded_box_2 = parsed_2[0]["bbox_xyxy"]
    print(f"Decoded Box: {decoded_box_2} | Status: {status_2} ({detail_2})")
    assert decoded_box_2 == [480.0, 92.0, 640.0, 180.0]
    print("  -> PASS: Normalized float array successfully parsed into valid pixel coordinates.\n")

    # 3. Parsing Status Classification
    print("[Probe 3 - Parsing Status Classification]")
    _, status_clean_empty, _, _ = adapter.parse_grounding_output_with_status("No components found.", img_w, img_h)
    assert status_clean_empty == "no_objects_detected"
    _, status_corrupt, detail_corrupt, _ = adapter.parse_grounding_output_with_status("The board has some defects on layer 3.", img_w, img_h)
    assert status_corrupt == "parse_failed"
    print(f"  Empty output status: '{status_clean_empty}'")
    print(f"  Unparseable text status: '{status_corrupt}' -> {detail_corrupt}")
    print("  -> PASS: Explicit parsing status successfully separates clean absence from parsing failure.\n")

    # 4. DeepSeek-VL-1.3B Architectural Dialogue Baseline
    print("[Probe 4 - DeepSeek-VL-1.3B Architectural Dialogue Baseline]")
    ds_adapter = object.__new__(DeepSeekAdapter)
    conversational_text = "This image presents a printed circuit board containing multiple capacitors and resistors."
    ds_parsed = ds_adapter.parse_grounding_output(conversational_text, img_w, img_h)
    assert len(ds_parsed) == 0
    print(f"Input: '{conversational_text[:60]}...'")
    print(f"Parsed objects count: {len(ds_parsed)}")
    print("  -> PASS: Dialogue text correctly produces 0 boxes without artificial fabrication.\n")

    print("=" * 70)
    print("HISTORICAL TEXT REPLAY ASSERTIONS PASSED (no new model inference).")
    print("=" * 70)


if __name__ == "__main__":
    run_coordinate_protocol_audit()
