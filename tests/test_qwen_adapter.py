"""Unit tests for QwenAdapter coordinate mapping and output parsing."""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pcb_vlm.geometry.box_ops import (
    xyxy_to_xywh,
    xywh_to_xyxy,
    qwen_norm_to_xyxy,
    xyxy_to_qwen_norm,
    compute_iou,
)
from pcb_vlm.models.qwen_adapter import QwenAdapter


class TestQwenAdapterGeometry(unittest.TestCase):
    def test_coordinate_roundtrip(self):
        img_w, img_h = 4000, 3000
        original_xyxy = [100.0, 200.0, 500.0, 600.0]
        norm_box = xyxy_to_qwen_norm(original_xyxy, img_w, img_h)
        reconstructed = qwen_norm_to_xyxy(norm_box, img_w, img_h)

        # Allow small rounding tolerance due to 1000-scale discrete bins
        self.assertAlmostEqual(original_xyxy[0], reconstructed[0], delta=5.0)
        self.assertAlmostEqual(original_xyxy[1], reconstructed[1], delta=5.0)
        self.assertAlmostEqual(original_xyxy[2], reconstructed[2], delta=5.0)
        self.assertAlmostEqual(original_xyxy[3], reconstructed[3], delta=5.0)

    def test_full_image_boundary(self):
        img_w, img_h = 1920, 1080
        norm_box = (0, 0, 1000, 1000)
        xyxy = qwen_norm_to_xyxy(norm_box, img_w, img_h)
        self.assertEqual(xyxy, [0.0, 0.0, 1920.0, 1080.0])

    def test_parse_standard_grounding_output(self):
        adapter = object.__new__(QwenAdapter)
        sample_text = (
            "Here are the detected components: "
            "resistor <|box_start|>(100,200),(300,400)<|box_end|> "
            "capacitor <|box_start|>(500,600),(700,800)<|box_end|>"
        )
        img_w, img_h = 1000, 1000
        detected = adapter.parse_grounding_output(sample_text, img_w, img_h)

        self.assertEqual(len(detected), 2)
        # Canonical Qwen outputs (xmin, ymin), (xmax, ymax) -> xyxy is [xmin, ymin, xmax, ymax]
        self.assertEqual(detected[0]["label"], "resistor")
        self.assertEqual(detected[0]["bbox_xyxy"], [100.0, 200.0, 300.0, 400.0])

        self.assertEqual(detected[1]["label"], "capacitor")
        self.assertEqual(detected[1]["bbox_xyxy"], [500.0, 600.0, 700.0, 800.0])

    def test_parse_empty_and_corrupt(self):
        adapter = object.__new__(QwenAdapter)
        self.assertEqual(adapter.parse_grounding_output("", 1000, 1000), [])
        self.assertEqual(adapter.parse_grounding_output("No components found.", 1000, 1000), [])
        self.assertEqual(adapter.parse_grounding_output("<|box_start|>corrupt<|box_end|>", 1000, 1000), [])

    def test_asymmetric_aspect_ratio_and_axis_order(self):
        # 4000x2000 (2:1 aspect ratio)
        img_w, img_h = 4000, 2000
        true_box = [3000.0, 200.0, 3800.0, 600.0]
        norm_box = xyxy_to_qwen_norm(true_box, img_w, img_h)
        # Expected canonical x-first: xmin=750, ymin=100, xmax=950, ymax=300
        self.assertEqual(norm_box, (750, 100, 950, 300))
        reconstructed = qwen_norm_to_xyxy(norm_box, img_w, img_h)
        self.assertEqual(reconstructed, true_box)

    def test_official_qwen_tokens_parsing(self):
        adapter = object.__new__(QwenAdapter)
        token_str = "<|object_ref_start|>diode<|object_ref_end|><|box_start|>(150,250),(450,650)<|box_end|>"
        img_w, img_h = 1000, 500
        parsed = adapter.parse_grounding_output(token_str, img_w, img_h)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["label"], "diode")
        # xmin=150/1000*1000=150, ymin=250/1000*500=125, xmax=450/1000*1000=450, ymax=650/1000*500=325
        self.assertEqual(parsed[0]["bbox_xyxy"], [150.0, 125.0, 450.0, 325.0])

    def test_user_real_probe_top_right_rectangle(self):
        # User probe empirical output on 800x400: (652, 95), (916, 375)
        adapter = object.__new__(QwenAdapter)
        probe_text = "(652,95),(916,375)"
        img_w, img_h = 800, 400
        gt_box = [520.0, 40.0, 740.0, 140.0]
        parsed = adapter.parse_grounding_output(probe_text, img_w, img_h)
        self.assertEqual(len(parsed), 1)
        iou = compute_iou(parsed[0]["bbox_xyxy"], gt_box)
        self.assertGreater(iou, 0.85)

    def test_user_real_probe_normalized_floats(self):
        # User probe empirical output with float array: [0.6, 0.23, 0.8, 0.45]
        adapter = object.__new__(QwenAdapter)
        probe_text = "[0.6, 0.23, 0.8, 0.45]"
        img_w, img_h = 800, 400
        parsed = adapter.parse_grounding_output(probe_text, img_w, img_h)
        self.assertEqual(len(parsed), 1)
        # x1 = 0.6 * 800 = 480, y1 = 0.23 * 400 = 92, x2 = 0.8 * 800 = 640, y2 = 0.45 * 400 = 180
        self.assertEqual(parsed[0]["bbox_xyxy"], [480.0, 92.0, 640.0, 180.0])

    def test_parsing_status_reporting(self):
        adapter = object.__new__(QwenAdapter)
        # 1. Success
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status("[0.1, 0.2, 0.5, 0.6]", 1000, 1000)
        self.assertEqual(status, "success")
        self.assertEqual(len(boxes), 1)
        self.assertFalse(trunc)

        # 2. Explicit no objects
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status("No objects found.", 1000, 1000)
        self.assertEqual(status, "no_objects_detected")
        self.assertEqual(len(boxes), 0)

        # 3. Parse failed (conversational / unparseable)
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status("Unable to find anything in this area.", 1000, 1000)
        self.assertEqual(status, "parse_failed")
        self.assertEqual(len(boxes), 0)

        # 4. Empty string is parse_failed, NOT no_objects_detected
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status("", 1000, 1000)
        self.assertEqual(status, "parse_failed")

    def test_label_preservation_bracketed_format(self):
        adapter = object.__new__(QwenAdapter)
        # Verify that 'resistor [100, 100, 300, 300]' retains 'resistor' instead of becoming 'component'
        text = "resistor [100, 100, 300, 300]"
        boxes = adapter.parse_grounding_output(text, 1000, 1000)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "resistor")
        self.assertEqual(boxes[0]["bbox_xyxy"], [100.0, 100.0, 300.0, 300.0])

    def test_mixed_format_parsing(self):
        adapter = object.__new__(QwenAdapter)
        # Text containing both token format and bracketed format
        text = (
            "resistor <|box_start|>(100,200),(300,400)<|box_end|> "
            "ic [500, 600, 700, 800]"
        )
        boxes = adapter.parse_grounding_output(text, 1000, 1000)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0]["label"], "resistor")
        self.assertEqual(boxes[1]["label"], "ic")

    def test_truncation_detection(self):
        adapter = object.__new__(QwenAdapter)
        # Output truncated at token limit in the middle of a second box
        text = "resistor <|box_start|>(100,200),(300,400)<|box_end|> ic <|box_start|>(500,600),"
        boxes, status, detail, is_truncated = adapter.parse_grounding_output_with_status(text, 1000, 1000)
        self.assertEqual(len(boxes), 1)
        self.assertTrue(is_truncated)
        self.assertEqual(status, "partial_success")

    def test_word_boundary_none(self):
        adapter = object.__new__(QwenAdapter)
        # Word containing 'none' as substring like 'component' or 'nonexistent' should not trigger no_objects_detected
        text = "This component is unclear."
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status(text, 1000, 1000)
        self.assertNotEqual(status, "no_objects_detected")
        self.assertEqual(status, "parse_failed")

    def test_python_dict_label_retention(self):
        adapter = object.__new__(QwenAdapter)
        text = "[{'label': 'connector', 'bbox': [125.0, 100.0, 183.0, 150.0], 'conf_score': None}]"
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status(text, 800, 400)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "connector", "Python single-quote dict must preserve 'connector' label, not default to 'component'")
        self.assertEqual(status, "success")

    def test_none_in_descriptive_sentence_not_no_objects(self):
        adapter = object.__new__(QwenAdapter)
        text = "none of the resistors on the top were clearly visible."
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status(text, 800, 400)
        self.assertNotEqual(status, "no_objects_detected", "Descriptive text containing 'none' must NOT be classified as no_objects_detected")
        self.assertEqual(status, "parse_failed")

    def test_bare_box_labeled_unknown(self):
        adapter = object.__new__(QwenAdapter)
        text = "<|box_start|>(100,200),(300,400)<|box_end|>"
        boxes, status, detail, trunc = adapter.parse_grounding_output_with_status(text, 1000, 1000)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "unknown", "Bare boxes without category must be preserved as 'unknown', not guessed as a category")

    def test_wrong_model_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            QwenAdapter(model_id="unrelated/does-not-exist")


if __name__ == "__main__":
    unittest.main()


