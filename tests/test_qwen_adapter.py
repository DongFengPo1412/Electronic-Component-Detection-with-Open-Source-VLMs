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
        # We test parsing method without instantiating heavy model weights
        adapter = object.__new__(QwenAdapter)
        sample_text = (
            "Here are the detected components: "
            "resistor <|box_start|>(100,200),(300,400)<|box_end|> "
            "capacitor <|box_start|>(500,600),(700,800)<|box_end|>"
        )
        img_w, img_h = 1000, 1000
        detected = adapter.parse_grounding_output(sample_text, img_w, img_h)

        self.assertEqual(len(detected), 2)
        # Note Qwen outputs (ymin, xmin), (ymax, xmax) -> xyxy is [xmin, ymin, xmax, ymax]
        self.assertEqual(detected[0]["label"], "resistor")
        self.assertEqual(detected[0]["bbox_xyxy"], [200.0, 100.0, 400.0, 300.0])

        self.assertEqual(detected[1]["label"], "capacitor")
        self.assertEqual(detected[1]["bbox_xyxy"], [600.0, 500.0, 800.0, 700.0])

    def test_parse_empty_and_corrupt(self):
        adapter = object.__new__(QwenAdapter)
        self.assertEqual(adapter.parse_grounding_output("", 1000, 1000), [])
        self.assertEqual(adapter.parse_grounding_output("No components found.", 1000, 1000), [])
        self.assertEqual(adapter.parse_grounding_output("<|box_start|>corrupt<|box_end|>", 1000, 1000), [])


if __name__ == "__main__":
    unittest.main()
