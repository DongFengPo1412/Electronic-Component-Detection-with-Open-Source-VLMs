"""Unit tests for DeepSeek-VL2 adapter and grounding token parser."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pcb_vlm.models.deepseek_adapter import DeepSeekAdapter


class TestDeepSeekAdapter(unittest.TestCase):

    def setUp(self):
        self.adapter = object.__new__(DeepSeekAdapter)

    def test_native_grounding_parsing_single_box(self):
        text = "<|ref|>ic<|/ref|><|det|>[[200, 300, 600, 700]]<|/det|>"
        boxes = self.adapter.parse_grounding_output(text, img_w=1000, img_h=1000)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "ic")
        self.assertEqual(boxes[0]["bbox_xyxy"], [300.0, 200.0, 700.0, 600.0])

    def test_native_grounding_parsing_multi_boxes(self):
        text = (
            "<|ref|>resistor<|/ref|><|det|>[[100, 150, 200, 250], [300, 350, 400, 450]]<|/det|>"
            "<|ref|>capacitor<|/ref|><|det|>[[500, 500, 600, 600]]<|/det|>"
        )
        boxes = self.adapter.parse_grounding_output(text, img_w=1000, img_h=1000)
        self.assertEqual(len(boxes), 3)
        self.assertEqual(boxes[0]["label"], "resistor")
        self.assertEqual(boxes[1]["label"], "resistor")
        self.assertEqual(boxes[2]["label"], "capacitor")
        self.assertEqual(boxes[0]["bbox_xyxy"], [150.0, 100.0, 250.0, 200.0])
        self.assertEqual(boxes[2]["bbox_xyxy"], [500.0, 500.0, 600.0, 600.0])

    def test_fallback_bracket_parsing(self):
        text = "Detected: ic [100, 200, 300, 400]"
        boxes = self.adapter.parse_grounding_output(text, img_w=1000, img_h=1000)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["label"], "ic")
        self.assertEqual(boxes[0]["bbox_xyxy"], [200.0, 100.0, 400.0, 300.0])

    def test_build_prompt(self):
        prompt = self.adapter.build_prompt(["ic", "resistor"])
        self.assertIn("ic, resistor", prompt)
        self.assertTrue(prompt.startswith("<|ref|>"))
        self.assertTrue(prompt.endswith("<|det|>"))


if __name__ == "__main__":
    unittest.main()
