"""Unit tests for SFT dataset construction and spatial coordinate sorting."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from build_sft_data import (
    bbox_intersection_area,
    build_qwen_grounding_target
)


class TestSFTBuilder(unittest.TestCase):

    def test_bbox_intersection_area(self):
        # Perfect overlap
        b1 = [0, 0, 100, 100]
        self.assertEqual(bbox_intersection_area(b1, b1), 10000.0)

        # 50% overlap
        b2 = [50, 0, 150, 100]
        self.assertEqual(bbox_intersection_area(b1, b2), 5000.0)

        # No overlap
        b3 = [200, 200, 300, 300]
        self.assertEqual(bbox_intersection_area(b1, b3), 0.0)

    def test_spatial_sorting_in_grounding_target(self):
        # Unsorted objects (lower one first, then higher one)
        objs = [
            {"label": "resistor", "norm_box": [500, 200, 600, 300]}, # y=500
            {"label": "ic", "norm_box": [100, 100, 300, 300]},       # y=100
            {"label": "capacitor", "norm_box": [100, 400, 200, 500]}  # y=100, x=400
        ]
        target_str = build_qwen_grounding_target(objs, 1000, 1000)

        # First should be 'ic' (y=100, x=100), then 'capacitor' (y=100, x=400), then 'resistor' (y=500)
        ic_pos = target_str.find("ic")
        cap_pos = target_str.find("capacitor")
        res_pos = target_str.find("resistor")

        self.assertTrue(ic_pos < cap_pos, "ic should appear before capacitor due to x-ordering at same y")
        self.assertTrue(cap_pos < res_pos, "capacitor should appear before resistor due to y-ordering")

        # Verify Qwen grounding token format
        self.assertIn("<|object_ref_start|>ic<|object_ref_end|><|box_start|>(100,100),(300,300)<|box_end|>", target_str)

    def test_empty_objects_grounding_target(self):
        self.assertEqual(build_qwen_grounding_target([], 1000, 1000), "None")


if __name__ == "__main__":
    unittest.main()
