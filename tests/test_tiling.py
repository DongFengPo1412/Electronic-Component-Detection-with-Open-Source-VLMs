"""Unit tests for tiling generation, coordinate remapping, and NMS merging."""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pcb_vlm.geometry.tiling import (
    generate_tiles,
    map_tile_box_to_global,
    nms_merge_predictions,
)


class TestTiling(unittest.TestCase):
    def test_tile_coverage(self):
        img_w, img_h = 3000, 2000
        tile_size = 1000
        overlap_ratio = 0.20  # stride = 800

        tiles = generate_tiles(img_w, img_h, tile_size=tile_size, overlap_ratio=overlap_ratio)
        self.assertGreater(len(tiles), 0)

        # Check all corners are covered
        covered_left = any(t["bbox"][0] == 0 for t in tiles)
        covered_top = any(t["bbox"][1] == 0 for t in tiles)
        covered_right = any(t["bbox"][2] == img_w for t in tiles)
        covered_bottom = any(t["bbox"][3] == img_h for t in tiles)

        self.assertTrue(covered_left)
        self.assertTrue(covered_top)
        self.assertTrue(covered_right)
        self.assertTrue(covered_bottom)

    def test_coordinate_remapping(self):
        img_w, img_h = 4000, 3000
        offset_x, offset_y = 800, 1600
        local_box = [50.0, 60.0, 200.0, 300.0]

        global_box = map_tile_box_to_global(local_box, offset_x, offset_y, img_w, img_h)
        self.assertEqual(global_box, [850.0, 1660.0, 1000.0, 1900.0])

    def test_nms_merging_same_category(self):
        preds = [
            {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "score": 0.95},
            # Near duplicate with high IoU
            {"label": "resistor", "bbox_xyxy": [105.0, 102.0, 202.0, 198.0], "score": 0.80},
            # Distinct box far away
            {"label": "resistor", "bbox_xyxy": [500.0, 500.0, 600.0, 600.0], "score": 0.90},
        ]
        merged = nms_merge_predictions(preds, iou_threshold=0.50)
        self.assertEqual(len(merged), 2)
        # Best scored should be kept
        self.assertEqual(merged[0]["score"], 0.95)

    def test_nms_different_categories_preserved(self):
        # A resistor and capacitor in close proximity should NOT suppress each other
        preds = [
            {"label": "resistor", "bbox_xyxy": [100.0, 100.0, 200.0, 200.0], "score": 0.95},
            {"label": "capacitor", "bbox_xyxy": [105.0, 105.0, 200.0, 200.0], "score": 0.90},
        ]
        merged = nms_merge_predictions(preds, iou_threshold=0.50)
        self.assertEqual(len(merged), 2)

    def test_composite_tile_status_aggregation(self):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
        from infer_batch import compute_composite_tile_status

        # 1. All slices parse_failed -> parse_failed
        s1 = compute_composite_tile_status({"parse_failed": 4}, total_tiles=4, num_merged_objects=0)
        self.assertEqual(s1, "parse_failed")

        # 2. All slices parse_filtered -> parse_filtered (NOT no_objects_detected)
        s2 = compute_composite_tile_status({"parse_filtered": 4}, total_tiles=4, num_merged_objects=0)
        self.assertEqual(s2, "parse_filtered")

        # 3. All slices genuinely clean no_objects_detected -> no_objects_detected
        s3 = compute_composite_tile_status({"no_objects_detected": 4}, total_tiles=4, num_merged_objects=0)
        self.assertEqual(s3, "no_objects_detected")

        # 4. Partial failure without objects -> partial_failed
        s4 = compute_composite_tile_status({"no_objects_detected": 3, "parse_failed": 1}, total_tiles=4, num_merged_objects=0)
        self.assertEqual(s4, "partial_failed")

        # 5. Some filtered, some no_objects, 0 objects -> parse_filtered
        s5 = compute_composite_tile_status({"no_objects_detected": 2, "parse_filtered": 2}, total_tiles=4, num_merged_objects=0)
        self.assertEqual(s5, "parse_filtered")

        # 6. Objects detected but some tiles were truncated/filtered -> partial_success
        s6 = compute_composite_tile_status({"success": 2, "parse_filtered": 1, "partial_success": 1}, total_tiles=4, num_merged_objects=3)
        self.assertEqual(s6, "partial_success")

        # 7. Clean success
        s7 = compute_composite_tile_status({"success": 4}, total_tiles=4, num_merged_objects=5)
        self.assertEqual(s7, "success")


if __name__ == "__main__":
    unittest.main()
