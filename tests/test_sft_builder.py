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
        # Unsorted objects (norm_box is [xmin, ymin, xmax, ymax])
        objs = [
            {"label": "resistor", "norm_box": [200, 500, 300, 600]},  # x=200, y=500
            {"label": "ic", "norm_box": [100, 100, 300, 300]},        # x=100, y=100
            {"label": "capacitor", "norm_box": [400, 100, 500, 200]}  # x=400, y=100
        ]
        target_str = build_qwen_grounding_target(objs, 1000, 1000)

        # First should be 'ic' (y=100, x=100), then 'capacitor' (y=100, x=400), then 'resistor' (y=500, x=200)
        ic_pos = target_str.find("ic")
        cap_pos = target_str.find("capacitor")
        res_pos = target_str.find("resistor")

        self.assertTrue(ic_pos < cap_pos, "ic should appear before capacitor due to x-ordering at same y")
        self.assertTrue(cap_pos < res_pos, "capacitor should appear before resistor due to y-ordering")

        # Verify Qwen grounding token format
        self.assertIn("<|object_ref_start|>ic<|object_ref_end|><|box_start|>(100,100),(300,300)<|box_end|>", target_str)

    def test_empty_objects_grounding_target(self):
        self.assertEqual(build_qwen_grounding_target([], 1000, 1000), "None")

    def test_edge_target_not_labeled_as_negative(self):
        import tempfile
        from PIL import Image
        from build_sft_data import process_split_sft

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a 2000x1000 dummy image
            img_path = os.path.join(tmpdir, "test_img.png")
            img = Image.new("RGB", (2000, 1000), color=(100, 100, 100))
            img.save(img_path)

            manifest = [{"image_id": 1, "image_path": img_path, "width": 2000, "height": 1000, "folder": "board_A"}]
            # Place a component at x: [950, 1050], which straddles across tile 0 [0, 1024] and tile 1
            # In tile 0 [0, 1024], inter width is 74 (from 950 to 1024), area is 74*100 = 7400 out of total area 10000 (0.74 visibility)
            # In tile 1 [819, 1843] (with 0.2 overlap), etc.
            # Let's place a component at [1000, 100, 1100, 200] (area 10000).
            # In a tile of [0, 0, 1024, 1024]: intersection is [1000, 100, 1024, 200], area is 24*100 = 2400 (24% visibility < 50%).
            coco_anns = {
                1: [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1000, 100, 100, 100]}]
            }
            cat_map = {1: "resistor"}

            crops_dir = os.path.join(tmpdir, "crops")
            records = process_split_sft(
                split_name="train",
                target_image_ids=[1],
                manifest=manifest,
                coco_anns_by_img=coco_anns,
                cat_id_to_name=cat_map,
                target_classes=["resistor"],
                output_dir=tmpdir,
                crops_dir=crops_dir,
                tile_size=1024,
                overlap_ratio=0.0, # non-overlapping tiles: tile 0 is [0, 1024], tile 1 is [976, 2000]
                min_visibility=0.50
            )

            # Check that tile 0 (which has 24% of the resistor) is NEVER output as a negative sample labeled "None"
            for r in records:
                if r.get("is_negative", False):
                    # Check tile bbox: if it intersects the component [1000, 100, 1100, 200], it must NOT be negative!
                    tx1, ty1, tx2, ty2 = r["tile_bbox"]
                    inter = max(0, min(tx2, 1100) - max(tx1, 1000)) * max(0, min(ty2, 200) - max(ty1, 100))
                    self.assertEqual(inter, 0, f"Negative sample {r['id']} overlaps with ground truth component!")


if __name__ == "__main__":
    unittest.main()

