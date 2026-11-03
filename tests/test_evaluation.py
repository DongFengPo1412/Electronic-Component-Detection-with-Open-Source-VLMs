"""
Unit and Integration Tests for PCB Component Detection Evaluator.
Verifies all boundary conditions and failure modes required by Stage 07.
"""

import sys
import os
import unittest

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.evaluate import (
    calculate_iou,
    xyxy_to_xywh,
    xywh_to_xyxy,
    is_valid_box,
    evaluate_detections
)

class TestEvaluator(unittest.TestCase):

    def test_coordinate_conversions(self):
        box_xyxy = [10.0, 20.0, 50.0, 80.0]
        xywh = xyxy_to_xywh(box_xyxy)
        self.assertEqual(xywh, [10.0, 20.0, 40.0, 60.0])
        recovered_xyxy = xywh_to_xyxy(xywh)
        self.assertEqual(recovered_xyxy, box_xyxy)

    def test_iou_calculation(self):
        # Perfect overlap
        b1 = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(calculate_iou(b1, b1), 1.0)

        # 50% horizontal overlap
        b2 = [5.0, 0.0, 15.0, 10.0]
        # Inter = 5*10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
        self.assertAlmostEqual(calculate_iou(b1, b2), 1.0 / 3.0)

        # No overlap
        b3 = [20.0, 20.0, 30.0, 30.0]
        self.assertAlmostEqual(calculate_iou(b1, b3), 0.0)

    def test_box_validity(self):
        self.assertTrue(is_valid_box([10, 10, 50, 50], 100, 100))
        # Negative coords
        self.assertFalse(is_valid_box([-5, 10, 50, 50]))
        # Inverted coords (x2 < x1)
        self.assertFalse(is_valid_box([50, 10, 10, 50]))
        # Zero area
        self.assertFalse(is_valid_box([10, 10, 10, 50]))

    def test_perfect_prediction_full_score(self):
        gt = {
            "images": [{"id": "img1"}],
            "annotations": [
                {"image_id": "img1", "category_id": "resistor", "bbox_xyxy": [10, 10, 50, 50]},
                {"image_id": "img1", "category_id": "capacitor", "bbox_xyxy": [100, 100, 150, 150]}
            ]
        }
        pred = {
            "predictions": [
                {
                    "image_id": "img1",
                    "status": "success",
                    "objects": [
                        {"label": "resistor", "bbox_xyxy": [10, 10, 50, 50], "score": 0.95},
                        {"label": "capacitor", "bbox_xyxy": [100, 100, 150, 150], "score": 0.90}
                    ]
                }
            ]
        }
        res = evaluate_detections(gt, pred, iou_threshold=0.5)
        self.assertEqual(res["micro"]["precision"], 1.0)
        self.assertEqual(res["micro"]["recall"], 1.0)
        self.assertEqual(res["micro"]["f1"], 1.0)
        self.assertEqual(res["parse_success_rate"], 1.0)

    def test_duplicate_prediction_penalizes_fp(self):
        # 1 GT, 2 identical predictions -> 1 TP, 1 FP -> Precision=0.5, Recall=1.0
        gt = {
            "images": [{"id": "img1"}],
            "annotations": [
                {"image_id": "img1", "category_id": "resistor", "bbox_xyxy": [10, 10, 50, 50]}
            ]
        }
        pred = {
            "predictions": [
                {
                    "image_id": "img1",
                    "status": "success",
                    "objects": [
                        {"label": "resistor", "bbox_xyxy": [10, 10, 50, 50], "score": 0.95},
                        {"label": "resistor", "bbox_xyxy": [10, 10, 50, 50], "score": 0.90} # Duplicate
                    ]
                }
            ]
        }
        res = evaluate_detections(gt, pred, iou_threshold=0.5)
        self.assertEqual(res["per_class"]["resistor"]["tp"], 1)
        self.assertEqual(res["per_class"]["resistor"]["fp"], 1)
        self.assertEqual(res["per_class"]["resistor"]["fn"], 0)
        self.assertAlmostEqual(res["micro"]["precision"], 0.5)
        self.assertAlmostEqual(res["micro"]["recall"], 1.0)

    def test_wrong_category(self):
        # GT is resistor, predicted capacitor -> 0 TP, 1 FP (capacitor), 1 FN (resistor)
        gt = {
            "images": [{"id": "img1"}],
            "annotations": [
                {"image_id": "img1", "category_id": "resistor", "bbox_xyxy": [10, 10, 50, 50]}
            ]
        }
        pred = {
            "predictions": [
                {
                    "image_id": "img1",
                    "status": "success",
                    "objects": [
                        {"label": "capacitor", "bbox_xyxy": [10, 10, 50, 50]}
                    ]
                }
            ]
        }
        res = evaluate_detections(gt, pred, iou_threshold=0.5)
        self.assertEqual(res["per_class"]["resistor"]["fn"], 1)
        self.assertEqual(res["per_class"]["capacitor"]["fp"], 1)
        self.assertEqual(res["micro"]["precision"], 0.0)
        self.assertEqual(res["micro"]["recall"], 0.0)

    def test_dense_components_scale(self):
        # 120 components on single PCB
        annotations = []
        predictions = []
        for i in range(120):
            box = [float(i * 10), 0.0, float(i * 10 + 8), 8.0]
            annotations.append({"image_id": "pcb_dense", "category_id": "resistor", "bbox_xyxy": box})
            predictions.append({"label": "resistor", "bbox_xyxy": box})

        gt = {"images": [{"id": "pcb_dense"}], "annotations": annotations}
        pred = {"predictions": [{"image_id": "pcb_dense", "status": "success", "objects": predictions}]}

        res = evaluate_detections(gt, pred, iou_threshold=0.5)
        self.assertEqual(res["per_class"]["resistor"]["tp"], 120)
        self.assertEqual(res["per_class"]["resistor"]["gt_support"], 120)
        self.assertEqual(res["micro"]["precision"], 1.0)
    def test_unknown_category_consistency(self):
        from scripts.evaluate import compute_official_coco_eval
        gt = {
            "images": [{"id": 1}],
            "categories": [{"id": 3, "name": "ic"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}
            ]
        }
        # Prediction has the right coordinates, but category is 'component'
        pred = {
            "predictions": [
                {
                    "image_id": 1,
                    "status": "success",
                    "objects": [
                        {"label": "component", "bbox_xyxy": [10, 10, 50, 50], "score": 0.95}
                    ]
                }
            ]
        }
        res_std = evaluate_detections(gt, pred, iou_threshold=0.5)
        self.assertEqual(res_std["micro"]["precision"], 0.0)
        self.assertEqual(res_std["micro"]["recall"], 0.0)
        self.assertEqual(res_std["per_class"]["ic"]["fn"], 1)

        res_coco = compute_official_coco_eval(gt, pred)
        self.assertEqual(res_coco["AP_50"], 0.0, "Unknown category 'component' must NOT match 'ic' in COCO eval!")

    def test_parse_failed_status_consistency(self):
        from scripts.evaluate import compute_official_coco_eval
        gt = {
            "images": [{"id": 1}],
            "categories": [{"id": 3, "name": "ic"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}
            ]
        }
        # Prediction has boxes but status is parse_failed
        pred = {
            "predictions": [
                {
                    "image_id": 1,
                    "status": "parse_failed",
                    "objects": [
                        {"label": "ic", "bbox_xyxy": [10, 10, 50, 50], "score": 0.95}
                    ]
                }
            ]
        }
        res_std = evaluate_detections(gt, pred, iou_threshold=0.5)
        self.assertEqual(res_std["tp"], 0)
        self.assertEqual(res_std["fp"], 0)
        self.assertEqual(res_std["fn"], 1)

        res_coco = compute_official_coco_eval(gt, pred)
        self.assertEqual(res_coco["total_detections"], 0)
        self.assertEqual(res_coco["AP_50"], 0.0)

    def test_verify_lora_adapter_provenance_strictness(self):
        import tempfile
        import json
        from scripts.run_test_matrix import verify_lora_adapter_provenance

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "adapter_metadata.json")

            # Case 1: Incomplete metadata (only canonical_split_verified=true)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"canonical_split_verified": True}, f)
            with self.assertRaises(ValueError) as ctx:
                verify_lora_adapter_provenance(tmpdir, "fake_split_hash", ["test_board_1"])
            self.assertIn("lacks verified split_sha256 metadata", str(ctx.exception))

            # Case 2: Mismatched split hash
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"split_sha256": "wrong_hash", "train_boards": ["train_board_1"]}, f)
            with self.assertRaises(ValueError) as ctx:
                verify_lora_adapter_provenance(tmpdir, "canonical_hash", ["test_board_1"])
            self.assertIn("does not match canonical test split hash", str(ctx.exception))

            # Case 3: Missing train_boards
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"split_sha256": "canonical_hash", "train_boards": []}, f)
            with self.assertRaises(ValueError) as ctx:
                verify_lora_adapter_provenance(tmpdir, "canonical_hash", ["test_board_1"])
            self.assertIn("lacks train_boards manifest", str(ctx.exception))

            # Case 4: Overlapping train_boards (data leakage)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"split_sha256": "canonical_hash", "train_boards": ["board_A", "leak_board"]}, f)
            with self.assertRaises(ValueError) as ctx:
                verify_lora_adapter_provenance(tmpdir, "canonical_hash", ["leak_board", "board_C"])
            self.assertIn("overlap with canonical test boards", str(ctx.exception))

            # Case 5: Valid, fully verified, disjoint provenance
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"split_sha256": "canonical_hash", "train_boards": ["board_A", "board_B"]}, f)
            # Should not raise any error
            verify_lora_adapter_provenance(tmpdir, "canonical_hash", ["board_C", "board_D"])

    def test_nan_and_inf_box_rejected(self):
        self.assertFalse(is_valid_box([float("nan"), 10, 50, 50]))
        self.assertFalse(is_valid_box([10, float("nan"), 50, 50]))
        self.assertFalse(is_valid_box([10, 10, float("nan"), 50]))
        self.assertFalse(is_valid_box([10, 10, 50, float("nan")]))
        self.assertFalse(is_valid_box([10, 10, float("inf"), 50]))
        self.assertFalse(is_valid_box([10, 10, 50, -float("inf")]))

    def test_out_of_bounds_box_rejected(self):
        # Image is 1000x1000
        self.assertFalse(is_valid_box([0, 0, 1500, 1500], img_w=1000, img_h=1000))
        self.assertFalse(is_valid_box([1050, 10, 1080, 50], img_w=1000, img_h=1000))
        self.assertFalse(is_valid_box([10, 1050, 50, 1080], img_w=1000, img_h=1000))
        self.assertTrue(is_valid_box([10, 10, 50, 50], img_w=1000, img_h=1000))

    def test_duplicate_records_consistency(self):
        from scripts.evaluate import extract_clean_predictions
        gt = {
            "images": [{"id": 1, "width": 1000, "height": 1000}],
            "categories": [{"id": 1, "name": "resistor"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}]
        }
        # Two records for the same image 1
        pred = {
            "predictions": [
                {"image_id": 1, "status": "success", "objects": [{"label": "resistor", "bbox_xyxy": [10, 10, 50, 50]}]},
                {"image_id": 1, "status": "success", "objects": [{"label": "resistor", "bbox_xyxy": [10, 10, 50, 50]}]}
            ]
        }
        pred_by_img, coco_detections, stats = extract_clean_predictions(gt, pred)
        self.assertEqual(len(pred_by_img[str(1)]), 1, "Ordinary evaluator should have exactly 1 record's boxes")
        self.assertEqual(len(coco_detections), 1, "COCO evaluator should have exactly 1 record's boxes, matching ordinary")
        self.assertEqual(stats["duplicate_records_count"], 1)

    def test_partial_failed_and_unexpected_status_not_successful(self):
        gt = {
            "images": [{"id": 1}, {"id": 2}],
            "categories": [{"id": 1, "name": "resistor"}],
            "annotations": []
        }
        pred = {
            "predictions": [
                {"image_id": 1, "status": "partial_failed", "objects": []},
                {"image_id": 2, "status": "some_unexpected_state", "objects": []}
            ]
        }
        res = evaluate_detections(gt, pred)
        self.assertEqual(res["complete_parse_images"], 0)
        self.assertEqual(res["parse_failed_images"], 1)
        self.assertEqual(res["unexpected_status_images"], 1)
        self.assertEqual(res["parse_success_rate"], 0.0, "partial_failed and unexpected must NOT yield 100% parse success")

    def test_missing_prediction_denominator(self):
        gt = {
            "images": [{"id": 1}, {"id": 2}],
            "categories": [{"id": 1, "name": "resistor"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox_xyxy": [10, 10, 50, 50]},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox_xyxy": [10, 10, 50, 50]}
            ]
        }
        # Only 1 image provided in predictions
        pred = {
            "predictions": [
                {"image_id": 1, "status": "success", "objects": [{"label": "resistor", "bbox_xyxy": [10, 10, 50, 50]}]}
            ]
        }
        res = evaluate_detections(gt, pred)
        self.assertEqual(res["total_images"], 2)
        self.assertEqual(res["complete_parse_images"], 1)
        self.assertEqual(res["missing_prediction_images"], 1)
        self.assertAlmostEqual(res["parse_success_rate"], 0.5, places=2, msg="Denominator must be total expected images (2), not provided (1)")

    def test_partial_success_retains_boxes_for_eval(self):
        gt = {
            "images": [{"id": 1}],
            "categories": [{"id": 1, "name": "resistor"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox_xyxy": [10, 10, 50, 50]}
            ]
        }
        pred = {
            "predictions": [
                {
                    "image_id": 1,
                    "status": "partial_success",
                    "objects": [
                        {"label": "resistor", "bbox_xyxy": [10, 10, 50, 50], "score": 0.9}
                    ]
                }
            ]
        }
        res = evaluate_detections(gt, pred)
        self.assertEqual(res["tp"], 1, "Partial success must retain valid boxes for evaluation")
        self.assertEqual(res["partial_parse_images"], 1)
        self.assertEqual(res["complete_parse_images"], 0)
        self.assertAlmostEqual(res["effective_parse_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()



