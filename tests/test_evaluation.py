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
        self.assertEqual(res["micro"]["recall"], 1.0)

if __name__ == "__main__":
    unittest.main()
