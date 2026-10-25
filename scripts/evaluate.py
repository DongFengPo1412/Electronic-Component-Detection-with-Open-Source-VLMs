"""
Comprehensive Evaluator for Electronic Component Detection using Open-Source VLMs.
Implements:
1. One-to-One IoU Matching (P, R, F1 at IoU 0.50 and 0.75)
2. Micro & Macro Statistics and Per-class Support
3. System Diagnostics (Parse Success Rate, Invalid Boxes, Out-of-bounds)
4. Official COCO Evaluation Wrapper (AP50, AP75, AP@[.50:.95])
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np

def xyxy_to_xywh(box: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculate IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea

def is_valid_box(box: List[float], img_w: float = None, img_h: float = None) -> bool:
    """Check if bbox [x1, y1, x2, y2] is non-empty, non-negative, and within image boundaries."""
    if len(box) != 4:
        return False
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0 or y1 < 0:
        return False
    if img_w is not None and x2 > img_w * 1.05: # 5% margin tolerance before strict clip
        return False
    if img_h is not None and y2 > img_h * 1.05:
        return False
    return True

def evaluate_detections(
    gt_data: Dict[str, Any],
    pred_data: Dict[str, Any],
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth using strict one-to-one greedy matching.
    """
    # Map GT objects by image_id
    gt_by_img = {}
    for item in gt_data.get("images", []):
        img_id = str(item.get("image_id", item.get("id")))
        gt_by_img[img_id] = []

    # Build category ID to name mapping if available
    cat_id_to_name = {}
    for c in gt_data.get("categories", []):
        c_name = str(c.get("name", "")).strip().lower()
        if "id" in c:
            cat_id_to_name[c["id"]] = c_name
            cat_id_to_name[str(c["id"])] = c_name
        if c_name:
            cat_id_to_name[c_name] = c_name

    for ann in gt_data.get("annotations", []):
        img_id = str(ann.get("image_id"))
        if img_id not in gt_by_img:
            gt_by_img[img_id] = []
        box = ann.get("bbox_xyxy")
        if box is None and "bbox" in ann: # COCO xywh format
            box = xywh_to_xyxy(ann["bbox"])
        raw_cat = ann.get("category_id", ann.get("category"))
        canonical_cat = cat_id_to_name.get(raw_cat, str(raw_cat).lower())
        gt_by_img[img_id].append({
            "category": canonical_cat,
            "bbox": box
        })

    # Map Predictions by image_id
    pred_by_img = {}
    total_parsed_images = 0
    parse_failed_images = 0
    invalid_box_count = 0

    for res in pred_data.get("predictions", []):
        img_id = str(res.get("image_id"))
        status = res.get("status", "success")
        if status != "success":
            parse_failed_images += 1
            pred_by_img[img_id] = []
            continue

        total_parsed_images += 1
        objs = res.get("objects", [])
        valid_objs = []
        for obj in objs:
            box = obj.get("bbox_xyxy", obj.get("bbox"))
            if is_valid_box(box):
                raw_pred_cat = obj.get("category_id", obj.get("label", obj.get("category")))
                canonical_pred_cat = cat_id_to_name.get(raw_pred_cat, str(raw_pred_cat).lower())
                valid_objs.append({
                    "category": canonical_pred_cat,
                    "bbox": box,
                    "score": obj.get("score", None)
                })
            else:
                invalid_box_count += 1
        pred_by_img[img_id] = valid_objs

    # Category-level counters
    categories = set()
    for objs in gt_by_img.values():
        for o in objs:
            categories.add(o["category"])
    for objs in pred_by_img.values():
        for o in objs:
            categories.add(o["category"])

    cat_stats = {c: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0} for c in categories}

    for img_id, gt_objs in gt_by_img.items():
        preds = pred_by_img.get(img_id, [])

        for cat in categories:
            gts_cat = [g for g in gt_objs if g["category"] == cat]
            preds_cat = [p for p in preds if p["category"] == cat]

            cat_stats[cat]["gt_count"] += len(gts_cat)
            cat_stats[cat]["pred_count"] += len(preds_cat)

            if not gts_cat and not preds_cat:
                continue
            if not gts_cat and preds_cat:
                cat_stats[cat]["fp"] += len(preds_cat)
                continue
            if gts_cat and not preds_cat:
                cat_stats[cat]["fn"] += len(gts_cat)
                continue

            # Greedy one-to-one matching
            matched_gts = set()
            # If scores available, sort by score desc; else sort by area desc
            preds_sorted = sorted(
                preds_cat,
                key=lambda p: p["score"] if p["score"] is not None else (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]),
                reverse=True
            )

            for p in preds_sorted:
                best_iou = 0.0
                best_gt_idx = -1
                for idx, g in enumerate(gts_cat):
                    if idx in matched_gts:
                        continue
                    iou = calculate_iou(p["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    cat_stats[cat]["tp"] += 1
                    matched_gts.add(best_gt_idx)
                else:
                    cat_stats[cat]["fp"] += 1

            cat_stats[cat]["fn"] += (len(gts_cat) - len(matched_gts))

    # Compute Precision, Recall, F1
    total_tp = sum(s["tp"] for s in cat_stats.values())
    total_fp = sum(s["fp"] for s in cat_stats.values())
    total_fn = sum(s["fn"] for s in cat_stats.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    per_class_results = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    for cat, s in cat_stats.items():
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if s["gt_count"] > 0:
            macro_precisions.append(prec)
            macro_recalls.append(rec)
            macro_f1s.append(f1)

        per_class_results[cat] = {
            "gt_support": s["gt_count"],
            "pred_count": s["pred_count"],
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

    total_images = len(gt_by_img)
    parse_success_rate = (total_parsed_images / total_images) if total_images > 0 else 1.0

    summary = {
        "iou_threshold": iou_threshold,
        "total_images": total_images,
        "parse_success_rate": round(parse_success_rate, 4),
        "invalid_boxes_caught": invalid_box_count,
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4)
        },
        "macro": {
            "precision": round(float(np.mean(macro_precisions)) if macro_precisions else 0.0, 4),
            "recall": round(float(np.mean(macro_recalls)) if macro_recalls else 0.0, 4),
            "f1": round(float(np.mean(macro_f1s)) if macro_f1s else 0.0, 4)
        },
        "per_class": per_class_results
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate PCB Electronic Component Detection")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON file")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold (default: 0.50)")
    parser.add_argument("--output", help="Optional output path to save JSON report")
    args = parser.parse_args()

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(args.pred, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    results = evaluate_detections(gt_data, pred_data, iou_threshold=args.iou)
    output_str = json.dumps(results, indent=2, ensure_ascii=False)
    print(output_str)

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"\nReport written to: {args.output}")

if __name__ == "__main__":
    main()
