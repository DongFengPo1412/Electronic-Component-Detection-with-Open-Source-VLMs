"""
Comprehensive Evaluator for Electronic Component Detection using Open-Source VLMs.
Implements:
1. Unified Clean Prediction Extraction (Status filtering, Category mapping, Box validation)
2. One-to-One IoU Matching (P, R, F1 at IoU 0.50 and 0.75)
3. Micro & Macro Statistics and Per-class Support
4. System Diagnostics (Parse Success Rate, Invalid Boxes, Unknown Categories)
5. Official COCO Evaluation Wrapper (AP50, AP75, AP@[.50:.95])
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional, Sequence
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


import math


def is_valid_box(box: Any, img_w: Optional[float] = None, img_h: Optional[float] = None) -> bool:
    """Check if bbox [x1, y1, x2, y2] is non-empty, finite, non-negative, and strictly within image boundaries."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    for c in box:
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            return False
        if math.isnan(c) or math.isinf(c):
            return False
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if x2 <= x1 or y2 <= y1:
        return False
    if x1 < 0.0 or y1 < 0.0:
        return False
    if img_w is not None:
        if x1 >= img_w or x2 > img_w:
            return False
    if img_h is not None:
        if y1 >= img_h or y2 > img_h:
            return False
    return True


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pcb_vlm.config import CANONICAL_CATEGORY_COCO_LIST
DEFAULT_PCB_CATEGORIES = CANONICAL_CATEGORY_COCO_LIST



def extract_clean_predictions(
    gt_data: Dict[str, Any],
    pred_data: Dict[str, Any],
    expected_image_ids: Optional[List[Any]] = None
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Unified extraction, validation, and category resolution for both standard and COCO evaluators.
    Guarantees:
    1. Failed predictions contribute 0 boxes to BOTH evaluators.
    2. Unknown categories NEVER default to a valid class in COCO or standard eval.
    3. Invalid boxes (negative, zero width/height, NaN, out of image bounds) are filtered out.
    4. Exact same deduplicated predictions and boxes are shared between standard and COCO evaluators.
    5. Detailed status counts (complete, partial, failed, missing, unexpected, truncated).
    """
    cat_id_to_name = {}
    cat_name_to_id = {}
    categories_source = gt_data.get("categories")
    if not categories_source:
        categories_source = list(DEFAULT_PCB_CATEGORIES)
        for ann in gt_data.get("annotations", []):
            raw_c = ann.get("category_id", ann.get("category"))
            if raw_c is not None and not any(c["name"] == str(raw_c).lower() for c in categories_source):
                categories_source.append({"id": len(categories_source) + 1, "name": str(raw_c).lower()})

    for c in categories_source:
        c_name = str(c.get("name", "")).strip().lower()
        c_id = c.get("id")
        if c_id is not None:
            cat_id_to_name[c_id] = c_name
            cat_id_to_name[str(c_id)] = c_name
            try:
                cat_name_to_id[str(c_id)] = int(c_id)
                cat_name_to_id[int(c_id)] = int(c_id)
            except ValueError:
                cat_name_to_id[str(c_id)] = c_id
        if c_name:
            cat_id_to_name[c_name] = c_name
            if c_id is not None:
                try:
                    cat_name_to_id[c_name] = int(c_id)
                except ValueError:
                    cat_name_to_id[c_name] = c_id

    # Index image dimensions from GT
    img_dims: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for item in gt_data.get("images", []):
        img_id_str = str(item.get("image_id", item.get("id")))
        w = item.get("width")
        h = item.get("height")
        img_dims[img_id_str] = (float(w) if w is not None else None, float(h) if h is not None else None)

    # Determine expected images
    if expected_image_ids is not None:
        target_ids = [str(x) for x in expected_image_ids]
    else:
        target_ids = [str(item.get("image_id", item.get("id"))) for item in gt_data.get("images", [])]

    total_expected_count = len(target_ids)

    # Initialize pred_by_img for all expected images
    pred_by_img: Dict[str, List[Dict[str, Any]]] = {iid: [] for iid in target_ids}

    # Deduplicate incoming prediction records deterministically
    preds_list = pred_data if isinstance(pred_data, list) else pred_data.get("predictions", [])
    deduped_preds: Dict[str, Dict[str, Any]] = {}
    duplicate_records_count = 0

    for res in preds_list:
        img_id_str = str(res.get("image_id"))
        if img_id_str in deduped_preds:
            duplicate_records_count += 1
        deduped_preds[img_id_str] = res

    # Detailed status counters
    stats: Dict[str, Any] = {
        "total_expected_images": total_expected_count,
        "complete_parse_images": 0,
        "partial_parse_images": 0,
        "no_objects_detected_images": 0,
        "parse_failed_images": 0,
        "inference_failed_images": 0,
        "missing_prediction_images": 0,
        "unexpected_status_images": 0,
        "truncated_images": 0,
        "successful_parse_count": 0,  # complete + partial + no_objects
        "invalid_box_count": 0,
        "unknown_category_count": 0,
        "duplicate_records_count": duplicate_records_count,
        "out_of_bounds_box_count": 0
    }

    VALID_SUCCESS_STATUSES = {"success", "partial_success", "no_objects_detected"}
    VALID_FAILED_STATUSES = {"parse_failed", "parse_filtered", "partial_failed", "inference_failed", "missing_prediction"}

    for iid in target_ids:
        if iid not in deduped_preds:
            stats["missing_prediction_images"] += 1
            pred_by_img[iid] = []
            continue

        res = deduped_preds[iid]
        raw_status = res.get("status")
        status = str(raw_status).strip() if raw_status is not None else "unexpected"

        if res.get("is_truncated", False):
            stats["truncated_images"] += 1

        img_w, img_h = img_dims.get(iid, (None, None))

        if status not in VALID_SUCCESS_STATUSES and status not in VALID_FAILED_STATUSES:
            # Unknown / unexpected status is recorded as failure
            stats["unexpected_status_images"] += 1
            pred_by_img[iid] = []
            continue

        if status == "inference_failed":
            stats["inference_failed_images"] += 1
            pred_by_img[iid] = []
            continue

        if status in ["parse_failed", "parse_filtered", "partial_failed"]:
            stats["parse_failed_images"] += 1
            pred_by_img[iid] = []
            continue

        if status == "no_objects_detected":
            stats["no_objects_detected_images"] += 1
            stats["complete_parse_images"] += 1
            stats["successful_parse_count"] += 1
            pred_by_img[iid] = []
            continue

        if status == "partial_success":
            stats["partial_parse_images"] += 1
            stats["successful_parse_count"] += 1
        elif status == "success":
            stats["complete_parse_images"] += 1
            stats["successful_parse_count"] += 1

        # Extract and validate objects
        objs = res.get("objects", [])
        valid_objs = []

        for obj in objs:
            box = obj.get("bbox_xyxy", obj.get("bbox"))
            if not is_valid_box(box, img_w=img_w, img_h=img_h):
                stats["invalid_box_count"] += 1
                continue

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            raw_cat = obj.get("category_id", obj.get("label", obj.get("category")))
            raw_cat_str = str(raw_cat).strip().lower() if raw_cat is not None else ""
            try:
                score = float(obj.get("score", 1.0)) if obj.get("score") is not None else 1.0
                if not math.isfinite(score):
                    raise ValueError('Nonfinite score')
            except (TypeError, ValueError):
                stats['invalid_box_count'] += 1
                continue

            if raw_cat_str in cat_name_to_id:
                canonical_cat = cat_id_to_name.get(cat_name_to_id[raw_cat_str], raw_cat_str)
                canonical_id = cat_name_to_id[raw_cat_str]
            elif raw_cat in cat_id_to_name:
                canonical_cat = cat_id_to_name[raw_cat]
                canonical_id = cat_name_to_id.get(canonical_cat, 999999)
            else:
                stats["unknown_category_count"] += 1
                canonical_cat = f"unknown_{raw_cat_str}" if raw_cat_str else "unknown"
                canonical_id = 999999

            valid_objs.append({
                "category": canonical_cat,
                "category_id": canonical_id,
                "bbox": [x1, y1, x2, y2],
                "score": score
            })

        pred_by_img[iid] = valid_objs

    # Populate coco_detections exclusively from the final deduplicated pred_by_img
    coco_detections: List[Dict[str, Any]] = []
    for iid_str, objs in pred_by_img.items():
        try:
            iid_int = int(iid_str)
        except (ValueError, TypeError):
            iid_int = None
        if iid_int is not None:
            for obj in objs:
                x1, y1, x2, y2 = obj["bbox"]
                coco_detections.append({
                    "image_id": iid_int,
                    "category_id": obj["category_id"],
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": obj["score"]
                })

    # Rates based strictly on total_expected_images as the denominator
    denom = max(1, total_expected_count)
    stats["complete_parse_rate"] = round(stats["complete_parse_images"] / denom, 4)
    stats["parse_success_rate"] = stats["complete_parse_rate"]
    stats["effective_parse_rate"] = round(stats["successful_parse_count"] / denom, 4)

    return pred_by_img, coco_detections, stats


def scope_ground_truth(gt, expected):
    if expected is None:
        return gt
    ids = {str(i) for i in expected}
    available = {str(i.get('id', i.get('image_id'))) for i in gt.get('images', [])}
    if not ids.issubset(available):
        raise ValueError('Expected evaluation images absent from GT')
    return {**gt, 'images':[i for i in gt['images'] if str(i.get('id', i.get('image_id'))) in ids],
            'annotations':[a for a in gt.get('annotations', []) if str(a['image_id']) in ids]}


def match_category_predictions_one_to_one(
    gts_cat: Sequence[Dict[str, Any]],
    preds_cat: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Match predictions to GTs within one category using evaluate_detections greedy order.

    Predictions are sorted stably by ``score`` descending (or box area when ``score``
    is ``None``). Each prediction is greedily paired with the unmatched GT in the same
    category that maximizes IoU strictly (`>`), accepting the match when
    ``best_iou >= iou_threshold``.
    """
    matched_gts = set()
    pred_to_gt: Dict[int, Tuple[int, float]] = {}
    gt_to_pred: Dict[int, Tuple[int, float]] = {}
    matched_pairs: List[Tuple[int, int, float]] = []

    preds_sorted = sorted(
        enumerate(preds_cat),
        key=lambda item: (
            item[1]["score"]
            if item[1].get("score") is not None
            else (item[1]["bbox"][2] - item[1]["bbox"][0]) * (item[1]["bbox"][3] - item[1]["bbox"][1])
        ),
        reverse=True,
    )

    tp = 0
    fp = 0
    for pred_idx, p in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, g in enumerate(gts_cat):
            if gt_idx in matched_gts:
                continue
            iou = calculate_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gts.add(best_gt_idx)
            pred_to_gt[pred_idx] = (best_gt_idx, float(best_iou))
            gt_to_pred[best_gt_idx] = (pred_idx, float(best_iou))
            matched_pairs.append((pred_idx, best_gt_idx, float(best_iou)))
        else:
            fp += 1

    return {
        "matched_pairs": matched_pairs,
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
        "tp": tp,
        "fp": fp,
        "fn": len(gts_cat) - len(matched_gts),
    }


def evaluate_detections(
    gt_data: Dict[str, Any],
    pred_data: Dict[str, Any],
    iou_threshold: float = 0.50,
    expected_image_ids: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth using strict one-to-one greedy matching.
    """
    gt_data = scope_ground_truth(gt_data, expected_image_ids)
    # Map GT objects by image_id
    gt_by_img = {}
    for item in gt_data.get("images", []):
        img_id = str(item.get("image_id", item.get("id")))
        gt_by_img[img_id] = []

    # Build category ID to name mapping
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

    # Unified clean extraction
    pred_by_img, _, parse_stats = extract_clean_predictions(gt_data, pred_data, expected_image_ids=expected_image_ids)

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

            # Greedy one-to-one matching (canonical helper shared with showcase/replay)
            match_res = match_category_predictions_one_to_one(gts_cat, preds_cat, iou_threshold=iou_threshold)
            cat_stats[cat]["tp"] += match_res["tp"]
            cat_stats[cat]["fp"] += match_res["fp"]
            cat_stats[cat]["fn"] += match_res["fn"]

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

    for cat in sorted(categories):
        s = cat_stats[cat]
        c_tp = s["tp"]
        c_fp = s["fp"]
        c_fn = s["fn"]
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0

        per_class_results[cat] = {
            "gt_support": s["gt_count"],
            "pred_count": s["pred_count"],
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1": round(c_f1, 4)
        }

        if s["gt_count"] > 0:
            macro_precisions.append(c_prec)
            macro_recalls.append(c_rec)
            macro_f1s.append(c_f1)

    total_predictions = sum(s["pred_count"] for s in cat_stats.values())
    total_gt = sum(s["gt_count"] for s in cat_stats.values())
    total_images = len(gt_by_img)

    summary = {
        "iou_threshold": iou_threshold,
        "total_images": total_images,
        "total_gt": total_gt,
        "total_predictions": total_predictions,
        "successful_parse_images": parse_stats["successful_parse_count"],
        "complete_parse_images": parse_stats["complete_parse_images"],
        "partial_parse_images": parse_stats["partial_parse_images"],
        "no_objects_detected_images": parse_stats["no_objects_detected_images"],
        "parse_failed_images": parse_stats["parse_failed_images"],
        "inference_failed_images": parse_stats["inference_failed_images"],
        "missing_prediction_images": parse_stats["missing_prediction_images"],
        "unexpected_status_images": parse_stats["unexpected_status_images"],
        "truncated_images": parse_stats["truncated_images"],
        "parse_success_rate": parse_stats["parse_success_rate"],
        "complete_parse_rate": parse_stats["complete_parse_rate"],
        "effective_parse_rate": parse_stats["effective_parse_rate"],
        "invalid_box_count": parse_stats["invalid_box_count"],
        "unknown_category_count": parse_stats["unknown_category_count"],
        "duplicate_records_count": parse_stats["duplicate_records_count"],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
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


def compute_official_coco_eval(
    gt_coco_dict: Dict[str, Any],
    pred_data: Dict[str, Any],
    pre_extracted_coco_detections: Optional[List[Dict[str, Any]]] = None,
    expected_image_ids: Optional[List[Any]] = None,
    evaluation_config: Optional[str] = None
) -> Dict[str, Any]:
    """Computes official COCO AP@[.50:.95], AP50, AP75 via pycocotools."""
    gt_coco_dict = scope_ground_truth(gt_coco_dict, expected_image_ids)
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import tempfile
    except ImportError:
        return {"error": "pycocotools not installed"}

    if pre_extracted_coco_detections is not None:
        coco_detections = pre_extracted_coco_detections
    else:
        _, coco_detections, _ = extract_clean_predictions(gt_coco_dict, pred_data, expected_image_ids=expected_image_ids)

    gt_dict_copy = dict(gt_coco_dict)
    if "categories" not in gt_dict_copy or not gt_dict_copy["categories"]:
        gt_dict_copy["categories"] = DEFAULT_PCB_CATEGORIES

    import copy
    import io
    import contextlib
    import yaml
    config_path = evaluation_config or os.path.join(os.path.dirname(__file__), "..", "configs", "evaluation.yaml")
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    limits = config["metrics"]["coco_max_dets"]
    if limits[:3] != [1, 10, 100] or any(not isinstance(n, int) or n <= 0 for n in limits):
        raise ValueError("COCO maxDets must start with [1,10,100]")
    def calculate(limit):
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = copy.deepcopy(gt_dict_copy)
            coco_gt.dataset.setdefault("info", {})
            coco_gt.createIndex()
            for ann in coco_gt.dataset.get("annotations", []):
                ann.setdefault("iscrowd", 0)
                ann.setdefault("area", ann["bbox"][2] * ann["bbox"][3])
            if not coco_detections:
                return {"mAP_50_95": 0.0, "AP_50": 0.0, "AP_75": 0.0, "maxDets": [1, 10, limit]}
            coco_dt = coco_gt.loadRes(coco_detections)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.maxDets = [1, 10, limit]
            evaluator.evaluate(); evaluator.accumulate()
            # Read the actual precision tensor: summarize() assumes maxDets=100.
            precision = evaluator.eval["precision"][:, :, :, 0, 2]
            def mean(values):
                valid = values[values > -1]
                return round(float(valid.mean()), 4) if valid.size else -1.0
            result = {"mAP_50_95": mean(precision), "AP_50": mean(precision[0]),
                      "AP_75": mean(precision[5]), "maxDets": [1, 10, limit]}
            for i, label in enumerate(("small", "medium", "large"), 1):
                result["AP_" + label] = mean(evaluator.eval["precision"][:, :, :, i, 2])
            result["AR_50_95"] = mean(evaluator.eval["recall"][:, :, 0, 2])
            return result
    result = calculate(100)
    result["total_detections"] = len(coco_detections)
    result["score_policy"] = "Provided scores retained; missing scores tied at 1.0, uncalibrated VLM AP is supplementary"
    if max(limits) > 100:
        result["dense_extended"] = calculate(max(limits))
    return result


def filter_gt_and_preds_by_split(
    gt_data: Dict[str, Any],
    pred_data: Dict[str, Any],
    splits_data: Optional[Dict[str, Any]] = None,
    split_name: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Strictly filter ground truth and align predictions by the split's target image IDs."""
    if not splits_data or not split_name:
        return gt_data, pred_data

    target_image_ids = set(splits_data["splits"][split_name]["image_ids"])
    print(f"[Evaluation Scope] Filtering to split '{split_name}' ({len(target_image_ids)} images: {sorted(list(target_image_ids))})")

    # Filter GT images and annotations
    filtered_images = [img for img in gt_data.get("images", []) if (img.get("id") in target_image_ids or img.get("image_id") in target_image_ids)]
    filtered_anns = [ann for ann in gt_data.get("annotations", []) if ann.get("image_id") in target_image_ids]

    filtered_gt = {
        "images": filtered_images,
        "annotations": filtered_anns,
        "categories": gt_data.get("categories", [])
    }

    # Ensure every target image has an entry in predictions
    pred_list = pred_data.get("predictions", [])
    preds_by_iid = {p.get("image_id"): p for p in pred_list}
    aligned_preds = []
    for iid in sorted(list(target_image_ids)):
        if iid in preds_by_iid:
            aligned_preds.append(preds_by_iid[iid])
        else:
            aligned_preds.append({
                "image_id": iid,
                "status": "missing_prediction",
                "objects": [],
                "num_detected": 0
            })

    filtered_pred = {
        "predictions": aligned_preds
    }
    return filtered_gt, filtered_pred


def evaluate_standard_benchmark(gt_data: Dict[str, Any], pred_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates standard detection metrics at IoU 0.50, 0.75, multi-threshold averages,
    official COCO AP, and diagnostic loose threshold IoU 0.10.
    """
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold_results = {}
    f1_list = []
    precision_list = []
    recall_list = []

    for t in thresholds:
        res = evaluate_detections(gt_data, pred_data, iou_threshold=t)
        threshold_results[f"iou_{int(t*100)}"] = {
            "precision": res["micro"]["precision"],
            "recall": res["micro"]["recall"],
            "f1": res["micro"]["f1"],
            "tp": sum(c["tp"] for c in res["per_class"].values()),
            "fp": sum(c["fp"] for c in res["per_class"].values()),
            "fn": sum(c["fn"] for c in res["per_class"].values()),
        }
        f1_list.append(res["micro"]["f1"])
        precision_list.append(res["micro"]["precision"])
        recall_list.append(res["micro"]["recall"])

    # Official COCO eval via pycocotools
    official_coco = compute_official_coco_eval(gt_data, pred_data)

    # Diagnostic-only loose threshold
    diag_res = evaluate_detections(gt_data, pred_data, iou_threshold=0.10)
    diag_summary = {
        "note": "Diagnostic-only loose overlap (IoU >= 0.10) for spatial hallucination analysis; NOT a valid metric for precision detection claims.",
        "precision": diag_res["micro"]["precision"],
        "recall": diag_res["micro"]["recall"],
        "f1": diag_res["micro"]["f1"],
        "tp": sum(c["tp"] for c in diag_res["per_class"].values()),
        "fp": sum(c["fp"] for c in diag_res["per_class"].values()),
        "fn": sum(c["fn"] for c in diag_res["per_class"].values()),
    }

    res_50 = threshold_results["iou_50"]
    res_75 = threshold_results["iou_75"]

    return {
        "standard_evaluation": {
            "iou_50": res_50,
            "iou_75": res_75,
            "mean_precision_across_iou_thresholds": round(float(np.mean(precision_list)), 4),
            "mean_recall_across_iou_thresholds": round(float(np.mean(recall_list)), 4),
            "mean_f1_across_iou_thresholds": round(float(np.mean(f1_list)), 4),
            "official_coco_metrics": official_coco
        },
        "diagnostic_auxiliary": {
            "iou_10_loose": diag_summary
        },
        "detailed_per_class_iou_50": evaluate_detections(gt_data, pred_data, iou_threshold=0.50)["per_class"]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PCB Electronic Component Detection")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON/JSONL file")
    parser.add_argument("--splits", help="Optional path to splits.json for split-scoped evaluation")
    parser.add_argument("--split-name", help="Split name to evaluate (e.g. test, val)")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold (default: 0.50)")
    parser.add_argument("--benchmark", action="store_true", help="Run full multi-threshold benchmark (AP50, AP75, official COCO AP, and diagnostic)")
    parser.add_argument("--output", help="Optional output path to save JSON report")
    args = parser.parse_args()

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    if args.pred.endswith(".jsonl"):
        predictions = []
        with open(args.pred, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        pred_data = {"predictions": predictions}
    else:
        with open(args.pred, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

    # Apply split scoping if requested
    if args.splits and args.split_name:
        with open(args.splits, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        gt_data, pred_data = filter_gt_and_preds_by_split(gt_data, pred_data, splits_data, args.split_name)

    if args.benchmark:
        results = evaluate_standard_benchmark(gt_data, pred_data)
    else:
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
