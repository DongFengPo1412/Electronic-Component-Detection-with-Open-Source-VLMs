"""
Error Analysis and Ablation Attribution Script (Stage 17).
Analyzes:
1. Error taxonomy: False Positives, Missed Detections (FN), Classification Confusions, Localization Drift.
2. Error distribution by target scale: Small (<32x32), Medium (32-96), Large (>96x96).
3. Representative case studies with precise coordinates.
"""

import os
import sys
import json
from collections import defaultdict


def box_iou(box1, box2):
    # box1: [x1, y1, x2, y2], box2: [x, y, w, h]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[0] + box2[2])
    y2 = min(box1[3], box2[1] + box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def analyze_errors_for_predictions(preds, coco_gts):
    # Group GT by image_id
    gt_by_img = defaultdict(list)
    for ann in coco_gts:
        gt_by_img[ann["image_id"]].append(ann)

    # Scale definition: COCO standard
    # Small: area < 1024 px2 (32x32)
    # Medium: 1024 <= area < 9216 (96x96)
    # Large: area >= 9216
    scale_counts = {"small": 0, "medium": 0, "large": 0}
    scale_recalled = {"small": 0, "medium": 0, "large": 0}

    total_predictions = 0
    true_positives_05 = 0
    true_positives_01 = 0
    localization_drifts = 0 # 0.1 <= IoU < 0.5
    class_confusions = 0 # IoU >= 0.1 but different class
    false_positives = 0 # IoU < 0.1 with any GT

    confusion_pairs = defaultdict(int)

    for item in preds:
        img_id = item["image_id"]
        objs = item.get("objects", [])
        gts = gt_by_img.get(img_id, [])
        total_predictions += len(objs)

        for p in objs:
            p_box = p["bbox_xyxy"]
            p_cls = p["label"]

            best_iou = 0.0
            best_gt = None
            for g in gts:
                iou = box_iou(p_box, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g

            if best_gt is not None and best_iou >= 0.10:
                if best_gt["category_name"] == p_cls:
                    if best_iou >= 0.50:
                        true_positives_05 += 1
                    else:
                        localization_drifts += 1
                    true_positives_01 += 1
                else:
                    class_confusions += 1
                    confusion_pairs[f"{p_cls} -> {best_gt['category_name']}"] += 1
            else:
                false_positives += 1

    # Audit GT recall by scale
    for ann in coco_gts:
        area = ann["area"]
        scale = "small" if area < 1024 else ("medium" if area < 9216 else "large")
        scale_counts[scale] += 1

        # Check if recalled at IoU >= 0.10
        img_preds = []
        for item in preds:
            if item["image_id"] == ann["image_id"]:
                img_preds = item.get("objects", [])
                break

        recalled = False
        for p in img_preds:
            if p["label"] == ann["category_name"] and box_iou(p["bbox_xyxy"], ann["bbox"]) >= 0.10:
                recalled = True
                break
        if recalled:
            scale_recalled[scale] += 1

    return {
        "total_predictions": total_predictions,
        "true_positives_05": true_positives_05,
        "true_positives_01": true_positives_01,
        "localization_drifts": localization_drifts,
        "class_confusions": class_confusions,
        "false_positives": false_positives,
        "top_confusion_pairs": sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5],
        "scale_analysis": {
            "small": {
                "total": scale_counts["small"],
                "recalled": scale_recalled["small"],
                "recall_rate": round(100 * scale_recalled["small"] / max(1, scale_counts["small"]), 2)
            },
            "medium": {
                "total": scale_counts["medium"],
                "recalled": scale_recalled["medium"],
                "recall_rate": round(100 * scale_recalled["medium"] / max(1, scale_counts["medium"]), 2)
            },
            "large": {
                "total": scale_counts["large"],
                "recalled": scale_recalled["large"],
                "recall_rate": round(100 * scale_recalled["large"] / max(1, scale_counts["large"]), 2)
            }
        }
    }


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    coco_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\coco_annotations.json"
    splits_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\splits.json"
    test_runs_dir = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_final_test_matrix"

    with open(coco_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_ids = set(splits["splits"]["test"]["image_ids"])
    test_gts = [ann for ann in coco["annotations"] if ann["image_id"] in test_ids]

    analysis_results = {}
    exp_folders = {
        "Q0": "Q0_qwen_base_whole",
        "Q1": "Q1_qwen_base_tiling",
        "Q2": "Q2_qwen_lora_whole",
        "Q3": "Q3_qwen_lora_tiling"
    }

    for exp_id, folder in exp_folders.items():
        preds_file = os.path.join(test_runs_dir, folder, "predictions.jsonl")
        if os.path.exists(preds_file):
            with open(preds_file, "r", encoding="utf-8") as f:
                preds = [json.loads(line) for line in f if line.strip()]
            analysis_results[exp_id] = analyze_errors_for_predictions(preds, test_gts)

    out_file = os.path.join(test_runs_dir, "error_analysis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    print(f"Error analysis saved to {out_file}")
    for exp_id, res in analysis_results.items():
        print(f"\n--- {exp_id} Error Breakdown ---")
        print(f"Total Detections:   {res['total_predictions']}")
        print(f"TP (IoU>=0.50):     {res['true_positives_05']}")
        print(f"Localization Drift: {res['localization_drifts']} (0.10 <= IoU < 0.50)")
        print(f"Class Confusion:    {res['class_confusions']}")
        print(f"False Positives:    {res['false_positives']}")
        print(f"Scale Recalls:      Small={res['scale_analysis']['small']['recall_rate']}%, Medium={res['scale_analysis']['medium']['recall_rate']}%, Large={res['scale_analysis']['large']['recall_rate']}%")


if __name__ == "__main__":
    main()
