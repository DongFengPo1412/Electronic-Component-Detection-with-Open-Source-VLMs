"""Run real GPU inference on fixed validation image (val_crop and DuetWIFI_Bottom)
using the newly fixed Qwen and DeepSeek adapters and evaluate metrics.
"""

import os
import sys
import json
import time
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pcb_vlm.models.qwen_adapter import QwenAdapter
from pcb_vlm.models.deepseek_adapter import DeepSeekAdapter
from evaluate import evaluate_detections, compute_official_coco_eval

CACHE_ROOT = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024"
AUDIT_DIR = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-followup-audit-20261027"

VAL_CROP_PATH = os.path.join(AUDIT_DIR, "val_crop.png")
DUETWIFI_BOTTOM_PATH = os.path.join(CACHE_ROOT, "datasets", "raw", "pcb_wacv_2019", "pcb_wacv_2019", "DuetWIFI_Bottom", "DuetWIFI_Bottom.jpg")
COCO_ANNO_PATH = os.path.join(CACHE_ROOT, "annotations_v2_canonical", "coco_annotations.json")

QWEN_MODEL_DIR = os.path.join(CACHE_ROOT, "models", "models", "Qwen--Qwen2-VL-2B-Instruct", "snapshots", "master")
QWEN_LORA_V2_DIR = os.path.join(CACHE_ROOT, "runs", "stage14_lora_training_v2", "best_adapter")
DEEPSEEK_MODEL_DIR = os.path.join(CACHE_ROOT, "models", "models", "deepseek-ai--deepseek-vl-1.3b-chat", "snapshots", "master")

OUTPUT_DIR = os.path.join(AUDIT_DIR, "validation_inference_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_ground_truth_for_crop():
    # DuetWIFI_Bottom crop [1380, 1139, 1380+512, 1139+512]
    # Exactly 4 LEDs inside this crop
    return [
        {"category_name": "led", "category_id": 5, "bbox_xyxy": [146.0, 237.0, 366.0, 276.0]},
        {"category_name": "led", "category_id": 5, "bbox_xyxy": [148.0, 299.0, 368.0, 338.0]},
        {"category_name": "led", "category_id": 5, "bbox_xyxy": [148.0, 359.0, 368.0, 398.0]},
        {"category_name": "led", "category_id": 5, "bbox_xyxy": [150.0, 421.0, 370.0, 460.0]},
    ]


def run_val_crop_inference():
    print(f"=== Running Real Validation Inference on: {VAL_CROP_PATH} ===")
    im = Image.open(VAL_CROP_PATH)
    w, h = im.size

    gt_boxes = get_ground_truth_for_crop()
    gt_dataset = {
        "images": [{"id": 1001, "file_name": "val_crop.png", "width": w, "height": h}],
        "annotations": [
            {
                "id": idx + 1,
                "image_id": 1001,
                "category_id": g["category_id"],
                "category_name": g["category_name"],
                "bbox": [g["bbox_xyxy"][0], g["bbox_xyxy"][1], g["bbox_xyxy"][2] - g["bbox_xyxy"][0], g["bbox_xyxy"][3] - g["bbox_xyxy"][1]],
                "area": (g["bbox_xyxy"][2] - g["bbox_xyxy"][0]) * (g["bbox_xyxy"][3] - g["bbox_xyxy"][1]),
                "iscrowd": 0
            }
            for idx, g in enumerate(gt_boxes)
        ],
        "categories": [
            {"id": 1, "name": "capacitor"},
            {"id": 2, "name": "resistor"},
            {"id": 3, "name": "ic"},
            {"id": 4, "name": "connector"},
            {"id": 5, "name": "led"},
            {"id": 6, "name": "transistor"},
            {"id": 7, "name": "diode"},
            {"id": 8, "name": "inductor"},
            {"id": 9, "name": "button"},
            {"id": 10, "name": "switch"},
            {"id": 11, "name": "clock"},
        ]
    }

    results = {}

    # 1. Qwen2-VL LoRA V2
    print("\n--- 1. Evaluating Qwen2-VL LoRA V2 ---")
    qwen_cache_file = os.path.join(OUTPUT_DIR, "qwen_lora_raw.json")
    if os.path.exists(qwen_cache_file):
        with open(qwen_cache_file, "r", encoding="utf-8") as f:
            qwen_lora_res = json.load(f)
        print(f"Loaded cached Qwen LoRA prediction from {qwen_cache_file}")
    else:
        qwen_lora = QwenAdapter(model_path=QWEN_MODEL_DIR, adapter_path=QWEN_LORA_V2_DIR)
        qwen_lora_res = qwen_lora.predict_image(VAL_CROP_PATH, max_pixels=512*512, max_new_tokens=256)
        with open(qwen_cache_file, "w", encoding="utf-8") as f:
            json.dump(qwen_lora_res, f, ensure_ascii=False, indent=2)
        del qwen_lora
        torch.cuda.empty_cache()

    print(f"Qwen LoRA status: {qwen_lora_res.get('status')}")
    print(f"Qwen LoRA detected: {len(qwen_lora_res.get('objects', []))} objects")
    print(f"Qwen LoRA truncated: {qwen_lora_res.get('is_truncated')}")
    print(f"Qwen LoRA latency: {qwen_lora_res.get('latency_ms')} ms, VRAM: {qwen_lora_res.get('peak_vram_mb')} MB")

    # Evaluate Qwen LoRA
    pred_item_qwen = {
        "image_id": 1001,
        "status": qwen_lora_res.get("status", "success"),
        "status_detail": qwen_lora_res.get("status_detail", ""),
        "is_truncated": qwen_lora_res.get("is_truncated", False),
        "raw_output": qwen_lora_res.get("raw_output", ""),
        "latency_ms": qwen_lora_res.get("latency_ms", 0.0),
        "peak_vram_mb": qwen_lora_res.get("peak_vram_mb", 0.0),
        "objects": [
            {"label": o["label"], "bbox_xyxy": o["bbox_xyxy"], "score": o.get("score", 1.0)}
            for o in qwen_lora_res.get("objects", [])
        ]
    }
    qwen_eval = evaluate_detections(gt_dataset, {"predictions": [pred_item_qwen]}, iou_threshold=0.50)
    qwen_coco = compute_official_coco_eval(gt_dataset, {"predictions": [pred_item_qwen]})

    results["qwen_lora_v2"] = {
        "model_name": "Qwen2-VL-2B (LoRA V2)",
        "inference": qwen_lora_res,
        "metrics_eval": qwen_eval,
        "coco_eval": qwen_coco
    }

    # 2. DeepSeek-VL-1.3B
    print("\n--- 2. Evaluating DeepSeek-VL-1.3B ---")
    ds_cache_file = os.path.join(OUTPUT_DIR, "deepseek_raw.json")
    if os.path.exists(ds_cache_file):
        with open(ds_cache_file, "r", encoding="utf-8") as f:
            ds_res = json.load(f)
        print(f"Loaded cached DeepSeek prediction from {ds_cache_file}")
    else:
        ds_adapter = DeepSeekAdapter(model_path=DEEPSEEK_MODEL_DIR)
        ds_res = ds_adapter.predict_image(VAL_CROP_PATH)
        with open(ds_cache_file, "w", encoding="utf-8") as f:
            json.dump(ds_res, f, ensure_ascii=False, indent=2)
        del ds_adapter
        torch.cuda.empty_cache()

    print(f"DeepSeek status: {ds_res.get('status')}")
    print(f"DeepSeek detected: {len(ds_res.get('objects', []))} objects")
    print(f"DeepSeek detail: {ds_res.get('status_detail')}")
    print(f"DeepSeek latency: {ds_res.get('latency_ms')} ms, VRAM: {ds_res.get('peak_vram_mb')} MB")

    pred_item_ds = {
        "image_id": 1001,
        "status": ds_res.get("status", "parse_failed"),
        "status_detail": ds_res.get("status_detail", ""),
        "raw_output": ds_res.get("raw_output", ""),
        "latency_ms": ds_res.get("latency_ms", 0.0),
        "peak_vram_mb": ds_res.get("peak_vram_mb", 0.0),
        "objects": [
            {"label": o["label"], "bbox_xyxy": o["bbox_xyxy"], "score": o.get("score", 1.0)}
            for o in ds_res.get("objects", [])
        ]
    }
    ds_eval = evaluate_detections(gt_dataset, {"predictions": [pred_item_ds]}, iou_threshold=0.50)
    ds_coco = compute_official_coco_eval(gt_dataset, {"predictions": [pred_item_ds]})

    results["deepseek_base"] = {
        "model_name": "DeepSeek-VL-1.3B (Chat Base)",
        "inference": ds_res,
        "metrics_eval": ds_eval,
        "coco_eval": ds_coco
    }

    out_file = os.path.join(OUTPUT_DIR, "val_crop_real_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved real validation results to {out_file}")


if __name__ == "__main__":
    run_val_crop_inference()
