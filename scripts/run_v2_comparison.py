"""Comparative Evaluation Script: Qwen2-VL V2 vs DeepSeek-VL vs Ground Truth.
Runs comprehensive evaluations on the 5 unseen test PCB boards and generates structured metrics.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pcb_vlm.models.qwen_adapter import QwenAdapter
from pcb_vlm.models.deepseek_adapter import DeepSeekAdapter
from evaluate import evaluate_detections


TEST_BOARDS = [
    {"name": "DuetWIFI_Top", "folder": "DuetWIFI_Top", "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\DuetWIFI_Top\DuetWIFI_Top.jpg", "image_id": 11},
    {"name": "STK644", "folder": "STK644", "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\STK644\STK644.jpg", "image_id": 27},
    {"name": "XCM-307A_Top", "folder": "XCM-307A_Top", "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\XCM-307A_Top\XCM-307A_Top.jpg", "image_id": 41},
    {"name": "ML450_Top1", "folder": "ML450_Top1", "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\ML450_Top1\ML450_Top1.jpg", "image_id": 19},
    {"name": "XLVDSproSupply_Bottom1", "folder": "XLVDSproSupply_Bottom1", "path": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\XLVDSproSupply_Bottom1\XLVDSproSupply_Bottom1.jpg", "image_id": 42}
]

COCO_PATH = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\coco_annotations.json"
OUTPUT_DIR = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_v2_comparison"
QWEN_BASE = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master"
QWEN_LORA_V1 = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage14_lora_training\final_adapter"
QWEN_LORA_V2 = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage14_lora_training_v2\best_adapter"
DEEPSEEK_BASE = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master"


def evaluate_model_on_boards(model_name: str, adapter, test_boards: List[Dict[str, Any]], coco_data: Dict[str, Any]):
    print(f"\n=======================================================", flush=True)
    print(f"Evaluating Model: {model_name}", flush=True)
    print(f"=======================================================", flush=True)
    predictions = []
    latencies = []

    for b in test_boards:
        iid = b["image_id"]
        ipath = b["path"]
        name = b["name"]
        print(f"Running inference on {name} (ID {iid})...", flush=True)
        t0 = time.time()
        res = adapter.predict_image(ipath, max_new_tokens=200)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)
        num_dets = len(res.get("objects", []))
        print(f"  -> Detected {num_dets} objects in {lat:.1f}ms", flush=True)
        predictions.append({
            "image_id": iid,
            "image_path": ipath,
            "objects": res.get("objects", []),
            "latency_ms": lat
        })

    # Evaluate against COCO GT
    test_iids = set(b["image_id"] for b in test_boards)
    split_coco = {
        "images": [img for img in coco_data["images"] if img["id"] in test_iids],
        "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in test_iids],
        "categories": coco_data["categories"]
    }

    metrics_by_iou = {}
    for iou in [0.50, 0.25, 0.10]:
        m = evaluate_detections(split_coco, {"predictions": predictions}, iou_threshold=iou)
        metrics_by_iou[str(iou)] = {
            "iou": iou,
            "precision": round(m["micro"]["precision"] * 100, 2),
            "recall": round(m["micro"]["recall"] * 100, 2),
            "f1": round(m["micro"]["f1"] * 100, 2)
        }

    total_dets = sum(len(p["objects"]) for p in predictions)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "model_name": model_name,
        "total_detections": total_dets,
        "avg_latency_ms": round(avg_lat, 2),
        "metrics_by_iou": metrics_by_iou,
        "predictions": predictions
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(COCO_PATH, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    summary_file = os.path.join(OUTPUT_DIR, "comparison_summary.json")
    all_results = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    # 1. Evaluate Qwen2-VL LoRA V1 (Previous generation with ladder hallucination)
    if "qwen2_vl_lora_v1" not in all_results and os.path.exists(QWEN_LORA_V1):
        print("\nLoading Qwen2-VL LoRA V1 (Previous iteration)...", flush=True)
        from peft import PeftModel
        import torch
        qwen_v1 = QwenAdapter(model_id=QWEN_BASE, device="cuda")
        qwen_v1.model = PeftModel.from_pretrained(qwen_v1.model, QWEN_LORA_V1)
        qwen_v1.model.eval()
        res_qwen_v1 = evaluate_model_on_boards("Qwen2-VL-2B (LoRA V1)", qwen_v1, TEST_BOARDS, coco_data)
        all_results["qwen2_vl_lora_v1"] = res_qwen_v1
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        del qwen_v1
        torch.cuda.empty_cache()

    # 2. Evaluate Qwen2-VL LoRA V2 (Improved generation with negative sampling & anti-repetition)
    if "qwen2_vl_lora_v2" not in all_results and os.path.exists(QWEN_LORA_V2):
        print("\nLoading Qwen2-VL LoRA V2 (Current iteration)...", flush=True)
        from peft import PeftModel
        import torch
        qwen_v2 = QwenAdapter(model_id=QWEN_BASE, device="cuda")
        qwen_v2.model = PeftModel.from_pretrained(qwen_v2.model, QWEN_LORA_V2)
        qwen_v2.model.eval()
        res_qwen_v2 = evaluate_model_on_boards("Qwen2-VL-2B (LoRA V2)", qwen_v2, TEST_BOARDS, coco_data)
        all_results["qwen2_vl_lora_v2"] = res_qwen_v2
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        del qwen_v2
        torch.cuda.empty_cache()

    # 3. Evaluate DeepSeek-VL-1.3B (Advisor's recommendation benchmark)
    if "deepseek_vl_base" not in all_results and os.path.exists(DEEPSEEK_BASE):
        print("\nLoading DeepSeek-VL-1.3B...", flush=True)
        import torch
        deepseek = DeepSeekAdapter(model_id=DEEPSEEK_BASE, device="cuda")
        deepseek.load_model()
        res_deepseek = evaluate_model_on_boards("DeepSeek-VL-1.3B (Base)", deepseek, TEST_BOARDS, coco_data)
        all_results["deepseek_vl_base"] = res_deepseek
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        del deepseek
        torch.cuda.empty_cache()

    print(f"\nAll comparison metrics saved to {summary_file}!", flush=True)


if __name__ == "__main__":
    main()
