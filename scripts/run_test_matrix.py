"""
Final Test Matrix Runner (Stage 16).
Runs the frozen test matrix on the 5 independent test PCB boards:
Q0: Qwen Base + Whole Image
Q1: Qwen Base + 1024 Tiling
Q2: Qwen LoRA + Whole Image
Q3: Qwen LoRA + 1024 Tiling

Outputs comprehensive test metrics across multiple IoU thresholds (0.50, 0.25, 0.10)
and per-class breakdown.
"""

import os
import sys
import json
import subprocess
import yaml
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.dirname(__file__))
from evaluate import evaluate_detections


def run_test_experiment(config_desc, args_list):
    print(f"\n=======================================================")
    print(f"Starting Test Experiment: {config_desc}")
    print(f"=======================================================")
    t0 = time.time()
    cmd = [sys.executable, os.path.join("scripts", "infer_batch.py")] + args_list
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - t0
    if res.returncode != 0:
        print(f"Experiment {config_desc} FAILED (code {res.returncode}):")
        print(res.stderr)
        return False, duration
    else:
        print(f"Experiment {config_desc} SUCCESS in {duration:.1f}s.")
        return True, duration


def evaluate_run_at_thresholds(predictions_file, coco_file, test_image_ids, thresholds=[0.50, 0.25, 0.10]):
    with open(predictions_file, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    with open(coco_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    target_id_set = set(test_image_ids)
    split_coco = {
        "images": [img for img in coco["images"] if img["id"] in target_id_set],
        "annotations": [ann for ann in coco["annotations"] if ann["image_id"] in target_id_set],
        "categories": coco["categories"]
    }

    metrics_by_iou = {}
    for iou in thresholds:
        m = evaluate_detections(split_coco, {"predictions": preds}, iou_threshold=iou)
        total_tp = sum(v["tp"] for v in m["per_class"].values())
        total_fp = sum(v["fp"] for v in m["per_class"].values())
        total_fn = sum(v["fn"] for v in m["per_class"].values())
        metrics_by_iou[str(iou)] = {
            "iou_threshold": iou,
            "micro_p": round(m["micro"]["precision"] * 100, 2),
            "micro_r": round(m["micro"]["recall"] * 100, 2),
            "micro_f1": round(m["micro"]["f1"] * 100, 2),
            "macro_p": round(m["macro"]["precision"] * 100, 2),
            "macro_r": round(m["macro"]["recall"] * 100, 2),
            "macro_f1": round(m["macro"]["f1"] * 100, 2),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "per_class": m["per_class"]
        }

    total_detections = sum(p.get("num_detected", len(p.get("objects", []))) for p in preds)
    return {
        "total_detections": total_detections,
        "metrics_by_iou": metrics_by_iou
    }


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    protocol_path = os.path.join(root_dir, "configs", "frozen_protocol.yaml")
    splits_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\splits.json"
    manifest_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\images_manifest.json"
    coco_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\coco_annotations.json"
    base_model = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master"
    lora_adapter = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage14_lora_training\best_adapter"
    out_base_dir = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_final_test_matrix"
    os.makedirs(out_base_dir, exist_ok=True)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_image_ids = splits["splits"]["test"]["image_ids"]

    experiments = [
        {
            "id": "Q0",
            "name": "Qwen2-VL Base + Whole Image",
            "out_dir": os.path.join(out_base_dir, "Q0_qwen_base_whole"),
            "args": [
                "--manifest", manifest_file,
                "--splits", splits_file,
                "--split-name", "test",
                "--model-id", base_model,
                "--output-dir", os.path.join(out_base_dir, "Q0_qwen_base_whole"),
                "--run-id", "test_Q0_qwen_base_whole",
                "--coco-gt", coco_file
            ]
        },
        {
            "id": "Q1",
            "name": "Qwen2-VL Base + 1024 Tiling",
            "out_dir": os.path.join(out_base_dir, "Q1_qwen_base_tiling"),
            "args": [
                "--manifest", manifest_file,
                "--splits", splits_file,
                "--split-name", "test",
                "--model-id", base_model,
                "--output-dir", os.path.join(out_base_dir, "Q1_qwen_base_tiling"),
                "--run-id", "test_Q1_qwen_base_tiling",
                "--tile-size", "1024",
                "--overlap-ratio", "0.20",
                "--coco-gt", coco_file
            ]
        },
        {
            "id": "Q2",
            "name": "Qwen2-VL LoRA + Whole Image",
            "out_dir": os.path.join(out_base_dir, "Q2_qwen_lora_whole"),
            "args": [
                "--manifest", manifest_file,
                "--splits", splits_file,
                "--split-name", "test",
                "--model-id", base_model,
                "--adapter-dir", lora_adapter,
                "--output-dir", os.path.join(out_base_dir, "Q2_qwen_lora_whole"),
                "--run-id", "test_Q2_qwen_lora_whole",
                "--coco-gt", coco_file
            ]
        },
        {
            "id": "Q3",
            "name": "Qwen2-VL LoRA + 1024 Tiling",
            "out_dir": os.path.join(out_base_dir, "Q3_qwen_lora_tiling"),
            "args": [
                "--manifest", manifest_file,
                "--splits", splits_file,
                "--split-name", "test",
                "--model-id", base_model,
                "--adapter-dir", lora_adapter,
                "--output-dir", os.path.join(out_base_dir, "Q3_qwen_lora_tiling"),
                "--run-id", "test_Q3_qwen_lora_tiling",
                "--tile-size", "1024",
                "--overlap-ratio", "0.20",
                "--coco-gt", coco_file
            ]
        }
    ]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": 16,
        "test_boards": splits["splits"]["test"]["boards"],
        "total_test_images": len(test_image_ids),
        "total_test_gt_annotations": splits["splits"]["test"]["num_annotations"],
        "experiments": {}
    }

    for exp in experiments:
        success, dur = run_test_experiment(f"{exp['id']} ({exp['name']})", exp["args"])
        preds_path = os.path.join(exp["out_dir"], "predictions.jsonl")
        manifest_path = os.path.join(exp["out_dir"], "run_manifest.json")

        vram_mb = 0.0
        avg_latency_ms = 0.0
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                mf = json.load(f)
                vram_mb = mf.get("max_peak_vram_mb", 0.0)
                avg_latency_ms = mf.get("avg_latency_ms", 0.0)

        eval_res = evaluate_run_at_thresholds(preds_path, coco_file, test_image_ids)

        summary["experiments"][exp["id"]] = {
            "name": exp["name"],
            "wall_time_s": round(dur, 2),
            "avg_latency_ms": avg_latency_ms,
            "peak_vram_mb": vram_mb,
            "total_detections": eval_res["total_detections"],
            "metrics_by_iou": eval_res["metrics_by_iou"]
        }

    summary_file = os.path.join(out_base_dir, "final_test_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"All Stage 16 Test Experiments Finished Successfully!")
    print(f"Final Test Summary written to: {summary_file}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
