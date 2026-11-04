"""
Final Test Matrix Runner (Stage 16).
Runs the frozen test matrix on the independent test PCB boards:
Q0: Qwen Base + Whole Image (Canonical Baseline)
Q1: Qwen Base + 1024 Tiling (Canonical Baseline)
Q2: Qwen LoRA + Whole Image (When canonical LoRA provided)
Q3: Qwen LoRA + 1024 Tiling (When canonical LoRA provided)

Outputs comprehensive test metrics across multiple IoU thresholds (0.50, 0.25, 0.10),
official COCO AP (mAP_50_95, AP_50, AP_75), and per-class breakdown.
"""

import os
import sys
import json
import subprocess
import yaml
import time
import argparse
import hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from typing import List, Dict, Any, Optional

try:
    from evaluate import evaluate_detections, compute_official_coco_eval
except ImportError:
    from scripts.evaluate import evaluate_detections, compute_official_coco_eval


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
    if not os.path.exists(predictions_file):
        return {
            "total_detections": 0,
            "metrics_by_iou": {},
            "official_coco_metrics": {"error": "predictions file missing"}
        }

    with open(predictions_file, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    with open(coco_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    target_id_set = set(test_image_ids)
    split_coco = {
        "images": [img for img in coco["images"] if (img.get("id") in target_id_set or img.get("image_id") in target_id_set)],
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

    official_coco = compute_official_coco_eval(split_coco, {"predictions": preds})
    total_detections = sum(p.get("num_detected", len(p.get("objects", []))) for p in preds)

    return {
        "total_detections": total_detections,
        "metrics_by_iou": metrics_by_iou,
        "official_coco_metrics": official_coco
    }


def verify_lora_adapter_provenance(lora_adapter_dir: str, splits_sha256: str, test_boards: List[str]):
    """Strictly verify LoRA adapter provenance against canonical split hash and boards."""
    if not lora_adapter_dir:
        return
    if not os.path.isdir(lora_adapter_dir):
        raise FileNotFoundError(lora_adapter_dir)

    # Check for metadata/manifest in adapter_dir or its parent directory
    candidates = [
        os.path.join(lora_adapter_dir, "adapter_metadata.json"),
        os.path.join(lora_adapter_dir, "run_manifest.json"),
        os.path.join(os.path.dirname(lora_adapter_dir), "run_manifest.json"),
        os.path.join(os.path.dirname(lora_adapter_dir), "training_history.json")
    ]
    manifest = {}
    for c in candidates:
        if os.path.exists(c) and c.endswith(".json"):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        manifest.update(data)
            except Exception:
                pass

    adapter_split_hash = manifest.get("split_sha256") or manifest.get("config", {}).get("split_sha256")
    train_boards = manifest.get("train_boards") or manifest.get("config", {}).get("train_boards", [])

    # Path-level check for known legacy training runs
    normalized_adapter = os.path.normpath(lora_adapter_dir).lower()
    is_legacy_path = ("stage14_lora_training" in normalized_adapter and "stage14_lora_training_canonical" not in normalized_adapter)

    if is_legacy_path:
        raise ValueError(
            f"[Contamination Error] Cannot evaluate old LoRA checkpoint '{lora_adapter_dir}' on canonical test boards. "
            f"The old LoRA training set contained boards (such as Spartan3, ThingMagicM6e, Virtex6) that belong to the canonical test set. "
            f"Please train a new adapter on annotations_v2_canonical before running on canonical test boards."
        )

    if not adapter_split_hash:
        raise ValueError(
            f"[Provenance Error] LoRA adapter at '{lora_adapter_dir}' lacks verified split_sha256 metadata. "
            f"To prevent data leakage, adapters must record 'split_sha256' matching canonical {splits_sha256[:12]}."
        )

    if adapter_split_hash != splits_sha256:
        raise ValueError(
            f"[Contamination Error] LoRA adapter was trained with split hash '{adapter_split_hash[:12]}', "
            f"which does not match canonical test split hash '{splits_sha256[:12]}'."
        )

    if not train_boards:
        raise ValueError(
            f"[Provenance Error] LoRA adapter at '{lora_adapter_dir}' lacks train_boards manifest. "
            f"Cannot verify absence of board-level contamination without explicit train_boards list."
        )

    leakage = set(train_boards).intersection(set(test_boards))
    if leakage:
        raise ValueError(
            f"[Contamination Error] LoRA adapter was trained on boards that overlap with canonical test boards: {leakage}"
        )

    from pcb_vlm.provenance import load_protocol, verify_adapter, sha256
    protocol = load_protocol()
    split_path = os.path.join(protocol['dataset_canonical_dir'], 'splits.json')
    if sha256(split_path) != splits_sha256 or set(test_boards) != set(protocol['test_split']['boards']):
        raise ValueError('Provenance check does not use the frozen split')
    verify_adapter(lora_adapter_dir, split_path, protocol['models']['qwen_base'])


def main():
    parser = argparse.ArgumentParser(description="Run test matrix evaluation.")
    parser.add_argument("--splits", default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations_v2_canonical\splits.json", help="Path to splits.json")
    parser.add_argument("--manifest", default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations_v2_canonical\images_manifest.json", help="Path to images_manifest.json")
    parser.add_argument("--coco", default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations_v2_canonical\coco_annotations.json", help="Path to coco_annotations.json")
    parser.add_argument("--output-dir", default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_canonical_test_matrix", help="Base output directory")
    parser.add_argument("--lora-adapter", default=None, help="Optional newly trained canonical LoRA adapter path")
    args = parser.parse_args()

    splits_file = args.splits
    manifest_file = args.manifest
    coco_file = args.coco
    out_base_dir = args.output_dir
    base_model = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master"
    lora_adapter = args.lora_adapter
    from pcb_vlm.provenance import load_protocol, verify_adapter
    protocol = load_protocol()
    if hashlib.sha256(open(splits_file, "rb").read()).hexdigest() != protocol["split_sha256"]:
        raise ValueError("Matrix split differs from frozen protocol")
    if lora_adapter:
        verify_adapter(lora_adapter, splits_file, base_model)
    os.makedirs(out_base_dir, exist_ok=True)

    with open(splits_file, "rb") as f:
        splits_sha256 = hashlib.sha256(f.read()).hexdigest()

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_image_ids = splits["splits"]["test"]["image_ids"]
    test_boards = splits["splits"]["test"]["boards"]
    print(f"Loaded canonical test split (SHA-256: {splits_sha256[:12]}...)")
    print(f"Test Boards ({len(test_boards)}): {test_boards}")
    print(f"Test Images ({len(test_image_ids)}): {test_image_ids}")

    # Strict Data Contamination Verification
    if lora_adapter:
        verify_lora_adapter_provenance(lora_adapter, splits_sha256, test_boards)

    experiments = [
        {
            "id": "Q0",
            "name": "Qwen2-VL Base + Whole Image (Canonical Baseline)",
            "out_dir": os.path.join(out_base_dir, "Q0_qwen_base_whole"),
            "args": [
                "--manifest", manifest_file,
                "--splits", splits_file,
                "--split-name", "test",
                "--model-id", base_model,
                "--output-dir", os.path.join(out_base_dir, "Q0_qwen_base_whole"),
                "--run-id", "canonical_test_Q0_qwen_base_whole",
                "--coco-gt", coco_file
            ]
        },
        {
            "id": "Q1",
            "name": "Qwen2-VL Base + 1024 Tiling (Canonical Baseline)",
            "out_dir": os.path.join(out_base_dir, "Q1_qwen_base_tiling"),
            "args": [
                "--manifest", manifest_file,
                "--splits", splits_file,
                "--split-name", "test",
                "--model-id", base_model,
                "--output-dir", os.path.join(out_base_dir, "Q1_qwen_base_tiling"),
                "--run-id", "canonical_test_Q1_qwen_base_tiling",
                "--tile-size", "1024",
                "--overlap-ratio", "0.20",
                "--coco-gt", coco_file
            ]
        }
    ]

    if lora_adapter and os.path.exists(lora_adapter):
        experiments.extend([
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
                    "--run-id", "canonical_test_Q2_qwen_lora_whole",
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
                    "--run-id", "canonical_test_Q3_qwen_lora_tiling",
                    "--tile-size", "1024",
                    "--overlap-ratio", "0.20",
                    "--coco-gt", coco_file
                ]
            }
        ])
    else:
        print("[Notice] Running clean base model benchmarks (Q0, Q1) on canonical test split. Provide a canonical --lora-adapter to execute Q2/Q3.")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": 16,
        "splits_sha256": splits_sha256,
        "test_boards": splits["splits"]["test"]["boards"],
        "total_test_images": len(test_image_ids),
        "total_test_gt_annotations": splits["splits"]["test"]["num_annotations"],
        "experiments": {}
    }

    for experiment in experiments:
        experiment["args"].extend(["--protocol", os.path.join(os.path.dirname(__file__), "..", "configs", "frozen_protocol.yaml")])
    all_experiments_successful = True
    for exp in experiments:
        success, dur = run_test_experiment(f"{exp['id']} ({exp['name']})", exp["args"])
        if not success:
            all_experiments_successful = False
            summary["experiments"][exp["id"]] = {
                "name": exp["name"],
                "status": "failed",
                "wall_time_s": round(dur, 2),
                "error": "Inference subprocess failed. See console output for traceback."
            }
            print(f"Skipping evaluation aggregation for failed experiment {exp['id']}.")
            continue

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
        if eval_res['official_coco_metrics'].get('error'):
            all_experiments_successful = False

        summary["experiments"][exp["id"]] = {
            "name": exp["name"],
            "status": "evaluation_failed" if eval_res["official_coco_metrics"].get("error") else "success",
            "wall_time_s": round(dur, 2),
            "avg_latency_ms": avg_latency_ms,
            "peak_vram_mb": vram_mb,
            "total_detections": eval_res["total_detections"],
            "metrics_by_iou": eval_res["metrics_by_iou"],
            "official_coco_metrics": eval_res["official_coco_metrics"]
        }

    summary_file = os.path.join(out_base_dir, "final_test_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    if all_experiments_successful:
        print(f"All Stage 16 Test Experiments Finished Successfully!")
    else:
        print(f"Test Matrix Completed with Errors. Some experiments failed.")
    summary["execution_status"] = "completed" if all_experiments_successful else "partial_failed"
    with open(summary_file, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Summary written to: {summary_file}")
    if not all_experiments_successful:
        sys.exit(2)
    print(f"=======================================================")


if __name__ == "__main__":
    main()
