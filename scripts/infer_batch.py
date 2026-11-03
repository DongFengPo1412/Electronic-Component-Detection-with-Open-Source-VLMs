"""Batch inference script with caching, resume, hardware tracking, and automated evaluation."""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pcb_vlm.models.qwen_adapter import QwenAdapter
from pcb_vlm.models.deepseek_adapter import DeepSeekAdapter
from pcb_vlm.geometry.tiling import generate_tiles, map_tile_box_to_global, nms_merge_predictions
from evaluate import evaluate_detections, compute_official_coco_eval


def get_adapter_hash(adapter_dir: Optional[str]) -> str:
    """Computes a true SHA-256 hash of the actual file contents of adapter weights and configs."""
    if not adapter_dir or not os.path.exists(adapter_dir):
        return "no_adapter"
    hasher = hashlib.sha256()
    for root, _, files in os.walk(adapter_dir):
        for fname in sorted(files):
            if fname.endswith((".json", ".safetensors", ".bin", ".pt")):
                fpath = os.path.join(root, fname)
                hasher.update(fname.encode("utf-8"))
                with open(fpath, "rb") as fp:
                    while chunk := fp.read(65536):
                        hasher.update(chunk)
    return hasher.hexdigest()[:16]


POSTPROCESS_VERSION = "2.2_canonical_xfirst"


def compute_prediction_cache_key(
    image_sha256: str,
    model_id: str,
    prompt: str,
    tile_size: Optional[int],
    overlap: Optional[float],
    adapter_dir: Optional[str] = None,
    max_pixels: int = 768 * 28 * 28,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.25
) -> str:
    hasher = hashlib.sha256()
    adapter_hash = get_adapter_hash(adapter_dir)
    content = (
        f"{image_sha256}_{model_id}_{prompt}_{tile_size}_{overlap}_"
        f"{adapter_hash}_{max_pixels}_{max_new_tokens}_{repetition_penalty}_{POSTPROCESS_VERSION}"
    )
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


def compute_composite_tile_status(tile_status_counts: Dict[str, int], total_tiles: int, num_merged_objects: int) -> str:
    """Compute overall image detection status from individual tile statuses."""
    failed_tiles = (
        tile_status_counts.get("parse_failed", 0) +
        tile_status_counts.get("failed", 0) +
        tile_status_counts.get("inference_failed", 0)
    )
    filtered_tiles = tile_status_counts.get("parse_filtered", 0)
    partial_tiles = tile_status_counts.get("partial_success", 0)
    no_obj_tiles = tile_status_counts.get("no_objects_detected", 0)

    if num_merged_objects > 0:
        if failed_tiles > 0 or partial_tiles > 0 or filtered_tiles > 0:
            return "partial_success"
        else:
            return "success"
    else:
        if failed_tiles == total_tiles:
            return "parse_failed"
        elif filtered_tiles == total_tiles:
            return "parse_filtered"
        elif no_obj_tiles == total_tiles:
            return "no_objects_detected"
        elif failed_tiles > 0:
            return "partial_failed"
        elif filtered_tiles > 0:
            return "parse_filtered"
        elif partial_tiles > 0:
            return "partial_success"
        else:
            return "no_objects_detected"


def run_batch_inference(
    manifest_path: str,
    splits_path: str,
    split_name: str,
    model_id: str,
    output_dir: str,
    run_id: str,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    tile_size: Optional[int] = None,
    overlap_ratio: float = 0.20,
    eval_coco_path: Optional[str] = None,
    max_pixels: int = 768 * 28 * 28,
    adapter_dir: Optional[str] = None,
    model_family: str = "qwen",
    image_ids: Optional[List[int]] = None,
    limit: Optional[int] = None,
    no_cache: bool = False,
    prompt_text: Optional[str] = None,
    prompt_type: str = "original"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, "predictions.jsonl")

    # Load splits and manifest
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_image_ids = set(splits["splits"][split_name]["image_ids"])
    if image_ids:
        target_image_ids = target_image_ids.intersection(set(image_ids))
    selected_ids = sorted(list(target_image_ids))
    if limit and limit > 0:
        selected_ids = selected_ids[:limit]
        target_image_ids = set(selected_ids)

    target_images = [img for img in manifest if img["image_id"] in target_image_ids]

    print(f"Loaded split '{split_name}': {len(target_images)} target images (selected IDs: {selected_ids}).")

    # Load existing cached predictions if resuming
    cached_predictions = {}
    if not no_cache and os.path.exists(predictions_file):
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cached_predictions[item.get("cache_key")] = item
        print(f"Loaded {len(cached_predictions)} cached predictions from {predictions_file}.")
    elif no_cache and os.path.exists(predictions_file):
        # Truncate existing file if force fresh
        open(predictions_file, "w", encoding="utf-8").close()
        print("Force fresh execution: cleared previous predictions file.")

    # Load model adapter
    if model_family.lower() == "deepseek":
        adapter = DeepSeekAdapter(
            model_id=model_id,
            device=device,
            cache_dir=cache_dir
        )
        adapter.load_model()
        if adapter_dir and os.path.exists(adapter_dir):
            print(f"Loading LoRA adapter for DeepSeek from {adapter_dir}...")
            from peft import PeftModel
            adapter.model.language_model = PeftModel.from_pretrained(adapter.model.language_model, adapter_dir)
            adapter.model.eval()
            print("DeepSeek LoRA adapter successfully mounted.")
    else:
        adapter = QwenAdapter(
            model_id=model_id,
            device=device,
            cache_dir=cache_dir
        )
        # Mount LoRA adapter if provided
        if adapter_dir and os.path.exists(adapter_dir):
            print(f"Loading LoRA adapter from {adapter_dir}...")
            from peft import PeftModel
            adapter.model = PeftModel.from_pretrained(adapter.model, adapter_dir)
            adapter.model.eval()
            print("LoRA adapter successfully mounted.")

    if prompt_text:
        prompt = prompt_text
    elif prompt_type == "improved":
        prompt = (
            "Detect and locate all electronic components on this PCB image. "
            "Target classes: capacitor, resistor, ic, connector, led, transistor, diode, inductor, button, switch, clock. "
            "Output format: strictly use normalized bounding box tokens in [0, 1000] range: label <|box_start|>(xmin,ymin),(xmax,ymax)<|box_end|>. "
            "Rules:\n"
            "1. Do NOT output silkscreen text, component reference names (e.g. D1, C1, R1), or text descriptions.\n"
            "2. Every detected object MUST have four coordinates (xmin,ymin),(xmax,ymax). Do NOT output center points.\n"
            "3. Do NOT repeat different category labels at the exact same bounding box coordinates.\n"
            "4. If no target components are visible, output 'None'."
        )
    else:
        prompt = adapter.build_prompt()

    print(f"Prompt strategy: '{prompt_type}'. Prompt length: {len(prompt)} chars.")

    results = []
    total_latency_ms = 0.0
    peak_vram_records = []

    for idx, img_entry in enumerate(target_images):
        iid = img_entry["image_id"]
        ipath = img_entry["image_path"]
        isha = img_entry["sha256"]
        w = img_entry["width"]
        h = img_entry["height"]

        cache_key = compute_prediction_cache_key(isha, model_id, prompt, tile_size, overlap_ratio, adapter_dir, max_pixels)

        if cache_key in cached_predictions:
            print(f"[{idx+1}/{len(target_images)}] Image {iid} ({img_entry['folder']}): CACHED")
            pred_item = cached_predictions[cache_key]
            results.append(pred_item)
            total_latency_ms += pred_item.get("latency_ms", 0.0)
            peak_vram_records.append(pred_item.get("peak_vram_mb", 0.0))
            continue

        print(f"[{idx+1}/{len(target_images)}] Inferring on image {iid} ({img_entry['folder']})...")
        t0 = time.time()

        try:
            if tile_size is not None and tile_size > 0:
                # Tiling / SAHI inference
                from PIL import Image
                pil_img = Image.open(ipath).convert("RGB")
                tiles = generate_tiles(w, h, tile_size=tile_size, overlap_ratio=overlap_ratio)
                patch_predictions = []
                tile_raw_outputs = []
                tile_status_counts = {"success": 0, "no_objects_detected": 0, "parse_failed": 0, "partial_success": 0}

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                for t_idx, t in enumerate(tiles):
                    bx1, by1, bx2, by2 = t["bbox"]
                    crop_tile = pil_img.crop((bx1, by1, bx2, by2))

                    # Direct in-memory PIL Image inference (eliminates JPEG compression artifacts)
                    tile_res = adapter.predict_image(crop_tile, prompt=prompt, max_pixels=max_pixels)

                    t_status = tile_res.get("status", "unknown")
                    t_detail = tile_res.get("status_detail", "")
                    t_truncated = tile_res.get("is_truncated", False)
                    if t_status not in tile_status_counts:
                        tile_status_counts[t_status] = 0
                    tile_status_counts[t_status] += 1

                    tile_raw_outputs.append({
                        "tile_idx": t_idx,
                        "bbox": t["bbox"],
                        "status": t_status,
                        "status_detail": t_detail,
                        "is_truncated": t_truncated,
                        "generated_tokens": tile_res.get("generated_tokens"),
                        "token_budget": tile_res.get("token_budget"),
                        "raw_output": tile_res.get("raw_output", ""),
                        "num_detected": len(tile_res.get("objects", []))
                    })

                    for obj in tile_res.get("objects", []):
                        g_box = map_tile_box_to_global(obj["bbox_xyxy"], t["offset_x"], t["offset_y"], w, h)
                        patch_predictions.append({
                            "label": obj["label"],
                            "bbox_xyxy": g_box,
                            "score": obj.get("score")
                        })

                merged_objects = nms_merge_predictions(patch_predictions, iou_threshold=0.50)
                latency_ms = (time.time() - t0) * 1000
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                total_t = len(tiles)
                overall_status = compute_composite_tile_status(tile_status_counts, total_t, len(merged_objects))
                clean_success_tiles = tile_status_counts.get("success", 0)
                no_obj_tiles = tile_status_counts.get("no_objects_detected", 0)
                filtered_tiles = tile_status_counts.get("parse_filtered", 0)
                failed_tiles = (
                    tile_status_counts.get("parse_failed", 0) +
                    tile_status_counts.get("failed", 0) +
                    tile_status_counts.get("inference_failed", 0)
                )
                partial_tiles = tile_status_counts.get("partial_success", 0)

                status_summary = (
                    f"Aggregated {total_t} tiles ({clean_success_tiles} success, {no_obj_tiles} no_objects, "
                    f"{filtered_tiles} filtered, {failed_tiles} failed, {partial_tiles} partial/truncated); "
                    f"{len(patch_predictions)} raw boxes merged to {len(merged_objects)} via NMS"
                )

                any_tile_truncated = any(t.get("is_truncated", False) for t in tile_raw_outputs)

                pred_item = {
                    "cache_key": cache_key,
                    "image_id": iid,
                    "folder": img_entry["folder"],
                    "width": w,
                    "height": h,
                    "tile_size": tile_size,
                    "overlap_ratio": overlap_ratio,
                    "num_tiles": total_t,
                    "tile_status_counts": tile_status_counts,
                    "objects": merged_objects,
                    "num_detected": len(merged_objects),
                    "status": overall_status,
                    "status_detail": status_summary,
                    "raw_output_summary": status_summary,
                    "tile_raw_outputs": tile_raw_outputs,
                    "is_truncated": any_tile_truncated,
                    "requested_model": getattr(adapter, "requested_model", model_id),
                    "actual_model_path": getattr(adapter, "model_path", model_id),
                    "latency_ms": round(latency_ms, 2),
                    "peak_vram_mb": round(peak_vram_mb, 2)
                }
            else:
                # Whole image inference
                pred_item_raw = adapter.predict_image(ipath, prompt=prompt, max_pixels=max_pixels)
                pred_item = {
                    "cache_key": cache_key,
                    "image_id": iid,
                    "folder": img_entry["folder"],
                    "width": w,
                    "height": h,
                    "tile_size": None,
                    "objects": pred_item_raw["objects"],
                    "num_detected": pred_item_raw["num_detected"],
                    "status": pred_item_raw.get("status", "success" if pred_item_raw["num_detected"] > 0 else "no_objects_detected"),
                    "status_detail": pred_item_raw.get("status_detail", ""),
                    "raw_output": pred_item_raw.get("raw_output", ""),
                    "is_truncated": pred_item_raw.get("is_truncated", False),
                    "is_syntax_truncated": pred_item_raw.get("is_syntax_truncated", False),
                    "generated_tokens": pred_item_raw.get("generated_tokens"),
                    "token_budget": pred_item_raw.get("token_budget"),
                    "finish_reason": pred_item_raw.get("finish_reason"),
                    "truncation_reason": pred_item_raw.get("truncation_reason"),
                    "requested_model": getattr(adapter, "requested_model", model_id),
                    "actual_model_path": getattr(adapter, "model_path", model_id),
                    "latency_ms": pred_item_raw["latency_ms"],
                    "peak_vram_mb": pred_item_raw["peak_vram_mb"]
                }
        except Exception as e:
            print(f"[Inference Error on Image {iid}]: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pred_item = {
                "cache_key": cache_key,
                "image_id": iid,
                "folder": img_entry.get("folder", "unknown"),
                "width": w,
                "height": h,
                "tile_size": tile_size,
                "objects": [],
                "num_detected": 0,
                "status": "inference_failed",
                "status_detail": f"Inference execution failed: {str(e)}",
                "is_truncated": False,
                "requested_model": getattr(adapter, "requested_model", model_id),
                "actual_model_path": getattr(adapter, "model_path", model_id),
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "peak_vram_mb": 0.0
            }

        results.append(pred_item)
        total_latency_ms += pred_item["latency_ms"]
        peak_vram_records.append(pred_item["peak_vram_mb"])

        # Append to jsonl immediately to guarantee persistence on partial runs
        with open(predictions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pred_item, ensure_ascii=False) + "\n")

    manifest_record = {
        "run_id": run_id,
        "split_name": split_name,
        "model_id": model_id,
        "adapter_dir": adapter_dir,
        "tile_size": tile_size,
        "overlap_ratio": overlap_ratio,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "total_images": len(target_images),
        "avg_latency_ms": round(total_latency_ms / len(target_images), 2) if target_images else 0.0,
        "max_peak_vram_mb": round(max(peak_vram_records), 2) if peak_vram_records else 0.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predictions_file": os.path.abspath(predictions_file)
    }

    manifest_out = os.path.join(output_dir, "run_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_record, f, indent=2, ensure_ascii=False)

    print(f"\nRun manifest written to: {manifest_out}")

    # Automated Evaluation if COCO GT is provided
    if eval_coco_path and os.path.exists(eval_coco_path):
        print(f"\nComputing evaluation metrics against COCO GT for split '{split_name}'...")
        with open(eval_coco_path, "r", encoding="utf-8") as f:
            full_coco = json.load(f)
        target_id_set = set(target_image_ids)
        split_coco = {
            "images": [img for img in full_coco.get("images", []) if img.get("id") in target_id_set or img.get("image_id") in target_id_set],
            "annotations": [ann for ann in full_coco.get("annotations", []) if ann.get("image_id") in target_id_set],
            "categories": full_coco.get("categories", [])
        }
        eval_metrics = evaluate_detections(split_coco, {"predictions": results})
        official_coco = compute_official_coco_eval(split_coco, {"predictions": results})
        eval_metrics["official_coco"] = official_coco

        # Compute strict per-image breakdown
        per_image_metrics = {}
        for iid in sorted(list(target_id_set)):
            sub_gt = {
                "images": [img for img in split_coco["images"] if (img.get("id") == iid or img.get("image_id") == iid)],
                "annotations": [ann for ann in split_coco["annotations"] if ann.get("image_id") == iid],
                "categories": split_coco.get("categories", [])
            }
            sub_preds = {"predictions": [p for p in results if p.get("image_id") == iid]}
            sub_eval = evaluate_detections(sub_gt, sub_preds)
            sub_coco = compute_official_coco_eval(sub_gt, sub_preds)
            pred_item = sub_preds["predictions"][0] if sub_preds["predictions"] else {}
            per_image_metrics[str(iid)] = {
                "image_id": iid,
                "status": pred_item.get("status", "unknown"),
                "total_gt": len(sub_gt["annotations"]),
                "num_detected": len(pred_item.get("objects", [])),
                "tp": sub_eval.get("tp", 0),
                "fp": sub_eval.get("fp", 0),
                "fn": sub_eval.get("fn", 0),
                "precision": round(sub_eval.get("micro", {}).get("precision", 0.0), 4),
                "recall": round(sub_eval.get("micro", {}).get("recall", 0.0), 4),
                "f1": round(sub_eval.get("micro", {}).get("f1", 0.0), 4),
                "ap50": round(sub_coco.get("AP_50", 0.0), 4),
                "tps_by_class": {c: v["tp"] for c, v in sub_eval.get("per_class", {}).items() if v.get("tp", 0) > 0}
            }
        eval_metrics["per_image"] = per_image_metrics

        eval_out = os.path.join(output_dir, "evaluation_metrics.json")
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
        manifest_record["evaluation"] = eval_metrics
        print(f"Evaluation metrics saved to: {eval_out}")
        print(f"Official COCO AP50: {official_coco.get('AP_50', 0.0)}, mAP50:95: {official_coco.get('mAP_50_95', 0.0)}")

    return manifest_record


def main():
    parser = argparse.ArgumentParser(description="Run batch inference with caching and evaluation.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to images_manifest.json.")
    parser.add_argument("--splits", type=str, required=True, help="Path to splits.json.")
    parser.add_argument("--split-name", type=str, default="val", help="Split name to run (val, test, train).")
    parser.add_argument("--image-ids", type=str, default=None, help="Comma-separated image IDs to run.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process.")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh inference without reading cache.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID.")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Path to fine-tuned LoRA adapter.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for predictions.")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Model cache directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size for patch inference.")
    parser.add_argument("--overlap-ratio", type=float, default=0.20, help="Tile overlap ratio.")
    parser.add_argument("--coco-gt", type=str, default=None, help="Path to coco_annotations.json for evaluation.")
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28, help="Max pixels for vision encoder.")
    parser.add_argument("--model-family", type=str, default="qwen", choices=["qwen", "deepseek"], help="Model family to run.")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt string.")
    parser.add_argument("--prompt-type", type=str, default="original", choices=["original", "improved"], help="Prompt strategy preset.")
    args = parser.parse_args()

    parsed_image_ids = [int(x.strip()) for x in args.image_ids.split(",") if x.strip()] if args.image_ids else None

    run_batch_inference(
        manifest_path=args.manifest,
        splits_path=args.splits,
        split_name=args.split_name,
        model_id=args.model_id,
        output_dir=args.output_dir,
        run_id=args.run_id,
        cache_dir=args.cache_dir,
        device=args.device,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        eval_coco_path=args.coco_gt,
        max_pixels=args.max_pixels,
        adapter_dir=args.adapter_dir,
        model_family=args.model_family,
        image_ids=parsed_image_ids,
        limit=args.limit,
        no_cache=args.no_cache,
        prompt_text=args.prompt,
        prompt_type=args.prompt_type
    )


if __name__ == "__main__":
    main()
