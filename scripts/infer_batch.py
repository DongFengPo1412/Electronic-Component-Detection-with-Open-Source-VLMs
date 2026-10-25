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
from pcb_vlm.geometry.tiling import generate_tiles, map_tile_box_to_global, nms_merge_predictions
from evaluate import evaluate_detections


def compute_prediction_cache_key(
    image_sha256: str,
    model_id: str,
    prompt: str,
    tile_size: Optional[int],
    overlap: Optional[float]
) -> str:
    hasher = hashlib.sha256()
    content = f"{image_sha256}_{model_id}_{prompt}_{tile_size}_{overlap}"
    hasher.update(content.encode("utf-8"))
    return hasher.hexdigest()


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
    eval_coco_path: Optional[str] = None
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, "predictions.jsonl")

    # Load splits and manifest
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_image_ids = set(splits["splits"][split_name]["image_ids"])
    target_images = [img for img in manifest if img["image_id"] in target_image_ids]

    print(f"Loaded split '{split_name}': {len(target_images)} target images.")

    # Load existing cached predictions if resuming
    cached_predictions = {}
    if os.path.exists(predictions_file):
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cached_predictions[item.get("cache_key")] = item
        print(f"Loaded {len(cached_predictions)} cached predictions from {predictions_file}.")

    # Load model adapter
    adapter = QwenAdapter(
        model_id=model_id,
        device=device,
        cache_dir=cache_dir
    )
    prompt = adapter.build_prompt()

    results = []
    total_latency_ms = 0.0
    peak_vram_records = []

    for idx, img_entry in enumerate(target_images):
        iid = img_entry["image_id"]
        ipath = img_entry["image_path"]
        isha = img_entry["sha256"]
        w = img_entry["width"]
        h = img_entry["height"]

        cache_key = compute_prediction_cache_key(isha, model_id, prompt, tile_size, overlap_ratio)

        if cache_key in cached_predictions:
            print(f"[{idx+1}/{len(target_images)}] Image {iid} ({img_entry['folder']}): CACHED")
            pred_item = cached_predictions[cache_key]
            results.append(pred_item)
            continue

        print(f"[{idx+1}/{len(target_images)}] Inferring on image {iid} ({img_entry['folder']})...")
        t0 = time.time()

        if tile_size is not None and tile_size > 0:
            # Tiling / SAHI inference
            from PIL import Image
            pil_img = Image.open(ipath).convert("RGB")
            tiles = generate_tiles(w, h, tile_size=tile_size, overlap_ratio=overlap_ratio)
            patch_predictions = []

            for t in tiles:
                bx1, by1, bx2, by2 = t["bbox"]
                crop_tile = pil_img.crop((bx1, by1, bx2, by2))
                temp_tile_path = os.path.join(output_dir, f"_temp_tile_{iid}.jpg")
                crop_tile.save(temp_tile_path)

                tile_res = adapter.predict_image(temp_tile_path, prompt=prompt)
                if os.path.exists(temp_tile_path):
                    os.remove(temp_tile_path)

                for obj in tile_res["objects"]:
                    g_box = map_tile_box_to_global(obj["bbox_xyxy"], t["offset_x"], t["offset_y"], w, h)
                    patch_predictions.append({
                        "label": obj["label"],
                        "bbox_xyxy": g_box,
                        "score": obj.get("score")
                    })

            merged_objects = nms_merge_predictions(patch_predictions, iou_threshold=0.50)
            latency_ms = (time.time() - t0) * 1000
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            pred_item = {
                "cache_key": cache_key,
                "image_id": iid,
                "folder": img_entry["folder"],
                "width": w,
                "height": h,
                "tile_size": tile_size,
                "overlap_ratio": overlap_ratio,
                "num_tiles": len(tiles),
                "objects": merged_objects,
                "num_detected": len(merged_objects),
                "latency_ms": round(latency_ms, 2),
                "peak_vram_mb": round(peak_vram_mb, 2)
            }
        else:
            # Whole image inference
            pred_item_raw = adapter.predict_image(ipath, prompt=prompt)
            pred_item = {
                "cache_key": cache_key,
                "image_id": iid,
                "folder": img_entry["folder"],
                "width": w,
                "height": h,
                "tile_size": None,
                "objects": pred_item_raw["objects"],
                "num_detected": pred_item_raw["num_detected"],
                "latency_ms": pred_item_raw["latency_ms"],
                "peak_vram_mb": pred_item_raw["peak_vram_mb"]
            }

        results.append(pred_item)
        total_latency_ms += pred_item["latency_ms"]
        peak_vram_records.append(pred_item["peak_vram_mb"])

        # Append to jsonl
        with open(predictions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pred_item, ensure_ascii=False) + "\n")

    # Run Manifest
    manifest_record = {
        "run_id": run_id,
        "split_name": split_name,
        "model_id": model_id,
        "tile_size": tile_size,
        "overlap_ratio": overlap_ratio,
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
        print(f"\nComputing evaluation metrics against COCO GT...")
        with open(eval_coco_path, "r", encoding="utf-8") as f:
            coco_gt = json.load(f)
        eval_metrics = evaluate_detections(coco_gt, {"predictions": results})
        eval_out = os.path.join(output_dir, "evaluation_metrics.json")
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
        manifest_record["evaluation"] = eval_metrics
        print(f"Evaluation metrics saved to: {eval_out}")

    return manifest_record


def main():
    parser = argparse.ArgumentParser(description="Run batch inference with caching and evaluation.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to images_manifest.json.")
    parser.add_argument("--splits", type=str, required=True, help="Path to splits.json.")
    parser.add_argument("--split-name", type=str, default="val", help="Split name to run (val, test, train).")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for predictions.")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Model cache directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile size for patch inference.")
    parser.add_argument("--overlap-ratio", type=float, default=0.20, help="Tile overlap ratio.")
    parser.add_argument("--coco-gt", type=str, default=None, help="Path to coco_annotations.json for evaluation.")
    args = parser.parse_args()

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
        eval_coco_path=args.coco_gt
    )


if __name__ == "__main__":
    main()
