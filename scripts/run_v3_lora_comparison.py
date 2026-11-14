"""Frozen V3 base/LoRA comparison; local weights, validation only, raw tile evidence."""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

from evaluate import compute_official_coco_eval, evaluate_detections, scope_ground_truth
from infer_batch import compute_composite_tile_status
from pcb_vlm.geometry.tiling import generate_tiles, map_tile_box_to_global, nms_merge_predictions
from pcb_vlm.models.qwen_adapter import QwenAdapter
from pcb_vlm.provenance import code_identity, model_identity, sha256, source_identity, validate_partition, verify_adapter


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def diagnostic_tile(manifest_entry, size=1024, overlap=0.2):
    """Select geometrically, independently of labels and model predictions."""
    w, h = manifest_entry["width"], manifest_entry["height"]
    return min(
        generate_tiles(w, h, size, overlap),
        key=lambda t: (((t["bbox"][0] + t["bbox"][2]) / 2 - w / 2) ** 2 + ((t["bbox"][1] + t["bbox"][3]) / 2 - h / 2) ** 2, t["tile_id"]),
    )


def crop_ground_truth(gt, image_id, bbox):
    """All intersecting targets, clipped; never select by SFT budgets or classes."""
    x1, y1, x2, y2 = bbox
    anns = []
    for a in gt["annotations"]:
        if a["image_id"] != image_id:
            continue
        x, y, w, h = a["bbox"]
        left = max(x, x1)
        top = max(y, y1)
        right = min(x + w, x2)
        bottom = min(y + h, y2)
        if right <= left or bottom <= top:
            continue
        anns.append({
            **a,
            "image_id": image_id,
            "bbox": [left - x1, top - y1, right - left, bottom - top],
            "bbox_xyxy": [left - x1, top - y1, right - x1, bottom - y1],
            "area": (right - left) * (bottom - top),
            "source_annotation_id": a["id"],
        })
    return {**gt, "images": [{"id": image_id, "width": x2 - x1, "height": y2 - y1}], "annotations": anns}


def detection_rank(summary, step):
    tp, fp, fn = (summary[k] for k in ("tp", "fp", "fn"))
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return (f1, recall, -fp, -step)


def metrics(gt, preds, raw_tiles, boards):
    ids = [i["id"] for i in gt["images"]]
    primary = evaluate_detections(gt, preds, 0.5, ids)
    counts = collections.Counter(r["status"] for r in raw_tiles)
    return {
        "primary": primary,
        "diagnostic_iou025": evaluate_detections(gt, preds, 0.25, ids),
        "diagnostic_iou010": evaluate_detections(gt, preds, 0.1, ids),
        "coco": compute_official_coco_eval(gt, preds, expected_image_ids=ids),
        "per_image": {str(i): evaluate_detections(gt, preds, 0.5, [i]) for i in ids},
        "per_board": {
            b: evaluate_detections(gt, preds, 0.5, [i for i in ids if boards[i] == b])
            for b in sorted(set(boards[i] for i in ids))
        },
        "tiles": {
            "count": len(raw_tiles),
            "status_counts": dict(counts),
            "effective_parse_rate": sum(
                counts[s] for s in ("success", "partial_success", "no_objects_detected")
            ) / max(1, len(raw_tiles)),
            "truncation_count": sum(r["is_truncated"] for r in raw_tiles),
            "exact_duplicate_count": sum(
                r.get("parse_audit", {}).get("duplicate_boxes", r.get("parse_audit", {}).get("duplicate_count", 0))
                for r in raw_tiles
            ),
            "total_model_generation_s": sum(r["latency_ms"] for r in raw_tiles) / 1000,
            "peak_allocated_mib": max((r["peak_vram_mb"] for r in raw_tiles), default=0),
        },
    }


def restore_global(raw, image):
    objects = []
    for row in raw:
        x, y, _, _ = row["tile_bbox"]
        objects.extend({
            **o,
            "bbox_xyxy": map_tile_box_to_global(o["bbox_xyxy"], x, y, image["width"], image["height"]),
            "score": None,
        } for o in row["objects"])
    merged = nms_merge_predictions(objects, iou_threshold=0.5)
    return {
        "image_id": image["image_id"],
        "objects": merged,
        "status": compute_composite_tile_status(collections.Counter(r["status"] for r in raw), len(raw), len(merged)),
        "is_truncated": any(r["is_truncated"] for r in raw),
        "nms_before": len(objects),
        "nms_after": len(merged),
        "num_tiles": len(raw),
    }


def run(plan_path, mode, adapter_path=None, name="base"):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    data = Path(plan["dataset"])
    root = Path(plan["task_root"])
    output = root / "evaluations" / name / mode
    output.mkdir(parents=True, exist_ok=False)

    gt = json.loads((data / "coco_annotations.json").read_text(encoding="utf-8"))
    splits = json.loads((data / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "images_manifest.json").read_text(encoding="utf-8"))
    validate_partition(splits, manifest)

    for p, digest in plan["input_sha256"].items():
        if sha256(p) != digest:
            raise ValueError(f"Frozen input changed: {p}")
    if source_identity() != plan["source_tree_sha256"]:
        raise ValueError("Source changed after experiment freeze")
    if model_identity(plan["model"]) != plan["model_content_sha256"]:
        raise ValueError("Model identity mismatch")
    if adapter_path:
        verify_adapter(adapter_path, data / "splits.json", plan["model"])

    val_ids = splits["splits"]["val"]["image_ids"]
    images = [r for r in manifest if r["image_id"] in val_ids]
    boards = {r["image_id"]: r["physical_board_id"] for r in images}
    selected = {r["image_id"]: r for r in plan["diagnostic_tiles"]}
    if mode == "diagnostic":
        images = [r for r in images if r["image_id"] in selected]
    test_ids = set(splits["splits"]["test"]["image_ids"])
    if any(r["image_id"] in test_ids for r in images):
        raise ValueError("Refusing evaluation on test split image_ids")

    from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    torch.set_num_threads(2)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    adapter = QwenAdapter.__new__(QwenAdapter)
    adapter.device = "cuda"
    adapter.requested_model = plan["model"]
    adapter.model_path = plan["model"]
    adapter.processor = AutoProcessor.from_pretrained(plan["model"], local_files_only=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        plan["model"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": "cuda"},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    exact_reload = None
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        saved = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
        loaded = get_peft_model_state_dict(model)
        exact_reload = saved.keys() == loaded.keys() and all(
            torch.equal(v.cpu(), loaded[k].detach().cpu()) for k, v in saved.items()
        )
        if not exact_reload:
            raise ValueError("Adapter tensor readback mismatch")

    adapter.model = model
    adapter.model.eval()
    adapter.model.config.use_cache = True

    run_manifest = {
        "plan_sha256": sha256(plan_path),
        "git_commit": code_identity(),
        "source_tree_sha256": source_identity(),
        "mode": mode,
        "name": name,
        "adapter_path": adapter_path,
        "adapter_sha256": sha256(Path(adapter_path) / "adapter_model.safetensors") if adapter_path else None,
        "exact_adapter_reload": exact_reload,
        "model_content_sha256": plan["model_content_sha256"],
        "quantization": "identical NF4/double-quant/bf16, prepare_model_for_kbit_training casts retained for both",
        "score_policy": "No VLM confidence; primary area-ranked matching; COCO equal-score supplementary",
        "status": "running",
        "expected_image_ids": [r["image_id"] for r in images],
    }
    write_json(output / "run_manifest.json", run_manifest)

    predictions = []
    all_tiles = []
    diagnostic_predictions = []
    diagnostic_gt = {**gt, "images": [], "annotations": []}
    diag_raw = []
    started = time.monotonic()

    with (output / "tile_predictions.jsonl").open("x", encoding="utf-8") as log:
        for image in images:
            if sha256(image["image_path"]) != image["sha256"]:
                raise ValueError("Image changed")
            pil = Image.open(image["image_path"]).convert("RGB")
            iid = image["image_id"]
            rows = []
            image_start = time.monotonic()
            tiles = (
                [selected[iid]]
                if mode == "diagnostic"
                else generate_tiles(image["width"], image["height"], plan["tile_size"], plan["overlap_ratio"])
            )
            for tile in tiles:
                if time.monotonic() - started > plan["evaluation_wall_seconds_per_process"]:
                    raise TimeoutError("Frozen evaluation time budget exceeded")
                try:
                    result = adapter.predict_image(
                        pil.crop(tuple(tile["bbox"])),
                        prompt=plan["prompt"],
                        max_pixels=plan["max_pixels"],
                        max_new_tokens=plan["max_new_tokens"],
                        repetition_penalty=plan["repetition_penalty"],
                        coordinate_protocol="xyxy_norm1000",
                    )
                except Exception as exc:
                    failure = {"image_id": iid, "tile_bbox": tile["bbox"], "status": "inference_failed", "error": repr(exc)}
                    log.write(json.dumps(failure) + "\n")
                    log.flush()
                    run_manifest.update(status="failed", error=repr(exc))
                    write_json(output / "run_manifest.json", run_manifest)
                    raise
                result.update(
                    image_id=iid,
                    tile_id=tile["tile_id"],
                    tile_bbox=tile["bbox"],
                    physical_board_id=image["physical_board_id"],
                )
                log.write(json.dumps(result) + "\n")
                log.flush()
                rows.append(result)
                all_tiles.append(result)
                print(
                    f"{name}/{mode}: image={iid} tile={tile['tile_id']} tokens={result['generated_tokens']} "
                    f"boxes={len(result['objects'])} status={result['status']}",
                    flush=True,
                )
                if iid in selected and tile["bbox"] == selected[iid]["bbox"]:
                    diagnostic_predictions.append({"image_id": iid, **{k: result[k] for k in ("objects", "status", "is_truncated")}})
                    clipped = crop_ground_truth(gt, iid, tile["bbox"])
                    diagnostic_gt["images"].extend(clipped["images"])
                    diagnostic_gt["annotations"].extend(clipped["annotations"])
                    diag_raw.append(result)

            prediction = restore_global(rows, image) if mode == "full" else diagnostic_predictions[-1]
            prediction["wall_s"] = time.monotonic() - image_start
            predictions.append(prediction)
            write_json(output / "predictions.json", predictions)

    eval_gt = scope_ground_truth(gt, val_ids) if mode == "full" else diagnostic_gt
    results = metrics(eval_gt, predictions, all_tiles, boards)
    results["wall_inference_s"] = time.monotonic() - started
    results["nms_removed"] = sum(p.get("nms_before", 0) - p.get("nms_after", 0) for p in predictions)
    write_json(output / "metrics.json", results)
    write_json(output / "ground_truth.json", eval_gt)

    if mode == "full":
        seen_diag = {p["image_id"] for p in diagnostic_predictions}
        missing_diag = sorted(set(selected) & {r["image_id"] for r in images} - seen_diag)
        if missing_diag:
            raise ValueError(f"Selected diagnostic tiles did not match full tiling grid for image_ids={missing_diag}")
        write_json(output / "diagnostic_metrics.json", metrics(diagnostic_gt, diagnostic_predictions, diag_raw, boards))
        write_json(output / "diagnostic_predictions.json", diagnostic_predictions)
        write_json(output / "diagnostic_gt.json", diagnostic_gt)

    run_manifest.update(
        status="completed",
        inference_wall_s=time.monotonic() - started,
        actual_image_ids=[p["image_id"] for p in predictions],
        actual_tiles=len(all_tiles),
    )
    write_json(output / "run_manifest.json", run_manifest)
    print(json.dumps(results["primary"], indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--mode", choices=["full", "diagnostic"], required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--name", default="base")
    a = ap.parse_args()
    run(a.plan, a.mode, a.adapter, a.name)
