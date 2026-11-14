"""Same YOLO checkpoint: full image vs tiled inference, native confidence and unified V3 evaluator."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parent)]

from evaluate import compute_official_coco_eval, evaluate_detections, is_valid_box, scope_ground_truth
from pcb_vlm.data.yolo_v3 import read, write
from pcb_vlm.geometry.tiling import generate_tiles, map_tile_box_to_global, nms_merge_predictions
from pcb_vlm.provenance import sha256, source_identity, validate_partition


def yolo_objects(boxes, classes, scores, names, width, height):
    """Convert raw YOLO predictions into canonical pixel xyxy dicts with strict validation."""
    out = []
    for box, cls, score in zip(boxes, classes, scores):
        if not is_valid_box(box, width, height) or int(cls) != cls or not 0 <= int(cls) < len(names) or not 0 <= score <= 1:
            raise ValueError("Invalid YOLO box/class/confidence")
        out.append({"label": names[int(cls)], "bbox_xyxy": list(box), "score": float(score)})
    return out


def confidence_subset(predictions, threshold=0.25):
    return [{**r, "objects": [o for o in r["objects"] if o["score"] >= threshold]} for r in predictions]


def evaluate_prediction_files(gt, preds, boards, threshold=0.25):
    ids = [i["id"] for i in gt["images"]]
    fixed = confidence_subset(preds, threshold)
    return {
        "primary_conf": threshold,
        "primary": evaluate_detections(gt, fixed, 0.5, ids),
        "ap_collection_conf": 0.001,
        "coco": compute_official_coco_eval(gt, preds, expected_image_ids=ids),
        "per_image": {str(i): evaluate_detections(gt, fixed, 0.5, [i]) for i in ids},
        "per_board": {
            b: evaluate_detections(gt, fixed, 0.5, [i for i in ids if boards[i] == b])
            for b in sorted(set(boards.values()))
        },
    }


def run(plan_path, weights, mode, split="val"):
    """Run whole-image or tiled YOLO inference against a frozen V3 evaluation plan."""
    plan = read(plan_path)
    root = Path(plan["task_root"])
    from train_yolo_v3 import configure_runtime

    configure_runtime(root)
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    for f, d in plan["input_sha256"].items():
        if sha256(f) != d:
            raise ValueError(f"Input changed: {f}")
    if source_identity() != plan["source_tree_sha256"]:
        raise ValueError("Frozen code changed")

    output_name = f"inference_{mode}" if split == "val" else f"inference_{split}_{mode}"
    output = root / output_name
    output.mkdir(exist_ok=False)

    dataset_dir = Path(plan["canonical_dataset"])
    gt = read(dataset_dir / "coco_annotations.json")
    splits = read(dataset_dir / "splits.json")
    manifest = read(dataset_dir / "images_manifest.json")
    validate_partition(splits, manifest)

    ids = splits["splits"][split]["image_ids"]
    gt = scope_ground_truth(gt, ids)
    if {im["id"] for im in gt["images"]} != set(ids):
        raise ValueError(f"Scoped GT image IDs differ from split '{split}'")

    images = [r for r in manifest if r["image_id"] in ids]
    boards = {r["image_id"]: r["physical_board_id"] for r in images}
    names = read(root / "yolo_data/provenance.json")["names"]

    model = YOLO(weights)
    if list(model.names.values()) != names:
        raise ValueError("Model class names do not match dataset provenance")

    conf = plan["inference"]
    imgsz = int(conf.get("imgsz", 640))
    tile_size = int(conf.get("tile_size", imgsz))
    overlap_ratio = float(conf.get("overlap_ratio", 0.20))
    iou_nms = float(conf.get("iou_nms", 0.50))

    params = dict(
        imgsz=imgsz,
        conf=conf["ap_collection_conf"],
        iou=iou_nms,
        max_det=conf["max_det_per_input"],
        device=0,
        half=False,
        rect=False,
        augment=False,
        agnostic_nms=False,
        verbose=False,
    )
    torch.set_num_threads(2)

    warm = time.perf_counter()
    model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), **params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warm = time.perf_counter() - warm

    shapes = []
    handle = model.predictor.model.model.register_forward_pre_hook(
        lambda module, args: shapes.append(list(args[0].shape))
    )
    predictions = []
    all_tiles = []
    started = time.perf_counter()

    try:
        with (output / "tile_predictions.jsonl").open("x", encoding="utf-8") as log:
            for im in images:
                if sha256(im["image_path"]) != im["sha256"]:
                    raise ValueError(f"Image hash changed: {im['image_path']}")
                pil = Image.open(im["image_path"]).convert("RGB")
                tiles = (
                    generate_tiles(im["width"], im["height"], tile_size, overlap_ratio)
                    if mode == "tiled"
                    else [{"tile_id": 0, "bbox": [0, 0, im["width"], im["height"]], "offset_x": 0, "offset_y": 0}]
                )
                objects = []
                begin = time.perf_counter()
                for t in tiles:
                    crop = pil.crop(t["bbox"])
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    tic = time.perf_counter()
                    res = model.predict(crop, **params)[0]
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tic

                    rows = yolo_objects(
                        res.boxes.xyxy.cpu().tolist(),
                        res.boxes.cls.cpu().tolist(),
                        res.boxes.conf.cpu().tolist(),
                        names,
                        crop.width,
                        crop.height,
                    )
                    peak_mib = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
                    raw = {
                        "image_id": im["image_id"],
                        "physical_board_id": im["physical_board_id"],
                        "tile_id": t["tile_id"],
                        "tile_bbox": t["bbox"],
                        "actual_input_shape": shapes[-1] if shapes else None,
                        "objects": rows,
                        "wall_s": elapsed,
                        "native_speed_ms": res.speed,
                        "peak_allocated_mib": peak_mib,
                        "cap_hit": len(rows) >= conf["max_det_per_input"],
                    }
                    all_tiles.append(raw)
                    log.write(json.dumps(raw) + "\n")
                    log.flush()
                    objects.extend({
                        **o,
                        "bbox_xyxy": map_tile_box_to_global(
                            o["bbox_xyxy"], t["offset_x"], t["offset_y"], im["width"], im["height"]
                        ),
                    } for o in rows)

                merged = nms_merge_predictions(objects, iou_nms)
                item = {
                    "image_id": im["image_id"],
                    "objects": merged,
                    "status": "success" if merged else "no_objects_detected",
                    "is_truncated": False,
                    "num_tiles": len(tiles),
                    "nms_before": len(objects),
                    "nms_after": len(merged),
                    "wall_s": time.perf_counter() - begin,
                }
                predictions.append(item)
                write(output / "predictions.json", predictions)
                print(mode, "image", im["image_id"], "tiles", len(tiles), "before", len(objects), "after", len(merged), flush=True)
    finally:
        handle.remove()

    eval_metrics = evaluate_prediction_files(gt, predictions, boards, conf["primary_conf"])
    eval_metrics["runtime"] = {
        "total_wall_s": time.perf_counter() - started,
        "warmup_s": warm,
        "calls": len(all_tiles),
        "nms_removed": sum(p["nms_before"] - p["nms_after"] for p in predictions),
        "max_det_per_input": conf["max_det_per_input"],
        "cap_hits": sum(r["cap_hit"] for r in all_tiles),
        "peak_allocated_mib": max((r["peak_allocated_mib"] for r in all_tiles), default=0.0),
    }
    write(output / "metrics.json", eval_metrics)
    write(output / "ground_truth.json", gt)
    write(output / "run_manifest.json", {
        "completed": True,
        "split": split,
        "plan_sha256": sha256(plan_path),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "mode": mode,
        "params": params,
        "gt_sha256": sha256(dataset_dir / "coco_annotations.json"),
        "eval_ids": ids,
        "source_tree_sha256": source_identity(),
        "confidence_policy": "Primary P/R/F1 at .25; AP from .001 native real scores; no 1.0 score replacement.",
    })
    print(json.dumps(eval_metrics["primary"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--mode", choices=["whole", "tiled"], required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    a = p.parse_args()
    run(a.plan, a.weights, a.mode, split=a.split)
