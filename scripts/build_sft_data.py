"""
Build Multimodal SFT (Supervised Fine-Tuning) datasets for Qwen2-VL.
Generates:
1. Ground truth conversation pairs in native Qwen2-VL grounding format.
2. High-resolution tile patches (1024x1024) and whole-image samples.
3. Strict physical board isolation (Train / Val split separation, Zero leakage).
4. Spatial coordinate sorting (top-to-bottom, left-to-right) to minimize sequence ambiguity.
"""

import os
import sys
import json
import argparse
import hashlib
from typing import List, Dict, Any, Tuple
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pcb_vlm.geometry.tiling import generate_tiles


def bbox_intersection_area(b1: List[float], b2: List[float]) -> float:
    """Calculate intersection area between [x1, y1, x2, y2] boxes."""
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    return interW * interH


def build_qwen_grounding_target(objects: List[Dict[str, Any]], img_w: int, img_h: int) -> str:
    """
    Format objects into Qwen2-VL native grounding string:
    <|object_ref_start|>label<|object_ref_end|><|box_start|>(xmin,ymin),(xmax,ymax)<|box_end|>
    Sorted spatially top-to-bottom (ymin), then left-to-right (xmin).
    """
    if not objects:
        return "None"

    # Sort objects top-to-bottom (ymin), left-to-right (xmin)
    # norm_box is [xmin, ymin, xmax, ymax]
    sorted_objs = sorted(objects, key=lambda o: (o["norm_box"][1], o["norm_box"][0]))

    tokens = []
    for obj in sorted_objs:
        label = obj["label"]
        xmin, ymin, xmax, ymax = obj["norm_box"]
        tokens.append(f"<|object_ref_start|>{label}<|object_ref_end|><|box_start|>({xmin},{ymin}),({xmax},{ymax})<|box_end|>")

    return ", ".join(tokens)


def process_split_sft(
    split_name: str,
    target_image_ids: List[int],
    manifest: List[Dict[str, Any]],
    coco_anns_by_img: Dict[int, List[Dict[str, Any]]],
    cat_id_to_name: Dict[int, str],
    target_classes: List[str],
    output_dir: str,
    crops_dir: str,
    tile_size: int = 1024,
    overlap_ratio: float = 0.20,
    min_visibility: float = 0.50
) -> List[Dict[str, Any]]:
    os.makedirs(crops_dir, exist_ok=True)
    classes_str = ", ".join(target_classes)
    prompt = f"<image>\nLocate all {classes_str} on this PCB image. Output each bounding box in normalized [0, 1000] coordinates: label <|box_start|>(xmin,ymin),(xmax,ymax)<|box_end|>"

    manifest_by_id = {item["image_id"]: item for item in manifest}
    sft_records = []
    total_components_included = 0

    for iid in target_image_ids:
        if iid not in manifest_by_id:
            raise ValueError(f"Missing image ID: {iid}")
        entry = manifest_by_id[iid]
        ipath = entry["image_path"]
        w = entry["width"]
        h = entry["height"]
        folder = entry["folder"]

        if not os.path.exists(ipath):
            raise FileNotFoundError(ipath)

        pil_img = Image.open(ipath).convert("RGB")
        raw_anns = coco_anns_by_img.get(iid, [])

        # Parse ground truth boxes into [x1, y1, x2, y2, label]
        gt_boxes = []
        for ann in raw_anns:
            cid = ann.get("category_id")
            label = cat_id_to_name.get(cid, "component")
            if target_classes and label not in target_classes:
                continue
            bx, by, bw, bh = ann["bbox"] # xywh
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            gt_boxes.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "area": max(0.0, bw * bh)
            })

        # 1. Whole image sample
        whole_objs = []
        for g in gt_boxes:
            x1, y1, x2, y2 = g["bbox"]
            xmin = int(round(max(0.0, min(1000.0, (x1 / float(w)) * 1000))))
            ymin = int(round(max(0.0, min(1000.0, (y1 / float(h)) * 1000))))
            xmax = int(round(max(0.0, min(1000.0, (x2 / float(w)) * 1000))))
            ymax = int(round(max(0.0, min(1000.0, (y2 / float(h)) * 1000))))
            if xmax > xmin and ymax > ymin:
                whole_objs.append({
                    "label": g["label"],
                    "norm_box": [xmin, ymin, xmax, ymax]
                })

        if whole_objs:
            sft_records.append({
                "id": f"{split_name}_whole_{iid}",
                "image": os.path.abspath(ipath),
                "width": w,
                "height": h,
                "board_folder": folder,
                "conversations": [
                    {"from": "user", "value": prompt},
                    {"from": "assistant", "value": build_qwen_grounding_target(whole_objs, w, h)}
                ],
                "num_objects": len(whole_objs)
            })
            total_components_included += len(whole_objs)

        # 2. Tile samples
        tiles = generate_tiles(w, h, tile_size=tile_size, overlap_ratio=overlap_ratio)
        for t_idx, t in enumerate(tiles):
            tx1, ty1, tx2, ty2 = t["bbox"]
            tile_w = tx2 - tx1
            tile_h = ty2 - ty1

            has_any_intersection = False
            tile_objs = []
            for g in gt_boxes:
                gx1, gy1, gx2, gy2 = g["bbox"]
                inter_area = bbox_intersection_area([gx1, gy1, gx2, gy2], [tx1, ty1, tx2, ty2])
                orig_area = g["area"]
                if orig_area <= 0:
                    continue

                if inter_area > 0:
                    has_any_intersection = True

                visibility = inter_area / orig_area
                if visibility >= min_visibility:
                    # Map to tile local coordinates
                    lx1 = max(0.0, min(float(tile_w), gx1 - tx1))
                    ly1 = max(0.0, min(float(tile_h), gy1 - ty1))
                    lx2 = max(0.0, min(float(tile_w), gx2 - tx1))
                    ly2 = max(0.0, min(float(tile_h), gy2 - ty1))

                    xmin = int(round(max(0.0, min(1000.0, (lx1 / float(tile_w)) * 1000))))
                    ymin = int(round(max(0.0, min(1000.0, (ly1 / float(tile_h)) * 1000))))
                    xmax = int(round(max(0.0, min(1000.0, (lx2 / float(tile_w)) * 1000))))
                    ymax = int(round(max(0.0, min(1000.0, (ly2 / float(tile_h)) * 1000))))

                    if xmax > xmin and ymax > ymin:
                        tile_objs.append({
                            "label": g["label"],
                            "norm_box": [xmin, ymin, xmax, ymax]
                        })

            if tile_objs:
                # Positive tile sample containing components meeting visibility threshold
                crop_fname = f"{split_name}_{iid}_tile_{t_idx:03d}.png"
                crop_path = os.path.join(crops_dir, crop_fname)
                crop_img = pil_img.crop((tx1, ty1, tx2, ty2))
                crop_img.save(crop_path, format="PNG")

                sft_records.append({
                    "id": f"{split_name}_tile_{iid}_{t_idx:03d}",
                    "image": os.path.abspath(crop_path),
                    "orig_image_id": iid,
                    "width": tile_w,
                    "height": tile_h,
                    "board_folder": folder,
                    "tile_bbox": [tx1, ty1, tx2, ty2],
                    "tile_offset": [tx1, ty1],
                    "conversations": [
                        {"from": "user", "value": prompt},
                        {"from": "assistant", "value": build_qwen_grounding_target(tile_objs, tile_w, tile_h)}
                    ],
                    "num_objects": len(tile_objs),
                    "is_negative": False,
                    "sft_version": "2.3_canonical"
                })
                total_components_included += len(tile_objs)
            else:
                # tile_objs is empty: distinguish truly empty tile from excluded boundary component
                if has_any_intersection:
                    # Boundary components are visible on the edge but below min_visibility.
                    # STRICT: Must NOT label as 'None', which would train model that visible edge components are non-objects.
                    continue
                else:
                    # Truly empty tile (0 components intersecting with tile bbox).
                    # Hard Negative Mining: sample 1 in 3 clean empty tiles
                    if t_idx % 3 == 0:
                        crop_fname = f"{split_name}_{iid}_tile_neg_{t_idx:03d}.png"
                        crop_path = os.path.join(crops_dir, crop_fname)
                        crop_img = pil_img.crop((tx1, ty1, tx2, ty2))
                        crop_img.save(crop_path, format="PNG")

                        sft_records.append({
                            "id": f"{split_name}_tile_neg_{iid}_{t_idx:03d}",
                            "image": os.path.abspath(crop_path),
                            "orig_image_id": iid,
                            "width": tile_w,
                            "height": tile_h,
                            "board_folder": folder,
                            "tile_bbox": [tx1, ty1, tx2, ty2],
                            "tile_offset": [tx1, ty1],
                            "conversations": [
                                {"from": "user", "value": prompt},
                                {"from": "assistant", "value": "None"}
                            ],
                            "num_objects": 0,
                            "is_negative": True,
                            "sft_version": "2.3_canonical"
                        })

    return sft_records


def main():
    parser = argparse.ArgumentParser(description="Construct Qwen2-VL Multimodal SFT datasets.")
    parser.add_argument("--coco-gt", type=str, required=True, help="Path to coco_annotations.json.")
    parser.add_argument("--splits", type=str, required=True, help="Path to splits.json.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to images_manifest.json.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for SFT data.")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile crop size.")
    parser.add_argument("--overlap-ratio", type=float, default=0.20, help="Overlap ratio between tiles.")
    parser.add_argument("--min-visibility", type=float, default=0.50, help="Min component visible area ratio.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError("SFT rebuild requires a new empty output directory")
    os.makedirs(args.output_dir, exist_ok=True)
    crops_dir = os.path.join(args.output_dir, "crops")

    with open(args.coco_gt, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    from pcb_vlm.provenance import validate_partition
    from pcb_vlm.config import validate_category_consistency
    validate_partition(splits, manifest)
    validate_category_consistency([c["name"] for c in coco["categories"]])
    # Categories
    cat_id_to_name = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}
    target_classes = sorted(list(set(cat_id_to_name.values())))

    # Group COCO annotations by image_id
    coco_anns_by_img = {}
    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        coco_anns_by_img.setdefault(iid, []).append(ann)

    print(f"Building SFT dataset for target classes: {target_classes}")

    # Build Train SFT
    train_ids = splits["splits"]["train"]["image_ids"]
    print(f"\nProcessing Train split ({len(train_ids)} images)...")
    train_records = process_split_sft(
        split_name="train",
        target_image_ids=train_ids,
        manifest=manifest,
        coco_anns_by_img=coco_anns_by_img,
        cat_id_to_name=cat_id_to_name,
        target_classes=target_classes,
        output_dir=args.output_dir,
        crops_dir=os.path.join(crops_dir, "train"),
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        min_visibility=args.min_visibility
    )

    # Build Val SFT
    val_ids = splits["splits"]["val"]["image_ids"]
    print(f"\nProcessing Val split ({len(val_ids)} images)...")
    val_records = process_split_sft(
        split_name="val",
        target_image_ids=val_ids,
        manifest=manifest,
        coco_anns_by_img=coco_anns_by_img,
        cat_id_to_name=cat_id_to_name,
        target_classes=target_classes,
        output_dir=args.output_dir,
        crops_dir=os.path.join(crops_dir, "val"),
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        min_visibility=args.min_visibility
    )

    from pcb_vlm.provenance import stamp_samples
    stamp_samples(train_records, "train", args.splits, manifest)
    stamp_samples(val_records, "val", args.splits, manifest)

    train_out = os.path.join(args.output_dir, "train_sft.json")
    val_out = os.path.join(args.output_dir, "val_sft.json")

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)

    # Compute dataset hash
    h = hashlib.sha256()
    with open(train_out, "rb") as f:
        h.update(f.read())
    with open(val_out, "rb") as f:
        h.update(f.read())
    dataset_hash = h.hexdigest()

    summary = {
        "dataset_hash": dataset_hash,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "train_components": sum(r["num_objects"] for r in train_records),
        "val_components": sum(r["num_objects"] for r in val_records),
        "target_classes": target_classes,
        "tile_size": args.tile_size,
        "overlap_ratio": args.overlap_ratio,
        "min_visibility": args.min_visibility,
        "train_file": os.path.abspath(train_out),
        "val_file": os.path.abspath(val_out)
    }

    summary_out = os.path.join(args.output_dir, "sft_summary.json")
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== SFT Dataset Construction Complete ===")
    print(f"  Dataset SHA-256: {dataset_hash}")
    print(f"  Train Samples:   {len(train_records)} (Total objects: {summary['train_components']})")
    print(f"  Val Samples:     {len(val_records)} (Total objects: {summary['val_components']})")
    print(f"  Summary saved:   {summary_out}")


if __name__ == "__main__":
    main()
