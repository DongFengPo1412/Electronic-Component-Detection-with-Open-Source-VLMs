"""Partition PCB dataset strictly by physical board IDs into train/val/test splits."""

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict, Counter
from typing import Dict, Any, List


def make_splits(
    coco_path: str,
    manifest_path: str,
    output_path: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, Any]:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from pcb_vlm.provenance import physical_id
    if os.path.exists(output_path):
        raise FileExistsError("Protect existing split; choose a new output path")
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Group images and annotations by board_group_id
    board_to_images = defaultdict(list)
    for img in manifest:
        board = physical_id(img["folder"])
        if img["board_group_id"] != board:
            raise ValueError("Manifest board identity mismatch")
        board_to_images[board].append(img)

    board_to_cat_counts = defaultdict(Counter)
    for ann in coco["annotations"]:
        source = next(i for i in manifest if i["image_id"] == ann["image_id"])
        bg = physical_id(source["folder"])
        if ann["board_group_id"] != bg:
            raise ValueError("Annotation board identity mismatch")
        cid = ann["category_id"]
        board_to_cat_counts[bg][cid] += 1

    board_ids = sorted(list(board_to_images.keys()))
    num_boards = len(board_ids)

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_boards = list(board_ids)
    rng.shuffle(shuffled_boards)

    n_val = max(1, int(round(num_boards * val_ratio)))
    n_test = max(1, int(round(num_boards * test_ratio)))
    n_train = num_boards - n_val - n_test

    test_boards = sorted(shuffled_boards[:n_test])
    val_boards = sorted(shuffled_boards[n_test:n_test + n_val])
    train_boards = sorted(shuffled_boards[n_test + n_val:])

    # Gather images and annotations for each split
    splits_data = {
        "train": {"boards": train_boards, "images": [], "annotations_count": 0, "categories": Counter()},
        "val": {"boards": val_boards, "images": [], "annotations_count": 0, "categories": Counter()},
        "test": {"boards": test_boards, "images": [], "annotations_count": 0, "categories": Counter()}
    }

    board_split_map = {}
    for b in train_boards:
        board_split_map[b] = "train"
    for b in val_boards:
        board_split_map[b] = "val"
    for b in test_boards:
        board_split_map[b] = "test"

    for img in manifest:
        s = board_split_map[img["board_group_id"]]
        splits_data[s]["images"].append(img)

    for ann in coco["annotations"]:
        s = board_split_map[ann["board_group_id"]]
        splits_data[s]["annotations_count"] += 1
        splits_data[s]["categories"][ann["category_id"]] += 1

    # Format output JSON
    split_export = {
        "metadata": {
            "dataset": coco.get("info", {}).get("description", "PCB Component Dataset"),
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_physical_boards": num_boards,
            "total_images": len(manifest),
            "total_annotations": len(coco["annotations"])
        },
        "splits": {
            "train": {
                "boards": train_boards,
                "image_ids": [img["image_id"] for img in splits_data["train"]["images"]],
                "image_folders": [img["folder"] for img in splits_data["train"]["images"]],
                "num_images": len(splits_data["train"]["images"]),
                "num_annotations": splits_data["train"]["annotations_count"],
                "category_counts": {str(k): v for k, v in splits_data["train"]["categories"].items()}
            },
            "val": {
                "boards": val_boards,
                "image_ids": [img["image_id"] for img in splits_data["val"]["images"]],
                "image_folders": [img["folder"] for img in splits_data["val"]["images"]],
                "num_images": len(splits_data["val"]["images"]),
                "num_annotations": splits_data["val"]["annotations_count"],
                "category_counts": {str(k): v for k, v in splits_data["val"]["categories"].items()}
            },
            "test": {
                "boards": test_boards,
                "image_ids": [img["image_id"] for img in splits_data["test"]["images"]],
                "image_folders": [img["folder"] for img in splits_data["test"]["images"]],
                "num_images": len(splits_data["test"]["images"]),
                "num_annotations": splits_data["test"]["annotations_count"],
                "category_counts": {str(k): v for k, v in splits_data["test"]["categories"].items()}
            }
        }
    }

    # Dump and compute split_hash
    raw_str = json.dumps(split_export, indent=2, sort_keys=True)
    split_hash = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    split_export["metadata"]["split_hash"] = split_hash

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_export, f, indent=2, ensure_ascii=False)

    return split_export


def main():
    parser = argparse.ArgumentParser(description="Create deterministic board-level train/val/test splits.")
    parser.add_argument("--coco", type=str, required=True, help="Path to coco_annotations.json.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to images_manifest.json.")
    parser.add_argument("--output", type=str, required=True, help="Path to output splits.json.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    print(f"Creating splits with seed={args.seed}...")
    res = make_splits(args.coco, args.manifest, args.output, seed=args.seed)
    meta = res["metadata"]
    sp = res["splits"]
    print("\n=== Splits Created ===")
    print(f"  Split Hash: {meta['split_hash']}")
    print(f"  Train: {len(sp['train']['boards'])} boards, {sp['train']['num_images']} images, {sp['train']['num_annotations']} components")
    print(f"  Val:   {len(sp['val']['boards'])} boards, {sp['val']['num_images']} images, {sp['val']['num_annotations']} components")
    print(f"  Test:  {len(sp['test']['boards'])} boards, {sp['test']['num_images']} images, {sp['test']['num_annotations']} components")


if __name__ == "__main__":
    main()
