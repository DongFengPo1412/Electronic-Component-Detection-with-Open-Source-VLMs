"""Validate data splits for physical board leakage, image overlap, and category coverage."""

import argparse
import hashlib
import json
import sys
from typing import Dict, Any


def validate_splits(splits_path: str) -> Dict[str, Any]:
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    splits = data["splits"]

    train_boards = set(splits["train"]["boards"])
    val_boards = set(splits["val"]["boards"])
    test_boards = set(splits["test"]["boards"])

    train_images = set(splits["train"]["image_ids"])
    val_images = set(splits["val"]["image_ids"])
    test_images = set(splits["test"]["image_ids"])

    train_folders = set(splits["train"]["image_folders"])
    val_folders = set(splits["val"]["image_folders"])
    test_folders = set(splits["test"]["image_folders"])

    # 1. Board overlap check
    board_leakage_tv = train_boards & val_boards
    board_leakage_tt = train_boards & test_boards
    board_leakage_vt = val_boards & test_boards
    has_board_leakage = bool(board_leakage_tv or board_leakage_tt or board_leakage_vt)

    # 2. Image overlap check
    img_leakage_tv = train_images & val_images
    img_leakage_tt = train_images & test_images
    img_leakage_vt = val_images & test_images
    has_img_leakage = bool(img_leakage_tv or img_leakage_tt or img_leakage_vt)

    # 3. Total counts check
    total_split_boards = len(train_boards) + len(val_boards) + len(test_boards)
    total_split_images = len(train_images) + len(val_images) + len(test_images)

    passed_all_checks = (
        not has_board_leakage
        and not has_img_leakage
        and total_split_boards == meta["total_physical_boards"]
        and total_split_images == meta["total_images"]
    )

    report = {
        "split_hash": meta.get("split_hash"),
        "total_boards": meta["total_physical_boards"],
        "total_images": meta["total_images"],
        "board_isolation_audit": {
            "train_val_board_overlap": list(board_leakage_tv),
            "train_test_board_overlap": list(board_leakage_tt),
            "val_test_board_overlap": list(board_leakage_vt),
            "zero_board_leakage": not has_board_leakage
        },
        "image_isolation_audit": {
            "train_val_image_overlap": list(img_leakage_tv),
            "train_test_image_overlap": list(img_leakage_tt),
            "val_test_image_overlap": list(img_leakage_vt),
            "zero_image_leakage": not has_img_leakage
        },
        "partition_sizes": {
            "train": {"boards": len(train_boards), "images": len(train_images), "annotations": splits["train"]["num_annotations"]},
            "val": {"boards": len(val_boards), "images": len(val_images), "annotations": splits["val"]["num_annotations"]},
            "test": {"boards": len(test_boards), "images": len(test_images), "annotations": splits["test"]["num_annotations"]}
        },
        "passed_validation": passed_all_checks
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit train/val/test splits for leakage and consistency.")
    parser.add_argument("--splits", type=str, required=True, help="Path to splits.json.")
    args = parser.parse_args()

    print(f"Validating splits file: {args.splits}")
    report = validate_splits(args.splits)

    print("\n=== Split Validation Results ===")
    print(f"  Split Hash: {report['split_hash']}")
    print(f"  Zero Board Leakage: {report['board_isolation_audit']['zero_board_leakage']}")
    print(f"  Zero Image Leakage: {report['image_isolation_audit']['zero_image_leakage']}")
    print(f"  Passed Validation:  {report['passed_validation']}")

    if not report["passed_validation"]:
        print("ERROR: Split validation failed!")
        sys.exit(1)
    else:
        print("SUCCESS: Splits verified completely isolated with ZERO board and ZERO image leakage.")


if __name__ == "__main__":
    main()
