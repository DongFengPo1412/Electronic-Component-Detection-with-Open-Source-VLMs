"""Audit converted COCO GT PCB dataset for integrity, legality, distribution and board statistics."""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Any
import numpy as np


def audit_coco_dataset(coco_path: str, manifest_path: str, output_report: str = None) -> Dict[str, Any]:
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat for cat in coco["categories"]}
    annotations = coco["annotations"]

    # 1. Integrity and boundary checks
    illegal_boxes = []
    out_of_bounds = []
    zero_area_boxes = []

    cat_counts = Counter()
    cat_areas = defaultdict(list)
    board_counts = Counter()
    board_cat_counts = defaultdict(Counter)

    for ann in annotations:
        aid = ann["id"]
        iid = ann["image_id"]
        cid = ann["category_id"]
        bbox = ann["bbox"] # [x, y, w, h]
        x, y, w, h = bbox
        board_group = ann.get("board_group_id", "unknown")

        img = images.get(iid)
        if not img:
            illegal_boxes.append({"ann_id": aid, "error": f"Image id {iid} not found"})
            continue

        img_w, img_h = img["width"], img["height"]

        if w <= 0 or h <= 0 or ann.get("area", 0) <= 0:
            zero_area_boxes.append({"ann_id": aid, "bbox": bbox})

        if x < 0 or y < 0 or (x + w) > img_w + 1e-3 or (y + h) > img_h + 1e-3:
            out_of_bounds.append({"ann_id": aid, "bbox": bbox, "img_size": [img_w, img_h]})

        cat_counts[cid] += 1
        cat_areas[cid].append(w * h)
        board_counts[board_group] += 1
        board_cat_counts[board_group][cid] += 1

    # Distribution summary
    cat_stats = {}
    for cid, cat in categories.items():
        cname = cat["name"]
        cnt = cat_counts[cid]
        areas = cat_areas[cid]
        if areas:
            cat_stats[cname] = {
                "count": cnt,
                "min_area": round(float(np.min(areas)), 1),
                "median_area": round(float(np.median(areas)), 1),
                "mean_area": round(float(np.mean(areas)), 1),
                "max_area": round(float(np.max(areas)), 1)
            }
        else:
            cat_stats[cname] = {"count": 0, "min_area": 0, "median_area": 0, "mean_area": 0, "max_area": 0}

    # Board statistics
    board_stats = {}
    for board, total_cnt in board_counts.most_common():
        board_stats[board] = {
            "total_components": total_cnt,
            "images": [m["file_name"] for m in manifest if m["board_group_id"] == board]
        }

    resolutions = [(img["width"], img["height"]) for img in coco["images"]]
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]

    audit_summary = {
        "dataset_name": coco.get("info", {}).get("description", "PCB Component Dataset"),
        "total_images": len(coco["images"]),
        "total_annotations": len(annotations),
        "total_physical_boards": len(board_counts),
        "resolutions": {
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
            "min_height": int(np.min(heights)),
            "max_height": int(np.max(heights))
        },
        "quality_audit": {
            "illegal_boxes_count": len(illegal_boxes),
            "out_of_bounds_count": len(out_of_bounds),
            "zero_area_boxes_count": len(zero_area_boxes),
            "passed_sanity_check": (len(illegal_boxes) == 0 and len(out_of_bounds) == 0 and len(zero_area_boxes) == 0)
        },
        "category_statistics": cat_stats,
        "physical_board_statistics": board_stats
    }

    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from pcb_vlm.provenance import physical_id, sha256
    by_id = {e['image_id']: e for e in manifest}
    source_errors = [e['image_id'] for e in manifest if not os.path.isfile(e['image_path']) or sha256(e['image_path']) != e['sha256']]
    hashes = Counter(e['sha256'] for e in manifest)
    coverage = {}
    for cid, category in categories.items():
        image_ids = {a['image_id'] for a in annotations if a['category_id'] == cid}
        coverage[category['name']] = {'images': len(image_ids), 'physical_boards': len({physical_id(by_id[i]['folder']) for i in image_ids})}
    from pcb_vlm.data.raw_labels import audit_annotation_types
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs/classes.yaml')
    with open(config_path, encoding='utf-8') as stream:
        class_config = yaml.safe_load(stream)
    patterns = {pat.lower():cat['id'] for cat in class_config['categories'] for pat in cat.get('raw_patterns',[])}
    semantics = audit_annotation_types(coco, manifest, patterns, class_config.get('ignore_patterns',[]))
    audit_summary['raw_xml_semantic_audit'] = semantics
    audit_summary['quality_audit']['passed_sanity_check'] &= semantics['passed']
    audit_summary['source_errors'] = source_errors
    audit_summary['duplicate_image_hashes'] = [h for h,n in hashes.items() if n > 1]
    audit_summary['category_coverage'] = coverage
    audit_summary['quality_audit']['passed_sanity_check'] &= not source_errors and not audit_summary['duplicate_image_hashes']
    if output_report:
        os.makedirs(os.path.dirname(output_report), exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2, ensure_ascii=False)

    return audit_summary


def main():
    parser = argparse.ArgumentParser(description="Audit COCO GT PCB dataset.")
    parser.add_argument("--coco", type=str, required=True, help="Path to coco_annotations.json.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to images_manifest.json.")
    parser.add_argument("--report", type=str, default=None, help="Output path for audit report JSON.")
    args = parser.parse_args()

    print(f"Auditing COCO annotations at: {args.coco}")
    summary = audit_coco_dataset(args.coco, args.manifest, args.report)

    print("\n=== Dataset Quality Audit ===")
    qa = summary["quality_audit"]
    print(f"  Illegal boxes: {qa['illegal_boxes_count']}")
    print(f"  Out of bounds: {qa['out_of_bounds_count']}")
    print(f"  Zero area boxes: {qa['zero_area_boxes_count']}")
    print(f"  Passed sanity check: {qa['passed_sanity_check']}")

    print("\n=== Category Distribution ===")
    for cname, st in summary["category_statistics"].items():
        print(f"  {cname:15s}: count={st['count']:5d}, median_area={st['median_area']:8.1f} px^2")

    print(f"\nTotal Physical Boards: {summary['total_physical_boards']}")
    print("Dataset audit complete.")
    if not qa["passed_sanity_check"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
