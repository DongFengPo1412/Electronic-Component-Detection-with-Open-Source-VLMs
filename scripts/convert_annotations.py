"""Convert PCB dataset raw XML annotations to standardized COCO GT JSON and manifest."""

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Tuple
import yaml
import cv2
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def load_classes_config(config_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    categories = cfg["categories"]
    from pcb_vlm.config import get_category_definitions
    if [(c["id"], c["name"]) for c in categories] != [(c["id"], c["name"]) for c in get_category_definitions()]:
        raise ValueError("Category ID/name mapping differs from classes.yaml")
    pattern_to_id = {}
    for cat in categories:
        cid = cat["id"]
        for pat in cat.get("raw_patterns", []):
            pattern_to_id[pat.lower()] = cid
    ignore_patterns = [p.lower() for p in cfg.get("ignore_patterns", [])]
    return categories, pattern_to_id, ignore_patterns


CANONICAL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "board_canonical_map.json")
_CANONICAL_CACHE = None

def get_canonical_map() -> Dict[str, str]:
    global _CANONICAL_CACHE
    if _CANONICAL_CACHE is None:
        if os.path.exists(CANONICAL_MAP_PATH):
            with open(CANONICAL_MAP_PATH, "r", encoding="utf-8") as f:
                _CANONICAL_CACHE = json.load(f).get("mapping", {})
        else:
            _CANONICAL_CACHE = {}
    return _CANONICAL_CACHE

def extract_board_group_id(folder_name: str) -> str:
    """Extract physical board ID using human-verified canonical mapping table first."""
    cmap = get_canonical_map()
    if folder_name in cmap:
        return cmap[folder_name]

    raise ValueError(f"Unmapped physical board folder: {folder_name}")


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def match_category(raw_name: str, pattern_to_id: Dict[str, int], ignore_patterns: List[str]) -> Tuple[int, str]:
    clean_name = raw_name.strip().lower().replace('"', '')
    # Check ignore first
    for ign in ignore_patterns:
        if ign in clean_name and not any(k in clean_name for k in ["capacitor", "resistor", "ic", "diode", "transistor", "led"]):
            return 0, "ignored"

    # Check categories
    for pat, cid in pattern_to_id.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(pat) + r"(?![a-z0-9])", clean_name):
            return cid, pat

    return 0, "unmatched"


def convert_dataset(raw_dir: str, config_path: str, output_dir: str, viz_dir: str = None) -> Dict[str, Any]:
    categories, pattern_to_id, ignore_patterns = load_classes_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    coco_images = []
    coco_annotations = []
    images_manifest = []

    ann_id = 1
    board_folders = sorted(os.listdir(raw_dir))
    
    total_raw_boxes = 0
    total_valid_components = 0
    total_ignored = 0
    total_clipped = 0

    cid_to_name = {c["id"]: c["name"] for c in categories}
    color_palette = {
        1: (0, 165, 255),   # capacitor: orange
        2: (0, 255, 0),     # resistor: green
        3: (0, 0, 255),     # ic: red
        4: (255, 255, 0),   # connector: cyan
        5: (255, 0, 255),   # led: magenta
        6: (255, 140, 0),   # transistor: dark orange
        7: (147, 20, 255),  # diode: pink
        8: (211, 0, 148),   # inductor: purple
        9: (0, 215, 255),   # button: gold
        10: (128, 128, 0),  # switch: teal
        11: (180, 105, 255) # clock: hot pink
    }

    for folder_idx, folder in enumerate(board_folders):
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]
        if not xml_files:
            continue

        xml_path = os.path.join(folder_path, xml_files[0])
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find image file
        img_filename = root.find("filename").text if root.find("filename") is not None else None
        img_path = None
        if img_filename:
            candidate = os.path.join(folder_path, img_filename)
            if os.path.exists(candidate):
                img_path = candidate

        if not img_path:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(folder_path, folder + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if not img_path:
            print(f"Warning: Image file not found for folder {folder}, skipping.")
            continue

        # Image dimensions
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print(f"Warning: Could not read image at {img_path}, skipping.")
            continue
        h, w = img_cv.shape[:2]

        image_id = folder_idx + 1
        board_group = extract_board_group_id(folder)
        img_sha256 = compute_sha256(img_path)

        image_entry = {
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "folder": folder,
            "width": w,
            "height": h,
            "board_group_id": board_group,
            "sha256": img_sha256,
            "absolute_path": os.path.abspath(img_path),
            "relative_path": os.path.relpath(img_path, raw_dir)
        }
        coco_images.append(image_entry)

        board_component_count = 0
        img_viz = img_cv.copy() if (viz_dir and folder_idx < 5) else None

        for obj in root.findall("object"):
            total_raw_boxes += 1
            raw_name = obj.find("name").text
            cid, match_info = match_category(raw_name, pattern_to_id, ignore_patterns)
            if cid == 0:
                total_ignored += 1
                continue

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Coordinate validation and clipping
            orig_box = (xmin, ymin, xmax, ymax)
            xmin = max(0.0, min(float(w - 1), xmin))
            ymin = max(0.0, min(float(h - 1), ymin))
            xmax = max(0.0, min(float(w), xmax))
            ymax = max(0.0, min(float(h), ymax))

            if (xmin, ymin, xmax, ymax) != orig_box:
                total_clipped += 1

            bw = xmax - xmin
            bh = ymax - ymin
            area = bw * bh

            if bw <= 1.0 or bh <= 1.0 or area <= 0.0:
                continue

            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": cid,
                "category_name": cid_to_name[cid],
                "bbox": [round(xmin, 2), round(ymin, 2), round(bw, 2), round(bh, 2)],
                "bbox_xyxy": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
                "board_group_id": board_group
            }
            coco_annotations.append(ann)
            ann_id += 1
            board_component_count += 1
            total_valid_components += 1

            # Visualization
            if img_viz is not None:
                color = color_palette.get(cid, (255, 255, 255))
                cv2.rectangle(img_viz, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)
                label_txt = f"{cid_to_name[cid]}"
                cv2.putText(img_viz, label_txt, (int(xmin), max(20, int(ymin) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if img_viz is not None:
            # Resize for visual preview if too large
            max_dim = 1600
            scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
            if scale < 1.0:
                preview = cv2.resize(img_viz, (int(w * scale), int(h * scale)))
            else:
                preview = img_viz
            viz_out = os.path.join(viz_dir, f"{folder}_gt_preview.jpg")
            cv2.imwrite(viz_out, preview)

        images_manifest.append({
            "image_id": image_id,
            "folder": folder,
            "file_name": os.path.basename(img_path),
            "board_group_id": board_group,
            "width": w,
            "height": h,
            "sha256": img_sha256,
            "num_components": board_component_count,
            "image_path": os.path.abspath(img_path)
        })

    # Prepare COCO format
    coco_output = {
        "info": {
            "description": "Electronic Component Detection Dataset (Georgia Tech PCB WACV 2019 Cleaned)",
            "version": "1.0.0",
            "year": 2026,
            "date_created": "2026-10-25"
        },
        "licenses": [{"id": 1, "name": "Academic / Non-Commercial Research"}],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories
    }

    coco_path = os.path.join(output_dir, "coco_annotations.json")
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    manifest_path = os.path.join(output_dir, "images_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(images_manifest, f, indent=2, ensure_ascii=False)

    summary = {
        "total_images": len(coco_images),
        "total_unique_physical_boards": len(set(img["board_group_id"] for img in coco_images)),
        "total_raw_boxes": total_raw_boxes,
        "total_valid_components": total_valid_components,
        "total_ignored_non_components": total_ignored,
        "total_clipped_boundary_boxes": total_clipped,
        "coco_file": coco_path,
        "manifest_file": manifest_path
    }

    summary_path = os.path.join(output_dir, "conversion_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert PCB XML annotations to unified COCO GT format.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Path to raw pcb_wacv_2019 dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to classes.yaml.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for COCO GT.")
    parser.add_argument("--viz-dir", type=str, default=None, help="Directory to save sample GT preview images.")
    args = parser.parse_args()

    print("Running annotation conversion...")
    summary = convert_dataset(args.raw_dir, args.config, args.output_dir, args.viz_dir)
    print("\n=== Conversion Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nAnnotation conversion finished successfully!")


if __name__ == "__main__":
    main()
