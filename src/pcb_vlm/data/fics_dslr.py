"""Minimal isolated parser, validator, and COCO/manifest exporter for FICS-PCB DSLR.

Deliberately decoupled from legacy WACV 11-class ontology (`pcb_vlm.config`) and
the fixed 29-board WACV map (`configs/board_canonical_map.json`).
Never uses `eval()` and never applies the 2px classification crop expansion from
upstream `extractCSV_dslr.py` (`adjust_coord`).
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

FICS_RAW_TO_CANONICAL: Dict[str, str] = {
    "capacitors": "capacitor",
    "resistors": "resistor",
    "inductors": "inductor",
    "transistors": "transistor",
    "diodes": "diode",
    "ICs": "ic",
}

FICS_CANONICAL_CATEGORIES: List[Dict[str, Any]] = [
    {"id": 1, "name": "capacitor", "raw_type": "capacitors"},
    {"id": 2, "name": "resistor", "raw_type": "resistors"},
    {"id": 3, "name": "inductor", "raw_type": "inductors"},
    {"id": 4, "name": "transistor", "raw_type": "transistors"},
    {"id": 5, "name": "diode", "raw_type": "diodes"},
    {"id": 6, "name": "ic", "raw_type": "ICs"},
]

FICS_CATEGORY_NAME_TO_ID: Dict[str, int] = {
    item["name"]: int(item["id"]) for item in FICS_CANONICAL_CATEGORIES
}

_FILENAME_RE = re.compile(r"^(s\d+)_([a-zA-Z0-9]+)$")


def file_sha256(path: Path) -> str:
    """Return lowercase hex SHA-256 of a file without modifying it."""
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_board_and_side(stem: str) -> Tuple[str, str]:
    """Parse `<board>_<side>` stems such as `s18_front` -> (`s18`, `front`)."""
    match = _FILENAME_RE.match(stem)
    if not match:
        raise ValueError(f"Unrecognized FICS DSLR filename stem: {stem!r}")
    return match.group(1), match.group(2).lower()


def parse_component_location(raw_loc: str) -> Tuple[Dict[str, Any], List[float], str]:
    """Safely parse `component_location` into (raw_location, bbox_xyxy, note).

    Uses strict JSON parsing first with `ast.literal_eval` fallback for Python-style
    single-quoted literals; never invokes `eval()`. Raises `ValueError` on unknown
    shapes or non-finite/degenerate coordinates instead of mutating boxes.
    """
    if not isinstance(raw_loc, str) or not raw_loc.strip():
        raise ValueError("Empty component_location string")
    try:
        loc = json.loads(raw_loc)
    except json.JSONDecodeError:
        try:
            loc = ast.literal_eval(raw_loc)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Malformed component_location literal: {exc}") from exc

    if not isinstance(loc, dict):
        raise ValueError(f"Expected dict for component_location, got {type(loc).__name__}")

    shape = loc.get("name")
    if shape == "rect":
        for key in ("x", "y", "width", "height"):
            if key not in loc:
                raise ValueError(f"Missing rect key '{key}'")
        x = float(loc["x"])
        y = float(loc["y"])
        w = float(loc["width"])
        h = float(loc["height"])
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("Non-finite rect coordinates")
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"Non-positive rect size w={w}, h={h}")
        raw_norm = {"shape": "rect", "x": x, "y": y, "width": w, "height": h}
        return raw_norm, [x, y, x + w, y + h], "original_rect_unexpanded"

    if shape == "polygon":
        xs_raw = loc.get("all_points_x")
        ys_raw = loc.get("all_points_y")
        if not isinstance(xs_raw, list) or not isinstance(ys_raw, list):
            raise ValueError("Polygon missing all_points_x / all_points_y lists")
        if len(xs_raw) < 3 or len(xs_raw) != len(ys_raw):
            raise ValueError("Invalid polygon point count or x/y length mismatch")
        xs = [float(v) for v in xs_raw]
        ys = [float(v) for v in ys_raw]
        if not all(math.isfinite(v) for v in xs + ys):
            raise ValueError("Non-finite polygon coordinates")
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate polygon bbox [{x1}, {y1}, {x2}, {y2}]")
        twice_area = sum(
            xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
            for i in range(len(xs))
        )
        if abs(twice_area) <= 0.0:
            raise ValueError(f"Collinear or zero-area polygon points")
        raw_norm = {"shape": "polygon", "all_points_x": xs, "all_points_y": ys}
        return raw_norm, [x1, y1, x2, y2], "axis_aligned_envelope_from_polygon_unexpanded"

    raise ValueError(f"Unsupported component_location shape: {shape!r}")


def validate_box_within_image(bbox_xyxy: List[float], img_w: int, img_h: int) -> None:
    """Validate [x1, y1, x2, y2] strictly inside [0, img_w] x [0, img_h]."""
    x1, y1, x2, y2 = bbox_xyxy
    if not all(math.isfinite(v) for v in bbox_xyxy):
        raise ValueError("Non-finite bbox coordinate")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox {bbox_xyxy}")
    if x1 < 0.0 or y1 < 0.0 or x2 > float(img_w) or y2 > float(img_h):
        raise ValueError(f"Out-of-bounds bbox {bbox_xyxy} for image {img_w}x{img_h}")


def validate_physical_board_disjoint_splits(
    splits: Dict[str, List[str]],
    manifest: List[Dict[str, Any]],
) -> None:
    """Enforce zero image overlap and zero physical-board leakage across splits."""
    by_id = {entry["image_id"]: entry for entry in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("Duplicate image_id in FICS manifest")

    seen_images: set[str] = set()
    seen_boards: set[str] = set()
    for split_name, image_ids in splits.items():
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Duplicate image_id inside split {split_name!r}")
        boards_in_split: set[str] = set()
        for iid in image_ids:
            if iid not in by_id:
                raise ValueError(f"Unknown image_id {iid!r} in split {split_name!r}")
            if iid in seen_images:
                raise ValueError(f"Image {iid!r} assigned to multiple splits")
            seen_images.add(iid)
            boards_in_split.add(by_id[iid]["physical_board_id"])
        overlap_boards = seen_boards.intersection(boards_in_split)
        if overlap_boards:
            raise ValueError(
                f"Physical board leakage across splits in {split_name!r}: {sorted(overlap_boards)}"
            )
        seen_boards.update(boards_in_split)


def audit_and_build_fics_dslr_bundle(
    sample_root: Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """Audit `<sample_root>/DSLR/{img,annotation}` and build COCO + manifest + checks.

    Expects `sample_root` to point to e.g. `.../extracted/s18`.
    """
    sample_root = Path(sample_root)
    img_dir = sample_root / "DSLR" / "img"
    ann_dir = sample_root / "DSLR" / "annotation"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing DSLR/img or DSLR/annotation under {sample_root}")

    img_files = sorted(p for p in img_dir.iterdir() if p.is_file() and not p.name.startswith("._"))
    csv_files = sorted(p for p in ann_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith("._"))

    img_by_stem: Dict[str, Path] = {}
    for p in img_files:
        if p.stem in img_by_stem:
            raise ValueError(f"Duplicate image stem '{p.stem}' in {img_dir}: {[str(img_by_stem[p.stem]), str(p)]}")
        img_by_stem[p.stem] = p

    csv_by_stem: Dict[str, Path] = {}
    for p in csv_files:
        if p.stem in csv_by_stem:
            raise ValueError(f"Duplicate CSV stem '{p.stem}' in {ann_dir}: {[str(csv_by_stem[p.stem]), str(p)]}")
        csv_by_stem[p.stem] = p
    imgs_without_csv = sorted(set(img_by_stem) - set(csv_by_stem))
    csvs_without_img = sorted(set(csv_by_stem) - set(img_by_stem))
    if strict and (imgs_without_csv or csvs_without_img):
        raise ValueError(
            f"Image/CSV stem mismatch: imgs_without_csv={imgs_without_csv}, csvs_without_img={csvs_without_img}"
        )

    manifest: List[Dict[str, Any]] = []
    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    object_mappings: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []
    raw_type_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()

    ann_id = 1
    for img_idx, stem in enumerate(sorted(set(img_by_stem).intersection(csv_by_stem)), start=1):
        img_path = img_by_stem[stem]
        csv_path = csv_by_stem[stem]
        board_id, side = parse_board_and_side(stem)
        img_sha = file_sha256(img_path)
        csv_sha = file_sha256(csv_path)

        with Image.open(img_path) as im:
            im.load()
            width, height = im.size
            exif = im.getexif()
            orientation = exif.get(274) if exif else None
            img_format = im.format
            img_mode = im.mode
            n_frames = getattr(im, "n_frames", 1)

        if strict and orientation not in (None, 1):
            raise ValueError(f"Unsupported non-identity EXIF orientation {orientation} in {img_path}")
        if strict and n_frames != 1:
            raise ValueError(f"Unexpected multi-page image ({n_frames} frames) in {img_path}")

        manifest.append({
            "image_id": stem,
            "coco_image_id": img_idx,
            "board_id": board_id,
            "physical_board_id": board_id,
            "side": side,
            "file_name": img_path.name,
            "image_path": str(img_path),
            "image_sha256": img_sha,
            "csv_path": str(csv_path),
            "csv_sha256": csv_sha,
            "format": img_format,
            "mode": img_mode,
            "n_frames": n_frames,
            "exif_orientation": orientation,
            "width": width,
            "height": height,
        })
        coco_images.append({
            "id": img_idx,
            "sample_id": stem,
            "file_name": img_path.name,
            "width": width,
            "height": height,
            "board_id": board_id,
            "side": side,
            "sha256": img_sha,
        })

        seen_boxes: Dict[Tuple[float, float, float, float], int] = {}
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_cols = {"image_name", "component_location", "component_type"}
            if not reader.fieldnames or not required_cols.issubset(reader.fieldnames):
                raise ValueError(f"Missing required CSV columns in {csv_path}: {reader.fieldnames}")

            for row_num, row in enumerate(reader, start=2):
                row_issues: List[str] = []
                row_img_name = (row.get("image_name") or "").strip()
                raw_type = (row.get("component_type") or "").strip()
                text_on_comp = (row.get("text_on_component") or "").strip()
                logo = (row.get("logo") or "").strip()
                raw_type_counts[raw_type] += 1

                if row_img_name != img_path.name:
                    row_issues.append(f"image_name_mismatch:{row_img_name}")

                if raw_type not in FICS_RAW_TO_CANONICAL:
                    row_issues.append(f"unknown_component_type:{raw_type}")
                    canon_name = f"UNKNOWN:{raw_type}"
                    cat_id = -1
                else:
                    canon_name = FICS_RAW_TO_CANONICAL[raw_type]
                    cat_id = FICS_CATEGORY_NAME_TO_ID[canon_name]

                raw_loc_norm: Optional[Dict[str, Any]] = None
                bbox_xyxy: Optional[List[float]] = None
                conversion_note = ""
                try:
                    raw_loc_norm, bbox_xyxy, conversion_note = parse_component_location(
                        row.get("component_location") or ""
                    )
                    validate_box_within_image(bbox_xyxy, width, height)
                except ValueError as exc:
                    row_issues.append(str(exc))

                if bbox_xyxy is not None:
                    bkey = tuple(bbox_xyxy)
                    if bkey in seen_boxes:
                        row_issues.append(f"duplicate_bbox_of_row:{seen_boxes[bkey]}")
                    else:
                        seen_boxes[bkey] = row_num

                if row_issues:
                    anomalies.append({
                        "csv": csv_path.name,
                        "csv_row": row_num,
                        "issues": row_issues,
                        "raw_component_type": raw_type,
                    })
                    if strict:
                        raise ValueError(f"Strict FICS CSV validation failed at row {row_num}: {row_issues}")
                    continue

                assert bbox_xyxy is not None and raw_loc_norm is not None
                x1, y1, x2, y2 = bbox_xyxy
                w_box, h_box = x2 - x1, y2 - y1
                canonical_counts[canon_name] += 1

                object_mappings.append({
                    "annotation_id": ann_id,
                    "source_csv": str(csv_path.name),
                    "csv_row": row_num,
                    "image_id": stem,
                    "image_name": row_img_name,
                    "board_id": board_id,
                    "side": side,
                    "raw_component_type": raw_type,
                    "canonical_category": canon_name,
                    "category_id": cat_id,
                    "text_on_component": text_on_comp,
                    "logo": logo,
                    "raw_location": raw_loc_norm,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, w_box, h_box],
                    "area_px2": w_box * h_box,
                    "conversion_note": conversion_note,
                    "status": "kept",
                })
                coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w_box, h_box],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "area": w_box * h_box,
                    "iscrowd": 0,
                    "raw_component_type": raw_type,
                    "source_csv": str(csv_path.name),
                    "csv_row": row_num,
                    "text_on_component": text_on_comp,
                    "logo": logo,
                })
                ann_id += 1

    coco_bundle = {
        "info": {
            "dataset": "FICS-PCB_DSLR_Diagnostic_Sample",
            "version": "fics_dslr_diag_v1",
            "split_policy": "format_diagnostic_only_no_train_val_test_split",
        },
        "images": coco_images,
        "categories": FICS_CANONICAL_CATEGORIES,
        "annotations": coco_annotations,
    }

    checks = {
        "image_csv_mutual_coverage": not imgs_without_csv and not csvs_without_img and len(manifest) > 0,
        "object_count_conserved": len(coco_annotations) == len(object_mappings) and len(coco_annotations) + len(anomalies) == sum(raw_type_counts.values()),
        "zero_anomalies": len(anomalies) == 0,
        "manifest_nonempty": len(manifest) > 0,
    }

    return {
        "manifest": manifest,
        "coco": coco_bundle,
        "object_mappings": object_mappings,
        "raw_type_counts": dict(raw_type_counts),
        "canonical_counts": dict(canonical_counts),
        "anomalies": anomalies,
        "checks": checks,
    }
