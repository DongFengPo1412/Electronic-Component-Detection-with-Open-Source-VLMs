"""Tiling (Slicing Aided Hyper Inference) module for high-resolution PCB images.

Splits large images into overlapping patches, restores local tile predictions
to global coordinates, and performs NMS merging across overlapping patches.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def generate_tiles(
    img_w: int,
    img_h: int,
    tile_size: int = 1024,
    overlap_ratio: float = 0.20
) -> List[Dict[str, Any]]:
    """Generate bounding boxes for overlapping tiles across an image.
    
    Returns a list of tile dicts:
    [
        {"tile_id": 0, "bbox": [x1, y1, x2, y2], "width": tw, "height": th, "offset_x": x1, "offset_y": y1},
        ...
    ]
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0.0, 1.0).")

    stride = max(1, int(tile_size * (1.0 - overlap_ratio)))

    x_steps = list(range(0, max(1, img_w - tile_size + 1), stride))
    if not x_steps or x_steps[-1] + tile_size < img_w:
        x_steps.append(max(0, img_w - tile_size))
    # Deduplicate while preserving order
    x_steps = sorted(list(set(x_steps)))

    y_steps = list(range(0, max(1, img_h - tile_size + 1), stride))
    if not y_steps or y_steps[-1] + tile_size < img_h:
        y_steps.append(max(0, img_h - tile_size))
    y_steps = sorted(list(set(y_steps)))

    tiles = []
    tile_id = 0
    for y in y_steps:
        for x in x_steps:
            x2 = min(img_w, x + tile_size)
            y2 = min(img_h, y + tile_size)
            tiles.append({
                "tile_id": tile_id,
                "bbox": [x, y, x2, y2],
                "offset_x": x,
                "offset_y": y,
                "width": x2 - x,
                "height": y2 - y
            })
            tile_id += 1

    return tiles


def map_tile_box_to_global(
    local_box: List[float],
    offset_x: int,
    offset_y: int,
    img_w: int,
    img_h: int
) -> List[float]:
    """Map a local [x1, y1, x2, y2] detection inside a tile to global image coordinates."""
    lx1, ly1, lx2, ly2 = local_box
    gx1 = max(0.0, min(float(img_w), lx1 + offset_x))
    gy1 = max(0.0, min(float(img_h), ly1 + offset_y))
    gx2 = max(0.0, min(float(img_w), lx2 + offset_x))
    gy2 = max(0.0, min(float(img_h), ly2 + offset_y))
    return [round(gx1, 2), round(gy1, 2), round(gx2, 2), round(gy2, 2)]


def nms_merge_predictions(
    predictions: List[Dict[str, Any]],
    iou_threshold: float = 0.50
) -> List[Dict[str, Any]]:
    """Per-class Non-Maximum Suppression (NMS) for merging patch predictions.
    
    Supports both scored predictions and non-scored (order/area-prioritized) predictions.
    """
    if not predictions:
        return []

    # Group by category
    by_category = {}
    for pred in predictions:
        cat = pred["label"]
        by_category.setdefault(cat, []).append(pred)

    merged = []
    for cat, items in by_category.items():
        # Sort items: if score is present, sort descending by score; else sort descending by area
        def sort_key(item):
            score = item.get("score")
            if score is not None:
                return float(score)
            x1, y1, x2, y2 = item["bbox_xyxy"]
            return (x2 - x1) * (y2 - y1)

        sorted_items = sorted(items, key=sort_key, reverse=True)
        keep = []

        while sorted_items:
            best = sorted_items.pop(0)
            keep.append(best)
            b_box = best["bbox_xyxy"]
            remaining = []

            for other in sorted_items:
                o_box = other["bbox_xyxy"]
                # Compute IoU
                inter_x1 = max(b_box[0], o_box[0])
                inter_y1 = max(b_box[1], o_box[1])
                inter_x2 = min(b_box[2], o_box[2])
                inter_y2 = min(b_box[3], o_box[3])

                iw = max(0.0, inter_x2 - inter_x1)
                ih = max(0.0, inter_y2 - inter_y1)
                inter_area = iw * ih

                area_b = (b_box[2] - b_box[0]) * (b_box[3] - b_box[1])
                area_o = (o_box[2] - o_box[0]) * (o_box[3] - o_box[1])
                union_area = area_b + area_o - inter_area

                iou = (inter_area / union_area) if union_area > 0 else 0.0

                if iou < iou_threshold:
                    remaining.append(other)

            sorted_items = remaining

        merged.extend(keep)

    return merged
