"""Geometry and bounding box transformation utilities."""

from typing import List, Tuple, Union
import numpy as np


def xyxy_to_xywh(box: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)]


def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]


def qwen_norm_to_xyxy(
    norm_box: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    axis_order: str = "xyxy"
) -> List[float]:
    """Convert Qwen normalized box in [0, 1000] to image pixel [x1, y1, x2, y2].

    By default uses canonical x-first: norm_box = (xmin, ymin, xmax, ymax).
    If axis_order == 'yxyx', interprets as (ymin, xmin, ymax, xmax).
    """
    if axis_order not in ('xyxy', 'yxyx'):
        raise ValueError(f'Unsupported axis order: {axis_order}')
    from .coordinate_protocol import to_pixels
    return to_pixels(norm_box, img_w, img_h, axis_order + '_norm1000')


def xyxy_to_qwen_norm(
    xyxy: List[float],
    img_w: int,
    img_h: int,
    axis_order: str = "xyxy"
) -> Tuple[int, int, int, int]:
    """Convert image pixel [x1, y1, x2, y2] to Qwen normalized integers in [0, 1000].

    By default returns canonical x-first: (xmin, ymin, xmax, ymax).
    If axis_order == 'yxyx', returns (ymin, xmin, ymax, xmax).
    """
    if axis_order not in ('xyxy', 'yxyx'):
        raise ValueError(f'Unsupported axis order: {axis_order}')
    from .coordinate_protocol import to_pixels
    x1, y1, x2, y2 = to_pixels(xyxy, img_w, img_h, 'xyxy_pixels')
    norm = (int(round(x1/img_w*1000)), int(round(y1/img_h*1000)),
            int(round(x2/img_w*1000)), int(round(y2/img_h*1000)))
    return norm if axis_order == 'xyxy' else (norm[1], norm[0], norm[3], norm[2])


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area
