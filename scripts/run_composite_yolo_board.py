"""CLI runner for GT-free composite YOLO11n class-aware tiled inference on one board image."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    DEFAULT_ULTRALYTICS_OVERLAY,
    EXPECTED_YOLO11N_BEST_SHA256,
    run_yolo11n_class_aware_board,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run GT-free class-aware YOLO11n tiled detection.")
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    yolo_spec = (plan.get("models") or {}).get("yolo11n") or {}
    q_items = (plan.get("queues") or {}).get("yolo11n") or []
    if not q_items:
        raise ValueError("Missing yolo11n queue items in plan.json")

    board_id = str(q_items[0]["board_id"])
    board_meta = (plan.get("boards") or {})[board_id]
    tiles = [
        {
            "tile_id": int(q["tile_id"]),
            "tile_offset_x": int(q["tile_offset_x"]),
            "tile_offset_y": int(q["tile_offset_y"]),
            "width": int(q["width"]),
            "height": int(q["height"]),
            "path": str(q["path"]),
            "sha256": str(q["sha256"]),
        }
        for q in q_items
    ]
    precomputed = {
        "orig_wh": board_meta["orig_wh"],
        "roi_bbox_half_open": [
            int(board_meta["roi_offset_x"]),
            int(board_meta["roi_offset_y"]),
            int(board_meta["roi_offset_x"]) + int(board_meta["roi_width"]),
            int(board_meta["roi_offset_y"]) + int(board_meta["roi_height"]),
        ],
        "roi_offset_x": int(board_meta["roi_offset_x"]),
        "roi_offset_y": int(board_meta["roi_offset_y"]),
        "roi_width": int(board_meta["roi_width"]),
        "roi_height": int(board_meta["roi_height"]),
        "tiles": tiles,
    }
    src_img = pathlib.Path(str(board_meta.get("source_image_path") or q_items[0]["path"]))
    res = run_yolo11n_class_aware_board(
        image_path=src_img,
        board_id=board_id,
        weights_path=pathlib.Path(str(yolo_spec["local_dir"])),
        expected_weights_sha256=str(yolo_spec.get("weights_sha256") or ""),
        detector_preset=str(yolo_spec.get("detector_preset") or "fics_yolo11n_r15"),
        expected_class_schema=str(yolo_spec.get("expected_class_schema") or "fics_6"),
        overlay_dir=str(yolo_spec.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY),
        output_dir=args.output_dir,
        tile_size=int(board_meta.get("tile_size", 640)),
        overlap_ratio=float(board_meta.get("tile_overlap_ratio", 0.20)),
        imgsz=640,
        conf_low=0.05,
        conf_official=0.25,
        per_class_conf=yolo_spec.get("per_class_conf"),
        nms_iou=0.50,
        precomputed_roi_tiles=precomputed,
    )
    print(json.dumps(res["summary"], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
