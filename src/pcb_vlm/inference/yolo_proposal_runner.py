"""GT-free Stage A YOLO11n candidate proposal locator runner.

Runs frozen WACV YOLO11n weights (best.pt, SHA-256 5378f977c9d3c56d9d6456ee4e4791b662fa8c294af5f5488f570b5e5a3bb4e1)
across all 1024px 20%-overlap tiles of a board image, discards all WACV category IDs/names,
performs per-tile and whole-board cross-tile class-agnostic NMS at IoU=0.50 with deterministic
tie-breaking by (-score, tile_id, tile_box_index), and persists full provenance & deduplication traces.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pcb_vlm.data.fics_dslr import file_sha256
from pcb_vlm.geometry.tiling import map_tile_box_to_global
from pcb_vlm.inference.gpu_gate import probe_gpu_gate

EXPECTED_YOLO11N_BEST_SHA256 = "5378f977c9d3c56d9d6456ee4e4791b662fa8c294af5f5488f570b5e5a3bb4e1"
DEFAULT_ULTRALYTICS_OVERLAY = r"C:\Users\Liu\.codex\cache\tasks\pcb-train-prep-20261104\dependency-overlay"


def _compute_iou_xyxy(b1: Sequence[float], b2: Sequence[float]) -> float:
    xA = max(float(b1[0]), float(b2[0]))
    yA = max(float(b1[1]), float(b2[1]))
    xB = min(float(b1[2]), float(b2[2]))
    yB = min(float(b1[3]), float(b2[3]))
    iw = max(0.0, xB - xA)
    ih = max(0.0, yB - yA)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = a1 + a2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def class_agnostic_nms_with_dedup_trace(
    proposals: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
    board_id: str,
) -> List[Dict[str, Any]]:
    """Run deterministic class-agnostic NMS sorted by (-score, tile_id, tile_box_index) and record suppressed boxes."""
    ordered = sorted(
        (dict(p) for p in proposals),
        key=lambda x: (-float(x["score"]), int(x["tile_id"]), int(x["tile_box_index"])),
    )
    kept: List[Dict[str, Any]] = []
    while ordered:
        best = ordered.pop(0)
        rank = len(kept)
        best_box = best["bbox_xyxy"]
        suppressed_ids: List[str] = []
        suppressed_details: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for other in ordered:
            ov = _compute_iou_xyxy(best_box, other["bbox_xyxy"])
            if ov >= float(iou_threshold):
                suppressed_ids.append(str(other["pre_nms_id"]))
                suppressed_details.append({
                    "pre_nms_id": str(other["pre_nms_id"]),
                    "tile_id": int(other["tile_id"]),
                    "call_id": str(other["call_id"]),
                    "tile_box_index": int(other["tile_box_index"]),
                    "score": round(float(other["score"]), 6),
                    "bbox_xyxy": [round(float(v), 2) for v in other["bbox_xyxy"]],
                    "iou_with_kept": round(float(ov), 4),
                })
            else:
                remaining.append(other)
        ordered = remaining
        kept.append({
            "prediction_id": f"{board_id}_yolo_p{rank:03d}",
            "proposal_id": f"{board_id}_yolo_p{rank:03d}",
            "pre_nms_id": str(best["pre_nms_id"]),
            "board_id": str(board_id),
            "image_id": int(best.get("image_id", 1)),
            "rank": rank,
            "orig_order": rank,
            "label": "proposal",
            "score": round(float(best["score"]), 6),
            "bbox_xyxy": [round(float(v), 2) for v in best_box],
            "local_bbox_xyxy": [round(float(v), 2) for v in best["local_bbox_xyxy"]],
            "tile_id": int(best["tile_id"]),
            "call_id": str(best["call_id"]),
            "tile_box_index": int(best["tile_box_index"]),
            "suppressed_count": len(suppressed_ids),
            "suppressed_pre_nms_ids": suppressed_ids,
            "suppressed_details": suppressed_details,
        })
    return kept


def run_yolo_proposal_locator(
    *,
    plan_path: pathlib.Path,
    output_dir: pathlib.Path,
    gpu_gate_fn: Callable[[float, float], Dict[str, Any]] = probe_gpu_gate,
    execute: bool = False,
) -> Dict[str, Any]:
    """Execute tiled YOLO11n class-agnostic proposal generation under dual VRAM gate."""
    t0 = time.perf_counter()
    plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
    model_spec = plan["models"]["yolo11n"]
    weights_path = pathlib.Path(model_spec["local_dir"])
    expected_w_sha = str(model_spec.get("weights_sha256") or EXPECTED_YOLO11N_BEST_SHA256).lower()

    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO11n weight file missing: {weights_path}")
    actual_w_sha = file_sha256(weights_path).lower()
    if actual_w_sha != expected_w_sha:
        raise ValueError(f"YOLO11n weight SHA-256 mismatch: {actual_w_sha} != {expected_w_sha}")

    queue_items = list(plan["queues"]["yolo11n"])
    if not queue_items:
        raise ValueError("Empty tile queue for yolo11n")
    board_id = str(queue_items[0]["board_id"])
    board_meta = plan["boards"][board_id]
    orig_w, orig_h = int(board_meta["orig_wh"][0]), int(board_meta["orig_wh"][1])

    run_dir = pathlib.Path(output_dir) / "run_yolo11n"
    prop_dir = pathlib.Path(output_dir).parent / "yolo_proposals"
    run_dir.mkdir(parents=True, exist_ok=True)
    prop_dir.mkdir(parents=True, exist_ok=True)

    if not execute:
        return {
            "dry_run": True,
            "weights_path": str(weights_path),
            "weights_sha256": actual_w_sha,
            "planned_calls": len(queue_items),
        }

    res_limits = plan.get("resource_limits") or {}
    min_smi = float(res_limits.get("gpu_gate_min_smi_free_mib", 5500.0))
    min_torch = float(res_limits.get("gpu_gate_min_torch_free_mib", 6144.0))
    gate = gpu_gate_fn(min_smi, min_torch)
    (run_dir / "gpu_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not bool(gate.get("passed")):
        raise RuntimeError(f"GPU VRAM gate failed: {gate}")

    ov_dir = str(model_spec.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY)
    if ov_dir and pathlib.Path(ov_dir).is_dir() and ov_dir not in sys.path:
        sys.path.insert(0, ov_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import numpy as np
    import torch
    import ultralytics
    from PIL import Image
    from ultralytics import YOLO

    op_cfg = plan.get("yolo_operating_point") or {}
    imgsz = int(op_cfg.get("imgsz", 640))
    conf_thresh = float(op_cfg.get("conf", 0.05))
    tile_nms_iou = float(op_cfg.get("tile_nms_iou", 0.50))
    max_det = int(op_cfg.get("max_det_per_tile", 3000))
    cross_tile_nms_iou = float(op_cfg.get("cross_tile_class_agnostic_nms_iou", 0.50))
    primary_top_k = int(op_cfg.get("primary_top_k", 200))

    t_load0 = time.perf_counter()
    model = YOLO(str(weights_path))
    predict_params = dict(
        imgsz=imgsz,
        conf=conf_thresh,
        iou=tile_nms_iou,
        max_det=max_det,
        device=0,
        half=False,
        rect=False,
        augment=False,
        agnostic_nms=True,
        verbose=False,
    )
    torch.set_num_threads(2)
    model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), **predict_params)
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
    load_s = round(time.perf_counter() - t_load0, 3)

    load_meta = {
        "model_key": "yolo11n",
        "weights_path": str(weights_path),
        "weights_sha256": actual_w_sha,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "load_and_warmup_seconds": load_s,
        "wacv_classes_discarded": True,
        "wacv_to_fics_mapping_used": False,
        "frozen_operating_point": {
            "imgsz": imgsz,
            "conf": conf_thresh,
            "tile_nms_iou": tile_nms_iou,
            "tile_agnostic_nms": True,
            "max_det_per_tile": max_det,
            "cross_tile_class_agnostic_nms_iou": cross_tile_nms_iou,
            "tie_break_rule": "(-score, tile_id, tile_box_index)",
            "primary_top_k": primary_top_k,
            "diagnostic_tiers": ["top_100", "top_200", "all_post_nms_conf005", "pre_nms_all"],
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "worker_load_meta.json").write_text(
        json.dumps(load_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    shapes: List[List[int]] = []
    hook_handle = model.predictor.model.model.register_forward_pre_hook(
        lambda module, args: shapes.append(list(args[0].shape))
    )

    raw_jsonl_path = run_dir / "raw_outputs_all.jsonl"
    tile_jsonl_path = prop_dir / "tile_predictions.jsonl"
    pre_nms_proposals: List[Dict[str, Any]] = []
    cap_hits = 0

    try:
        with raw_jsonl_path.open("w", encoding="utf-8") as raw_fh, tile_jsonl_path.open("w", encoding="utf-8") as tile_fh:
            for item in queue_items:
                tile_p = pathlib.Path(item["path"])
                actual_tile_sha = file_sha256(tile_p).lower()
                exp_tile_sha = str(item["sha256"]).lower()
                if actual_tile_sha != exp_tile_sha:
                    raise ValueError(f"Tile SHA-256 mismatch for {tile_p}: {actual_tile_sha} != {exp_tile_sha}")

                with Image.open(tile_p) as im_raw:
                    pil_img = im_raw.convert("RGB")
                tw, th = pil_img.size
                if (tw, th) != (int(item["width"]), int(item["height"])):
                    raise ValueError(f"Tile size mismatch for {tile_p}: {(tw, th)} != {(item['width'], item['height'])}")

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(0)
                    torch.cuda.synchronize(0)
                t_call0 = time.perf_counter()
                res = model.predict(pil_img, **predict_params)[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize(0)
                gpu_ms = round((time.perf_counter() - t_call0) * 1000.0, 2)
                peak_mib = round(torch.cuda.max_memory_reserved(0) / (1024.0 * 1024.0), 2) if torch.cuda.is_available() else 0.0

                # Extract ONLY boxes and confidence scores; discard res.boxes.cls (WACV classes) completely
                boxes_xyxy = res.boxes.xyxy.cpu().tolist()
                scores = res.boxes.conf.cpu().tolist()
                if len(boxes_xyxy) >= max_det:
                    cap_hits += 1

                # Deterministic per-tile ordering & validation
                raw_tile_candidates = []
                for b_idx, (bx, sc) in enumerate(zip(boxes_xyxy, scores)):
                    x1 = max(0.0, min(float(tw), float(bx[0])))
                    y1 = max(0.0, min(float(th), float(bx[1])))
                    x2 = max(0.0, min(float(tw), float(bx[2])))
                    y2 = max(0.0, min(float(th), float(bx[3])))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    raw_tile_candidates.append({
                        "orig_idx": b_idx,
                        "local_bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "score": round(float(sc), 6),
                    })

                raw_tile_candidates.sort(key=lambda r: (-float(r["score"]), int(r["orig_idx"])))
                tile_objs: List[Dict[str, Any]] = []
                for r_cand in raw_tile_candidates:
                    if any(_compute_iou_xyxy(r_cand["local_bbox_xyxy"], k_obj["local_bbox_xyxy"]) >= tile_nms_iou for k_obj in tile_objs):
                        continue
                    t_box_idx = len(tile_objs)
                    tile_objs.append({
                        "tile_box_index": t_box_idx,
                        "label": "proposal",
                        "local_bbox_xyxy": r_cand["local_bbox_xyxy"],
                        "score": r_cand["score"],
                    })

                tid = int(item["tile_id"])
                cid = str(item["call_id"])
                off_x = int(item["roi_offset_x"]) + int(item["tile_offset_x"])
                off_y = int(item["roi_offset_y"]) + int(item["tile_offset_y"])

                for t_obj in tile_objs:
                    gbox = map_tile_box_to_global(
                        t_obj["local_bbox_xyxy"],
                        offset_x=off_x,
                        offset_y=off_y,
                        img_w=orig_w,
                        img_h=orig_h,
                    )
                    if gbox[2] <= gbox[0] or gbox[3] <= gbox[1]:
                        continue
                    pre_idx = len(pre_nms_proposals)
                    pre_nms_proposals.append({
                        "pre_nms_id": f"{board_id}_yolo_raw{pre_idx:04d}",
                        "board_id": board_id,
                        "image_id": int(item.get("image_id", 1)),
                        "tile_id": tid,
                        "call_id": cid,
                        "tile_box_index": int(t_obj["tile_box_index"]),
                        "label": "proposal",
                        "score": float(t_obj["score"]),
                        "local_bbox_xyxy": t_obj["local_bbox_xyxy"],
                        "bbox_xyxy": gbox,
                    })

                rec = {
                    "call_id": cid,
                    "call_index": int(item["call_index"]),
                    "board_id": board_id,
                    "image_id": int(item.get("image_id", 1)),
                    "mode": "YOLO_T_LOC",
                    "tile_id": tid,
                    "image_path": str(tile_p),
                    "input_sha256": actual_tile_sha,
                    "expected_sha256": exp_tile_sha,
                    "width": tw,
                    "height": th,
                    "orig_input_width": tw,
                    "orig_input_height": th,
                    "roi_offset_x": int(item["roi_offset_x"]),
                    "roi_offset_y": int(item["roi_offset_y"]),
                    "tile_offset_x": int(item["tile_offset_x"]),
                    "tile_offset_y": int(item["tile_offset_y"]),
                    "actual_input_shape": shapes[-1] if shapes else [1, 3, imgsz, imgsz],
                    "proposal_objects": tile_objs,
                    "parsed_count": len(tile_objs),
                    "raw_output": json.dumps(tile_objs, ensure_ascii=False),
                    "gpu_generate_ms": gpu_ms,
                    "native_speed_ms": dict(res.speed) if hasattr(res, "speed") and res.speed else {},
                    "cuda_max_reserved_mib": peak_mib,
                    "cap_hit": len(boxes_xyxy) >= max_det,
                    "ok": True,
                    "is_oom": False,
                    "is_timeout": False,
                    "watchdog_killed": False,
                }
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                raw_fh.write(line)
                raw_fh.flush()
                tile_fh.write(line)
                tile_fh.flush()
    finally:
        hook_handle.remove()

    all_post_nms = class_agnostic_nms_with_dedup_trace(
        pre_nms_proposals,
        iou_threshold=cross_tile_nms_iou,
        board_id=board_id,
    )
    top100 = list(all_post_nms[:100])
    top200 = list(all_post_nms[:primary_top_k])

    pre_p = prop_dir / "proposals_pre_nms.json"
    all_p = prop_dir / "proposals_all_post_nms.json"
    t100_p = prop_dir / "proposals_top100.json"
    t200_p = prop_dir / "proposals_top200.json"

    pre_p.write_text(json.dumps(pre_nms_proposals, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    all_p.write_text(json.dumps(all_post_nms, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    t100_p.write_text(json.dumps(top100, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    t200_p.write_text(json.dumps(top200, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stage_hashes = {
        "board_id": board_id,
        "weights_sha256": actual_w_sha,
        "tile_predictions_jsonl_sha256": file_sha256(tile_jsonl_path).lower(),
        "proposals_pre_nms_sha256": file_sha256(pre_p).lower(),
        "proposals_all_post_nms_sha256": file_sha256(all_p).lower(),
        "proposals_top100_sha256": file_sha256(t100_p).lower(),
        "proposals_top200_sha256": file_sha256(t200_p).lower(),
        "counts": {
            "tiles": len(queue_items),
            "pre_nms": len(pre_nms_proposals),
            "all_post_nms_conf005": len(all_post_nms),
            "top_100": len(top100),
            "top_200_primary": len(top200),
            "cap_hits_at_3000": cap_hits,
        },
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }
    (prop_dir / "proposal_stage_hashes.json").write_text(
        json.dumps(stage_hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (run_dir / "session_summary.json").write_text(
        json.dumps({"stage_status": "completed", **stage_hashes}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"stage_status": "completed", **stage_hashes}
