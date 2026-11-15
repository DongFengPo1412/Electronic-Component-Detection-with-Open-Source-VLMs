"""GT-free end-to-end orchestration helpers: ROI/tiling, prediction reconstruction, verification, and B_other export.

This thin module composes frozen repository primitives (`tiling`, `fics_replay`,
`candidate_verifier`) strictly without reading ground-truth annotations so both
the final held-out evaluation and the upcoming interactive demo can share one
verified GT-free code path.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

from pcb_vlm.data.fics_dslr import file_sha256
from pcb_vlm.evaluation.candidate_verifier import (
    ALLOWED_SEVEN_LABELS,
    TARGET_SIX_CLASSES,
    apply_verifier_filter_policy,
    build_candidate_verifier_manifest,
    parse_single_label_json,
    verify_verifier_manifest_integrity,
)
from pcb_vlm.evaluation.fics_replay import (
    FICS_SIX_CLASSES,
    classify_call_termination_and_parse_v2,
)
from pcb_vlm.geometry.tiling import (
    generate_tiles,
    map_tile_box_to_global,
    nms_merge_predictions,
)


def compute_board_roi_and_tiles(
    img_path: pathlib.Path,
    *,
    roi_pad: int = 32,
    white_thresh: int = 248,
    tile_size: int = 1024,
    overlap_ratio: float = 0.20,
    export_dir: Optional[pathlib.Path] = None,
    board_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute any(RGB < 248)+32px ROI and 1024px overlapped tiles, optionally saving PNGs."""
    src_path = pathlib.Path(img_path)
    with Image.open(src_path) as raw_im:
        im_rgb = ImageOps.exif_transpose(raw_im).convert("RGB")
        orig_w, orig_h = im_rgb.size
        arr = np.asarray(im_rgb)

    mask = np.any(arr < int(white_thresh), axis=2)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        x1, y1, x2, y2 = 0, 0, orig_w, orig_h
    else:
        x1 = max(0, int(xs.min()) - int(roi_pad))
        y1 = max(0, int(ys.min()) - int(roi_pad))
        x2 = min(orig_w, int(xs.max()) + 1 + int(roi_pad))
        y2 = min(orig_h, int(ys.max()) + 1 + int(roi_pad))

    roi_w = x2 - x1
    roi_h = y2 - y1
    tiles = generate_tiles(roi_w, roi_h, tile_size=int(tile_size), overlap_ratio=float(overlap_ratio))

    bname = board_key or src_path.stem
    roi_info: Optional[Dict[str, Any]] = None
    tile_entries: List[Dict[str, Any]] = []

    if export_dir is not None:
        out_root = pathlib.Path(export_dir)
        tiles_dir = out_root / "tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        roi_im = im_rgb.crop((x1, y1, x2, y2))
        roi_path = out_root / f"{bname}_roi.png"
        roi_im.save(roi_path)
        roi_info = {
            "path": str(roi_path),
            "sha256": file_sha256(roi_path),
            "width": roi_w,
            "height": roi_h,
        }
        for t in tiles:
            tid = int(t["tile_id"])
            tx1, ty1, tx2, ty2 = t["bbox"]
            tile_im = roi_im.crop((tx1, ty1, tx2, ty2))
            tile_path = tiles_dir / f"tile_{tid:02d}.png"
            tile_im.save(tile_path)
            tile_entries.append({
                "tile_id": tid,
                "bbox_roi_xyxy": [int(tx1), int(ty1), int(tx2), int(ty2)],
                "tile_offset_x": int(tx1),
                "tile_offset_y": int(ty1),
                "width": int(tx2 - tx1),
                "height": int(ty2 - ty1),
                "path": str(tile_path),
                "sha256": file_sha256(tile_path),
            })
    else:
        for t in tiles:
            tx1, ty1, tx2, ty2 = t["bbox"]
            tile_entries.append({
                "tile_id": int(t["tile_id"]),
                "bbox_roi_xyxy": [int(tx1), int(ty1), int(tx2), int(ty2)],
                "tile_offset_x": int(tx1),
                "tile_offset_y": int(ty1),
                "width": int(tx2 - tx1),
                "height": int(ty2 - ty1),
            })

    return {
        "image_path": str(src_path),
        "orig_wh": [orig_w, orig_h],
        "roi_bbox_half_open": [x1, y1, x2, y2],
        "roi_offset_x": x1,
        "roi_offset_y": y1,
        "roi_width": roi_w,
        "roi_height": roi_h,
        "tile_count": len(tile_entries),
        "roi_export": roi_info,
        "tiles": tile_entries,
    }


def load_jsonl_records(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read non-empty JSONL lines from disk preserving file order."""
    rows: List[Dict[str, Any]] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def reconstruct_board_predictions_gt_free(
    *,
    board_id: str,
    image_id: int,
    orig_wh: Tuple[int, int],
    raw_records: Sequence[Mapping[str, Any]],
    mode: str = "T",
    expected_call_ids: Optional[Sequence[str]] = None,
    eos_token_ids: Sequence[int] = (151645, 151643),
    max_new_tokens: int = 512,
    nms_iou_threshold: float = 0.50,
    allowed_classes: Sequence[str] = FICS_SIX_CLASSES,
) -> Dict[str, Any]:
    """Reconstruct global board predictions from raw JSONL records without any GT.

    Applies cross-tile NMS only for tiled (`T`) mode; whole-ROI (`W`) mode preserves
    raw decoded detections identically to the formal baseline replay.
    """
    if str(mode) not in ("W", "T"):
        raise ValueError(f"Unsupported detection mode: {mode!r}; expected 'W' or 'T'")

    unit_rows = [
        dict(r) for r in raw_records
        if str(r.get("board_id")) == str(board_id) and str(r.get("mode")) == str(mode)
    ]
    duplicate_call_ids: List[str] = []
    unexpected_call_ids: List[str] = []
    if expected_call_ids is not None:
        expected_list = [str(cid) for cid in expected_call_ids]
        expected_set = set(expected_list)
        if len(expected_list) != len(expected_set):
            seen_exp: set[str] = set()
            for cid in expected_list:
                if cid in seen_exp and cid not in duplicate_call_ids:
                    duplicate_call_ids.append(cid)
                seen_exp.add(cid)
        by_cid: Dict[str, Dict[str, Any]] = {}
        for r in unit_rows:
            cid = str(r.get("call_id", ""))
            if cid not in expected_set:
                unexpected_call_ids.append(cid)
            elif cid in by_cid:
                if cid not in duplicate_call_ids:
                    duplicate_call_ids.append(cid)
            else:
                by_cid[cid] = r
        missing = [cid for cid in expected_list if cid not in by_cid]
        ordered = [by_cid[cid] for cid in expected_list if cid in by_cid]
        planned_calls = len(expected_list)
    else:
        missing = []
        seen_cids: set[str] = set()
        for r in unit_rows:
            cid = str(r.get("call_id", ""))
            if cid in seen_cids and cid not in duplicate_call_ids:
                duplicate_call_ids.append(cid)
            seen_cids.add(cid)
        ordered = sorted(unit_rows, key=lambda r: (int(r.get("call_index", 0)), str(r.get("call_id", ""))))
        planned_calls = len(ordered)

    failed = [
        str(r.get("call_id"))
        for r in ordered
        if (not r.get("ok")) or r.get("is_oom") or r.get("is_timeout") or r.get("watchdog_killed")
    ]
    if missing or failed or duplicate_call_ids or unexpected_call_ids or not ordered:
        return {
            "board_id": str(board_id),
            "image_id": int(image_id),
            "mode": str(mode),
            "technical_complete": False,
            "missing_call_ids": missing,
            "failed_call_ids": failed,
            "duplicate_call_ids": duplicate_call_ids,
            "unexpected_call_ids": unexpected_call_ids,
            "planned_calls": planned_calls,
            "completed_calls": len(ordered) - len(failed),
            "predictions": [],
        }

    orig_w, orig_h = int(orig_wh[0]), int(orig_wh[1])
    raw_global_preds: List[Dict[str, Any]] = []
    for rec in ordered:
        if "raw_output" in rec and rec["raw_output"] is not None:
            raw_text = str(rec["raw_output"])
        elif "raw_text" in rec and rec["raw_text"] is not None:
            raw_text = str(rec["raw_text"])
        else:
            raise KeyError(f"Record {rec.get('call_id')!r} missing 'raw_output'/'raw_text'")

        in_w = rec["orig_input_width"] if rec.get("orig_input_width") is not None else rec["width"]
        in_h = rec["orig_input_height"] if rec.get("orig_input_height") is not None else rec["height"]

        v2 = classify_call_termination_and_parse_v2(
            raw_text=raw_text,
            width=int(in_w),
            height=int(in_h),
            generated_tokens=rec.get("generated_tokens"),
            max_new_tokens=int(rec.get("max_new_tokens") or max_new_tokens),
            generated_token_ids=rec.get("generated_token_ids"),
            eos_token_ids=eos_token_ids,
            protocol="xyxy_norm1000",
            allowed_classes=allowed_classes,
        )
        off_x = int(rec["tile_offset_x"]) + int(rec["roi_offset_x"])
        off_y = int(rec["tile_offset_y"]) + int(rec["roi_offset_y"])
        for obj in v2["objects"]:
            gbox = map_tile_box_to_global(
                obj["bbox_xyxy"],
                offset_x=off_x,
                offset_y=off_y,
                img_w=orig_w,
                img_h=orig_h,
            )
            raw_global_preds.append({
                "board_id": str(board_id),
                "image_id": int(image_id),
                "label": obj["label"],
                "bbox_xyxy": gbox,
                "score": 1.0,
            })

    if str(mode) == "T":
        after_merge = nms_merge_predictions(raw_global_preds, iou_threshold=float(nms_iou_threshold))
        nms_applied = True
    else:
        after_merge = list(raw_global_preds)
        nms_applied = False

    structured = [
        {
            "prediction_id": f"{board_id}_{mode}_p{idx:03d}",
            "board_id": str(board_id),
            "image_id": int(image_id),
            "orig_order": idx,
            "label": p["label"],
            "bbox_xyxy": p["bbox_xyxy"],
            "score": float(p.get("score", 1.0)),
        }
        for idx, p in enumerate(after_merge)
    ]

    return {
        "board_id": str(board_id),
        "image_id": int(image_id),
        "mode": str(mode),
        "nms_applied": nms_applied,
        "technical_complete": True,
        "planned_calls": len(ordered),
        "completed_calls": len(ordered),
        "raw_global_predictions": len(raw_global_preds),
        "post_nms_predictions": len(structured),
        "predictions": structured,
    }


def prepare_verifier_manifest_gt_free(
    *,
    reconstructed_by_board: Sequence[Mapping[str, Any]],
    board_image_paths: Mapping[str, pathlib.Path],
    crops_dir: pathlib.Path,
    inference_spec: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    pad_fraction: float = 0.10,
    min_edge_upscale: int = 224,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build and integrity-verify a neutral GT-free verifier manifest from reconstructed predictions."""
    all_preds: List[Dict[str, Any]] = []
    for unit in reconstructed_by_board:
        if not unit.get("technical_complete"):
            raise RuntimeError(
                f"Cannot prepare verifier manifest for incomplete board {unit.get('board_id')!r}: "
                f"missing={unit.get('missing_call_ids')}, failed={unit.get('failed_call_ids')}"
            )
        all_preds.extend(dict(p) for p in unit.get("predictions") or [])

    manifest = build_candidate_verifier_manifest(
        source_predictions=all_preds,
        board_image_paths=board_image_paths,
        crops_dir=pathlib.Path(crops_dir),
        inference_spec=inference_spec,
        source_provenance=source_provenance,
        pad_fraction=float(pad_fraction),
        min_edge_upscale=int(min_edge_upscale),
    )
    integrity = verify_verifier_manifest_integrity(manifest)
    return manifest, integrity


def export_verified_predictions_gt_free(
    *,
    manifest: Mapping[str, Any],
    verifier_rows: Sequence[Mapping[str, Any]],
    allowed_labels: Sequence[str] = ALLOWED_SEVEN_LABELS,
    target_classes: Sequence[str] = TARGET_SIX_CLASSES,
    policy: str = "B_other",
    allow_incomplete: bool = False,
    output_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Apply frozen verifier filter policy cleanly without GT and return per-board kept objects."""
    expected_call_ids = {
        str(p.get("unique_call_id") or p.get("call_id"))
        for p in manifest["predictions"]
        if (p.get("unique_call_id") or p.get("call_id"))
    }
    duplicate_call_ids: List[str] = []
    unexpected_call_ids: List[str] = []
    records_by_cid: Dict[str, Dict[str, Any]] = {}
    for row in verifier_rows:
        cid = str(row["call_id"])
        if cid not in expected_call_ids:
            unexpected_call_ids.append(cid)
            continue
        if cid in records_by_cid:
            if cid not in duplicate_call_ids:
                duplicate_call_ids.append(cid)
            continue
        rec = dict(row)
        if rec.get("ok") and ("strict_json_ok" not in rec or "strict_label" not in rec):
            v_txt = rec.get("raw_text") if rec.get("raw_text") is not None else rec.get("raw_output")
            parsed = parse_single_label_json(str(v_txt or ""), allowed_labels=allowed_labels)
            rec["strict_json_ok"] = parsed["strict_json_ok"]
            rec["strict_label"] = parsed["strict_label"]
            rec["compat_regex_label"] = parsed["compat_regex_label"]
            rec["parse_error"] = parsed["parse_error"]
        records_by_cid[cid] = rec

    filtered = apply_verifier_filter_policy(
        predictions=manifest["predictions"],
        verification_records_by_call_id=records_by_cid,
        policy=policy,
        target_classes=target_classes,
    )
    if (not filtered["technical_complete"] or duplicate_call_ids or unexpected_call_ids) and not allow_incomplete:
        raise RuntimeError(
            "Verifier outputs are technically incomplete or contain duplicate/unexpected call IDs; "
            "refusing silent background rejection: "
            + json.dumps(
                {
                    "counts": filtered["counts"],
                    "duplicate_call_ids": duplicate_call_ids,
                    "unexpected_call_ids": unexpected_call_ids,
                },
                ensure_ascii=False,
            )
        )

    preds_list = list(manifest["predictions"])
    img_id_by_board = {str(p["board_id"]): p.get("image_id") for p in preds_list}
    decisions = list(filtered["decisions"])
    kept_by_board = filtered["kept_predictions_by_board"]

    board_ids = sorted(set(img_id_by_board.keys()) | set(kept_by_board.keys()))
    by_board: Dict[str, Dict[str, Any]] = {}
    for bid in board_ids:
        b_decs = [d for d in decisions if str(d["board_id"]) == bid]
        b_kept = list(kept_by_board.get(bid, []))
        image_id = img_id_by_board.get(bid)
        by_board[bid] = {
            "board_id": bid,
            "image_id": image_id,
            "total_candidates": len(b_decs),
            "kept_count": len(b_kept),
            "rejected_count": len(b_decs) - len(b_kept),
            "pred_record": {
                "image_id": image_id,
                "objects": b_kept,
            },
            "decisions": b_decs,
        }

    payload: Dict[str, Any] = {
        "schema_version": "gt_free_verified_predictions_v1",
        "policy": policy,
        "technical_complete": bool(
            filtered["technical_complete"] and not duplicate_call_ids and not unexpected_call_ids
        ),
        "summary": filtered["counts"],
        "per_board": by_board,
    }
    if output_path is not None:
        out_p = pathlib.Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
