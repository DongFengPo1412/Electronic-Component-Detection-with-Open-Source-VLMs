"""Offline overlapping-tile consistency false-positive filtering (Candidate A).

Design guarantees:
1. Operates strictly on model `T` (tiled) raw predictions and tile geometry; never reads GT
   annotations when deciding whether to keep or filter any prediction or group.
2. Preserves pre-NMS per-call/per-tile provenance (`call_id`, `tile_id`, `pred_idx`) and
   counts distinct supporting `tile_id`s per deterministic group (never counting duplicate
   predictions inside the same tile as multi-view support).
3. Groups same-board, same-class raw global predictions using a deterministic total order
   and a frozen group representative box (`rep_box`) so cluster formation is invariant to
   input row traversal order.
4. Counts opportunity tiles strictly from complete frozen tile geometry (tiles whose global
   rectangle completely contains `rep_box`). Only filters groups when
   `opportunity_count >= min_opportunity_tiles` and `support_count < min_support_tiles`;
   single-coverage boxes (`opportunity_count == 1`) and oversized boxes (`opportunity_count == 0`)
   are preserved unchanged.
5. Does not apply cross-class suppression or rewrite labels; after filtering, runs standard
   per-board class-aware NMS (`iou_threshold=0.50`, `score=1.0`).
6. Refuses to fabricate votes when required `tile_id` or tile geometry is missing/invalid.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pcb_vlm.geometry.box_ops import compute_iou
from pcb_vlm.geometry.tiling import (
    generate_tiles,
    map_tile_box_to_global,
    nms_merge_predictions,
)
from pcb_vlm.evaluation.fics_replay import (
    FICS_SIX_CLASSES,
    classify_call_termination_and_parse_v2,
    load_fics_six_class_config,
    repo_eval,
    validate_coco_schema_and_categories,
)


DEFAULT_METHOD_A_SPEC: Dict[str, Any] = {
    "method_id": "candidate_A_tile_consistency_v1",
    "description": (
        "GT-free offline overlapping-tile consistency filter for tiled (T) VLM predictions "
        "with deterministic frozen representative-box grouping and opportunity-aware gating."
    ),
    "algorithm": {
        "group_scope": "same_board_same_class",
        "group_iou_threshold": 0.50,
        "deterministic_sort_key": "(-area, x1, y1, x2, y2, tile_id, call_id, pred_idx)",
        "representative_box_policy": "frozen_first_seed_box_in_group",
        "support_metric": "distinct_tile_ids_in_group",
        "opportunity_metric": "distinct_frozen_tiles_completely_containing_rep_box",
        "containment_eps": 1e-6,
        "filter_condition": "opportunity_count >= min_opportunity_tiles and support_count < min_support_tiles",
        "min_opportunity_tiles": 2,
        "min_support_tiles": 2,
        "oversized_or_single_coverage_policy": "keep",
        "cross_class_suppression": False,
        "label_rewrite": False,
        "post_filter_nms_iou_threshold": 0.50,
        "post_filter_score": 1.0,
        "missing_tile_provenance_policy": "mark_unusable_and_refuse",
    },
    "coordinate_system": {
        "global_box_format": "xyxy_pixels_clipped_to_full_board_image",
        "tile_offset_formula": "tile_global_xy = (roi_offset_x + tile_offset_x, roi_offset_y + tile_offset_y)",
    },
}


def _is_int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _merge_spec(spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "method_id": DEFAULT_METHOD_A_SPEC["method_id"],
        "description": DEFAULT_METHOD_A_SPEC["description"],
        "algorithm": dict(DEFAULT_METHOD_A_SPEC["algorithm"]),
        "coordinate_system": dict(DEFAULT_METHOD_A_SPEC["coordinate_system"]),
    }
    if not spec:
        return out
    if "algorithm" in spec and isinstance(spec["algorithm"], dict):
        out["algorithm"].update(spec["algorithm"])
    for key in (
        "group_iou_threshold",
        "min_opportunity_tiles",
        "min_support_tiles",
        "containment_eps",
        "post_filter_nms_iou_threshold",
        "post_filter_score",
    ):
        if key in spec:
            out["algorithm"][key] = spec[key]
    return out


def validate_and_build_tile_geometry(
    t_records: Sequence[Dict[str, Any]],
    *,
    expected_roi_size: Optional[Tuple[int, int]] = None,
    tile_size: int = 1024,
    overlap_ratio: float = 0.20,
) -> Tuple[bool, Optional[str], Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate T-call provenance and construct global tile geometry without fabricating votes.

    Returns `(usable, unusable_reason, tiles_by_id, ordered_records)`.
    """
    if not t_records:
        return False, "missing_t_records", {}, []

    seen_tiles: Dict[int, Dict[str, Any]] = {}
    records_by_tile: Dict[int, Dict[str, Any]] = {}
    roi_offsets: Optional[Tuple[int, int]] = None

    for rec in t_records:
        cid = rec.get("call_id")
        if not isinstance(cid, str) or not cid.strip():
            return False, "record_missing_call_id", {}, []
        if str(rec.get("mode")) != "T":
            return False, f"non_tile_mode_record:{rec.get('mode')!r}", {}, []
        if (not rec.get("ok")) or rec.get("is_oom") or rec.get("is_timeout") or rec.get("watchdog_killed"):
            return False, f"incomplete_or_failed_call:{cid}", {}, []

        tid = rec.get("tile_id")
        if not _is_int_not_bool(tid) or tid < 0:
            return False, f"missing_or_invalid_tile_id:{cid}", {}, []

        req_ints = ("roi_offset_x", "roi_offset_y", "tile_offset_x", "tile_offset_y", "orig_input_width", "orig_input_height")
        for key in req_ints:
            if not _is_int_not_bool(rec.get(key)):
                return False, f"missing_or_nonint_{key}:{cid}", {}, []

        tw = int(rec["orig_input_width"])
        th = int(rec["orig_input_height"])
        if tw <= 0 or th <= 0:
            return False, f"nonpositive_tile_dims:{cid}", {}, []

        rox = int(rec["roi_offset_x"])
        roy = int(rec["roi_offset_y"])
        if rox < 0 or roy < 0 or int(rec["tile_offset_x"]) < 0 or int(rec["tile_offset_y"]) < 0:
            return False, f"negative_offset:{cid}", {}, []

        if roi_offsets is None:
            roi_offsets = (rox, roy)
        elif roi_offsets != (rox, roy):
            return False, f"inconsistent_roi_offsets:{roi_offsets}_vs_{(rox, roy)}", {}, []

        if tid in seen_tiles:
            return False, f"duplicate_tile_id:{tid}", {}, []

        gx1 = rox + int(rec["tile_offset_x"])
        gy1 = roy + int(rec["tile_offset_y"])
        seen_tiles[tid] = {
            "tile_id": tid,
            "call_id": cid,
            "tile_offset_x": int(rec["tile_offset_x"]),
            "tile_offset_y": int(rec["tile_offset_y"]),
            "roi_offset_x": rox,
            "roi_offset_y": roy,
            "width": tw,
            "height": th,
            "bbox": [float(gx1), float(gy1), float(gx1 + tw), float(gy1 + th)],
        }
        records_by_tile[tid] = rec

    expected_n = len(seen_tiles)
    if set(seen_tiles.keys()) != set(range(expected_n)):
        return False, f"non_contiguous_tile_ids:{sorted(seen_tiles.keys())}", {}, []

    if expected_roi_size is not None:
        roi_w, roi_h = int(expected_roi_size[0]), int(expected_roi_size[1])
        expected_tiles = generate_tiles(roi_w, roi_h, tile_size=tile_size, overlap_ratio=overlap_ratio)
        if len(expected_tiles) != len(seen_tiles):
            return (
                False,
                f"tile_count_mismatch:observed={len(seen_tiles)}_expected={len(expected_tiles)}",
                {},
                [],
            )
        for exp in expected_tiles:
            tid = int(exp["tile_id"])
            obs = seen_tiles.get(tid)
            if obs is None:
                return False, f"missing_expected_tile_id:{tid}", {}, []
            if (
                obs["tile_offset_x"] != int(exp["offset_x"])
                or obs["tile_offset_y"] != int(exp["offset_y"])
                or obs["width"] != int(exp["width"])
                or obs["height"] != int(exp["height"])
            ):
                return False, f"tile_geometry_mismatch:tile_{tid}", {}, []

    ordered_records = [records_by_tile[tid] for tid in sorted(seen_tiles.keys())]
    return True, None, seen_tiles, ordered_records


def box_completely_contained_in_tile(
    box_xyxy: Sequence[float],
    tile_bbox_xyxy: Sequence[float],
    *,
    eps: float = 1e-6,
) -> bool:
    """Return True iff `box_xyxy` lies completely within `tile_bbox_xyxy` (within `eps`)."""
    bx1, by1, bx2, by2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])
    tx1, ty1, tx2, ty2 = (
        float(tile_bbox_xyxy[0]),
        float(tile_bbox_xyxy[1]),
        float(tile_bbox_xyxy[2]),
        float(tile_bbox_xyxy[3]),
    )
    if bx2 <= bx1 or by2 <= by1:
        return False
    return (bx1 >= tx1 - eps) and (by1 >= ty1 - eps) and (bx2 <= tx2 + eps) and (by2 <= ty2 + eps)


def compute_opportunity_tiles(
    rep_box_xyxy: Sequence[float],
    tiles_by_id: Dict[int, Dict[str, Any]],
    *,
    eps: float = 1e-6,
) -> List[int]:
    """Return sorted `tile_id`s from complete tile geometry that completely contain `rep_box_xyxy`."""
    return sorted(
        tid
        for tid, tile in tiles_by_id.items()
        if box_completely_contained_in_tile(rep_box_xyxy, tile["bbox"], eps=eps)
    )


def _ensure_parsed_v2(
    rec: Dict[str, Any],
    *,
    max_new_tokens: int = 512,
    eos_token_ids: Sequence[int] = (151645, 151643),
    protocol: str = "xyxy_norm1000",
    allowed_classes: Sequence[str] = FICS_SIX_CLASSES,
) -> Dict[str, Any]:
    if isinstance(rec.get("v2"), dict) and isinstance(rec["v2"].get("objects"), list):
        return rec["v2"]
    return classify_call_termination_and_parse_v2(
        raw_text=str(rec.get("raw_output") or ""),
        width=int(rec["orig_input_width"]),
        height=int(rec["orig_input_height"]),
        generated_tokens=rec.get("generated_tokens"),
        max_new_tokens=int(rec.get("max_new_tokens") or max_new_tokens),
        generated_token_ids=rec.get("generated_token_ids"),
        eos_token_ids=eos_token_ids,
        protocol=str(rec.get("coordinate_protocol") or protocol),
        allowed_classes=allowed_classes,
    )


def filter_board_raw_predictions_by_tile_consistency(
    t_records: Sequence[Dict[str, Any]],
    *,
    board_id: str,
    img_width: int,
    img_height: int,
    expected_roi_size: Optional[Tuple[int, int]] = None,
    tile_size: int = 1024,
    overlap_ratio: float = 0.20,
    spec: Optional[Dict[str, Any]] = None,
    eos_token_ids: Sequence[int] = (151645, 151643),
    max_new_tokens: int = 512,
    protocol: str = "xyxy_norm1000",
    allowed_classes: Sequence[str] = FICS_SIX_CLASSES,
) -> Dict[str, Any]:
    """Run GT-free Candidate A consistency filtering on a single board's `T` call records.

    This function accepts NO ground-truth annotations and returns both baseline (unfiltered)
    post-NMS predictions and Candidate A filtered post-NMS predictions plus group audit data.
    """
    t0 = time.perf_counter()
    merged_spec = _merge_spec(spec)
    algo = merged_spec["algorithm"]
    group_iou = float(algo["group_iou_threshold"])
    nms_iou = float(algo["post_filter_nms_iou_threshold"])
    min_opp = int(algo["min_opportunity_tiles"])
    min_sup = int(algo["min_support_tiles"])
    eps = float(algo["containment_eps"])
    score_val = float(algo["post_filter_score"])

    usable, reason, tiles_by_id, ordered_records = validate_and_build_tile_geometry(
        t_records,
        expected_roi_size=expected_roi_size,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
    )
    if not usable:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "board_id": board_id,
            "usable": False,
            "unusable_reason": reason,
            "cpu_overhead_ms": round(elapsed_ms, 3),
        }

    raw_items: List[Dict[str, Any]] = []
    any_len_trunc = False
    has_no_objects_call = False
    for rec in ordered_records:
        v2 = _ensure_parsed_v2(
            rec,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            protocol=protocol,
            allowed_classes=allowed_classes,
        )
        if v2.get("length_limited_without_eos"):
            any_len_trunc = True
        if v2.get("status") == "no_objects_detected":
            has_no_objects_call = True
        tid = int(rec["tile_id"])
        cid = str(rec["call_id"])
        ox = int(rec["tile_offset_x"]) + int(rec["roi_offset_x"])
        oy = int(rec["tile_offset_y"]) + int(rec["roi_offset_y"])
        for pred_idx, obj in enumerate(v2["objects"]):
            gbox = map_tile_box_to_global(
                obj["bbox_xyxy"],
                offset_x=ox,
                offset_y=oy,
                img_w=int(img_width),
                img_h=int(img_height),
            )
            x1, y1, x2, y2 = float(gbox[0]), float(gbox[1]), float(gbox[2]), float(gbox[3])
            if x2 <= x1 or y2 <= y1:
                continue
            raw_items.append({
                "raw_idx": len(raw_items),
                "board_id": board_id,
                "label": str(obj["label"]).strip().lower(),
                "bbox_xyxy": [x1, y1, x2, y2],
                "area": (x2 - x1) * (y2 - y1),
                "tile_id": tid,
                "call_id": cid,
                "pred_idx": int(pred_idx),
                "score": score_val,
            })

    # Group deterministically per class, completely independent of input row traversal order.
    by_class: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in raw_items:
        by_class[item["label"]].append(item)

    groups: List[Dict[str, Any]] = []
    kept_raw_indices: set[int] = set()
    opp_hist: collections.Counter[int] = collections.Counter()
    sup_hist: collections.Counter[int] = collections.Counter()

    group_counter = 0
    for cls_name in sorted(by_class.keys()):
        cls_items = sorted(
            by_class[cls_name],
            key=lambda it: (
                -float(it["area"]),
                float(it["bbox_xyxy"][0]),
                float(it["bbox_xyxy"][1]),
                float(it["bbox_xyxy"][2]),
                float(it["bbox_xyxy"][3]),
                int(it["tile_id"]),
                str(it["call_id"]),
                int(it["pred_idx"]),
            ),
        )
        assigned = [False] * len(cls_items)
        for i, seed in enumerate(cls_items):
            if assigned[i]:
                continue
            assigned[i] = True
            rep_box = list(seed["bbox_xyxy"])
            members = [seed]
            for j in range(i + 1, len(cls_items)):
                if assigned[j]:
                    continue
                if compute_iou(cls_items[j]["bbox_xyxy"], rep_box) >= group_iou:
                    assigned[j] = True
                    members.append(cls_items[j])

            support_tiles = sorted({int(m["tile_id"]) for m in members})
            support_count = len(support_tiles)
            opp_tiles = compute_opportunity_tiles(rep_box, tiles_by_id, eps=eps)
            opp_count = len(opp_tiles)

            opp_hist[opp_count] += 1
            sup_hist[support_count] += 1

            is_filtered = (opp_count >= min_opp) and (support_count < min_sup)
            if not is_filtered:
                for m in members:
                    kept_raw_indices.add(int(m["raw_idx"]))

            if opp_count == 0:
                reason_tag = "kept_oversized_or_uncontained_opp0"
            elif opp_count < min_opp:
                reason_tag = "kept_single_coverage_opp1"
            elif not is_filtered:
                reason_tag = "kept_multi_tile_supported"
            else:
                reason_tag = "filtered_unconfirmed_in_overlap"

            groups.append({
                "group_id": f"{board_id}_g{group_counter:04d}",
                "board_id": board_id,
                "label": cls_name,
                "rep_box_xyxy": rep_box,
                "seed_tile_id": int(seed["tile_id"]),
                "seed_call_id": str(seed["call_id"]),
                "member_count": len(members),
                "support_tile_ids": support_tiles,
                "support_count": support_count,
                "opportunity_tile_ids": opp_tiles,
                "opportunity_count": opp_count,
                "filtered_out": bool(is_filtered),
                "decision_reason": reason_tag,
                "members": [
                    {
                        "call_id": m["call_id"],
                        "tile_id": m["tile_id"],
                        "pred_idx": m["pred_idx"],
                        "bbox_xyxy": m["bbox_xyxy"],
                    }
                    for m in members
                ],
            })
            group_counter += 1

    baseline_nms_in = [
        {"board_id": board_id, "label": it["label"], "bbox_xyxy": list(it["bbox_xyxy"]), "score": score_val}
        for it in raw_items
    ]
    filtered_nms_in = [
        {"board_id": board_id, "label": it["label"], "bbox_xyxy": list(it["bbox_xyxy"]), "score": score_val}
        for it in raw_items
        if it["raw_idx"] in kept_raw_indices
    ]

    baseline_post_nms = nms_merge_predictions(baseline_nms_in, iou_threshold=nms_iou)
    filtered_post_nms = nms_merge_predictions(filtered_nms_in, iou_threshold=nms_iou)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "board_id": board_id,
        "usable": True,
        "unusable_reason": None,
        "spec": merged_spec,
        "tile_count": len(tiles_by_id),
        "raw_prediction_count": len(raw_items),
        "kept_raw_prediction_count": len(filtered_nms_in),
        "baseline_post_nms_predictions": baseline_post_nms,
        "filtered_post_nms_predictions": filtered_post_nms,
        "group_count": len(groups),
        "filtered_group_count": sum(1 for g in groups if g["filtered_out"]),
        "kept_group_count": sum(1 for g in groups if not g["filtered_out"]),
        "opportunity_count_hist": {str(k): int(opp_hist[k]) for k in sorted(opp_hist.keys())},
        "support_count_hist": {str(k): int(sup_hist[k]) for k in sorted(sup_hist.keys())},
        "groups": groups,
        "any_length_truncated": any_len_trunc,
        "has_no_objects_call": has_no_objects_call,
        "cpu_overhead_ms": round(elapsed_ms, 3),
    }


def _match_predictions_to_gt_for_audit(
    gt_anns: Sequence[Dict[str, Any]],
    preds: Sequence[Dict[str, Any]],
    cat_id_to_name: Dict[int, str],
    *,
    iou_threshold: float = 0.50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute exact 1-to-1 TP/FP/FN details matching `scripts/evaluate.py` semantics for case audit."""
    gts_norm = []
    for ann in gt_anns:
        x, y, w, h = ann["bbox"]
        gts_norm.append({
            "ann_id": ann["id"],
            "label": cat_id_to_name[ann["category_id"]],
            "bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
            "matched": False,
        })

    # Sort preds identically to scripts/evaluate.py (-score descending when present; all 1.0 preserves order)
    ordered_preds = sorted(
        list(enumerate(preds)),
        key=lambda pair: float(pair[1].get("score", 1.0)),
        reverse=True,
    )

    tp_cases: List[Dict[str, Any]] = []
    fp_cases: List[Dict[str, Any]] = []
    for orig_idx, pred in ordered_preds:
        plbl = str(pred["label"]).strip().lower()
        pbox = [float(v) for v in pred["bbox_xyxy"]]
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, gt in enumerate(gts_norm):
            if gt["matched"] or gt["label"] != plbl:
                continue
            iou = compute_iou(pbox, gt["bbox_xyxy"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        if best_gt_idx >= 0:
            gts_norm[best_gt_idx]["matched"] = True
            tp_cases.append({
                "pred_idx": orig_idx,
                "label": plbl,
                "bbox_xyxy": pbox,
                "matched_gt_id": gts_norm[best_gt_idx]["ann_id"],
                "matched_gt_bbox_xyxy": gts_norm[best_gt_idx]["bbox_xyxy"],
                "iou": round(best_iou, 4),
            })
        else:
            # Find highest IoU with any GT (for diagnostic context)
            max_any_iou = 0.0
            nearest_gt_label = None
            for gt in gts_norm:
                val = compute_iou(pbox, gt["bbox_xyxy"])
                if val > max_any_iou:
                    max_any_iou = val
                    nearest_gt_label = gt["label"]
            fp_cases.append({
                "pred_idx": orig_idx,
                "label": plbl,
                "bbox_xyxy": pbox,
                "max_any_gt_iou": round(max_any_iou, 4),
                "nearest_gt_label": nearest_gt_label,
            })

    fn_cases = [gt for gt in gts_norm if not gt["matched"]]
    return tp_cases, fp_cases, fn_cases


def evaluate_tile_consistency_vs_baseline(
    *,
    board_records_by_board: Dict[str, List[Dict[str, Any]]],
    combined_coco: Dict[str, Any],
    spec: Optional[Dict[str, Any]] = None,
    roi_sizes_by_board: Optional[Dict[str, Tuple[int, int]]] = None,
    eos_token_ids: Sequence[int] = (151645, 151643),
    max_new_tokens: int = 512,
    tile_size: int = 1024,
    overlap_ratio: float = 0.20,
    eval_iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Evaluate Candidate A against baseline T across multiple boards (GT used only in final scoring)."""
    cfg = load_fics_six_class_config()
    allowed_classes = tuple(str(c["name"]).strip().lower() for c in cfg["classes"])
    board_img_map = validate_coco_schema_and_categories(combined_coco, expected_classes=cfg["classes"])
    cat_id_to_name = {int(c["id"]): str(c["name"]).strip().lower() for c in cfg["classes"]}

    per_board: Dict[str, Any] = {}
    micro_base = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0}
    micro_cand = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0}
    total_cpu_ms = 0.0

    for bid in sorted(board_records_by_board.keys()):
        if bid not in board_img_map:
            raise ValueError(f"Board {bid!r} missing from combined COCO images")
        img_meta = board_img_map[bid]
        coco_img_id = img_meta["id"]
        orig_w = int(img_meta["width"])
        orig_h = int(img_meta["height"])
        anns = [a for a in combined_coco.get("annotations", []) if str(a["image_id"]) == str(coco_img_id)]
        board_coco = {
            "images": [img_meta],
            "categories": list(combined_coco["categories"]),
            "annotations": anns,
        }

        roi_sz = roi_sizes_by_board.get(bid) if roi_sizes_by_board else None
        # Note: GT-free filter call strictly receives only records and image dimensions
        filt = filter_board_raw_predictions_by_tile_consistency(
            board_records_by_board[bid],
            board_id=bid,
            img_width=orig_w,
            img_height=orig_h,
            expected_roi_size=roi_sz,
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            spec=spec,
            eos_token_ids=eos_token_ids,
            max_new_tokens=max_new_tokens,
            allowed_classes=allowed_classes,
        )
        total_cpu_ms += float(filt.get("cpu_overhead_ms", 0.0))
        if not filt["usable"]:
            per_board[bid] = {
                "usable": False,
                "unusable_reason": filt["unusable_reason"],
            }
            continue

        def _make_payload(preds_list: Sequence[Dict[str, Any]], default_status_empty: str) -> Dict[str, Any]:
            if preds_list:
                status = "partial_success" if filt["any_length_truncated"] else "success"
            elif filt["has_no_objects_call"]:
                status = default_status_empty
            else:
                status = "parse_failed"
            return {
                "predictions": [{
                    "image_id": coco_img_id,
                    "status": status,
                    "is_truncated": bool(filt["any_length_truncated"]),
                    "objects": [
                        {"label": p["label"], "bbox_xyxy": p["bbox_xyxy"], "score": p["score"]}
                        for p in preds_list
                    ],
                }]
            }

        base_payload = _make_payload(filt["baseline_post_nms_predictions"], "no_objects_detected")
        cand_payload = _make_payload(filt["filtered_post_nms_predictions"], "no_objects_detected")

        base_eval = repo_eval.evaluate_detections(board_coco, base_payload, iou_threshold=eval_iou_threshold)
        cand_eval = repo_eval.evaluate_detections(board_coco, cand_payload, iou_threshold=eval_iou_threshold)

        b_micro = base_eval["micro"]
        c_micro = cand_eval["micro"]

        base_tp, base_fp, _ = _match_predictions_to_gt_for_audit(
            anns, filt["baseline_post_nms_predictions"], cat_id_to_name, iou_threshold=eval_iou_threshold
        )
        cand_tp, cand_fp, _ = _match_predictions_to_gt_for_audit(
            anns, filt["filtered_post_nms_predictions"], cat_id_to_name, iou_threshold=eval_iou_threshold
        )

        base_tp_gt_ids = {t["matched_gt_id"] for t in base_tp}
        cand_tp_gt_ids = {t["matched_gt_id"] for t in cand_tp}
        lost_gt_ids = sorted(base_tp_gt_ids - cand_tp_gt_ids)
        gained_gt_ids = sorted(cand_tp_gt_ids - base_tp_gt_ids)

        kept_ratio = (
            float(cand_eval["total_predictions"]) / float(base_eval["total_predictions"])
            if base_eval["total_predictions"] > 0
            else 1.0
        )

        # Annotate filtered groups with ex-post GT outcome for case studies only
        filtered_groups_audit = []
        for g in filt["groups"]:
            if not g["filtered_out"]:
                continue
            rbox = g["rep_box_xyxy"]
            glbl = g["label"]
            hit_same = [
                a["id"]
                for a in anns
                if cat_id_to_name[a["category_id"]] == glbl
                and compute_iou(rbox, [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]) >= eval_iou_threshold
            ]
            filtered_groups_audit.append({
                "group_id": g["group_id"],
                "label": glbl,
                "rep_box_xyxy": rbox,
                "seed_tile_id": g["seed_tile_id"],
                "support_tile_ids": g["support_tile_ids"],
                "opportunity_tile_ids": g["opportunity_tile_ids"],
                "opportunity_count": g["opportunity_count"],
                "support_count": g["support_count"],
                "matched_same_class_gt_ids": hit_same,
            })

        per_board[bid] = {
            "usable": True,
            "tile_count": filt["tile_count"],
            "raw_predictions": filt["raw_prediction_count"],
            "kept_raw_predictions": filt["kept_raw_prediction_count"],
            "group_count": filt["group_count"],
            "filtered_group_count": filt["filtered_group_count"],
            "opportunity_count_hist": filt["opportunity_count_hist"],
            "support_count_hist": filt["support_count_hist"],
            "cpu_overhead_ms": filt["cpu_overhead_ms"],
            "baseline": {
                "tp": base_eval["tp"],
                "fp": base_eval["fp"],
                "fn": base_eval["fn"],
                "gt": base_eval["total_gt"],
                "predictions": base_eval["total_predictions"],
                "precision": b_micro["precision"],
                "recall": b_micro["recall"],
                "f1": b_micro["f1"],
                "per_class": base_eval["per_class"],
            },
            "candidate_a": {
                "tp": cand_eval["tp"],
                "fp": cand_eval["fp"],
                "fn": cand_eval["fn"],
                "gt": cand_eval["total_gt"],
                "predictions": cand_eval["total_predictions"],
                "precision": c_micro["precision"],
                "recall": c_micro["recall"],
                "f1": c_micro["f1"],
                "per_class": cand_eval["per_class"],
            },
            "delta": {
                "tp_diff": int(cand_eval["tp"]) - int(base_eval["tp"]),
                "fp_diff": int(cand_eval["fp"]) - int(base_eval["fp"]),
                "fn_diff": int(cand_eval["fn"]) - int(base_eval["fn"]),
                "lost_tp": int(base_eval["tp"]) - int(cand_eval["tp"]),
                "removed_fp": int(base_eval["fp"]) - int(cand_eval["fp"]),
                "precision_diff": round(float(c_micro["precision"]) - float(b_micro["precision"]), 6),
                "recall_diff": round(float(c_micro["recall"]) - float(b_micro["recall"]), 6),
                "f1_diff": round(float(c_micro["f1"]) - float(b_micro["f1"]), 6),
                "kept_post_nms_ratio": round(kept_ratio, 6),
                "lost_gt_ids": lost_gt_ids,
                "gained_gt_ids": gained_gt_ids,
            },
            "filtered_groups_audit": filtered_groups_audit,
        }

        micro_base["tp"] += int(base_eval["tp"])
        micro_base["fp"] += int(base_eval["fp"])
        micro_base["fn"] += int(base_eval["fn"])
        micro_base["gt"] += int(base_eval["total_gt"])
        micro_base["preds"] += int(base_eval["total_predictions"])

        micro_cand["tp"] += int(cand_eval["tp"])
        micro_cand["fp"] += int(cand_eval["fp"])
        micro_cand["fn"] += int(cand_eval["fn"])
        micro_cand["gt"] += int(cand_eval["total_gt"])
        micro_cand["preds"] += int(cand_eval["total_predictions"])

    def _prf(d: Dict[str, int]) -> Dict[str, Any]:
        p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {**d, "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}

    mb = _prf(micro_base)
    mc = _prf(micro_cand)
    return {
        "spec": _merge_spec(spec),
        "per_board": per_board,
        "micro_summary": {
            "baseline": mb,
            "candidate_a": mc,
            "removed_fp": mb["fp"] - mc["fp"],
            "lost_tp": mb["tp"] - mc["tp"],
            "f1_diff": round(mc["f1"] - mb["f1"], 6),
            "kept_post_nms_ratio": round(mc["preds"] / mb["preds"], 6) if mb["preds"] > 0 else 1.0,
        },
        "total_cpu_overhead_ms": round(total_cpu_ms, 3),
    }
