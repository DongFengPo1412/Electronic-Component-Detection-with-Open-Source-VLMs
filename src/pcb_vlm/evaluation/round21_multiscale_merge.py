"""Round 21 multiscale 448px weak-class supplement merge & one-to-one coverage decomposition.

Hard invariants:
- Only accepts 448px predictions with `label in {"diode", "transistor"}` AND `score >= 0.25`.
- Never modifies `capacitor`, `resistor`, `inductor`, or `ic` boxes (100% bit-for-bit identical to base A0/A1).
- Protects A0 640px official boxes: any 448px D/Q candidate with original-coordinate `IoU >= 0.50`
  against ANY A0 box (same-class or cross-class) is suppressed and logged as a conflict.
- Deterministic order (`(-score, prediction_id)`), full tile/score/scale traceability.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pcb_vlm.evaluation.composite_detector import compute_iou_xyxy

R21_SUPPLEMENT_TILE_SIZE: int = 448
R21_SUPPLEMENT_OVERLAP_RATIO: float = 0.20
R21_SUPPLEMENT_IMGSZ: int = 640
R21_SUPPLEMENT_CONF_MIN: float = 0.25
R21_DEDUP_IOU_THRESHOLD: float = 0.50
R21_WEAK_CLASSES: Tuple[str, ...] = ("diode", "transistor")
R21_BASE_PROTECTED_CLASSES: Tuple[str, ...] = ("capacitor", "resistor", "inductor", "ic")


def _pred_key(pred: Mapping[str, Any], fallback_idx: int) -> str:
    pid = str(pred.get("prediction_id") or "").strip()
    if pid:
        return pid
    box = [round(float(v), 2) for v in pred["bbox_xyxy"]]
    return f"idx{fallback_idx}_{pred.get('label')}_{box[0]}_{box[1]}_{box[2]}_{box[3]}"


def _verify_protected_identical(
    *,
    base_preds: Sequence[Mapping[str, Any]],
    merged_preds: Sequence[Mapping[str, Any]],
    arm_name: str,
) -> None:
    def _sig(items: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Tuple[float, ...], float]]:
        out: List[Tuple[str, Tuple[float, ...], float]] = []
        for it in items:
            lbl = str(it.get("label") or "").strip().lower()
            if lbl in R21_BASE_PROTECTED_CLASSES:
                out.append((lbl, tuple(float(v) for v in it["bbox_xyxy"]), float(it.get("score", 0.0))))
        return out

    if _sig(base_preds) != _sig(merged_preds):
        raise AssertionError(f"Round 21 invariant violated in {arm_name}: protected classes changed!")


def merge_r21_448_dq_supplement(
    *,
    a0_preds_640: Sequence[Mapping[str, Any]],
    preds_448_025: Sequence[Mapping[str, Any]],
    a1_preds_640: Optional[Sequence[Mapping[str, Any]]] = None,
    dedup_iou_threshold: float = R21_DEDUP_IOU_THRESHOLD,
    conf_min: float = R21_SUPPLEMENT_CONF_MIN,
) -> Dict[str, Any]:
    """Merge 448px D/Q (>=0.25) supplement onto 640px A0 (and optional A1) with A0-priority IoU dedup."""
    t0 = time.perf_counter()

    # Filter 448px candidates strictly to diode/transistor with score >= 0.25
    raw_cands: List[Tuple[int, Mapping[str, Any], str]] = []
    for idx, p in enumerate(preds_448_025):
        lbl = str(p.get("label") or "").strip().lower()
        score = float(p.get("score", 0.0))
        if lbl in R21_WEAK_CLASSES and score >= float(conf_min):
            raw_cands.append((idx, p, _pred_key(p, idx)))

    raw_cands.sort(key=lambda t: (-float(t[1].get("score", 0.0)), t[2]))

    accepted_448: List[Dict[str, Any]] = []
    conflict_suppressed: List[Dict[str, Any]] = []

    for _, cand, pid in raw_cands:
        c_box = [float(v) for v in cand["bbox_xyxy"]]
        c_lbl = str(cand.get("label") or "").strip().lower()
        c_score = float(cand.get("score", 0.0))
        tile_id = cand.get("tile_id")
        if tile_id is None and isinstance(cand.get("source_tiles"), list) and cand["source_tiles"]:
            tile_id = cand["source_tiles"][0]

        # 1. Check against ALL A0 640px official boxes first (protect A0 priority)
        best_a0_iou = 0.0
        best_a0_item: Optional[Mapping[str, Any]] = None
        for a0_idx, a0_item in enumerate(a0_preds_640):
            iou = compute_iou_xyxy(c_box, a0_item["bbox_xyxy"])
            if iou > best_a0_iou:
                best_a0_iou = iou
                best_a0_item = a0_item

        if best_a0_item is not None and best_a0_iou >= float(dedup_iou_threshold):
            a0_lbl = str(best_a0_item.get("label") or "").strip().lower()
            reason = "overlap_a0_same_class" if a0_lbl == c_lbl else "overlap_a0_cross_class"
            conflict_suppressed.append({
                "prediction_id": pid,
                "label": c_lbl,
                "r15_score": c_score,
                "bbox_xyxy": c_box,
                "source_tile_id": tile_id,
                "source_tiles": cand.get("source_tiles"),
                "conflict_reason": reason,
                "conflict_with_prediction_id": str(best_a0_item.get("prediction_id") or ""),
                "conflict_with_label": a0_lbl,
                "conflict_iou": round(best_a0_iou, 6),
            })
            continue

        # 2. Check against already-accepted 448px boxes
        best_448_iou = 0.0
        best_448_item: Optional[Mapping[str, Any]] = None
        for acc_item in accepted_448:
            iou = compute_iou_xyxy(c_box, acc_item["bbox_xyxy"])
            if iou > best_448_iou:
                best_448_iou = iou
                best_448_item = acc_item

        if best_448_item is not None and best_448_iou >= float(dedup_iou_threshold):
            acc_lbl = str(best_448_item.get("label") or "").strip().lower()
            reason = "overlap_448_same_class" if acc_lbl == c_lbl else "overlap_448_cross_class"
            conflict_suppressed.append({
                "prediction_id": pid,
                "label": c_lbl,
                "r15_score": c_score,
                "bbox_xyxy": c_box,
                "source_tile_id": tile_id,
                "source_tiles": cand.get("source_tiles"),
                "conflict_reason": reason,
                "conflict_with_prediction_id": str(best_448_item.get("prediction_id") or ""),
                "conflict_with_label": acc_lbl,
                "conflict_iou": round(best_448_iou, 6),
            })
            continue

        # 3. Accept 448px D/Q box with full traceability
        acc = copy.deepcopy(dict(cand))
        acc["orig_r15_label"] = c_lbl
        acc["aligned_fics6_label"] = c_lbl
        acc["box_source"] = "r21_448px_dq_supplement"
        acc["is_low_conf_supplement"] = False
        acc["is_multiscale_448_supplement"] = True
        acc["scale_tile_size"] = R21_SUPPLEMENT_TILE_SIZE
        acc["source_tile_id"] = tile_id
        acc["conflict_checked"] = True
        acc["vlm_intervened"] = False
        accepted_448.append(acc)

    def _tag_base(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        tagged: List[Dict[str, Any]] = []
        for it in items:
            cp = copy.deepcopy(dict(it))
            cp.setdefault("orig_r15_label", str(cp.get("label") or "").strip().lower())
            cp.setdefault("box_source", "r15_official_025")
            cp.setdefault("is_low_conf_supplement", False)
            cp.setdefault("is_multiscale_448_supplement", False)
            cp.setdefault("scale_tile_size", 640)
            tagged.append(cp)
        return tagged

    a0_tagged = _tag_base(a0_preds_640)
    a0_plus_448 = a0_tagged + copy.deepcopy(accepted_448)
    _verify_protected_identical(base_preds=a0_preds_640, merged_preds=a0_plus_448, arm_name="A0_plus_448")

    a1_plus_448: Optional[List[Dict[str, Any]]] = None
    if a1_preds_640 is not None:
        a1_tagged = _tag_base(a1_preds_640)
        a1_plus_448 = a1_tagged + copy.deepcopy(accepted_448)
        _verify_protected_identical(base_preds=a1_preds_640, merged_preds=a1_plus_448, arm_name="A1_plus_448")

    return {
        "a0_plus_448": a0_plus_448,
        "a1_plus_448": a1_plus_448,
        "accepted_448_boxes": accepted_448,
        "conflict_suppressed": conflict_suppressed,
        "summary": {
            "raw_448_dq_025_candidates": len(raw_cands),
            "accepted_448_count": len(accepted_448),
            "accepted_448_diode": sum(1 for b in accepted_448 if b["label"] == "diode"),
            "accepted_448_transistor": sum(1 for b in accepted_448 if b["label"] == "transistor"),
            "conflict_suppressed_count": len(conflict_suppressed),
            "wall_seconds": round(time.perf_counter() - t0, 4),
        },
    }


def _one_to_one_class_agnostic_match(
    gts: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
) -> Dict[int, Dict[str, Any]]:
    """Greedy one-to-one matching by prediction score descending (class-agnostic IoU >= 0.50)."""
    sorted_preds = sorted(
        enumerate(preds),
        key=lambda t: (-float(t[1].get("score", 0.0)), str(t[1].get("prediction_id") or f"idx{t[0]}")),
    )
    used_gt: Set[int] = set()
    matched_by_gt: Dict[int, Dict[str, Any]] = {}
    for _, p in sorted_preds:
        best_gi, best_iou = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in used_gt:
                continue
            iou = compute_iou_xyxy(p["bbox_xyxy"], g["bbox_xyxy"])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= float(iou_threshold):
            used_gt.add(best_gi)
            matched_by_gt[best_gi] = {
                "prediction_id": p.get("prediction_id"),
                "pred_label": str(p.get("label") or "").lower(),
                "pred_score": float(p.get("score", 0.0)),
                "iou": round(best_iou, 4),
            }
    return matched_by_gt


def compute_one_to_one_coverage_split(
    *,
    gts: Sequence[Mapping[str, Any]],
    pool_640_005: Sequence[Mapping[str, Any]],
    pool_448_005: Sequence[Mapping[str, Any]],
    preds_448_025: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Split every D/Q GT into: covered_by_640_low_pool vs newly_covered_only_by_448 vs missed_by_both."""
    dq_indices = [i for i, g in enumerate(gts) if str(g.get("label") or "").lower() in R21_WEAK_CLASSES]
    m_640 = _one_to_one_class_agnostic_match(gts, pool_640_005, iou_threshold=iou_threshold)
    m_448_low = _one_to_one_class_agnostic_match(gts, pool_448_005, iou_threshold=iou_threshold)
    m_448_off = _one_to_one_class_agnostic_match(gts, preds_448_025, iou_threshold=iou_threshold)

    counts = {
        "dq_gt_total": len(dq_indices),
        "covered_by_640_low_pool": 0,
        "newly_covered_only_by_448_pool005": 0,
        "newly_covered_only_by_448_off025": 0,
        "missed_by_both": 0,
        "by_class": {
            "diode": {"gt": 0, "covered_640_005": 0, "new_448_005": 0, "new_448_025": 0, "missed_both": 0},
            "transistor": {"gt": 0, "covered_640_005": 0, "new_448_005": 0, "new_448_025": 0, "missed_both": 0},
        },
    }
    gt_records: List[Dict[str, Any]] = []

    for gi in dq_indices:
        glbl = str(gts[gi]["label"]).lower()
        in_640 = gi in m_640
        in_448_005 = gi in m_448_low
        in_448_025 = gi in m_448_off
        counts["by_class"][glbl]["gt"] += 1

        if in_640:
            status = "covered_by_640_low_pool"
            counts["covered_by_640_low_pool"] += 1
            counts["by_class"][glbl]["covered_640_005"] += 1
        elif in_448_005:
            status = "newly_covered_only_by_448_pool005"
            counts["newly_covered_only_by_448_pool005"] += 1
            counts["by_class"][glbl]["new_448_005"] += 1
            if in_448_025:
                counts["newly_covered_only_by_448_off025"] += 1
                counts["by_class"][glbl]["new_448_025"] += 1
        else:
            status = "missed_by_both"
            counts["missed_by_both"] += 1
            counts["by_class"][glbl]["missed_both"] += 1

        gt_records.append({
            "gt_index": gi,
            "gt_label": glbl,
            "bbox_xyxy": [float(v) for v in gts[gi]["bbox_xyxy"]],
            "coverage_status": status,
            "match_640_005": m_640.get(gi),
            "match_448_005": m_448_low.get(gi),
            "match_448_025": m_448_off.get(gi),
        })

    return {"summary": counts, "gt_records": gt_records}
