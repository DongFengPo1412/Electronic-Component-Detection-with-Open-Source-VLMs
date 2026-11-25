"""Round 19 post-processing module: IC-only rejection (A1) & low-conf D/Q supplementation (A2/A3).

Hard invariants:
- `capacitor`, `resistor`, `inductor` boxes (`bbox_xyxy`, `label`, `score`, relative order) are NEVER modified.
- Retained A0 official (`score >= 0.25`) boxes NEVER move (`bbox_xyxy`) or change their YOLO `score`.
- A1 ONLY inspects `label == "ic"` and only drops (`reject_other`); all cross-class relabeling is disabled.
- A2 ONLY admits weak classes (`diode`, `transistor`) from `low_conf_pool_005` (`score_min <= score < 0.25`);
  abstention NEVER adds a box; deterministic A0-protecting IoU >= 0.50 dedup runs in original coordinates.
- A3 combines A1-retained official boxes + A2-accepted low-conf D/Q boxes cleanly.
"""

from __future__ import annotations

import copy
import pathlib
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from PIL import Image
import torch

from pcb_vlm.evaluation.composite_detector import compute_iou_xyxy
from pcb_vlm.evaluation.crop_verifier_r18 import (
    R18_CLASSES,
    extract_square_context_crop,
    score_crops_with_model,
)

R19_A1_T_REJECT_OTHER: float = 0.75
R19_A1_M_REJECT_OTHER: float = 0.25
R19_LOW_CONF_UPPER: float = 0.25
R19_DEDUP_IOU_THRESHOLD: float = 0.50
R19_WEAK_CLASSES: Tuple[str, ...] = ("diode", "transistor")
R19_UNTOUCHED_CLASSES: Tuple[str, ...] = ("capacitor", "resistor", "inductor")


def _pred_key(pred: Mapping[str, Any], fallback_idx: int) -> str:
    pid = str(pred.get("prediction_id") or "").strip()
    if pid:
        return pid
    box = [round(float(v), 2) for v in pred["bbox_xyxy"]]
    return f"idx{fallback_idx}_{pred.get('label')}_{box[0]}_{box[1]}_{box[2]}_{box[3]}"


def _top1_top2_margin(probs_by_class: Mapping[str, float]) -> Tuple[str, float, str, float, float]:
    ranked = sorted(((str(c), float(p)) for c, p in probs_by_class.items()), key=lambda x: (-x[1], x[0]))
    top1_cls, top1_p = ranked[0]
    top2_cls, top2_p = ranked[1]
    return top1_cls, top1_p, top2_cls, top2_p, top1_p - top2_p


def compute_r19_crop_probs_for_boxes(
    *,
    image_path: pathlib.Path,
    boxes: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Dict[str, float]]:
    """Compute ResNet-18 4-class probabilities once per unique `prediction_id`."""
    unique_items: List[Tuple[str, Sequence[float]]] = []
    seen: Set[str] = set()
    for idx, pred in enumerate(boxes):
        k = _pred_key(pred, idx)
        if k in seen:
            continue
        seen.add(k)
        unique_items.append((k, pred["bbox_xyxy"]))

    if not unique_items:
        return {}

    crops: List[Image.Image] = []
    with Image.open(image_path) as im_raw:
        rgb = im_raw.convert("RGB")
        for _, box in unique_items:
            crops.append(extract_square_context_crop(rgb, box))

    probs_list = score_crops_with_model(model, crops, device=device, batch_size=batch_size)
    return {k: probs for (k, _), probs in zip(unique_items, probs_list)}


def tag_r19_a0_predictions(official_preds_025: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return deep-copied A0 official boxes with provenance metadata."""
    out: List[Dict[str, Any]] = []
    for pred in official_preds_025:
        item = copy.deepcopy(dict(pred))
        item["orig_r15_label"] = str(item.get("label") or "").strip().lower()
        item["box_source"] = "r15_official_025"
        item["is_low_conf_supplement"] = False
        item["r19_arm"] = "A0"
        out.append(item)
    return out


def apply_r19_a1_ic_reject_only(
    *,
    official_preds_025: Sequence[Mapping[str, Any]],
    probs_by_pred_id: Optional[Mapping[str, Mapping[str, float]]] = None,
    image_path: Optional[pathlib.Path] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    t_reject_other: float = R19_A1_T_REJECT_OTHER,
    m_reject_other: float = R19_A1_M_REJECT_OTHER,
) -> Dict[str, Any]:
    """Arm A1: ONLY inspect `label == 'ic'`; drop iff top1=='other' with p>=0.75 & margin>=0.25."""
    t0 = time.perf_counter()
    probs_map: Dict[str, Mapping[str, float]] = dict(probs_by_pred_id or {})
    if probs_by_pred_id is None:
        if image_path is None or model is None or device is None:
            raise ValueError("Either probs_by_pred_id or (image_path, model, device) must be provided.")
        ic_boxes = [p for p in official_preds_025 if str(p.get("label") or "").strip().lower() == "ic"]
        probs_map = compute_r19_crop_probs_for_boxes(
            image_path=image_path, boxes=ic_boxes, model=model, device=device
        )

    kept_preds: List[Dict[str, Any]] = []
    dropped_ic_boxes: List[Dict[str, Any]] = []
    ic_inspected = 0
    ic_kept = 0

    for idx, pred in enumerate(official_preds_025):
        item = copy.deepcopy(dict(pred))
        lbl = str(item.get("label") or "").strip().lower()
        pid = _pred_key(item, idx)
        item["orig_r15_label"] = lbl
        item["box_source"] = "r15_official_025"
        item["is_low_conf_supplement"] = False
        item["r19_arm"] = "A1"

        if lbl != "ic":
            kept_preds.append(item)
            continue

        ic_inspected += 1
        probs = probs_map.get(pid)
        if probs is None:
            raise KeyError(f"Missing crop probabilities for IC prediction {pid!r}")
        top1_cls, top1_p, top2_cls, top2_p, margin = _top1_top2_margin(probs)
        if top1_cls == "other" and top1_p >= float(t_reject_other) and margin >= float(m_reject_other):
            dropped_ic_boxes.append({
                "prediction_id": item.get("prediction_id") or pid,
                "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
                "r15_score": float(item.get("score", 0.0)),
                "orig_r15_label": "ic",
                "top1_class": top1_cls,
                "top1_prob": round(top1_p, 6),
                "top2_class": top2_cls,
                "margin": round(margin, 6),
                "probs": {str(k): round(float(v), 6) for k, v in probs.items()},
            })
            continue

        ic_kept += 1
        item["verifier_decision"] = "keep_ic_a1"
        item["verifier_source"] = "r18_resnet18_crop_verifier"
        item["verifier_top1_class"] = top1_cls
        item["verifier_top1_prob"] = round(top1_p, 6)
        item["verifier_margin"] = round(margin, 6)
        kept_preds.append(item)

    return {
        "predictions": kept_preds,
        "dropped_ic_boxes": dropped_ic_boxes,
        "summary": {
            "t_reject_other": float(t_reject_other),
            "m_reject_other": float(m_reject_other),
            "input_count": len(official_preds_025),
            "output_count": len(kept_preds),
            "ic_inspected": ic_inspected,
            "ic_kept": ic_kept,
            "ic_dropped": len(dropped_ic_boxes),
            "wall_seconds": round(time.perf_counter() - t0, 4),
        },
    }


def select_r19_a2_low_conf_candidates(
    *,
    official_preds_025: Sequence[Mapping[str, Any]],
    low_conf_pool_005: Sequence[Mapping[str, Any]],
    rule_spec: Mapping[str, Any],
    probs_by_pred_id: Optional[Mapping[str, Mapping[str, float]]] = None,
    a1_dropped_pred_ids: Optional[Set[str]] = None,
    dedup_iou_threshold: float = R19_DEDUP_IOU_THRESHOLD,
) -> Dict[str, Any]:
    """Arm A2: Admit low-conf weak-class (`diode`/`transistor`) boxes with A0-protecting IoU dedup."""
    rule_id = str(rule_spec["rule_id"])
    rule_family = str(rule_spec["rule_family"])
    score_min = float(rule_spec["score_min"])
    t_confirm = float(rule_spec.get("t_confirm", 0.0))
    m_confirm = float(rule_spec.get("m_confirm", 0.0))
    t_ic_to_dq = float(rule_spec.get("t_ic_to_dq", 0.85))
    m_ic_to_dq = float(rule_spec.get("m_ic_to_dq", 0.15))

    if rule_family not in ("yolo_only", "cls_same_class", "cls_same_or_ic_to_dq"):
        raise ValueError(f"Unsupported R19 A2 rule_family: {rule_family!r}")

    a1_dropped_set = set(a1_dropped_pred_ids or set())
    probs_map = probs_by_pred_id or {}

    pre_dedup_candidates: List[Dict[str, Any]] = []
    pool_in_range = 0
    rejected_by_rule_count = 0

    for idx, cand in enumerate(low_conf_pool_005):
        score = float(cand.get("score", 0.0))
        if not (score_min <= score < R19_LOW_CONF_UPPER):
            continue
        pool_in_range += 1
        orig_lbl = str(cand.get("label") or "").strip().lower()
        pid = _pred_key(cand, idx)

        proposed_label: Optional[str] = None
        decision_reason = ""
        cls_evidence: Optional[Dict[str, Any]] = None

        if rule_family == "yolo_only":
            if orig_lbl in R19_WEAK_CLASSES:
                proposed_label = orig_lbl
                decision_reason = "yolo_score_keep_dq"
        elif rule_family == "cls_same_class":
            if orig_lbl in R19_WEAK_CLASSES:
                probs = probs_map.get(pid)
                if probs is None:
                    raise KeyError(f"Missing crop probabilities for low-conf candidate {pid!r}")
                top1_cls, top1_p, _, _, margin = _top1_top2_margin(probs)
                cls_evidence = {"top1_class": top1_cls, "top1_prob": round(top1_p, 6), "margin": round(margin, 6), "probs": {str(k): round(float(v), 6) for k, v in probs.items()}}
                if top1_cls == orig_lbl and top1_p >= t_confirm and margin >= m_confirm:
                    proposed_label = orig_lbl
                    decision_reason = "cls_confirm_same_dq"
        else:  # cls_same_or_ic_to_dq
            if orig_lbl in R19_WEAK_CLASSES:
                probs = probs_map.get(pid)
                if probs is None:
                    raise KeyError(f"Missing crop probabilities for low-conf candidate {pid!r}")
                top1_cls, top1_p, _, _, margin = _top1_top2_margin(probs)
                cls_evidence = {"top1_class": top1_cls, "top1_prob": round(top1_p, 6), "margin": round(margin, 6), "probs": {str(k): round(float(v), 6) for k, v in probs.items()}}
                if top1_cls == orig_lbl and top1_p >= t_confirm and margin >= m_confirm:
                    proposed_label = orig_lbl
                    decision_reason = "cls_confirm_same_dq"
            elif orig_lbl == "ic":
                probs = probs_map.get(pid)
                if probs is None:
                    raise KeyError(f"Missing crop probabilities for low-conf IC candidate {pid!r}")
                top1_cls, top1_p, _, _, margin = _top1_top2_margin(probs)
                cls_evidence = {"top1_class": top1_cls, "top1_prob": round(top1_p, 6), "margin": round(margin, 6), "probs": {str(k): round(float(v), 6) for k, v in probs.items()}}
                if top1_cls in R19_WEAK_CLASSES and top1_p >= t_ic_to_dq and margin >= m_ic_to_dq:
                    proposed_label = top1_cls
                    decision_reason = "cls_relabel_low_ic_to_dq"

        if proposed_label is None:
            rejected_by_rule_count += 1
            continue

        pre_dedup_candidates.append({
            "raw": cand,
            "pid": pid,
            "orig_label": orig_lbl,
            "proposed_label": proposed_label,
            "score": score,
            "decision_reason": decision_reason,
            "cls_evidence": cls_evidence,
        })

    # Deterministic order: highest YOLO score first, then prediction_id
    pre_dedup_candidates.sort(key=lambda c: (-float(c["score"]), str(c["pid"])))

    accepted_low: List[Dict[str, Any]] = []
    dedup_suppressed: List[Dict[str, Any]] = []
    a0_tagged = tag_r19_a0_predictions(official_preds_025)

    for cand in pre_dedup_candidates:
        c_box = cand["raw"]["bbox_xyxy"]
        c_lbl = cand["proposed_label"]

        # 1. Check against ALL A0 official boxes first (protect A0)
        best_a0_iou = 0.0
        best_a0_item: Optional[Mapping[str, Any]] = None
        for idx_a0, a0_item in enumerate(a0_tagged):
            iou = compute_iou_xyxy(c_box, a0_item["bbox_xyxy"])
            if iou > best_a0_iou:
                best_a0_iou = iou
                best_a0_item = a0_item

        if best_a0_item is not None and best_a0_iou >= float(dedup_iou_threshold):
            a0_lbl = str(best_a0_item.get("label") or "").strip().lower()
            a0_pid = str(best_a0_item.get("prediction_id") or "")
            conflict_reason = "overlap_a0_same_class" if a0_lbl == c_lbl else "overlap_a0_cross_class"
            dedup_suppressed.append({
                "prediction_id": cand["pid"],
                "bbox_xyxy": [float(v) for v in c_box],
                "r15_score": float(cand["score"]),
                "orig_r15_label": cand["orig_label"],
                "proposed_label": c_lbl,
                "conflict_reason": conflict_reason,
                "conflict_with_prediction_id": a0_pid,
                "conflict_with_label": a0_lbl,
                "conflict_iou": round(best_a0_iou, 6),
                "conflict_was_a1_dropped_ic": bool(a0_pid in a1_dropped_set),
            })
            continue

        # 2. Check against already-accepted low-conf boxes
        best_low_iou = 0.0
        best_low_item: Optional[Mapping[str, Any]] = None
        for low_item in accepted_low:
            iou = compute_iou_xyxy(c_box, low_item["bbox_xyxy"])
            if iou > best_low_iou:
                best_low_iou = iou
                best_low_item = low_item

        if best_low_item is not None and best_low_iou >= float(dedup_iou_threshold):
            low_lbl = str(best_low_item.get("label") or "").strip().lower()
            low_pid = str(best_low_item.get("prediction_id") or "")
            conflict_reason = "overlap_low_conf_same_class" if low_lbl == c_lbl else "overlap_low_conf_cross_class"
            dedup_suppressed.append({
                "prediction_id": cand["pid"],
                "bbox_xyxy": [float(v) for v in c_box],
                "r15_score": float(cand["score"]),
                "orig_r15_label": cand["orig_label"],
                "proposed_label": c_lbl,
                "conflict_reason": conflict_reason,
                "conflict_with_prediction_id": low_pid,
                "conflict_with_label": low_lbl,
                "conflict_iou": round(best_low_iou, 6),
                "conflict_was_a1_dropped_ic": False,
            })
            continue

        # 3. Accept!
        accepted = copy.deepcopy(dict(cand["raw"]))
        accepted["orig_r15_label"] = cand["orig_label"]
        accepted["label"] = c_lbl
        accepted["aligned_fics6_label"] = c_lbl
        accepted["box_source"] = "r19_low_conf_dq_yolo" if rule_family == "yolo_only" else "r19_low_conf_dq_cls"
        accepted["is_low_conf_supplement"] = True
        accepted["low_conf_rule_id"] = rule_id
        accepted["r19_decision_reason"] = cand["decision_reason"]
        accepted["vlm_intervened"] = False
        if cand["cls_evidence"] is not None:
            accepted["verifier_top1_class"] = cand["cls_evidence"]["top1_class"]
            accepted["verifier_top1_prob"] = cand["cls_evidence"]["top1_prob"]
            accepted["verifier_margin"] = cand["cls_evidence"]["margin"]
        accepted_low.append(accepted)

    return {
        "a2_predictions": a0_tagged + accepted_low,
        "accepted_low_boxes": accepted_low,
        "dedup_suppressed_boxes": dedup_suppressed,
        "rejected_by_rule_count": rejected_by_rule_count,
        "summary": {
            "rule_id": rule_id,
            "rule_family": rule_family,
            "score_min": score_min,
            "pool_in_range_count": pool_in_range,
            "pre_dedup_admitted_count": len(pre_dedup_candidates),
            "dedup_suppressed_count": len(dedup_suppressed),
            "accepted_low_count": len(accepted_low),
            "a2_total_count": len(a0_tagged) + len(accepted_low),
        },
    }


def verify_r19_untouched_invariants(
    *,
    a0_preds: Sequence[Mapping[str, Any]],
    derived_preds: Sequence[Mapping[str, Any]],
    arm_name: str,
) -> None:
    """Assert `capacitor`, `resistor`, `inductor` are 100% bit-for-bit identical in coords/scores/order."""
    def _sig(items: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Tuple[float, ...], float]]:
        out: List[Tuple[str, Tuple[float, ...], float]] = []
        for it in items:
            lbl = str(it.get("label") or "").strip().lower()
            if lbl in R19_UNTOUCHED_CLASSES:
                out.append((lbl, tuple(float(v) for v in it["bbox_xyxy"]), float(it.get("score", 0.0))))
        return out

    s0 = _sig(a0_preds)
    s1 = _sig(derived_preds)
    if s0 != s1:
        raise AssertionError(f"Round 19 invariant violated in {arm_name}: capacitor/resistor/inductor changed!")


def apply_r19_four_arms(
    *,
    official_preds_025: Sequence[Mapping[str, Any]],
    low_conf_pool_005: Sequence[Mapping[str, Any]],
    a2_rule_spec: Mapping[str, Any],
    probs_by_pred_id: Optional[Mapping[str, Mapping[str, float]]] = None,
    image_path: Optional[pathlib.Path] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    t_reject_other: float = R19_A1_T_REJECT_OTHER,
    m_reject_other: float = R19_A1_M_REJECT_OTHER,
) -> Dict[str, Any]:
    """Run A0, A1, A2, and A3 from a single base detection (`official_preds_025` + `low_conf_pool_005`)."""
    t0 = time.perf_counter()
    rule_family = str(a2_rule_spec["rule_family"])
    score_min = float(a2_rule_spec["score_min"])

    probs_map: Dict[str, Mapping[str, float]] = dict(probs_by_pred_id or {})
    if probs_by_pred_id is None:
        needed_boxes: List[Mapping[str, Any]] = [
            p for p in official_preds_025 if str(p.get("label") or "").strip().lower() == "ic"
        ]
        if rule_family == "cls_same_class":
            needed_boxes.extend(
                p for p in low_conf_pool_005
                if score_min <= float(p.get("score", 0.0)) < R19_LOW_CONF_UPPER
                and str(p.get("label") or "").strip().lower() in R19_WEAK_CLASSES
            )
        elif rule_family == "cls_same_or_ic_to_dq":
            needed_boxes.extend(
                p for p in low_conf_pool_005
                if score_min <= float(p.get("score", 0.0)) < R19_LOW_CONF_UPPER
                and str(p.get("label") or "").strip().lower() in ("diode", "transistor", "ic")
            )
        if needed_boxes:
            if image_path is None or model is None or device is None:
                raise ValueError("image_path, model, and device are required when probs_by_pred_id is None.")
            probs_map = compute_r19_crop_probs_for_boxes(
                image_path=image_path, boxes=needed_boxes, model=model, device=device
            )

    a0_preds = tag_r19_a0_predictions(official_preds_025)
    a1_res = apply_r19_a1_ic_reject_only(
        official_preds_025=official_preds_025,
        probs_by_pred_id=probs_map,
        t_reject_other=t_reject_other,
        m_reject_other=m_reject_other,
    )
    a1_dropped_ids = {str(d["prediction_id"]) for d in a1_res["dropped_ic_boxes"]}
    a2_res = select_r19_a2_low_conf_candidates(
        official_preds_025=official_preds_025,
        low_conf_pool_005=low_conf_pool_005,
        rule_spec=a2_rule_spec,
        probs_by_pred_id=probs_map,
        a1_dropped_pred_ids=a1_dropped_ids,
    )

    a3_preds = copy.deepcopy(a1_res["predictions"])
    for item in a3_preds:
        item["r19_arm"] = "A3"
    for low_item in a2_res["accepted_low_boxes"]:
        cp = copy.deepcopy(low_item)
        cp["r19_arm"] = "A3"
        a3_preds.append(cp)

    verify_r19_untouched_invariants(a0_preds=a0_preds, derived_preds=a1_res["predictions"], arm_name="A1")
    verify_r19_untouched_invariants(a0_preds=a0_preds, derived_preds=a2_res["a2_predictions"], arm_name="A2")
    verify_r19_untouched_invariants(a0_preds=a0_preds, derived_preds=a3_preds, arm_name="A3")

    return {
        "a0_predictions": a0_preds,
        "a1_predictions": a1_res["predictions"],
        "a2_predictions": a2_res["a2_predictions"],
        "a3_predictions": a3_preds,
        "a1_dropped_ic_boxes": a1_res["dropped_ic_boxes"],
        "a2_accepted_low_boxes": a2_res["accepted_low_boxes"],
        "a2_dedup_suppressed_boxes": a2_res["dedup_suppressed_boxes"],
        "summary": {
            "a0_count": len(a0_preds),
            "a1_summary": a1_res["summary"],
            "a2_summary": a2_res["summary"],
            "a3_count": len(a3_preds),
            "wall_seconds": round(time.perf_counter() - t0, 4),
        },
    }
