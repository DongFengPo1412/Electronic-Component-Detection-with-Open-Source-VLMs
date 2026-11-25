"""Round 14 composite PCB component detector, bounded Qwen3 verifier, language-spatial query engine, and 3-way evaluator.

Strict invariants:
- Zero Ground Truth reading during inference, candidate selection, VLM verification, or query execution.
- Preserve YOLO11n raw WACV class IDs/names and real confidence scores; ONLY the 6 FICS-PCB classes map into
  aligned FICS predictions. Non-FICS extra classes (connector, led, button, switch, clock) are NEVER auto-merged
  (for example, `led` is NEVER merged into `diode`).
- Save BOTH low-threshold candidate pool (conf >= 0.05, NO Top-200 cap) and official detections (conf >= 0.25).
- Deterministic bounded hard-candidate selection (<= 48 candidates/board) and conservative VLM verification:
  format error / non-normal EOS / technical failure / uncertain sub-threshold NEVER discards or fabricates boxes.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

from pcb_vlm.data.fics_dslr import file_sha256
from pcb_vlm.evaluation.candidate_verifier import (
    ALLOWED_SEVEN_LABELS,
    DEFAULT_VERIFIER_PROMPT,
    parse_single_label_json,
)
from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES
from pcb_vlm.evaluation.gt_free_pipeline import (
    compute_board_roi_and_tiles,
    prepare_verifier_manifest_gt_free,
)
from pcb_vlm.evaluation.language_instruction import (
    CLASS_CN_DISPLAY,
    _ALL_CLASSES_PHRASES,
    _CLASS_PATTERNS,
    _COUNT_PATTERNS,
    _NEGATION_PHRASES,
    _OUT_OF_SCOPE_PATTERNS,
    _strip_spaces,
)
from pcb_vlm.geometry.tiling import map_tile_box_to_global
from pcb_vlm.inference.gpu_gate import probe_gpu_gate

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import evaluate as repo_eval  # noqa: E402

EXPECTED_YOLO11N_BEST_SHA256 = "5378f977c9d3c56d9d6456ee4e4791b662fa8c294af5f5488f570b5e5a3bb4e1"
EXPECTED_FICS_R15_BEST_SHA256 = "4d92fd1d8093ea7498c13b3aac6dd5a6e5c21b9965b96667270883aed571009e"
DEFAULT_COMPOSITE_DETECTOR_PRESET = "fics_yolo11n_r15"
DEFAULT_ULTRALYTICS_OVERLAY = r"C:\Users\Liu\.codex\cache\tasks\pcb-train-prep-20261104\dependency-overlay"

YOLO_WACV_ID_TO_RAW_NAME: Dict[int, str] = {
    0: "capacitor",
    1: "resistor",
    2: "ic",
    3: "connector",
    4: "led",
    5: "transistor",
    6: "diode",
    7: "inductor",
    8: "button",
    9: "switch",
    10: "clock",
}

YOLO_FICS6_ID_TO_RAW_NAME: Dict[int, str] = {
    idx: name for idx, name in enumerate(FICS_SIX_CLASSES)
}

YOLO_DETECTOR_PRESET_SCHEMAS: Dict[str, Dict[int, str]] = {
    "wacv_11": YOLO_WACV_ID_TO_RAW_NAME,
    "fics_6": YOLO_FICS6_ID_TO_RAW_NAME,
}

SUPPORTED_COMPOSITE_DETECTOR_PRESETS: Tuple[str, ...] = (
    "fics_yolo11n_r15",
    "wacv_yolo11n",
    "fics_yolo11n_r15_ic_cal",
    "fics_yolo11n_r17_cand",
    "fics_yolo11n_r15_crop_verify_r18",
    "fics_yolo11n_r15_r19_a1_ic_reject",
    "fics_yolo11n_r15_r19_a3_combined",
    "fics_yolo11n_r15_r20_a1_ic_relabel",
    "fics_yolo11n_r15_r21_a1_plus_448",
)

COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA: Dict[str, str] = {
    "fics_yolo11n_r15": "fics_6",
    "wacv_yolo11n": "wacv_11",
    "fics_yolo11n_r15_ic_cal": "fics_6",
    "fics_yolo11n_r17_cand": "fics_6",
    "fics_yolo11n_r15_crop_verify_r18": "fics_6",
    "fics_yolo11n_r15_r19_a1_ic_reject": "fics_6",
    "fics_yolo11n_r15_r19_a3_combined": "fics_6",
    "fics_yolo11n_r15_r20_a1_ic_relabel": "fics_6",
    "fics_yolo11n_r15_r21_a1_plus_448": "fics_6",
}

WACV_TO_FICS6_ALIGNED: Dict[str, str] = {c: c for c in FICS_SIX_CLASSES}
WACV_EXTRA_NON_FICS_CLASSES: Tuple[str, ...] = ("connector", "led", "button", "switch", "clock")

COMPOSITE_REGION_NAMES: Dict[str, str] = {
    "unconstrained": "整图范围",
    "left_half": "原图左半边 (cx < W/2)",
    "right_half": "原图右半边 (cx >= W/2)",
    "top_half": "原图上半边 (cy < H/2)",
    "bottom_half": "原图下半边 (cy >= H/2)",
}

_LEFT_REGION_PATTERNS: Tuple[str, ...] = (
    r"左半边", r"左半侧", r"左半部", r"左半区", r"左半", r"左侧", r"左边", r"\bleft\s+half\b", r"\bleft\s+side\b",
)
_RIGHT_REGION_PATTERNS: Tuple[str, ...] = (
    r"右半边", r"右半侧", r"右半部", r"右半区", r"右半", r"右侧", r"右边", r"\bright\s+half\b", r"\bright\s+side\b",
)
_TOP_REGION_PATTERNS: Tuple[str, ...] = (
    r"上半边", r"上半侧", r"上半部", r"上半区", r"上半", r"上侧", r"上边", r"\btop\s+half\b", r"\bupper\s+half\b",
)
_BOTTOM_REGION_PATTERNS: Tuple[str, ...] = (
    r"下半边", r"下半侧", r"下半部", r"下半区", r"下半", r"下侧", r"下边", r"\bbottom\s+half\b", r"\blower\s+half\b",
)
_UNSUPPORTED_COMPOSITE_REGION_PATTERNS: Tuple[str, ...] = (
    r"左上", r"右上", r"左下", r"右下", r"中间", r"中央", r"中心", r"角落", r"边缘", r"附近", r"旁边",
    r"\btop[-\s]?(?:left|right)\b", r"\bbottom[-\s]?(?:left|right)\b", r"\bcorner\b", r"\bcenter\b", r"\bmiddle\b",
)


def compute_iou_xyxy(b1: Sequence[float], b2: Sequence[float]) -> float:
    x_a = max(float(b1[0]), float(b2[0]))
    y_a = max(float(b1[1]), float(b2[1]))
    x_b = min(float(b1[2]), float(b2[2]))
    y_b = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = a1 + a2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def query_nvidia_smi_used_mib(gpu_index: int = 0) -> Optional[float]:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                f"--id={int(gpu_index)}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None
    return None


def normalize_yolo_raw_class(
    raw_cls_id: int,
    model_names: Mapping[Any, Any],
    *,
    expected_schema: str = "wacv_11",
) -> Dict[str, Any]:
    """Resolve raw YOLO class name and explicit FICS-6 alignment without merging non-FICS classes."""
    fallback_map = YOLO_DETECTOR_PRESET_SCHEMAS.get(str(expected_schema or "wacv_11"), YOLO_WACV_ID_TO_RAW_NAME)
    if int(raw_cls_id) in model_names:
        raw_name = str(model_names[int(raw_cls_id)]).strip().lower()
    elif str(raw_cls_id) in model_names:
        raw_name = str(model_names[str(raw_cls_id)]).strip().lower()
    elif int(raw_cls_id) in fallback_map:
        raw_name = fallback_map[int(raw_cls_id)]
    else:
        raw_name = f"unknown_{raw_cls_id}"
    aligned = WACV_TO_FICS6_ALIGNED.get(raw_name)
    return {
        "raw_yolo_class_id": int(raw_cls_id),
        "raw_yolo_label": raw_name,
        "aligned_fics6_label": aligned,
        "is_fics6_aligned": aligned is not None,
    }


def class_wise_nms_with_trace(
    detections: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
    label_key: str = "raw_yolo_label",
    board_id: str,
    id_prefix: str = "yc",
) -> List[Dict[str, Any]]:
    """Deterministic per-class cross-tile NMS sorted by (-score, tile_id, tile_box_index, orig_order)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for idx, d in enumerate(detections):
        cp = dict(d)
        cp.setdefault("_fallback_order", idx)
        lbl = str(cp.get(label_key) or cp.get("label") or "").strip().lower()
        grouped.setdefault(lbl, []).append(cp)

    kept_all: List[Dict[str, Any]] = []
    for lbl in sorted(grouped.keys()):
        ordered = sorted(
            grouped[lbl],
            key=lambda x: (
                -float(x["score"]),
                int(x.get("tile_id") if x.get("tile_id") is not None else -1),
                int(x.get("tile_box_index") if x.get("tile_box_index") is not None else -1),
                int(x.get("_fallback_order", 0)),
            ),
        )
        while ordered:
            best = ordered.pop(0)
            best_box = best["bbox_xyxy"]
            suppressed: List[Dict[str, Any]] = []
            remaining: List[Dict[str, Any]] = []
            for other in ordered:
                ov = compute_iou_xyxy(best_box, other["bbox_xyxy"])
                if ov >= float(iou_threshold):
                    suppressed.append({
                        "pre_nms_id": other.get("pre_nms_id"),
                        "tile_id": other.get("tile_id"),
                        "label": lbl,
                        "score": round(float(other["score"]), 6),
                        "iou_with_kept": round(float(ov), 4),
                    })
                else:
                    remaining.append(other)
            ordered = remaining
            best_out = dict(best)
            best_out.pop("_fallback_order", None)
            best_out["bbox_xyxy"] = [round(float(v), 2) for v in best_box]
            best_out["score"] = round(float(best["score"]), 6)
            best_out["suppressed_count"] = len(suppressed) + int(best_out.get("suppressed_count") or 0)
            if suppressed:
                best_out["suppressed_details"] = list(best_out.get("suppressed_details") or []) + suppressed
            kept_all.append(best_out)

    kept_all.sort(
        key=lambda x: (
            -float(x["score"]),
            str(x.get("label") or x.get("raw_yolo_label") or ""),
            int(x.get("tile_id") if x.get("tile_id") is not None else -1),
            int(x.get("tile_box_index") if x.get("tile_box_index") is not None else -1),
        )
    )
    out: List[Dict[str, Any]] = []
    for rank, item in enumerate(kept_all):
        row = dict(item)
        pid = row.get("prediction_id") or f"{board_id}_{id_prefix}{rank:04d}"
        row["prediction_id"] = pid
        row["board_id"] = str(board_id)
        row["image_id"] = int(row.get("image_id", 1))
        row["rank"] = rank
        row["orig_order"] = rank
        out.append(row)
    return out


def filter_fics6_pool_by_class_thresholds(
    low_conf_pool_fics6: Sequence[Mapping[str, Any]],
    *,
    default_conf: float = 0.25,
    per_class_conf: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Deterministically filter an existing FICS-6 low-conf pool using per-class thresholds."""
    thresholds = {str(k).strip().lower(): float(v) for k, v in (per_class_conf or {}).items()}
    kept: List[Dict[str, Any]] = []
    for det in low_conf_pool_fics6:
        cp = dict(det)
        lbl = str(cp.get("aligned_fics6_label") or cp.get("label") or "").strip().lower()
        if lbl not in WACV_TO_FICS6_ALIGNED:
            continue
        thr = float(thresholds.get(lbl, default_conf))
        if float(cp.get("score", 0.0)) >= thr:
            cp["label"] = lbl
            kept.append(cp)
    for idx, item in enumerate(kept):
        item["orig_order"] = idx
        item["rank"] = idx
    return kept


def split_aligned_and_extra_predictions(
    nms_detections: Sequence[Mapping[str, Any]],
    *,
    conf_official: float = 0.25,
    per_class_conf: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Separate FICS-6 aligned detections (0.05 pool & official threshold) from extra WACV classes."""
    thresholds = {str(k).strip().lower(): float(v) for k, v in (per_class_conf or {}).items()}
    low_conf_pool_fics6: List[Dict[str, Any]] = []
    official_fics6: List[Dict[str, Any]] = []
    extra_non_fics_005: List[Dict[str, Any]] = []
    extra_non_fics_025: List[Dict[str, Any]] = []

    for det in nms_detections:
        cp = dict(det)
        score = float(cp["score"])
        aligned = cp.get("aligned_fics6_label")
        is_aligned = bool(cp.get("is_fics6_aligned")) and (aligned in WACV_TO_FICS6_ALIGNED)
        if is_aligned:
            lbl = str(aligned)
            cp["label"] = lbl
            cp.setdefault("detector_source", "yolo11n")
            cp.setdefault("vlm_intervened", False)
            cp.setdefault("vlm_decision", "not_selected")
            low_conf_pool_fics6.append(cp)
            cls_thr = float(thresholds.get(lbl, conf_official))
            if score >= cls_thr:
                official_fics6.append(dict(cp))
        else:
            cp["label"] = str(cp.get("raw_yolo_label") or cp.get("label") or "unknown")
            extra_non_fics_005.append(cp)
            if score >= float(conf_official):
                extra_non_fics_025.append(dict(cp))

    for i, item in enumerate(low_conf_pool_fics6):
        item["orig_order"] = i
        item["rank"] = i
    for i, item in enumerate(official_fics6):
        item["orig_order"] = i
        item["rank"] = i

    def _raw_counts(items: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for it in items:
            k = str(it.get("raw_yolo_label") or it.get("label") or "unknown")
            counts[k] = counts.get(k, 0) + 1
        return counts

    return {
        "low_conf_pool_fics6": low_conf_pool_fics6,
        "official_predictions_fics6": official_fics6,
        "raw_non_fics_detections_005": extra_non_fics_005,
        "raw_non_fics_detections_025": extra_non_fics_025,
        "non_fics_class_counts_005": _raw_counts(extra_non_fics_005),
        "non_fics_class_counts_025": _raw_counts(extra_non_fics_025),
    }


def select_hard_candidates_for_vlm(
    low_conf_pool_fics6: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int = 48,
    conf_official: float = 0.25,
    uncertain_high: float = 0.45,
    rescue_low: float = 0.12,
    conflict_iou: float = 0.50,
) -> Dict[str, Any]:
    """Deterministic GT-free selector picking <= 48 hard candidates per board for Qwen3 verification.

    Priority order:
      1) cross_class_conflict: boxes with score >= conf_official overlapping (IoU >= 0.50) with
         another score >= conf_official box of a DIFFERENT FICS-6 class.
      2) low_conf_official_uncertain: boxes with conf_official <= score < uncertain_high (sorted
         by score ascending so most borderline predictions are verified first; capped at 24 slots).
      3) borderline_rescue_pool: boxes with rescue_low <= score < conf_official not overlapping
         (IoU >= 0.50) any strong >= uncertain_high official box (sorted by score descending).
      4) fallback_low_conf: remaining non-overlapping boxes in [0.05, rescue_low) sorted descending.
    """
    cap = max(0, int(max_candidates))
    pool = [dict(x) for x in low_conf_pool_fics6 if float(x.get("score", 0.0)) >= 0.05]
    official = [x for x in pool if float(x["score"]) >= float(conf_official)]
    strong_official = [x for x in official if float(x["score"]) >= float(uncertain_high)]

    selected: List[Dict[str, Any]] = []
    seen_ids = set()

    def _add(item: Mapping[str, Any], reason: str, priority: int, extra: Optional[Mapping[str, Any]] = None) -> bool:
        if len(selected) >= cap:
            return False
        pid = str(item["prediction_id"])
        if pid in seen_ids:
            return False
        seen_ids.add(pid)
        rec = dict(item)
        rec["selection_reason"] = reason
        rec["selection_priority"] = priority
        if extra:
            rec.update(dict(extra))
        selected.append(rec)
        return True

    # 1) Same-location cross-class conflicts among official >= 0.25 predictions
    conflict_rows: List[Tuple[float, str, Dict[str, Any], List[str]]] = []
    for i, a in enumerate(official):
        partners: List[str] = []
        max_pair_sc = float(a["score"])
        for j, b in enumerate(official):
            if i == j or str(a["label"]) == str(b["label"]):
                continue
            if compute_iou_xyxy(a["bbox_xyxy"], b["bbox_xyxy"]) >= float(conflict_iou):
                partners.append(f"{b['prediction_id']}({b['label']}:{b['score']:.3f})")
                max_pair_sc = max(max_pair_sc, float(b["score"]))
        if partners:
            conflict_rows.append((max_pair_sc, str(a["prediction_id"]), a, partners))
    conflict_rows.sort(key=lambda t: (-t[0], t[1]))
    for max_sc, _pid, a, partners in conflict_rows:
        _add(a, "cross_class_conflict", 1, {"conflict_partners": partners, "max_pair_score": round(max_sc, 4)})

    # 2) Borderline official detections: 0.25 <= score < 0.45 (cap at 24 slots when cap==48)
    uncertain_cap = min(cap, max(1, cap // 2))
    uncertain_official = sorted(
        [x for x in official if float(x["score"]) < float(uncertain_high)],
        key=lambda x: (float(x["score"]), str(x["prediction_id"])),
    )
    uncertain_added = 0
    for item in uncertain_official:
        if uncertain_added >= uncertain_cap or len(selected) >= cap:
            break
        if _add(item, "low_conf_official_uncertain", 2):
            uncertain_added += 1

    # 3) Borderline below-0.25 rescue pool: 0.12 <= score < 0.25 not covered by strong official boxes
    def _covered_by_strong(cand: Mapping[str, Any]) -> bool:
        return any(
            compute_iou_xyxy(cand["bbox_xyxy"], s["bbox_xyxy"]) >= float(conflict_iou)
            for s in strong_official
        )

    sub_mid = sorted(
        [
            x for x in pool
            if float(rescue_low) <= float(x["score"]) < float(conf_official)
            and not _covered_by_strong(x)
        ],
        key=lambda x: (-float(x["score"]), str(x["prediction_id"])),
    )
    for item in sub_mid:
        if len(selected) >= cap:
            break
        _add(item, "borderline_rescue_pool", 3)

    # 4) Fallback low-conf non-overlapping 0.05..0.12
    sub_low = sorted(
        [
            x for x in pool
            if float(x["score"]) < float(rescue_low)
            and not _covered_by_strong(x)
        ],
        key=lambda x: (-float(x["score"]), str(x["prediction_id"])),
    )
    for item in sub_low:
        if len(selected) >= cap:
            break
        _add(item, "fallback_low_conf_rescue_pool", 4)

    reason_counts: Dict[str, int] = {}
    for s in selected:
        r = str(s["selection_reason"])
        reason_counts[r] = reason_counts.get(r, 0) + 1

    return {
        "max_candidates": cap,
        "selected_count": len(selected),
        "reason_counts": reason_counts,
        "selected_candidates": selected,
    }


def apply_bounded_vlm_verification(
    *,
    board_id: str,
    low_conf_pool_fics6: Sequence[Mapping[str, Any]],
    selected_candidates: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    verifier_rows: Sequence[Mapping[str, Any]],
    conf_official: float = 0.25,
    rescue_low: float = 0.12,
    nms_iou: float = 0.50,
) -> Dict[str, Any]:
    """Apply conservative bounded Qwen3 verification decisions on top of YOLO11n FICS-6 pool."""
    rows_by_uid = {str(r["call_id"]): dict(r) for r in verifier_rows if r.get("call_id")}
    man_by_pid = {str(p["prediction_id"]): dict(p) for p in manifest.get("predictions") or []}
    sel_by_pid = {str(c["prediction_id"]): dict(c) for c in selected_candidates}

    audit_records: List[Dict[str, Any]] = []
    pre_final_nms_kept: List[Dict[str, Any]] = []

    decision_counts: Dict[str, int] = {
        "unselected_yolo_kept": 0,
        "unselected_low_conf_ignored": 0,
        "confirmed_yolo_class": 0,
        "class_corrected": 0,
        "rescued_low_conf_agree": 0,
        "rejected_background": 0,
        "borderline_rejected_other": 0,
        "borderline_not_rescued_conservative": 0,
        "vlm_fallback_keep_yolo": 0,
    }

    for src in low_conf_pool_fics6:
        pid = str(src["prediction_id"])
        yolo_label = str(src["label"]).strip().lower()
        yolo_score = float(src["score"])
        is_official = yolo_score >= float(conf_official)

        if pid not in sel_by_pid:
            if is_official:
                decision_counts["unselected_yolo_kept"] += 1
                kept = dict(src)
                kept["label"] = yolo_label
                kept["yolo_original_label"] = yolo_label
                kept["yolo_original_score"] = round(yolo_score, 6)
                kept["detector_source"] = "yolo11n"
                kept["vlm_intervened"] = False
                kept["vlm_decision"] = "not_selected_kept"
                pre_final_nms_kept.append(kept)
            else:
                decision_counts["unselected_low_conf_ignored"] += 1
            continue

        sel_meta = sel_by_pid[pid]
        man_entry = man_by_pid.get(pid) or {}
        uid = man_entry.get("unique_call_id")
        vrow = rows_by_uid.get(str(uid)) if uid else None
        crop_valid = bool(man_entry.get("crop_valid"))

        vlm_raw: Optional[str] = None
        vlm_parsed_label: Optional[str] = None
        normal_eos = False
        parse_ok = False
        tech_ok = False

        if crop_valid and vrow is not None:
            raw_val = vrow.get("raw_output") if vrow.get("raw_output") is not None else vrow.get("raw_text")
            vlm_raw = str(raw_val or "")
            tech_ok = bool(vrow.get("ok")) and not bool(vrow.get("is_oom")) and not bool(vrow.get("is_timeout")) and not bool(vrow.get("watchdog_killed"))
            normal_eos = bool(vrow["has_normal_eos"]) if "has_normal_eos" in vrow else bool(vrow.get("eos_terminated", False))
            parsed = parse_single_label_json(vlm_raw, allowed_labels=ALLOWED_SEVEN_LABELS)
            parse_ok = bool(parsed.get("strict_json_ok"))
            vlm_parsed_label = parsed.get("strict_label")

        final_label: Optional[str] = None
        rescued = False
        class_changed = False
        vlm_intervened = False

        if not (crop_valid and tech_ok and normal_eos and parse_ok and vlm_parsed_label in ALLOWED_SEVEN_LABELS):
            decision = "vlm_fallback_keep_yolo"
            if is_official:
                final_label = yolo_label
        elif vlm_parsed_label == "other":
            vlm_intervened = True
            decision = "rejected_background" if is_official else "borderline_rejected_other"
            final_label = None
        else:
            # Strict valid 6-class FICS label
            vlm_intervened = True
            if is_official:
                final_label = str(vlm_parsed_label)
                if final_label == yolo_label:
                    decision = "confirmed_yolo_class"
                else:
                    decision = "class_corrected"
                    class_changed = True
            else:
                if yolo_score >= float(rescue_low) and str(vlm_parsed_label) == yolo_label:
                    decision = "rescued_low_conf_agree"
                    final_label = yolo_label
                    rescued = True
                else:
                    decision = "borderline_not_rescued_conservative"
                    final_label = None

        decision_counts[decision] += 1
        audit_records.append({
            "prediction_id": pid,
            "selection_reason": sel_meta.get("selection_reason"),
            "selection_priority": sel_meta.get("selection_priority"),
            "bbox_xyxy": src["bbox_xyxy"],
            "yolo_original_label": yolo_label,
            "yolo_original_score": round(yolo_score, 6),
            "unique_call_id": uid,
            "crop_sha256": man_entry.get("crop_sha256"),
            "crop_valid": crop_valid,
            "vlm_technical_ok": tech_ok,
            "vlm_normal_eos": normal_eos,
            "vlm_strict_json_ok": parse_ok,
            "vlm_raw_output": vlm_raw,
            "vlm_parsed_label": vlm_parsed_label,
            "final_decision": decision,
            "final_label": final_label,
            "vlm_intervened": vlm_intervened,
            "rescued_from_low_conf": rescued,
            "class_changed": class_changed,
            "vlm_gpu_ms": vrow.get("gpu_generate_ms") if vrow else None,
        })

        if final_label is not None:
            kept = dict(src)
            kept["label"] = final_label
            kept["yolo_original_label"] = yolo_label
            kept["yolo_original_score"] = round(yolo_score, 6)
            kept["detector_source"] = "yolo11n+qwen3" if vlm_intervened else "yolo11n"
            kept["vlm_intervened"] = vlm_intervened
            kept["vlm_decision"] = decision
            kept["vlm_parsed_label"] = vlm_parsed_label
            kept["rescued_from_low_conf"] = rescued
            kept["class_changed"] = class_changed
            pre_final_nms_kept.append(kept)

    final_preds = class_wise_nms_with_trace(
        pre_final_nms_kept,
        iou_threshold=float(nms_iou),
        label_key="label",
        board_id=board_id,
        id_prefix="comp",
    )

    return {
        "final_predictions": final_preds,
        "pre_final_nms_count": len(pre_final_nms_kept),
        "post_final_nms_count": len(final_preds),
        "decision_counts": decision_counts,
        "audit_records": audit_records,
    }


def run_yolo11n_class_aware_board(
    *,
    image_path: pathlib.Path,
    board_id: str,
    weights_path: pathlib.Path,
    expected_weights_sha256: str = EXPECTED_YOLO11N_BEST_SHA256,
    detector_preset: str = DEFAULT_COMPOSITE_DETECTOR_PRESET,
    expected_class_schema: Optional[str] = None,
    overlay_dir: str = DEFAULT_ULTRALYTICS_OVERLAY,
    output_dir: pathlib.Path,
    tile_size: int = 640,
    overlap_ratio: float = 0.20,
    imgsz: int = 640,
    conf_low: float = 0.05,
    conf_official: float = 0.25,
    per_class_conf: Optional[Mapping[str, float]] = None,
    nms_iou: float = 0.50,
    max_det_per_tile: int = 3000,
    gpu_gate_fn: Callable[[float, float], Dict[str, Any]] = probe_gpu_gate,
    precomputed_roi_tiles: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run GT-free tiled YOLO11n class-aware inference on one whole PCB board image."""
    t_start = time.perf_counter()
    preset_id = str(detector_preset or DEFAULT_COMPOSITE_DETECTOR_PRESET).strip() or DEFAULT_COMPOSITE_DETECTOR_PRESET
    if preset_id not in SUPPORTED_COMPOSITE_DETECTOR_PRESETS:
        raise ValueError(f"Unsupported detector_preset={preset_id!r}; expected one of {SUPPORTED_COMPOSITE_DETECTOR_PRESETS}")
    schema_key = str(expected_class_schema or COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA[preset_id]).strip()
    if schema_key not in YOLO_DETECTOR_PRESET_SCHEMAS:
        raise ValueError(f"Unsupported expected_class_schema={schema_key!r}")
    expected_id_map = YOLO_DETECTOR_PRESET_SCHEMAS[schema_key]
    detector_src_tag = (
        "yolo11n_fics_r15_ic_cal"
        if preset_id == "fics_yolo11n_r15_ic_cal"
        else ("yolo11n_fics_r15" if preset_id == "fics_yolo11n_r15" else "yolo11n")
    )

    img_p = pathlib.Path(image_path)
    w_p = pathlib.Path(weights_path)
    if not img_p.is_file():
        raise FileNotFoundError(f"Board image not found: {img_p}")
    if not w_p.is_file():
        raise FileNotFoundError(f"YOLO11n weight file not found: {w_p}")
    actual_w_sha = file_sha256(w_p).lower()
    if expected_weights_sha256 and actual_w_sha != expected_weights_sha256.lower():
        raise ValueError(f"YOLO11n weights SHA-256 mismatch: {actual_w_sha} != {expected_weights_sha256}")

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = out_dir / "tiles"

    if precomputed_roi_tiles is not None:
        geom = dict(precomputed_roi_tiles)
    else:
        geom = compute_board_roi_and_tiles(
            img_p,
            roi_pad=32,
            white_thresh=248,
            tile_size=int(tile_size),
            overlap_ratio=float(overlap_ratio),
            export_dir=tiles_dir,
            board_key=board_id,
        )

    gate = gpu_gate_fn(4096.0, 6144.0)
    (out_dir / "gpu_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not bool(gate.get("passed")):
        raise RuntimeError(f"GPU VRAM gate failed: {gate}")

    smi_before = query_nvidia_smi_used_mib(0)
    if overlay_dir and pathlib.Path(overlay_dir).is_dir() and overlay_dir not in sys.path:
        sys.path.insert(0, overlay_dir)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    t_load0 = time.perf_counter()
    model = YOLO(str(w_p))
    model_names = getattr(model, "names", None) or {}
    norm_model_names = {int(k): str(v).strip().lower() for k, v in model_names.items()}
    if norm_model_names != expected_id_map:
        raise ValueError(
            f"Unexpected YOLO model.names for schema={schema_key!r}: {norm_model_names!r} != {expected_id_map!r}"
        )

    predict_params = dict(
        imgsz=int(imgsz),
        conf=float(conf_low),
        iou=float(nms_iou),
        max_det=int(max_det_per_tile),
        device=0,
        half=False,
        rect=False,
        augment=False,
        agnostic_nms=False,
        verbose=False,
    )
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    model.predict(np.zeros((int(imgsz), int(imgsz), 3), dtype=np.uint8), **predict_params)
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
    load_s = round(time.perf_counter() - t_load0, 3)

    shapes: List[List[int]] = []
    hook_handle = model.predictor.model.model.register_forward_pre_hook(
        lambda module, args: shapes.append(list(args[0].shape))
    )

    orig_w, orig_h = int(geom["orig_wh"][0]), int(geom["orig_wh"][1])
    roi_off_x = int(geom["roi_offset_x"])
    roi_off_y = int(geom["roi_offset_y"])

    raw_jsonl_path = out_dir / "yolo_raw_outputs_all.jsonl"
    pre_nms_all: List[Dict[str, Any]] = []
    peak_smi = smi_before or 0.0
    cap_hits = 0

    try:
        with raw_jsonl_path.open("w", encoding="utf-8") as fh:
            for call_idx, tile in enumerate(geom["tiles"]):
                tid = int(tile["tile_id"])
                tp = pathlib.Path(tile["path"])
                actual_tile_sha = file_sha256(tp).lower()
                if actual_tile_sha != str(tile["sha256"]).lower():
                    raise ValueError(f"Tile SHA mismatch for {tp}")

                with Image.open(tp) as im_raw:
                    pil_img = im_raw.convert("RGB")
                tw, th = pil_img.size

                if torch.cuda.is_available():
                    torch.cuda.synchronize(0)
                t0 = time.perf_counter()
                res = model.predict(pil_img, **predict_params)[0]
                if torch.cuda.is_available():
                    torch.cuda.synchronize(0)
                gpu_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                boxes_xyxy = res.boxes.xyxy.cpu().tolist()
                scores = res.boxes.conf.cpu().tolist()
                cls_ids = [int(c) for c in res.boxes.cls.cpu().tolist()]
                if len(boxes_xyxy) >= int(max_det_per_tile):
                    cap_hits += 1

                tile_objs: List[Dict[str, Any]] = []
                off_x = roi_off_x + int(tile["tile_offset_x"])
                off_y = roi_off_y + int(tile["tile_offset_y"])
                cid = f"yolo11n_{board_id}_T{tid:02d}"

                raw_rows: List[Dict[str, Any]] = []
                for b_idx, (bx, sc, rc_id) in enumerate(zip(boxes_xyxy, scores, cls_ids)):
                    x1 = max(0.0, min(float(tw), float(bx[0])))
                    y1 = max(0.0, min(float(th), float(bx[1])))
                    x2 = max(0.0, min(float(tw), float(bx[2])))
                    y2 = max(0.0, min(float(th), float(bx[3])))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cls_info = normalize_yolo_raw_class(rc_id, model_names, expected_schema=schema_key)
                    raw_rows.append({
                        "orig_idx": b_idx,
                        "local_bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "score": round(float(sc), 6),
                        **cls_info,
                    })
                raw_rows.sort(key=lambda r: (-float(r["score"]), int(r["orig_idx"])))

                for tb_idx, row in enumerate(raw_rows):
                    gbox = map_tile_box_to_global(
                        row["local_bbox_xyxy"],
                        offset_x=off_x,
                        offset_y=off_y,
                        img_w=orig_w,
                        img_h=orig_h,
                    )
                    if gbox[2] <= gbox[0] or gbox[3] <= gbox[1]:
                        continue
                    obj = {
                        "tile_box_index": tb_idx,
                        "local_bbox_xyxy": row["local_bbox_xyxy"],
                        "bbox_xyxy": [round(float(v), 2) for v in gbox],
                        "score": row["score"],
                        "raw_yolo_class_id": row["raw_yolo_class_id"],
                        "raw_yolo_label": row["raw_yolo_label"],
                        "aligned_fics6_label": row["aligned_fics6_label"],
                        "is_fics6_aligned": row["is_fics6_aligned"],
                    }
                    tile_objs.append(obj)
                    pre_nms_all.append({
                        "pre_nms_id": f"{board_id}_yolo_raw{len(pre_nms_all):05d}",
                        "board_id": board_id,
                        "image_id": 1,
                        "tile_id": tid,
                        "call_id": cid,
                        "tile_box_index": tb_idx,
                        "label": row["aligned_fics6_label"] or row["raw_yolo_label"],
                        "score": row["score"],
                        "local_bbox_xyxy": row["local_bbox_xyxy"],
                        "bbox_xyxy": [round(float(v), 2) for v in gbox],
                        "raw_yolo_class_id": row["raw_yolo_class_id"],
                        "raw_yolo_label": row["raw_yolo_label"],
                        "aligned_fics6_label": row["aligned_fics6_label"],
                        "is_fics6_aligned": row["is_fics6_aligned"],
                        "detector_source": detector_src_tag,
                        "detector_preset": preset_id,
                    })

                cur_smi = query_nvidia_smi_used_mib(0)
                if cur_smi is not None:
                    peak_smi = max(peak_smi, cur_smi)

                rec = {
                    "call_id": cid,
                    "call_index": call_idx,
                    "board_id": board_id,
                    "image_id": 1,
                    "mode": "COMPOSITE_YOLO",
                    "tile_id": tid,
                    "image_path": str(tp),
                    "input_sha256": actual_tile_sha,
                    "width": tw,
                    "height": th,
                    "roi_offset_x": roi_off_x,
                    "roi_offset_y": roi_off_y,
                    "tile_offset_x": int(tile["tile_offset_x"]),
                    "tile_offset_y": int(tile["tile_offset_y"]),
                    "actual_input_shape": shapes[-1] if shapes else [1, 3, int(imgsz), int(imgsz)],
                    "detections": tile_objs,
                    "parsed_count": len(tile_objs),
                    "raw_output": json.dumps(tile_objs, ensure_ascii=False),
                    "gpu_generate_ms": gpu_ms,
                    "native_speed_ms": dict(res.speed) if hasattr(res, "speed") and res.speed else {},
                    "ok": True,
                    "is_oom": False,
                    "is_timeout": False,
                    "watchdog_killed": False,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        hook_handle.remove()

    alloc_mib = round(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0), 2) if torch.cuda.is_available() else 0.0
    resv_mib = round(torch.cuda.max_memory_reserved(0) / (1024.0 * 1024.0), 2) if torch.cuda.is_available() else 0.0
    smi_after = query_nvidia_smi_used_mib(0)
    if smi_after is not None:
        peak_smi = max(peak_smi, smi_after)

    nms_all = class_wise_nms_with_trace(
        pre_nms_all,
        iou_threshold=float(nms_iou),
        label_key="raw_yolo_label",
        board_id=board_id,
        id_prefix="yc",
    )
    split = split_aligned_and_extra_predictions(
        nms_all,
        conf_official=float(conf_official),
        per_class_conf=per_class_conf,
    )
    wall_s = round(time.perf_counter() - t_start, 3)
    class_conf_map = {
        c: float((per_class_conf or {}).get(c, conf_official))
        for c in FICS_SIX_CLASSES
    }

    summary = {
        "schema_version": "r16_yolo11n_class_aware_v1",
        "board_id": board_id,
        "detector_preset": preset_id,
        "expected_class_schema": schema_key,
        "qwen3_participated": False,
        "source_image_path": str(img_p),
        "source_image_sha256": file_sha256(img_p).lower(),
        "weights_path": str(w_p),
        "weights_sha256": actual_w_sha,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "geometry": {
            "orig_wh": geom["orig_wh"],
            "roi_bbox_half_open": geom["roi_bbox_half_open"],
            "roi_offset_x": roi_off_x,
            "roi_offset_y": roi_off_y,
            "roi_width": int(geom["roi_width"]),
            "roi_height": int(geom["roi_height"]),
            "tile_size": int(tile_size),
            "overlap_ratio": float(overlap_ratio),
            "tile_count": int(len(geom["tiles"])),
        },
        "operating_point": {
            "imgsz": int(imgsz),
            "conf_low_pool": float(conf_low),
            "conf_official": float(conf_official),
            "class_conf_thresholds": class_conf_map,
            "nms_iou": float(nms_iou),
            "class_wise_nms": True,
            "top_k_cap": None,
            "extra_wacv_classes_merged_into_fics6": False,
        },
        "counts": {
            "pre_nms_all": len(pre_nms_all),
            "post_nms_all_005": len(nms_all),
            "low_conf_pool_fics6_005": len(split["low_conf_pool_fics6"]),
            "official_fics6_025": len(split["official_predictions_fics6"]),
            "non_fics_extra_005": len(split["raw_non_fics_detections_005"]),
            "non_fics_extra_025": len(split["raw_non_fics_detections_025"]),
            "non_fics_class_counts_005": split["non_fics_class_counts_005"],
            "non_fics_class_counts_025": split["non_fics_class_counts_025"],
            "cap_hits": cap_hits,
        },
        "timing_and_vram": {
            "load_and_warmup_seconds": load_s,
            "total_wall_seconds": wall_s,
            "cuda_max_memory_allocated_mib": alloc_mib,
            "cuda_max_memory_reserved_mib": resv_mib,
            "nvidia_smi_before_mib": smi_before,
            "nvidia_smi_peak_mib": round(peak_smi, 2) if peak_smi else None,
            "nvidia_smi_after_mib": smi_after,
        },
    }

    (out_dir / "yolo_low_conf_pool_005.json").write_text(
        json.dumps(split["low_conf_pool_fics6"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "yolo_official_025.json").write_text(
        json.dumps(split["official_predictions_fics6"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "yolo_raw_non_fics_extra.json").write_text(
        json.dumps({
            "conf_005": split["raw_non_fics_detections_005"],
            "conf_025": split["raw_non_fics_detections_025"],
            "counts_005": split["non_fics_class_counts_005"],
            "counts_025": split["non_fics_class_counts_025"],
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "yolo_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary,
        "geometry": geom,
        "raw_jsonl_path": str(raw_jsonl_path),
        **split,
    }


_COMPOSITE_EXTRA_OUT_OF_SCOPE: Tuple[Tuple[str, str], ...] = (
    ("LED(非FICS六类)", r"\bleds?\b|发光二极管|指示灯"),
    ("时钟器件/clock(非FICS六类)", r"\bclocks?\b|时钟芯片"),
)


def parse_composite_query(query_text: str) -> Dict[str, Any]:
    """Parse natural-language class (single/multi/all-6) and original-image spatial region query."""
    cleaned = _strip_spaces(query_text)
    if not cleaned:
        return {
            "ok": False,
            "error_code": "empty_query",
            "error_message": "查询指令不能为空。示例：「只找电阻，但只统计左半边」「只找电容并计数」「找电阻和电容」。",
            "raw_query": cleaned,
        }
    low = cleaned.lower()

    for scope_name, pat in _COMPOSITE_EXTRA_OUT_OF_SCOPE + _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "out_of_scope_class",
                "error_message": f"查询目标超出 FICS-PCB 六类标准词表：{scope_name}（不会将 LED/connector 并入二极管或其他类别）。",
                "raw_query": cleaned,
            }
    for pat in _NEGATION_PHRASES:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "exclusion_unsupported",
                "error_message": "暂不支持否定/排除类表达（如“除了/不要”），请直接指定要查找的类别与区域。",
                "raw_query": cleaned,
            }
    for pat in _UNSUPPORTED_COMPOSITE_REGION_PATTERNS:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "unsupported_spatial_region",
                "error_message": f"当前复合空间查询支持“左半边 / 右半边 / 上半边 / 下半边”（未开启区域词：{pat}）。",
                "raw_query": cleaned,
            }

    has_left = any(re.search(p, low) for p in _LEFT_REGION_PATTERNS)
    has_right = any(re.search(p, low) for p in _RIGHT_REGION_PATTERNS)
    has_top = any(re.search(p, low) for p in _TOP_REGION_PATTERNS)
    has_bottom = any(re.search(p, low) for p in _BOTTOM_REGION_PATTERNS)

    active_regions = [
        name for name, flag in (
            ("left_half", has_left),
            ("right_half", has_right),
            ("top_half", has_top),
            ("bottom_half", has_bottom),
        ) if flag
    ]
    if len(active_regions) >= 2:
        return {
            "ok": False,
            "error_code": "ambiguous_spatial_region",
            "error_message": f"空间约束存在多个区域词：{active_regions}，请仅指定单个半边区域。",
            "raw_query": cleaned,
        }
    region_mode = active_regions[0] if active_regions else "unconstrained"

    wants_all = any(re.search(p, low) for p in _ALL_CLASSES_PHRASES)
    matched_classes = [cls for cls, pats in _CLASS_PATTERNS if any(re.search(p, low) for p in pats)]
    if wants_all or (region_mode != "unconstrained" and not matched_classes and any(w in low for w in ("器件", "元器件", "元件", "component", "统计", "全部", "所有"))):
        target_classes = list(FICS_SIX_CLASSES)
    elif matched_classes:
        target_classes = matched_classes
    else:
        return {
            "ok": False,
            "error_code": "unrecognized_target_class",
            "error_message": "未识别到 FICS-PCB 六类目标器件（电容/电阻/电感/晶体管/二极管/集成电路）。",
            "raw_query": cleaned,
        }

    count_requested = any(re.search(p, low) for p in _COUNT_PATTERNS)
    target_cn = [CLASS_CN_DISPLAY.get(c, c) for c in target_classes]
    return {
        "ok": True,
        "raw_query": cleaned,
        "target_classes": target_classes,
        "target_classes_cn": target_cn,
        "region_mode": region_mode,
        "region_cn": COMPOSITE_REGION_NAMES[region_mode],
        "count_requested": count_requested,
        "normalized_summary": f"目标={'/'.join(target_classes)} | 区域={region_mode} | 计数={'是' if count_requested else '默认输出'}",
    }


def box_in_composite_region(
    bbox_xyxy: Sequence[float],
    *,
    orig_wh: Sequence[int],
    region_mode: str,
) -> Tuple[bool, Dict[str, float]]:
    """Evaluate box center membership in original-image coordinates [0..W, 0..H]."""
    orig_w, orig_h = float(orig_wh[0]), float(orig_wh[1])
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    rm = str(region_mode or "unconstrained").lower()
    if rm == "left_half":
        inside = cx < (orig_w / 2.0)
    elif rm == "right_half":
        inside = cx >= (orig_w / 2.0)
    elif rm == "top_half":
        inside = cy < (orig_h / 2.0)
    elif rm == "bottom_half":
        inside = cy >= (orig_h / 2.0)
    elif rm == "unconstrained":
        inside = True
    else:
        raise ValueError(f"Unsupported region_mode={region_mode!r}")
    return inside, {"cx": round(cx, 2), "cy": round(cy, 2)}


def execute_composite_query(
    *,
    predictions: Sequence[Mapping[str, Any]],
    query_text: str,
    orig_wh: Sequence[int],
    base_job_mode: str,
    vlm_calls_count: int = 0,
    query_id: Optional[str] = None,
    detector_preset: str = "wacv_yolo11n",
    yolo_weights_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Filter composite detections by natural-language class and original-image spatial constraints."""
    parsed = parse_composite_query(query_text)
    if not parsed.get("ok"):
        return {
            "ok": False,
            **parsed,
        }

    target_set = set(parsed["target_classes"])
    region_mode = str(parsed["region_mode"])
    orig_w, orig_h = int(orig_wh[0]), int(orig_wh[1])

    matched_boxes: List[Dict[str, Any]] = []
    class_matched_outside_region = 0
    for item in predictions:
        lbl = str(item.get("label") or "").strip().lower()
        if lbl not in target_set:
            continue
        inside, center = box_in_composite_region(
            item["bbox_xyxy"],
            orig_wh=(orig_w, orig_h),
            region_mode=region_mode,
        )
        if not inside:
            class_matched_outside_region += 1
            continue
        cp = dict(item)
        cp["center_xy"] = [center["cx"], center["cy"]]
        cp["detector_source"] = str(cp.get("detector_source") or "yolo11n")
        cp["vlm_intervened"] = bool(cp.get("vlm_intervened", False))
        cp["box_source"] = str(cp.get("box_source") or ("r15_official_025" if cp["detector_source"] == "yolo11n" else cp["detector_source"]))
        cp["is_low_conf_supplement"] = bool(cp.get("is_low_conf_supplement", False))
        if cp.get("r20_relabel_decision"):
            cp["r20_relabel_decision"] = str(cp["r20_relabel_decision"])
        if cp.get("r20_relabel_source"):
            cp["r20_relabel_source"] = str(cp["r20_relabel_source"])
        if "is_multiscale_448_supplement" in cp:
            cp["is_multiscale_448_supplement"] = bool(cp["is_multiscale_448_supplement"])
        if "scale_tile_size" in cp:
            cp["scale_tile_size"] = int(cp["scale_tile_size"])
        if "source_tile_id" in cp and cp["source_tile_id"] is not None:
            cp["source_tile_id"] = cp["source_tile_id"]
        matched_boxes.append(cp)

    class_counts = {c: 0 for c in FICS_SIX_CLASSES}
    source_counts: Dict[str, int] = {}
    box_source_counts: Dict[str, int] = {}
    low_conf_supplement_count = 0
    vlm_intervened_count = 0
    for m in matched_boxes:
        lbl = str(m["label"])
        if lbl in class_counts:
            class_counts[lbl] += 1
        src = str(m["detector_source"])
        source_counts[src] = source_counts.get(src, 0) + 1
        bsrc = str(m["box_source"])
        box_source_counts[bsrc] = box_source_counts.get(bsrc, 0) + 1
        if m.get("is_low_conf_supplement"):
            low_conf_supplement_count += 1
        if m.get("vlm_intervened"):
            vlm_intervened_count += 1

    split_desc = {
        "unconstrained": f"整图全幅 [0..{orig_w}, 0..{orig_h}]",
        "left_half": f"原图坐标系左半边：框中心 cx < {orig_w / 2.0:.1f} px (原图宽 {orig_w} px)",
        "right_half": f"原图坐标系右半边：框中心 cx >= {orig_w / 2.0:.1f} px (原图宽 {orig_w} px)",
        "top_half": f"原图坐标系上半边：框中心 cy < {orig_h / 2.0:.1f} px (原图高 {orig_h} px)",
        "bottom_half": f"原图坐标系下半边：框中心 cy >= {orig_h / 2.0:.1f} px (原图高 {orig_h} px)",
    }[region_mode]

    if str(base_job_mode) == "COMPOSITE_YOLO_QWEN3":
        exec_path = "composite_with_qwen3_bounded_verify"
        exec_label = "筛选已生成的检测结果（YOLO11n + Qwen3 有限难样本核验后结果 · CPU 筛选，本次查询不新增 GPU 调用）"
    else:
        exec_path = "composite_existing_result_filter"
        exec_label = "筛选已生成的检测结果（CPU，本次查询不新增 GPU 调用）"

    return {
        "ok": True,
        "query_id": query_id or f"cq_{hashlib.sha1(parsed['raw_query'].encode('utf-8')).hexdigest()[:8]}",
        "query_spec": parsed,
        "query_execution_mechanism": "筛选已生成的检测结果",
        "execution_path": exec_path,
        "execution_path_label": exec_label,
        "gpu_called_for_query": False,
        "base_job_mode": str(base_job_mode),
        "base_job_vlm_calls_count": int(vlm_calls_count),
        "detector_preset": str(detector_preset or "wacv_yolo11n"),
        "yolo_weights_sha256": yolo_weights_sha256,
        "model_participation": {
            "stage1_detector_preset": str(detector_preset or "wacv_yolo11n"),
            "stage1_weights_sha256": yolo_weights_sha256,
            "stage2_vlm_bounded_verify": str(base_job_mode) == "COMPOSITE_YOLO_QWEN3",
            "query_stage_gpu_called": False,
            "query_mechanism": "筛选已生成的检测结果",
        },
        "coordinate_convention": {
            "coord_frame": "original_image_xyxy",
            "orig_wh": [orig_w, orig_h],
            "center_rule": "cx=(x1+x2)/2, cy=(y1+y2)/2",
            "split_axis": "x" if region_mode in ("left_half", "right_half") else ("y" if region_mode in ("top_half", "bottom_half") else None),
            "split_value_px": (orig_w / 2.0) if region_mode in ("left_half", "right_half") else ((orig_h / 2.0) if region_mode in ("top_half", "bottom_half") else None),
            "explanation": split_desc,
        },
        "predicted_count": len(matched_boxes),
        "class_matched_excluded_by_region": class_matched_outside_region,
        "class_counts": class_counts,
        "detector_source_breakdown": source_counts,
        "box_source_breakdown": box_source_counts,
        "low_conf_supplement_count": low_conf_supplement_count,
        "vlm_intervened_box_count": vlm_intervened_count,
        "predictions": matched_boxes,
        "gt_loaded": False,
    }


def render_composite_query_overlay_png(
    *,
    roi_image_path: pathlib.Path,
    roi_offset_x: int,
    roi_offset_y: int,
    orig_wh: Sequence[int],
    predictions: Sequence[Mapping[str, Any]],
    region_mode: str = "unconstrained",
) -> bytes:
    """Render composite overlay highlighting VLM-intervened boxes and spatial half split line."""
    import io

    with Image.open(roi_image_path) as im:
        rgb = im.convert("RGB")
        max_side = 2400
        scale = 1.0
        if max(rgb.width, rgb.height) > max_side:
            scale = max_side / float(max(rgb.width, rgb.height))
            rgb = rgb.resize(
                (max(1, int(round(rgb.width * scale))), max(1, int(round(rgb.height * scale)))),
                Image.Resampling.BILINEAR,
            )

    draw = ImageDraw.Draw(rgb)
    font = ImageFont.load_default()
    stroke_w = max(2, int(round(max(rgb.width, rgb.height) / 550)))

    orig_w, orig_h = float(orig_wh[0]), float(orig_wh[1])
    if region_mode in ("left_half", "right_half"):
        split_x = ((orig_w / 2.0) - float(roi_offset_x)) * scale
        if 0 <= split_x <= rgb.width:
            for y in range(0, rgb.height, 18):
                draw.line([(split_x, y), (split_x, min(rgb.height, y + 10))], fill=(250, 204, 21), width=max(2, stroke_w))
    elif region_mode in ("top_half", "bottom_half"):
        split_y = ((orig_h / 2.0) - float(roi_offset_y)) * scale
        if 0 <= split_y <= rgb.height:
            for x in range(0, rgb.width, 18):
                draw.line([(x, split_y), (min(rgb.width, x + 10), split_y)], fill=(250, 204, 21), width=max(2, stroke_w))

    palette: Dict[str, Tuple[int, int, int]] = {
        "capacitor": (56, 189, 248),
        "resistor": (34, 197, 94),
        "inductor": (250, 204, 21),
        "transistor": (192, 132, 252),
        "diode": (251, 146, 60),
        "ic": (45, 212, 191),
    }
    for pred in predictions:
        box = pred.get("bbox_xyxy")
        if not box or len(box) != 4:
            continue
        x1 = max(0.0, min(rgb.width - 1.0, (float(box[0]) - roi_offset_x) * scale))
        y1 = max(0.0, min(rgb.height - 1.0, (float(box[1]) - roi_offset_y) * scale))
        x2 = max(x1 + 2.0, min(float(rgb.width), (float(box[2]) - roi_offset_x) * scale))
        y2 = max(y1 + 2.0, min(float(rgb.height), (float(box[3]) - roi_offset_y) * scale))
        lbl = str(pred.get("label") or "?").lower()
        col = palette.get(lbl, (56, 189, 248))
        if pred.get("vlm_intervened"):
            col = (217, 70, 239)
        draw.rectangle([x1, y1, x2, y2], outline=col, width=stroke_w)
        sc = pred.get("score")
        tag = f"{lbl[:3]} {float(sc):.2f}" if sc is not None else lbl
        if pred.get("vlm_intervened"):
            tag = f"[VLM] {tag}"
        ty = max(2.0, y1 - 12.0)
        draw.rectangle([x1, ty, min(float(rgb.width), x1 + len(tag) * 6.5 + 6), ty + 12.0], fill=(15, 23, 42))
        draw.text((x1 + 2.0, ty + 1.0), tag, fill=col, font=font)

    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


def _prf1(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "precision_pct": round(prec * 100.0, 2),
        "recall_pct": round(rec * 100.0, 2),
        "f1_pct": round(f1 * 100.0, 2),
    }


def evaluate_composite_board(
    *,
    gts: Sequence[Mapping[str, Any]],
    predictions_025: Sequence[Mapping[str, Any]],
    low_conf_pool_005: Optional[Sequence[Mapping[str, Any]]] = None,
    roi_bbox_half_open: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Compute class-agnostic coverage, class-aware metrics, per-class/size stats, and FN error decomposition."""
    # Canonical class-aware 1-to-1 IoU>=0.50
    all_matched_pred_idx = set()
    all_matched_gt_idx = set()
    matched_pairs: List[Dict[str, Any]] = []
    per_class: Dict[str, Dict[str, Any]] = {}

    gt_list = [dict(g) for g in gts]
    pred_list = [dict(p) for p in predictions_025]

    for cls in FICS_SIX_CLASSES:
        gt_indices = [i for i, g in enumerate(gt_list) if str(g["label"]).lower() == cls]
        pred_indices = [i for i, p in enumerate(pred_list) if str(p["label"]).lower() == cls]
        res = repo_eval.match_category_predictions_one_to_one(
            [{"bbox": gt_list[i]["bbox_xyxy"], "category": cls} for i in gt_indices],
            [{"bbox": pred_list[i]["bbox_xyxy"], "category": cls, "score": float(pred_list[i].get("score", 0.0))} for i in pred_indices],
            iou_threshold=0.50,
        )
        for p_loc, g_loc, ov in res["matched_pairs"]:
            gi = gt_indices[g_loc]
            pi = pred_indices[p_loc]
            all_matched_gt_idx.add(gi)
            all_matched_pred_idx.add(pi)
            matched_pairs.append({
                "gt_idx": gi,
                "gt_id": gt_list[gi].get("id"),
                "pred_idx": pi,
                "prediction_id": pred_list[pi].get("prediction_id"),
                "label": cls,
                "iou": round(float(ov), 4),
            })
        cls_m = _prf1(res["tp"], res["fp"], res["fn"])
        per_class[cls] = {
            "gt_count": len(gt_indices),
            "pred_count": len(pred_indices),
            "count_abs_error": abs(len(pred_indices) - len(gt_indices)),
            **cls_m,
        }

    overall = _prf1(len(all_matched_pred_idx), len(pred_list) - len(all_matched_pred_idx), len(gt_list) - len(all_matched_gt_idx))
    overall["gt_total"] = len(gt_list)
    overall["pred_total"] = len(pred_list)
    overall["count_abs_error"] = abs(len(pred_list) - len(gt_list))

    # Class-agnostic candidate coverage at conf>=0.25 and conf>=0.05 (IoU 0.50 & 0.25)
    def _ca(preds_seq: Sequence[Mapping[str, Any]], iou_t: float) -> Dict[str, Any]:
        m = repo_eval.match_category_predictions_one_to_one(
            [{"bbox": g["bbox_xyxy"], "category": "any"} for g in gt_list],
            [{"bbox": p["bbox_xyxy"], "category": "any", "score": float(p.get("score", 0.0))} for p in preds_seq],
            iou_threshold=iou_t,
        )
        rec = (m["tp"] / len(gt_list)) if gt_list else 0.0
        return {"matched_gt": m["tp"], "gt_total": len(gt_list), "recall": round(rec, 6), "recall_pct": round(rec * 100.0, 2), "pred_count": len(preds_seq)}

    ca_025_iou50 = _ca(pred_list, 0.50)
    ca_025_iou25 = _ca(pred_list, 0.25)
    pool_005 = [dict(x) for x in low_conf_pool_005] if low_conf_pool_005 is not None else pred_list
    ca_005_iou50 = _ca(pool_005, 0.50)
    ca_005_iou25 = _ca(pool_005, 0.25)

    # Error decomposition for missed GTs (FN)
    fn_wrong_class = 0
    fn_low_iou = 0
    fn_score_filtered_005 = 0
    fn_true_miss = 0
    wrong_class_confusions: Dict[str, int] = {}

    for gi, g in enumerate(gt_list):
        if gi in all_matched_gt_idx:
            continue
        gbox = g["bbox_xyxy"]
        glbl = str(g["label"]).lower()
        best_025_iou = 0.0
        best_025_pred = None
        for p in pred_list:
            ov = compute_iou_xyxy(gbox, p["bbox_xyxy"])
            if ov > best_025_iou:
                best_025_iou = ov
                best_025_pred = p
        if best_025_iou >= 0.50 and best_025_pred is not None:
            fn_wrong_class += 1
            ck = f"{glbl}->{str(best_025_pred['label']).lower()}"
            wrong_class_confusions[ck] = wrong_class_confusions.get(ck, 0) + 1
        elif best_025_iou >= 0.25:
            fn_low_iou += 1
        else:
            best_005_same = 0.0
            best_005_any = 0.0
            for p in pool_005:
                ov = compute_iou_xyxy(gbox, p["bbox_xyxy"])
                if ov > best_005_any:
                    best_005_any = ov
                if str(p["label"]).lower() == glbl and ov > best_005_same:
                    best_005_same = ov
            if best_005_any >= 0.50:
                fn_score_filtered_005 += 1
            else:
                fn_true_miss += 1

    # Duplicate FP on already-matched GT
    dup_fp_same_cls = 0
    dup_fp_diff_cls = 0
    for pi, p in enumerate(pred_list):
        if pi in all_matched_pred_idx:
            continue
        for gi in all_matched_gt_idx:
            if compute_iou_xyxy(p["bbox_xyxy"], gt_list[gi]["bbox_xyxy"]) >= 0.50:
                if str(p["label"]).lower() == str(gt_list[gi]["label"]).lower():
                    dup_fp_same_cls += 1
                else:
                    dup_fp_diff_cls += 1
                break

    # Per-size-bin breakdown (original-image pixel area)
    def _size_bin(bbox_xyxy: Sequence[float]) -> str:
        area = max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0])) * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
        if area < 2500.0:
            return "small"
        if area < 20000.0:
            return "medium"
        return "large"

    per_size_bin: Dict[str, Dict[str, Any]] = {}
    for bname, desc in (
        ("small", "area < 2500 px^2 (<50x50)"),
        ("medium", "2500 <= area < 20000 px^2"),
        ("large", "area >= 20000 px^2"),
    ):
        g_idx_bin = [i for i, g in enumerate(gt_list) if _size_bin(g["bbox_xyxy"]) == bname]
        tp_b = sum(1 for i in g_idx_bin if i in all_matched_gt_idx)
        fn_b = len(g_idx_bin) - tp_b
        ca025_b = sum(
            1
            for i in g_idx_bin
            if any(compute_iou_xyxy(gt_list[i]["bbox_xyxy"], p["bbox_xyxy"]) >= 0.50 for p in pred_list)
        )
        ca005_b = sum(
            1
            for i in g_idx_bin
            if any(compute_iou_xyxy(gt_list[i]["bbox_xyxy"], p["bbox_xyxy"]) >= 0.50 for p in pool_005)
        )
        rec_b = (tp_b / len(g_idx_bin)) if g_idx_bin else 0.0
        per_size_bin[bname] = {
            "rule": desc,
            "gt_count": len(g_idx_bin),
            "tp": tp_b,
            "fn": fn_b,
            "recall": round(rec_b, 6),
            "recall_pct": round(rec_b * 100.0, 2),
            "ca_025_iou50_any_pred_covered": ca025_b,
            "ca_005_iou50_any_pred_covered": ca005_b,
        }

    roi_gt_audit: Optional[Dict[str, Any]] = None
    if roi_bbox_half_open is not None and len(roi_bbox_half_open) == 4:
        rx1, ry1, rx2, ry2 = [float(v) for v in roi_bbox_half_open]
        gt_out_roi = 0
        gt_part_roi = 0
        for g in gt_list:
            gx1, gy1, gx2, gy2 = [float(v) for v in g["bbox_xyxy"]]
            ga = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
            ia = max(0.0, min(gx2, rx2) - max(gx1, rx1)) * max(0.0, min(gy2, ry2) - max(gy1, ry1))
            if ia <= 0.0:
                gt_out_roi += 1
            elif ga > 0.0 and (ia / ga) < 0.999:
                gt_part_roi += 1
        roi_gt_audit = {
            "roi_bbox_half_open": [int(roi_bbox_half_open[0]), int(roi_bbox_half_open[1]), int(roi_bbox_half_open[2]), int(roi_bbox_half_open[3])],
            "gt_outside_roi": gt_out_roi,
            "gt_partially_clipped_by_roi": gt_part_roi,
        }

    return {
        "class_aware": overall,
        "per_class": per_class,
        "per_size_bin": per_size_bin,
        "class_agnostic_coverage": {
            "official_025_iou50": ca_025_iou50,
            "official_025_iou25": ca_025_iou25,
            "pool_005_iou50": ca_005_iou50,
            "pool_005_iou25": ca_005_iou25,
        },
        "error_decomposition": {
            "fn_total": overall["fn"],
            "fn_wrong_class_at_iou50": fn_wrong_class,
            "fn_wrong_class_confusions": wrong_class_confusions,
            "fn_low_iou_025_to_050": fn_low_iou,
            "fn_score_filtered_below_025_present_in_005": fn_score_filtered_005,
            "fn_true_localization_miss": fn_true_miss,
            "roi_gt_audit": roi_gt_audit,
            "fp_total": overall["fp"],
            "fp_duplicate_same_class_on_matched_gt": dup_fp_same_cls,
            "fp_conflict_diff_class_on_matched_gt": dup_fp_diff_cls,
        },
        "matched_pairs": matched_pairs,
    }


def evaluate_three_way_comparison(
    *,
    per_board_results: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute combined Micro-F1 across boards and evaluate pre-registered promotion gate."""
    def _combine(arm_key: str) -> Dict[str, Any]:
        tp = sum(int(per_board_results[b][arm_key]["class_aware"]["tp"]) for b in per_board_results)
        fp = sum(int(per_board_results[b][arm_key]["class_aware"]["fp"]) for b in per_board_results)
        fn = sum(int(per_board_results[b][arm_key]["class_aware"]["fn"]) for b in per_board_results)
        return _prf1(tp, fp, fn)

    combined = {
        arm: _combine(arm)
        for arm in ("qwen3_r5", "yolo11n_direct", "yolo11n_qwen3_composite")
    }
    yolo_f1 = combined["yolo11n_direct"]["f1"]
    comp_f1 = combined["yolo11n_qwen3_composite"]["f1"]
    delta_f1_pp = round((comp_f1 - yolo_f1) * 100.0, 2)
    tp_no_drop = all(
        int(per_board_results[b]["yolo11n_qwen3_composite"]["class_aware"]["tp"])
        >= int(per_board_results[b]["yolo11n_direct"]["class_aware"]["tp"])
        for b in per_board_results
    )
    gate_passed = (delta_f1_pp >= 3.0) and tp_no_drop

    return {
        "combined_micro": combined,
        "promotion_gate": {
            "rule": "combined_micro_f1_gain >= +3.0 pp AND no TP drop on S20 or S4",
            "delta_micro_f1_pp": delta_f1_pp,
            "no_tp_drop_on_any_board": tp_no_drop,
            "passed": gate_passed,
            "recommended_baseline": "yolo11n_qwen3_composite" if gate_passed else "yolo11n_direct",
        },
    }


def load_coco_gts_for_single_board(coco_path: pathlib.Path, *, image_id: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load COCO GT annotations for one board image cleanly."""
    doc = json.loads(pathlib.Path(coco_path).read_text(encoding="utf-8"))
    images = list(doc.get("images") or [])
    if image_id is not None:
        chosen = next((im for im in images if int(im["id"]) == int(image_id)), None)
        if chosen is None:
            raise KeyError(f"image_id={image_id} not found in {coco_path}")
    else:
        if len(images) != 1:
            raise ValueError(f"Expected 1 image in {coco_path}, found {len(images)}")
        chosen = images[0]
    tid = int(chosen["id"])
    cat_map = {int(c["id"]): str(c["name"]).strip().lower() for c in doc.get("categories") or []}
    gts: List[Dict[str, Any]] = []
    for ann in doc.get("annotations") or []:
        if int(ann["image_id"]) != tid:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        lbl = cat_map[int(ann["category_id"])]
        if lbl not in WACV_TO_FICS6_ALIGNED:
            continue
        gts.append({
            "id": int(ann["id"]),
            "label": lbl,
            "bbox_xyxy": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
        })
    return gts, chosen
