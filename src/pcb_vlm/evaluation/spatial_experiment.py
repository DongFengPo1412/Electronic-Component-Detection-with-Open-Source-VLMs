"""Experimental spatial-restriction PCB instruction parser, evaluator, and process-group deadline guard."""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pcb_vlm.evaluation.language_instruction import (
    CLASS_CN_DISPLAY,
    _ACTION_PATTERNS,
    _ALL_CLASSES_PHRASES,
    _CLASS_PATTERNS,
    _COUNT_PATTERNS,
    _NEGATION_PHRASES,
    _OUT_OF_SCOPE_PATTERNS,
    _SIX_CLASS_SET,
    _strip_spaces,
    build_single_class_detection_prompt,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import evaluate as repo_eval  # noqa: E402

SPATIAL_HALF_REGIONS: Tuple[str, ...] = ("unconstrained", "left_half", "right_half")

REGION_CN_DISPLAY: Dict[str, str] = {
    "unconstrained": "整图不限区域",
    "left_half": "图像左半边 (cx < W/2)",
    "right_half": "图像右半边 (cx >= W/2)",
}

_LEFT_HALF_PATTERNS: Tuple[str, ...] = (
    r"左半边", r"左半侧", r"左半部", r"左半区", r"左半", r"左侧", r"左边", r"\bleft\s+half\b", r"\bleft\s+side\b",
)
_RIGHT_HALF_PATTERNS: Tuple[str, ...] = (
    r"右半边", r"右半侧", r"右半部", r"右半区", r"右半", r"右侧", r"右边", r"\bright\s+half\b", r"\bright\s+side\b",
)
_UNSUPPORTED_SPATIAL_PATTERNS: Tuple[str, ...] = (
    r"左上", r"右上", r"左下", r"右下", r"上半", r"下半", r"上边", r"下边", r"上侧", r"下侧",
    r"上方", r"下方", r"中间", r"中央", r"中心", r"角落", r"边缘", r"附近", r"旁边",
    r"\btop[-\s]?(?:left|right|half)\b", r"\bbottom[-\s]?(?:left|right|half)\b",
    r"\bupper\b", r"\blower\b", r"\bcorner\b", r"\bcenter\b", r"\bmiddle\b", r"\bnear(?:by)?\b",
)


def _pr_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
    }


def _extract_xyxy(box_like: Mapping[str, Any]) -> List[float]:
    raw = box_like.get("bbox_xyxy") or box_like.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"Expected 4-element bbox in {box_like!r}")
    return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]


def build_spatial_detection_prompt(
    target_class: str,
    region_mode: str,
    *,
    coordinate_protocol: str = "xyxy_norm1000",
) -> Tuple[str, str]:
    """Build deterministic single-class prompt for unconstrained, left_half, or right_half queries."""
    cls = str(target_class or "").strip().lower()
    if cls not in _SIX_CLASS_SET:
        raise ValueError(f"Unsupported target_class={target_class!r}")
    rmode = str(region_mode or "").strip().lower()
    if rmode not in SPATIAL_HALF_REGIONS:
        raise ValueError(f"Unsupported region_mode={region_mode!r}; expected one of {SPATIAL_HALF_REGIONS}")
    if rmode == "unconstrained":
        return build_single_class_detection_prompt(
            cls, coordinate_protocol=coordinate_protocol, prompt_variant="r8_base"
        )
    if coordinate_protocol == "xyxy_norm1":
        coord_spec = "[0.0, 1.0]"
        mid = "0.5"
    elif coordinate_protocol == "xyxy_norm1000":
        coord_spec = "[0, 1000]"
        mid = "500"
    else:
        raise ValueError(f"Unsupported coordinate_protocol={coordinate_protocol!r}")

    if rmode == "left_half":
        region_desc = f"the LEFT HALF of this PCB image (horizontal box center x < {mid} in {coord_spec} coordinates)"
        exclude_desc = f"in the right half (x >= {mid})"
        empty_desc = "in the left half"
    else:
        region_desc = f"the RIGHT HALF of this PCB image (horizontal box center x >= {mid} in {coord_spec} coordinates)"
        exclude_desc = f"in the left half (x < {mid})"
        empty_desc = "in the right half"

    prompt = (
        f"Locate and count ONLY visible {cls} components located in {region_desc}. "
        f"Do NOT output any component {exclude_desc} and do NOT output any component other than {cls}. "
        f"Return a JSON array of objects with keys bbox_2d and label. "
        f"Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized to {coord_spec} relative to this full input image. "
        f'Use the exact label "{cls}" for every detected {cls}. '
        f"If no {cls} components are visible {empty_desc}, return []. Do not include explanations."
    )
    return prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_spatial_pcb_instruction(
    user_text: str,
    *,
    coordinate_protocol: str = "xyxy_norm1000",
) -> Dict[str, Any]:
    """Parse experimental half-image spatial single-class instruction on unchanged full input image."""
    cleaned = _strip_spaces(user_text)
    if not cleaned:
        return {
            "ok": False,
            "error_code": "empty_instruction",
            "error_message": "指令不能为空。示例：「只找电阻，限图像左半边并计数」。",
            "raw_instruction": cleaned,
        }
    low = cleaned.lower()

    for pat in _ALL_CLASSES_PHRASES:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "multi_class_use_standard_mode",
                "error_message": "实验性空间指令仅支持单个六类目标加左/右半边约束。",
                "raw_instruction": cleaned,
            }
    for scope_name, pat in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "out_of_scope_class",
                "error_message": f"目标超出 FICS-PCB 六类标准词表：{scope_name}。",
                "raw_instruction": cleaned,
            }
    for pat in _NEGATION_PHRASES:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "exclusion_instruction_unsupported",
                "error_message": "暂不支持否定/排除类表达，请直接指定目标类别与左/右半边。",
                "raw_instruction": cleaned,
            }

    for pat in _UNSUPPORTED_SPATIAL_PATTERNS:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "spatial_qualifier_out_of_bounded_protocol",
                "error_message": f"实验协议仅支持“图像左半边 / 图像右半边”（命中受限区域词：{pat}）。",
                "raw_instruction": cleaned,
            }

    has_left = any(re.search(p, low) for p in _LEFT_HALF_PATTERNS)
    has_right = any(re.search(p, low) for p in _RIGHT_HALF_PATTERNS)
    if has_left and has_right:
        return {
            "ok": False,
            "error_code": "ambiguous_spatial_region",
            "error_message": "指令同时包含左半与右半描述，请仅指定其中一个半区。",
            "raw_instruction": cleaned,
        }
    region_mode = "left_half" if has_left else ("right_half" if has_right else "unconstrained")

    matched_classes = [cls for cls, pats in _CLASS_PATTERNS if any(re.search(p, low) for p in pats)]
    if len(matched_classes) >= 2:
        return {
            "ok": False,
            "error_code": "multi_class_not_single_target",
            "error_message": f"实验入口仅支持单一目标类别，检测到多类：{matched_classes}",
            "raw_instruction": cleaned,
        }
    if not matched_classes:
        return {
            "ok": False,
            "error_code": "unrecognized_target_class",
            "error_message": "未识别到 FICS-PCB 六类目标器件（capacitor/resistor/inductor/transistor/diode/ic）。",
            "raw_instruction": cleaned,
        }

    target_cls = matched_classes[0]
    count_requested = any(re.search(p, low) for p in _COUNT_PATTERNS)
    has_action = any(re.search(p, low) for p in _ACTION_PATTERNS)
    if not has_action and not count_requested and len(cleaned) > 28:
        return {
            "ok": False,
            "error_code": "ambiguous_instruction_intent",
            "error_message": "指令缺少明确查找或计数意图。",
            "raw_instruction": cleaned,
        }

    prompt_text, prompt_sha = build_spatial_detection_prompt(
        target_cls, region_mode, coordinate_protocol=coordinate_protocol
    )
    return {
        "ok": True,
        "raw_instruction": cleaned,
        "target_class": target_cls,
        "target_class_cn": CLASS_CN_DISPLAY[target_cls],
        "region_mode": region_mode,
        "region_cn": REGION_CN_DISPLAY[region_mode],
        "count_requested": bool(count_requested),
        "normalized_intent": f"find_and_count_{target_cls}_{region_mode}" if count_requested else f"find_{target_cls}_{region_mode}",
        "prompt_variant": "r11_spatial_bounded_v1" if region_mode != "unconstrained" else "r8_base",
        "execution_source": "direct_vlm_spatial",
        "experimental_spatial": True,
        "input_cropped": False,
        "cross_region_boxes_preserved": True,
        "normalized_prompt": prompt_text,
        "prompt_sha256": prompt_sha,
        "coordinate_protocol": coordinate_protocol,
    }


def classify_box_half(bbox_xyxy: Sequence[float], image_width: float) -> str:
    """Assign a box cleanly by horizontal center: cx < 0.5*W -> left_half, cx >= 0.5*W -> right_half."""
    if image_width <= 0:
        raise ValueError(f"image_width must be positive, got {image_width}")
    cx = 0.5 * (float(bbox_xyxy[0]) + float(bbox_xyxy[2]))
    return "left_half" if cx < (0.5 * float(image_width)) else "right_half"


def audit_spatial_predictions(
    same_class_preds: Sequence[Mapping[str, Any]],
    *,
    image_width: float,
    region_mode: str,
) -> Dict[str, Any]:
    """Audit in-region vs cross-region boxes WITHOUT dropping out-of-region predictions."""
    rmode = str(region_mode or "unconstrained").strip().lower()
    if rmode not in SPATIAL_HALF_REGIONS:
        raise ValueError(f"Unsupported region_mode={region_mode!r}")

    annotated: List[Dict[str, Any]] = []
    in_region: List[Dict[str, Any]] = []
    cross_region: List[Dict[str, Any]] = []

    for pred in same_class_preds:
        bbox = _extract_xyxy(pred)
        cx = 0.5 * (bbox[0] + bbox[2])
        assigned_half = classify_box_half(bbox, image_width)
        is_in_region = True if rmode == "unconstrained" else (assigned_half == rmode)
        item = dict(pred)
        item["bbox_center_x"] = round(cx, 3)
        item["assigned_half"] = assigned_half
        item["in_requested_region"] = bool(is_in_region)
        annotated.append(item)
        if is_in_region:
            in_region.append(item)
        else:
            cross_region.append(item)

    total = len(annotated)
    viol_cnt = len(cross_region)
    return {
        "region_mode": rmode,
        "region_cn": REGION_CN_DISPLAY[rmode],
        "image_width": float(image_width),
        "center_split_x": round(0.5 * float(image_width), 3),
        "raw_same_class_total": total,
        "in_region_count": len(in_region),
        "cross_region_violation_count": viol_cnt,
        "cross_region_violation_rate": round(viol_cnt / total, 6) if total > 0 else 0.0,
        "all_predictions_annotated": annotated,
        "in_region_predictions": in_region,
        "cross_region_violation_predictions": cross_region,
    }


def score_spatial_arm_against_gt(
    *,
    same_class_preds: Sequence[Mapping[str, Any]],
    unauthorized_preds: Sequence[Mapping[str, Any]],
    gt_boxes_for_class: Sequence[Mapping[str, Any]],
    image_width: float,
    region_mode: str,
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Score one spatial arm (`left_half` or `right_half`) identically for GT and predicted centers."""
    rmode = str(region_mode or "").strip().lower()
    if rmode not in ("left_half", "right_half"):
        raise ValueError(f"Scoring requires left_half or right_half, got {region_mode!r}")

    pred_audit = audit_spatial_predictions(same_class_preds, image_width=image_width, region_mode=rmode)

    in_region_gt: List[Dict[str, Any]] = []
    opposite_gt: List[Dict[str, Any]] = []
    for g in gt_boxes_for_class:
        bbox = _extract_xyxy(g)
        half = classify_box_half(bbox, image_width)
        rec = {"bbox": bbox, "assigned_half": half}
        if half == rmode:
            in_region_gt.append(rec)
        else:
            opposite_gt.append(rec)

    preds_for_match = [
        {"bbox": _extract_xyxy(p), "score": p.get("score")}
        for p in pred_audit["in_region_predictions"]
    ]
    matched = repo_eval.match_category_predictions_one_to_one(
        in_region_gt, preds_for_match, iou_threshold=float(iou_threshold)
    )
    tp = int(matched["tp"])
    in_region_fp = int(matched["fp"])
    fn = int(matched["fn"])
    viol_cnt = int(pred_audit["cross_region_violation_count"])
    strict_fp = in_region_fp + viol_cnt

    in_region_prf = _pr_f1(tp, in_region_fp, fn)
    strict_prf = _pr_f1(tp, strict_fp, fn)
    gt_cnt = len(in_region_gt)
    in_reg_cnt = len(pred_audit["in_region_predictions"])
    raw_cnt = len(pred_audit["all_predictions_annotated"])
    unauth_cnt = len(unauthorized_preds)

    is_half_absent = gt_cnt == 0
    return {
        "region_mode": rmode,
        "gt_in_region_count": gt_cnt,
        "gt_opposite_half_count": len(opposite_gt),
        "raw_same_class_pred_count": raw_cnt,
        "in_region_pred_count": in_reg_cnt,
        "cross_region_violation_count": viol_cnt,
        "cross_region_violation_rate": pred_audit["cross_region_violation_rate"],
        "unauthorized_class_count": unauth_cnt,
        "tp": tp,
        "in_region_fp": in_region_fp,
        "strict_fp": strict_fp,
        "fn": fn,
        "in_region_precision": in_region_prf["precision"],
        "in_region_recall": in_region_prf["recall"],
        "in_region_f1": in_region_prf["f1"],
        "strict_precision": strict_prf["precision"],
        "strict_f1": strict_prf["f1"],
        "in_region_count_abs_error": abs(in_reg_cnt - gt_cnt),
        "raw_count_abs_error": abs(raw_cnt - gt_cnt),
        "is_half_absent": is_half_absent,
        "half_absent_fp_total": raw_cnt if is_half_absent else None,
        "half_absent_in_region_fp": in_reg_cnt if is_half_absent else None,
        "strict_valid_empty": (raw_cnt == 0 and unauth_cnt == 0) if is_half_absent else None,
        "matched_pairs": matched["matched_pairs"],
    }


def aggregate_spatial_scores(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool micro TP/FP/FN, violations, and absent-half counts across a list of scored spatial arms."""
    tp = sum(int(r["tp"]) for r in records)
    in_fp = sum(int(r["in_region_fp"]) for r in records)
    strict_fp = sum(int(r["strict_fp"]) for r in records)
    fn = sum(int(r["fn"]) for r in records)
    viol = sum(int(r["cross_region_violation_count"]) for r in records)
    raw_tot = sum(int(r["raw_same_class_pred_count"]) for r in records)
    in_tot = sum(int(r["in_region_pred_count"]) for r in records)
    gt_tot = sum(int(r["gt_in_region_count"]) for r in records)
    unauth = sum(int(r["unauthorized_class_count"]) for r in records)
    absent = [r for r in records if r.get("is_half_absent")]
    pos = [r for r in records if not r.get("is_half_absent")]

    in_prf = _pr_f1(tp, in_fp, fn)
    strict_prf = _pr_f1(tp, strict_fp, fn)
    return {
        "eval_count": len(records),
        "positive_eval_count": len(pos),
        "half_absent_eval_count": len(absent),
        "gt_in_region_total": gt_tot,
        "raw_same_class_total": raw_tot,
        "in_region_pred_total": in_tot,
        "cross_region_violation_total": viol,
        "cross_region_violation_rate": round(viol / raw_tot, 6) if raw_tot > 0 else 0.0,
        "unauthorized_class_total": unauth,
        "tp": tp,
        "in_region_fp": in_fp,
        "strict_fp": strict_fp,
        "fn": fn,
        "in_region_precision": in_prf["precision"],
        "in_region_recall": in_prf["recall"],
        "in_region_f1": in_prf["f1"],
        "strict_precision": strict_prf["precision"],
        "strict_f1": strict_prf["f1"],
        "in_region_count_abs_error_sum": sum(int(r["in_region_count_abs_error"]) for r in records),
        "raw_count_abs_error_sum": sum(int(r["raw_count_abs_error"]) for r in records),
        "half_absent_fp_total": sum(int(r.get("half_absent_fp_total") or 0) for r in absent),
        "half_absent_in_region_fp_total": sum(int(r.get("half_absent_in_region_fp") or 0) for r in absent),
        "half_absent_strict_valid_empty_passes": sum(1 for r in absent if r.get("strict_valid_empty")),
    }


def evaluate_spatial_promotion_gates(
    *,
    direct_left_positive_tp: int,
    direct_right_positive_tp: int,
    direct_pooled: Mapping[str, Any],
    cpu_pooled: Mapping[str, Any],
    per_eval_pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    distinct_fresh_calls_verified: bool,
    auditable_cross_region_logged: bool,
    max_cross_region_violation_rate: float = 0.25,
) -> Dict[str, Any]:
    """Evaluate pre-registered admission + promotion gates for enabling spatial UI interaction."""
    no_pos_collapse = not any(
        int(cpu["gt_in_region_count"]) > 0 and int(cpu["tp"]) > 0 and int(direct["tp"]) == 0
        for direct, cpu in per_eval_pairs
    )
    gates = {
        "distinct_fresh_calls_verified": bool(distinct_fresh_calls_verified),
        "both_sides_positive_hits": bool(direct_left_positive_tp > 0 and direct_right_positive_tp > 0),
        "auditable_cross_region_logged": bool(auditable_cross_region_logged),
        "beats_or_matches_cpu_postfilter_tp": int(direct_pooled["tp"]) >= int(cpu_pooled["tp"]),
        "beats_or_matches_cpu_postfilter_strict_f1": float(direct_pooled["strict_f1"]) >= float(cpu_pooled["strict_f1"]),
        "cross_region_violation_bounded": float(direct_pooled["cross_region_violation_rate"]) <= float(max_cross_region_violation_rate),
        "no_positive_half_collapse": bool(no_pos_collapse),
    }
    admission_ok = (
        gates["distinct_fresh_calls_verified"]
        and gates["both_sides_positive_hits"]
        and gates["auditable_cross_region_logged"]
    )
    promote_ok = all(gates.values())
    return {
        "admission_gates_passed": admission_ok,
        "promote_spatial_ui": promote_ok,
        "recommended_ui_action": (
            "keep_main_single_class_scope_and_isolation" if not promote_ok else "eligible_for_bounded_spatial_ui"
        ),
        "gates": gates,
    }


def terminate_pid_tree_force(pid: int) -> str:
    """Forcefully terminate a subprocess tree on Windows (`taskkill /T /F`) or POSIX process group."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return "taskkill_tree_force"
        except Exception:
            return "taskkill_failed"
    try:
        os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
        return "killpg_sigkill"
    except Exception:
        return "killpg_failed"


def run_with_process_group_deadline(
    cmd: Sequence[str],
    *,
    global_deadline_epoch: float,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    stdout_path: Optional[pathlib.Path] = None,
    stderr_path: Optional[pathlib.Path] = None,
    poll_interval_s: float = 0.25,
) -> Dict[str, Any]:
    """Run a child command in its own process group and enforce `global_deadline_epoch` via tree kill."""
    t0 = time.monotonic()
    remaining = float(global_deadline_epoch) - t0
    if remaining <= 0.0:
        return {
            "returncode": -1,
            "timed_out_global_deadline": True,
            "termination_method": "pre_launch_deadline_expired",
            "elapsed_seconds": 0.0,
            "pid": None,
        }

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(cwd) if cwd else None,
        "env": dict(env) if env is not None else None,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    out_fh = open(stdout_path, "w", encoding="utf-8") if stdout_path else None
    err_fh = open(stderr_path, "w", encoding="utf-8") if stderr_path else None
    popen_kwargs["stdout"] = out_fh if out_fh else subprocess.PIPE
    popen_kwargs["stderr"] = err_fh if err_fh else subprocess.PIPE

    timed_out = False
    term_method: Optional[str] = None
    proc: Optional[subprocess.Popen[str]] = None
    stdout_tail = ""
    stderr_tail = ""
    try:
        proc = subprocess.Popen(list(cmd), **popen_kwargs)
        while proc.poll() is None:
            if time.monotonic() >= float(global_deadline_epoch):
                timed_out = True
                term_method = terminate_pid_tree_force(proc.pid)
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    pass
                break
            time.sleep(min(float(poll_interval_s), max(0.05, float(global_deadline_epoch) - time.monotonic())))
        if not timed_out and out_fh is None and err_fh is None:
            out_txt, err_txt = proc.communicate(timeout=5.0)
            stdout_tail = (out_txt or "")[-2000:]
            stderr_tail = (err_txt or "")[-2000:]
    finally:
        if out_fh is not None:
            out_fh.close()
        if err_fh is not None:
            err_fh.close()

    return {
        "returncode": int(proc.returncode) if (proc is not None and proc.returncode is not None) else -1,
        "timed_out_global_deadline": bool(timed_out),
        "termination_method": term_method,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        "pid": int(proc.pid) if proc is not None else None,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
