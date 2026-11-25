"""Natural-language single-class PCB detection instruction parser, prompt builder, and compliance auditor."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES

_SIX_CLASS_SET = frozenset(FICS_SIX_CLASSES)

CLASS_CN_DISPLAY: Dict[str, str] = {
    "capacitor": "电容",
    "resistor": "电阻",
    "inductor": "电感",
    "transistor": "晶体管",
    "diode": "二极管",
    "ic": "集成电路",
}

# Ordered longest/most-specific patterns first.
_CLASS_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("capacitor", (r"贴片电容", r"电解电容", r"钽电容", r"电容器", r"电容", r"\bcapacitors?\b")),
    ("resistor", (r"贴片电阻", r"色环电阻", r"排阻", r"电阻器", r"电阻", r"\bresistors?\b")),
    ("inductor", (r"功率电感", r"贴片电感", r"电感器", r"电感", r"磁珠", r"\binductors?\b")),
    ("transistor", (r"场效应管", r"三极管", r"mosfet", r"mos管", r"晶体管", r"\btransistors?\b")),
    # Must check 二极管 before ic/diode collisions
    ("diode", (r"发光二极管", r"稳压二极管", r"肖特基二极管", r"稳压管", r"二极管", r"\bdiodes?\b")),
    ("ic", (r"集成电路", r"微控制器", r"单片机", r"主控芯片", r"芯片", r"(?<![a-z0-9])ics?(?![a-z0-9])")),
]

_SPATIAL_PATTERNS: Tuple[str, ...] = (
    r"左上", r"右上", r"左下", r"右下", r"左半", r"右半", r"上半", r"下半",
    r"左边", r"右边", r"左侧", r"右侧", r"上方", r"下方", r"中间", r"中央",
    r"区域", r"局部", r"附近", r"旁边", r"里面", r"边缘",
    r"\btop[-\s]?left\b", r"\btop[-\s]?right\b", r"\bbottom[-\s]?left\b", r"\bbottom[-\s]?right\b",
    r"\bleft\s+(?:half|side|region)\b", r"\bright\s+(?:half|side|region)\b",
    r"\bupper\b", r"\blower\b", r"\bcorner\b", r"\bregion\b", r"\bnear(?:by)?\b",
)

_OUT_OF_SCOPE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("连接器/接插件", r"连接器|接插件|插接件|\bconnectors?\b"),
    ("排针/插座", r"排针|插槽|插座|\bheaders?\b|\bsockets?\b"),
    ("开关/按键", r"拨码开关|开关|按键|按钮|\bswitch(?:es)?\b|\bbuttons?\b"),
    ("晶振", r"晶体振荡器|晶振|\bcrystals?\b|\boscillators?\b"),
    ("保险丝", r"保险丝|\bfuses?\b"),
    ("变压器", r"变压器|\btransformers?\b"),
    ("蜂鸣器/换能器", r"蜂鸣器|换能器|\bbuzzers?\b|\btransducers?\b"),
    ("传感器/继电器/电位器", r"传感器|继电器|电位器|\bsensors?\b|\brelays?\b|\bpotentiometers?\b"),
    ("数码管", r"数码管|七段|\bseven[-\s]?seg(?:ment)?\b"),
    ("焊盘/丝印/过孔/背景", r"焊盘|丝印|过孔|铜箔|\bpads?\b|\bsilkscreen\b|\bvias?\b"),
)

_ALL_CLASSES_PHRASES: Tuple[str, ...] = (
    r"所有六类", r"全部六类", r"全部元器件", r"所有元器件", r"所有器件", r"全部器件", r"六类",
    r"\ball\s+(?:six|6|components?)\b",
)

_NEGATION_PHRASES: Tuple[str, ...] = (
    r"除了", r"不要", r"排除", r"非", r"\bexcept\b", r"\bwithout\b", r"\bexclude\b",
)

_COUNT_PATTERNS: Tuple[str, ...] = (
    r"计数", r"数量", r"几个", r"多少", r"统计", r"总数", r"\bcount\b", r"\bhow\s+many\b",
)

_ACTION_PATTERNS: Tuple[str, ...] = (
    r"找", r"检测", r"识别", r"定位", r"标注", r"框出", r"查找", r"找出", r"统计", r"计数",
    r"\bfind\b", r"\blocate\b", r"\bdetect\b", r"\bcount\b", r"\bshow\b",
)


def _strip_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_single_class_detection_prompt(
    target_class: str,
    *,
    coordinate_protocol: str = "xyxy_norm1000",
    prompt_variant: str = "r8_base",
) -> Tuple[str, str]:
    """Build deterministic single-class locate-and-count prompt and its SHA-256."""
    cls = str(target_class or "").strip().lower()
    if cls not in _SIX_CLASS_SET:
        raise ValueError(f"Unsupported single-class prompt target: {target_class!r}")
    if coordinate_protocol == "xyxy_norm1":
        coord_spec = "[0.0, 1.0]"
    elif coordinate_protocol == "xyxy_norm1000":
        coord_spec = "[0, 1000]"
    else:
        raise ValueError(f"Unsupported coordinate_protocol={coordinate_protocol!r}")

    variant = str(prompt_variant or "r8_base").strip() or "r8_base"
    if variant == "r8_base":
        prompt = (
            f"Locate and count all visible {cls} components on this PCB image. "
            f"Do NOT output any component other than {cls}. "
            f"Return a JSON array of objects with keys bbox_2d and label. "
            f"Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized to {coord_spec} relative to this input image. "
            f"Use the exact label \"{cls}\" for every detected {cls}. "
            f"If no {cls} components are visible, return []. Do not include explanations."
        )
    elif variant == "r10_visible_evidence":
        prompt = (
            f"Locate and count all visible {cls} components on this PCB image. "
            f"Do NOT output any component other than {cls}. "
            f"Only box a {cls} if its actual physical component body is clearly visible; "
            f"strictly exclude bare solder pads, empty footprints, pins or headers without a {cls} body, "
            f"silkscreen text or outlines, and bare PCB surface. "
            f"Return a JSON array of objects with keys bbox_2d and label. "
            f"Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized to {coord_spec} relative to this input image. "
            f"Use the exact label \"{cls}\" for every detected {cls}. "
            f"If no {cls} components are clearly visible, return []. Do not include explanations."
        )
    else:
        raise ValueError(f"Unsupported prompt_variant={prompt_variant!r}")
    return prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_pcb_language_instruction(
    user_text: str,
    *,
    coordinate_protocol: str = "xyxy_norm1000",
    prompt_variant: str = "r8_base",
) -> Dict[str, Any]:
    """Parse natural-language single-class locate/count instruction into canonical spec or explicit error."""
    cleaned = _strip_spaces(user_text)
    if not cleaned:
        return {
            "ok": False,
            "error_code": "empty_instruction",
            "error_message": "指令不能为空。请输入六类范围内的单类查找指令，例如「只找电阻并计数」。",
            "raw_instruction": cleaned,
        }

    low = cleaned.lower()

    for pat in _SPATIAL_PATTERNS:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "spatial_restriction_unsupported",
                "error_message": f"当前版本暂不支持带空间区域限制的指令（命中限定词：{pat}），留待下一步扩展；本轮仅支持全板单类定位与计数，如「只找电阻并计数」。",
                "raw_instruction": cleaned,
            }

    for pat in _ALL_CLASSES_PHRASES:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "multi_class_use_standard_mode",
                "error_message": "您输入的是多类/全六类检测意图，请直接使用上方“标准六类检测（W/T）”；单类指令入口仅用于按单一指定类别重跑推理（如「只找电阻并计数」）。",
                "raw_instruction": cleaned,
            }

    for scope_name, pat in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "out_of_scope_class",
                "error_message": f"目标超出本系统 FICS-PCB 六类标准词表（检测到范围外对象：{scope_name}）。仅支持：电容 (capacitor)、电阻 (resistor)、电感 (inductor)、晶体管 (transistor)、二极管 (diode)、集成电路 (ic)。",
                "raw_instruction": cleaned,
            }

    for pat in _NEGATION_PHRASES:
        if re.search(pat, low):
            return {
                "ok": False,
                "error_code": "exclusion_instruction_unsupported",
                "error_message": "当前不支持否定/排除类指令（如“除了…之外”）。请直接指定单一目标类别，例如「只找电容并计数」。",
                "raw_instruction": cleaned,
            }

    matched_classes: List[str] = []
    for cls, pats in _CLASS_PATTERNS:
        if any(re.search(p, low) for p in pats):
            matched_classes.append(cls)

    if len(matched_classes) >= 2:
        cn_list = "、".join(f"{CLASS_CN_DISPLAY[c]} ({c})" for c in matched_classes)
        return {
            "ok": False,
            "error_code": "multi_class_not_single_target",
            "error_message": f"指令中同时包含多个目标类别（{cn_list}）。按指令重推入口仅支持单一目标类别，或请改用标准六类入口。",
            "raw_instruction": cleaned,
            "matched_classes": matched_classes,
        }

    if len(matched_classes) == 0:
        return {
            "ok": False,
            "error_code": "unrecognized_target_class",
            "error_message": "未能识别到 FICS-PCB 六类范围内的目标器件。支持类别：电容 (capacitor)、电阻 (resistor)、电感 (inductor)、晶体管 (transistor)、二极管 (diode)、集成电路 (ic)。",
            "raw_instruction": cleaned,
        }

    target_cls = matched_classes[0]
    count_requested = any(re.search(p, low) for p in _COUNT_PATTERNS)
    # Require either clear action/count intent or standalone class phrase with <= 24 chars
    has_action = any(re.search(p, low) for p in _ACTION_PATTERNS)
    if not has_action and not count_requested and len(cleaned) > 24:
        return {
            "ok": False,
            "error_code": "ambiguous_instruction_intent",
            "error_message": "指令表述不清晰，请包含明确查找或计数意图，例如「只找电阻并计数」。",
            "raw_instruction": cleaned,
        }

    prompt_text, prompt_sha = build_single_class_detection_prompt(
        target_cls,
        coordinate_protocol=coordinate_protocol,
        prompt_variant=prompt_variant,
    )
    return {
        "ok": True,
        "raw_instruction": cleaned,
        "target_class": target_cls,
        "target_class_cn": CLASS_CN_DISPLAY[target_cls],
        "count_requested": bool(count_requested),
        "normalized_intent": f"find_and_count_{target_cls}" if count_requested else f"find_{target_cls}",
        "prompt_variant": str(prompt_variant or "r8_base"),
        "normalized_prompt": prompt_text,
        "prompt_sha256": prompt_sha,
        "coordinate_protocol": coordinate_protocol,
    }


def _extract_raw_json_labels(raw_text: str) -> List[str]:
    """Extract raw JSON object 'label' fields from a raw VLM output string for unauthorized-label auditing."""
    txt = str(raw_text or "")
    for tok in ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>"):
        txt = txt.replace(tok, "")
    txt = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", txt.strip(), flags=re.IGNORECASE)
    m = re.search(r"\[.*\]", txt, flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    if not isinstance(arr, list):
        return []
    out: List[str] = []
    for item in arr:
        if isinstance(item, Mapping) and item.get("label") is not None:
            out.append(str(item["label"]).strip().lower())
    return out


def audit_single_class_predictions(
    *,
    all_post_nms_preds: Sequence[Mapping[str, Any]],
    raw_global_predictions: Sequence[Mapping[str, Any]],
    raw_records: Optional[Sequence[Mapping[str, Any]]],
    target_class: str,
) -> Dict[str, Any]:
    """Separate target-class predictions from unauthorized classes without hiding pre-filter violation rates."""
    tgt = str(target_class).strip().lower()
    if tgt not in _SIX_CLASS_SET:
        raise ValueError(f"Unsupported target_class={target_class!r}")

    target_preds: List[Dict[str, Any]] = []
    unauth_preds: List[Dict[str, Any]] = []
    unauth_cls_breakdown: Counter = Counter()
    for p in all_post_nms_preds:
        lbl = str(p.get("label") or "").strip().lower()
        if lbl == tgt:
            target_preds.append(dict(p))
        else:
            unauth_preds.append(dict(p))
            unauth_cls_breakdown[lbl or "unknown"] += 1

    raw_unauth_count = sum(1 for p in raw_global_predictions if str(p.get("label") or "").strip().lower() != tgt)
    raw_unauth_labels: Counter = Counter()
    for r in raw_records or ():
        for lbl in _extract_raw_json_labels(str(r.get("raw_output") or "")):
            if lbl != tgt:
                raw_unauth_labels[lbl or "empty"] += 1

    pre_total = len(all_post_nms_preds)
    unauth_cnt = len(unauth_preds)
    unauth_rate = round(unauth_cnt / pre_total, 6) if pre_total > 0 else 0.0

    return {
        "target_class": tgt,
        "target_count": len(target_preds),
        "target_predictions": target_preds,
        "unauthorized_predictions": unauth_preds,
        "pre_filter_post_nms_total": pre_total,
        "unauthorized_post_nms_count": unauth_cnt,
        "unauthorized_post_nms_rate": unauth_rate,
        "unauthorized_class_breakdown": dict(sorted(unauth_cls_breakdown.items())),
        "raw_global_total": len(raw_global_predictions),
        "unauthorized_raw_global_count": raw_unauth_count,
        "raw_json_unauthorized_labels": dict(sorted(raw_unauth_labels.items())),
    }
