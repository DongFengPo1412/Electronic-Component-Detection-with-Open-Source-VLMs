"""Final GT-free whole-board PCB component detection demo backend & research replay service.

Composes frozen repository primitives (`tiling`, `fics_replay`, `candidate_verifier`,
`fics_matrix`, `candidate_verifier_runner`, `evaluate`, and `gt_free_pipeline`) without
reading default ground truth during live inference and without hardcoding private paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from pcb_vlm.data.fics_dslr import file_sha256
from pcb_vlm.evaluation.candidate_verifier import (
    ALLOWED_SEVEN_LABELS,
    DEFAULT_VERIFIER_PROMPT,
)
from pcb_vlm.evaluation.fics_replay import (
    FICS_SIX_CLASSES,
    classify_call_termination_and_parse_v2,
    load_fics_six_class_config,
    validate_coco_schema_and_categories,
)
from pcb_vlm.evaluation.composite_detector import (
    COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA,
    SUPPORTED_COMPOSITE_DETECTOR_PRESETS,
    apply_bounded_vlm_verification,
    box_in_composite_region,
    evaluate_composite_board,
    execute_composite_query,
    render_composite_query_overlay_png,
    run_yolo11n_class_aware_board,
    select_hard_candidates_for_vlm,
)
from pcb_vlm.evaluation.gt_free_pipeline import (
    compute_board_roi_and_tiles,
    export_dual_crop_consensus_gt_free,
    export_verified_predictions_gt_free,
    load_jsonl_records,
    prepare_verifier_manifest_gt_free,
    reconstruct_board_predictions_gt_free,
)
from pcb_vlm.evaluation.language_instruction import (
    audit_single_class_predictions,
    parse_pcb_language_instruction,
)
from pcb_vlm.evaluation.spatial_experiment import (
    audit_spatial_predictions,
    parse_spatial_pcb_instruction,
)
from pcb_vlm.geometry.tiling import generate_tiles

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import evaluate as repo_eval  # noqa: E402

import numpy as np  # noqa: E402

MODEL_ORDER: Tuple[str, ...] = ("qwen2", "deepseek", "qwen3")
EXPERIMENTAL_MODEL_ORDER: Tuple[str, ...] = ("yolo11n", "yolo11n_qwen3")
COMPOSITE_MODEL_ORDER: Tuple[str, ...] = ("yolo11n_composite", "yolo11n_qwen3_composite")
ALL_SUPPORTED_MODELS: Tuple[str, ...] = MODEL_ORDER + EXPERIMENTAL_MODEL_ORDER + COMPOSITE_MODEL_ORDER
DEFAULT_YOLO11N_WEIGHTS_PATH = r"C:\Users\Liu\.codex\cache\tasks\pcb-yolo-v3-r7-20261104\training\weights\best.pt"
DEFAULT_YOLO11N_WEIGHTS_SHA256 = "5378f977c9d3c56d9d6456ee4e4791b662fa8c294af5f5488f570b5e5a3bb4e1"
DEFAULT_FICS_R15_WEIGHTS_PATH = r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\weights\fics_yolo11n_r15_best.pt"
DEFAULT_FICS_R15_WEIGHTS_SHA256 = "4d92fd1d8093ea7498c13b3aac6dd5a6e5c21b9965b96667270883aed571009e"
DEFAULT_ULTRALYTICS_OVERLAY_DIR = r"C:\Users\Liu\.codex\cache\tasks\pcb-train-prep-20261104\dependency-overlay"
SUPPORTED_IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
TERMINAL_JOB_STATUSES: Tuple[str, ...] = ("completed", "failed", "cancelled", "incomplete")
MAX_UPLOAD_BYTES: int = 256 * 1024 * 1024
MIN_IMAGE_DIM: int = 32
MAX_IMAGE_DIM: int = 16384
MAX_IMAGE_PIXELS: int = 100_000_000
MAX_TILE_COUNT: int = 120
COMPOSITE_MAX_TILE_COUNT: int = 300
RECOMMENDED_LIVE_T_B_OTHER_MAX_TILES: int = 24
FROZEN_DETECTION_PROMPT = (
    "Locate all visible electronic components belonging to these classes: "
    "capacitor, resistor, inductor, transistor, diode, ic. "
    "Return a JSON array of objects with keys bbox_2d and label. "
    "Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized to [0, 1000] relative to this input image. "
    "Use exactly one of the listed class names for each object. "
    "If no target components are visible, return []. Do not include explanations."
)
FROZEN_DETECTION_PROMPT_SHA256 = hashlib.sha256(FROZEN_DETECTION_PROMPT.encode("utf-8")).hexdigest()
R3_QWEN2_NORM1_PROMPT = (
    "Locate all visible electronic components belonging to these classes: "
    "capacitor, resistor, inductor, transistor, diode, ic. "
    "Return a JSON array of objects with keys bbox_2d and label. "
    "Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized to [0.0, 1.0] relative to this input image. "
    "Use exactly one of the listed class names for each object. "
    "If no target components are visible, return []. Do not include explanations."
)
R3_QWEN2_NORM1_PROMPT_SHA256 = hashlib.sha256(R3_QWEN2_NORM1_PROMPT.encode("utf-8")).hexdigest()
R3_DEEPSEEK_SHORT_PROMPT = (
    "Return ONLY a JSON array of tight bounding boxes for visible PCB components. "
    "Allowed labels: capacitor, resistor, inductor, transistor, diode, ic. "
    'Each item must be {"bbox_2d": [xmin, ymin, xmax, ymax], "label": "<class>"} '
    "with 4 actual numeric coordinates in [0, 1000] (0 <= xmin < xmax <= 1000, 0 <= ymin < ymax <= 1000). "
    "If no target component is visible, return []. No prose."
)
R3_DEEPSEEK_SHORT_PROMPT_SHA256 = hashlib.sha256(R3_DEEPSEEK_SHORT_PROMPT.encode("utf-8")).hexdigest()

SUPPORTED_PRESETS: Tuple[str, ...] = (
    "baseline",
    "r3_candidate",
    "r3_qwen2_norm1",
    "r3_deepseek_short_json",
    "r3_qwen3_greedy",
    "r4_highres",
    "r4_qwen3_greedy_maxpx589824",
    "r4_qwen2_norm1_maxpx451584",
    "r5_tile768",
    "r5_qwen3_greedy_maxpx589824_tile768",
    "r6_b_other_strict",
    "r6_qwen3_tile768_b_other_strict",
    "r7_dual_crop_consensus",
    "r7_qwen3_tile768_dual_crop_consensus",
    "r9_nf4_base",
    "r9_qwen3_nf4_base_tile768",
    "r9_nf4_adapter",
    "r9_qwen3_nf4_adapter_tile768",
    "r10_evidence_prompt",
    "r10_qwen3_bf16_tile768_evidence_prompt",
)


def resolve_model_preset_spec(model_key: str, preset: Optional[str] = None) -> Dict[str, Any]:
    """Resolve explicit opt-in Round 3/4/5 single-intervention config or return common baseline."""
    mk = str(model_key or "").strip().lower()
    raw_p = str(preset or "baseline").strip() or "baseline"
    if raw_p not in SUPPORTED_PRESETS:
        raise DemoError(
            f"不支持的配置预设 {raw_p!r}；可选值：{', '.join(SUPPORTED_PRESETS)}。",
            status_code=400,
        )

    baseline_spec = {
        "preset_group": "baseline",
        "preset_id": "baseline",
        "preset_display_name": "共同基准配置 (xyxy_norm1000)",
        "coordinate_protocol": "xyxy_norm1000",
        "prompt": FROZEN_DETECTION_PROMPT,
        "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
        "explicit_do_sample_false": False,
        "qwen_min_pixels": 65536,
        "qwen_max_pixels": 200704,
        "tile_size": 1024,
        "tile_overlap_ratio": 0.20,
        "intervention_note": "冻结共同基准：xyxy_norm1000，原版 6 类提示词，原有 generate 调用合同，max_pixels=200,704，1024px 切片。",
    }
    if raw_p == "baseline":
        return baseline_spec

    if mk not in MODEL_ORDER:
        raise DemoError(f"纯 VLM 单干预预设 {raw_p!r} 不适用于实验入口 {mk!r}。", status_code=400)

    r3_by_model: Dict[str, Dict[str, Any]] = {
        "qwen2": {
            "preset_group": "r3_candidate",
            "preset_id": "r3_qwen2_norm1",
            "preset_display_name": "第3轮候选 · Qwen2 统一 xyxy_norm1 契约",
            "coordinate_protocol": "xyxy_norm1",
            "prompt": R3_QWEN2_NORM1_PROMPT,
            "prompt_sha256": R3_QWEN2_NORM1_PROMPT_SHA256,
            "explicit_do_sample_false": False,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 200704,
            "tile_size": 1024,
            "tile_overlap_ratio": 0.20,
            "intervention_note": "仅将提示词与确定性解析统一为 xyxy_norm1 ([0.0, 1.0])；保持 max_pixels=200,704 与原有解码。",
        },
        "deepseek": {
            "preset_group": "r3_candidate",
            "preset_id": "r3_deepseek_short_json",
            "preset_display_name": "第3轮候选 · DeepSeek 短结构化 JSON 指令",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": R3_DEEPSEEK_SHORT_PROMPT,
            "prompt_sha256": R3_DEEPSEEK_SHORT_PROMPT_SHA256,
            "explicit_do_sample_false": False,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 200704,
            "tile_size": 1024,
            "tile_overlap_ratio": 0.20,
            "intervention_note": "仅替换为短结构化 JSON 指令，要求实际 4 坐标并约定空图返回 []；固定 384x384 预处理与 [0,1000] 协议。",
        },
        "qwen3": {
            "preset_group": "r3_candidate",
            "preset_id": "r3_qwen3_greedy",
            "preset_display_name": "第3轮候选 · Qwen3 显式贪心解码修正",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 200704,
            "tile_size": 1024,
            "tile_overlap_ratio": 0.20,
            "intervention_note": "保持原提示词、512 token 与 200,704 视觉上限不变；仅在 model.generate 显式传入 do_sample=False 进入 GREEDY_SEARCH。",
        },
    }

    r4_by_model: Dict[str, Dict[str, Any]] = {
        "qwen2": {
            "preset_group": "r4_highres",
            "preset_id": "r4_qwen2_norm1_maxpx451584",
            "preset_display_name": "第4轮候选 · Qwen2 norm1 视觉像素上限提升 (451,584)",
            "coordinate_protocol": "xyxy_norm1",
            "prompt": R3_QWEN2_NORM1_PROMPT,
            "prompt_sha256": R3_QWEN2_NORM1_PROMPT_SHA256,
            "explicit_do_sample_false": False,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 451584,
            "tile_size": 1024,
            "tile_overlap_ratio": 0.20,
            "intervention_note": "相对 r3_qwen2_norm1 仅将 max_pixels 从 200,704 提高至 451,584；保持 0-1 提示与解析、512 token、GREEDY_SEARCH 与 65,536 下限不变。",
        },
        "qwen3": {
            "preset_group": "r4_highres",
            "preset_id": "r4_qwen3_greedy_maxpx589824",
            "preset_display_name": "第4轮候选 · Qwen3 贪心视觉像素上限提升 (589,824)",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 589824,
            "tile_size": 1024,
            "tile_overlap_ratio": 0.20,
            "intervention_note": "相对 r3_qwen3_greedy 仅将 max_pixels 从 200,704 提高至 589,824；保持原提示词、xyxy_norm1000、显式 do_sample=False 贪心解码、512 token 与 65,536 下限不变。",
        },
    }

    r5_by_model: Dict[str, Dict[str, Any]] = {
        "qwen3": {
            "preset_group": "r5_tile768",
            "preset_id": "r5_qwen3_greedy_maxpx589824_tile768",
            "preset_display_name": "第5轮候选 · Qwen3 768px 小切片 + 589,824 高像素上限",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 589824,
            "tile_size": 768,
            "tile_overlap_ratio": 0.20,
            "verifier_policy": "B_other",
            "intervention_note": "相对 r4_qwen3_greedy_maxpx589824 仅将切片尺寸由 1024 缩小至 768（保持 20% 重叠率与贴边回拉规则）；锁定 max_pixels=589,824、min_pixels=65,536、原提示词、xyxy_norm1000、显式 do_sample=False 与 512 token 不变。",
        },
    }

    r6_by_model: Dict[str, Dict[str, Any]] = {
        "qwen3": {
            "preset_group": "r6_b_other_strict",
            "preset_id": "r6_qwen3_tile768_b_other_strict",
            "preset_display_name": "第6轮候选 · Qwen3 768px + 同模型严格 B_other 背景拒识（异常保留审计）",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 589824,
            "tile_size": 768,
            "tile_overlap_ratio": 0.20,
            "verifier_policy": "B_other_strict_keep_anomaly",
            "verifier_pad_fraction": 0.10,
            "intervention_note": "基于第 5 轮 Qwen3 768px/589,824 高召回候选，仅使用同一 Qwen3-VL 模型执行 10% 外扩局部裁剪二次核验（32 token + 显式 GREEDY_SEARCH）：严格仅在输出 other 时剔除候选，有效 6 类保留原框原类别；格式/截断/技术异常保留原框并独立计数审计。",
        },
    }

    r7_by_model: Dict[str, Dict[str, Any]] = {
        "qwen3": {
            "preset_group": "r7_dual_crop_consensus",
            "preset_id": "r7_qwen3_tile768_dual_crop_consensus",
            "preset_display_name": "第7轮候选 · Qwen3 768px + 双裁片一致背景拒识 (10% + 40%)",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 589824,
            "tile_size": 768,
            "tile_overlap_ratio": 0.20,
            "verifier_policy": "B_other_dual_crop_consensus",
            "verifier_pad_fraction": 0.10,
            "verifier_stage2_pad_fraction": 0.40,
            "intervention_note": "基于冻结 768px 切片候选（基础框来自已保存或当前真实推理）：先按 10% 裁片核验，仅对判为 other 的候选再用固定 40% 更宽上下文裁片复核（32 token + 显式 GREEDY_SEARCH）；仅当 10% 与 40% 两次均严格输出 other 才删除，任一次判为 6 类目标或格式/技术异常均保留原框与原类别。",
        },
    }

    r9_by_key: Dict[str, Dict[str, Any]] = {
        "r9_qwen3_nf4_base_tile768": {
            "preset_group": "r9_nf4_base",
            "preset_id": "r9_qwen3_nf4_base_tile768",
            "preset_display_name": "第9轮对照基准 · Qwen3-VL-2B NF4 4-bit Base (768px)",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 589824,
            "tile_size": 768,
            "tile_overlap_ratio": 0.20,
            "quantization": "nf4",
            "use_configured_r9_adapter": False,
            "adapter_id": "base_nf4",
            "intervention_note": "第 9 轮同精度配对基准：Qwen3-VL-2B-Instruct 4-bit NF4 无适配器基线，锁定 768px/20% 切片、589,824 视觉像素上限与显式贪心解码。",
        },
        "r9_qwen3_nf4_adapter_tile768": {
            "preset_group": "r9_nf4_adapter",
            "preset_id": "r9_qwen3_nf4_adapter_tile768",
            "preset_display_name": "第9轮微调候选 · Qwen3-VL-2B NF4 + QLoRA 适配器 (768px)",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 589824,
            "tile_size": 768,
            "tile_overlap_ratio": 0.20,
            "quantization": "nf4",
            "use_configured_r9_adapter": True,
            "adapter_id": "r9_qlora_adapter",
            "intervention_note": "第 9 轮同精度监督微调：Qwen3-VL-2B-Instruct 4-bit NF4 + QLoRA 语言条件多任务适配器，锁定 768px/20% 切片、589,824 视觉像素上限与显式贪心解码。",
        },
    }

    r10_by_key: Dict[str, Dict[str, Any]] = {
        "r10_qwen3_bf16_tile768_evidence_prompt": {
            "preset_group": "r10_evidence_prompt",
            "preset_id": "r10_qwen3_bf16_tile768_evidence_prompt",
            "preset_display_name": "第10轮提示对照 · Qwen3 BF16 可见器件本体证据约束 (768px)",
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": FROZEN_DETECTION_PROMPT,
            "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
            "instruction_prompt_variant": "r10_visible_evidence",
            "explicit_do_sample_false": True,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 589824,
            "tile_size": 768,
            "tile_overlap_ratio": 0.20,
            "intervention_note": "第 10 轮单因素提示对照：保持 BF16、768px/20% 重叠、max_pixels=589,824 与显式 greedy 解码完全不变，仅在单类指令提示中增加可见物理器件本体证据及排除裸焊盘/空焊盘/无本体引脚/纯丝印与板面约束。",
        },
    }

    if raw_p == "r3_candidate" or raw_p in ("r3_qwen2_norm1", "r3_deepseek_short_json", "r3_qwen3_greedy"):
        target = r3_by_model[mk]
        if raw_p != "r3_candidate" and raw_p != target["preset_id"]:
            raise DemoError(
                f"预设 {raw_p!r} 仅适用于对应模型，不能用于 {mk!r}。",
                status_code=400,
            )
        return dict(target)

    if raw_p in ("r5_tile768", "r5_qwen3_greedy_maxpx589824_tile768"):
        if mk != "qwen3":
            raise DemoError(
                f"第 5 轮 768px 小切片预设 {raw_p!r} 仅适用于晋级对照模型 qwen3，不能用于 {mk!r}。",
                status_code=400,
            )
        return dict(r5_by_model["qwen3"])

    if raw_p in ("r6_b_other_strict", "r6_qwen3_tile768_b_other_strict"):
        if mk != "qwen3":
            raise DemoError(
                f"第 6 轮同模型背景拒识预设 {raw_p!r} 仅适用于 qwen3，不能用于 {mk!r}。",
                status_code=400,
            )
        return dict(r6_by_model["qwen3"])

    if raw_p in ("r7_dual_crop_consensus", "r7_qwen3_tile768_dual_crop_consensus"):
        if mk != "qwen3":
            raise DemoError(
                f"第 7 轮双裁片一致背景拒识预设 {raw_p!r} 仅适用于 qwen3，不能用于 {mk!r}。",
                status_code=400,
            )
        return dict(r7_by_model["qwen3"])

    if raw_p in ("r9_nf4_base", "r9_qwen3_nf4_base_tile768"):
        if mk != "qwen3":
            raise DemoError(
                f"第 9 轮 NF4 对照预设 {raw_p!r} 仅适用于 qwen3，不能用于 {mk!r}。",
                status_code=400,
            )
        return dict(r9_by_key["r9_qwen3_nf4_base_tile768"])

    if raw_p in ("r9_nf4_adapter", "r9_qwen3_nf4_adapter_tile768"):
        if mk != "qwen3":
            raise DemoError(
                f"第 9 轮 NF4 QLoRA 适配器预设 {raw_p!r} 仅适用于 qwen3，不能用于 {mk!r}。",
                status_code=400,
            )
        return dict(r9_by_key["r9_qwen3_nf4_adapter_tile768"])

    if raw_p in ("r10_evidence_prompt", "r10_qwen3_bf16_tile768_evidence_prompt"):
        if mk != "qwen3":
            raise DemoError(
                f"第 10 轮 BF16 器件证据提示预设 {raw_p!r} 仅适用于 qwen3，不能用于 {mk!r}。",
                status_code=400,
            )
        return dict(r10_by_key["r10_qwen3_bf16_tile768_evidence_prompt"])

    if mk == "deepseek":
        raise DemoError(
            f"DeepSeek-VL v1 采用固定 384×384 方形预处理，不支持动态高像素预设 {raw_p!r}；请使用受控局部诊断或独立训练入口。",
            status_code=400,
        )
    target_r4 = r4_by_model[mk]
    if raw_p != "r4_highres" and raw_p != target_r4["preset_id"]:
        raise DemoError(
            f"预设 {raw_p!r} 仅适用于对应模型，不能用于 {mk!r}。",
            status_code=400,
        )
    return dict(target_r4)
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen2": "Qwen2-VL-2B-Instruct",
    "deepseek": "deepseek-vl-1.3b-chat（v1）",
    "qwen3": "Qwen3-VL-2B-Instruct",
    "yolo11n": "YOLO11n 专用候选定位器（阶段 A · 丢弃 WACV 类别）",
    "yolo11n_qwen3": "YOLO 定位 + Qwen3 分类（阶段 B · Top-200 候选 + 10% 外扩裁剪分类）",
    "yolo11n_composite": "复合检测 · YOLO11n 6类直接检测 (640px切片 · 保留原置信度)",
    "yolo11n_qwen3_composite": "复合检测 · YOLO11n + Qwen3 有限难样本核验 (<=48 候选)",
}
MODEL_VISUAL_PREPROCESS_NOTES: Dict[str, str] = {
    "qwen2": "Qwen2-VL 原生动态分辨率预算 (min_pixels=65,536, max_pixels=200,704, SDPA)",
    "deepseek": "DeepSeek-VL v1 原生固定正方形缩放 (384×384 + [127,127,127] 灰边填充)",
    "qwen3": "Qwen3-VL 原生动态分辨率预算 (shortest_edge=65,536, longest_edge=200,704, SDPA + 独立依赖覆盖)",
    "yolo11n": "YOLO11n (WACV best.pt) 1024px 切片 -> imgsz=640, conf=0.05, tile/global 类无关 NMS IoU=0.50, Top-200 候选（完全丢弃 WACV 类别）",
    "yolo11n_qwen3": "阶段 A 冻结 Top-200 候选框 + 阶段 B 10% 外扩裁剪 (min_edge=224) -> Qwen3-VL 严格单标签 6 类 + other 过滤",
    "yolo11n_composite": "YOLO11n (WACV best.pt) 640px/20% 重叠切片 · imgsz=640 · 各类独立 NMS 0.50 · 默认 0.25 (同步保存 0.05 候选池 · 额外类单列不并入 diodes)",
    "yolo11n_qwen3_composite": "YOLO11n 640px 切片对齐 6 类 + 固定确定性规则筛选 <=48 难/低分/冲突候选 -> Qwen3-VL BF16 局部裁片保守核验与最终 NMS",
}


def _format_visual_preprocess_note(mk: str, preset_spec: Optional[Mapping[str, Any]] = None) -> str:
    """Return dynamic visual/tiling note reflecting opt-in max_pixels and tile_size presets."""
    spec = preset_spec or {}
    tile_sz = int(spec.get("tile_size") or 1024)
    ov_pct = int(round(float(spec.get("tile_overlap_ratio", 0.20)) * 100))
    tile_desc = f", 切片={tile_sz}px/{ov_pct}%重叠" if tile_sz != 1024 else ""
    if mk == "qwen3":
        min_px = int(spec.get("qwen_min_pixels") or 65536)
        max_px = int(spec.get("qwen_max_pixels") or 200704)
        if max_px != 200704 or min_px != 65536 or tile_sz != 1024:
            return f"Qwen3-VL 原生动态分辨率预算 (shortest_edge={min_px:,}, longest_edge={max_px:,}{tile_desc}, SDPA + 独立依赖覆盖)"
    elif mk == "qwen2":
        min_px = int(spec.get("qwen_min_pixels") or 65536)
        max_px = int(spec.get("qwen_max_pixels") or 200704)
        if max_px != 200704 or min_px != 65536 or tile_sz != 1024:
            return f"Qwen2-VL 原生动态分辨率预算 (min_pixels={min_px:,}, max_pixels={max_px:,}{tile_desc}, SDPA)"
    return MODEL_VISUAL_PREPROCESS_NOTES.get(mk, "")
CLASS_CN_NAMES: Dict[str, str] = {
    "capacitor": "电容",
    "resistor": "电阻",
    "inductor": "电感",
    "transistor": "晶体管",
    "diode": "二极管",
    "ic": "集成电路",
}
CLASS_DISPLAY_NAMES: Dict[str, str] = {
    k: f"{v} ({k})" for k, v in CLASS_CN_NAMES.items()
}
MODE_DISPLAY_NAMES: Dict[str, str] = {
    "W": "W（整 ROI 推理）",
    "T": "T（切片推理 + NMS）",
    "T_B_other": "T + B_other（切片+背景核验 · 冻结方法）",
    "YOLO_T_LOC": "YOLO_T_LOC（1024px 切片 · 类无关候选框定位 Top-200）",
    "YOLO_T_QWEN3_CLS": "YOLO_T_QWEN3_CLS（YOLO 定位 Top-200 + Qwen3 裁剪分类）",
    "COMPOSITE_YOLO": "COMPOSITE_YOLO（复合检测 · YOLO11n 640px切片 6类直接输出）",
    "COMPOSITE_YOLO_QWEN3": "COMPOSITE_YOLO_QWEN3（复合检测 · YOLO11n + Qwen3 有限难样本核验）",
}


def _classify_job_outcome(
    status: str,
    *,
    kept_predictions_count: Optional[int] = None,
    termination_v2: Optional[Mapping[str, Any]] = None,
    technical_complete: Optional[bool] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a single-model job state into explicit user-facing detection outcomes.

    Strictly distinguishes:
    - waiting / queued
    - running
    - completed_with_boxes (valid target predictions parsed)
    - completed_zero_boxes (technical run finished with 0 valid boxes; never equated to 'no components on PCB',
      and distinguishes explicit empty array [] from length-limited or unparseable output)
    - failed / incomplete (never disguised as empty predictions)
    - cancelled
    """
    st = str(status or "").strip()
    if st in ("waiting", "queued"):
        return {
            "outcome_code": "waiting",
            "outcome_label": "等待中",
            "has_valid_boxes": False,
            "zero_box_reason": None,
            "outcome_note": "任务在串行队列中等待启动。",
        }
    if st in ("running", "running_base_inference", "preparing_verifier", "running_verifier"):
        return {
            "outcome_code": "running",
            "outcome_label": "运行中",
            "has_valid_boxes": False,
            "zero_box_reason": None,
            "outcome_note": "正在独占单 GPU 执行推理。",
        }
    if st == "cancelled":
        return {
            "outcome_code": "cancelled",
            "outcome_label": "已取消",
            "has_valid_boxes": False,
            "zero_box_reason": None,
            "outcome_note": error_message or "任务已被用户取消，不生成预测成绩。",
        }
    if st in ("failed", "incomplete") or technical_complete is False:
        return {
            "outcome_code": "failed",
            "outcome_label": "运行失败" if st == "failed" else "推理未完整（失败）",
            "has_valid_boxes": False,
            "zero_box_reason": None,
            "outcome_note": error_message or "推理未完整结束或门禁未通过，拒绝伪装为空预测。",
        }
    if st == "completed":
        cnt = int(kept_predictions_count or 0)
        tv2 = dict(termination_v2 or {})
        tot = int(tv2.get("total_calls") or 0)
        exp_empty = int(tv2.get("explicit_empty") or 0)
        len_lim = int(tv2.get("length_limited_without_eos") or 0)
        syn_inc = int(tv2.get("syntax_or_format_incomplete_after_eos") or 0)
        if cnt > 0:
            note = f"模型预测检出 {cnt} 个目标框（GT-free 未做精度评价，框数不等于板上真实元器件总数）。"
            if len_lim > 0:
                note += f" 注意：{len_lim}/{tot} 次调用达 token 上限未闭合（已解析闭合前缀中的合法框）。"
            return {
                "outcome_code": "completed_with_boxes",
                "outcome_label": f"完成（检出 {cnt} 框）",
                "has_valid_boxes": True,
                "zero_box_reason": None,
                "outcome_note": note,
            }
        if tot > 0 and exp_empty == tot:
            reason = "explicit_empty_array"
            lbl = "无有效框（显式输出 []）"
            note = (
                "模型正常结束并显式输出空数组 []。“无有效框”仅表示模型在当前模式下未检出目标，"
                "不等于电路板上没有元器件。"
            )
        elif len_lim > 0:
            reason = "length_limited_without_eos"
            lbl = "无有效框（达 token 上限截断）"
            note = (
                f"模型推理完成但 {len_lim}/{tot} 次调用达 512 token 上限未正常闭合且未解析出合法目标框；"
                "此状态属于模型输出不合规/截断导致的零有效框，不伪装为显式空预测 []，也不等于板上无元器件。"
            )
        elif syn_inc > 0:
            reason = "format_non_compliant_zero_valid_boxes"
            lbl = "无有效框（输出格式不合规）"
            note = (
                "模型输出结束但未满足严格 6 类 JSON 坐标契约，解析后有效框为 0；"
                "不伪装为显式空预测 []，也不等于板上无元器件。"
            )
        else:
            reason = "zero_kept_after_filter"
            lbl = "无有效框（过滤后 0 框）"
            note = "推理完成但过滤后保留有效预测框为 0；“无有效框”不等于电路板上没有元器件。"
        return {
            "outcome_code": "completed_zero_boxes",
            "outcome_label": lbl,
            "has_valid_boxes": False,
            "zero_box_reason": reason,
            "outcome_note": note,
        }
    return {
        "outcome_code": st or "unknown",
        "outcome_label": st or "未知状态",
        "has_valid_boxes": False,
        "zero_box_reason": None,
        "outcome_note": error_message or "",
    }


class DemoError(Exception):
    """Structured domain error mapped to HTTP status codes cleanly."""

    def __init__(self, message: str, *, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.details = dict(details or {})


class BusyGpuJobError(DemoError):
    """Raised when a second GPU job is submitted while another job owns the single GPU lock."""

    def __init__(self, message: str, *, active_job_id: str) -> None:
        super().__init__(message, status_code=409, details={"active_job_id": active_job_id})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_filename(raw_name: str, default_stem: str = "board_upload") -> str:
    """Sanitize uploaded filenames against path traversal and unsafe characters."""
    base = pathlib.Path(str(raw_name or "").replace("\\", "/")).name
    if not base or base in (".", ".."):
        return f"{default_stem}.png"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pathlib.Path(base).stem).strip("._") or default_stem
    suffix = pathlib.Path(base).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        suffix = ".png"
    return f"{stem[:80]}{suffix}"


def normalize_mode(mode: str) -> str:
    m = str(mode or "").strip()
    if m in ("W", "T", "T_B_other", "YOLO_T_LOC", "YOLO_T_QWEN3_CLS", "COMPOSITE_YOLO", "COMPOSITE_YOLO_QWEN3"):
        return m
    if m in ("T+B_other", "T + B_other", "B_other"):
        return "T_B_other"
    raise DemoError(
        f"不支持的检测模式：{mode!r}；仅支持 W、T、T_B_other、YOLO_T_LOC、YOLO_T_QWEN3_CLS、COMPOSITE_YOLO 或 COMPOSITE_YOLO_QWEN3。",
        status_code=400,
    )


def validate_model_and_mode(model_key: str, mode: str) -> Tuple[str, str]:
    mk = str(model_key or "").strip().lower()
    if mk not in ALL_SUPPORTED_MODELS:
        raise DemoError(f"不支持的模型：{model_key!r}；仅支持 qwen2、deepseek、qwen3、yolo11n/yolo11n_qwen3 或复合检测 yolo11n_composite/yolo11n_qwen3_composite。", status_code=400)
    norm_mode = normalize_mode(mode)
    if norm_mode == "T_B_other" and mk != "qwen3":
        raise DemoError(
            "冻结二次核验方法 T + B_other 仅适用于 Qwen3-VL-2B-Instruct，不支持在其他模型上启用。",
            status_code=400,
        )
    if norm_mode == "YOLO_T_LOC" and mk != "yolo11n":
        raise DemoError("阶段 A 专用候选定位模式 YOLO_T_LOC 仅适用于 model_key='yolo11n'。", status_code=400)
    if norm_mode == "YOLO_T_QWEN3_CLS" and mk != "yolo11n_qwen3":
        raise DemoError("阶段 B 组合实验模式 YOLO_T_QWEN3_CLS 仅适用于 model_key='yolo11n_qwen3'。", status_code=400)
    if norm_mode == "COMPOSITE_YOLO" and mk != "yolo11n_composite":
        raise DemoError("复合检测模式 COMPOSITE_YOLO 仅适用于 model_key='yolo11n_composite'。", status_code=400)
    if norm_mode == "COMPOSITE_YOLO_QWEN3" and mk != "yolo11n_qwen3_composite":
        raise DemoError("复合核验模式 COMPOSITE_YOLO_QWEN3 仅适用于 model_key='yolo11n_qwen3_composite'。", status_code=400)
    if mk in EXPERIMENTAL_MODEL_ORDER and norm_mode not in ("YOLO_T_LOC", "YOLO_T_QWEN3_CLS"):
        raise DemoError(f"实验入口 {mk!r} 不能使用模式 {norm_mode!r}。", status_code=400)
    if mk in COMPOSITE_MODEL_ORDER and norm_mode not in ("COMPOSITE_YOLO", "COMPOSITE_YOLO_QWEN3"):
        raise DemoError(f"复合检测入口 {mk!r} 不能使用模式 {norm_mode!r}。", status_code=400)
    return mk, norm_mode


def _write_json_atomic(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON via a collision-free unique temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:12]}.tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _match_preds_with_gt(
    gts: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    *,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Match board GT boxes and predictions one-to-one using repository evaluate.py logic."""
    gts_by_cat: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for gi, g in enumerate(gts):
        cat = str(g["label"]).strip().lower()
        gts_by_cat.setdefault(cat, []).append(
            (gi, {"bbox": [float(v) for v in g["bbox_xyxy"]], "category": cat})
        )

    preds_by_cat: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for pi, p in enumerate(preds):
        cat = str(p["label"]).strip().lower()
        score = float(p.get("score", 1.0)) if p.get("score") is not None else 1.0
        preds_by_cat.setdefault(cat, []).append(
            (pi, {"bbox": [float(v) for v in p["bbox_xyxy"]], "category": cat, "score": score})
        )

    pred_match: Dict[int, Tuple[int, float]] = {}
    gt_match: Dict[int, Tuple[int, float]] = {}
    for cat in sorted(set(gts_by_cat) & set(preds_by_cat)):
        match_res = repo_eval.match_category_predictions_one_to_one(
            [item[1] for item in gts_by_cat[cat]],
            [item[1] for item in preds_by_cat[cat]],
            iou_threshold=iou_thresh,
        )
        for p_idx, g_idx, ov in match_res["matched_pairs"]:
            pi = preds_by_cat[cat][p_idx][0]
            gi = gts_by_cat[cat][g_idx][0]
            pred_match[pi] = (int(gts[gi]["id"]), round(ov, 4))
            gt_match[gi] = (pi, round(ov, 4))

    gt_rows: List[Dict[str, Any]] = []
    for gi, g in enumerate(gts):
        row = dict(g)
        if gi in gt_match:
            row["match_status"] = "TP_GT"
            row["matched_iou"] = gt_match[gi][1]
        else:
            row["match_status"] = "FN"
            row["matched_iou"] = None
        gt_rows.append(row)

    pred_rows: List[Dict[str, Any]] = []
    for pi, p in enumerate(preds):
        row = dict(p)
        if pi in pred_match:
            row["match_status"] = "TP"
            row["matched_gt_id"] = pred_match[pi][0]
            row["matched_iou"] = pred_match[pi][1]
        else:
            row["match_status"] = "FP"
            row["matched_gt_id"] = None
            row["matched_iou"] = None
        pred_rows.append(row)

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in FICS_SIX_CLASSES:
        c_gts = [g for g in gt_rows if g["label"] == cls]
        c_preds = [p for p in pred_rows if p["label"] == cls]
        c_tp = sum(1 for p in c_preds if p["match_status"] == "TP")
        c_fp = len(c_preds) - c_tp
        c_fn = len(c_gts) - c_tp
        c_p = (c_tp / len(c_preds)) if c_preds else 0.0
        if len(c_gts) == 0:
            c_r: Optional[float] = None
            c_f1: Optional[float] = None
            r_disp = "N/A (GT=0)"
            f1_disp = "N/A (GT=0)"
        else:
            c_r = round(c_tp / len(c_gts), 6)
            c_f1 = round((2 * c_p * (c_tp / len(c_gts)) / (c_p + (c_tp / len(c_gts)))) if (c_p + (c_tp / len(c_gts))) > 0 else 0.0, 6)
            r_disp = f"{c_r * 100:.2f}%"
            f1_disp = f"{c_f1 * 100:.2f}%"
        per_class[cls] = {
            "gt_count": len(c_gts),
            "pred_count": len(c_preds),
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": round(c_p, 6),
            "recall": c_r,
            "recall_display": r_disp,
            "f1": c_f1,
            "f1_display": f1_disp,
        }

    tp_tot = len(pred_match)
    fp_tot = len(pred_rows) - tp_tot
    fn_tot = len(gt_rows) - tp_tot
    prec = (tp_tot / len(pred_rows)) if pred_rows else 0.0
    if len(gt_rows) == 0:
        rec: Optional[float] = None
        f1: Optional[float] = None
        rec_disp = "N/A (GT=0)"
        f1_disp_tot = "N/A (GT=0)"
    else:
        rec_raw = tp_tot / len(gt_rows)
        rec = round(rec_raw, 6)
        f1 = round((2 * prec * rec_raw / (prec + rec_raw)) if (prec + rec_raw) > 0 else 0.0, 6)
        rec_disp = f"{rec * 100:.2f}%"
        f1_disp_tot = f"{f1 * 100:.2f}%"
    return {
        "metrics": {
            "tp": tp_tot,
            "fp": fp_tot,
            "fn": fn_tot,
            "total_gt": len(gt_rows),
            "total_predictions": len(pred_rows),
            "precision": round(prec, 6),
            "recall": rec,
            "recall_display": rec_disp,
            "f1": f1,
            "f1_display": f1_disp_tot,
            "per_class": per_class,
        },
        "gts": gt_rows,
        "predictions": pred_rows,
    }


def _match_preds_with_gt_class_agnostic(
    gts: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    *,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Class-agnostic 1-to-1 matching for evaluating candidate proposal localization (never treated as final 6-class detection TP)."""
    any_gts = [{"bbox": [float(v) for v in g["bbox_xyxy"]], "category": "any"} for g in gts]
    any_preds = [
        {
            "bbox": [float(v) for v in p["bbox_xyxy"]],
            "category": "any",
            "score": float(p.get("score", 1.0)) if p.get("score") is not None else 1.0,
        }
        for p in preds
    ]
    match_res = repo_eval.match_category_predictions_one_to_one(
        any_gts,
        any_preds,
        iou_threshold=iou_thresh,
    )
    pred_match: Dict[int, Tuple[int, float]] = {}
    gt_match: Dict[int, Tuple[int, float]] = {}
    for p_idx, g_idx, ov in match_res["matched_pairs"]:
        pred_match[p_idx] = (int(gts[g_idx]["id"]), round(float(ov), 4))
        gt_match[g_idx] = (p_idx, round(float(ov), 4))

    gt_rows: List[Dict[str, Any]] = []
    for gi, g in enumerate(gts):
        row = dict(g)
        if gi in gt_match:
            row["match_status"] = "TP_GT"
            row["matched_iou"] = gt_match[gi][1]
        else:
            row["match_status"] = "FN"
            row["matched_iou"] = None
        gt_rows.append(row)

    pred_rows: List[Dict[str, Any]] = []
    for pi, p in enumerate(preds):
        row = dict(p)
        if pi in pred_match:
            row["match_status"] = "TP"
            row["matched_gt_id"] = pred_match[pi][0]
            row["matched_iou"] = pred_match[pi][1]
        else:
            row["match_status"] = "FP"
            row["matched_gt_id"] = None
            row["matched_iou"] = None
        pred_rows.append(row)

    per_class: Dict[str, Dict[str, Any]] = {}
    for cls in FICS_SIX_CLASSES:
        c_gts = [g for g in gt_rows if g["label"] == cls]
        c_tp = sum(1 for g in c_gts if g["match_status"] == "TP_GT")
        c_fn = len(c_gts) - c_tp
        if len(c_gts) == 0:
            c_r: Optional[float] = None
            r_disp = "N/A (GT=0)"
        else:
            c_r = round(c_tp / len(c_gts), 6)
            r_disp = f"{c_r * 100:.2f}%"
        per_class[cls] = {
            "gt_count": len(c_gts),
            "pred_count": 0,
            "tp": c_tp,
            "fp": 0,
            "fn": c_fn,
            "precision": 0.0,
            "recall": c_r,
            "recall_display": r_disp,
            "f1": None,
            "f1_display": "N/A (类无关定位)",
        }

    tp_tot = len(pred_match)
    fp_tot = len(pred_rows) - tp_tot
    fn_tot = len(gt_rows) - tp_tot
    prec = (tp_tot / len(pred_rows)) if pred_rows else 0.0
    rec_raw = (tp_tot / len(gt_rows)) if gt_rows else 0.0
    rec = round(rec_raw, 6) if gt_rows else None
    f1 = round((2 * prec * rec_raw / (prec + rec_raw)) if (prec + rec_raw) > 0 else 0.0, 6) if gt_rows else None
    return {
        "metrics": {
            "eval_protocol": "class_agnostic_proposal_localization_not_final_detection_tp",
            "tp": tp_tot,
            "fp": fp_tot,
            "fn": fn_tot,
            "total_gt": len(gt_rows),
            "total_predictions": len(pred_rows),
            "precision": round(prec, 6),
            "recall": rec,
            "recall_display": f"{rec * 100:.2f}%" if rec is not None else "N/A (GT=0)",
            "f1": f1,
            "f1_display": f"{f1 * 100:.2f}%" if f1 is not None else "N/A (GT=0)",
            "per_class": per_class,
        },
        "gts": gt_rows,
        "predictions": pred_rows,
    }


def _count_classes(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {cls: 0 for cls in FICS_SIX_CLASSES}
    for p in predictions:
        lbl = str(p.get("label") or p.get("orig_pred_label") or "").strip().lower()
        if lbl in counts:
            counts[lbl] += 1
    return counts


def terminate_process_tree(proc: subprocess.Popen[Any], wait_timeout_s: float = 8.0) -> None:
    """Safely terminate a subprocess and all of its child workers on Windows or POSIX."""
    if proc.poll() is not None:
        return
    pid = int(proc.pid)
    try:
        if sys.platform.startswith("win"):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(5.0, wait_timeout_s),
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=min(3.0, wait_timeout_s))
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
    finally:
        try:
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def render_roi_overlay_png(
    *,
    roi_image_path: pathlib.Path,
    roi_offset_x: int,
    roi_offset_y: int,
    predictions: Sequence[Mapping[str, Any]],
    rejected_candidates: Sequence[Mapping[str, Any]] = (),
    show_rejected: bool = False,
) -> bytes:
    """Render an annotated ROI PNG image matching GT-free or verified boxes."""
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

    def _draw_box(item: Mapping[str, Any], color: Tuple[int, int, int], prefix: str) -> None:
        box = item.get("bbox_xyxy") or item.get("orig_pred_bbox_xyxy")
        if not box or len(box) != 4:
            return
        x1 = (float(box[0]) - roi_offset_x) * scale
        y1 = (float(box[1]) - roi_offset_y) * scale
        x2 = (float(box[2]) - roi_offset_x) * scale
        y2 = (float(box[3]) - roi_offset_y) * scale
        x1_c = max(0.0, min(rgb.width - 1.0, x1))
        y1_c = max(0.0, min(rgb.height - 1.0, y1))
        x2_c = max(x1_c + 2.0, min(float(rgb.width), x2))
        y2_c = max(y1_c + 2.0, min(float(rgb.height), y2))
        draw.rectangle([x1_c, y1_c, x2_c, y2_c], outline=color, width=stroke_w)
        lbl = str(item.get("label") or item.get("orig_pred_label") or "?")
        text = f"{prefix}{lbl}"
        ty = max(2.0, y1_c - 12.0)
        draw.rectangle([x1_c, ty, min(float(rgb.width), x1_c + len(text) * 6.5 + 6), ty + 12.0], fill=(15, 23, 42))
        draw.text((x1_c + 2.0, ty + 1.0), text, fill=color, font=font)

    if show_rejected:
        for rej in rejected_candidates:
            _draw_box(rej, (244, 63, 94), "Filtered:")

    palette: Dict[str, Tuple[int, int, int]] = {
        "capacitor": (56, 189, 248),
        "resistor": (34, 197, 94),
        "inductor": (250, 204, 21),
        "transistor": (192, 132, 252),
        "diode": (251, 146, 60),
        "ic": (45, 212, 191),
    }
    for pred in predictions:
        lbl = str(pred.get("label") or pred.get("orig_pred_label") or "").lower()
        ms = pred.get("match_status")
        if ms == "TP":
            col = (56, 189, 248)
            pref = f"TP:{pred.get('matched_iou', '')} "
        elif ms == "FP":
            col = (239, 68, 68)
            pref = "FP:"
        else:
            col = palette.get(lbl, (56, 189, 248))
            pref = ""
        _draw_box(pred, col, pref)

    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


class ResearchReplayStore:
    """Read-only 8-board (6 dev/historical + 2 blind held-out) research replay store."""

    def __init__(self, index_path: Optional[pathlib.Path]) -> None:
        self.index_path = pathlib.Path(index_path) if index_path else None
        self.available = False
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.scope_isolation: Dict[str, Any] = {}
        self.method_extensions: Dict[str, Any] = {}
        self.heldout_t_b_other: Optional[Dict[str, Any]] = None
        self._roi_cache: Dict[Tuple[str, str, int, Optional[str]], bytes] = {}
        if self.index_path and self.index_path.is_file():
            self._load()

    @staticmethod
    def _safe_is_file(path_like: Any) -> bool:
        if not path_like:
            return False
        try:
            return pathlib.Path(str(path_like)).is_file()
        except OSError:
            return False

    @staticmethod
    def _verify_entry(base_dir: pathlib.Path, entry: Mapping[str, Any], ctx: str) -> pathlib.Path:
        raw_p = pathlib.Path(str(entry["path"]))
        p = raw_p if raw_p.is_absolute() else (base_dir / raw_p)
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Research artifact missing ({ctx}): {p}")
        exp = entry.get("sha256")
        if exp:
            act = file_sha256(p)
            if act.lower() != str(exp).strip().lower():
                raise ValueError(f"SHA-256 mismatch for ({ctx}): {act} != {exp}")
        return p

    def _load(self) -> None:
        assert self.index_path is not None
        idx = json.loads(self.index_path.read_text(encoding="utf-8"))
        base_dir = self.index_path.parent
        self.scope_isolation = dict(idx.get("scope_isolation") or {})

        for ds_key, ds_meta in (idx.get("datasets") or {}).items():
            verified_arts = {
                ak: self._verify_entry(base_dir, av, f"{ds_key}.artifacts.{ak}")
                for ak, av in (ds_meta.get("artifacts") or {}).items()
            }
            rep = json.loads(verified_arts["replay_report"].read_text(encoding="utf-8"))
            coco = json.loads(verified_arts["gt_coco"].read_text(encoding="utf-8"))
            cat_map = {int(c["id"]): str(c["name"]).lower() for c in coco["categories"]}

            boards: Dict[str, Dict[str, Any]] = {}
            img_id_to_board: Dict[int, str] = {}
            for img in coco["images"]:
                bid = str(img.get("board_id") or pathlib.Path(str(img["file_name"])).stem)
                img_id_to_board[int(img["id"])] = bid
                boards[bid] = {
                    "board_id": bid,
                    "coco_image_id": int(img["id"]),
                    "full_width": int(img["width"]),
                    "full_height": int(img["height"]),
                    "file_name": img.get("file_name"),
                    "gts": [],
                    "roi_offset_x": 0,
                    "roi_offset_y": 0,
                    "roi_width": int(img["width"]),
                    "roi_height": int(img["height"]),
                    "roi_image_path": None,
                    "roi_sha256": None,
                }
            for ann in coco["annotations"]:
                bid = img_id_to_board[int(ann["image_id"])]
                x, y, w, h = [float(v) for v in ann["bbox"]]
                boards[bid]["gts"].append({
                    "id": int(ann["id"]),
                    "label": cat_map[int(ann["category_id"])],
                    "bbox_xyxy": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                })

            raw_by_mb: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for mk in MODEL_ORDER:
                raw_p = self._verify_entry(base_dir, ds_meta["raw_jsonl"][mk], f"{ds_key}.raw_jsonl.{mk}")
                eos_ids = (100001,) if mk == "deepseek" else (151645, 151643)
                for line in raw_p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    bid = str(r["board_id"])
                    if bid in boards and r.get("mode") == "W" and boards[bid]["roi_image_path"] is None:
                        cand_str = str(r.get("image_path") or "")
                        boards[bid]["roi_image_path"] = cand_str if self._safe_is_file(cand_str) else r.get("image_path")
                        boards[bid]["roi_sha256"] = r.get("expected_sha256") or r.get("input_sha256")
                        boards[bid]["roi_offset_x"] = int(r.get("roi_offset_x", 0))
                        boards[bid]["roi_offset_y"] = int(r.get("roi_offset_y", 0))
                        boards[bid]["roi_width"] = int(r.get("orig_input_width", boards[bid]["full_width"]))
                        boards[bid]["roi_height"] = int(r.get("orig_input_height", boards[bid]["full_height"]))
                    v2 = classify_call_termination_and_parse_v2(
                        raw_text=str(r.get("raw_output") or ""),
                        width=int(r["orig_input_width"]),
                        height=int(r["orig_input_height"]),
                        generated_tokens=r.get("generated_tokens"),
                        max_new_tokens=int(r.get("max_new_tokens", 512)),
                        generated_token_ids=r.get("generated_token_ids"),
                        eos_token_ids=eos_ids,
                    )
                    raw_by_mb.setdefault((mk, bid), []).append({
                        "call_id": r["call_id"],
                        "mode": r["mode"],
                        "tile_id": r.get("tile_id"),
                        "width": int(r["orig_input_width"]),
                        "height": int(r["orig_input_height"]),
                        "generated_tokens": r.get("generated_tokens"),
                        "gpu_generate_ms": r.get("gpu_generate_ms"),
                        "cuda_max_reserved_mib": r.get("cuda_max_reserved_mib"),
                        "normal_eos": v2["normal_eos"],
                        "length_limited_without_eos": v2["length_limited_without_eos"],
                        "strict_schema_ok": v2["strict_schema_ok"],
                        "explicit_empty": v2["explicit_empty"],
                        "status": v2["status"],
                        "parsed_count": len(v2["objects"]),
                        "raw_output": str(r.get("raw_output") or ""),
                    })

            b_other_data = None
            b_manifest_preds_by_pid: Dict[str, Mapping[str, Any]] = {}
            if "b_other_predictions_gt_free" in verified_arts:
                b_other_data = json.loads(verified_arts["b_other_predictions_gt_free"].read_text(encoding="utf-8"))
                if "candidate_manifest" in verified_arts:
                    cman = json.loads(verified_arts["candidate_manifest"].read_text(encoding="utf-8"))
                    b_manifest_preds_by_pid = {str(p["prediction_id"]): p for p in cman.get("predictions") or []}
                self.heldout_t_b_other = {
                    "predictions": b_other_data,
                    "results": json.loads(verified_arts["heldout_results"].read_text(encoding="utf-8"))
                    if "heldout_results" in verified_arts else None,
                }

            self.datasets[ds_key] = {
                "dataset_key": ds_key,
                "label": ds_meta["label"],
                "split_role": ds_meta["split_role"],
                "split_group": ds_meta.get("split_group", "historical_or_dev"),
                "call_counts_per_model": dict(ds_meta.get("call_counts_per_model") or {}),
                "boards": boards,
                "replay": rep,
                "b_other_data": b_other_data,
                "b_manifest_preds_by_pid": b_manifest_preds_by_pid,
                "raw_by_mb": raw_by_mb,
                "provenance": {
                    k: {"path": str(v), "sha256": ds_meta["artifacts"][k].get("sha256")}
                    for k, v in verified_arts.items()
                },
                "raw_provenance": {
                    mk: {"path": ds_meta["raw_jsonl"][mk]["path"], "sha256": ds_meta["raw_jsonl"][mk].get("sha256")}
                    for mk in MODEL_ORDER
                },
            }

        for ek, ev in (idx.get("method_extensions") or {}).items():
            ep = self._verify_entry(base_dir, ev, f"method_extensions.{ek}")
            self.method_extensions[ek] = json.loads(ep.read_text(encoding="utf-8"))
        self.available = True

    def get_overview(self) -> Dict[str, Any]:
        if not self.available:
            return {"available": False, "notice": "未配置 research results_index.json，只读回放模块处于未加载状态。", "datasets": {}}

        ds_out: Dict[str, Any] = {}
        for ds_key, ds in self.datasets.items():
            rep = ds["replay"]
            models_summary = {}
            for mk in MODEL_ORDER:
                comb = {}
                for mode in ("W", "T"):
                    ev = rep["models"][mk]["combined"][mode]["evaluate_detections_iou50"]
                    comb[mode] = {
                        k: ev[k] for k in ("tp", "fp", "fn", "total_gt", "total_predictions", "micro")
                    }
                models_summary[mk] = {
                    "termination_v2": rep["models"][mk]["termination_v2"],
                    "combined": comb,
                }
            if ds_key == "heldout_blind_s19_s16" and self.heldout_t_b_other and self.heldout_t_b_other.get("results"):
                hr_b = self.heldout_t_b_other["results"]["groups"]["qwen3_T_B_other"]["combined"]
                models_summary["qwen3"]["combined"]["T_B_other"] = {
                    "tp": hr_b["tp"],
                    "fp": hr_b["fp"],
                    "fn": hr_b["fn"],
                    "total_gt": hr_b["gt"],
                    "total_predictions": hr_b["predictions"],
                    "micro": {
                        "precision": hr_b["precision"],
                        "recall": hr_b["recall"],
                        "f1": hr_b["f1"],
                    },
                }

            ds_out[ds_key] = {
                "label": ds["label"],
                "split_role": ds["split_role"],
                "split_group": ds["split_group"],
                "call_counts_per_model": ds["call_counts_per_model"],
                "boards": [
                    {
                        "board_id": bid,
                        "coco_image_id": b["coco_image_id"],
                        "gt_count": len(b["gts"]),
                        "roi_width": b["roi_width"],
                        "roi_height": b["roi_height"],
                        "roi_image_available": self._safe_is_file(b["roi_image_path"]),
                    }
                    for bid, b in ds["boards"].items()
                ],
                "models": models_summary,
            }

        method_cards: Dict[str, Any] = {}
        if "candidate_A_combined6" in self.method_extensions:
            ca = self.method_extensions["candidate_A_combined6"]
            method_cards["method_A_six_boards_268_gt"] = {
                "scope": "6 historical/dev boards (268 GT)",
                "micro_summary": ca.get("micro_summary"),
                "per_board": ca.get("per_board"),
            }
        if "dev_four_method_comparison" in self.method_extensions:
            dfm = self.method_extensions["dev_four_method_comparison"]
            method_cards["dev_four_methods_s4_s20_163_gt"] = {
                "scope": "Dev expansion s4+s20 (163 GT)",
                "preregistered_selection": dfm.get("preregistered_selection"),
                "methods": {
                    m_k: {
                        "combined": {
                            k: v for k, v in m_v["combined"].items()
                            if k in ("gt", "predictions", "tp", "fp", "fn", "precision", "recall", "f1", "per_class")
                        }
                    }
                    for m_k, m_v in (dfm.get("methods") or {}).items()
                },
            }
        if "heldout_results" in self.method_extensions:
            hr = self.method_extensions["heldout_results"]
            method_cards["heldout_blind_s19_s16_107_gt"] = {
                "scope": "Held-out blind s19+s16 (107 GT)",
                "qwen3_T_B_other": hr["groups"]["qwen3_T_B_other"]["combined"],
                "verifier_audit": hr.get("verifier_output_audit"),
                "descriptive_8_board_overview": hr.get("descriptive_8_board_375_gt_overview"),
            }

        return {
            "available": True,
            "notice": "8 板（375 GT）真实研究结果只读回放：严格区分历史/开发集（6 板 268 GT）与最终盲测保留集（2 板 107 GT）。",
            "scope_isolation": self.scope_isolation,
            "datasets": ds_out,
            "method_cards": method_cards,
        }

    def get_board_view(self, *, dataset_key: str, board_id: str, model_key: str, mode: str) -> Dict[str, Any]:
        if not self.available:
            raise DemoError("未配置 research results_index.json。", status_code=404)
        if dataset_key not in self.datasets:
            raise DemoError(f"未知数据集：{dataset_key!r}", status_code=404)
        ds = self.datasets[dataset_key]
        if board_id not in ds["boards"]:
            raise DemoError(f"板卡 {board_id!r} 不在数据集 {dataset_key!r} 中。", status_code=404)
        if model_key not in MODEL_ORDER:
            raise DemoError(f"不支持的模型：{model_key!r}", status_code=400)

        norm_mode = normalize_mode(mode)
        b = ds["boards"][board_id]
        if norm_mode == "T_B_other":
            if model_key != "qwen3" or not ds.get("b_other_data"):
                raise DemoError(
                    f"数据集 {dataset_key!r} / 模型 {model_key!r} 未运行冻结 T+B_other 回放。",
                    status_code=400,
                )
            b_board = ds["b_other_data"]["per_board"][board_id]
            kept_objs = list(b_board["pred_record"]["objects"])
            matched = _match_preds_with_gt(b["gts"], kept_objs, iou_thresh=0.50)
            man_map = ds.get("b_manifest_preds_by_pid") or {}
            pre_preds = [
                {
                    "prediction_id": d["prediction_id"],
                    "label": d["orig_pred_label"],
                    "bbox_xyxy": man_map[str(d["prediction_id"])]["orig_pred_bbox_xyxy"],
                    "verifier_label": d.get("strict_label"),
                    "kept": bool(d.get("retained")),
                    "decision_reason": d.get("reason"),
                }
                for d in b_board.get("decisions") or []
                if str(d["prediction_id"]) in man_map
            ]
            call_mode = "T"
            calls = [c for c in ds["raw_by_mb"].get((model_key, board_id), []) if c["mode"] == call_mode]
            return {
                "dataset_key": dataset_key,
                "dataset_label": ds["label"],
                "split_group": ds["split_group"],
                "board_id": board_id,
                "coco_image_id": b["coco_image_id"],
                "model_key": model_key,
                "mode": norm_mode,
                "call_counts_per_model": ds["call_counts_per_model"],
                "roi": {
                    "offset_x": b["roi_offset_x"],
                    "offset_y": b["roi_offset_y"],
                    "width": b["roi_width"],
                    "height": b["roi_height"],
                    "image_available": self._safe_is_file(b["roi_image_path"]),
                    "sha256": b["roi_sha256"],
                },
                "metrics": matched["metrics"],
                "gts": matched["gts"],
                "predictions": matched["predictions"],
                "pre_verifier_candidates": pre_preds,
                "verifier_summary": {
                    "total_candidates": b_board["total_candidates"],
                    "kept_count": b_board["kept_count"],
                    "rejected_count": b_board["rejected_count"],
                },
                "calls": calls,
                "provenance": ds["provenance"],
                "raw_provenance": ds["raw_provenance"],
            }

        mrep = ds["replay"]["models"][model_key]["per_board"][board_id][norm_mode]
        preds_raw = list((mrep.get("pred_record") or {}).get("objects") or [])
        matched = _match_preds_with_gt(b["gts"], preds_raw, iou_thresh=0.50)
        calls = [c for c in ds["raw_by_mb"].get((model_key, board_id), []) if c["mode"] == norm_mode]
        return {
            "dataset_key": dataset_key,
            "dataset_label": ds["label"],
            "split_group": ds["split_group"],
            "board_id": board_id,
            "coco_image_id": b["coco_image_id"],
            "model_key": model_key,
            "mode": norm_mode,
            "call_counts_per_model": ds["call_counts_per_model"],
            "roi": {
                "offset_x": b["roi_offset_x"],
                "offset_y": b["roi_offset_y"],
                "width": b["roi_width"],
                "height": b["roi_height"],
                "image_available": self._safe_is_file(b["roi_image_path"]),
                "sha256": b["roi_sha256"],
            },
            "metrics": matched["metrics"],
            "gts": matched["gts"],
            "predictions": matched["predictions"],
            "pre_verifier_candidates": None,
            "calls": calls,
            "provenance": ds["provenance"],
            "raw_provenance": ds["raw_provenance"],
        }

    def render_image_jpeg(
        self,
        *,
        dataset_key: str,
        board_id: str,
        max_width: int = 1600,
        crop: Optional[str] = None,
    ) -> bytes:
        if not self.available or dataset_key not in self.datasets:
            raise DemoError(f"未知数据集：{dataset_key!r}", status_code=404)
        ds = self.datasets[dataset_key]
        if board_id not in ds["boards"]:
            raise DemoError(f"板卡 {board_id!r} 不在 {dataset_key!r}", status_code=404)
        b = ds["boards"][board_id]
        img_str = b.get("roi_image_path")
        if not self._safe_is_file(img_str):
            raise DemoError(f"板卡 ROI 图像不可用：{img_str!r}", status_code=404)
        exp_sha = b.get("roi_sha256")
        if exp_sha and file_sha256(pathlib.Path(img_str)).lower() != str(exp_sha).lower():
            raise DemoError("ROI 图像 SHA-256 校验失败，拒绝显示错位图像。", status_code=500)

        key = (dataset_key, board_id, int(max_width), crop)
        if key in self._roi_cache:
            return self._roi_cache[key]
        with Image.open(img_str) as im:
            rgb = im.convert("RGB")
            if crop:
                parts = [int(float(v)) for v in crop.split(",")]
                if len(parts) != 4:
                    raise DemoError(f"无效局部裁剪参数：{crop!r}", status_code=400)
                x1 = max(0, min(rgb.width - 1, parts[0]))
                y1 = max(0, min(rgb.height - 1, parts[1]))
                x2 = max(x1 + 1, min(rgb.width, parts[2]))
                y2 = max(y1 + 1, min(rgb.height, parts[3]))
                rgb = rgb.crop((x1, y1, x2, y2))
            if max_width > 0 and rgb.width > max_width:
                new_h = max(1, int(round(rgb.height * (max_width / rgb.width))))
                rgb = rgb.resize((max_width, new_h), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=88)
            data = buf.getvalue()
        self._roi_cache[key] = data
        return data


class LiveDemoService:
    """GT-free single-board live inference job manager, queue lock, exporter, and optional GT evaluator."""

    def __init__(
        self,
        *,
        workspace_root: pathlib.Path,
        runtime_config_path: Optional[pathlib.Path] = None,
        research_store: Optional[ResearchReplayStore] = None,
        job_runner: Optional[Callable[[Dict[str, Any], threading.Event, "LiveDemoService"], None]] = None,
        verifier_runner: Optional[Callable[[Dict[str, Any], pathlib.Path, threading.Event, "LiveDemoService"], None]] = None,
        composite_yolo_runner: Optional[Callable[..., Dict[str, Any]]] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        self.workspace_root = pathlib.Path(workspace_root).resolve()
        self.uploads_dir = self.workspace_root / "uploads"
        self.jobs_dir = self.workspace_root / "jobs"
        self.comparisons_dir = self.workspace_root / "comparisons"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.comparisons_dir.mkdir(parents=True, exist_ok=True)

        self.runtime_config_path = pathlib.Path(runtime_config_path).resolve() if runtime_config_path else None
        self.runtime_config: Dict[str, Any] = {}
        if self.runtime_config_path and self.runtime_config_path.is_file():
            loaded = yaml.safe_load(self.runtime_config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.runtime_config = loaded

        self.python_executable = str(
            python_executable
            or self.runtime_config.get("python_executable")
            or sys.executable
        )
        self.research_store = research_store
        self._custom_job_runner = job_runner
        self._custom_verifier_runner = verifier_runner
        self._custom_composite_yolo_runner = composite_yolo_runner

        self._lock = threading.Lock()
        self._job_state_locks_guard = threading.Lock()
        self._job_state_locks: Dict[str, threading.Lock] = {}
        self._comparison_locks: Dict[str, threading.Lock] = {}
        self._active_job_id: Optional[str] = None
        self._active_comparison_id: Optional[str] = None
        self._cancel_events: Dict[str, threading.Event] = {}
        self._comparison_cancel_events: Dict[str, threading.Event] = {}
        self._cancelled_comparisons: set[str] = set()
        self._active_procs: Dict[str, subprocess.Popen[Any]] = {}
        self._job_threads: Dict[str, threading.Thread] = {}
        self._comparison_threads: Dict[str, threading.Thread] = {}
        self._preview_cache: Dict[Tuple[str, str, int, Optional[str]], bytes] = {}

    def shutdown(self) -> None:
        """Cancel any active comparison queue or GPU job and terminate subprocess trees before exit."""
        with self._lock:
            active_cmp_ids = list(self._comparison_cancel_events.keys())
            active_ids = list(self._cancel_events.keys())
        for cid in active_cmp_ids:
            try:
                self.cancel_comparison(cid)
            except Exception:
                pass
        for jid in active_ids:
            try:
                self.cancel_job(jid)
            except Exception:
                pass
        with self._lock:
            cmp_threads = list(self._comparison_threads.values())
            threads = list(self._job_threads.values())
        for th in cmp_threads:
            if th.is_alive():
                th.join(timeout=8.0)
        for th in threads:
            if th.is_alive():
                th.join(timeout=8.0)

    def get_ui_defaults(self) -> Dict[str, Any]:
        """Return cold-start UI defaults from runtime_config (preserving isolated 3-model baseline entry)."""
        cfg = self.runtime_config or {}
        def_model = str(cfg.get("default_model") or "qwen3").strip().lower() or "qwen3"
        def_mode = str(cfg.get("default_mode") or "T").strip() or "T"
        def_preset = str(cfg.get("default_preset") or "r5_qwen3_greedy_maxpx589824_tile768").strip() or "r5_qwen3_greedy_maxpx589824_tile768"
        def_inst_var = str(cfg.get("default_instruction_prompt_variant") or "r8_base").strip() or "r8_base"
        cmp_mode = str(cfg.get("three_model_compare_default_mode") or "W").strip() or "W"
        cmp_preset = str(cfg.get("three_model_compare_default_preset") or "baseline").strip() or "baseline"
        rt_models = cfg.get("models") or {}
        fics_rt = rt_models.get("yolo11n_fics_r15") or {}
        wacv_rt = rt_models.get("yolo11n") or {}
        ic_cal_rt = rt_models.get("yolo11n_fics_r15_ic_cal") or {}
        r17_cand_rt = rt_models.get("yolo11n_fics_r17_cand") or {}
        r18_cv_rt = rt_models.get("yolo11n_fics_r15_crop_verify_r18") or {}
        r19_a1_rt = rt_models.get("yolo11n_fics_r15_r19_a1_ic_reject") or {}
        r19_a3_rt = rt_models.get("yolo11n_fics_r15_r19_a3_combined") or {}
        r20_rt = rt_models.get("yolo11n_fics_r15_r20_a1_ic_relabel") or {}
        fics_path_str = str(fics_rt.get("local_dir") or DEFAULT_FICS_R15_WEIGHTS_PATH).strip()
        fics_available = bool(fics_path_str and pathlib.Path(fics_path_str).is_file())
        presets_list: List[Dict[str, Any]] = [
            {
                "preset_id": "fics_yolo11n_r15",
                "display_name": "FICS-PCB 6类域适配 YOLO11n（R16 默认 · 统一 0.25 阈值）",
                "expected_class_schema": "fics_6",
                "weights_sha256": str(fics_rt.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256),
                "available": fics_available,
            },
            {
                "preset_id": "wacv_yolo11n",
                "display_name": "WACV YOLO11n 旧基线（11类对齐6类 · 对照用）",
                "expected_class_schema": "wacv_11",
                "weights_sha256": str(wacv_rt.get("weights_sha256") or DEFAULT_YOLO11N_WEIGHTS_SHA256),
                "available": True,
            },
        ]
        if r17_cand_rt:
            r17_path_str = str(r17_cand_rt.get("local_dir") or "").strip()
            presets_list.append({
                "preset_id": "fics_yolo11n_r17_cand",
                "display_name": "FICS-PCB 6类域适配候选（R17 弱类扩充 · 隔离候选 · 0.25 阈值）",
                "expected_class_schema": "fics_6",
                "weights_sha256": str(r17_cand_rt.get("weights_sha256") or "").strip().lower(),
                "available": bool(r17_path_str and pathlib.Path(r17_path_str).is_file()),
            })
        if r18_cv_rt:
            r18_yolo_p = str(r18_cv_rt.get("local_dir") or fics_path_str).strip()
            r18_cv_p = str(r18_cv_rt.get("crop_verifier_weights") or "").strip()
            presets_list.append({
                "preset_id": "fics_yolo11n_r15_crop_verify_r18",
                "display_name": "FICS R15 + R18 局部分类核验候选（冻结框 · IC/二极管/晶体管核验）",
                "expected_class_schema": "fics_6",
                "weights_sha256": str(r18_cv_rt.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower(),
                "crop_verifier_sha256": str(r18_cv_rt.get("crop_verifier_sha256") or "").strip().lower(),
                "available": bool(
                    r18_yolo_p and pathlib.Path(r18_yolo_p).is_file()
                    and r18_cv_p and pathlib.Path(r18_cv_p).is_file()
                ),
            })
        if r19_a1_rt:
            r19_a1_yolo_p = str(r19_a1_rt.get("local_dir") or fics_path_str).strip()
            r19_a1_cv_p = str(r19_a1_rt.get("crop_verifier_weights") or "").strip()
            presets_list.append({
                "preset_id": "fics_yolo11n_r15_r19_a1_ic_reject",
                "display_name": "FICS R15 + R19 A1 仅 IC 拒识（冻结框 · 禁用跨类改判）",
                "expected_class_schema": "fics_6",
                "weights_sha256": str(r19_a1_rt.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower(),
                "crop_verifier_sha256": str(r19_a1_rt.get("crop_verifier_sha256") or "").strip().lower(),
                "available": bool(
                    r19_a1_yolo_p and pathlib.Path(r19_a1_yolo_p).is_file()
                    and r19_a1_cv_p and pathlib.Path(r19_a1_cv_p).is_file()
                ),
            })
        if r19_a3_rt:
            r19_a3_yolo_p = str(r19_a3_rt.get("local_dir") or fics_path_str).strip()
            r19_a3_cv_p = str(r19_a3_rt.get("crop_verifier_weights") or "").strip()
            presets_list.append({
                "preset_id": "fics_yolo11n_r15_r19_a3_combined",
                "display_name": "FICS R15 + R19 A3 组合（IC 拒识 + 低分 D/Q 补检）",
                "expected_class_schema": "fics_6",
                "weights_sha256": str(r19_a3_rt.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower(),
                "crop_verifier_sha256": str(r19_a3_rt.get("crop_verifier_sha256") or "").strip().lower(),
                "available": bool(
                    r19_a3_yolo_p and pathlib.Path(r19_a3_yolo_p).is_file()
                    and r19_a3_cv_p and pathlib.Path(r19_a3_cv_p).is_file()
                ),
            })
        if r20_rt:
            r20_yolo_p = str(r20_rt.get("local_dir") or fics_path_str).strip()
            r20_cv_p = str(r20_rt.get("crop_verifier_weights") or "").strip()
            r20_hd_p = str(r20_rt.get("r20_head_weights") or "").strip()
            presets_list.append({
                "preset_id": "fics_yolo11n_r15_r20_a1_ic_relabel",
                "display_name": "FICS R15 + R19 A1 + R20 定向 IC 错类修正（冻结框）",
                "expected_class_schema": "fics_6",
                "weights_sha256": str(r20_rt.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower(),
                "available": bool(
                    r20_yolo_p and pathlib.Path(r20_yolo_p).is_file()
                    and r20_cv_p and pathlib.Path(r20_cv_p).is_file()
                    and r20_hd_p and pathlib.Path(r20_hd_p).is_file()
                ),
            })
        if ic_cal_rt:
            ic_path_str = str(ic_cal_rt.get("local_dir") or fics_path_str).strip()
            ic_thr = float(ic_cal_rt.get("ic_conf_threshold") or 0.45)
            presets_list.append({
                "preset_id": "fics_yolo11n_r15_ic_cal",
                "display_name": f"FICS R15 试验预设 · 仅 IC 阈值={ic_thr:.2f}（其余 0.25）",
                "expected_class_schema": "fics_6",
                "weights_sha256": str(ic_cal_rt.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256),
                "ic_conf_threshold": ic_thr,
                "available": bool(ic_path_str and pathlib.Path(ic_path_str).is_file()),
            })
        return {
            "single_model_defense": {
                "model_key": def_model,
                "mode": def_mode,
                "preset": def_preset,
                "instruction_prompt_variant": def_inst_var,
                "tile_size": 768 if "tile768" in def_preset else 1024,
                "tile_overlap_ratio": 0.20,
                "qwen_min_pixels": 65536,
                "qwen_max_pixels": 589824 if "589824" in def_preset else 200704,
                "max_new_tokens": 512,
                "do_sample": False,
                "quantization": "bf16",
            },
            "three_model_compare": {
                "model_keys": list(MODEL_ORDER),
                "mode": cmp_mode,
                "preset": cmp_preset,
                "tile_size": 1024,
                "tile_overlap_ratio": 0.20,
                "qwen_max_pixels": 200704,
                "deepseek_image_size": 384,
            },
            "composite_detector": {
                "default_detector_preset": str(cfg.get("default_composite_detector_preset") or "fics_yolo11n_r15"),
                "presets": presets_list,
            },
        }

    def _upload_dir(self, upload_id: str) -> pathlib.Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", str(upload_id or "")):
            raise DemoError(f"无效 upload_id：{upload_id!r}", status_code=400)
        d = (self.uploads_dir / upload_id).resolve()
        if d.parent != self.uploads_dir or not d.is_dir():
            raise DemoError(f"上传记录不存在：{upload_id!r}", status_code=404)
        return d

    def _job_dir(self, job_id: str) -> pathlib.Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", str(job_id or "")):
            raise DemoError(f"无效 job_id：{job_id!r}", status_code=400)
        d = (self.jobs_dir / job_id).resolve()
        if d.parent != self.jobs_dir or not d.is_dir():
            raise DemoError(f"任务不存在：{job_id!r}", status_code=404)
        return d

    def _comparison_dir(self, comparison_id: str) -> pathlib.Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", str(comparison_id or "")):
            raise DemoError(f"无效 comparison_id：{comparison_id!r}", status_code=400)
        d = (self.comparisons_dir / comparison_id).resolve()
        if d.parent != self.comparisons_dir or not d.is_dir():
            raise DemoError(f"同图对比记录不存在：{comparison_id!r}", status_code=404)
        return d

    def register_upload(self, *, raw_bytes: bytes, raw_filename: str) -> Dict[str, Any]:
        """Validate uploaded board image, compute GT-free ROI/tiling pre-flight, and persist inputs."""
        if not raw_bytes:
            raise DemoError("上传文件内容为空，请选择有效的 PCB 图像文件。", status_code=400)
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise DemoError(
                f"上传文件超出大小限制（当前 {len(raw_bytes)} 字节，最大允许 {MAX_UPLOAD_BYTES} 字节 / 256 MiB）。",
                status_code=413,
            )

        raw_suffix = pathlib.Path(str(raw_filename or "").strip()).suffix.lower()
        if raw_suffix and raw_suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise DemoError(
                f"不支持的图像文件扩展名 {raw_suffix!r}；请上传单帧 PNG、JPG/JPEG、TIF/TIFF、BMP 或 WEBP 图像。",
                status_code=400,
            )

        safe_name = _safe_filename(raw_filename)
        upload_id = f"up_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        udir = self.uploads_dir / upload_id
        udir.mkdir(parents=True, exist_ok=True)
        src_path = udir / safe_name
        src_path.write_bytes(raw_bytes)

        try:
            with Image.open(src_path) as im:
                n_frames = int(getattr(im, "n_frames", 1) or 1)
                if n_frames > 1:
                    raise ValueError(f"检测到多帧或多页图像 (n_frames={n_frames})，仅支持单帧 PCB 图像")
                im_rgb = ImageOps.exif_transpose(im).convert("RGB")
                orig_w, orig_h = im_rgb.size
                if orig_w < MIN_IMAGE_DIM or orig_h < MIN_IMAGE_DIM:
                    raise ValueError(f"图像尺寸过小 ({orig_w}x{orig_h})，低于最小检测要求 {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}")
                if orig_w > MAX_IMAGE_DIM or orig_h > MAX_IMAGE_DIM or (orig_w * orig_h) > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        f"图像分辨率超限 ({orig_w}x{orig_h}={orig_w * orig_h:,} 像素)，"
                        f"超出单机安全预检上限 (最大边长 {MAX_IMAGE_DIM}，最大总像素 {MAX_IMAGE_PIXELS:,})"
                    )
                arr_check = np.asarray(im_rgb)
                non_white_pixels = int(np.count_nonzero(np.any(arr_check < 248, axis=2)))
                pixel_std = float(arr_check.std())
                if non_white_pixels < 64 or pixel_std < 1.0:
                    raise ValueError("图像内容全空、全白或为纯色背景，未检测到有效 PCB 板卡前景区域")
        except Exception as exc:
            shutil.rmtree(udir, ignore_errors=True)
            raise DemoError(f"图像预检未通过（无法解析或不适合检测）：{exc}", status_code=400) from exc

        board_key = pathlib.Path(safe_name).stem or upload_id
        inputs_dir = udir / "inputs"
        geom = compute_board_roi_and_tiles(
            src_path,
            roi_pad=32,
            white_thresh=248,
            tile_size=1024,
            overlap_ratio=0.20,
            export_dir=inputs_dir,
            board_key=board_key,
        )
        if int(geom["tile_count"]) <= 0 or int(geom["tile_count"]) > MAX_TILE_COUNT:
            shutil.rmtree(udir, ignore_errors=True)
            raise DemoError(
                f"ROI 切片数量超限（tile_count={geom['tile_count']}，单机上限 {MAX_TILE_COUNT}），拒绝启动超界推理。",
                status_code=400,
            )

        img_sha = file_sha256(src_path)
        meta = {
            "schema_version": "fics_upload_preflight_v1",
            "upload_id": upload_id,
            "safe_filename": safe_name,
            "board_id": board_key,
            "source_image_path": str(src_path),
            "source_image_sha256": img_sha,
            "bytes": len(raw_bytes),
            "created_at": _now_iso(),
            "orig_wh": geom["orig_wh"],
            "roi_bbox_half_open": geom["roi_bbox_half_open"],
            "roi_offset_x": geom["roi_offset_x"],
            "roi_offset_y": geom["roi_offset_y"],
            "roi_width": geom["roi_width"],
            "roi_height": geom["roi_height"],
            "tile_count": geom["tile_count"],
            "roi_export": geom["roi_export"],
            "tiles": geom["tiles"],
            "gt_required": False,
        }
        _write_json_atomic(udir / "upload_meta.json", meta)
        return self._public_upload_summary(meta)

    @staticmethod
    def _public_upload_summary(meta: Mapping[str, Any]) -> Dict[str, Any]:
        tc = int(meta["tile_count"])
        within_live_t_budget = tc <= RECOMMENDED_LIVE_T_B_OTHER_MAX_TILES
        return {
            "upload_id": meta["upload_id"],
            "safe_filename": meta["safe_filename"],
            "board_id": meta["board_id"],
            "source_image_sha256": meta["source_image_sha256"],
            "bytes": meta.get("bytes"),
            "orig_wh": meta["orig_wh"],
            "roi_bbox_half_open": meta["roi_bbox_half_open"],
            "roi_offset_x": meta["roi_offset_x"],
            "roi_offset_y": meta["roi_offset_y"],
            "roi_width": meta["roi_width"],
            "roi_height": meta["roi_height"],
            "tile_count": tc,
            "roi_sha256": (meta.get("roi_export") or {}).get("sha256"),
            "gt_required": False,
            "read_only_playback_only": bool(meta.get("read_only_playback_only")),
            "playback_preview_notice": meta.get("playback_preview_notice"),
            "recommended_live_t_b_other_max_tiles": RECOMMENDED_LIVE_T_B_OTHER_MAX_TILES,
            "within_recommended_live_t_b_other_budget": within_live_t_budget,
            "available_modes": [
                {
                    "mode": "W",
                    "display_name": MODE_DISPLAY_NAMES["W"],
                    "supported_models": list(MODEL_ORDER),
                    "planned_base_calls_per_model": 1,
                    "is_default_comparison_mode": True,
                    "workload_note": "每模型 1 次整图 ROI 调用（三模型串行对比共 3 次调用，各模型使用各自冻结原生视觉预处理）",
                },
                {
                    "mode": "T",
                    "display_name": MODE_DISPLAY_NAMES["T"],
                    "supported_models": list(MODEL_ORDER),
                    "planned_base_calls_per_model": tc,
                    "is_default_comparison_mode": False,
                    "workload_note": f"每模型 {tc} 次 1024×1024 切片调用 + 0.50 IoU NMS 去重",
                },
                {
                    "mode": "T_B_other",
                    "display_name": MODE_DISPLAY_NAMES["T_B_other"],
                    "supported_models": ["qwen3"],
                    "planned_base_calls_per_model": tc,
                    "is_default_comparison_mode": False,
                    "within_recommended_live_budget": within_live_t_budget,
                    "workload_note": (
                        f"仅限 Qwen3 单独冻结增强：{tc} 次切片调用 + 候选框 10% 外扩裁剪二次核验 "
                        f"({'符合 <=24 切片现场推荐预算' if within_live_t_budget else f'当前 {tc} 切片超出 <=24 现场快速预算，建议优先使用 W 同图对比'})"
                    ),
                },
            ],
        }

    def get_upload_summary(self, upload_id: str) -> Dict[str, Any]:
        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
        return self._public_upload_summary(meta)

    def _resolve_upload_image_path(self, udir: pathlib.Path, meta: Mapping[str, Any], src_kind: str) -> Tuple[pathlib.Path, bool]:
        """Resolve upload image path portably across moved/isolated workspaces.

        Returns (resolved_path, is_derived_preview_only).
        """
        if src_kind == "orig":
            raw_p = str(meta.get("source_image_path") or "").strip()
            candidates: List[pathlib.Path] = []
            if raw_p:
                p = pathlib.Path(raw_p)
                candidates.append(p if p.is_absolute() else (udir / p))
            if meta.get("safe_filename"):
                candidates.append(udir / str(meta["safe_filename"]))
            for cand in candidates:
                if cand.is_file():
                    return cand.resolve(), False
            raise DemoError(
                "外置全分辨率原图未随当前轻量交付包附带，无法显示完整原图视图；"
                "请切换为「ROI 裁切视图」（只读预览）或按外置路径说明挂载原图。",
                status_code=404,
            )

        roi_meta = meta.get("roi_export") or {}
        raw_roi = str(roi_meta.get("path") or "").strip()
        roi_candidates: List[pathlib.Path] = []
        if raw_roi:
            p = pathlib.Path(raw_roi)
            roi_candidates.append(p if p.is_absolute() else (udir / p))
            roi_candidates.append(udir / "inputs" / p.name)
        for cand in roi_candidates:
            if cand.is_file():
                return cand.resolve(), False

        raw_prev = str(roi_meta.get("playback_preview_path") or "").strip()
        if raw_prev:
            pp = pathlib.Path(raw_prev)
            pp = pp if pp.is_absolute() else (udir / pp)
            if pp.is_file():
                return pp.resolve(), True
        raise DemoError("ROI 预览图像文件缺失。", status_code=404)

    def render_upload_roi_jpeg(
        self,
        *,
        upload_id: str,
        max_width: int = 1600,
        crop: Optional[str] = None,
        source: str = "roi",
    ) -> bytes:
        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
        src_kind = "orig" if str(source or "roi").strip().lower() == "orig" else "roi"
        img_p, is_preview = self._resolve_upload_image_path(udir, meta, src_kind)
        key = (upload_id, src_kind, int(max_width), crop)
        if key in self._preview_cache:
            return self._preview_cache[key]
        with Image.open(img_p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB") if src_kind == "orig" else im.convert("RGB")
            if crop:
                parts = [float(v) for v in crop.split(",")]
                if len(parts) != 4:
                    raise DemoError(f"无效局部裁剪参数：{crop!r}", status_code=400)
                if is_preview and int(meta.get("roi_width") or 0) > 0 and int(meta.get("roi_height") or 0) > 0:
                    sx = rgb.width / float(meta["roi_width"])
                    sy = rgb.height / float(meta["roi_height"])
                    parts = [parts[0] * sx, parts[1] * sy, parts[2] * sx, parts[3] * sy]
                x1 = max(0, min(rgb.width - 1, int(round(parts[0]))))
                y1 = max(0, min(rgb.height - 1, int(round(parts[1]))))
                x2 = max(x1 + 1, min(rgb.width, int(round(parts[2]))))
                y2 = max(y1 + 1, min(rgb.height, int(round(parts[3]))))
                rgb = rgb.crop((x1, y1, x2, y2))
            if max_width > 0 and rgb.width > max_width:
                new_h = max(1, int(round(rgb.height * (max_width / rgb.width))))
                rgb = rgb.resize((max_width, new_h), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=88)
            data = buf.getvalue()
        self._preview_cache[key] = data
        return data

    def _resolve_upload_tiles_for_geometry(
        self,
        upload_id: str,
        meta: Mapping[str, Any],
        *,
        tile_size: int = 1024,
        overlap_ratio: float = 0.20,
        max_tile_count: int = MAX_TILE_COUNT,
    ) -> List[Dict[str, Any]]:
        """Return existing 1024px tiles or deterministically crop & cache alternate tile geometry (e.g. 768px)."""
        ts = int(tile_size)
        ov = float(overlap_ratio)
        if ts == 1024 and abs(ov - 0.20) < 1e-9:
            return [dict(t) for t in (meta.get("tiles") or [])]

        udir = self._upload_dir(upload_id)
        inputs_dir = udir / "inputs"
        ov_pct = int(round(ov * 100))
        manifest_p = inputs_dir / f"tiles_{ts}_ov{ov_pct}_manifest.json"
        if manifest_p.is_file():
            try:
                cached = json.loads(manifest_p.read_text(encoding="utf-8"))
                cached_tiles = cached.get("tiles") or []
                if (
                    int(cached.get("tile_size", -1)) == ts
                    and abs(float(cached.get("overlap_ratio", -1.0)) - ov) < 1e-9
                    and cached_tiles
                    and all(pathlib.Path(str(t["path"])).is_file() for t in cached_tiles)
                ):
                    return [dict(t) for t in cached_tiles]
            except Exception:
                pass

        roi_path, is_preview = self._resolve_upload_image_path(udir, meta, "roi")
        if is_preview:
            raise DemoError("只读回放示例缺少完整 ROI 原图，无法重新生成非默认切片几何。", status_code=400)

        roi_w = int(meta["roi_width"])
        roi_h = int(meta["roi_height"])
        tiles = generate_tiles(roi_w, roi_h, tile_size=ts, overlap_ratio=ov)
        if int(len(tiles)) <= 0 or int(len(tiles)) > int(max_tile_count):
            raise DemoError(
                f"切片几何 tile_size={ts} 产生 {len(tiles)} 片，超出单机上限 {max_tile_count}。",
                status_code=400,
            )

        out_dir = inputs_dir / f"tiles_{ts}_ov{ov_pct}"
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved: List[Dict[str, Any]] = []
        with Image.open(roi_path) as raw_roi:
            roi_im = raw_roi.convert("RGB")
            if roi_im.size != (roi_w, roi_h):
                raise DemoError(
                    f"ROI 图像尺寸 {roi_im.size} 与 upload_meta ({roi_w}, {roi_h}) 不一致。",
                    status_code=500,
                )
            for t in tiles:
                tid = int(t["tile_id"])
                tx1, ty1, tx2, ty2 = [int(v) for v in t["bbox"]]
                tile_path = out_dir / f"tile_{tid:02d}.png"
                roi_im.crop((tx1, ty1, tx2, ty2)).save(tile_path)
                resolved.append({
                    "tile_id": tid,
                    "bbox_roi_xyxy": [tx1, ty1, tx2, ty2],
                    "tile_offset_x": tx1,
                    "tile_offset_y": ty1,
                    "width": int(tx2 - tx1),
                    "height": int(ty2 - ty1),
                    "path": str(tile_path),
                    "sha256": file_sha256(tile_path),
                })

        _write_json_atomic(
            manifest_p,
            {
                "schema_version": "fics_upload_alt_tiling_v1",
                "upload_id": str(upload_id),
                "source_image_sha256": str(meta.get("source_image_sha256") or ""),
                "roi_width": roi_w,
                "roi_height": roi_h,
                "tile_size": ts,
                "overlap_ratio": ov,
                "tile_count": len(resolved),
                "tiles": resolved,
            },
        )
        return resolved

    def _resolve_r9_runtime_adapter(
        self,
        model_key: str,
        preset_spec: Mapping[str, Any],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not preset_spec.get("use_configured_r9_adapter"):
            aid = str(preset_spec.get("adapter_id")).strip() if preset_spec.get("adapter_id") else None
            return None, aid, None
        rt_models = self.runtime_config.get("models") or {}
        rt_mk = rt_models.get(model_key) or {}
        raw_ad = str(rt_mk.get("r9_adapter_dir") or rt_mk.get("adapter_path") or "").strip()
        if not raw_ad:
            raise DemoError(
                f"第 9 轮 QLoRA 适配器预设要求先配置 runtime_config.models.{model_key}.r9_adapter_dir，当前未配置。",
                status_code=400,
            )
        ad_p = pathlib.Path(raw_ad)
        if not ad_p.is_dir() or not (ad_p / "adapter_config.json").is_file():
            raise DemoError(
                f"第 9 轮 QLoRA 适配器目录无效或缺少 adapter_config.json：{ad_p}",
                status_code=400,
            )
        resolved_id = str(
            rt_mk.get("r9_adapter_id")
            or rt_mk.get("adapter_id")
            or ad_p.name
            or preset_spec.get("adapter_id")
            or "r9_qlora_adapter"
        ).strip()
        resolved_sha: Optional[str] = None
        for fname in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
            cand = ad_p / fname
            if cand.is_file():
                resolved_sha = hashlib.sha256(cand.read_bytes()).hexdigest()
                break
        return str(ad_p), resolved_id, resolved_sha

    def _resolve_composite_detector_spec(
        self,
        detector_preset: Optional[str] = None,
        *,
        require_files: bool = True,
    ) -> Dict[str, Any]:
        """Resolve explicit composite YOLO11n weight preset (`fics_yolo11n_r15` default, `wacv_yolo11n`, `fics_yolo11n_r15_ic_cal`)."""
        cfg_default = str((self.runtime_config or {}).get("default_composite_detector_preset") or "fics_yolo11n_r15").strip()
        dp = str(detector_preset or cfg_default or "fics_yolo11n_r15").strip() or "fics_yolo11n_r15"
        if dp not in SUPPORTED_COMPOSITE_DETECTOR_PRESETS:
            raise DemoError(
                f"不支持的复合检测权重预设：{dp!r}；可选值：{', '.join(SUPPORTED_COMPOSITE_DETECTOR_PRESETS)}。",
                status_code=400,
            )
        rt_models = self.runtime_config.get("models") or {}
        if dp == "fics_yolo11n_r15":
            cfg_entry = rt_models.get("yolo11n_fics_r15") or {}
            local_dir = str(cfg_entry.get("local_dir") or DEFAULT_FICS_R15_WEIGHTS_PATH).strip()
            exp_sha = str(cfg_entry.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower()
            ov_dir = str(cfg_entry.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY_DIR)
            if require_files and (not local_dir or not pathlib.Path(local_dir).is_file()):
                raise DemoError(
                    f"未找到默认 FICS 域适配权重 models.yolo11n_fics_r15：{local_dir or '(未配置)'}",
                    status_code=500,
                )
            return {
                "detector_preset": "fics_yolo11n_r15",
                "display_name": "FICS-PCB 6类域适配 YOLO11n（R16 默认 · 统一 0.25 阈值）",
                "expected_class_schema": COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA["fics_yolo11n_r15"],
                "repo_id": "FICS-PCB-6Class-YOLO11n-R15",
                "local_dir": local_dir,
                "weights_sha256": exp_sha,
                "overlay_dir": ov_dir,
                "per_class_conf": None,
            }
        if dp == "fics_yolo11n_r15_ic_cal":
            cfg_entry = rt_models.get("yolo11n_fics_r15_ic_cal") or {}
            fics_entry = rt_models.get("yolo11n_fics_r15") or {}
            local_dir = str(cfg_entry.get("local_dir") or fics_entry.get("local_dir") or DEFAULT_FICS_R15_WEIGHTS_PATH).strip()
            exp_sha = str(cfg_entry.get("weights_sha256") or fics_entry.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower()
            ov_dir = str(cfg_entry.get("overlay_dir") or fics_entry.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY_DIR)
            ic_thr = float(cfg_entry.get("ic_conf_threshold") or 0.45)
            if require_files and (not local_dir or not pathlib.Path(local_dir).is_file()):
                raise DemoError(
                    f"未找到试验校准预设权重：{local_dir or '(未配置)'}",
                    status_code=500,
                )
            return {
                "detector_preset": "fics_yolo11n_r15_ic_cal",
                "display_name": f"FICS R15 试验预设 · 仅 IC 阈值={ic_thr:.2f}（其余 0.25）",
                "expected_class_schema": COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA["fics_yolo11n_r15_ic_cal"],
                "repo_id": "FICS-PCB-6Class-YOLO11n-R15-IC-Cal",
                "local_dir": local_dir,
                "weights_sha256": exp_sha,
                "overlay_dir": ov_dir,
                "per_class_conf": {"ic": ic_thr},
            }
        if dp == "fics_yolo11n_r17_cand":
            cfg_entry = rt_models.get("yolo11n_fics_r17_cand") or {}
            local_dir = str(cfg_entry.get("local_dir") or "").strip()
            exp_sha = str(cfg_entry.get("weights_sha256") or "").strip().lower()
            ov_dir = str(cfg_entry.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY_DIR)
            if require_files and (not local_dir or not pathlib.Path(local_dir).is_file()):
                raise DemoError(
                    f"未找到 R17 弱类扩充候选权重 models.yolo11n_fics_r17_cand：{local_dir or '(未配置)'}",
                    status_code=500,
                )
            return {
                "detector_preset": "fics_yolo11n_r17_cand",
                "display_name": "FICS-PCB 6类域适配候选（R17 弱类扩充 · 隔离候选 · 0.25 阈值）",
                "expected_class_schema": COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA["fics_yolo11n_r17_cand"],
                "repo_id": "FICS-PCB-6Class-YOLO11n-R17-Cand",
                "local_dir": local_dir,
                "weights_sha256": exp_sha,
                "overlay_dir": ov_dir,
                "per_class_conf": None,
            }
        if dp == "fics_yolo11n_r15_crop_verify_r18":
            cfg_entry = rt_models.get("yolo11n_fics_r15_crop_verify_r18") or {}
            fics_entry = rt_models.get("yolo11n_fics_r15") or {}
            local_dir = str(cfg_entry.get("local_dir") or fics_entry.get("local_dir") or DEFAULT_FICS_R15_WEIGHTS_PATH).strip()
            exp_sha = str(cfg_entry.get("weights_sha256") or fics_entry.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower()
            ov_dir = str(cfg_entry.get("overlay_dir") or fics_entry.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY_DIR)
            cv_weights = str(cfg_entry.get("crop_verifier_weights") or "").strip()
            cv_sha = str(cfg_entry.get("crop_verifier_sha256") or "").strip().lower()
            if require_files and (not local_dir or not pathlib.Path(local_dir).is_file()):
                raise DemoError(f"未找到 R18 候选底层 FICS R15 权重：{local_dir or '(未配置)'}", status_code=500)
            if require_files and (not cv_weights or not pathlib.Path(cv_weights).is_file()):
                raise DemoError(f"未找到 R18 局部分类核验权重 crop_verifier_weights：{cv_weights or '(未配置)'}", status_code=500)
            return {
                "detector_preset": "fics_yolo11n_r15_crop_verify_r18",
                "display_name": "FICS R15 + R18 局部分类核验候选（冻结框 · IC/二极管/晶体管核验）",
                "expected_class_schema": COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA["fics_yolo11n_r15_crop_verify_r18"],
                "repo_id": "FICS-PCB-6Class-YOLO11n-R15-CropVerify-R18",
                "local_dir": local_dir,
                "weights_sha256": exp_sha,
                "overlay_dir": ov_dir,
                "per_class_conf": None,
                "crop_verifier_weights": cv_weights,
                "crop_verifier_sha256": cv_sha,
                "t_relabel": float(cfg_entry.get("t_relabel", 0.85)),
                "m_relabel": float(cfg_entry.get("m_relabel", 0.15)),
                "t_reject_other": float(cfg_entry.get("t_reject_other", 0.75)),
                "m_reject_other": float(cfg_entry.get("m_reject_other", 0.25)),
            }
        if dp in ("fics_yolo11n_r15_r19_a1_ic_reject", "fics_yolo11n_r15_r19_a3_combined"):
            cfg_key = "yolo11n_fics_r15_r19_a1_ic_reject" if dp == "fics_yolo11n_r15_r19_a1_ic_reject" else "yolo11n_fics_r15_r19_a3_combined"
            cfg_entry = rt_models.get(cfg_key) or {}
            fics_entry = rt_models.get("yolo11n_fics_r15") or {}
            local_dir = str(cfg_entry.get("local_dir") or fics_entry.get("local_dir") or DEFAULT_FICS_R15_WEIGHTS_PATH).strip()
            exp_sha = str(cfg_entry.get("weights_sha256") or fics_entry.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower()
            ov_dir = str(cfg_entry.get("overlay_dir") or fics_entry.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY_DIR)
            cv_weights = str(cfg_entry.get("crop_verifier_weights") or "").strip()
            cv_sha = str(cfg_entry.get("crop_verifier_sha256") or "").strip().lower()
            if require_files and (not local_dir or not pathlib.Path(local_dir).is_file()):
                raise DemoError(f"未找到 R19 候选底层 FICS R15 权重：{local_dir or '(未配置)'}", status_code=500)
            if require_files and (not cv_weights or not pathlib.Path(cv_weights).is_file()):
                raise DemoError(f"未找到 R19 局部分类核验权重 crop_verifier_weights：{cv_weights or '(未配置)'}", status_code=500)
            spec_out: Dict[str, Any] = {
                "detector_preset": dp,
                "display_name": (
                    "FICS R15 + R19 A1 仅 IC 拒识（冻结框 · 禁用跨类改判）"
                    if dp == "fics_yolo11n_r15_r19_a1_ic_reject"
                    else "FICS R15 + R19 A3 组合（IC 拒识 + 低分 D/Q 补检）"
                ),
                "expected_class_schema": COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA[dp],
                "repo_id": "FICS-PCB-6Class-YOLO11n-R15-R19-A1" if dp == "fics_yolo11n_r15_r19_a1_ic_reject" else "FICS-PCB-6Class-YOLO11n-R15-R19-A3",
                "local_dir": local_dir,
                "weights_sha256": exp_sha,
                "overlay_dir": ov_dir,
                "per_class_conf": None,
                "crop_verifier_weights": cv_weights,
                "crop_verifier_sha256": cv_sha,
                "t_reject_other": float(cfg_entry.get("t_reject_other", 0.75)),
                "m_reject_other": float(cfg_entry.get("m_reject_other", 0.25)),
            }
            if dp == "fics_yolo11n_r15_r19_a3_combined":
                spec_out["a2_rule_spec"] = dict(cfg_entry.get("a2_rule_spec") or {})
            return spec_out
        if dp == "fics_yolo11n_r15_r20_a1_ic_relabel":
            cfg_entry = rt_models.get("yolo11n_fics_r15_r20_a1_ic_relabel") or {}
            fics_entry = rt_models.get("yolo11n_fics_r15") or {}
            local_dir = str(cfg_entry.get("local_dir") or fics_entry.get("local_dir") or DEFAULT_FICS_R15_WEIGHTS_PATH).strip()
            exp_sha = str(cfg_entry.get("weights_sha256") or fics_entry.get("weights_sha256") or DEFAULT_FICS_R15_WEIGHTS_SHA256).strip().lower()
            ov_dir = str(cfg_entry.get("overlay_dir") or fics_entry.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY_DIR)
            cv_weights = str(cfg_entry.get("crop_verifier_weights") or "").strip()
            cv_sha = str(cfg_entry.get("crop_verifier_sha256") or "").strip().lower()
            hd_weights = str(cfg_entry.get("r20_head_weights") or "").strip()
            hd_sha = str(cfg_entry.get("r20_head_sha256") or "").strip().lower()
            imgnet_ckpt = str(cfg_entry.get("imagenet_ckpt_path") or r"C:\Users\Liu\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth").strip()
            if require_files and (not local_dir or not pathlib.Path(local_dir).is_file()):
                raise DemoError(f"未找到 R20 底层 FICS R15 权重：{local_dir or '(未配置)'}", status_code=500)
            if require_files and (not cv_weights or not pathlib.Path(cv_weights).is_file()):
                raise DemoError(f"未找到 R20 A1 权重 crop_verifier_weights：{cv_weights or '(未配置)'}", status_code=500)
            if require_files and (not hd_weights or not pathlib.Path(hd_weights).is_file()):
                raise DemoError(f"未找到 R20 头权重 r20_head_weights：{hd_weights or '(未配置)'}", status_code=500)
            return {
                "detector_preset": dp,
                "display_name": "FICS R15 + R19 A1 + R20 定向 IC 错类修正（冻结框）",
                "expected_class_schema": COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA[dp],
                "repo_id": "FICS-PCB-6Class-YOLO11n-R15-R20-IC-Relabel",
                "local_dir": local_dir,
                "weights_sha256": exp_sha,
                "overlay_dir": ov_dir,
                "per_class_conf": None,
                "crop_verifier_weights": cv_weights,
                "crop_verifier_sha256": cv_sha,
                "t_reject_other": float(cfg_entry.get("t_reject_other", 0.75)),
                "m_reject_other": float(cfg_entry.get("m_reject_other", 0.25)),
                "r20_head_weights": hd_weights,
                "r20_head_sha256": hd_sha,
                "imagenet_ckpt_path": imgnet_ckpt,
                "t_relabel": float(cfg_entry.get("t_relabel", 0.80)),
                "m_relabel": float(cfg_entry.get("m_relabel", 0.25)),
            }
        yolo_rt = rt_models.get("yolo11n") or {}
        w_path = str(yolo_rt.get("local_dir") or DEFAULT_YOLO11N_WEIGHTS_PATH)
        if require_files and not pathlib.Path(w_path).is_file():
            raise DemoError(f"未找到指定本地 YOLO11n 权重文件：{w_path}", status_code=500)
        return {
            "detector_preset": "wacv_yolo11n",
            "display_name": "WACV YOLO11n 旧基线（11类对齐6类 · 对照用）",
            "expected_class_schema": COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA["wacv_yolo11n"],
            "repo_id": "WACV-V3-YOLO11n-R7-best",
            "local_dir": w_path,
            "weights_sha256": str(yolo_rt.get("weights_sha256") or DEFAULT_YOLO11N_WEIGHTS_SHA256).lower(),
            "overlay_dir": str(yolo_rt.get("overlay_dir") or DEFAULT_ULTRALYTICS_OVERLAY_DIR),
            "per_class_conf": None,
        }

    def _build_single_job_plan(
        self,
        *,
        job_id: str,
        meta: Mapping[str, Any],
        model_key: str,
        mode: str,
        preset: str = "baseline",
        detector_preset: Optional[str] = None,
        verifier_only: bool = False,
        instruction_spec: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct a clean single-board execution plan matching frozen repository protocol."""
        is_composite = mode in ("COMPOSITE_YOLO", "COMPOSITE_YOLO_QWEN3")
        is_yolo_exp = mode in ("YOLO_T_LOC", "YOLO_T_QWEN3_CLS") or is_composite
        preset_spec = resolve_model_preset_spec(model_key, "baseline" if is_yolo_exp else preset)
        resolved_adapter_path, resolved_adapter_id, resolved_adapter_sha256 = self._resolve_r9_runtime_adapter(model_key, preset_spec)
        quantization = str(preset_spec.get("quantization") or "bf16").strip().lower()
        base_mode = "COMPOSITE_YOLO" if is_composite else ("YOLO_T_LOC" if is_yolo_exp else ("T" if mode == "T_B_other" else mode))
        tile_size = 640 if is_composite else int(preset_spec.get("tile_size") or 1024)
        tile_overlap_ratio = 0.20 if is_composite else float(preset_spec.get("tile_overlap_ratio", 0.20))
        rt_models = self.runtime_config.get("models") or {}
        need_rt_check = (self._custom_verifier_runner is None) if verifier_only else (
            (self._custom_composite_yolo_runner is None and self._custom_job_runner is None)
            if is_composite
            else (self._custom_job_runner is None)
        )
        comp_det_spec = self._resolve_composite_detector_spec(
            detector_preset if is_composite else "wacv_yolo11n",
            require_files=bool(need_rt_check and is_yolo_exp),
        )
        if need_rt_check:
            if is_yolo_exp:
                if mode in ("YOLO_T_QWEN3_CLS", "COMPOSITE_YOLO_QWEN3") and (
                    ("qwen3" not in rt_models or not rt_models["qwen3"].get("local_dir"))
                    and self._custom_verifier_runner is None
                ):
                    raise DemoError("未在 runtime_config 中配置模型 'qwen3' 的本地权重路径。", status_code=500)
            elif model_key not in rt_models or not rt_models[model_key].get("local_dir"):
                raise DemoError(
                    f"未在 runtime_config 中配置模型 {model_key!r} 的本地权重路径。",
                    status_code=500,
                )

        model_specs: Dict[str, Any] = {}
        for mk, repo_id, attn, ov_used in (
            ("qwen2", "Qwen/Qwen2-VL-2B-Instruct", "sdpa", False),
            ("deepseek", "deepseek-ai/deepseek-vl-1.3b-chat", "native_deepseek_vl", False),
            ("qwen3", "Qwen/Qwen3-VL-2B-Instruct", "sdpa", True),
        ):
            rt = rt_models.get(mk) or {}
            spec: Dict[str, Any] = {
                "repo_id": repo_id,
                "local_dir": str(rt.get("local_dir") or f"/unconfigured/{mk}"),
                "attn_implementation": attn,
                "overlay_used": bool(rt.get("overlay_dir")) if mk == "qwen3" else ov_used,
            }
            if mk == "qwen3":
                q3_min = int(preset_spec.get("qwen_min_pixels", 65536))
                q3_max = int(preset_spec.get("qwen_max_pixels", 200704)) if model_key == "qwen3" else 200704
                spec["revision"] = "89644892e4d85e24eaac8bacfd4f463576704203"
                spec["overlay_dir"] = str(rt.get("overlay_dir") or "")
                spec["require_overlay"] = bool(rt.get("require_overlay", True))
                spec["visual_budget"] = {"shortest_edge": q3_min, "longest_edge": q3_max}
                spec["quantization"] = quantization if model_key == "qwen3" else "bf16"
                spec["adapter_path"] = resolved_adapter_path if model_key == "qwen3" else None
                spec["adapter_id"] = resolved_adapter_id if model_key == "qwen3" else None
            elif mk == "qwen2":
                q2_min = int(preset_spec.get("qwen_min_pixels", 65536))
                q2_max = int(preset_spec.get("qwen_max_pixels", 200704)) if model_key == "qwen2" else 200704
                spec["visual_budget"] = {"min_pixels": q2_min, "max_pixels": q2_max}
            else:
                spec["visual_budget"] = {"native_square_size": 384, "padding_color": [127, 127, 127], "max_pixels_applies": False}
            model_specs[mk] = spec

        model_specs["yolo11n"] = {
            "repo_id": comp_det_spec["repo_id"],
            "detector_preset": comp_det_spec["detector_preset"],
            "detector_display_name": comp_det_spec["display_name"],
            "expected_class_schema": comp_det_spec["expected_class_schema"],
            "local_dir": comp_det_spec["local_dir"],
            "weights_sha256": comp_det_spec["weights_sha256"],
            "overlay_dir": comp_det_spec["overlay_dir"],
            "per_class_conf": comp_det_spec.get("per_class_conf"),
            "wacv_classes_discarded": (mode in ("YOLO_T_LOC", "YOLO_T_QWEN3_CLS")),
        }

        board_id = str(meta["board_id"])
        queue_key = "yolo11n" if is_yolo_exp else model_key
        inst_tgt = str(instruction_spec["target_class"]) if instruction_spec else None
        call_prefix = f"{queue_key}_inst_{job_id[-8:]}_{inst_tgt}" if inst_tgt else f"{queue_key}_demo"
        queue_items: List[Dict[str, Any]] = []
        if base_mode == "W":
            roi = meta["roi_export"]
            planned_tile_count = int(meta["tile_count"])
            queue_items.append({
                "board_id": board_id,
                "tag": "W",
                "mode": "W",
                "image_id": 1,
                "tile_id": None,
                "path": str(roi["path"]),
                "sha256": str(roi["sha256"]),
                "width": int(roi["width"]),
                "height": int(roi["height"]),
                "roi_offset_x": int(meta["roi_offset_x"]),
                "roi_offset_y": int(meta["roi_offset_y"]),
                "tile_offset_x": 0,
                "tile_offset_y": 0,
                "call_index": 0,
                "call_id": f"{call_prefix}_{board_id}_W",
            })
        else:
            resolved_tiles = self._resolve_upload_tiles_for_geometry(
                str(meta["upload_id"]),
                meta,
                tile_size=tile_size,
                overlap_ratio=tile_overlap_ratio,
                max_tile_count=COMPOSITE_MAX_TILE_COUNT if is_composite else MAX_TILE_COUNT,
            )
            planned_tile_count = len(resolved_tiles)
            for idx, tile in enumerate(resolved_tiles):
                tid = int(tile["tile_id"])
                queue_items.append({
                    "board_id": board_id,
                    "tag": f"T{tid:02d}",
                    "mode": base_mode,
                    "image_id": 1,
                    "tile_id": tid,
                    "path": str(tile["path"]),
                    "sha256": str(tile["sha256"]),
                    "width": int(tile["width"]),
                    "height": int(tile["height"]),
                    "roi_offset_x": int(meta["roi_offset_x"]),
                    "roi_offset_y": int(meta["roi_offset_y"]),
                    "tile_offset_x": int(tile["tile_offset_x"]),
                    "tile_offset_y": int(tile["tile_offset_y"]),
                    "call_index": idx,
                    "call_id": f"{call_prefix}_{board_id}_T{tid:02d}",
                })

        active_prompt = str(instruction_spec["normalized_prompt"]) if instruction_spec else str(preset_spec["prompt"])
        active_prompt_sha = str(instruction_spec["prompt_sha256"]) if instruction_spec else str(preset_spec["prompt_sha256"])

        gate_cfg = self.runtime_config.get("gpu_gate") or {}
        plan_doc: Dict[str, Any] = {
            "schema_version": "fics_demo_job_plan_v1",
            "job_id": job_id,
            "created_at": _now_iso(),
            "model_execution_order": [queue_key],
            "models": model_specs,
            "inference_config": {
                "dtype": "bfloat16",
                "quantization": quantization,
                "adapter_path": resolved_adapter_path,
                "adapter_id": resolved_adapter_id,
                "adapter_sha256": resolved_adapter_sha256,
                "batch_size": 1,
                "do_sample": False,
                "explicit_do_sample_false": bool(preset_spec["explicit_do_sample_false"]),
                "repetition_penalty": 1.0,
                "max_new_tokens": 512,
                "qwen_min_pixels": int(preset_spec.get("qwen_min_pixels", 65536)),
                "qwen_max_pixels": int(preset_spec.get("qwen_max_pixels", 200704)),
                "tile_size": tile_size,
                "tile_overlap_ratio": tile_overlap_ratio,
                "deepseek_image_size": 384,
                "coordinate_protocol": str(preset_spec["coordinate_protocol"]),
                "prompt": active_prompt,
                "prompt_sha256": active_prompt_sha,
                "instruction_prompt_variant": (
                    str(instruction_spec.get("prompt_variant"))
                    if instruction_spec and instruction_spec.get("prompt_variant")
                    else str(preset_spec.get("instruction_prompt_variant") or "r8_base")
                ),
                "preset_id": str(preset_spec["preset_id"]),
                "preset_display_name": str(preset_spec["preset_display_name"]),
                "intervention_note": str(preset_spec["intervention_note"]),
            },
            "yolo_operating_point": {
                "imgsz": 640,
                "conf": 0.05,
                "tile_nms_iou": 0.50,
                "tile_agnostic_nms": True,
                "max_det_per_tile": 3000,
                "cross_tile_class_agnostic_nms_iou": 0.50,
                "tie_break_rule": "(-score, tile_id, tile_box_index)",
                "primary_top_k": 200,
                "wacv_classes_discarded": True,
            },
            "parsing_and_eval_policy": {
                "classes": list(FICS_SIX_CLASSES),
                "nms_iou_threshold": 0.50,
            },
            "resource_limits": {
                "gpu_gate_min_smi_free_mib": float(gate_cfg.get("min_free_mib", 5500.0)),
                "gpu_gate_min_torch_free_mib": 6144.0,
                "model_load_timeout_seconds": 180.0,
                "per_call_timeout_seconds": 90.0,
                "per_model_gpu_timeout_seconds": 1800.0 if is_yolo_exp else 2400.0,
                "total_gpu_timeout_seconds": 1800.0 if is_yolo_exp else 3600.0,
            },
            "boards": {
                board_id: {
                    "board_id": board_id,
                    "image_id": 1,
                    "source_image_path": meta["source_image_path"],
                    "orig_wh": meta["orig_wh"],
                    "roi_offset_x": meta["roi_offset_x"],
                    "roi_offset_y": meta["roi_offset_y"],
                    "roi_width": meta["roi_width"],
                    "roi_height": meta["roi_height"],
                    "tile_size": tile_size,
                    "tile_overlap_ratio": tile_overlap_ratio,
                    "tile_count": planned_tile_count,
                }
            },
            "queues": {queue_key: queue_items},
        }
        if queue_key != model_key:
            plan_doc["queues"][model_key] = queue_items
        return plan_doc

    def submit_job(
        self,
        *,
        upload_id: str,
        model_key: str,
        mode: str,
        preset: Optional[str] = None,
        detector_preset: Optional[str] = None,
        source_proposal_job_id: Optional[str] = None,
        user_instruction: Optional[str] = None,
        allow_experimental_spatial: bool = False,
        _from_comparison_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Acquire single-GPU-job lock, persist GT-free plan, and launch background execution."""
        mk, norm_mode = validate_model_and_mode(model_key, mode)
        is_composite_pre = norm_mode in ("COMPOSITE_YOLO", "COMPOSITE_YOLO_QWEN3")
        is_yolo_exp_pre = norm_mode in ("YOLO_T_LOC", "YOLO_T_QWEN3_CLS") or is_composite_pre
        effective_detector_preset: Optional[str] = None
        if is_composite_pre:
            effective_preset = "baseline"
            effective_detector_preset = str(
                detector_preset
                or (self.runtime_config or {}).get("default_composite_detector_preset")
                or "fics_yolo11n_r15"
            ).strip() or "fics_yolo11n_r15"
            if effective_detector_preset not in SUPPORTED_COMPOSITE_DETECTOR_PRESETS:
                raise DemoError(
                    f"不支持的复合检测权重预设：{effective_detector_preset!r}；可选值：{', '.join(SUPPORTED_COMPOSITE_DETECTOR_PRESETS)}。",
                    status_code=400,
                )
        elif is_yolo_exp_pre:
            effective_preset = "baseline"
        elif preset is not None and str(preset).strip():
            effective_preset = str(preset).strip()
        else:
            cfg_def_preset = str(self.runtime_config.get("default_preset") or "").strip()
            cfg_def_model = str(self.runtime_config.get("default_model") or "qwen3").strip().lower()
            if cfg_def_preset and mk == cfg_def_model and norm_mode in ("W", "T") and _from_comparison_id is None:
                effective_preset = cfg_def_preset
            else:
                effective_preset = "baseline"
        preset_spec = resolve_model_preset_spec(mk, effective_preset)

        parsed_inst: Optional[Dict[str, Any]] = None
        if user_instruction is not None:
            if norm_mode not in ("W", "T"):
                raise DemoError(
                    f"自然语言按指令查找与计数仅支持直接 VLM 检测模式 W 或 T，不支持 {norm_mode!r}。",
                    status_code=400,
                )
            if allow_experimental_spatial:
                parsed_inst = parse_spatial_pcb_instruction(
                    user_instruction,
                    coordinate_protocol=str(preset_spec["coordinate_protocol"]),
                )
            else:
                parsed_inst = parse_pcb_language_instruction(
                    user_instruction,
                    coordinate_protocol=str(preset_spec["coordinate_protocol"]),
                    prompt_variant=str(preset_spec.get("instruction_prompt_variant") or "r8_base"),
                )
            if not parsed_inst.get("ok"):
                raise DemoError(
                    str(parsed_inst.get("error_message") or "无法识别自然语言指令。"),
                    status_code=400,
                    details=parsed_inst,
                )
        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
        if meta.get("read_only_playback_only"):
            raise DemoError(
                "当前上传记录为轻量离线交付包内置的只读回放示例（派生预览图仅供展示，不可作为原图重推）；"
                "如需真实推理，请上传本地新图或按文档挂载外置原图与模型权重。",
                status_code=400,
            )

        verified_src_proposal_id: Optional[str] = None
        if source_proposal_job_id:
            if norm_mode != "YOLO_T_QWEN3_CLS":
                raise DemoError("仅阶段 B 组合实验模式 YOLO_T_QWEN3_CLS 支持指定 source_proposal_job_id。", status_code=400)
            src_jdir = self._job_dir(str(source_proposal_job_id))
            src_spec = json.loads((src_jdir / "job_spec.json").read_text(encoding="utf-8"))
            if src_spec.get("upload_id") != upload_id or str(src_spec.get("source_image_sha256") or "").lower() != str(meta["source_image_sha256"]).lower():
                raise DemoError("指定的候选框定位任务不属于当前上传图像，严禁跨图复用候选框。", status_code=400)
            if src_spec.get("mode") != "YOLO_T_LOC":
                raise DemoError(f"指定的 source_proposal_job_id ({source_proposal_job_id}) 不是 YOLO_T_LOC 定位任务。", status_code=400)
            if not (src_jdir / "yolo_proposals" / "proposals_top200.json").is_file():
                raise DemoError(f"指定的定位任务 {source_proposal_job_id} 尚未生成冻结的 proposals_top200.json。", status_code=400)
            verified_src_proposal_id = str(source_proposal_job_id)

        with self._lock:
            if self._active_job_id is not None:
                raise BusyGpuJobError(
                    f"已有 GPU 任务正在运行 ({self._active_job_id})，请等待其完成或点击取消后再发起新任务。",
                    active_job_id=self._active_job_id,
                )
            if self._active_comparison_id is not None and _from_comparison_id != self._active_comparison_id:
                raise BusyGpuJobError(
                    f"已有同图多模型对比队列正在运行 ({self._active_comparison_id})，请等待队列完成或点击取消后再发起新任务。",
                    active_job_id=self._active_comparison_id,
                )
            if _from_comparison_id is not None:
                cmp_ev = self._comparison_cancel_events.get(_from_comparison_id)
                if (
                    _from_comparison_id in self._cancelled_comparisons
                    or cmp_ev is None
                    or cmp_ev.is_set()
                ):
                    raise InterruptedError("同图对比队列已接受取消，拒绝启动后续未派发模型。")
            job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            jdir = self.jobs_dir / job_id
            jdir.mkdir(parents=True, exist_ok=True)
            plan = self._build_single_job_plan(
                job_id=job_id,
                meta=meta,
                model_key=mk,
                mode=norm_mode,
                preset=effective_preset,
                detector_preset=effective_detector_preset,
                instruction_spec=parsed_inst,
            )
            _write_json_atomic(jdir / "plan.json", plan)

            is_composite = norm_mode in ("COMPOSITE_YOLO", "COMPOSITE_YOLO_QWEN3")
            is_yolo_exp = norm_mode in ("YOLO_T_LOC", "YOLO_T_QWEN3_CLS")
            base_mode = "COMPOSITE_YOLO" if is_composite else ("YOLO_T_LOC" if is_yolo_exp else ("T" if norm_mode == "T_B_other" else norm_mode))
            expected_base_calls = len(plan["queues"][mk])
            planned_board = (plan.get("boards") or {}).get(str(meta["board_id"])) or {}
            job_tile_size = int(planned_board.get("tile_size", preset_spec.get("tile_size", 1024)))
            job_tile_overlap = float(planned_board.get("tile_overlap_ratio", preset_spec.get("tile_overlap_ratio", 0.20)))
            job_tile_count = int(planned_board.get("tile_count", meta["tile_count"]))
            job_spec = {
                "job_id": job_id,
                "comparison_id": _from_comparison_id,
                "upload_id": upload_id,
                "board_id": meta["board_id"],
                "safe_filename": meta["safe_filename"],
                "source_image_path": meta["source_image_path"],
                "source_image_sha256": meta["source_image_sha256"],
                "orig_wh": meta["orig_wh"],
                "roi_offset_x": meta["roi_offset_x"],
                "roi_offset_y": meta["roi_offset_y"],
                "roi_width": meta["roi_width"],
                "roi_height": meta["roi_height"],
                "roi_image_path": meta["roi_export"]["path"],
                "tile_size": job_tile_size,
                "tile_overlap_ratio": job_tile_overlap,
                "tile_count": job_tile_count,
                "model_key": mk,
                "model_display_name": MODEL_DISPLAY_NAMES[mk],
                "visual_preprocess_note": _format_visual_preprocess_note(mk, preset_spec),
                "mode": norm_mode,
                "mode_display_name": MODE_DISPLAY_NAMES[norm_mode],
                "base_mode": base_mode,
                "preset": str(preset_spec["preset_id"]),
                "preset_group": str(preset_spec["preset_group"]),
                "preset_display_name": str(preset_spec["preset_display_name"]),
                "detector_preset": (
                    str((plan.get("models") or {}).get("yolo11n", {}).get("detector_preset") or "wacv_yolo11n")
                    if is_composite else None
                ),
                "detector_display_name": (
                    str((plan.get("models") or {}).get("yolo11n", {}).get("detector_display_name") or "")
                    if is_composite else None
                ),
                "expected_class_schema": (
                    str((plan.get("models") or {}).get("yolo11n", {}).get("expected_class_schema") or "")
                    if is_composite else None
                ),
                "yolo_weights_sha256": (
                    str((plan.get("models") or {}).get("yolo11n", {}).get("weights_sha256") or "")
                    if is_composite else None
                ),
                "coordinate_protocol": str(preset_spec["coordinate_protocol"]),
                "quantization": str((plan.get("inference_config") or {}).get("quantization") or preset_spec.get("quantization") or "bf16"),
                "adapter_path": (plan.get("inference_config") or {}).get("adapter_path"),
                "adapter_id": (plan.get("inference_config") or {}).get("adapter_id"),
                "adapter_sha256": (plan.get("inference_config") or {}).get("adapter_sha256"),
                "explicit_do_sample_false": bool(preset_spec["explicit_do_sample_false"]),
                "qwen_min_pixels": int(preset_spec.get("qwen_min_pixels", 65536)),
                "qwen_max_pixels": int(preset_spec.get("qwen_max_pixels", 200704)),
                "prompt_sha256": str(parsed_inst["prompt_sha256"] if parsed_inst else preset_spec["prompt_sha256"]),
                "instruction_spec": parsed_inst,
                "execution_mode_label": "按指令重新推理（独立 GPU 任务）" if parsed_inst else "标准六类检测",
                "verifier_policy": str(preset_spec.get("verifier_policy") or "B_other"),
                "verifier_pad_fraction": float(preset_spec.get("verifier_pad_fraction", 0.10)),
                "verifier_stage2_pad_fraction": (
                    float(preset_spec["verifier_stage2_pad_fraction"])
                    if preset_spec.get("verifier_stage2_pad_fraction") is not None
                    else None
                ),
                "intervention_note": str(preset_spec["intervention_note"]),
                "source_proposal_job_id": verified_src_proposal_id,
                "expected_base_calls": expected_base_calls,
                "expected_base_call_ids": [q["call_id"] for q in plan["queues"][mk]],
                "gt_loaded": False,
                "created_at": _now_iso(),
            }
            _write_json_atomic(jdir / "job_spec.json", job_spec)
            state = {
                "job_id": job_id,
                "status": "queued",
                "stage_label": "任务已入队，等待显存门禁检查",
                "started_at_epoch": time.time(),
                "finished_at_epoch": None,
                "base_calls_planned": expected_base_calls,
                "base_calls_completed": 0,
                "verifier_calls_planned": 0,
                "verifier_calls_completed": 0,
                "technical_complete": None,
                "error_message": None,
                "cancel_requested": False,
            }
            _write_json_atomic(jdir / "job_state.json", state)

            cancel_ev = threading.Event()
            self._cancel_events[job_id] = cancel_ev
            self._active_job_id = job_id

            th = threading.Thread(target=self._run_job_thread, args=(job_id, cancel_ev), daemon=True)
            self._job_threads[job_id] = th
            th.start()
            return self.get_job_status(job_id)

    _FORBIDDEN_CANDIDATE_LEAK_KEYS = frozenset({
        "ground_truth",
        "gt",
        "gt_id",
        "matched_gt_id",
        "tp",
        "fp",
        "fn",
        "is_tp",
        "match_status",
        "eval_status",
        "iou",
        "iou_with_gt",
        "orig_pred_label",
        "true_positive",
        "false_positive",
    })

    def _validate_reuse_stage1_payload(
        self,
        *,
        reuse_stage1: Mapping[str, Any],
        verifier_policy: str,
        source_image_sha256: str,
        norm_preds: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if verifier_policy != "B_other_dual_crop_consensus":
            raise DemoError("仅双裁片一致策略 B_other_dual_crop_consensus 支持复用 Stage-1 10% 核验结果。", status_code=400)
        if not isinstance(reuse_stage1, Mapping):
            raise DemoError("reuse_stage1 必须为 JSON 对象。", status_code=400)
        bad_rk = [k for k in reuse_stage1.keys() if str(k).strip().lower() in self._FORBIDDEN_CANDIDATE_LEAK_KEYS]
        if bad_rk:
            raise DemoError(f"reuse_stage1 包含禁止的真值/评测泄漏字段：{bad_rk}", status_code=400)

        man_p = pathlib.Path(str(reuse_stage1.get("manifest_path") or ""))
        raw_p = pathlib.Path(str(reuse_stage1.get("raw_outputs_path") or ""))
        if not man_p.is_file() or not raw_p.is_file():
            raise DemoError("reuse_stage1 指定的 manifest_path 或 raw_outputs_path 不存在。", status_code=400)

        s1_man = json.loads(man_p.read_text(encoding="utf-8"))
        s1_rows = load_jsonl_records(raw_p)
        crop_spec = s1_man.get("crop_spec") or {}
        if abs(float(crop_spec.get("pad_fraction") or 0.0) - 0.10) > 1e-6 or int(crop_spec.get("min_edge_upscale") or 0) != 224:
            raise DemoError("reuse_stage1 Stage-1 manifest 的裁片规格必须为 pad_fraction=0.10 且 min_edge_upscale=224。", status_code=400)

        inf_spec = s1_man.get("inference_spec") or {}
        exp_prompt_sha = hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest().lower()
        if (
            str(inf_spec.get("model_key") or "").lower() != "qwen3"
            or str(inf_spec.get("prompt_sha256") or "").lower() != exp_prompt_sha
            or int(inf_spec.get("max_new_tokens") or 0) != 32
            or inf_spec.get("do_sample") is not False
        ):
            raise DemoError("reuse_stage1 Stage-1 manifest 推理合同与冻结 Qwen3 核验器不匹配。", status_code=400)

        bimgs = s1_man.get("board_images") or {}
        if not bimgs or not any(str(v.get("sha256") or "").lower() == source_image_sha256.lower() for v in bimgs.values()):
            raise DemoError("reuse_stage1 原图 SHA-256 与当前上传图像不匹配，严禁跨图复用 Stage-1。", status_code=400)

        s1_preds = list(s1_man.get("predictions") or [])
        if len(s1_preds) != len(norm_preds):
            raise DemoError(f"reuse_stage1 候选数 ({len(s1_preds)}) 与输入候选数 ({len(norm_preds)}) 不一致。", status_code=400)
        for sp, npred in zip(s1_preds, norm_preds):
            if (
                str(sp.get("prediction_id")) != str(npred["prediction_id"])
                or int(sp.get("orig_order", -1)) != int(npred["orig_order"])
                or str(sp.get("orig_pred_label", "")).lower() != str(npred["label"]).lower()
                or [round(float(x), 4) for x in (sp.get("orig_pred_bbox_xyxy") or [])] != [round(float(x), 4) for x in npred["bbox_xyxy"]]
            ):
                raise DemoError("reuse_stage1 Stage-1 候选框、类别或顺序与输入候选不完全一致。", status_code=400)

        s1_ucalls = {str(u["call_id"]): u for u in (s1_man.get("unique_calls") or [])}
        if len(s1_rows) != len(s1_ucalls):
            raise DemoError(f"reuse_stage1 原始输出条数 ({len(s1_rows)}) 与唯一裁片数 ({len(s1_ucalls)}) 不一致。", status_code=400)
        seen_cids = set()
        for r in s1_rows:
            cid = str(r.get("call_id") or "")
            if cid not in s1_ucalls or cid in seen_cids:
                raise DemoError(f"reuse_stage1 包含重复或未知 call_id: {cid!r}。", status_code=400)
            seen_cids.add(cid)
            probe = r.get("effective_generation_probe") or {}
            if (
                not r.get("ok")
                or probe.get("effective_generation_mode") != "GenerationMode.GREEDY_SEARCH"
                or probe.get("effective_do_sample") is not False
                or str(r.get("input_sha256") or "").lower() != str(s1_ucalls[cid]["image_sha256"]).lower()
            ):
                raise DemoError(f"reuse_stage1 调用 {cid!r} 未满足 GREEDY_SEARCH / ok=True / SHA 一致校验。", status_code=400)

        return {
            "manifest_path": str(man_p),
            "manifest_sha256": file_sha256(man_p).lower(),
            "raw_outputs_path": str(raw_p),
            "raw_outputs_sha256": file_sha256(raw_p).lower(),
            "unique_calls": len(s1_ucalls),
            "total_candidates": len(s1_preds),
        }

    def submit_verifier_only_job(
        self,
        *,
        upload_id: str,
        source_predictions: Sequence[Mapping[str, Any]],
        preset: str = "r6_qwen3_tile768_b_other_strict",
        source_provenance: Optional[Mapping[str, Any]] = None,
        reuse_stage1: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run isolated Stage-2 Qwen3 crop verification on caller-frozen neutral candidate boxes."""
        preset_spec = resolve_model_preset_spec("qwen3", preset)
        verifier_policy = str(preset_spec.get("verifier_policy") or "B_other_strict_keep_anomaly")
        if source_provenance is not None:
            if not isinstance(source_provenance, Mapping):
                raise DemoError("source_provenance 必须为 JSON 对象。", status_code=400)
            bad_prov = [k for k in source_provenance.keys() if str(k).strip().lower() in self._FORBIDDEN_CANDIDATE_LEAK_KEYS]
            if bad_prov:
                raise DemoError(f"source_provenance 包含禁止的真值/评测泄漏字段：{bad_prov}", status_code=400)

        if not isinstance(source_predictions, (list, tuple)) or not source_predictions:
            raise DemoError("predictions 必须为非空候选框数组。", status_code=400)

        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
        if meta.get("read_only_playback_only"):
            raise DemoError("只读回放样例不可启动 GPU 二次核验任务；请上传真实图像。", status_code=400)

        orig_w, orig_h = int(meta["orig_wh"][0]), int(meta["orig_wh"][1])
        board_id = str(meta["board_id"])
        norm_preds: List[Dict[str, Any]] = []
        seen_pids = set()
        for idx, raw_item in enumerate(source_predictions):
            if not isinstance(raw_item, Mapping):
                raise DemoError(f"第 {idx} 个候选框格式非法，必须为对象。", status_code=400)
            bad_keys = [k for k in raw_item.keys() if str(k).strip().lower() in self._FORBIDDEN_CANDIDATE_LEAK_KEYS]
            if bad_keys:
                raise DemoError(f"候选框 #{idx} 包含禁止的标注/评测字段 {bad_keys}；二次核验入口禁止传真值信息。", status_code=400)
            pid = str(raw_item.get("prediction_id") or f"{board_id}_cand_{idx:03d}").strip()
            if not pid or pid in seen_pids:
                raise DemoError(f"候选框 #{idx} 的 prediction_id={pid!r} 缺失或重复。", status_code=400)
            seen_pids.add(pid)
            lbl = str(raw_item.get("label") or "").strip().lower()
            if lbl not in FICS_SIX_CLASSES:
                raise DemoError(f"候选框 {pid} 类别 {lbl!r} 不在 6 类目标内。", status_code=400)
            box = raw_item.get("bbox_xyxy")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise DemoError(f"候选框 {pid} 缺少合法 4 坐标 bbox_xyxy。", status_code=400)
            try:
                x1, y1, x2, y2 = [float(v) for v in box]
            except Exception as exc:
                raise DemoError(f"候选框 {pid} 坐标非数值：{exc}", status_code=400) from exc
            if x2 <= x1 or y2 <= y1 or x1 < -1e-6 or y1 < -1e-6 or x2 > orig_w + 1e-6 or y2 > orig_h + 1e-6:
                raise DemoError(f"候选框 {pid} 坐标越界或无效：{[x1, y1, x2, y2]}。", status_code=400)
            norm_preds.append({
                "prediction_id": pid,
                "board_id": board_id,
                "image_id": 1,
                "orig_order": int(raw_item.get("orig_order", idx)),
                "label": lbl,
                "bbox_xyxy": [
                    max(0.0, x1),
                    max(0.0, y1),
                    min(float(orig_w), x2),
                    min(float(orig_h), y2),
                ],
                "score": float(raw_item.get("score", 1.0)),
                "tile_id": raw_item.get("tile_id"),
                "call_id": raw_item.get("call_id"),
            })

        verified_reuse_s1 = (
            self._validate_reuse_stage1_payload(
                reuse_stage1=reuse_stage1,
                verifier_policy=verifier_policy,
                source_image_sha256=str(meta["source_image_sha256"]),
                norm_preds=norm_preds,
            )
            if reuse_stage1 is not None
            else None
        )

        with self._lock:
            if self._active_job_id is not None:
                raise BusyGpuJobError(
                    f"已有 GPU 任务正在运行 ({self._active_job_id})，请等待其完成或点击取消后再发起新任务。",
                    active_job_id=self._active_job_id,
                )
            if self._active_comparison_id is not None:
                raise BusyGpuJobError(
                    f"已有同图多模型对比队列正在运行 ({self._active_comparison_id})，请等待队列完成或点击取消后再发起新任务。",
                    active_job_id=self._active_comparison_id,
                )
            job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            jdir = self.jobs_dir / job_id
            jdir.mkdir(parents=True, exist_ok=True)
            plan = self._build_single_job_plan(
                job_id=job_id,
                meta=meta,
                model_key="qwen3",
                mode="W",
                preset=preset,
                verifier_only=True,
            )
            plan["queues"]["qwen3"] = []
            plan["verifier_only"] = True
            plan["verifier_policy"] = verifier_policy
            _write_json_atomic(jdir / "plan.json", plan)
            _write_json_atomic(jdir / "frozen_input_candidates.json", norm_preds)

            planned_board = (plan.get("boards") or {}).get(board_id) or {}
            job_tile_size = int(planned_board.get("tile_size", preset_spec.get("tile_size", 768)))
            job_tile_overlap = float(planned_board.get("tile_overlap_ratio", preset_spec.get("tile_overlap_ratio", 0.20)))
            job_tile_count = int(planned_board.get("tile_count", meta["tile_count"]))
            job_spec = {
                "job_id": job_id,
                "comparison_id": None,
                "upload_id": upload_id,
                "board_id": board_id,
                "safe_filename": meta["safe_filename"],
                "source_image_path": meta["source_image_path"],
                "source_image_sha256": meta["source_image_sha256"],
                "orig_wh": meta["orig_wh"],
                "roi_offset_x": meta["roi_offset_x"],
                "roi_offset_y": meta["roi_offset_y"],
                "roi_width": meta["roi_width"],
                "roi_height": meta["roi_height"],
                "roi_image_path": meta["roi_export"]["path"],
                "tile_size": job_tile_size,
                "tile_overlap_ratio": job_tile_overlap,
                "tile_count": job_tile_count,
                "model_key": "qwen3",
                "model_display_name": MODEL_DISPLAY_NAMES["qwen3"],
                "visual_preprocess_note": _format_visual_preprocess_note("qwen3", preset_spec),
                "mode": "T_B_other",
                "mode_display_name": MODE_DISPLAY_NAMES["T_B_other"] + "（冻结候选二阶段核验 · 基础框来自已保存真实推理）",
                "base_mode": "T",
                "preset": str(preset_spec["preset_id"]),
                "preset_group": str(preset_spec["preset_group"]),
                "preset_display_name": str(preset_spec["preset_display_name"]),
                "coordinate_protocol": str(preset_spec["coordinate_protocol"]),
                "quantization": str((plan.get("inference_config") or {}).get("quantization") or preset_spec.get("quantization") or "bf16"),
                "adapter_path": (plan.get("inference_config") or {}).get("adapter_path"),
                "adapter_id": (plan.get("inference_config") or {}).get("adapter_id"),
                "adapter_sha256": (plan.get("inference_config") or {}).get("adapter_sha256"),
                "explicit_do_sample_false": bool(preset_spec["explicit_do_sample_false"]),
                "qwen_min_pixels": int(preset_spec.get("qwen_min_pixels", 65536)),
                "qwen_max_pixels": int(preset_spec.get("qwen_max_pixels", 589824)),
                "prompt_sha256": str(preset_spec["prompt_sha256"]),
                "verifier_policy": verifier_policy,
                "verifier_pad_fraction": float(preset_spec.get("verifier_pad_fraction", 0.10)),
                "verifier_stage2_pad_fraction": (
                    float(preset_spec["verifier_stage2_pad_fraction"])
                    if preset_spec.get("verifier_stage2_pad_fraction") is not None
                    else None
                ),
                "reuse_stage1": verified_reuse_s1,
                "verifier_only": True,
                "source_provenance": dict(source_provenance) if source_provenance else {},
                "intervention_note": str(preset_spec["intervention_note"]),
                "source_proposal_job_id": None,
                "expected_base_calls": 0,
                "expected_base_call_ids": [],
                "gt_loaded": False,
                "created_at": _now_iso(),
            }
            _write_json_atomic(jdir / "job_spec.json", job_spec)
            state = {
                "job_id": job_id,
                "status": "queued",
                "stage_label": "二阶段候选核验任务已入队，等待裁剪与显存门禁检查",
                "started_at_epoch": time.time(),
                "finished_at_epoch": None,
                "base_calls_planned": 0,
                "base_calls_completed": 0,
                "verifier_calls_planned": len(norm_preds),
                "verifier_calls_completed": 0,
                "technical_complete": None,
                "error_message": None,
                "cancel_requested": False,
            }
            _write_json_atomic(jdir / "job_state.json", state)

            cancel_ev = threading.Event()
            self._cancel_events[job_id] = cancel_ev
            self._active_job_id = job_id

            th = threading.Thread(
                target=self._run_verifier_only_thread,
                args=(job_id, cancel_ev, norm_preds),
                daemon=True,
            )
            self._job_threads[job_id] = th
            th.start()
            return self.get_job_status(job_id)

    def _register_proc(self, job_id: str, proc: Optional[subprocess.Popen[Any]]) -> None:
        with self._lock:
            if proc is None:
                self._active_procs.pop(job_id, None)
            else:
                self._active_procs[job_id] = proc

    def _get_state_lock(self, job_id: str) -> threading.Lock:
        with self._job_state_locks_guard:
            lk = self._job_state_locks.get(job_id)
            if lk is None:
                lk = threading.Lock()
                self._job_state_locks[job_id] = lk
            return lk

    def _update_state(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        """Serialize per-job read-modify-write state updates and enforce terminal-state immutability."""
        jdir = self.jobs_dir / job_id
        state_p = jdir / "job_state.json"
        with self._get_state_lock(job_id):
            state = json.loads(state_p.read_text(encoding="utf-8"))
            cur_status = str(state.get("status") or "")
            if cur_status in TERMINAL_JOB_STATUSES:
                # Never overwrite a finalized terminal status/outcome from a late worker progress or cancel write.
                return state
            if state.get("cancel_requested") and updates.get("cancel_requested") is False:
                updates.pop("cancel_requested", None)
            if state.get("cancel_requested") and updates.get("status") not in TERMINAL_JOB_STATUSES:
                updates.pop("status", None)
                updates.pop("stage_label", None)
            state.update(updates)
            _write_json_atomic(state_p, state)
            return state

    def _run_subprocess_cancellable(
        self,
        *,
        job_id: str,
        cmd: Sequence[str],
        log_path: pathlib.Path,
        cancel_ev: threading.Event,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        popen_kwargs: Dict[str, Any] = {
            "cwd": str(_REPO_ROOT),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(list(cmd), **popen_kwargs)
        self._register_proc(job_id, proc)
        out_lines: List[str] = []

        def _drain() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                out_lines.append(line)

        drain_th = threading.Thread(target=_drain, daemon=True)
        drain_th.start()

        try:
            while proc.poll() is None:
                if cancel_ev.is_set():
                    terminate_process_tree(proc)
                    drain_th.join(timeout=2.0)
                    log_path.write_text("".join(out_lines), encoding="utf-8")
                    raise InterruptedError("用户取消了正在执行的推理子进程。")
                time.sleep(0.2)
            drain_th.join(timeout=2.0)
            log_path.write_text("".join(out_lines), encoding="utf-8")
            return int(proc.returncode or 0)
        finally:
            self._register_proc(job_id, None)

    def _execute_composite_job(
        self,
        *,
        job_id: str,
        job_spec: Mapping[str, Any],
        plan: Mapping[str, Any],
        cancel_ev: threading.Event,
    ) -> None:
        """Execute GT-free composite YOLO11n detection + optional <=48 bounded Qwen3 BF16 crop verification."""
        jdir = self._job_dir(job_id)
        board_id = str(job_spec["board_id"])
        mk = str(job_spec["model_key"])
        norm_mode = str(job_spec["mode"])
        comp_out = jdir / "base_runs" / "composite_yolo"
        comp_out.mkdir(parents=True, exist_ok=True)

        self._update_state(
            job_id,
            status="running_base_inference",
            stage_label=f"正在执行复合检测阶段 1/2：{MODEL_DISPLAY_NAMES[mk]} (640px/20% 切片 · 保留 0.05 候选与 0.25 正式框)",
        )
        if self._custom_composite_yolo_runner is not None:
            yolo_out = self._custom_composite_yolo_runner(job_spec, plan, comp_out, cancel_ev, self)
        else:
            cmd = [
                self.python_executable,
                str(_SCRIPTS_ROOT / "run_composite_yolo_board.py"),
                "--plan", str(jdir / "plan.json"),
                "--output-dir", str(comp_out),
            ]
            rc_y = self._run_subprocess_cancellable(
                job_id=job_id,
                cmd=cmd,
                log_path=jdir / "composite_yolo_runner.log",
                cancel_ev=cancel_ev,
            )
            if rc_y != 0:
                raise RuntimeError(f"复合 YOLO11n 推理子进程退出码 {rc_y}")
            yolo_out = {
                "summary": json.loads((comp_out / "yolo_run_summary.json").read_text(encoding="utf-8")),
                "low_conf_pool_fics6": json.loads((comp_out / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8")),
                "official_predictions_fics6": json.loads((comp_out / "yolo_official_025.json").read_text(encoding="utf-8")),
                "extra_info": json.loads((comp_out / "yolo_raw_non_fics_extra.json").read_text(encoding="utf-8")),
                "raw_records": load_jsonl_records(comp_out / "yolo_raw_outputs_all.jsonl"),
            }

        if cancel_ev.is_set():
            raise InterruptedError("任务已被用户取消。")

        yolo_summary = yolo_out["summary"]
        low_conf_pool = list(yolo_out["low_conf_pool_fics6"])
        official_preds = list(yolo_out["official_predictions_fics6"])
        extra_info = yolo_out.get("extra_info") or (
            json.loads((comp_out / "yolo_raw_non_fics_extra.json").read_text(encoding="utf-8"))
            if (comp_out / "yolo_raw_non_fics_extra.json").is_file()
            else {"counts_005": {}, "counts_025": {}}
        )
        raw_records = list(yolo_out.get("raw_records") or load_jsonl_records(comp_out / "yolo_raw_outputs_all.jsonl"))

        if len(raw_records) != int(job_spec["expected_base_calls"]):
            self._update_state(
                job_id,
                status="incomplete",
                stage_label="复合 YOLO11n 切片推理未完整结束",
                finished_at_epoch=time.time(),
                technical_complete=False,
                error_message=f"已完成切片数 {len(raw_records)} != 计划 {job_spec['expected_base_calls']}",
            )
            return

        yolo_calls = [
            {
                "call_id": r["call_id"],
                "mode": "COMPOSITE_YOLO",
                "tile_id": r.get("tile_id"),
                "width": int(r["width"]),
                "height": int(r["height"]),
                "generated_tokens": None,
                "gpu_generate_ms": r.get("gpu_generate_ms"),
                "cuda_max_reserved_mib": (yolo_summary.get("timing_and_vram") or {}).get("cuda_max_memory_reserved_mib"),
                "status": "success" if int(r.get("parsed_count", 0)) > 0 else "no_objects_detected",
                "parsed_count": int(r.get("parsed_count", 0)),
                "raw_output": str(r.get("raw_output") or "[]"),
            }
            for r in raw_records
        ]
        calls_detail = {
            "termination_v2": {
                "total_calls": len(raw_records),
                "normal_eos": len(raw_records),
                "length_limited_without_eos": 0,
                "syntax_or_format_incomplete_after_eos": 0,
                "strict_schema_ok": len(raw_records),
                "explicit_empty": sum(1 for r in raw_records if int(r.get("parsed_count", 0)) == 0),
            },
            "calls": yolo_calls,
        }
        recon_comp = {
            "board_id": board_id,
            "image_id": 1,
            "mode": norm_mode,
            "nms_applied": True,
            "technical_complete": True,
            "planned_calls": len(raw_records),
            "completed_calls": len(raw_records),
            "raw_global_predictions": int((yolo_summary.get("counts") or {}).get("pre_nms_all", len(low_conf_pool))),
            "post_nms_predictions": len(low_conf_pool),
            "predictions": official_preds,
        }

        if norm_mode == "COMPOSITE_YOLO":
            dp_used = str(job_spec.get("detector_preset") or "fics_yolo11n_r15")
            final_preds = official_preds
            pre_verifier_candidates = None
            r18_verifier_summary = None
            r19_postproc_summary = None
            filter_src = "composite_yolo_direct_025" if dp_used != "fics_yolo11n_r15_ic_cal" else "composite_yolo_ic_calibrated"
            if dp_used == "fics_yolo11n_r15_crop_verify_r18":
                from pcb_vlm.evaluation.crop_verifier_r18 import (
                    apply_r18_verifier_to_predictions,
                    load_r18_verifier_checkpoint,
                )
                import torch
                det_spec = self._resolve_composite_detector_spec(dp_used, require_files=True)
                src_image_path = pathlib.Path(str(job_spec["source_image_path"]))
                pre_verifier_candidates = official_preds
                custom_cv = getattr(self, "_custom_r18_crop_verifier", None)
                if custom_cv is not None:
                    cv_out = custom_cv(image_path=src_image_path, predictions=official_preds, det_spec=det_spec)
                else:
                    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    cv_model, _, cv_sha = load_r18_verifier_checkpoint(
                        pathlib.Path(str(det_spec["crop_verifier_weights"])),
                        expected_sha256=det_spec.get("crop_verifier_sha256") or None,
                        device=dev,
                    )
                    cv_out = apply_r18_verifier_to_predictions(
                        image_path=src_image_path,
                        predictions=official_preds,
                        model=cv_model,
                        device=dev,
                        weights_sha256=cv_sha,
                        t_relabel=float(det_spec["t_relabel"]),
                        m_relabel=float(det_spec["m_relabel"]),
                        t_reject_other=float(det_spec["t_reject_other"]),
                        m_reject_other=float(det_spec["m_reject_other"]),
                    )
                final_preds = list(cv_out["verified_predictions"])
                r18_verifier_summary = dict(cv_out["summary"])
                _write_json_atomic(comp_out / "yolo_r18_verified_025.json", final_preds)
                _write_json_atomic(comp_out / "r18_crop_verifier_decisions.json", list(cv_out["decisions_log"]))
                filter_src = "composite_yolo_r15_plus_r18_crop_verifier"
            elif dp_used in ("fics_yolo11n_r15_r19_a1_ic_reject", "fics_yolo11n_r15_r19_a3_combined"):
                from pcb_vlm.evaluation.crop_verifier_r18 import load_r18_verifier_checkpoint
                from pcb_vlm.evaluation.round19_postproc import (
                    apply_r19_a1_ic_reject_only,
                    apply_r19_four_arms,
                    tag_r19_a0_predictions,
                )
                import torch
                det_spec = self._resolve_composite_detector_spec(dp_used, require_files=True)
                src_image_path = pathlib.Path(str(job_spec["source_image_path"]))
                pre_verifier_candidates = official_preds
                dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                cv_model, _, cv_sha = load_r18_verifier_checkpoint(
                    pathlib.Path(str(det_spec["crop_verifier_weights"])),
                    expected_sha256=det_spec.get("crop_verifier_sha256") or None,
                    device=dev,
                )
                if dp_used == "fics_yolo11n_r15_r19_a1_ic_reject":
                    a1_out = apply_r19_a1_ic_reject_only(
                        official_preds_025=official_preds,
                        image_path=src_image_path,
                        model=cv_model,
                        device=dev,
                        t_reject_other=float(det_spec["t_reject_other"]),
                        m_reject_other=float(det_spec["m_reject_other"]),
                    )
                    final_preds = list(a1_out["predictions"])
                    audit_doc = {
                        "preset": dp_used,
                        "crop_verifier_weights_sha256": cv_sha,
                        "a0_count": len(official_preds),
                        "a1_summary": a1_out["summary"],
                        "a1_dropped_ic_boxes": a1_out["dropped_ic_boxes"],
                    }
                    _write_json_atomic(comp_out / "r19_a0_predictions.json", tag_r19_a0_predictions(official_preds))
                    _write_json_atomic(comp_out / "r19_a1_predictions.json", final_preds)
                    _write_json_atomic(comp_out / "r19_postproc_audit.json", audit_doc)
                    r19_postproc_summary = {"preset": dp_used, "crop_verifier_weights_sha256": cv_sha, **a1_out["summary"]}
                    filter_src = "composite_yolo_r15_plus_r19_a1_ic_reject"
                else:
                    a2_rule = dict(det_spec.get("a2_rule_spec") or {})
                    arms_out = apply_r19_four_arms(
                        official_preds_025=official_preds,
                        low_conf_pool_005=low_conf_pool,
                        a2_rule_spec=a2_rule,
                        image_path=src_image_path,
                        model=cv_model,
                        device=dev,
                        t_reject_other=float(det_spec["t_reject_other"]),
                        m_reject_other=float(det_spec["m_reject_other"]),
                    )
                    final_preds = list(arms_out["a3_predictions"])
                    audit_doc = {
                        "preset": dp_used,
                        "crop_verifier_weights_sha256": cv_sha,
                        "a2_rule_spec": a2_rule,
                        "summary": arms_out["summary"],
                        "a1_dropped_ic_boxes": arms_out["a1_dropped_ic_boxes"],
                        "a2_accepted_low_boxes": arms_out["a2_accepted_low_boxes"],
                        "a2_dedup_suppressed_boxes": arms_out["a2_dedup_suppressed_boxes"],
                    }
                    _write_json_atomic(comp_out / "r19_a0_predictions.json", arms_out["a0_predictions"])
                    _write_json_atomic(comp_out / "r19_a1_predictions.json", arms_out["a1_predictions"])
                    _write_json_atomic(comp_out / "r19_a2_predictions.json", arms_out["a2_predictions"])
                    _write_json_atomic(comp_out / "r19_a3_predictions.json", final_preds)
                    _write_json_atomic(comp_out / "r19_postproc_audit.json", audit_doc)
                    r19_postproc_summary = {"preset": dp_used, "crop_verifier_weights_sha256": cv_sha, "a2_rule_spec": a2_rule, **arms_out["summary"]}
                    filter_src = "composite_yolo_r15_plus_r19_a3_combined"
            elif dp_used == "fics_yolo11n_r15_r20_a1_ic_relabel":
                from pcb_vlm.evaluation.crop_verifier_r18 import load_r18_verifier_checkpoint
                from pcb_vlm.evaluation.round19_postproc import apply_r19_a1_ic_reject_only, tag_r19_a0_predictions
                from pcb_vlm.evaluation.round20_ic_relabel import apply_r20_targeted_ic_relabel, load_r20_head_checkpoint
                import torch
                det_spec = self._resolve_composite_detector_spec(dp_used, require_files=True)
                src_image_path = pathlib.Path(str(job_spec["source_image_path"]))
                pre_verifier_candidates = official_preds
                dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                cv_model, _, cv_sha = load_r18_verifier_checkpoint(
                    pathlib.Path(str(det_spec["crop_verifier_weights"])),
                    expected_sha256=det_spec.get("crop_verifier_sha256") or None,
                    device=dev,
                )
                a1_out = apply_r19_a1_ic_reject_only(
                    official_preds_025=official_preds,
                    image_path=src_image_path,
                    model=cv_model,
                    device=dev,
                    t_reject_other=float(det_spec["t_reject_other"]),
                    m_reject_other=float(det_spec["m_reject_other"]),
                )
                r20_bb, r20_hd, _, r20_sha = load_r20_head_checkpoint(
                    head_weights_path=pathlib.Path(str(det_spec["r20_head_weights"])),
                    imagenet_ckpt_path=pathlib.Path(str(det_spec["imagenet_ckpt_path"])),
                    expected_sha256=det_spec.get("r20_head_sha256") or None,
                    device=dev,
                )
                r20_out = apply_r20_targeted_ic_relabel(
                    a1_predictions=a1_out["predictions"],
                    image_path=src_image_path,
                    backbone=r20_bb,
                    head=r20_hd,
                    device=dev,
                    t_relabel=float(det_spec["t_relabel"]),
                    m_relabel=float(det_spec["m_relabel"]),
                )
                final_preds = list(r20_out["predictions"])
                _write_json_atomic(comp_out / "r20_a0_predictions.json", tag_r19_a0_predictions(official_preds))
                _write_json_atomic(comp_out / "r20_a1_predictions.json", a1_out["predictions"])
                _write_json_atomic(comp_out / "r20_a1_relabel_predictions.json", final_preds)
                _write_json_atomic(comp_out / "r20_relabel_audit.json", {
                    "preset": dp_used,
                    "a1_weights_sha256": cv_sha,
                    "r20_head_sha256": r20_sha,
                    "a1_summary": a1_out["summary"],
                    "r20_summary": r20_out["summary"],
                    "relabeled_boxes": r20_out["relabeled_boxes"],
                    "overlap_abstained": r20_out["overlap_abstained"],
                })
                r19_postproc_summary = {"preset": dp_used, "a1_summary": a1_out["summary"], "r20_summary": r20_out["summary"]}
                filter_src = "composite_yolo_r15_plus_r19_a1_plus_r20_ic_relabel"

            result_payload = self._build_final_result_payload(
                job_spec=job_spec,
                plan=plan,
                recon=recon_comp,
                kept_predictions=final_preds,
                pre_verifier_candidates=pre_verifier_candidates,
                verifier_summary=r18_verifier_summary,
                calls_detail=calls_detail,
            )
            result_payload["score_semantics_note"] = (
                "复合检测基线 (COMPOSITE_YOLO)：保留 YOLO11n 原生置信度 score in [0.25, 1.0] 与对齐 6 类标签，"
                "同步持久化 conf>=0.05 候选池（无 Top-200 截断）；额外类 (led/connector 等) 独立统计，绝不自动并入二极管或六类。"
            )
            result_payload["composite_metadata"] = {
                "execution_mode": "COMPOSITE_YOLO",
                "detector_preset": dp_used,
                "detector_display_name": job_spec.get("detector_display_name"),
                "expected_class_schema": job_spec.get("expected_class_schema") or "fics_6",
                "yolo_weights_sha256": (yolo_summary.get("weights_sha256") or job_spec.get("yolo_weights_sha256")),
                "class_conf_thresholds": (yolo_summary.get("operating_point") or {}).get("class_conf_thresholds") or {c: 0.25 for c in FICS_SIX_CLASSES},
                "qwen3_participated": False,
                "yolo_summary": yolo_summary,
                "low_conf_pool_005_count": len(low_conf_pool),
                "official_025_count": len(final_preds),
                "raw_non_fics_class_counts_025": extra_info.get("counts_025") or {},
                "raw_non_fics_class_counts_005": extra_info.get("counts_005") or {},
                "vlm_bounded_verification": None,
                "r18_crop_verification": r18_verifier_summary,
                "r19_postproc": r19_postproc_summary,
            }
            result_payload["export_provenance"] = {
                "weight_identity": {
                    "detector_preset": dp_used,
                    "weights_sha256": (yolo_summary.get("weights_sha256") or job_spec.get("yolo_weights_sha256")),
                    "expected_class_schema": job_spec.get("expected_class_schema") or "fics_6",
                    "class_conf_thresholds": (yolo_summary.get("operating_point") or {}).get("class_conf_thresholds") or {c: 0.25 for c in FICS_SIX_CLASSES},
                },
                "filter_source": filter_src,
                "qwen3_participated": False,
                "r18_crop_verification": r18_verifier_summary,
                "r19_postproc": r19_postproc_summary,
                "known_limitations": [
                    "GT-free 推理预测计数不等于电路板真实元器件总数。",
                    "FICS R15 在二极管 (diode) 与晶体管 (transistor) 上召回率仍偏弱（受训练集少样本限制），且 IC 类别仍存在一定高分误报。",
                ],
            }
            _write_json_atomic(jdir / "job_result.json", result_payload)
            self._update_state(
                job_id,
                status="completed",
                stage_label=f"复合 YOLO11n 6类检测完成（0.25正式框 {len(official_preds)} 个 / 0.05候选池 {len(low_conf_pool)} 个）",
                finished_at_epoch=time.time(),
                base_calls_completed=len(raw_records),
                technical_complete=True,
            )
            return

        # COMPOSITE_YOLO_QWEN3: bounded <=48 hard-candidate selection + conservative Qwen3 BF16 verification
        sel_res = select_hard_candidates_for_vlm(low_conf_pool, max_candidates=48)
        selected_cands = sel_res["selected_candidates"]
        _write_json_atomic(jdir / "selected_hard_candidates_48.json", sel_res)

        self._update_state(
            job_id,
            status="preparing_verifier",
            stage_label=f"正在裁剪有限难样本候选（{len(selected_cands)} <= 48 个）供 Qwen3 BF16 局部核验",
            base_calls_completed=len(raw_records),
            verifier_calls_planned=len(selected_cands),
        )
        q3_rt = (self.runtime_config.get("models") or {}).get("qwen3") or {}
        verifier_spec = {
            "model_key": "qwen3",
            "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "model_local_dir": str(q3_rt.get("local_dir") or ""),
            "overlay_dir": str(q3_rt.get("overlay_dir") or ""),
            "dtype": "bfloat16",
            "do_sample": False,
            "repetition_penalty": 1.0,
            "max_new_tokens": 32,
            "min_pixels": 65536,
            "max_pixels": 200704,
            "allowed_labels": list(ALLOWED_SEVEN_LABELS),
            "prompt": DEFAULT_VERIFIER_PROMPT,
            "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
        }
        prov = {
            "job_id": job_id,
            "board_id": board_id,
            "model_key": mk,
            "mode": norm_mode,
            "source_image_sha256": job_spec["source_image_sha256"],
        }
        if selected_cands:
            manifest, _ = prepare_verifier_manifest_gt_free(
                reconstructed_by_board=[{
                    "board_id": board_id,
                    "image_id": 1,
                    "mode": norm_mode,
                    "technical_complete": True,
                    "predictions": selected_cands,
                }],
                board_image_paths={board_id: pathlib.Path(job_spec["source_image_path"])},
                crops_dir=jdir / "verifier_crops",
                inference_spec=verifier_spec,
                source_provenance=prov,
                pad_fraction=0.25,
                min_edge_upscale=224,
            )
            manifest_path = jdir / "verifier_manifest.json"
            _write_json_atomic(manifest_path, manifest)
            ucalls = len(manifest.get("unique_calls") or [])
            self._update_state(job_id, verifier_calls_planned=ucalls)
            if ucalls > 0:
                verifier_rows = self._run_verifier_subprocess_or_mock(
                    job_id=job_id,
                    job_spec=job_spec,
                    manifest_path=manifest_path,
                    out_dir=jdir / "verifier_runs",
                    log_path=jdir / "verifier_runner.log",
                    stage_label=f"正在执行复合阶段 2/2：Qwen3 BF16 有限难样本核验 ({ucalls} <= 48 裁片)",
                    cancel_ev=cancel_ev,
                )
            else:
                verifier_rows = []
        else:
            manifest = {"predictions": [], "unique_calls": []}
            verifier_rows = []

        if cancel_ev.is_set():
            raise InterruptedError("任务已被用户取消。")

        v_res = apply_bounded_vlm_verification(
            board_id=board_id,
            low_conf_pool_fics6=low_conf_pool,
            selected_candidates=selected_cands,
            manifest=manifest,
            verifier_rows=verifier_rows,
            conf_official=0.25,
            rescue_low=0.12,
            nms_iou=0.50,
        )
        _write_json_atomic(jdir / "candidate_verification_audit.json", v_res)
        kept_preds = v_res["final_predictions"]

        v_sum = {
            "total_candidates": len(selected_cands),
            "retained_count": len(kept_preds),
            "rejected_as_other": int(v_res["decision_counts"]["rejected_background"]),
            "format_failure_filtered": 0,
            "technical_failure_filtered": 0,
            "total_anomalies_kept": int(v_res["decision_counts"]["vlm_fallback_keep_yolo"]),
            "rescued_low_conf_agree": int(v_res["decision_counts"]["rescued_low_conf_agree"]),
            "class_corrected": int(v_res["decision_counts"]["class_corrected"]),
            "confirmed_yolo_class": int(v_res["decision_counts"]["confirmed_yolo_class"]),
            "decision_counts": v_res["decision_counts"],
        }
        pre_cands_ui = []
        audit_by_pid = {str(a["prediction_id"]): a for a in v_res["audit_records"]}
        for p in official_preds:
            aud = audit_by_pid.get(str(p["prediction_id"]))
            kept_flag = True if aud is None else (aud["final_label"] is not None)
            pre_cands_ui.append({
                **p,
                "verifier_label": aud["vlm_parsed_label"] if aud else None,
                "kept": kept_flag,
                "decision_reason": aud["final_decision"] if aud else "unselected_yolo_kept",
            })

        result_payload = self._build_final_result_payload(
            job_spec=job_spec,
            plan=plan,
            recon=recon_comp,
            kept_predictions=kept_preds,
            pre_verifier_candidates=pre_cands_ui,
            verifier_summary=v_sum,
            calls_detail=calls_detail,
        )
        result_payload["score_semantics_note"] = (
            "复合检测 + Qwen3 有限核验 (COMPOSITE_YOLO_QWEN3)：仅对 <=48 个低分/模糊/冲突候选框调用 Qwen3 BF16 保守核验，"
            "解析失败或异常时保守回退保留 YOLO 原判；每框显式标注 detector_source 与 vlm_intervened。"
        )
        result_payload["composite_metadata"] = {
            "execution_mode": "COMPOSITE_YOLO_QWEN3",
            "detector_preset": job_spec.get("detector_preset") or "wacv_yolo11n",
            "detector_display_name": job_spec.get("detector_display_name"),
            "expected_class_schema": job_spec.get("expected_class_schema") or "wacv_11",
            "yolo_weights_sha256": (yolo_summary.get("weights_sha256") or job_spec.get("yolo_weights_sha256")),
            "yolo_summary": yolo_summary,
            "low_conf_pool_005_count": len(low_conf_pool),
            "official_025_count": len(official_preds),
            "raw_non_fics_class_counts_025": extra_info.get("counts_025") or {},
            "raw_non_fics_class_counts_005": extra_info.get("counts_005") or {},
            "selection": {
                "max_candidates": 48,
                "selected_count": len(selected_cands),
                "reason_counts": sel_res["reason_counts"],
            },
            "vlm_bounded_verification": {
                "verifier_unique_calls": len(verifier_rows),
                "decision_counts": v_res["decision_counts"],
                "audit_records": v_res["audit_records"],
            },
        }
        _write_json_atomic(jdir / "job_result.json", result_payload)
        self._update_state(
            job_id,
            status="completed",
            stage_label=(
                f"复合检测完成（YOLO 0.25={len(official_preds)} -> Qwen3核验 {len(selected_cands)} 候选"
                f" [救回={v_res['decision_counts']['rescued_low_conf_agree']}, 改类={v_res['decision_counts']['class_corrected']}, 拒识={v_res['decision_counts']['rejected_background']}]"
                f" -> 最终 {len(kept_preds)} 框）"
            ),
            finished_at_epoch=time.time(),
            base_calls_completed=len(raw_records),
            verifier_calls_completed=len(verifier_rows),
            technical_complete=True,
        )

    def _run_job_thread(self, job_id: str, cancel_ev: threading.Event) -> None:
        jdir = self.jobs_dir / job_id
        job_spec = json.loads((jdir / "job_spec.json").read_text(encoding="utf-8"))
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        mk = job_spec["model_key"]
        norm_mode = job_spec["mode"]
        base_mode = job_spec["base_mode"]
        board_id = job_spec["board_id"]

        try:
            if cancel_ev.is_set():
                raise InterruptedError("任务在启动前被用户取消。")

            if norm_mode in ("COMPOSITE_YOLO", "COMPOSITE_YOLO_QWEN3"):
                self._execute_composite_job(
                    job_id=job_id,
                    job_spec=job_spec,
                    plan=plan,
                    cancel_ev=cancel_ev,
                )
                return

            if norm_mode in ("YOLO_T_LOC", "YOLO_T_QWEN3_CLS"):
                base_out = jdir / "base_runs"
                prop_dir = jdir / "yolo_proposals"
                src_prop_jid = job_spec.get("source_proposal_job_id")
                if src_prop_jid:
                    src_jdir = self._job_dir(str(src_prop_jid))
                    src_prop_dir = src_jdir / "yolo_proposals"
                    src_hashes = json.loads((src_prop_dir / "proposal_stage_hashes.json").read_text(encoding="utf-8"))
                    src_t200 = src_prop_dir / "proposals_top200.json"
                    actual_t200_sha = file_sha256(src_t200).lower()
                    if actual_t200_sha != str(src_hashes["proposals_top200_sha256"]).lower():
                        raise RuntimeError(f"Frozen proposals_top200.json SHA-256 mismatch in {src_prop_jid}: {actual_t200_sha}")
                    shutil.copytree(src_prop_dir, prop_dir, dirs_exist_ok=True)
                    shutil.copytree(src_jdir / "base_runs" / "run_yolo11n", base_out / "run_yolo11n", dirs_exist_ok=True)
                    if mk != "yolo11n":
                        shutil.copytree(src_jdir / "base_runs" / "run_yolo11n", base_out / f"run_{mk}", dirs_exist_ok=True)
                else:
                    self._update_state(
                        job_id,
                        status="running_base_inference",
                        stage_label=f"正在执行阶段 A 专用候选定位 ({MODEL_DISPLAY_NAMES['yolo11n']} / 1024px 切片)",
                    )
                    y_cmd = [
                        self.python_executable,
                        str(_SCRIPTS_ROOT / "run_yolo11n_proposal.py"),
                        "--plan", str(jdir / "plan.json"),
                        "--output-dir", str(base_out),
                        "--execute",
                    ]
                    rc_y = self._run_subprocess_cancellable(
                        job_id=job_id,
                        cmd=y_cmd,
                        log_path=jdir / "yolo_proposal_runner.log",
                        cancel_ev=cancel_ev,
                    )
                    if rc_y != 0:
                        raise RuntimeError(f"阶段 A YOLO11n 候选定位子进程退出码 {rc_y}")
                    if mk != "yolo11n" and (base_out / "run_yolo11n").is_dir():
                        shutil.copytree(base_out / "run_yolo11n", base_out / f"run_{mk}", dirs_exist_ok=True)

                if cancel_ev.is_set():
                    raise InterruptedError("任务已被用户取消。")

                raw_jsonl = base_out / "run_yolo11n" / "raw_outputs_all.jsonl"
                raw_records = load_jsonl_records(raw_jsonl)
                if len(raw_records) != int(job_spec["expected_base_calls"]):
                    self._update_state(
                        job_id,
                        status="incomplete",
                        stage_label="阶段 A 切片候选定位未完整结束",
                        finished_at_epoch=time.time(),
                        technical_complete=False,
                        error_message=f"已完成切片数 {len(raw_records)} != 计划切片数 {job_spec['expected_base_calls']}",
                    )
                    return

                stage_hashes = json.loads((prop_dir / "proposal_stage_hashes.json").read_text(encoding="utf-8"))
                top200_preds = json.loads((prop_dir / "proposals_top200.json").read_text(encoding="utf-8"))
                yolo_calls = [
                    {
                        "call_id": r["call_id"],
                        "mode": "YOLO_T_LOC",
                        "tile_id": r.get("tile_id"),
                        "width": int(r["width"]),
                        "height": int(r["height"]),
                        "generated_tokens": None,
                        "gpu_generate_ms": r.get("gpu_generate_ms"),
                        "cuda_max_reserved_mib": r.get("cuda_max_reserved_mib"),
                        "status": "success" if int(r.get("parsed_count", 0)) > 0 else "no_objects_detected",
                        "parsed_count": int(r.get("parsed_count", 0)),
                        "raw_output": str(r.get("raw_output") or "[]"),
                    }
                    for r in raw_records
                ]
                calls_detail = {
                    "termination_v2": {
                        "total_calls": len(raw_records),
                        "normal_eos": len(raw_records),
                        "length_limited_without_eos": 0,
                        "syntax_or_format_incomplete_after_eos": 0,
                        "strict_schema_ok": len(raw_records),
                        "explicit_empty": sum(1 for r in raw_records if int(r.get("parsed_count", 0)) == 0),
                    },
                    "calls": yolo_calls,
                }
                recon_yolo = {
                    "board_id": board_id,
                    "image_id": 1,
                    "mode": "YOLO_T_LOC",
                    "nms_applied": True,
                    "technical_complete": True,
                    "planned_calls": len(raw_records),
                    "completed_calls": len(raw_records),
                    "raw_global_predictions": int(stage_hashes["counts"]["pre_nms"]),
                    "post_nms_predictions": int(stage_hashes["counts"]["all_post_nms_conf005"]),
                    "predictions": top200_preds,
                    "proposal_stage_hashes": stage_hashes,
                }

                if norm_mode == "YOLO_T_LOC":
                    result_payload = self._build_final_result_payload(
                        job_spec=job_spec,
                        plan=plan,
                        recon=recon_yolo,
                        kept_predictions=top200_preds,
                        pre_verifier_candidates=None,
                        verifier_summary=None,
                        calls_detail=calls_detail,
                    )
                    result_payload["score_semantics_note"] = (
                        "阶段 A 专用候选定位：输出真实 YOLO 排序置信度 score in [0.05, 1.0]，已完全丢弃 WACV 类别标签 (label='proposal')；"
                        "主操作点保留每板 Top-200 候选框，仅用于几何定位与二阶段裁剪，不直接当作最终 6 类检测 TP。"
                    )
                    result_payload["proposal_stage_hashes"] = stage_hashes
                    _write_json_atomic(jdir / "job_result.json", result_payload)
                    self._update_state(
                        job_id,
                        status="completed",
                        stage_label=f"阶段 A 专用候选定位完成（主操作点保留 {len(top200_preds)} 个候选框）",
                        finished_at_epoch=time.time(),
                        base_calls_completed=len(raw_records),
                        technical_complete=True,
                    )
                    return

                # Stage B: YOLO_T_QWEN3_CLS — classify all frozen Top-200 candidates with Qwen3-VL
                self._update_state(
                    job_id,
                    status="preparing_verifier",
                    stage_label=f"正在按 10% 外扩规则裁剪全部冻结主操作点候选（共 {len(top200_preds)} 个候选框）",
                    base_calls_completed=len(raw_records),
                    verifier_calls_planned=len(top200_preds),
                )
                q3_rt = (self.runtime_config.get("models") or {}).get("qwen3") or {}
                verifier_spec = {
                    "model_key": "qwen3",
                    "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
                    "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
                    "model_local_dir": str(q3_rt.get("local_dir") or ""),
                    "overlay_dir": str(q3_rt.get("overlay_dir") or ""),
                    "dtype": "bfloat16",
                    "do_sample": False,
                    "repetition_penalty": 1.0,
                    "max_new_tokens": 32,
                    "min_pixels": 65536,
                    "max_pixels": 200704,
                    "allowed_labels": list(ALLOWED_SEVEN_LABELS),
                    "prompt": DEFAULT_VERIFIER_PROMPT,
                    "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
                }
                manifest, _ = prepare_verifier_manifest_gt_free(
                    reconstructed_by_board=[recon_yolo],
                    board_image_paths={board_id: pathlib.Path(job_spec["source_image_path"])},
                    crops_dir=jdir / "verifier_crops",
                    inference_spec=verifier_spec,
                    source_provenance={
                        "job_id": job_id,
                        "source_proposal_job_id": src_prop_jid or job_id,
                        "proposals_top200_sha256": stage_hashes["proposals_top200_sha256"],
                        "board_id": board_id,
                        "model_key": mk,
                        "mode": "YOLO_T_QWEN3_CLS",
                        "source_image_sha256": job_spec["source_image_sha256"],
                    },
                )
                manifest_path = jdir / "verifier_manifest.json"
                _write_json_atomic(manifest_path, manifest)

                unique_calls = len(manifest.get("unique_calls") or [])
                self._update_state(job_id, verifier_calls_planned=unique_calls)
                verifier_out_dir = jdir / "verifier_runs"
                if unique_calls > 0:
                    self._update_state(
                        job_id,
                        status="running_verifier",
                        stage_label=f"正在执行阶段 B Qwen3-VL 候选框分类（共 {unique_calls} 个 10% 外扩裁剪）",
                    )
                    if self._custom_verifier_runner is not None:
                        self._custom_verifier_runner(job_spec, manifest_path, cancel_ev, self)
                    elif self._custom_job_runner is None:
                        v_cmd = [
                            self.python_executable,
                            str(_SCRIPTS_ROOT / "run_candidate_verifier.py"),
                            "--manifest", str(manifest_path),
                            "--output-dir", str(verifier_out_dir),
                            "--stage-timeout", "1800",
                            "--per-call-timeout", "90",
                            "--execute",
                        ]
                        rc_v = self._run_subprocess_cancellable(
                            job_id=job_id,
                            cmd=v_cmd,
                            log_path=jdir / "verifier_runner.log",
                            cancel_ev=cancel_ev,
                        )
                        if rc_v != 0:
                            raise RuntimeError(f"阶段 B Qwen3-VL 分类子进程退出码 {rc_v}")
                    v_raw_p = verifier_out_dir / "verifier_raw_outputs.jsonl"
                    if not v_raw_p.is_file():
                        raise RuntimeError("未找到阶段 B 分类原始输出 verifier_raw_outputs.jsonl。")
                    verifier_rows = load_jsonl_records(v_raw_p)
                else:
                    verifier_rows = []

                if cancel_ev.is_set():
                    raise InterruptedError("任务已被用户取消。")

                exported = export_verified_predictions_gt_free(
                    manifest=manifest,
                    verifier_rows=verifier_rows,
                    policy="YOLO_PROPOSAL_ASSIGN_QWEN3_LABEL",
                    output_path=jdir / "yolo_qwen3_classified_predictions_gt_free.json",
                )
                board_exp = exported["per_board"].get(board_id) or {"pred_record": {"objects": []}, "decisions": []}
                kept_objs = [
                    {
                        "prediction_id": f"{board_id}_YOLO_Q3_k{i:03d}",
                        "proposal_id": o.get("prediction_id"),
                        "board_id": board_id,
                        "image_id": 1,
                        "orig_order": i,
                        "label": o["label"],
                        "bbox_xyxy": o["bbox_xyxy"],
                        "score": float(o.get("score", 1.0)),
                        "tile_id": o.get("tile_id"),
                        "call_id": o.get("call_id"),
                        "unique_call_id": o.get("unique_call_id"),
                    }
                    for i, o in enumerate(board_exp["pred_record"]["objects"])
                ]
                manifest_by_pid = {str(p["prediction_id"]): p for p in manifest.get("predictions") or []}
                pre_candidates = [
                    {
                        "prediction_id": d["prediction_id"],
                        "label": d["strict_label"] if (d.get("retained") and d.get("strict_label")) else "proposal",
                        "orig_proposal_label": "proposal",
                        "bbox_xyxy": manifest_by_pid[str(d["prediction_id"])]["orig_pred_bbox_xyxy"],
                        "score": float(manifest_by_pid[str(d["prediction_id"])].get("score", 1.0)),
                        "tile_id": manifest_by_pid[str(d["prediction_id"])].get("tile_id"),
                        "call_id": manifest_by_pid[str(d["prediction_id"])].get("call_id"),
                        "unique_call_id": d.get("unique_call_id"),
                        "crop_sha256": manifest_by_pid[str(d["prediction_id"])].get("crop_sha256"),
                        "verifier_label": d.get("strict_label"),
                        "strict_json_ok": bool(d.get("strict_json_ok")),
                        "kept": bool(d.get("retained")),
                        "decision_reason": d.get("reason"),
                    }
                    for d in board_exp.get("decisions") or []
                ]
                result_payload = self._build_final_result_payload(
                    job_spec=job_spec,
                    plan=plan,
                    recon=recon_yolo,
                    kept_predictions=kept_objs,
                    pre_verifier_candidates=pre_candidates,
                    verifier_summary=exported["summary"],
                    calls_detail=calls_detail,
                )
                result_payload["score_semantics_note"] = (
                    "YOLO 定位 + Qwen3 分类（两阶段组合实验）：框坐标与排序分数 score in [0.05, 1.0] 来自阶段 A YOLO11n 候选定位器（已丢弃 WACV 类别），"
                    "6 类元器件标签来自阶段 B Qwen3-VL 对 10% 外扩裁剪的严格单标签分类（仅过滤 other，格式失败单列）。"
                )
                result_payload["proposal_stage_hashes"] = stage_hashes
                result_payload["classifier_manifest_sha256"] = file_sha256(manifest_path).lower()
                _write_json_atomic(jdir / "job_result.json", result_payload)
                self._update_state(
                    job_id,
                    status="completed",
                    stage_label=f"YOLO 定位 + Qwen3 分类完成（输入 {len(top200_preds)} 候选 -> 保留 {len(kept_objs)} 个 6 类预测框）",
                    finished_at_epoch=time.time(),
                    verifier_calls_completed=len(verifier_rows),
                    technical_complete=True,
                )
                return

            if self._custom_job_runner is not None:
                self._custom_job_runner(job_spec, cancel_ev, self)
            else:
                self._update_state(job_id, status="running_base_inference", stage_label=f"正在执行基线推理 ({MODEL_DISPLAY_NAMES[mk]} / {base_mode})")
                base_out = jdir / "base_runs"
                cmd = [
                    self.python_executable,
                    str(_SCRIPTS_ROOT / "run_fics_matrix.py"),
                    "--plan", str(jdir / "plan.json"),
                    "--output-dir", str(base_out),
                    "--models", mk,
                    "--execute",
                ]
                if self.runtime_config_path and self.runtime_config_path.is_file():
                    cmd.extend(["--runtime-config", str(self.runtime_config_path)])
                rc = self._run_subprocess_cancellable(
                    job_id=job_id,
                    cmd=cmd,
                    log_path=jdir / "base_runner.log",
                    cancel_ev=cancel_ev,
                )
                if rc != 0:
                    gate_p = base_out / f"run_{mk}" / "gpu_gate.json"
                    gate_msg = ""
                    if gate_p.is_file():
                        gdata = json.loads(gate_p.read_text(encoding="utf-8"))
                        if not gdata.get("passed"):
                            gate_msg = f"（显存门禁未通过：{gdata.get('reason') or gdata.get('failure_reason') or 'VRAM 不足'}）"
                    raise RuntimeError(f"基线推理子进程退出码 {rc}{gate_msg}")

            if cancel_ev.is_set():
                raise InterruptedError("任务已被用户取消。")

            # Reconstruct GT-free base predictions
            raw_jsonl = jdir / "base_runs" / f"run_{mk}" / "raw_outputs_all.jsonl"
            if not raw_jsonl.is_file():
                raise RuntimeError("未找到基线原始输出 raw_outputs_all.jsonl。")
            raw_records = load_jsonl_records(raw_jsonl)
            recon = reconstruct_board_predictions_gt_free(
                board_id=board_id,
                image_id=1,
                orig_wh=(int(job_spec["orig_wh"][0]), int(job_spec["orig_wh"][1])),
                raw_records=raw_records,
                mode=base_mode,
                expected_call_ids=job_spec["expected_base_call_ids"],
                eos_token_ids=(100001,) if mk == "deepseek" else (151645, 151643),
                max_new_tokens=512,
                nms_iou_threshold=0.50,
                coordinate_protocol=str(job_spec.get("coordinate_protocol") or "xyxy_norm1000"),
            )
            calls_detail = self._summarize_raw_calls(
                raw_records,
                mk=mk,
                base_mode=base_mode,
                coordinate_protocol=str(job_spec.get("coordinate_protocol") or "xyxy_norm1000"),
            )

            if not recon["technical_complete"]:
                self._update_state(
                    job_id,
                    status="incomplete",
                    stage_label="推理未完整结束，拒绝伪造为空图",
                    finished_at_epoch=time.time(),
                    technical_complete=False,
                    error_message=(
                        f"存在缺失、失败或异常 call_id：missing={recon.get('missing_call_ids')}, "
                        f"failed={recon.get('failed_call_ids')}"
                    ),
                )
                return

            if norm_mode != "T_B_other":
                all_post_nms = list(recon["predictions"])
                inst_spec = job_spec.get("instruction_spec")
                inst_compliance: Optional[Dict[str, Any]] = None
                all_pre_filter: Optional[List[Dict[str, Any]]] = None
                if isinstance(inst_spec, Mapping) and inst_spec.get("target_class"):
                    eos_ids = (100001,) if mk == "deepseek" else (151645, 151643)
                    raw_global_objs: List[Dict[str, Any]] = []
                    for r_raw in raw_records:
                        if str(r_raw.get("mode")) != base_mode:
                            continue
                        rw = int(r_raw.get("orig_input_width") or r_raw.get("width") or 1000)
                        rh = int(r_raw.get("orig_input_height") or r_raw.get("height") or 1000)
                        rp = str(r_raw.get("coordinate_protocol") or job_spec.get("coordinate_protocol") or "xyxy_norm1000")
                        v2_raw = classify_call_termination_and_parse_v2(
                            raw_text=str(r_raw.get("raw_output") if r_raw.get("raw_output") is not None else (r_raw.get("raw_text") or "")),
                            width=rw,
                            height=rh,
                            generated_tokens=r_raw.get("generated_tokens"),
                            max_new_tokens=int(r_raw.get("max_new_tokens") or 512),
                            generated_token_ids=r_raw.get("generated_token_ids"),
                            eos_token_ids=eos_ids,
                            protocol=rp,
                        )
                        raw_global_objs.extend(v2_raw["objects"])
                    audited = audit_single_class_predictions(
                        all_post_nms_preds=all_post_nms,
                        raw_global_predictions=raw_global_objs,
                        raw_records=[r for r in raw_records if str(r.get("mode")) == base_mode],
                        target_class=str(inst_spec["target_class"]),
                    )
                    preds = audited["target_predictions"]
                    all_pre_filter = all_post_nms
                    v2_stat = calls_detail["termination_v2"]
                    spatial_comp: Optional[Dict[str, Any]] = None
                    if inst_spec.get("experimental_spatial"):
                        sp_audit = audit_spatial_predictions(
                            preds,
                            image_width=float(job_spec["roi_width"]),
                            region_mode=str(inst_spec.get("region_mode") or "unconstrained"),
                        )
                        # Preserve ALL same-class boxes (including cross-region violations) and annotate them cleanly.
                        preds = sp_audit["all_predictions_annotated"]
                        spatial_comp = {
                            "region_mode": sp_audit["region_mode"],
                            "region_cn": sp_audit["region_cn"],
                            "image_width": sp_audit["image_width"],
                            "center_split_x": sp_audit["center_split_x"],
                            "raw_same_class_total": sp_audit["raw_same_class_total"],
                            "in_region_count": sp_audit["in_region_count"],
                            "cross_region_violation_count": sp_audit["cross_region_violation_count"],
                            "cross_region_violation_rate": sp_audit["cross_region_violation_rate"],
                            "input_cropped": bool(inst_spec.get("input_cropped", False)),
                            "cross_region_boxes_preserved": bool(inst_spec.get("cross_region_boxes_preserved", True)),
                            "execution_source": str(inst_spec.get("execution_source") or "direct_vlm_spatial"),
                        }
                    inst_compliance = {
                        "target_class": audited["target_class"],
                        "target_class_cn": inst_spec.get("target_class_cn"),
                        "deduplicated_target_count": audited["target_count"],
                        "pre_filter_post_nms_total": audited["pre_filter_post_nms_total"],
                        "unauthorized_post_nms_count": audited["unauthorized_post_nms_count"],
                        "unauthorized_post_nms_rate": audited["unauthorized_post_nms_rate"],
                        "unauthorized_class_breakdown": audited["unauthorized_class_breakdown"],
                        "unauthorized_predictions": audited["unauthorized_predictions"],
                        "raw_global_total": audited["raw_global_total"],
                        "unauthorized_raw_global_count": audited["unauthorized_raw_global_count"],
                        "raw_json_unauthorized_labels": audited["raw_json_unauthorized_labels"],
                        "technical_complete": bool(recon["technical_complete"]),
                        "valid_empty_calls": int(v2_stat.get("explicit_empty", 0)),
                        "truncated_calls": int(v2_stat.get("length_limited_without_eos", 0)),
                        "format_incomplete_calls": int(v2_stat.get("syntax_or_format_incomplete_after_eos", 0)),
                        "normal_eos_calls": int(v2_stat.get("normal_eos", 0)),
                        **({"spatial_compliance": spatial_comp} if spatial_comp is not None else {}),
                    }
                else:
                    preds = all_post_nms

                sum_p = jdir / "base_runs" / f"run_{mk}" / "summary.json"
                if sum_p.is_file():
                    try:
                        sum_obj = json.loads(sum_p.read_text(encoding="utf-8"))
                        wl = sum_obj.get("worker_loaded") or {}
                        if wl.get("adapter_sha256") and not job_spec.get("adapter_sha256"):
                            job_spec = dict(job_spec)
                            job_spec["adapter_sha256"] = wl.get("adapter_sha256")
                    except Exception:
                        pass

                result_payload = self._build_final_result_payload(
                    job_spec=job_spec,
                    plan=plan,
                    recon=recon,
                    kept_predictions=preds,
                    pre_verifier_candidates=None,
                    verifier_summary=None,
                    calls_detail=calls_detail,
                    instruction_compliance=inst_compliance,
                    all_pre_filter_predictions=all_pre_filter,
                )
                _write_json_atomic(jdir / "job_result.json", result_payload)
                self._update_state(
                    job_id,
                    status="completed",
                    stage_label="检测完成",
                    finished_at_epoch=time.time(),
                    base_calls_completed=len(raw_records),
                    technical_complete=True,
                )
                return

            # Stage 2 for T_B_other (qwen3 only)
            self._execute_verifier_stage(
                job_id=job_id,
                job_spec=job_spec,
                plan=plan,
                recon=recon,
                calls_detail=calls_detail,
                cancel_ev=cancel_ev,
                base_calls_completed=len(raw_records),
            )
        except InterruptedError as exc:
            self._update_state(
                job_id,
                status="cancelled",
                stage_label="已安全取消（子进程树已清理，保留当前部分日志）",
                finished_at_epoch=time.time(),
                technical_complete=False,
                error_message=str(exc),
            )
        except Exception as exc:
            self._update_state(
                job_id,
                status="failed",
                stage_label="推理或门禁检查未通过",
                finished_at_epoch=time.time(),
                technical_complete=False,
                error_message=str(exc),
            )
        finally:
            with self._lock:
                self._active_procs.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None

    @staticmethod
    def _summarize_verifier_calls(verifier_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Summarize Stage-2 verifier crop calls into `calls_detail` shape for live UI inspection."""
        calls: List[Dict[str, Any]] = []
        v2_counts = {
            "total_calls": 0,
            "normal_eos": 0,
            "length_limited_without_eos": 0,
            "syntax_or_format_incomplete_after_eos": 0,
            "strict_schema_ok": 0,
            "explicit_empty": 0,
        }
        for r in verifier_rows:
            ok_flag = bool(r.get("ok"))
            has_eos = bool(r.get("has_normal_eos"))
            strict_ok = bool(r.get("strict_json_ok"))
            v2_counts["total_calls"] += 1
            if ok_flag and has_eos:
                v2_counts["normal_eos"] += 1
            elif ok_flag and not has_eos:
                v2_counts["length_limited_without_eos"] += 1
            if ok_flag and has_eos and not strict_ok:
                v2_counts["syntax_or_format_incomplete_after_eos"] += 1
            if strict_ok:
                v2_counts["strict_schema_ok"] += 1
            calls.append({
                "call_id": r["call_id"],
                "mode": "T_B_other",
                "tile_id": None,
                "width": int(r.get("width") or 224),
                "height": int(r.get("height") or 224),
                "coordinate_protocol": "single_label_json",
                "prompt_sha256": r.get("prompt_sha256"),
                "planned_qwen_min_pixels": r.get("planned_qwen_min_pixels") or 65536,
                "planned_qwen_max_pixels": r.get("planned_qwen_max_pixels") or 200704,
                "processed_width_px": r.get("processed_width_px"),
                "processed_height_px": r.get("processed_height_px"),
                "processed_area_px": r.get("processed_area_px"),
                "image_grid_thw": r.get("image_grid_thw"),
                "merged_visual_tokens": r.get("merged_visual_tokens"),
                "effective_generation_probe": r.get("effective_generation_probe"),
                "generated_tokens": r.get("generated_tokens"),
                "gpu_generate_ms": r.get("gpu_generate_ms"),
                "cuda_max_reserved_mib": r.get("cuda_max_reserved_mib"),
                "status": "success" if strict_ok else ("failed" if not ok_flag else "format_anomaly"),
                "call_bucket": "verifier_strict_ok" if strict_ok else ("verifier_truncated" if not has_eos else "verifier_format_anomaly"),
                "normal_eos": has_eos,
                "length_limited_without_eos": bool(ok_flag and not has_eos),
                "strict_schema_ok": strict_ok,
                "explicit_empty": False,
                "syntax_or_format_incomplete_after_eos": bool(ok_flag and has_eos and not strict_ok),
                "parsed_count": 1 if strict_ok else 0,
                "strict_label": r.get("strict_label"),
                "raw_output": str(r.get("raw_output") or ""),
            })
        return {"termination_v2": v2_counts, "calls": calls}

    def _run_verifier_subprocess_or_mock(
        self,
        *,
        job_id: str,
        job_spec: Mapping[str, Any],
        manifest_path: pathlib.Path,
        out_dir: pathlib.Path,
        log_path: pathlib.Path,
        stage_label: str,
        cancel_ev: threading.Event,
    ) -> List[Dict[str, Any]]:
        self._update_state(job_id, status="running_verifier", stage_label=stage_label)
        if self._custom_verifier_runner is not None:
            self._custom_verifier_runner(job_spec, manifest_path, cancel_ev, self)
        else:
            v_cmd = [
                self.python_executable,
                str(_SCRIPTS_ROOT / "run_candidate_verifier.py"),
                "--manifest", str(manifest_path),
                "--output-dir", str(out_dir),
                "--execute",
            ]
            rc_v = self._run_subprocess_cancellable(
                job_id=job_id,
                cmd=v_cmd,
                log_path=log_path,
                cancel_ev=cancel_ev,
            )
            if rc_v != 0:
                raise RuntimeError(f"二次核验子进程退出码 {rc_v}")
        v_raw_p = out_dir / "verifier_raw_outputs.jsonl"
        if not v_raw_p.is_file():
            raise RuntimeError(f"未找到二次核验输出 {v_raw_p}。")
        return load_jsonl_records(v_raw_p)

    def _execute_verifier_stage(
        self,
        *,
        job_id: str,
        job_spec: Mapping[str, Any],
        plan: Mapping[str, Any],
        recon: Mapping[str, Any],
        calls_detail: Optional[Mapping[str, Any]],
        cancel_ev: threading.Event,
        base_calls_completed: int = 0,
    ) -> None:
        """Execute Stage-2 crop verification (single-pass or Round 7 dual-crop 10%+40% consensus)."""
        jdir = self._job_dir(job_id)
        board_id = str(job_spec["board_id"])
        mk = str(job_spec["model_key"])
        cand_preds = list(recon.get("predictions") or [])
        verifier_policy = str(job_spec.get("verifier_policy") or "B_other")
        self._update_state(
            job_id,
            status="preparing_verifier",
            stage_label=f"正在裁剪候选框准备二次核验（共 {len(cand_preds)} 个候选）",
            base_calls_completed=base_calls_completed,
            verifier_calls_planned=len(cand_preds),
        )
        q3_rt = (self.runtime_config.get("models") or {}).get("qwen3") or {}
        verifier_spec = {
            "model_key": "qwen3",
            "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "model_local_dir": str(q3_rt.get("local_dir") or ""),
            "overlay_dir": str(q3_rt.get("overlay_dir") or ""),
            "dtype": "bfloat16",
            "do_sample": False,
            "repetition_penalty": 1.0,
            "max_new_tokens": 32,
            "min_pixels": 65536,
            "max_pixels": 200704,
            "allowed_labels": list(ALLOWED_SEVEN_LABELS),
            "prompt": DEFAULT_VERIFIER_PROMPT,
            "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
        }
        prov = {
            "job_id": job_id,
            "board_id": board_id,
            "model_key": mk,
            "mode": str(job_spec.get("mode") or "T_B_other"),
            "source_image_sha256": job_spec["source_image_sha256"],
        }

        if verifier_policy != "B_other_dual_crop_consensus":
            pad1 = float(job_spec.get("verifier_pad_fraction") or 0.10)
            manifest, _ = prepare_verifier_manifest_gt_free(
                reconstructed_by_board=[recon],
                board_image_paths={board_id: pathlib.Path(job_spec["source_image_path"])},
                crops_dir=jdir / "verifier_crops",
                inference_spec=verifier_spec,
                source_provenance=prov,
                pad_fraction=pad1,
            )
            manifest_path = jdir / "verifier_manifest.json"
            _write_json_atomic(manifest_path, manifest)

            unique_calls = len(manifest.get("unique_calls") or [])
            self._update_state(job_id, verifier_calls_planned=unique_calls)
            verifier_out_dir = jdir / "verifier_runs"
            if unique_calls > 0:
                verifier_rows = self._run_verifier_subprocess_or_mock(
                    job_id=job_id,
                    job_spec=job_spec,
                    manifest_path=manifest_path,
                    out_dir=verifier_out_dir,
                    log_path=jdir / "verifier_runner.log",
                    stage_label=f"正在执行冻结二次核验 (B_other · {unique_calls} 个独立候选裁剪)",
                    cancel_ev=cancel_ev,
                )
            else:
                verifier_rows = []

            if cancel_ev.is_set():
                raise InterruptedError("任务已被用户取消。")

            exported = export_verified_predictions_gt_free(
                manifest=manifest,
                verifier_rows=verifier_rows,
                policy=verifier_policy,
                allow_incomplete=(verifier_policy == "B_other_strict_keep_anomaly"),
                output_path=jdir / "b_other_predictions_gt_free.json",
            )
            board_exp = exported["per_board"].get(board_id) or {"pred_record": {"objects": []}, "decisions": []}
            kept_objs = [
                {
                    "prediction_id": o.get("prediction_id") or f"{board_id}_T_B_other_k{i:03d}",
                    "board_id": board_id,
                    "image_id": 1,
                    "orig_order": o.get("orig_order", i),
                    "label": o["label"],
                    "bbox_xyxy": o["bbox_xyxy"],
                    "score": float(o.get("score", 1.0)),
                    **({"technical_anomaly": True, "anomaly_reason": o.get("anomaly_reason")} if o.get("technical_anomaly") else {}),
                }
                for i, o in enumerate(board_exp["pred_record"]["objects"])
            ]
            manifest_by_pid = {str(p["prediction_id"]): p for p in manifest.get("predictions") or []}
            pre_candidates = [
                {
                    "prediction_id": d["prediction_id"],
                    "label": d["orig_pred_label"],
                    "bbox_xyxy": manifest_by_pid[str(d["prediction_id"])]["orig_pred_bbox_xyxy"],
                    "score": float(manifest_by_pid[str(d["prediction_id"])].get("score", 1.0)),
                    "orig_order": manifest_by_pid[str(d["prediction_id"])].get("orig_order"),
                    "verifier_label": d.get("strict_label"),
                    "kept": bool(d.get("retained")),
                    "decision_reason": d.get("reason"),
                    "technical_anomaly": bool(d.get("technical_anomaly")),
                    "anomaly_reason": d.get("anomaly_reason"),
                }
                for d in board_exp.get("decisions") or []
            ]
            eff_calls_detail = calls_detail if calls_detail is not None else self._summarize_verifier_calls(verifier_rows)
            tech_ok = bool(exported.get("technical_complete"))
            result_payload = self._build_final_result_payload(
                job_spec=job_spec,
                plan=plan,
                recon=recon,
                kept_predictions=kept_objs,
                pre_verifier_candidates=pre_candidates,
                verifier_summary=exported["summary"],
                calls_detail=eff_calls_detail,
                technical_complete=tech_ok,
            )
            _write_json_atomic(jdir / "job_result.json", result_payload)
            if tech_ok:
                self._update_state(
                    job_id,
                    status="completed",
                    stage_label="二阶段背景拒识核验全部完成" if job_spec.get("verifier_only") else "切片推理与 B_other 冻结核验全部完成",
                    finished_at_epoch=time.time(),
                    verifier_calls_completed=len(verifier_rows),
                    technical_complete=True,
                )
            else:
                self._update_state(
                    job_id,
                    status="incomplete",
                    stage_label="二阶段核验存在缺失/失败调用（按异常保留原框但不计为技术通过）",
                    finished_at_epoch=time.time(),
                    verifier_calls_completed=len(verifier_rows),
                    technical_complete=False,
                    error_message=(
                        f"verifier missing={exported['summary'].get('missing_record_count')}, "
                        f"failed={exported['summary'].get('technical_failure_kept_anomaly')}, "
                        f"invalid_crop={exported['summary'].get('invalid_crop_kept_anomaly')}"
                    ),
                )
            return

        # Round 7 Dual-Crop Consensus (Stage-1 10% + Stage-2 40% ONLY on Stage-1 strict "other")
        reuse_cfg = job_spec.get("reuse_stage1")
        stage1_reused = reuse_cfg is not None
        if stage1_reused:
            s1_man = json.loads(pathlib.Path(reuse_cfg["manifest_path"]).read_text(encoding="utf-8"))
            for p in s1_man.get("predictions") or []:
                p["board_id"] = board_id
            s1_rows = load_jsonl_records(pathlib.Path(reuse_cfg["raw_outputs_path"]))
            _write_json_atomic(jdir / "stage1_reused_verifier_manifest.json", s1_man)
            (jdir / "stage1_reused_verifier_raw_outputs.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in s1_rows) + ("\n" if s1_rows else ""),
                encoding="utf-8",
            )
        else:
            pad1 = float(job_spec.get("verifier_pad_fraction") or 0.10)
            s1_man, _ = prepare_verifier_manifest_gt_free(
                reconstructed_by_board=[recon],
                board_image_paths={board_id: pathlib.Path(job_spec["source_image_path"])},
                crops_dir=jdir / "verifier_crops",
                inference_spec=verifier_spec,
                source_provenance={**prov, "stage": "stage1_pad10"},
                pad_fraction=pad1,
            )
            s1_man_p = jdir / "verifier_manifest.json"
            _write_json_atomic(s1_man_p, s1_man)
            s1_ucalls = len(s1_man.get("unique_calls") or [])
            self._update_state(job_id, verifier_calls_planned=s1_ucalls)
            if s1_ucalls > 0:
                s1_rows = self._run_verifier_subprocess_or_mock(
                    job_id=job_id,
                    job_spec=job_spec,
                    manifest_path=s1_man_p,
                    out_dir=jdir / "verifier_runs",
                    log_path=jdir / "verifier_runner.log",
                    stage_label=f"阶段 1/2：正在执行 10% 外扩局部核验（{s1_ucalls} 个唯一裁剪）",
                    cancel_ev=cancel_ev,
                )
            else:
                s1_rows = []

        if cancel_ev.is_set():
            raise InterruptedError("任务已被用户取消。")

        s1_eval = export_verified_predictions_gt_free(
            manifest=s1_man,
            verifier_rows=s1_rows,
            policy="B_other_strict_keep_anomaly",
            allow_incomplete=True,
        )
        s1_board_exp = s1_eval["per_board"].get(board_id) or {"decisions": []}
        s1_other_pids = {
            str(d["prediction_id"])
            for d in (s1_board_exp.get("decisions") or [])
            if d.get("reason") == "rejected_as_other"
        }
        s1_by_pid = {str(p["prediction_id"]): p for p in s1_man.get("predictions") or []}
        stage2_source_preds = [
            {
                "prediction_id": p["prediction_id"],
                "board_id": board_id,
                "image_id": p.get("image_id", 1),
                "orig_order": p.get("orig_order"),
                "label": p["orig_pred_label"],
                "bbox_xyxy": p["orig_pred_bbox_xyxy"],
                "score": float(p.get("score", 1.0)),
                "tile_id": p.get("tile_id"),
                "call_id": p.get("call_id"),
            }
            for p in (s1_man.get("predictions") or [])
            if str(p["prediction_id"]) in s1_other_pids
        ]

        pad2 = float(job_spec.get("verifier_stage2_pad_fraction") or 0.40)
        stage2_dir = jdir / "stage2"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        if stage2_source_preds:
            s2_recon = {
                "board_id": board_id,
                "image_id": 1,
                "mode": "T",
                "technical_complete": True,
                "raw_global_predictions": len(stage2_source_preds),
                "post_nms_predictions": len(stage2_source_preds),
                "predictions": stage2_source_preds,
            }
            s2_man, _ = prepare_verifier_manifest_gt_free(
                reconstructed_by_board=[s2_recon],
                board_image_paths={board_id: pathlib.Path(job_spec["source_image_path"])},
                crops_dir=stage2_dir / "verifier_crops",
                inference_spec=verifier_spec,
                source_provenance={**prov, "stage": "stage2_pad40", "pad_fraction": pad2},
                pad_fraction=pad2,
            )
            s2_man_p = stage2_dir / "verifier_manifest.json"
            _write_json_atomic(s2_man_p, s2_man)
            s2_ucalls = len(s2_man.get("unique_calls") or [])
            total_planned = s2_ucalls if stage1_reused else (len(s1_man.get("unique_calls") or []) + s2_ucalls)
            self._update_state(
                job_id,
                verifier_calls_planned=total_planned,
                stage_label=f"阶段 2/2：准备对 {len(stage2_source_preds)} 个 Stage-1 other 候选做 40% 裁片复判（{s2_ucalls} 个唯一裁片）",
            )
            if s2_ucalls > 0:
                s2_rows = self._run_verifier_subprocess_or_mock(
                    job_id=job_id,
                    job_spec=job_spec,
                    manifest_path=s2_man_p,
                    out_dir=stage2_dir / "verifier_runs",
                    log_path=stage2_dir / "verifier_runner.log",
                    stage_label=f"阶段 2/2：正在执行 40% 宽上下文复判（{s2_ucalls} 个唯一裁片）",
                    cancel_ev=cancel_ev,
                )
            else:
                s2_rows = []
        else:
            s2_man = None
            s2_rows = []

        if cancel_ev.is_set():
            raise InterruptedError("任务已被用户取消。")

        exported = export_dual_crop_consensus_gt_free(
            stage1_manifest=s1_man,
            stage1_verifier_rows=s1_rows,
            stage2_manifest=s2_man,
            stage2_verifier_rows=s2_rows,
            allow_incomplete=True,
            output_path=jdir / "b_other_dual_crop_predictions_gt_free.json",
        )
        _write_json_atomic(jdir / "b_other_predictions_gt_free.json", exported)
        board_exp = exported["per_board"].get(board_id) or {"pred_record": {"objects": []}, "decisions": []}
        kept_objs = [
            {
                "prediction_id": o.get("prediction_id") or f"{board_id}_R7_dual_k{i:03d}",
                "board_id": board_id,
                "image_id": 1,
                "orig_order": o.get("orig_order", i),
                "label": o["label"],
                "bbox_xyxy": o["bbox_xyxy"],
                "score": float(o.get("score", 1.0)),
                "stage1_unique_call_id": o.get("stage1_unique_call_id"),
                "stage2_unique_call_id": o.get("stage2_unique_call_id"),
                "stage1_strict_label": o.get("stage1_strict_label"),
                "stage2_strict_label": o.get("stage2_strict_label"),
                "retention_source": o.get("retention_source"),
                **({"technical_anomaly": True, "anomaly_reason": o.get("anomaly_reason")} if o.get("technical_anomaly") else {}),
            }
            for i, o in enumerate(board_exp["pred_record"]["objects"])
        ]
        pre_candidates = [
            {
                "prediction_id": d["prediction_id"],
                "label": d["orig_pred_label"],
                "bbox_xyxy": s1_by_pid[str(d["prediction_id"])]["orig_pred_bbox_xyxy"],
                "score": float(s1_by_pid[str(d["prediction_id"])].get("score", 1.0)),
                "orig_order": s1_by_pid[str(d["prediction_id"])].get("orig_order"),
                "verifier_label": d.get("strict_label"),
                "stage1_strict_label": d.get("stage1_strict_label"),
                "stage2_strict_label": d.get("stage2_strict_label"),
                "kept": bool(d.get("retained")),
                "decision_reason": d.get("reason"),
                "technical_anomaly": bool(d.get("technical_anomaly")),
                "anomaly_reason": d.get("anomaly_reason"),
            }
            for d in board_exp.get("decisions") or []
        ]
        if calls_detail is not None:
            eff_calls_detail = calls_detail
        elif stage1_reused:
            eff_calls_detail = self._summarize_verifier_calls(s2_rows)
        else:
            combined_rows = [
                {**r, "call_id": f"s1:{r['call_id']}"} for r in s1_rows
            ] + [
                {**r, "call_id": f"s2:{r['call_id']}"} for r in s2_rows
            ]
            eff_calls_detail = self._summarize_verifier_calls(combined_rows)

        tech_ok = bool(exported.get("technical_complete"))
        result_payload = self._build_final_result_payload(
            job_spec=job_spec,
            plan=plan,
            recon=recon,
            kept_predictions=kept_objs,
            pre_verifier_candidates=pre_candidates,
            verifier_summary=exported["summary"],
            calls_detail=eff_calls_detail,
            technical_complete=tech_ok,
        )
        result_payload["dual_crop_provenance"] = {
            "stage1_reused": stage1_reused,
            "stage1_pad_fraction": 0.10,
            "stage2_pad_fraction": pad2,
            "stage1_unique_calls": len(s1_man.get("unique_calls") or []),
            "stage2_unique_calls": len(s2_man.get("unique_calls") or []) if s2_man is not None else 0,
            "stage2_logical_candidates": len(stage2_source_preds),
            "base_boxes_origin": (
                "来自已保存的真实整板/切片推理（caller-frozen candidates）"
                if job_spec.get("verifier_only")
                else "来自本任务实时 768px 切片推理"
            ),
        }
        _write_json_atomic(jdir / "job_result.json", result_payload)
        completed_calls_cnt = len(s2_rows) if stage1_reused else (len(s1_rows) + len(s2_rows))
        if tech_ok:
            self._update_state(
                job_id,
                status="completed",
                stage_label="双裁片一致 (10%+40%) 背景拒识核验完成",
                finished_at_epoch=time.time(),
                verifier_calls_completed=completed_calls_cnt,
                technical_complete=True,
            )
        else:
            self._update_state(
                job_id,
                status="incomplete",
                stage_label="双裁片核验存在缺失/失败/无效裁剪（按异常保留原框但不计为技术通过）",
                finished_at_epoch=time.time(),
                verifier_calls_completed=completed_calls_cnt,
                technical_complete=False,
                error_message=(
                    f"dual_crop missing={exported['summary'].get('missing_record_kept_anomaly')}, "
                    f"failed={exported['summary'].get('technical_failure_kept_anomaly')}, "
                    f"invalid_crop={exported['summary'].get('invalid_crop_kept_anomaly')}"
                ),
            )

    def _run_verifier_only_thread(
        self,
        job_id: str,
        cancel_ev: threading.Event,
        frozen_candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        jdir = self._job_dir(job_id)
        job_spec = json.loads((jdir / "job_spec.json").read_text(encoding="utf-8"))
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        board_id = str(job_spec["board_id"])
        recon = {
            "board_id": board_id,
            "image_id": 1,
            "mode": "T",
            "technical_complete": True,
            "raw_global_predictions": len(frozen_candidates),
            "post_nms_predictions": len(frozen_candidates),
            "predictions": list(frozen_candidates),
        }
        try:
            if cancel_ev.is_set():
                raise InterruptedError("任务在启动前被用户取消。")
            self._execute_verifier_stage(
                job_id=job_id,
                job_spec=job_spec,
                plan=plan,
                recon=recon,
                calls_detail=None,
                cancel_ev=cancel_ev,
                base_calls_completed=0,
            )
        except InterruptedError as exc:
            self._update_state(
                job_id,
                status="cancelled",
                stage_label="已安全取消（子进程树已清理，保留当前部分日志）",
                finished_at_epoch=time.time(),
                technical_complete=False,
                error_message=str(exc),
            )
        except Exception as exc:
            self._update_state(
                job_id,
                status="failed",
                stage_label="二阶段核验或门禁检查未通过",
                finished_at_epoch=time.time(),
                technical_complete=False,
                error_message=str(exc),
            )
        finally:
            with self._lock:
                self._active_procs.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None

    @staticmethod
    def _summarize_raw_calls(
        raw_records: Sequence[Mapping[str, Any]],
        *,
        mk: str,
        base_mode: str,
        coordinate_protocol: str = "xyxy_norm1000",
    ) -> Dict[str, Any]:
        eos_ids = (100001,) if mk == "deepseek" else (151645, 151643)
        calls: List[Dict[str, Any]] = []
        v2_counts = {
            "total_calls": 0,
            "normal_eos": 0,
            "length_limited_without_eos": 0,
            "syntax_or_format_incomplete_after_eos": 0,
            "strict_schema_ok": 0,
            "explicit_empty": 0,
        }
        for r in raw_records:
            if str(r.get("mode")) != base_mode:
                continue
            in_w = int(r.get("orig_input_width") or r.get("width") or 1000)
            in_h = int(r.get("orig_input_height") or r.get("height") or 1000)
            raw_txt = str(r.get("raw_output") if r.get("raw_output") is not None else (r.get("raw_text") or ""))
            rec_proto = str(r.get("coordinate_protocol") or coordinate_protocol or "xyxy_norm1000")
            v2 = classify_call_termination_and_parse_v2(
                raw_text=raw_txt,
                width=in_w,
                height=in_h,
                generated_tokens=r.get("generated_tokens"),
                max_new_tokens=int(r.get("max_new_tokens") or 512),
                generated_token_ids=r.get("generated_token_ids"),
                eos_token_ids=eos_ids,
                protocol=rec_proto,
            )
            v2_counts["total_calls"] += 1
            for k in ("normal_eos", "length_limited_without_eos", "syntax_or_format_incomplete_after_eos", "strict_schema_ok", "explicit_empty"):
                if v2.get(k):
                    v2_counts[k] += 1
            calls.append({
                "call_id": r["call_id"],
                "mode": r.get("mode"),
                "tile_id": r.get("tile_id"),
                "width": in_w,
                "height": in_h,
                "coordinate_protocol": rec_proto,
                "prompt_sha256": r.get("prompt_sha256"),
                "planned_qwen_min_pixels": r.get("planned_qwen_min_pixels"),
                "planned_qwen_max_pixels": r.get("planned_qwen_max_pixels"),
                "processed_width_px": r.get("processed_width_px"),
                "processed_height_px": r.get("processed_height_px"),
                "processed_area_px": r.get("processed_area_px"),
                "image_grid_thw": r.get("image_grid_thw"),
                "merged_visual_tokens": r.get("merged_visual_tokens"),
                "effective_generation_probe": r.get("effective_generation_probe"),
                "generated_tokens": r.get("generated_tokens"),
                "gpu_generate_ms": r.get("gpu_generate_ms"),
                "cuda_max_reserved_mib": r.get("cuda_max_reserved_mib"),
                "status": v2["status"],
                "call_bucket": v2["call_bucket"],
                "normal_eos": bool(v2["normal_eos"]),
                "length_limited_without_eos": bool(v2["length_limited_without_eos"]),
                "strict_schema_ok": bool(v2["strict_schema_ok"]),
                "explicit_empty": bool(v2["explicit_empty"]),
                "syntax_or_format_incomplete_after_eos": bool(v2["syntax_or_format_incomplete_after_eos"]),
                "eos_incomplete_reason": v2.get("eos_incomplete_reason"),
                "parse_audit": v2.get("audit"),
                "parsed_count": len(v2["objects"]),
                "raw_output": raw_txt,
            })
        return {"termination_v2": v2_counts, "calls": calls}

    @staticmethod
    def _build_final_result_payload(
        *,
        job_spec: Mapping[str, Any],
        plan: Mapping[str, Any],
        recon: Mapping[str, Any],
        kept_predictions: Sequence[Mapping[str, Any]],
        pre_verifier_candidates: Optional[Sequence[Mapping[str, Any]]],
        verifier_summary: Optional[Mapping[str, Any]],
        calls_detail: Mapping[str, Any],
        technical_complete: bool = True,
        instruction_compliance: Optional[Mapping[str, Any]] = None,
        all_pre_filter_predictions: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        outcome = _classify_job_outcome(
            "completed" if technical_complete else "incomplete",
            kept_predictions_count=len(kept_predictions),
            termination_v2=calls_detail["termination_v2"],
            technical_complete=technical_complete,
        )
        eff_probes = [
            c["effective_generation_probe"]
            for c in calls_detail["calls"]
            if isinstance(c.get("effective_generation_probe"), dict)
        ]
        vis_tele = [
            {
                "planned_qwen_min_pixels": c.get("planned_qwen_min_pixels") or job_spec.get("qwen_min_pixels", 65536),
                "planned_qwen_max_pixels": c.get("planned_qwen_max_pixels") or job_spec.get("qwen_max_pixels", 200704),
                "processed_width_px": c.get("processed_width_px"),
                "processed_height_px": c.get("processed_height_px"),
                "processed_area_px": c.get("processed_area_px"),
                "image_grid_thw": c.get("image_grid_thw"),
                "merged_visual_tokens": c.get("merged_visual_tokens"),
            }
            for c in calls_detail["calls"]
            if c.get("processed_width_px") is not None or c.get("processed_area_px") is not None
        ]
        return {
            "schema_version": "fics_demo_job_result_v1",
            "job_id": job_spec["job_id"],
            "comparison_id": job_spec.get("comparison_id"),
            "upload_id": job_spec["upload_id"],
            "board_id": job_spec["board_id"],
            "safe_filename": job_spec["safe_filename"],
            "source_image_sha256": job_spec["source_image_sha256"],
            "model_key": job_spec["model_key"],
            "model_display_name": job_spec["model_display_name"],
            "visual_preprocess_note": job_spec.get("visual_preprocess_note") or MODEL_VISUAL_PREPROCESS_NOTES.get(job_spec["model_key"], ""),
            "mode": job_spec["mode"],
            "mode_display_name": job_spec["mode_display_name"],
            "preset": job_spec.get("preset", "baseline"),
            "preset_group": job_spec.get("preset_group", "baseline"),
            "preset_display_name": job_spec.get("preset_display_name", "共同基准配置 (xyxy_norm1000)"),
            "detector_preset": job_spec.get("detector_preset"),
            "detector_display_name": job_spec.get("detector_display_name"),
            "expected_class_schema": job_spec.get("expected_class_schema"),
            "yolo_weights_sha256": job_spec.get("yolo_weights_sha256"),
            "coordinate_protocol": job_spec.get("coordinate_protocol", plan["inference_config"].get("coordinate_protocol", "xyxy_norm1000")),
            "quantization": job_spec.get("quantization") or plan["inference_config"].get("quantization", "bf16"),
            "adapter_path": job_spec.get("adapter_path") or plan["inference_config"].get("adapter_path"),
            "adapter_id": job_spec.get("adapter_id") or plan["inference_config"].get("adapter_id"),
            "adapter_sha256": job_spec.get("adapter_sha256") or plan["inference_config"].get("adapter_sha256"),
            "explicit_do_sample_false": bool(job_spec.get("explicit_do_sample_false", False)),
            "planned_visual_budget": {
                "qwen_min_pixels": int(job_spec.get("qwen_min_pixels", plan["inference_config"].get("qwen_min_pixels", 65536))),
                "qwen_max_pixels": int(job_spec.get("qwen_max_pixels", plan["inference_config"].get("qwen_max_pixels", 200704))),
                "tile_size": int(job_spec.get("tile_size", plan["inference_config"].get("tile_size", 1024))),
                "tile_overlap_ratio": float(job_spec.get("tile_overlap_ratio", plan["inference_config"].get("tile_overlap_ratio", 0.20))),
            },
            "actual_visual_telemetry_summary": vis_tele[0] if vis_tele else None,
            "intervention_note": job_spec.get("intervention_note"),
            "effective_generation_probe_summary": eff_probes[0] if eff_probes else None,
            "score_semantics_note": "score=1.0 仅为离散坐标标签输出的固定占位值，不代表模型概率置信度。",
            "gt_evaluation_note": "模型预测 / 未做精度评价（GT-free）：未提供人工真值时不输出准确率/召回率/F1，预测框数量不等于板上真实元器件数量。",
            "gt_loaded": False,
            "post_hoc_eval": None,
            "orig_wh": job_spec["orig_wh"],
            "roi": {
                "offset_x": job_spec["roi_offset_x"],
                "offset_y": job_spec["roi_offset_y"],
                "width": job_spec["roi_width"],
                "height": job_spec["roi_height"],
                "tile_size": int(job_spec.get("tile_size", plan["inference_config"].get("tile_size", 1024))),
                "tile_overlap_ratio": float(job_spec.get("tile_overlap_ratio", plan["inference_config"].get("tile_overlap_ratio", 0.20))),
                "tile_count": job_spec["tile_count"],
                "image_path": job_spec["roi_image_path"],
            },
            "technical_complete": bool(technical_complete),
            "verifier_only": bool(job_spec.get("verifier_only", False)),
            "verifier_policy": job_spec.get("verifier_policy"),
            "outcome": outcome,
            "raw_global_predictions": recon.get("raw_global_predictions", len(kept_predictions)),
            "post_nms_candidates_count": recon.get("post_nms_predictions", len(kept_predictions)),
            "kept_predictions_count": len(kept_predictions),
            "class_counts": _count_classes(kept_predictions),
            "class_cn_names": dict(CLASS_CN_NAMES),
            "pre_verifier_class_counts": _count_classes(pre_verifier_candidates) if pre_verifier_candidates is not None else None,
            "predictions": list(kept_predictions),
            "pre_verifier_candidates": list(pre_verifier_candidates) if pre_verifier_candidates is not None else None,
            "verifier_summary": dict(verifier_summary) if verifier_summary is not None else None,
            "termination_v2": calls_detail["termination_v2"],
            "calls": calls_detail["calls"],
            "inference_config_sha256": {
                "prompt_sha256": plan["inference_config"]["prompt_sha256"],
            },
            **(
                {
                    "instruction_spec": dict(job_spec["instruction_spec"]),
                    "instruction_compliance": dict(instruction_compliance or {}),
                    "all_pre_filter_predictions": list(all_pre_filter_predictions or kept_predictions),
                    **(
                        {"spatial_compliance": dict(instruction_compliance["spatial_compliance"])}
                        if isinstance(instruction_compliance, Mapping)
                        and isinstance(instruction_compliance.get("spatial_compliance"), Mapping)
                        else {}
                    ),
                }
                if job_spec.get("instruction_spec")
                else {}
            ),
            "completed_at": _now_iso(),
        }

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Request safe cancellation for an active job and kill its subprocess tree."""
        self._job_dir(job_id)
        with self._lock:
            ev = self._cancel_events.get(job_id)
            proc = self._active_procs.get(job_id)
        if ev is not None:
            self._update_state(job_id, cancel_requested=True, stage_label="正在安全终止推理子进程...")
            ev.set()
        if proc is not None:
            terminate_process_tree(proc)
        return self.get_job_status(job_id)

    def list_jobs(self, limit: int = 30, upload_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List persisted jobs newest-first without failing on partial directories."""
        items: List[Dict[str, Any]] = []
        if not self.jobs_dir.is_dir():
            return items
        for d in sorted(self.jobs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not d.is_dir() or not (d / "job_state.json").is_file():
                continue
            try:
                st = self.get_job_status(d.name)
                if upload_id and st.get("upload_id") != upload_id:
                    continue
                items.append(st)
            except Exception:
                continue
            if len(items) >= int(limit):
                break
        return items

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        jdir = self._job_dir(job_id)
        with self._get_state_lock(job_id):
            spec = json.loads((jdir / "job_spec.json").read_text(encoding="utf-8"))
            state = json.loads((jdir / "job_state.json").read_text(encoding="utf-8"))
        mk = spec["model_key"]

        # Read real progress from jsonl files while active
        base_raw_p = jdir / "base_runs" / f"run_{mk}" / "raw_outputs_all.jsonl"
        if not base_raw_p.is_file() and (jdir / "base_runs" / "composite_yolo" / "yolo_raw_outputs_all.jsonl").is_file():
            base_raw_p = jdir / "base_runs" / "composite_yolo" / "yolo_raw_outputs_all.jsonl"
        elif not base_raw_p.is_file() and (jdir / "base_runs" / "run_yolo11n" / "raw_outputs_all.jsonl").is_file():
            base_raw_p = jdir / "base_runs" / "run_yolo11n" / "raw_outputs_all.jsonl"
        ver_raw_p = jdir / "verifier_runs" / "verifier_raw_outputs.jsonl"
        s2_ver_raw_p = jdir / "stage2" / "verifier_runs" / "verifier_raw_outputs.jsonl"
        stage1_reused = bool(spec.get("reuse_stage1"))
        s1_reused_calls = 0
        s1_reused_manifest_p = jdir / "stage1_reused_verifier_manifest.json"
        if stage1_reused and s1_reused_manifest_p.is_file():
            try:
                s1_reused_calls = len(json.loads(s1_reused_manifest_p.read_text(encoding="utf-8")).get("unique_calls") or [])
            except Exception:
                s1_reused_calls = 0
        base_done = len(load_jsonl_records(base_raw_p)) if base_raw_p.is_file() else int(state.get("base_calls_completed", 0))
        s1_new_done = 0 if stage1_reused else (len(load_jsonl_records(ver_raw_p)) if ver_raw_p.is_file() else 0)
        s2_new_done = len(load_jsonl_records(s2_ver_raw_p)) if s2_ver_raw_p.is_file() else 0
        if s1_new_done + s2_new_done > 0:
            ver_done = s1_new_done + s2_new_done
        else:
            ver_done = int(state.get("verifier_calls_completed", 0))

        base_plan = int(state.get("base_calls_planned") or spec["expected_base_calls"] or 0)
        ver_plan = int(state.get("verifier_calls_planned") or 0)
        if ver_plan == 0:
            s1_plan_u = 0
            s2_plan_u = 0
            if not stage1_reused and (jdir / "verifier_manifest.json").is_file():
                try:
                    s1_plan_u = len(json.loads((jdir / "verifier_manifest.json").read_text(encoding="utf-8")).get("unique_calls") or [])
                except Exception:
                    s1_plan_u = 0
            if (jdir / "stage2" / "verifier_manifest.json").is_file():
                try:
                    s2_plan_u = len(json.loads((jdir / "stage2" / "verifier_manifest.json").read_text(encoding="utf-8")).get("unique_calls") or [])
                except Exception:
                    s2_plan_u = 0
            ver_plan = s1_plan_u + s2_plan_u

        end_t = state.get("finished_at_epoch") or time.time()
        elapsed_s = round(max(0.0, float(end_t) - float(state["started_at_epoch"])), 1)
        total_plan = base_plan + ver_plan
        total_done = min(total_plan, base_done + ver_done)
        eta_s: Optional[float] = None
        if state["status"] not in ("completed", "failed", "cancelled", "incomplete") and total_done > 0 and total_plan > total_done:
            eta_s = round((elapsed_s / total_done) * (total_plan - total_done), 1)

        gate_p = jdir / "base_runs" / f"run_{mk}" / "gpu_gate.json"
        if not gate_p.is_file() and (jdir / "base_runs" / "composite_yolo" / "gpu_gate.json").is_file():
            gate_p = jdir / "base_runs" / "composite_yolo" / "gpu_gate.json"
        elif not gate_p.is_file() and (jdir / "base_runs" / "run_yolo11n" / "gpu_gate.json").is_file():
            gate_p = jdir / "base_runs" / "run_yolo11n" / "gpu_gate.json"
        gate_info = json.loads(gate_p.read_text(encoding="utf-8")) if gate_p.is_file() else None
        res_p = jdir / "job_result.json"
        has_result = res_p.is_file()
        kept_cnt: Optional[int] = None
        tv2: Optional[Dict[str, Any]] = None
        cls_counts: Optional[Dict[str, int]] = None
        if has_result:
            try:
                res_obj = json.loads(res_p.read_text(encoding="utf-8"))
                kept_cnt = int(res_obj.get("kept_predictions_count", 0))
                tv2 = res_obj.get("termination_v2")
                cls_counts = res_obj.get("class_counts")
            except Exception:
                pass
        outcome = _classify_job_outcome(
            state["status"],
            kept_predictions_count=kept_cnt,
            termination_v2=tv2,
            technical_complete=state.get("technical_complete"),
            error_message=state.get("error_message"),
        )

        return {
            "job_id": job_id,
            "comparison_id": spec.get("comparison_id"),
            "upload_id": spec["upload_id"],
            "board_id": spec["board_id"],
            "safe_filename": spec.get("safe_filename"),
            "source_image_sha256": spec.get("source_image_sha256"),
            "model_key": mk,
            "model_display_name": spec["model_display_name"],
            "visual_preprocess_note": spec.get("visual_preprocess_note") or MODEL_VISUAL_PREPROCESS_NOTES.get(mk, ""),
            "mode": spec["mode"],
            "mode_display_name": spec["mode_display_name"],
            "preset": spec.get("preset", "baseline"),
            "preset_group": spec.get("preset_group", "baseline"),
            "preset_display_name": spec.get("preset_display_name", "共同基准配置 (xyxy_norm1000)"),
            "coordinate_protocol": spec.get("coordinate_protocol", "xyxy_norm1000"),
            "quantization": spec.get("quantization", "bf16"),
            "adapter_id": spec.get("adapter_id"),
            "explicit_do_sample_false": bool(spec.get("explicit_do_sample_false", False)),
            "qwen_min_pixels": int(spec.get("qwen_min_pixels", 65536)),
            "qwen_max_pixels": int(spec.get("qwen_max_pixels", 200704)),
            "tile_size": int(spec.get("tile_size", 1024)),
            "tile_overlap_ratio": float(spec.get("tile_overlap_ratio", 0.20)),
            "tile_count": int(spec.get("tile_count", 0)),
            "status": state["status"],
            "stage_label": state["stage_label"],
            "outcome": outcome,
            "kept_predictions_count": kept_cnt,
            "class_counts": cls_counts,
            "cancel_requested": bool(state.get("cancel_requested")),
            "technical_complete": state.get("technical_complete"),
            "error_message": state.get("error_message"),
            "elapsed_seconds": elapsed_s,
            "eta_seconds": eta_s,
            "base_calls_planned": base_plan,
            "base_calls_completed": base_done,
            "verifier_calls_planned": ver_plan,
            "verifier_calls_completed": ver_done,
            "stage1_reused_calls": s1_reused_calls,
            "stage1_new_calls_completed": s1_new_done,
            "stage2_new_calls_completed": s2_new_done,
            "instruction_spec": spec.get("instruction_spec"),
            "gpu_gate": gate_info,
            "has_result": has_result,
        }

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        jdir = self._job_dir(job_id)
        res_p = jdir / "job_result.json"
        if not res_p.is_file():
            st = self.get_job_status(job_id)
            raise DemoError(
                f"任务 {job_id} 尚未生成最终结果（当前状态：{st['status']}）。",
                status_code=409 if st["status"] in ("queued", "running_base_inference", "preparing_verifier", "running_verifier") else 404,
            )
        return json.loads(res_p.read_text(encoding="utf-8"))

    def evaluate_job_with_coco(
        self,
        *,
        job_id: str,
        coco_gt: Dict[str, Any],
        target_image_id: Optional[int] = None,
        confirm_manual_mapping: bool = False,
        manual_mapping_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Strictly validate user-supplied COCO GT identity & dimensions for post-hoc evaluation."""
        jdir = self._job_dir(job_id)
        res = self.get_job_result(job_id)
        orig_w, orig_h = int(res["orig_wh"][0]), int(res["orig_wh"][1])
        safe_fname = str(res["safe_filename"])
        board_stem = pathlib.Path(safe_fname).stem.lower()
        uploaded_sha = str(res["source_image_sha256"]).strip().lower()

        try:
            validate_coco_schema_and_categories(coco_gt)
        except Exception as exc:
            raise DemoError(f"COCO GT 格式或 6 类类别定义校验失败：{exc}", status_code=400) from exc

        images = list(coco_gt.get("images") or [])
        if not images:
            raise DemoError("COCO GT 未包含任何 images 条目，无法开展事后对比。", status_code=400)

        chosen_img: Optional[Dict[str, Any]] = None
        if target_image_id is not None:
            matches = [im for im in images if int(im["id"]) == int(target_image_id)]
            if not matches:
                raise DemoError(f"COCO GT 中不存在指定 image_id={target_image_id}。", status_code=400)
            chosen_img = matches[0]
        elif len(images) == 1:
            chosen_img = images[0]
        else:
            sha_matches = [
                im for im in images
                if any(
                    str(im.get(k) or "").strip().lower() == uploaded_sha
                    for k in ("sha256", "image_sha256", "source_image_sha256", "file_sha256")
                )
            ]
            if len(sha_matches) == 1:
                chosen_img = sha_matches[0]
            else:
                stem_matches = [
                    im for im in images
                    if (
                        str(im.get("file_name") or "").strip()
                        and (
                            pathlib.Path(str(im.get("file_name") or "")).name.lower() == safe_fname.lower()
                            or pathlib.Path(str(im.get("file_name") or "")).stem.lower() == board_stem
                        )
                    )
                    or (str(im.get("board_id") or "").strip().lower() == board_stem and bool(board_stem))
                ]
                if len(stem_matches) == 1:
                    chosen_img = stem_matches[0]
                else:
                    avail = [
                        {"id": int(im["id"]), "file_name": im.get("file_name"), "width": im.get("width"), "height": im.get("height")}
                        for im in images
                    ]
                    raise DemoError(
                        "多图 COCO GT 未能唯一匹配上传板卡，请在提交时明确选择目标 image_id。",
                        status_code=400,
                        details={"available_images": avail},
                    )

        gt_w, gt_h = int(chosen_img["width"]), int(chosen_img["height"])
        if (gt_w, gt_h) != (orig_w, orig_h):
            raise DemoError(
                f"COCO GT 图像尺寸 ({gt_w}x{gt_h}) 与上传板卡原图 ({orig_w}x{orig_h}) 不一致，拒绝错配评价。",
                status_code=400,
            )

        # Verify declared SHA-256 & filename correspondence before accepting post-hoc GT
        declared_shas: List[str] = []
        for k in ("sha256", "image_sha256", "source_image_sha256", "file_sha256"):
            raw_v = chosen_img.get(k)
            if raw_v is not None and str(raw_v).strip():
                declared_shas.append(str(raw_v).strip().lower())
        if any(s != uploaded_sha for s in declared_shas):
            raise DemoError(
                "COCO GT 条目声明的图像 SHA-256 与上传原图存在已知哈希冲突，拒绝错配评价（即使同尺寸或人工确认亦不可绕过）。",
                status_code=400,
                details={
                    "reason": "sha256_conflict",
                    "uploaded_sha256": uploaded_sha,
                    "declared_sha256": declared_shas,
                    "coco_image_id": int(chosen_img["id"]),
                    "coco_file_name": chosen_img.get("file_name"),
                },
            )

        has_sha_match = bool(declared_shas) and all(s == uploaded_sha for s in declared_shas)
        coco_fname = str(chosen_img.get("file_name") or "").strip()
        coco_fname_base = pathlib.Path(coco_fname).name.lower() if coco_fname else ""
        coco_stem = pathlib.Path(coco_fname).stem.lower() if coco_fname else ""
        coco_bid = str(chosen_img.get("board_id") or "").strip().lower()
        has_name_match = bool(
            (coco_fname_base and coco_fname_base == safe_fname.lower())
            or (coco_stem and board_stem and coco_stem == board_stem)
            or (coco_bid and board_stem and coco_bid == board_stem)
        )

        if has_sha_match:
            image_identity_verified = True
            identity_basis = "sha256_match"
            mapping_source = "auto_sha256"
            identity_note = "已核对 COCO 条目声明的图像 SHA-256 与上传原图完全一致。"
        elif has_name_match:
            image_identity_verified = False
            identity_basis = "filename_and_dimensions_only"
            mapping_source = "auto_filename_match"
            identity_note = "仅依据文件名与像素尺寸自动匹配，COCO 未提供图像字节 SHA-256（未完成字节身份验证）。"
        elif confirm_manual_mapping:
            image_identity_verified = False
            identity_basis = "manual_user_confirmation"
            mapping_source = "user_confirmed_manual_mapping"
            identity_note = "由用户显式确认将不同名/改名图像映射至选定 COCO GT（未经过图像字节验证）。"
        else:
            raise DemoError(
                "上传图与目标 COCO GT 缺少身份对应依据（文件名不一致且无 SHA-256 匹配；单图或同尺寸本身不构成同图证明）。"
                "若为人为改名后的同一板卡，请勾选并显式确认人工映射。",
                status_code=400,
                details={
                    "requires_manual_confirmation": True,
                    "uploaded_filename": safe_fname,
                    "coco_image_id": int(chosen_img["id"]),
                    "coco_file_name": coco_fname or None,
                    "width": gt_w,
                    "height": gt_h,
                },
            )

        cat_map = {int(c["id"]): str(c["name"]).lower() for c in coco_gt["categories"]}
        target_id = int(chosen_img["id"])
        gts = []
        for ann in coco_gt.get("annotations") or []:
            if int(ann["image_id"]) != target_id:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            gts.append({
                "id": int(ann["id"]),
                "label": cat_map[int(ann["category_id"])],
                "bbox_xyxy": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
            })

        def _assign_box_to_eval_window(
            box_xyxy: Sequence[Any],
            windows: Sequence[Mapping[str, Any]],
        ) -> Optional[str]:
            if len(box_xyxy) != 4:
                return None
            cx = (float(box_xyxy[0]) + float(box_xyxy[2])) / 2.0
            cy = (float(box_xyxy[1]) + float(box_xyxy[3])) / 2.0
            for w in windows:
                wx1, wy1, wx2, wy2 = w["bbox_xyxy"]
                in_x = (wx1 <= cx < wx2) or (wx2 >= float(orig_w) and cx == wx2)
                in_y = (wy1 <= cy < wy2) or (wy2 >= float(orig_h) and cy == wy2)
                if in_x and in_y:
                    return str(w["window_id"])
            return None

        has_top_scope = "eval_scope" in coco_gt
        has_img_scope = "eval_scope" in chosen_img
        has_top_wins = "eval_windows" in coco_gt
        has_img_wins = "eval_windows" in chosen_img

        if has_top_scope and has_img_scope and coco_gt["eval_scope"] != chosen_img["eval_scope"]:
            raise DemoError("COCO 顶层与图像条目的 eval_scope 定义冲突，拒绝评分。", status_code=400)
        if has_top_wins and has_img_wins and coco_gt["eval_windows"] != chosen_img["eval_windows"]:
            raise DemoError("COCO 顶层与图像条目的 eval_windows 定义冲突，拒绝评分。", status_code=400)

        if has_img_scope or has_top_scope:
            raw_scope_val = chosen_img["eval_scope"] if has_img_scope else coco_gt["eval_scope"]
            if not isinstance(raw_scope_val, str) or raw_scope_val.strip().lower() not in ("full_image", "partial_windows"):
                raise DemoError(f"不支持的 eval_scope={raw_scope_val!r}，仅允许 'full_image' 或 'partial_windows'。", status_code=400)
            eval_scope = raw_scope_val.strip().lower()
        else:
            eval_scope = "full_image"

        raw_wins_present = has_img_wins or has_top_wins
        raw_eval_windows = chosen_img["eval_windows"] if has_img_wins else (coco_gt["eval_windows"] if has_top_wins else None)
        if eval_scope == "full_image" and raw_wins_present:
            raise DemoError("未声明 eval_scope='partial_windows' 却携带 eval_windows，存在范围冲突，拒绝评分。", status_code=400)

        eval_windows: List[Dict[str, Any]] = []
        is_partial_windows = (eval_scope == "partial_windows")
        if is_partial_windows:
            if not raw_wins_present or not isinstance(raw_eval_windows, list) or len(raw_eval_windows) == 0:
                raise DemoError("声明 eval_scope='partial_windows' 时必须提供非空 list 类型的 eval_windows。", status_code=400)
            seen_wids: set[str] = set()
            for idx_w, ew in enumerate(raw_eval_windows):
                if not isinstance(ew, Mapping):
                    raise DemoError(f"eval_windows[{idx_w}] 必须是包含 window_id 与 bbox_xyxy 的对象。", status_code=400)
                wid_str = str(ew.get("window_id") or "").strip()
                if not wid_str:
                    raise DemoError(f"eval_windows[{idx_w}] 缺少有效的 window_id。", status_code=400)
                if wid_str in seen_wids:
                    raise DemoError(f"eval_windows 存在重复 window_id={wid_str!r}，拒绝评分。", status_code=400)
                seen_wids.add(wid_str)
                raw_box = ew.get("bbox_xyxy")
                if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                    raise DemoError(f"eval_windows[{idx_w}] ({wid_str}) 的 bbox_xyxy 必须是长度为 4 的数值列表。", status_code=400)
                coords: List[float] = []
                for v in raw_box:
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                        raise DemoError(f"eval_windows[{idx_w}] ({wid_str}) 含非有限或非法坐标 {v!r}，拒绝评分。", status_code=400)
                    coords.append(float(v))
                wx1, wy1, wx2, wy2 = coords
                if wx1 < 0.0 or wy1 < 0.0 or wx2 > float(orig_w) or wy2 > float(orig_h) or wx1 >= wx2 or wy1 >= wy2:
                    raise DemoError(
                        f"eval_windows[{idx_w}] ({wid_str}) 坐标倒置或超出图像范围 ({orig_w}x{orig_h}): {coords}。",
                        status_code=400,
                    )
                eval_windows.append({
                    "window_id": wid_str,
                    "bbox_xyxy": [round(wx1, 2), round(wy1, 2), round(wx2, 2), round(wy2, 2)],
                })

            for i_w in range(len(eval_windows)):
                ax1, ay1, ax2, ay2 = eval_windows[i_w]["bbox_xyxy"]
                for j_w in range(i_w + 1, len(eval_windows)):
                    bx1, by1, bx2, by2 = eval_windows[j_w]["bbox_xyxy"]
                    if max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1)) > 0.0:
                        raise DemoError(
                            f"eval_windows 中 {eval_windows[i_w]['window_id']!r} 与 {eval_windows[j_w]['window_id']!r} 存在未定义处理的重叠区域，拒绝评分。",
                            status_code=400,
                        )

            out_of_win_gts = [
                g["id"] for g in gts
                if _assign_box_to_eval_window(g["bbox_xyxy"], eval_windows) is None
            ]
            if out_of_win_gts:
                raise DemoError(
                    f"存在 {len(out_of_win_gts)} 条 GT 标注超出声明的 eval_windows 边界，请修正标注或窗口定义。",
                    status_code=400,
                    details={"out_of_window_gt_ids": out_of_win_gts[:20]},
                )

        def _filter_to_eval_windows(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
            if not is_partial_windows:
                return [dict(it) for it in items]
            out_items: List[Dict[str, Any]] = []
            for it in items:
                pb = it.get("bbox_xyxy") or it.get("orig_pred_bbox_xyxy") or []
                if _assign_box_to_eval_window(pb, eval_windows) is not None:
                    out_items.append(dict(it))
            return out_items

        eval_preds = _filter_to_eval_windows(res["predictions"])
        eval_pre_candidates = (
            _filter_to_eval_windows(res["pre_verifier_candidates"])
            if res.get("pre_verifier_candidates") is not None
            else None
        )

        if res.get("mode") == "YOLO_T_LOC":
            kept_eval = _match_preds_with_gt_class_agnostic(gts, eval_preds, iou_thresh=0.50)
            aux_eval_25 = _match_preds_with_gt_class_agnostic(gts, eval_preds, iou_thresh=0.25)
            kept_eval["metrics"]["aux_class_agnostic_iou25_tp"] = aux_eval_25["metrics"]["tp"]
            kept_eval["metrics"]["aux_class_agnostic_iou25_recall"] = aux_eval_25["metrics"]["recall"]
            pre_eval = None
        else:
            kept_eval = _match_preds_with_gt(gts, eval_preds, iou_thresh=0.50)
            if eval_pre_candidates is not None:
                if res.get("mode") == "YOLO_T_QWEN3_CLS":
                    pre_eval = _match_preds_with_gt_class_agnostic(gts, eval_pre_candidates, iou_thresh=0.50)
                else:
                    pre_eval = _match_preds_with_gt(gts, eval_pre_candidates, iou_thresh=0.50)
            else:
                pre_eval = None
        class_agnostic_eval = (
            kept_eval["metrics"]
            if res.get("mode") == "YOLO_T_LOC"
            else _match_preds_with_gt_class_agnostic(gts, eval_preds, iou_thresh=0.50)["metrics"]
        )
        single_class_eval: Optional[Dict[str, Any]] = None
        if isinstance(res.get("instruction_spec"), Mapping) and res["instruction_spec"].get("target_class"):
            tgt_cls = str(res["instruction_spec"]["target_class"]).strip().lower()
            tgt_m = dict((kept_eval["metrics"].get("per_class") or {}).get(tgt_cls) or {})
            tgt_gt_cnt = int(tgt_m.get("gt_count", 0))
            tgt_pred_cnt = int(tgt_m.get("pred_count", len(eval_preds)))
            icomp = res.get("instruction_compliance") or {}
            single_class_eval = {
                "target_class": tgt_cls,
                "target_class_cn": str(res["instruction_spec"].get("target_class_cn") or CLASS_CN_NAMES.get(tgt_cls, tgt_cls)),
                "gt_count": tgt_gt_cnt,
                "pred_count": tgt_pred_cnt,
                "count_signed_error": tgt_pred_cnt - tgt_gt_cnt,
                "count_abs_error": abs(tgt_pred_cnt - tgt_gt_cnt),
                "tp": int(tgt_m.get("tp", 0)),
                "fp": int(tgt_m.get("fp", 0)),
                "fn": int(tgt_m.get("fn", 0)),
                "precision": tgt_m.get("precision"),
                "recall": tgt_m.get("recall"),
                "recall_display": tgt_m.get("recall_display"),
                "f1": tgt_m.get("f1"),
                "f1_display": tgt_m.get("f1_display"),
                "unauthorized_post_nms_count": None if is_partial_windows else int(icomp.get("unauthorized_post_nms_count", 0)),
                "unauthorized_post_nms_rate": None if is_partial_windows else float(icomp.get("unauthorized_post_nms_rate", 0.0)),
                "unauthorized_scope_note": "whole_board_compliance_not_comparable_to_partial_windows" if is_partial_windows else None,
            }

        eval_bundle = {
            "gt_evaluated": True,
            "eval_scope": "partial_windows" if is_partial_windows else "full_image",
            "whole_board_f1_valid": not is_partial_windows,
            "eval_windows": eval_windows if is_partial_windows else None,
            "window_attribution_rule": "box_center_half_open_with_image_edge_closure_unique_window" if is_partial_windows else None,
            "scope_notice": (
                f"局部已核窗口评分（{len(eval_windows)} 个窗口 / {len(gts)} GT），不产生整板 F1"
                if is_partial_windows
                else f"整图标注评分（{len(gts)} GT）"
            ),
            "coco_image_id": target_id,
            "coco_file_name": chosen_img.get("file_name"),
            "image_identity_verified": image_identity_verified,
            "identity_basis": identity_basis,
            "mapping_source": mapping_source,
            "identity_note": identity_note,
            "source_image_sha256": uploaded_sha,
            "coco_declared_sha256": declared_shas[0] if declared_shas else None,
            "manual_confirmation": {
                "confirmed": bool(confirm_manual_mapping),
                "reason": str(manual_mapping_reason or "").strip() or None,
                "uploaded_filename": safe_fname,
                "coco_image_id": target_id,
                "coco_file_name": chosen_img.get("file_name"),
            },
            "iou_threshold": 0.50,
            "metrics": kept_eval["metrics"],
            "class_agnostic_eval": class_agnostic_eval,
            "single_class_eval": single_class_eval,
            "gts": kept_eval["gts"],
            "predictions_matched": kept_eval["predictions"],
            "pre_verifier_metrics": pre_eval["metrics"] if pre_eval else None,
            "evaluated_at": _now_iso(),
        }
        res["gt_loaded"] = True
        res["post_hoc_eval"] = eval_bundle
        _write_json_atomic(jdir / "job_result.json", res)
        return res

    def execute_composite_query(
        self,
        *,
        job_id: str,
        query_text: str,
    ) -> Dict[str, Any]:
        """Run GT-free natural-language + spatial query over a completed composite detection job."""
        jdir = self._job_dir(job_id)
        res = self.get_job_result(job_id)
        mode = str(res.get("mode") or "")
        if mode not in ("COMPOSITE_YOLO", "COMPOSITE_YOLO_QWEN3"):
            raise DemoError(
                f"任务 {job_id} 模式为 {mode!r}，仅支持在复合检测任务 (COMPOSITE_YOLO / COMPOSITE_YOLO_QWEN3) 上执行复合自然语言与区域查询。",
                status_code=400,
            )
        comp_meta = res.get("composite_metadata") or {}
        vlm_info = comp_meta.get("vlm_bounded_verification") or {}
        vlm_calls_cnt = int(vlm_info.get("verifier_unique_calls") or 0)
        det_preset = str(comp_meta.get("detector_preset") or res.get("detector_preset") or "wacv_yolo11n")
        w_sha = comp_meta.get("yolo_weights_sha256") or res.get("yolo_weights_sha256")
        qid = f"cq_{time.strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"
        q_res = execute_composite_query(
            predictions=res["predictions"],
            query_text=query_text,
            orig_wh=res["orig_wh"],
            base_job_mode=mode,
            vlm_calls_count=vlm_calls_cnt,
            query_id=qid,
            detector_preset=det_preset,
            yolo_weights_sha256=w_sha,
        )
        if not q_res.get("ok"):
            raise DemoError(
                str(q_res.get("error_message") or "无法解析复合查询指令。"),
                status_code=400,
                details=q_res,
            )
        q_doc = {
            **q_res,
            "job_id": job_id,
            "upload_id": res["upload_id"],
            "board_id": res["board_id"],
            "safe_filename": res["safe_filename"],
            "source_image_sha256": res["source_image_sha256"],
            "orig_wh": res["orig_wh"],
            "roi": res["roi"],
            "overlay_url": f"/api/composite/query_overlay/{job_id}/{qid}",
            "created_at": _now_iso(),
        }
        q_dir = jdir / "composite_queries"
        q_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(q_dir / f"{qid}.json", q_doc)
        return q_doc

    def get_composite_query(self, *, job_id: str, query_id: str) -> Dict[str, Any]:
        jdir = self._job_dir(job_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", str(query_id or "")):
            raise DemoError(f"无效 query_id：{query_id!r}", status_code=400)
        qp = jdir / "composite_queries" / f"{query_id}.json"
        if not qp.is_file():
            raise DemoError(f"复合查询记录不存在：{query_id!r}", status_code=404)
        return json.loads(qp.read_text(encoding="utf-8"))

    def export_composite_query_overlay(self, *, job_id: str, query_id: str) -> Tuple[bytes, str, str]:
        q_doc = self.get_composite_query(job_id=job_id, query_id=query_id)
        roi_p = pathlib.Path(q_doc["roi"]["image_path"])
        if not roi_p.is_file():
            raise DemoError("ROI 图像不存在，无法渲染复合查询叠加图。", status_code=404)
        png_bytes = render_composite_query_overlay_png(
            roi_image_path=roi_p,
            roi_offset_x=int(q_doc["roi"]["offset_x"]),
            roi_offset_y=int(q_doc["roi"]["offset_y"]),
            orig_wh=q_doc["orig_wh"],
            predictions=q_doc["predictions"],
            region_mode=str(q_doc["query_spec"]["region_mode"]),
        )
        return png_bytes, "image/png", f"{job_id}_{query_id}_overlay.png"

    def evaluate_composite_query_posthoc_gt(
        self,
        *,
        job_id: str,
        query_id: str,
        coco_gt: Dict[str, Any],
        target_image_id: Optional[int] = None,
        confirm_manual_mapping: bool = False,
        manual_mapping_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Post-hoc COCO GT evaluation scoped identically to a composite query's target classes & spatial half."""
        q_doc = self.get_composite_query(job_id=job_id, query_id=query_id)
        full_eval_res = self.evaluate_job_with_coco(
            job_id=job_id,
            coco_gt=coco_gt,
            target_image_id=target_image_id,
            confirm_manual_mapping=confirm_manual_mapping,
            manual_mapping_reason=manual_mapping_reason,
        )
        phe = full_eval_res["post_hoc_eval"]
        all_gts = list(phe["gts"])
        target_set = set(q_doc["query_spec"]["target_classes"])
        region_mode = str(q_doc["query_spec"]["region_mode"])
        orig_wh = q_doc["orig_wh"]

        scoped_gts = [
            g for g in all_gts
            if str(g["label"]).lower() in target_set
            and box_in_composite_region(g["bbox_xyxy"], orig_wh=orig_wh, region_mode=region_mode)[0]
        ]
        matched = _match_preds_with_gt(scoped_gts, q_doc["predictions"], iou_thresh=0.50)
        m = matched["metrics"]
        gt_cnt = len(scoped_gts)
        pred_cnt = len(q_doc["predictions"])
        q_eval = {
            "gt_evaluated": True,
            "coco_image_id": phe["coco_image_id"],
            "image_identity_verified": phe["image_identity_verified"],
            "target_classes": sorted(target_set),
            "region_mode": region_mode,
            "gt_count": gt_cnt,
            "pred_count": pred_cnt,
            "count_signed_error": pred_cnt - gt_cnt,
            "count_abs_error": abs(pred_cnt - gt_cnt),
            "tp": int(m["tp"]),
            "fp": int(m["fp"]),
            "fn": int(m["fn"]),
            "precision": m["precision"],
            "recall": m["recall"],
            "recall_display": "N/A (GT=0)" if gt_cnt == 0 else m["recall_display"],
            "f1": m["f1"],
            "f1_display": "N/A (GT=0)" if gt_cnt == 0 else m["f1_display"],
            "per_class": m["per_class"],
            "evaluated_at": _now_iso(),
        }
        q_doc["gt_loaded"] = True
        q_doc["post_hoc_query_eval"] = q_eval
        q_doc["predictions_matched"] = matched["predictions"]
        q_doc["scoped_gts"] = matched["gts"]
        _write_json_atomic(self._job_dir(job_id) / "composite_queries" / f"{query_id}.json", q_doc)
        return q_doc

    def export_composite_query_json(self, *, job_id: str, query_id: str) -> Tuple[bytes, str, str]:
        """Export composite query JSON with original coordinates, weight identity, filter source, and limitations."""
        q_doc = self.get_composite_query(job_id=job_id, query_id=query_id)
        res = self.get_job_result(job_id)
        comp_meta = res.get("composite_metadata") or {}
        export_doc = {
            **q_doc,
            "weight_identity": {
                "detector_preset": comp_meta.get("detector_preset") or q_doc.get("detector_preset"),
                "weights_sha256": comp_meta.get("yolo_weights_sha256") or q_doc.get("yolo_weights_sha256"),
                "expected_class_schema": comp_meta.get("expected_class_schema") or "fics_6",
                "class_conf_thresholds": comp_meta.get("class_conf_thresholds") or {c: 0.25 for c in FICS_SIX_CLASSES},
            },
            "filter_source": q_doc.get("query_execution_mechanism") or "筛选已生成的检测结果",
            "qwen3_participated": bool(str(q_doc.get("base_job_mode") or "") == "COMPOSITE_YOLO_QWEN3"),
            "box_source_breakdown": q_doc.get("box_source_breakdown") or {},
            "low_conf_supplement_count": int(q_doc.get("low_conf_supplement_count") or 0),
            "r19_postproc_summary": comp_meta.get("r19_postproc"),
            "known_limitations": (res.get("export_provenance") or {}).get("known_limitations") or [
                "GT-free 查询基于已有检测结果筛选，不代表真实器件数。",
                "当前 FICS R15 模型在二极管/晶体管上召回率偏弱，IC 仍存在一定高分误报。",
            ],
        }
        return (
            json.dumps(export_doc, indent=2, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            f"{job_id}_{query_id}_query.json",
        )

    def export_job(self, *, job_id: str, kind: str, view: str = "kept") -> Tuple[bytes, str, str]:
        """Generate downloadable standard JSON, annotated PNG overlay, or Markdown summary."""
        res = self.get_job_result(job_id)
        k = str(kind or "").strip().lower()
        if k == "json":
            data = json.dumps(res, indent=2, ensure_ascii=False).encode("utf-8")
            return data, "application/json; charset=utf-8", f"{job_id}_detections.json"
        if k == "overlay_png":
            roi_p = pathlib.Path(res["roi"]["image_path"])
            is_preview = False
            if not roi_p.is_file():
                try:
                    udir = self._upload_dir(res["upload_id"])
                    meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
                    roi_p, is_preview = self._resolve_upload_image_path(udir, meta, "roi")
                except Exception:
                    raise DemoError("ROI 图像不存在，无法导出可视化图片。", status_code=404)
            use_pre = (view == "pre_verifier") and (res.get("pre_verifier_candidates") is not None)
            if use_pre:
                preds = res["pre_verifier_candidates"]
                rejected: List[Mapping[str, Any]] = []
            elif res.get("post_hoc_eval"):
                preds = res["post_hoc_eval"]["predictions_matched"]
                rejected = [c for c in (res.get("pre_verifier_candidates") or []) if not c.get("kept")]
            else:
                preds = res["predictions"]
                rejected = [c for c in (res.get("pre_verifier_candidates") or []) if not c.get("kept")]
            if is_preview:
                with Image.open(roi_p) as prev_im:
                    pw, ph = prev_im.size
                sx = pw / float(res["roi"]["width"])
                sy = ph / float(res["roi"]["height"])
                rox, roy = int(res["roi"]["offset_x"]), int(res["roi"]["offset_y"])

                def _scale_items(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
                    out_items: List[Dict[str, Any]] = []
                    for it in items:
                        box = it.get("bbox_xyxy") or it.get("orig_pred_bbox_xyxy")
                        if not box or len(box) != 4:
                            continue
                        cp = dict(it)
                        cp["bbox_xyxy"] = [
                            (float(box[0]) - rox) * sx,
                            (float(box[1]) - roy) * sy,
                            (float(box[2]) - rox) * sx,
                            (float(box[3]) - roy) * sy,
                        ]
                        out_items.append(cp)
                    return out_items

                png_bytes = render_roi_overlay_png(
                    roi_image_path=roi_p,
                    roi_offset_x=0,
                    roi_offset_y=0,
                    predictions=_scale_items(preds),
                    rejected_candidates=_scale_items(rejected),
                    show_rejected=(view == "with_rejected"),
                )
            else:
                png_bytes = render_roi_overlay_png(
                    roi_image_path=roi_p,
                    roi_offset_x=int(res["roi"]["offset_x"]),
                    roi_offset_y=int(res["roi"]["offset_y"]),
                    predictions=preds,
                    rejected_candidates=rejected,
                    show_rejected=(view == "with_rejected"),
                )
            return png_bytes, "image/png", f"{job_id}_{view}_overlay.png"
        if k in ("summary_md", "summary"):
            out_info = res.get("outcome") or _classify_job_outcome(
                "completed",
                kept_predictions_count=res.get("kept_predictions_count"),
                termination_v2=res.get("termination_v2"),
                technical_complete=res.get("technical_complete"),
            )
            lines = [
                f"# FICS-PCB 整板检测结果摘要 ({job_id})",
                "",
                f"- 上传文件名：`{res['safe_filename']}`",
                f"- 原图 SHA-256：`{res['source_image_sha256']}`",
                f"- 原图尺寸：`{res['orig_wh'][0]}x{res['orig_wh'][1]}` | ROI 偏移：`({res['roi']['offset_x']}, {res['roi']['offset_y']})` | ROI 尺寸：`{res['roi']['width']}x{res['roi']['height']}` | 切片数：`{res['roi']['tile_count']}`",
                f"- 模型：`{res['model_display_name']}` (`{res['model_key']}`)",
                f"- 量化 / 适配器：`{res.get('quantization', 'bf16')}` / `{res.get('adapter_id') or 'none'}`" + (f" (sha256={res.get('adapter_sha256')})" if res.get("adapter_sha256") else ""),
                f"- 原生视觉预处理：`{res.get('visual_preprocess_note') or MODEL_VISUAL_PREPROCESS_NOTES.get(res['model_key'], '')}`",
                f"- 模式：`{res['mode_display_name']}` (`{res['mode']}`)",
                f"- 基础框来源：`{'外部导入的已保存真实推理候选框' if res.get('verifier_only') else '本任务实时检测推理框'}`",
                f"- 技术完整性：`{res['technical_complete']}`",
                f"- 检测结果状态：**{out_info['outcome_label']}** — {out_info['outcome_note']}",
                f"- 保留预测总数：**{res['kept_predictions_count']}**（仅代表模型预测框数，不等于电路板真实元器件总数）",
                "",
            ]
            if res.get("instruction_spec"):
                ispec = res["instruction_spec"]
                icomp = res.get("instruction_compliance") or {}
                lines.extend([
                    "## 自然语言单类指令执行与遵从度（独立重新推理）",
                    f"- 用户输入指令：`{ispec.get('raw_instruction')}`",
                    f"- 解析目标类别：`{ispec.get('target_class_cn')}` (`{ispec.get('target_class')}`)",
                    f"- 实际传入模型 Prompt SHA-256：`{ispec.get('prompt_sha256')}`",
                    f"- 实际传入模型 Prompt：`{ispec.get('normalized_prompt')}`",
                    f"- 去重后目标计数：`{icomp.get('deduplicated_target_count') if res.get('technical_complete') else '未完成/失败'}`",
                    f"- 越权输出（过滤前 Post-NMS）：`{icomp.get('unauthorized_post_nms_count', 0)}/{icomp.get('pre_filter_post_nms_total', 0)}` (`{float(icomp.get('unauthorized_post_nms_rate', 0.0))*100:.2f}%`, 明细={icomp.get('unauthorized_class_breakdown')})",
                    "",
                ])
            lines.append("## 各类别检出计数（6 类）")
            for cls in FICS_SIX_CLASSES:
                cn = CLASS_CN_NAMES.get(cls, cls)
                lines.append(f"- {cn} (`{cls}`): {res['class_counts'].get(cls, 0)}")
            if res.get("composite_metadata"):
                cm = res["composite_metadata"]
                lines.extend([
                    "## 复合检测权重与来源溯源",
                    f"- 权重预设：`{cm.get('detector_preset')}` (`{cm.get('expected_class_schema')}`, SHA-256=`{cm.get('yolo_weights_sha256')}`)",
                    f"- 分类置信度阈值：`{json.dumps(cm.get('class_conf_thresholds') or {}, ensure_ascii=False)}`",
                    f"- Qwen3 参与类别输出：`{cm.get('qwen3_participated', res.get('mode') == 'COMPOSITE_YOLO_QWEN3')}`",
                    f"- 局限性提示：GT-free 预测框数不代表真实器件总数；FICS R15 在 diode/transistor 上召回仍偏弱，IC 仍存在一定误报。",
                    "",
                ])
            if res.get("verifier_summary"):
                vs = res["verifier_summary"]
                lines.extend([
                    "",
                    f"## 二次核验 ({res.get('verifier_policy') or 'B_other'}) 过滤统计",
                    f"- 候选总数 (T Post-NMS)：{vs['total_candidates']}",
                    f"- 核验保留总数：{vs['retained_count']}",
                    f"- 识别为 `other` 剔除：{vs['rejected_as_other']}",
                ])
                if vs.get("rejected_dual_crop_other_consensus") is not None:
                    lines.extend([
                        f"- 阶段 1 (10% 裁片) 直接保留 / 判 `other` 进入阶段 2：{vs.get('stage1_retained_count', 0)} / {vs.get('stage1_rejected_other_count', 0)}",
                        f"- 阶段 2 (40% 宽上下文) 救回有效 6 类目标：{vs.get('rescued_by_stage2_target_count', 0)} (exact={vs.get('rescued_stage2_exact_same', 0)}, cross_target={vs.get('rescued_stage2_cross_target', 0)})",
                        f"- 双裁片一致 (10% 与 40% 均 strict `other`) 剔除：{vs.get('rejected_dual_crop_other_consensus', 0)}",
                    ])
                lines.extend([
                    f"- 异常/缺失过滤：{vs['format_failure_filtered'] + vs['technical_failure_filtered']}",
                    f"- 异常保留总计 (格式/截断/技术/缺失/无效裁剪)：{vs.get('total_anomalies_kept', 0)} (format={vs.get('format_failure_kept_anomaly', 0)}, truncated={vs.get('truncated_kept_anomaly', 0)}, tech_fail={vs.get('technical_failure_kept_anomaly', 0)}, missing={vs.get('missing_record_kept_anomaly', 0)}, invalid_crop={vs.get('invalid_crop_kept_anomaly', 0)})",
                ])
            lines.extend(["", "## 事后 GT 评价状态"])
            if res.get("post_hoc_eval"):
                phe = res["post_hoc_eval"]
                m = phe["metrics"]
                id_verified = bool(phe.get("image_identity_verified"))
                r_disp = m.get("recall_display") or ("N/A (GT=0)" if m.get("recall") is None else f"{m['recall']*100:.2f}%")
                f1_disp = m.get("f1_display") or ("N/A (GT=0)" if m.get("f1") is None else f"{m['f1']*100:.2f}%")
                sce = phe.get("single_class_eval")
                if isinstance(sce, dict):
                    sr_disp = sce.get("recall_display") or ("N/A (GT=0)" if sce.get("recall") is None else f"{sce['recall']*100:.2f}%")
                    sf1_disp = sce.get("f1_display") or ("N/A (GT=0)" if sce.get("f1") is None else f"{sce['f1']*100:.2f}%")
                    sp_val = sce.get("precision")
                    sp_disp = "0.00%" if sp_val is None else f"{sp_val*100:.2f}%"
                    lines.append(
                        f"- 目标单类事后 GT 口径 (`{sce.get('target_class')}`, GT={sce.get('gt_count', 0)}, Pred={sce.get('pred_count', 0)}, abs_err={sce.get('count_abs_error', 0)})："
                        f"TP={sce.get('tp', 0)}, FP={sce.get('fp', 0)}, FN={sce.get('fn', 0)}, P={sp_disp}, R={sr_disp}, F1={sf1_disp}"
                    )
                    lines.append(
                        f"- 同板全 6 类参考口径 (GT={m.get('total_gt', m.get('gt_total', 0))}, IoU>=0.50)："
                        f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, P={m['precision']*100:.2f}%, R={r_disp}, F1={f1_disp}"
                    )
                else:
                    lines.append(
                        f"- 已对齐事后 COCO GT (image_id={phe['coco_image_id']}, file_name={phe.get('coco_file_name')}, IoU>=0.50)："
                        f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, P={m['precision']*100:.2f}%, R={r_disp}, F1={f1_disp}"
                    )
                if isinstance(phe.get("class_agnostic_eval"), dict):
                    ca = phe["class_agnostic_eval"]
                    ca_r = ca.get("recall_display") or ("N/A (GT=0)" if ca.get("recall") is None else f"{ca['recall']*100:.2f}%")
                    lines.append(
                        f"- 类无关定位基准 (Class-Agnostic IoU>=0.50)：TP={ca.get('tp', 0)}, FP={ca.get('fp', 0)}, FN={ca.get('fn', 0)}, P={ca.get('precision', 0)*100:.2f}%, R={ca_r}"
                    )
                lines.append(
                    f"- 图像身份对应：`image_identity_verified={id_verified}` | `mapping_source={phe.get('mapping_source')}` | {phe.get('identity_note')}"
                )
                pc = m.get("per_class") or {}
                for cls in FICS_SIX_CLASSES:
                    c = pc.get(cls) or {}
                    cr = c.get("recall_display") or ("N/A (GT=0)" if c.get("recall") is None else f"{c['recall']*100:.2f}%")
                    lines.append(
                        f"  - `{cls}`: GT={c.get('gt_count', 0)}, Pred={c.get('pred_count', 0)}, "
                        f"TP={c.get('tp', 0)}, FP={c.get('fp', 0)}, FN={c.get('fn', 0)}, R={cr}"
                    )
            else:
                lines.append(
                    "- **未评价（模型预测 / 未做精度评价）**：本次运行未载入 Ground Truth；全程严格遵循 GT-free 流程，"
                    "不计算准确率/召回率/F1，且 `score=1.0` 仅为离散坐标输出占位符，不代表 100% 置信度。"
                )
            text = ("\n".join(lines) + "\n").encode("utf-8")
            return text, "text/markdown; charset=utf-8", f"{job_id}_summary.md"
        raise DemoError(f"不支持的导出格式：{kind!r}", status_code=400)

    def _get_comparison_lock(self, comparison_id: str) -> threading.Lock:
        with self._job_state_locks_guard:
            lk = self._comparison_locks.get(comparison_id)
            if lk is None:
                lk = threading.Lock()
                self._comparison_locks[comparison_id] = lk
            return lk

    def _update_comparison(self, comparison_id: str, updater: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        cdir = self.comparisons_dir / comparison_id
        cp = cdir / "comparison.json"
        with self._get_comparison_lock(comparison_id):
            data = json.loads(cp.read_text(encoding="utf-8"))
            updater(data)
            _write_json_atomic(cp, data)
            return data

    def submit_comparison(
        self,
        *,
        upload_id: str,
        model_keys: Sequence[str],
        mode: str = "W",
        preset: str = "baseline",
        include_qwen3_b_other: bool = False,
    ) -> Dict[str, Any]:
        """Submit a same-image multi-model serial comparison queue sharing upload_id, SHA-256, and mode."""
        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
        if meta.get("read_only_playback_only"):
            raise DemoError(
                "当前上传记录为轻量离线交付包内置的只读回放示例（派生预览图仅供展示，不可作为原图重推）；"
                "如需真实推理，请上传本地新图或按文档挂载外置原图与模型权重。",
                status_code=400,
            )
        norm_mode = normalize_mode(mode)
        raw_preset = str(preset or "baseline").strip() or "baseline"
        if raw_preset not in SUPPORTED_PRESETS:
            raise DemoError(
                f"多模型同图对比不支持预设 {raw_preset!r}；可选值：{', '.join(SUPPORTED_PRESETS)}。",
                status_code=400,
            )

        cleaned_models: List[str] = []
        for raw_mk in model_keys or []:
            mk = str(raw_mk or "").strip().lower()
            if mk not in MODEL_ORDER:
                raise DemoError(f"不支持的对比模型：{raw_mk!r}；仅支持 qwen2、deepseek、qwen3。", status_code=400)
            if mk not in cleaned_models:
                cleaned_models.append(mk)
        if not cleaned_models:
            raise DemoError("请至少选择一个模型执行检测或同图对比。", status_code=400)
        if raw_preset not in ("baseline", "r3_candidate", "r4_highres") and len(cleaned_models) != 1:
            raise DemoError(
                f"多模型同图对比（{len(cleaned_models)} 个模型）需传入组预设 'baseline'、'r3_candidate' 或 'r4_highres'；单模型具体预设 {raw_preset!r} 仅允许用于对应单模型对比队列。",
                status_code=400,
            )
        if norm_mode == "T_B_other" and any(mk != "qwen3" for mk in cleaned_models):
            raise DemoError(
                "冻结方法 T + B_other 仅适用于 Qwen3-VL-2B-Instruct，不可作为多模型同配置对比基线；"
                "多模型同图对比请选择 W 或 T（可将 Qwen3 T+B_other 作为单独增强项开启）。",
                status_code=400,
            )

        slots: List[Dict[str, Any]] = []
        cmp_tile_sizes: set[int] = set()
        cmp_tile_overlaps: set[float] = set()
        cmp_tile_counts: set[int] = set()
        for idx, mk in enumerate(cleaned_models):
            p_spec = resolve_model_preset_spec(mk, raw_preset)
            slot_tile_size = int(p_spec.get("tile_size") or 1024)
            slot_tile_overlap = float(p_spec.get("tile_overlap_ratio", 0.20))
            if norm_mode != "W":
                slot_tiles = self._resolve_upload_tiles_for_geometry(
                    upload_id,
                    meta,
                    tile_size=slot_tile_size,
                    overlap_ratio=slot_tile_overlap,
                )
                slot_tile_count = len(slot_tiles)
            else:
                slot_tile_count = int(meta["tile_count"])
            cmp_tile_sizes.add(slot_tile_size)
            cmp_tile_overlaps.add(slot_tile_overlap)
            cmp_tile_counts.add(slot_tile_count)
            slot_role = (
                "baseline_comparison"
                if (norm_mode in ("W", "T") and p_spec["preset_group"] == "baseline")
                else ("r5_tile768_single_intervention" if p_spec["preset_group"] == "r5_tile768"
                      else ("r4_highres_single_intervention" if p_spec["preset_group"] == "r4_highres"
                            else ("r3_single_intervention_candidate" if p_spec["preset_group"] == "r3_candidate" else "separate_qwen3_enhancement")))
            )
            slots.append({
                "slot_index": idx,
                "slot_id": f"{mk}_{norm_mode}",
                "slot_role": slot_role,
                "model_key": mk,
                "model_display_name": MODEL_DISPLAY_NAMES[mk],
                "visual_preprocess_note": _format_visual_preprocess_note(mk, p_spec),
                "mode": norm_mode,
                "mode_display_name": MODE_DISPLAY_NAMES[norm_mode],
                "preset": str(p_spec["preset_id"]),
                "preset_group": str(p_spec["preset_group"]),
                "preset_display_name": str(p_spec["preset_display_name"]),
                "coordinate_protocol": str(p_spec["coordinate_protocol"]),
                "explicit_do_sample_false": bool(p_spec["explicit_do_sample_false"]),
                "qwen_min_pixels": int(p_spec.get("qwen_min_pixels", 65536)),
                "qwen_max_pixels": int(p_spec.get("qwen_max_pixels", 200704)),
                "tile_size": slot_tile_size,
                "tile_overlap_ratio": slot_tile_overlap,
                "tile_count": slot_tile_count,
                "prompt_sha256": str(p_spec["prompt_sha256"]),
                "intervention_note": str(p_spec["intervention_note"]),
                "queue_status": "waiting",
                "job_id": None,
                "error_message": None,
            })

        if include_qwen3_b_other and norm_mode != "T_B_other":
            q3_b_spec = resolve_model_preset_spec("qwen3", "baseline")
            slots.append({
                "slot_index": len(slots),
                "slot_id": "qwen3_T_B_other",
                "slot_role": "separate_qwen3_enhancement",
                "model_key": "qwen3",
                "model_display_name": MODEL_DISPLAY_NAMES["qwen3"],
                "visual_preprocess_note": MODEL_VISUAL_PREPROCESS_NOTES["qwen3"],
                "mode": "T_B_other",
                "mode_display_name": MODE_DISPLAY_NAMES["T_B_other"],
                "preset": str(q3_b_spec["preset_id"]),
                "preset_group": "baseline",
                "preset_display_name": str(q3_b_spec["preset_display_name"]),
                "coordinate_protocol": str(q3_b_spec["coordinate_protocol"]),
                "explicit_do_sample_false": bool(q3_b_spec["explicit_do_sample_false"]),
                "tile_size": 1024,
                "tile_overlap_ratio": 0.20,
                "tile_count": int(meta["tile_count"]),
                "prompt_sha256": str(q3_b_spec["prompt_sha256"]),
                "intervention_note": str(q3_b_spec["intervention_note"]),
                "queue_status": "waiting",
                "job_id": None,
                "error_message": None,
            })

        with self._lock:
            if self._active_job_id is not None or self._active_comparison_id is not None:
                busy_id = str(self._active_job_id or self._active_comparison_id)
                raise BusyGpuJobError(
                    f"已有 GPU 任务或对比队列正在运行 ({busy_id})，请等待其完成或点击取消后再发起新对比。",
                    active_job_id=busy_id,
                )
            comparison_id = f"cmp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            cdir = self.comparisons_dir / comparison_id
            cdir.mkdir(parents=True, exist_ok=True)
            eff_tile_size = next(iter(cmp_tile_sizes)) if len(cmp_tile_sizes) == 1 else 1024
            eff_tile_ov = next(iter(cmp_tile_overlaps)) if len(cmp_tile_overlaps) == 1 else 0.20
            eff_tile_cnt = next(iter(cmp_tile_counts)) if len(cmp_tile_counts) == 1 else int(meta["tile_count"])
            cmp_doc = {
                "schema_version": "fics_demo_comparison_v1",
                "comparison_id": comparison_id,
                "upload_id": upload_id,
                "board_id": meta["board_id"],
                "safe_filename": meta["safe_filename"],
                "source_image_sha256": meta["source_image_sha256"],
                "orig_wh": meta["orig_wh"],
                "roi": {
                    "bbox_half_open": meta["roi_bbox_half_open"],
                    "offset_x": meta["roi_offset_x"],
                    "offset_y": meta["roi_offset_y"],
                    "width": meta["roi_width"],
                    "height": meta["roi_height"],
                    "tile_size": eff_tile_size,
                    "tile_overlap_ratio": eff_tile_ov,
                    "tile_count": eff_tile_cnt,
                    "sha256": (meta.get("roi_export") or {}).get("sha256"),
                },
                "baseline_mode": norm_mode,
                "baseline_mode_display_name": MODE_DISPLAY_NAMES[norm_mode],
                "preset": raw_preset,
                "preset_display_name": (
                    "第 9 轮 micro-QLoRA 适配器候选 (r9_nf4_adapter)"
                    if raw_preset in ("r9_nf4_adapter", "r9_qwen3_nf4_adapter_tile768")
                    else ("第 9 轮 NF4 同精度基准 (r9_nf4_base)"
                          if raw_preset in ("r9_nf4_base", "r9_qwen3_nf4_base_tile768")
                          else ("第 5 轮 Qwen3 768px 小切片候选 (r5_tile768)"
                                if raw_preset in ("r5_tile768", "r5_qwen3_greedy_maxpx589824_tile768")
                                else ("第 4 轮高视觉预算候选 (r4_highres)"
                                      if raw_preset == "r4_highres"
                                      else ("第 3 轮单干预候选配置 (r3_candidate)" if raw_preset == "r3_candidate"
                                            else ("共同基准配置 (baseline)" if raw_preset == "baseline" else raw_preset)))))
                ),
                "model_keys": cleaned_models,
                "include_qwen3_b_other": bool(include_qwen3_b_other and norm_mode != "T_B_other"),
                "status": "running",
                "cancel_requested": False,
                "created_at": _now_iso(),
                "started_at_epoch": time.time(),
                "finished_at_epoch": None,
                "slots": slots,
            }
            _write_json_atomic(cdir / "comparison.json", cmp_doc)
            cancel_ev = threading.Event()
            self._comparison_cancel_events[comparison_id] = cancel_ev
            self._active_comparison_id = comparison_id

            th = threading.Thread(target=self._run_comparison_thread, args=(comparison_id, cancel_ev), daemon=True)
            self._comparison_threads[comparison_id] = th
            th.start()
            return self.get_comparison_status(comparison_id)

    def _run_comparison_thread(self, comparison_id: str, cancel_ev: threading.Event) -> None:
        cdir = self.comparisons_dir / comparison_id
        cmp_doc = json.loads((cdir / "comparison.json").read_text(encoding="utf-8"))
        upload_id = str(cmp_doc["upload_id"])
        num_slots = len(cmp_doc["slots"])

        try:
            for idx in range(num_slots):
                if cancel_ev.is_set():
                    def _mark_cancelled(d: Dict[str, Any], i: int = idx) -> None:
                        if d["slots"][i]["queue_status"] == "waiting":
                            d["slots"][i]["queue_status"] = "cancelled"
                            d["slots"][i]["error_message"] = "同图对比队列已取消，该模型未启动。"
                    self._update_comparison(comparison_id, _mark_cancelled)
                    continue

                slot = json.loads((cdir / "comparison.json").read_text(encoding="utf-8"))["slots"][idx]
                try:
                    st = self.submit_job(
                        upload_id=upload_id,
                        model_key=slot["model_key"],
                        mode=slot["mode"],
                        preset=str(slot.get("preset") or "baseline"),
                        _from_comparison_id=comparison_id,
                    )
                    jid = st["job_id"]

                    def _mark_started(d: Dict[str, Any], i: int = idx, job_id_val: str = jid) -> None:
                        d["slots"][i]["job_id"] = job_id_val
                        d["slots"][i]["queue_status"] = "running"
                    self._update_comparison(comparison_id, _mark_started)

                    j_th = self._job_threads.get(jid)
                    while j_th is not None and j_th.is_alive():
                        if cancel_ev.is_set():
                            self.cancel_job(jid)
                        j_th.join(timeout=0.1)

                    end_st = self.get_job_status(jid)

                    def _mark_done(d: Dict[str, Any], i: int = idx, est: Dict[str, Any] = end_st) -> None:
                        d["slots"][i]["queue_status"] = est["status"]
                        d["slots"][i]["error_message"] = est.get("error_message")
                    self._update_comparison(comparison_id, _mark_done)
                except InterruptedError:
                    def _mark_cancelled_atomic(d: Dict[str, Any], i: int = idx) -> None:
                        d["slots"][i]["queue_status"] = "cancelled"
                        d["slots"][i]["error_message"] = "同图对比队列已取消，该模型未启动。"
                    self._update_comparison(comparison_id, _mark_cancelled_atomic)
                except Exception as exc:
                    err_msg = str(exc)

                    def _mark_err(d: Dict[str, Any], i: int = idx, msg: str = err_msg) -> None:
                        d["slots"][i]["queue_status"] = "cancelled" if cancel_ev.is_set() else "failed"
                        d["slots"][i]["error_message"] = msg
                    self._update_comparison(comparison_id, _mark_err)

            def _finalize(d: Dict[str, Any]) -> None:
                d["finished_at_epoch"] = time.time()
                statuses = [s["queue_status"] for s in d["slots"]]
                if d.get("cancel_requested") or cancel_ev.is_set() or all(s == "cancelled" for s in statuses):
                    if any(s == "completed" for s in statuses):
                        d["status"] = "partial_cancelled"
                    else:
                        d["status"] = "cancelled"
                elif all(s == "completed" for s in statuses):
                    d["status"] = "completed"
                elif any(s == "completed" for s in statuses):
                    d["status"] = "partial_completed"
                else:
                    d["status"] = "failed"
            self._update_comparison(comparison_id, _finalize)
        finally:
            with self._lock:
                self._comparison_cancel_events.pop(comparison_id, None)
                self._cancelled_comparisons.discard(comparison_id)
                if self._active_comparison_id == comparison_id:
                    self._active_comparison_id = None

    def create_comparison_from_jobs(
        self,
        *,
        upload_id: str,
        mode: str,
        job_ids: Sequence[str],
        addon_job_id: Optional[str] = None,
        addon_job_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Bind existing single-model jobs into a verified same-image comparison record.

        Strictly enforces identical upload_id, identical source_image_sha256, and matching mode
        so predictions from different images or incompatible configurations can never be mixed.
        """
        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
        exp_sha = str(meta["source_image_sha256"]).strip().lower()
        norm_mode = normalize_mode(mode)
        if not job_ids:
            raise DemoError("至少需要提供一个 job_id 以构建同图对比。", status_code=400)

        slots: List[Dict[str, Any]] = []
        seen_models: List[str] = []
        for idx, raw_jid in enumerate(job_ids):
            jid = str(raw_jid or "").strip()
            st = self.get_job_status(jid)
            if st["upload_id"] != upload_id or str(st.get("source_image_sha256") or "").strip().lower() != exp_sha:
                raise DemoError(
                    f"任务 {jid} 属于其他上传图像 (upload_id={st['upload_id']})，严禁跨图混排对比结果。",
                    status_code=400,
                )
            if st["mode"] != norm_mode:
                raise DemoError(
                    f"任务 {jid} 的模式 ({st['mode']}) 与对比基线模式 ({norm_mode}) 不一致，拒绝混入同配置对比列。",
                    status_code=400,
                )
            mk = st["model_key"]
            if mk not in seen_models:
                seen_models.append(mk)
            slots.append({
                "slot_index": idx,
                "slot_id": f"{mk}_{norm_mode}",
                "slot_role": "baseline_comparison" if norm_mode in ("W", "T") else "separate_qwen3_enhancement",
                "model_key": mk,
                "model_display_name": MODEL_DISPLAY_NAMES[mk],
                "visual_preprocess_note": MODEL_VISUAL_PREPROCESS_NOTES[mk],
                "mode": norm_mode,
                "mode_display_name": MODE_DISPLAY_NAMES[norm_mode],
                "queue_status": st["status"],
                "job_id": jid,
                "error_message": st.get("error_message"),
            })

        all_addon_ids: List[str] = []
        if addon_job_id:
            all_addon_ids.append(str(addon_job_id))
        for raw_ajid in addon_job_ids or []:
            if raw_ajid and str(raw_ajid) not in all_addon_ids:
                all_addon_ids.append(str(raw_ajid))

        for ajid in all_addon_ids:
            ast = self.get_job_status(ajid)
            if ast["upload_id"] != upload_id or str(ast.get("source_image_sha256") or "").strip().lower() != exp_sha:
                raise DemoError(
                    f"增强任务 {ajid} 与当前上传图像身份不一致，拒绝绑定。",
                    status_code=400,
                )
            if (ast["model_key"], ast["mode"]) not in (
                ("qwen3", "T_B_other"),
                ("yolo11n", "YOLO_T_LOC"),
                ("yolo11n_qwen3", "YOLO_T_QWEN3_CLS"),
            ):
                raise DemoError("单独增强/实验对比项仅允许绑定 Qwen3 T_B_other、YOLO_T_LOC 或 YOLO_T_QWEN3_CLS 任务。", status_code=400)
            amk = ast["model_key"]
            amode = ast["mode"]
            slots.append({
                "slot_index": len(slots),
                "slot_id": f"{amk}_{amode}",
                "slot_role": "separate_qwen3_enhancement",
                "model_key": amk,
                "model_display_name": MODEL_DISPLAY_NAMES[amk],
                "visual_preprocess_note": MODEL_VISUAL_PREPROCESS_NOTES[amk],
                "mode": amode,
                "mode_display_name": MODE_DISPLAY_NAMES[amode],
                "queue_status": ast["status"],
                "job_id": ajid,
                "error_message": ast.get("error_message"),
            })

        comparison_id = f"cmp_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        cdir = self.comparisons_dir / comparison_id
        cdir.mkdir(parents=True, exist_ok=True)
        statuses = [s["queue_status"] for s in slots]
        overall_st = (
            "completed" if all(s == "completed" for s in statuses)
            else "partial_completed" if any(s == "completed" for s in statuses)
            else "failed"
        )
        cmp_doc = {
            "schema_version": "fics_demo_comparison_v1",
            "comparison_id": comparison_id,
            "upload_id": upload_id,
            "board_id": meta["board_id"],
            "safe_filename": meta["safe_filename"],
            "source_image_sha256": meta["source_image_sha256"],
            "orig_wh": meta["orig_wh"],
            "roi": {
                "bbox_half_open": meta["roi_bbox_half_open"],
                "offset_x": meta["roi_offset_x"],
                "offset_y": meta["roi_offset_y"],
                "width": meta["roi_width"],
                "height": meta["roi_height"],
                "tile_count": meta["tile_count"],
                "sha256": (meta.get("roi_export") or {}).get("sha256"),
            },
            "baseline_mode": norm_mode,
            "baseline_mode_display_name": MODE_DISPLAY_NAMES[norm_mode],
            "model_keys": seen_models,
            "include_qwen3_b_other": bool(addon_job_id),
            "status": overall_st,
            "cancel_requested": False,
            "created_at": _now_iso(),
            "started_at_epoch": time.time(),
            "finished_at_epoch": time.time(),
            "slots": slots,
        }
        _write_json_atomic(cdir / "comparison.json", cmp_doc)
        return self.get_comparison_status(comparison_id)

    def cancel_comparison(self, comparison_id: str) -> Dict[str, Any]:
        """Atomically mark comparison cancelled under mutex so no undispatched next slot can start."""
        cdir = self._comparison_dir(comparison_id)
        with self._lock:
            ev = self._comparison_cancel_events.get(comparison_id)
            active_jid = self._active_job_id
            if ev is not None:
                self._cancelled_comparisons.add(comparison_id)
                ev.set()
        if ev is not None:
            def _set_cancel(d: Dict[str, Any]) -> None:
                d["cancel_requested"] = True
                for s in d.get("slots") or []:
                    if s.get("queue_status") == "waiting":
                        s["queue_status"] = "cancelled"
                        s["error_message"] = "用户取消了同图对比队列，该模型未启动。"
            self._update_comparison(comparison_id, _set_cancel)
        cmp_doc = json.loads((cdir / "comparison.json").read_text(encoding="utf-8"))
        jids_to_cancel: List[str] = []
        for s in cmp_doc.get("slots") or []:
            jid = s.get("job_id")
            if jid and (jid == active_jid or s.get("queue_status") == "running"):
                jids_to_cancel.append(str(jid))
        if active_jid and active_jid not in jids_to_cancel:
            try:
                spec_p = self.jobs_dir / active_jid / "job_spec.json"
                if spec_p.is_file():
                    spec_doc = json.loads(spec_p.read_text(encoding="utf-8"))
                    if spec_doc.get("comparison_id") == comparison_id:
                        jids_to_cancel.append(str(active_jid))
            except Exception:
                pass
        for jid in jids_to_cancel:
            try:
                self.cancel_job(jid)
            except Exception:
                pass
        return self.get_comparison_status(comparison_id)

    def get_comparison_status(self, comparison_id: str) -> Dict[str, Any]:
        """Return enriched multi-model comparison status with strict same-image identity verification."""
        cdir = self._comparison_dir(comparison_id)
        with self._get_comparison_lock(comparison_id):
            doc = json.loads((cdir / "comparison.json").read_text(encoding="utf-8"))
        exp_uid = str(doc["upload_id"])
        exp_sha = str(doc["source_image_sha256"]).strip().lower()

        enriched_slots: List[Dict[str, Any]] = []
        total_elapsed = 0.0
        for s in doc.get("slots") or []:
            slot_out = dict(s)
            jid = s.get("job_id")
            if jid:
                st = self.get_job_status(jid)
                if st["upload_id"] != exp_uid or str(st.get("source_image_sha256") or "").strip().lower() != exp_sha:
                    raise DemoError(
                        f"检测到跨图污染异常：对比 {comparison_id} 绑定的任务 {jid} 图像身份不匹配。",
                        status_code=500,
                    )
                slot_out["queue_status"] = st["status"]
                slot_out["stage_label"] = st["stage_label"]
                slot_out["elapsed_seconds"] = st["elapsed_seconds"]
                slot_out["base_calls_planned"] = st["base_calls_planned"]
                slot_out["base_calls_completed"] = st["base_calls_completed"]
                slot_out["verifier_calls_planned"] = st["verifier_calls_planned"]
                slot_out["verifier_calls_completed"] = st["verifier_calls_completed"]
                slot_out["gpu_gate"] = st["gpu_gate"]
                slot_out["error_message"] = st.get("error_message") or s.get("error_message")
                slot_out["outcome"] = st["outcome"]
                slot_out["has_result"] = st["has_result"]
                total_elapsed += float(st.get("elapsed_seconds") or 0.0)
                if st["status"] == "completed" and st["has_result"]:
                    res = self.get_job_result(jid)
                    slot_out["preset"] = res.get("preset", s.get("preset", "baseline"))
                    slot_out["preset_display_name"] = res.get("preset_display_name", s.get("preset_display_name"))
                    slot_out["coordinate_protocol"] = res.get("coordinate_protocol", s.get("coordinate_protocol", "xyxy_norm1000"))
                    slot_out["explicit_do_sample_false"] = bool(res.get("explicit_do_sample_false", s.get("explicit_do_sample_false", False)))
                    slot_out["planned_visual_budget"] = res.get("planned_visual_budget") or {
                        "qwen_min_pixels": s.get("qwen_min_pixels", 65536),
                        "qwen_max_pixels": s.get("qwen_max_pixels", 200704),
                        "tile_size": s.get("tile_size", 1024),
                        "tile_overlap_ratio": s.get("tile_overlap_ratio", 0.20),
                    }
                    slot_out["actual_visual_telemetry_summary"] = res.get("actual_visual_telemetry_summary")
                    slot_out["effective_generation_probe_summary"] = res.get("effective_generation_probe_summary")
                    slot_out["kept_predictions_count"] = res["kept_predictions_count"]
                    slot_out["post_nms_candidates_count"] = res["post_nms_candidates_count"]
                    slot_out["class_counts"] = res["class_counts"]
                    slot_out["pre_verifier_class_counts"] = res.get("pre_verifier_class_counts")
                    slot_out["verifier_summary"] = res.get("verifier_summary")
                    slot_out["termination_v2"] = res["termination_v2"]
                    slot_out["predictions"] = res["predictions"]
                    slot_out["pre_verifier_candidates"] = res.get("pre_verifier_candidates")
                    slot_out["calls"] = res["calls"]
                    slot_out["post_hoc_eval"] = res.get("post_hoc_eval")
                    slot_out["exports"] = {
                        "json": f"/api/jobs/{jid}/export/json",
                        "overlay_png": f"/api/jobs/{jid}/export/overlay_png",
                        "summary_md": f"/api/jobs/{jid}/export/summary_md",
                    }
                else:
                    slot_out["kept_predictions_count"] = None
                    slot_out["class_counts"] = None
                    slot_out["termination_v2"] = None
                    slot_out["predictions"] = None
                    slot_out["exports"] = None
            else:
                q_st = str(s.get("queue_status") or "waiting")
                slot_out["stage_label"] = "队列等待中" if q_st == "waiting" else ("已取消（未启动）" if q_st == "cancelled" else q_st)
                slot_out["elapsed_seconds"] = 0.0
                slot_out["outcome"] = _classify_job_outcome(q_st, error_message=s.get("error_message"))
                slot_out["has_result"] = False
                slot_out["kept_predictions_count"] = None
                slot_out["class_counts"] = None
                slot_out["termination_v2"] = None
                slot_out["predictions"] = None
                slot_out["exports"] = None
            enriched_slots.append(slot_out)

        end_t = doc.get("finished_at_epoch") or time.time()
        wall_s = round(max(0.0, float(end_t) - float(doc.get("started_at_epoch") or end_t)), 1)
        return {
            "schema_version": "fics_demo_comparison_v1",
            "comparison_id": comparison_id,
            "upload_id": exp_uid,
            "board_id": doc["board_id"],
            "safe_filename": doc["safe_filename"],
            "source_image_sha256": doc["source_image_sha256"],
            "orig_wh": doc["orig_wh"],
            "roi": doc["roi"],
            "baseline_mode": doc["baseline_mode"],
            "baseline_mode_display_name": doc["baseline_mode_display_name"],
            "preset": doc.get("preset", "baseline"),
            "preset_display_name": doc.get("preset_display_name", "共同基准配置 (baseline)"),
            "model_keys": doc["model_keys"],
            "include_qwen3_b_other": doc.get("include_qwen3_b_other", False),
            "status": doc["status"],
            "cancel_requested": bool(doc.get("cancel_requested")),
            "created_at": doc["created_at"],
            "wall_elapsed_seconds": wall_s,
            "sum_job_elapsed_seconds": round(total_elapsed, 1),
            "class_cn_names": dict(CLASS_CN_NAMES),
            "gt_free_notice": (
                "模型预测 / 未做精度评价（GT-free）：当前无人工真值，仅展示各模型预测框与类别统计；"
                "预测框数不代表板上真实元器件数，不计算准确率/F1，score=1.0 仅为固定占位符而非概率置信度。"
            ),
            "zero_box_semantics_notice": (
                "“无有效框”仅表示该模型在当前模式下未解析出合法目标框，不等于电路板上没有元器件；"
                "模型输出截断、格式不合规或运行失败均如实标记，不伪装为显式空预测 []。"
            ),
            "slots": enriched_slots,
        }

    def list_comparisons(self, limit: int = 20, upload_id: Optional[str] = None) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not self.comparisons_dir.is_dir():
            return items
        for d in sorted(self.comparisons_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not d.is_dir() or not (d / "comparison.json").is_file():
                continue
            try:
                st = self.get_comparison_status(d.name)
                if upload_id and st.get("upload_id") != upload_id:
                    continue
                items.append(st)
            except Exception:
                continue
            if len(items) >= int(limit):
                break
        return items

    def export_comparison(self, *, comparison_id: str, kind: str = "markdown") -> Tuple[bytes, str, str]:
        """Export lightweight same-image multi-model comparison report (JSON or Markdown).

        Unfinished, cancelled, or failed models never emit fake empty prediction tables or export links.
        """
        cmp_data = self.get_comparison_status(comparison_id)
        k = str(kind or "markdown").strip().lower()
        if k == "json":
            export_doc = dict(cmp_data)
            # Strip heavy per-box/call arrays in lightweight JSON summary while preserving counts, status, and export refs
            compact_slots = []
            for s in cmp_data["slots"]:
                cs = {
                    key: s.get(key)
                    for key in (
                        "slot_index",
                        "slot_id",
                        "slot_role",
                        "model_key",
                        "model_display_name",
                        "visual_preprocess_note",
                        "mode",
                        "mode_display_name",
                        "preset",
                        "preset_display_name",
                        "coordinate_protocol",
                        "explicit_do_sample_false",
                        "planned_visual_budget",
                        "actual_visual_telemetry_summary",
                        "effective_generation_probe_summary",
                        "queue_status",
                        "job_id",
                        "elapsed_seconds",
                        "outcome",
                        "kept_predictions_count",
                        "post_nms_candidates_count",
                        "class_counts",
                        "verifier_summary",
                        "termination_v2",
                        "error_message",
                        "exports",
                    )
                }
                phe = s.get("post_hoc_eval")
                if isinstance(phe, Mapping) and phe.get("gt_evaluated"):
                    cs["post_hoc_eval"] = {
                        "gt_evaluated": True,
                        "eval_scope": phe.get("eval_scope"),
                        "whole_board_f1_valid": phe.get("whole_board_f1_valid"),
                        "eval_windows": phe.get("eval_windows"),
                        "window_attribution_rule": phe.get("window_attribution_rule"),
                        "scope_notice": phe.get("scope_notice"),
                        "coco_image_id": phe.get("coco_image_id"),
                        "coco_file_name": phe.get("coco_file_name"),
                        "image_identity_verified": bool(phe.get("image_identity_verified")),
                        "identity_basis": phe.get("identity_basis"),
                        "mapping_source": phe.get("mapping_source"),
                        "identity_note": phe.get("identity_note"),
                        "source_image_sha256": phe.get("source_image_sha256"),
                        "coco_declared_sha256": phe.get("coco_declared_sha256"),
                        "iou_threshold": phe.get("iou_threshold", 0.5),
                        "metrics": phe.get("metrics"),
                        "class_agnostic_eval": phe.get("class_agnostic_eval"),
                        "evaluated_at": phe.get("evaluated_at"),
                    }
                else:
                    cs["post_hoc_eval"] = None
                compact_slots.append(cs)
            export_doc["slots"] = compact_slots
            blob = (json.dumps(export_doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            return blob, "application/json; charset=utf-8", f"{comparison_id}_comparison.json"

        if k in ("markdown", "md", "summary_md"):
            roi = cmp_data["roi"]
            lines = [
                f"# FICS-PCB 同图多模型检测对比报告 ({comparison_id})",
                "",
                "## 1. 共同图像身份与坐标基准",
                f"- **上传文件名**：`{cmp_data['safe_filename']}` (`upload_id={cmp_data['upload_id']}`)",
                f"- **原图 SHA-256**：`{cmp_data['source_image_sha256']}`",
                f"- **原图尺寸**：`{cmp_data['orig_wh'][0]} × {cmp_data['orig_wh'][1]}` px",
                f"- **ROI 区域（阈值 <248 + 32px）**：`{roi['width']} × {roi['height']}` px | 偏移 `({roi['offset_x']}, {roi['offset_y']})` | {roi.get('tile_size', 1024)}px 切片数：`{roi['tile_count']}`",
                f"- **对比基线模式**：`{cmp_data['baseline_mode_display_name']}` (`{cmp_data['baseline_mode']}`)",
                f"- **队列总体状态**：`{cmp_data['status']}` | 累计模型推理耗时：`{cmp_data['sum_job_elapsed_seconds']}s`",
                f"- **评价口径声明**：{cmp_data['gt_free_notice']}",
                f"- **零框与异常状态说明**：{cmp_data['zero_box_semantics_notice']}",
                "",
                "## 2. 同配置基线模型对比汇总",
                "",
                "| 模型名称 | 模式 | 运行/结果状态 | 耗时 (s) | 预测总框数 | 电容 (capacitor) | 电阻 (resistor) | 电感 (inductor) | 晶体管 (transistor) | 二极管 (diode) | 集成电路 (ic) | 终止与解析说明 |",
                "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
            baseline_slots = [s for s in cmp_data["slots"] if s.get("slot_role") != "separate_qwen3_enhancement"]
            addon_slots = [s for s in cmp_data["slots"] if s.get("slot_role") == "separate_qwen3_enhancement"]

            def _row_for_slot(s: Mapping[str, Any]) -> str:
                out = s.get("outcome") or {}
                if s.get("queue_status") == "completed" and s.get("has_result"):
                    cc = s.get("class_counts") or {}
                    tv2 = s.get("termination_v2") or {}
                    term_txt = (
                        f"EOS={tv2.get('normal_eos', 0)}/{tv2.get('total_calls', 0)}, "
                        f"限长={tv2.get('length_limited_without_eos', 0)}, "
                        f"空数组={tv2.get('explicit_empty', 0)}"
                    )
                    return (
                        f"| `{s['model_display_name']}` | `{s['mode']}` | **{out.get('outcome_label')}** | "
                        f"{s.get('elapsed_seconds', 0.0)} | **{s.get('kept_predictions_count', 0)}** | "
                        f"{cc.get('capacitor', 0)} | {cc.get('resistor', 0)} | {cc.get('inductor', 0)} | "
                        f"{cc.get('transistor', 0)} | {cc.get('diode', 0)} | {cc.get('ic', 0)} | {term_txt} |"
                    )
                reason = s.get("error_message") or out.get("outcome_note") or "未完成有效推理"
                return (
                    f"| `{s['model_display_name']}` | `{s['mode']}` | **{out.get('outcome_label', s.get('queue_status'))}** | "
                    f"{s.get('elapsed_seconds', 0.0)} | —（未生成） | — | — | — | — | — | — | {reason} |"
                )

            for s in baseline_slots:
                lines.append(_row_for_slot(s))

            if addon_slots:
                lines.extend([
                    "",
                    "## 3. 单独冻结增强模式对照（非同配置 W 基线）",
                    "",
                    "| 模型名称 | 增强模式 | 运行/结果状态 | 耗时 (s) | 保留总框数 | 电容 | 电阻 | 电感 | 晶体管 | 二极管 | 集成电路 | 核验过滤与终止说明 |",
                    "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
                ])
                for s in addon_slots:
                    lines.append(_row_for_slot(s))

            lines.extend(["", "## 4. 逐模型原生预处理与导出文件引用"])
            for s in cmp_data["slots"]:
                out = s.get("outcome") or {}
                lines.append(f"### {s['model_display_name']} (`{s['model_key']}` · `{s['mode']}`)")
                lines.append(f"- **原生视觉预处理**：{s.get('visual_preprocess_note')}")
                lines.append(f"- **任务 ID**：`{s.get('job_id') or '未启动'}` | **状态**：{out.get('outcome_label')} — {out.get('outcome_note')}")
                exps = s.get("exports")
                if exps:
                    lines.append(f"- **单模型导出引用**：JSON `{exps['json']}` | 标注图 PNG `{exps['overlay_png']}` | 摘要 Markdown `{exps['summary_md']}`")
                else:
                    lines.append("- **单模型导出引用**：该模型未完成有效推理，不生成导出文件。")
                lines.append("")

            scored_slots = [
                s for s in cmp_data["slots"]
                if isinstance(s.get("post_hoc_eval"), Mapping) and s["post_hoc_eval"].get("gt_evaluated")
            ]
            if scored_slots:
                lines.extend([
                    "",
                    "## 5. 事后 COCO GT 评测汇总（IoU=0.50）",
                    "",
                    "| 模型名称 | TP/FP/FN | Precision | Recall | F1 | 预测/GT (差值) | GT 身份依据 |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
                ])
                for s in scored_slots:
                    phe = s["post_hoc_eval"]
                    m = phe.get("metrics") or {}
                    p_cnt = int(m.get("total_predictions", 0))
                    g_cnt = int(m.get("total_gt", 0))
                    diff = p_cnt - g_cnt
                    diff_s = f"+{diff}" if diff >= 0 else str(diff)
                    p_pct = f"{float(m.get('precision', 0.0)) * 100:.2f}%"
                    r_pct = f"{float(m.get('recall', 0.0)) * 100:.2f}%"
                    f1_pct = f"{float(m.get('f1', 0.0)) * 100:.2f}%"
                    lines.append(
                        f"| `{s['model_display_name']}` | {m.get('tp', 0)}/{m.get('fp', 0)}/{m.get('fn', 0)} | "
                        f"{p_pct} | {r_pct} | {f1_pct} | {p_cnt}/{g_cnt} ({diff_s}) | "
                        f"{phe.get('identity_basis') or '-'} ({phe.get('coco_file_name') or '-'}) |"
                    )

            blob = ("\n".join(lines) + "\n").encode("utf-8")
            return blob, "text/markdown; charset=utf-8", f"{comparison_id}_comparison.md"

        raise DemoError(f"不支持的对比报告导出格式：{kind!r}；仅支持 json 或 markdown。", status_code=400)
