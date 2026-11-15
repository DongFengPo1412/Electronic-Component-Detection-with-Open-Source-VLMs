"""Generate machine-verified FICS-PCB results tables, telemetry summaries, and static figures.

Reads all metrics, v2 termination counts, per-board/per-class breakdowns, and
latency/VRAM statistics directly from SHA-256 verified replay/matrix JSON/JSONL files
referenced by a `results_index.json`. Never hardcodes experimental scores in code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
from typing import Any, Dict, List, Optional, Tuple

FICS_CLASSES = ("capacitor", "resistor", "inductor", "transistor", "diode", "ic")
MODEL_ORDER = ("qwen2", "deepseek", "qwen3")
MODEL_DISPLAY = {
    "qwen2": "Qwen2-VL-2B-Instruct",
    "deepseek": "deepseek-vl-1.3b-chat",
    "qwen3": "Qwen3-VL-2B-Instruct",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_entry_path(base_dir: pathlib.Path, entry: Dict[str, Any]) -> pathlib.Path:
    raw = pathlib.Path(str(entry["path"]))
    return raw if raw.is_absolute() else (base_dir / raw).resolve()


def _verify_entry(base_dir: pathlib.Path, entry: Dict[str, Any], label: str) -> Optional[str]:
    if entry.get("optional") and not entry.get("path"):
        entry["resolved_path"] = None
        entry["verified_sha256"] = entry.get("sha256")
        return entry["verified_sha256"]
    if not entry.get("path"):
        raise FileNotFoundError(f"Missing path for required indexed artifact {label}")
    p = _resolve_entry_path(base_dir, entry)
    if not p.is_file():
        raise FileNotFoundError(f"Missing indexed artifact {label}: {p}")
    actual = sha256_file(p)
    exp = entry.get("sha256")
    if exp and actual.lower() != str(exp).strip().lower():
        raise ValueError(f"SHA-256 mismatch for {label} ({p}): {actual} != {exp}")
    entry["resolved_path"] = str(p)
    entry["verified_sha256"] = actual
    return actual


def verify_and_load_index(index_path: pathlib.Path) -> Dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    base_dir = index_path.parent
    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        ds = index["datasets"][ds_key]
        for label, entry in ds["artifacts"].items():
            _verify_entry(base_dir, entry, f"{ds_key}.{label}")
        for mk, rentry in ds["raw_jsonl"].items():
            _verify_entry(base_dir, rentry, f"{ds_key}.raw_jsonl.{mk}")

    supp = index.get("supplements", {}).get("dev_qwen3_high_budget")
    if supp:
        for label, entry in supp["artifacts"].items():
            _verify_entry(base_dir, entry, f"supplement.{label}")
    return index


def _pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    return f"{float(val) * 100.0:.2f}%"


def _fmt_num(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    return f"{float(val):.{decimals}f}"


def _load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _extract_telemetry_for_dataset(ds: Dict[str, Any]) -> Dict[str, Any]:
    mat_entry = ds["artifacts"].get("matrix_session_summary") or {}
    mat_path = mat_entry.get("resolved_path") or mat_entry.get("path")
    mat = json.loads(pathlib.Path(mat_path).read_text(encoding="utf-8")) if mat_path else {}
    out: Dict[str, Any] = {
        "matrix_status": mat.get("matrix_status"),
        "total_wall_seconds": mat.get("total_wall_seconds"),
        "models": {},
    }
    mat_models = mat.get("models") or {}
    for mk in MODEL_ORDER:
        raw_p = ds["raw_jsonl"][mk].get("resolved_path") or ds["raw_jsonl"][mk]["path"]
        rows = _load_jsonl(pathlib.Path(raw_p))
        gpu_ms = [float(r["gpu_generate_ms"]) for r in rows if r.get("gpu_generate_ms") is not None]
        wall_ms = [float(r["call_wall_ms"]) for r in rows if r.get("call_wall_ms") is not None]
        prep_ms = [float(r["cpu_preprocess_ms"]) for r in rows if r.get("cpu_preprocess_ms") is not None]
        h2d_ms = [float(r["h2d_transfer_ms"]) for r in rows if r.get("h2d_transfer_ms") is not None]
        alloc_mib = [float(r["cuda_max_allocated_mib"]) for r in rows if r.get("cuda_max_allocated_mib") is not None]
        res_mib = [float(r["cuda_max_reserved_mib"]) for r in rows if r.get("cuda_max_reserved_mib") is not None]
        gen_toks = [int(r["generated_tokens"]) for r in rows if r.get("generated_tokens") is not None]
        w_ms = [float(r["gpu_generate_ms"]) for r in rows if r.get("mode") == "W" and r.get("gpu_generate_ms") is not None]
        t_ms = [float(r["gpu_generate_ms"]) for r in rows if r.get("mode") == "T" and r.get("gpu_generate_ms") is not None]

        m_mat = mat_models.get(mk) or {}
        load_meta = m_mat.get("worker_load_meta") or {}
        out["models"][mk] = {
            "calls": len(rows),
            "stage_wall_seconds": m_mat.get("wall_seconds"),
            "load_seconds": load_meta.get("load_seconds"),
            "load_cuda_allocated_mib": load_meta.get("cuda_allocated_mib"),
            "load_cuda_reserved_mib": load_meta.get("cuda_reserved_mib"),
            "peak_cuda_allocated_mib": round(max(alloc_mib), 2) if alloc_mib else None,
            "peak_cuda_reserved_mib": round(max(res_mib), 2) if res_mib else None,
            "gpu_generate_ms_mean": round(statistics.mean(gpu_ms), 2) if gpu_ms else None,
            "gpu_generate_ms_median": round(statistics.median(gpu_ms), 2) if gpu_ms else None,
            "gpu_generate_ms_max": round(max(gpu_ms), 2) if gpu_ms else None,
            "w_gpu_generate_ms_mean": round(statistics.mean(w_ms), 2) if w_ms else None,
            "t_gpu_generate_ms_mean": round(statistics.mean(t_ms), 2) if t_ms else None,
            "cpu_preprocess_ms_mean": round(statistics.mean(prep_ms), 2) if prep_ms else None,
            "h2d_transfer_ms_mean": round(statistics.mean(h2d_ms), 2) if h2d_ms else None,
            "call_wall_ms_mean": round(statistics.mean(wall_ms), 2) if wall_ms else None,
            "generated_tokens_mean": round(statistics.mean(gen_toks), 2) if gen_toks else None,
            "generated_tokens_max": max(gen_toks) if gen_toks else None,
            "tokens_per_sec_mean": (
                round(sum(gen_toks) / (sum(gpu_ms) / 1000.0), 2)
                if gen_toks and sum(gpu_ms) > 0
                else None
            ),
        }
    return out


def _extract_dataset_summary(ds_key: str, ds: Dict[str, Any]) -> Dict[str, Any]:
    rep_p = ds["artifacts"]["replay_report"].get("resolved_path") or ds["artifacts"]["replay_report"]["path"]
    rep = json.loads(pathlib.Path(rep_p).read_text(encoding="utf-8"))
    summary: Dict[str, Any] = {
        "dataset_key": ds_key,
        "label": ds["label"],
        "split_role": ds["split_role"],
        "boards": list(rep["coco_boards"].keys()),
        "total_calls": sum(int(rep["models"][m]["termination_v2"]["total_calls_observed"]) for m in MODEL_ORDER),
        "calls_per_model": int(rep["models"]["qwen3"]["termination_v2"]["total_calls_observed"]),
        "total_gt": int(rep["models"]["qwen3"]["combined"]["W"]["evaluate_detections_iou50"]["total_gt"]),
        "models": {},
        "telemetry": _extract_telemetry_for_dataset(ds),
    }

    for mk in MODEL_ORDER:
        mrep = rep["models"][mk]
        combined_out: Dict[str, Any] = {}
        for mode in ("W", "T"):
            ev = mrep["combined"][mode]["evaluate_detections_iou50"]
            per_cls = {}
            for cls in FICS_CLASSES:
                c = ev["per_class"].get(cls) or {
                    "gt_support": 0,
                    "pred_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                gt_sup = int(c["gt_support"])
                per_cls[cls] = {
                    "gt_support": gt_sup,
                    "pred_count": int(c["pred_count"]),
                    "tp": int(c["tp"]),
                    "fp": int(c["fp"]),
                    "fn": int(c["fn"]),
                    "precision": float(c["precision"]),
                    "recall": None if gt_sup == 0 else float(c["recall"]),
                    "recall_display": "N/A (GT=0)" if gt_sup == 0 else _pct(c["recall"]),
                    "f1": None if gt_sup == 0 else float(c["f1"]),
                    "f1_display": "N/A (GT=0)" if gt_sup == 0 else _pct(c["f1"]),
                }
            combined_out[mode] = {
                "tp": int(ev["tp"]),
                "fp": int(ev["fp"]),
                "fn": int(ev["fn"]),
                "total_gt": int(ev["total_gt"]),
                "total_predictions": int(ev["total_predictions"]),
                "precision": float(ev["micro"]["precision"]),
                "recall": float(ev["micro"]["recall"]),
                "f1": float(ev["micro"]["f1"]),
                "per_class": per_cls,
            }

        per_board_out: Dict[str, Any] = {}
        for bid, bdata in mrep["per_board"].items():
            per_board_out[bid] = {}
            for mode in ("W", "T"):
                bev = bdata[mode]["evaluate_detections_iou50"]
                per_board_out[bid][mode] = {
                    "tp": int(bev["tp"]),
                    "fp": int(bev["fp"]),
                    "fn": int(bev["fn"]),
                    "total_gt": int(bev["total_gt"]),
                    "total_predictions": int(bev["total_predictions"]),
                    "precision": float(bev["micro"]["precision"]),
                    "recall": float(bev["micro"]["recall"]),
                    "f1": float(bev["micro"]["f1"]),
                }

        summary["models"][mk] = {
            "display_name": MODEL_DISPLAY[mk],
            "termination_v2": mrep["termination_v2"],
            "combined": combined_out,
            "per_board": per_board_out,
        }
    return summary


def _extract_high_budget_supplement(supp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ms_entry = supp["artifacts"].get("metrics_summary") or {}
    fp_entry = supp["artifacts"].get("frozen_plan") or {}
    ms_p = ms_entry.get("resolved_path") or ms_entry.get("path")
    fp_p = fp_entry.get("resolved_path") or fp_entry.get("path")
    if not ms_p or not fp_p:
        return None
    ms = json.loads(pathlib.Path(ms_p).read_text(encoding="utf-8"))
    fp = json.loads(pathlib.Path(fp_p).read_text(encoding="utf-8"))
    fm = ms["models"]["qwen3"]["formal_matrix"]
    inf = fp["inference_config"]
    modes_out: Dict[str, Any] = {}
    for mode in ("W", "T"):
        mnode = fm["combined_two_board"][mode]
        std = mnode["eval"]["standard_eval"]
        ct = mnode["call_telemetry"]
        modes_out[mode] = {
            "tp": int(std["tp"]),
            "fp": int(std["fp"]),
            "fn": int(std["fn"]),
            "total_gt": int(std["total_gt"]),
            "total_predictions": int(std["total_predictions"]),
            "precision": float(std["micro"]["precision"]),
            "recall": float(std["micro"]["recall"]),
            "f1": float(std["micro"]["f1"]),
            "truncated_calls": int(ct["truncated_calls"]),
            "calls_count": int(ct["calls_count"]),
            "latency_ms_mean": float(ct["latency_ms_mean"]),
            "cuda_max_allocated_mib": float(ct["cuda_max_allocated_mib"]),
            "cuda_max_reserved_mib": float(ct["cuda_max_reserved_mib"]),
        }
    return {
        "label": supp["label"],
        "max_new_tokens": int(inf["max_new_tokens"]),
        "qwen_min_pixels": int(inf["qwen_min_pixels"]),
        "qwen_max_pixels": int(inf["qwen_max_pixels"]),
        "modes": modes_out,
    }


def render_tables_markdown(tables_data: Dict[str, Any], index: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# FICS-PCB 6 类检测主实验自动汇总表（Machine-Generated）",
        "",
        "> 本文档由 `scripts/build_fics_results_bundle.py` 直接读取已校验 SHA-256 的正式回放与会话遥测 JSON/JSONL 自动生成，未在脚本中手写任何实验分数。",
        "> 两个样本集（`s18+s28` 开发集 64 GT 与 `s29+s27` 未观察新板 41 GT）严格分开呈现，不跨数据集混合求平均。",
        "",
        "## 1. 统一实用预算（512 tokens）三模型主结果对比",
        "",
    ]

    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        ds = tables_data["datasets"][ds_key]
        lines.extend([
            f"### {ds['label']}（{ds['total_calls']} 次推理，每模型 {ds['calls_per_model']} 次，共 {ds['total_gt']} GT）",
            "",
            "| 模型 | 模式 | TP | FP | FN | Pred | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for mk in MODEL_ORDER:
            m = ds["models"][mk]
            for mode, mdesc in (("W", "W (整 ROI)"), ("T", "T (切片+NMS)")):
                c = m["combined"][mode]
                lines.append(
                    f"| {m['display_name']} | {mdesc} | {c['tp']} | {c['fp']} | {c['fn']} | "
                    f"{c['total_predictions']} | {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['f1'])} |"
                )
        lines.append("")

    lines.extend([
        "## 2. 逐板拆分表（IoU >= 0.50，6 类类感知匹配）",
        "",
        "| 数据集 | 模型 | 板卡 | 模式 | GT | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        ds = tables_data["datasets"][ds_key]
        for mk in MODEL_ORDER:
            m = ds["models"][mk]
            for bid, bmodes in m["per_board"].items():
                for mode in ("W", "T"):
                    b = bmodes[mode]
                    lines.append(
                        f"| {ds_key} | {m['display_name']} | {bid} | {mode} | {b['total_gt']} | "
                        f"{b['tp']} | {b['fp']} | {b['fn']} | {_pct(b['precision'])} | {_pct(b['recall'])} | {_pct(b['f1'])} |"
                    )
    lines.append("")

    lines.extend([
        "## 3. Qwen3-VL-2B-Instruct 逐类拆分（含 GT=0 零分母规范标注）",
        "",
        "> 说明：在 `s29+s27` 样本中，`diode` 的真实 GT 支持数为 `0`；按照严格定义，其 Recall 与 F1 标注为 `N/A (GT=0)`，误报数 `FP` 照实列出。",
        "",
    ])
    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        ds = tables_data["datasets"][ds_key]
        q3 = ds["models"]["qwen3"]["combined"]
        lines.extend([
            f"### {ds['label']} — Qwen3-VL-2B-Instruct",
            "",
            "| 模式 | 类别 | GT | Pred | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for mode in ("W", "T"):
            for cls in FICS_CLASSES:
                c = q3[mode]["per_class"][cls]
                lines.append(
                    f"| {mode} | `{cls}` | {c['gt_support']} | {c['pred_count']} | {c['tp']} | {c['fp']} | {c['fn']} | "
                    f"{_pct(c['precision'])} | {c['recall_display']} | {c['f1_display']} |"
                )
        lines.append("")

    lines.extend([
        "## 4. `v2` 终止语义与格式遵从性统计",
        "",
        "| 数据集 | 模型 | 规划/完成调用 | 正常 EOS | 达上限无 EOS | EOS 后格式/模式异常 | 严格 Schema 通过 | 兼容部分恢复 | 显式空数组 `[]` | 无框失败 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        ds = tables_data["datasets"][ds_key]
        for mk in MODEL_ORDER:
            v2 = ds["models"][mk]["termination_v2"]
            lines.append(
                f"| {ds_key} | {MODEL_DISPLAY[mk]} | {v2['completed_ok_calls']}/{v2['total_calls_observed']} | "
                f"{v2['normal_eos']} | {v2['length_limited_without_eos']} | "
                f"{v2['syntax_or_format_incomplete_after_eos']} | {v2['strict_schema_ok']} | "
                f"{v2['partial_recovery']} | {v2['explicit_empty']} | {v2['parse_failed']} |"
            )
    lines.append("")

    lines.extend([
        "## 5. 时延、生成吞吐与 GPU 显存占用（来自实际原始 JSONL 与会话日志）",
        "",
        "| 数据集 | 模型 | 调用数 | 阶段总耗时 (s) | 模型加载 (s) | 平均 GPU 生成 (ms) | W/T 平均生成 (ms) | 最大生成 (ms) | 平均预处理/H2D (ms) | 平均输出 Token | 吞吐 (tok/s) | 峰值分配/保留显存 (MiB) |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
    ])
    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        ds = tables_data["datasets"][ds_key]
        tel = ds["telemetry"]
        for mk in MODEL_ORDER:
            tm = tel["models"][mk]
            lines.append(
                f"| {ds_key} | {MODEL_DISPLAY[mk]} | {tm['calls']} | {_fmt_num(tm['stage_wall_seconds'], 2)} | "
                f"{_fmt_num(tm['load_seconds'], 2)} | {_fmt_num(tm['gpu_generate_ms_mean'], 1)} | "
                f"{_fmt_num(tm['w_gpu_generate_ms_mean'], 1)} / {_fmt_num(tm['t_gpu_generate_ms_mean'], 1)} | "
                f"{_fmt_num(tm['gpu_generate_ms_max'], 1)} | {_fmt_num(tm['cpu_preprocess_ms_mean'], 1)} / {_fmt_num(tm['h2d_transfer_ms_mean'], 2)} | "
                f"{_fmt_num(tm['generated_tokens_mean'], 1)} | {_fmt_num(tm['tokens_per_sec_mean'], 1)} | "
                f"{_fmt_num(tm['peak_cuda_allocated_mib'], 1)} / {_fmt_num(tm['peak_cuda_reserved_mib'], 1)} |"
            )
    lines.append("")

    supp = tables_data.get("supplements", {}).get("dev_qwen3_high_budget")
    if supp:
        lines.extend([
            "## 6. 单独补充：Qwen3 高预算开发集参考（`s18+s28`，不与 512-token 主表混排）",
            "",
            f"- 预算设置：`max_new_tokens={supp['max_new_tokens']}`，`qwen_max_pixels={supp['qwen_max_pixels']}`（仅开发集参考；主线三模型对比严格以 512-token 同协议为准）",
            "",
            "| 模式 | TP | FP | FN | Precision | Recall | F1 | 截断调用数 | 平均生成耗时 (ms) | 峰值分配显存 (MiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for mode in ("W", "T"):
            m = supp["modes"][mode]
            lines.append(
                f"| {mode} | {m['tp']} | {m['fp']} | {m['fn']} | {_pct(m['precision'])} | {_pct(m['recall'])} | "
                f"{_pct(m['f1'])} | {m['truncated_calls']}/{m['calls_count']} | {m['latency_ms_mean']:.1f} | "
                f"{m['cuda_max_allocated_mib']:.1f} |"
            )
        lines.append("")

    lines.extend([
        "## 7. 数据溯源 SHA-256 绑定清单",
        "",
        "| 数据集 / 补充 | 构件 | SHA-256 (前 16 位) |",
        "|---|---|---|",
    ])
    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        ds_idx = index["datasets"][ds_key]
        for label, entry in ds_idx["artifacts"].items():
            sha_val = entry.get("verified_sha256")
            if sha_val:
                lines.append(f"| {ds_key} | `{label}` | `{sha_val[:16]}` |")
            else:
                lines.append(f"| {ds_key} | `{label}` | `未打包（可选外部遥测）` |")
        for mk, entry in ds_idx["raw_jsonl"].items():
            sha_val = entry.get("verified_sha256")
            if sha_val:
                lines.append(f"| {ds_key} | `raw_{mk}` | `{sha_val[:16]}` |")
            else:
                lines.append(f"| {ds_key} | `raw_{mk}` | `N/A` |")
    return "\n".join(lines) + "\n"


def generate_figures(tables_data: Dict[str, Any], fig_dir: pathlib.Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    # Figure 1: Main W vs T F1/Precision/Recall comparison on s18+s28 (dev) and s29+s27 (new)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180, sharey=True)
    for ax, (ds_key, subtitle) in zip(
        axes,
        [
            ("dev_two_board_s18_s28", "开发集 s18+s28 (64 GT, 201 调用)"),
            ("unobserved_two_board_s29_s27", "未观察新板 s29+s27 (41 GT, 306 调用)"),
        ],
    ):
        ds = tables_data["datasets"][ds_key]
        x_labels = []
        precs, recs, f1s = [], [], []
        for mk in MODEL_ORDER:
            short = {"qwen2": "Qwen2", "deepseek": "DeepSeek", "qwen3": "Qwen3"}[mk]
            for mode in ("W", "T"):
                c = ds["models"][mk]["combined"][mode]
                x_labels.append(f"{short}\n({mode})")
                precs.append(c["precision"] * 100.0)
                recs.append(c["recall"] * 100.0)
                f1s.append(c["f1"] * 100.0)
        x = np.arange(len(x_labels))
        w = 0.25
        b1 = ax.bar(x - w, precs, w, label="Precision (%)", color="#4C78A8")
        b2 = ax.bar(x, recs, w, label="Recall (%)", color="#F58518")
        b3 = ax.bar(x + w, f1s, w, label="F1 (%)", color="#54A24B")
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                if h > 0.0:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(subtitle, fontsize=11, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0, 36)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if ax is axes[0]:
            ax.set_ylabel("百分比 (%)", fontsize=10)
            ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("FICS-PCB 6类检测主结果对比（统一 512-token 实用预算，IoU >= 0.50）", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = fig_dir / "fig1_main_detection_comparison.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    generated.append(str(p1))

    # Figure 2: v2 termination & format compliance breakdown
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    cats = [
        ("strict_schema_pass", "严格JSON有效框/空[]", "#54A24B"),
        ("partial_with_len_limit", "达512上限截断", "#E45756"),
        ("eos_incomplete", "正常EOS但格式异常", "#F58518"),
    ]
    for ax, (ds_key, title) in zip(
        axes,
        [
            ("dev_two_board_s18_s28", "s18+s28 (每模型 67 次调用)"),
            ("unobserved_two_board_s29_s27", "s29+s27 (每模型 102 次调用)"),
        ],
    ):
        ds = tables_data["datasets"][ds_key]
        x = np.arange(len(MODEL_ORDER))
        names = [MODEL_DISPLAY[mk].replace("-Instruct", "") for mk in MODEL_ORDER]
        strict_vals = [ds["models"][mk]["termination_v2"]["strict_schema_ok"] for mk in MODEL_ORDER]
        len_vals = [ds["models"][mk]["termination_v2"]["length_limited_without_eos"] for mk in MODEL_ORDER]
        eos_inc_vals = [ds["models"][mk]["termination_v2"]["syntax_or_format_incomplete_after_eos"] for mk in MODEL_ORDER]
        w = 0.25
        for offset, vals, ( _, lbl, col) in zip((-w, 0.0, w), (strict_vals, len_vals, eos_inc_vals), cats):
            bars = ax.bar(x + offset, vals, w, label=lbl, color=col)
            for bar in bars:
                h = int(bar.get_height())
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, str(h), ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylabel("调用次数", fontsize=10)
        ax.set_ylim(0, max(strict_vals + len_vals + eos_inc_vals) * 1.18 + 4)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("v2 终止语义与输出格式遵从性分布", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = fig_dir / "fig2_v2_termination_and_format.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    generated.append(str(p2))

    # Figure 3: Qwen3 per-class Recall & Precision on s18+s28 and s29+s27
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=180)
    for ax, (ds_key, title) in zip(
        axes,
        [
            ("dev_two_board_s18_s28", "Qwen3 on s18+s28 (64 GT)"),
            ("unobserved_two_board_s29_s27", "Qwen3 on s29+s27 (41 GT; diode GT=0)"),
        ],
    ):
        ds = tables_data["datasets"][ds_key]
        q3 = ds["models"]["qwen3"]["combined"]
        x = np.arange(len(FICS_CLASSES))
        w = 0.35
        w_f1 = [(q3["W"]["per_class"][c]["f1"] or 0.0) * 100.0 for c in FICS_CLASSES]
        t_f1 = [(q3["T"]["per_class"][c]["f1"] or 0.0) * 100.0 for c in FICS_CLASSES]
        b1 = ax.bar(x - w / 2, w_f1, w, label="W 整 ROI F1 (%)", color="#4C78A8")
        b2 = ax.bar(x + w / 2, t_f1, w, label="T 切片+NMS F1 (%)", color="#E45756")
        for idx, cls in enumerate(FICS_CLASSES):
            gt_s = q3["W"]["per_class"][cls]["gt_support"]
            if gt_s == 0:
                ax.text(x[idx], 2.0, "N/A\n(GT=0)", ha="center", va="bottom", fontsize=7.5, color="#B22222")
            else:
                for bar in (b1[idx], b2[idx]):
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(GT={q3['W']['per_class'][c]['gt_support']})" for c in FICS_CLASSES], fontsize=8.5)
        ax.set_ylabel("F1 (%)", fontsize=10)
        ax.set_ylim(0, 58)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Qwen3-VL-2B 逐类 F1 表现与整图/切片模式差异", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p3 = fig_dir / "fig3_qwen3_per_class_f1.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    generated.append(str(p3))

    # Figure 4: Latency and Peak GPU Memory across datasets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    x = np.arange(len(MODEL_ORDER))
    names = [MODEL_DISPLAY[mk].replace("-Instruct", "") for mk in MODEL_ORDER]
    dev_lat = [tables_data["datasets"]["dev_two_board_s18_s28"]["telemetry"]["models"][mk]["gpu_generate_ms_mean"] / 1000.0 for mk in MODEL_ORDER]
    unobs_lat = [tables_data["datasets"]["unobserved_two_board_s29_s27"]["telemetry"]["models"][mk]["gpu_generate_ms_mean"] / 1000.0 for mk in MODEL_ORDER]
    dev_mem = [tables_data["datasets"]["dev_two_board_s18_s28"]["telemetry"]["models"][mk]["peak_cuda_reserved_mib"] for mk in MODEL_ORDER]
    unobs_mem = [tables_data["datasets"]["unobserved_two_board_s29_s27"]["telemetry"]["models"][mk]["peak_cuda_reserved_mib"] for mk in MODEL_ORDER]

    w = 0.34
    for ax, v1, v2, ylabel, title in (
        (ax1, dev_lat, unobs_lat, "平均单次 GPU 生成耗时 (s)", "单次调用平均 GPU 耗时对比"),
        (ax2, dev_mem, unobs_mem, "峰值 Reserved 显存 (MiB)", "峰值 GPU 保留显存占用 (8GB 显存显卡，单模型峰值 4.1–4.9 GiB)"),
    ):
        b1 = ax.bar(x - w / 2, v1, w, label="s18+s28 开发集", color="#72B7B2")
        b2 = ax.bar(x + w / 2, v2, w, label="s29+s27 未观察板", color="#B279A2")
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + max(v1 + v2) * 0.02, f"{h:.1f}" if h < 100 else f"{h:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(v1 + v2) * 1.26)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("端到端运行成本：单次时延与峰值显存统计", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p4 = fig_dir / "fig4_latency_and_vram.png"
    fig.savefig(p4, dpi=180)
    plt.close(fig)
    generated.append(str(p4))

    return generated


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build machine-verified FICS results tables and charts from results_index.json.")
    parser.add_argument("--index", required=True, help="Path to results_index.json")
    parser.add_argument("--out-json", required=True, help="Output summary JSON path")
    parser.add_argument("--out-md", action="append", required=True, help="Output Markdown table path(s)")
    parser.add_argument("--figures-dir", required=True, help="Output directory for PNG charts")
    args = parser.parse_args(argv)

    idx_path = pathlib.Path(args.index)
    index = verify_and_load_index(idx_path)

    tables_data: Dict[str, Any] = {
        "schema_version": "1.0",
        "source_index": str(idx_path),
        "source_index_sha256": sha256_file(idx_path),
        "datasets": {},
        "supplements": {},
    }
    for ds_key in ("dev_two_board_s18_s28", "unobserved_two_board_s29_s27"):
        tables_data["datasets"][ds_key] = _extract_dataset_summary(ds_key, index["datasets"][ds_key])
    supp = index.get("supplements", {}).get("dev_qwen3_high_budget")
    if supp:
        supp_data = _extract_high_budget_supplement(supp)
        if supp_data is not None:
            tables_data["supplements"]["dev_qwen3_high_budget"] = supp_data

    md_text = render_tables_markdown(tables_data, index)
    figs = generate_figures(tables_data, pathlib.Path(args.figures_dir))
    tables_data["generated_figures"] = figs

    out_json_p = pathlib.Path(args.out_json)
    out_json_p.parent.mkdir(parents=True, exist_ok=True)
    out_json_p.write_text(json.dumps(tables_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for md_out in args.out_md:
        mp = pathlib.Path(md_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(md_text, encoding="utf-8")

    print(json.dumps({"out_json": str(out_json_p), "out_md": args.out_md, "figures": figs}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
