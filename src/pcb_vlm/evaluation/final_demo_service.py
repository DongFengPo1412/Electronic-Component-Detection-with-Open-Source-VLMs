"""Final GT-free whole-board PCB component detection demo backend & research replay service.

Composes frozen repository primitives (`tiling`, `fics_replay`, `candidate_verifier`,
`fics_matrix`, `candidate_verifier_runner`, `evaluate`, and `gt_free_pipeline`) without
reading default ground truth during live inference and without hardcoding private paths.
"""

from __future__ import annotations

import hashlib
import io
import json
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
from pcb_vlm.evaluation.gt_free_pipeline import (
    compute_board_roi_and_tiles,
    export_verified_predictions_gt_free,
    load_jsonl_records,
    prepare_verifier_manifest_gt_free,
    reconstruct_board_predictions_gt_free,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import evaluate as repo_eval  # noqa: E402

MODEL_ORDER: Tuple[str, ...] = ("qwen2", "deepseek", "qwen3")
SUPPORTED_IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
TERMINAL_JOB_STATUSES: Tuple[str, ...] = ("completed", "failed", "cancelled", "incomplete")
FROZEN_DETECTION_PROMPT = (
    "Locate all visible electronic components belonging to these classes: "
    "capacitor, resistor, inductor, transistor, diode, ic. "
    "Return a JSON array of objects with keys bbox_2d and label. "
    "Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized to [0, 1000] relative to this input image. "
    "Use exactly one of the listed class names for each object. "
    "If no target components are visible, return []. Do not include explanations."
)
FROZEN_DETECTION_PROMPT_SHA256 = hashlib.sha256(FROZEN_DETECTION_PROMPT.encode("utf-8")).hexdigest()
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "qwen2": "Qwen2-VL-2B-Instruct",
    "deepseek": "deepseek-vl-1.3b-chat",
    "qwen3": "Qwen3-VL-2B-Instruct",
}
MODE_DISPLAY_NAMES: Dict[str, str] = {
    "W": "W（整 ROI 推理）",
    "T": "T（切片推理 + NMS）",
    "T_B_other": "T + B_other（切片+背景核验 · 冻结方法）",
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
    if m in ("W", "T", "T_B_other"):
        return m
    if m in ("T+B_other", "T + B_other", "B_other"):
        return "T_B_other"
    raise DemoError(f"不支持的检测模式：{mode!r}；仅支持 W、T 或 T_B_other。", status_code=400)


def validate_model_and_mode(model_key: str, mode: str) -> Tuple[str, str]:
    mk = str(model_key or "").strip().lower()
    if mk not in MODEL_ORDER:
        raise DemoError(f"不支持的模型：{model_key!r}；仅支持 qwen2、deepseek、qwen3。", status_code=400)
    norm_mode = normalize_mode(mode)
    if norm_mode == "T_B_other" and mk != "qwen3":
        raise DemoError(
            "冻结二次核验方法 T + B_other 仅适用于 Qwen3-VL-2B-Instruct，不支持在 qwen2 或 deepseek 上启用。",
            status_code=400,
        )
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
        os.replace(tmp, path)
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
                        cand_p = pathlib.Path(str(r.get("image_path") or ""))
                        boards[bid]["roi_image_path"] = str(cand_p) if cand_p.is_file() else r.get("image_path")
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
                        "roi_image_available": bool(b["roi_image_path"] and pathlib.Path(b["roi_image_path"]).is_file()),
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
                    "image_available": bool(b["roi_image_path"] and pathlib.Path(b["roi_image_path"]).is_file()),
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
                "image_available": bool(b["roi_image_path"] and pathlib.Path(b["roi_image_path"]).is_file()),
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
        if not img_str or not pathlib.Path(img_str).is_file():
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
        python_executable: Optional[str] = None,
    ) -> None:
        self.workspace_root = pathlib.Path(workspace_root).resolve()
        self.uploads_dir = self.workspace_root / "uploads"
        self.jobs_dir = self.workspace_root / "jobs"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

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

        self._lock = threading.Lock()
        self._job_state_locks_guard = threading.Lock()
        self._job_state_locks: Dict[str, threading.Lock] = {}
        self._active_job_id: Optional[str] = None
        self._cancel_events: Dict[str, threading.Event] = {}
        self._active_procs: Dict[str, subprocess.Popen[Any]] = {}
        self._job_threads: Dict[str, threading.Thread] = {}
        self._preview_cache: Dict[Tuple[str, int, Optional[str]], bytes] = {}

    def shutdown(self) -> None:
        """Cancel any active GPU job and terminate its subprocess tree before server exit."""
        with self._lock:
            active_ids = list(self._cancel_events.keys())
        for jid in active_ids:
            self.cancel_job(jid)
        with self._lock:
            threads = list(self._job_threads.values())
        for th in threads:
            if th.is_alive():
                th.join(timeout=8.0)

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

    def register_upload(self, *, raw_bytes: bytes, raw_filename: str) -> Dict[str, Any]:
        """Validate uploaded board image, compute GT-free ROI/tiling pre-flight, and persist inputs."""
        if not raw_bytes:
            raise DemoError("上传文件内容为空，请选择有效的 PCB 图像文件。", status_code=400)

        safe_name = _safe_filename(raw_filename)
        upload_id = f"up_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        udir = self.uploads_dir / upload_id
        udir.mkdir(parents=True, exist_ok=True)
        src_path = udir / safe_name
        src_path.write_bytes(raw_bytes)

        try:
            with Image.open(src_path) as im:
                im_rgb = ImageOps.exif_transpose(im).convert("RGB")
                _ = im_rgb.size
        except Exception as exc:
            shutil.rmtree(udir, ignore_errors=True)
            raise DemoError(f"无法将文件解析为有效图像：{exc}", status_code=400) from exc

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
        return {
            "upload_id": meta["upload_id"],
            "safe_filename": meta["safe_filename"],
            "board_id": meta["board_id"],
            "source_image_sha256": meta["source_image_sha256"],
            "orig_wh": meta["orig_wh"],
            "roi_bbox_half_open": meta["roi_bbox_half_open"],
            "roi_offset_x": meta["roi_offset_x"],
            "roi_offset_y": meta["roi_offset_y"],
            "roi_width": meta["roi_width"],
            "roi_height": meta["roi_height"],
            "tile_count": meta["tile_count"],
            "gt_required": False,
        }

    def render_upload_roi_jpeg(self, *, upload_id: str, max_width: int = 1600, crop: Optional[str] = None) -> bytes:
        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))
        roi_p = pathlib.Path(meta["roi_export"]["path"])
        if not roi_p.is_file():
            raise DemoError("ROI 图像文件缺失。", status_code=404)
        key = (upload_id, int(max_width), crop)
        if key in self._preview_cache:
            return self._preview_cache[key]
        with Image.open(roi_p) as im:
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
        self._preview_cache[key] = data
        return data

    def _build_single_job_plan(
        self,
        *,
        job_id: str,
        meta: Mapping[str, Any],
        model_key: str,
        mode: str,
    ) -> Dict[str, Any]:
        """Construct a clean single-board execution plan matching frozen repository protocol."""
        base_mode = "T" if mode == "T_B_other" else mode
        rt_models = self.runtime_config.get("models") or {}
        if self._custom_job_runner is None:
            if model_key not in rt_models or not rt_models[model_key].get("local_dir"):
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
                spec["revision"] = "89644892e4d85e24eaac8bacfd4f463576704203"
                spec["overlay_dir"] = str(rt.get("overlay_dir") or "")
                spec["require_overlay"] = bool(rt.get("require_overlay", True))
                spec["visual_budget"] = {"shortest_edge": 65536, "longest_edge": 200704}
            elif mk == "qwen2":
                spec["visual_budget"] = {"min_pixels": 65536, "max_pixels": 200704}
            else:
                spec["visual_budget"] = {"native_square_size": 384, "padding_color": [127, 127, 127], "max_pixels_applies": False}
            model_specs[mk] = spec

        board_id = str(meta["board_id"])
        queue_items: List[Dict[str, Any]] = []
        if base_mode == "W":
            roi = meta["roi_export"]
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
                "call_id": f"{model_key}_demo_{board_id}_W",
            })
        else:
            for idx, tile in enumerate(meta["tiles"]):
                tid = int(tile["tile_id"])
                queue_items.append({
                    "board_id": board_id,
                    "tag": f"T{tid:02d}",
                    "mode": "T",
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
                    "call_id": f"{model_key}_demo_{board_id}_T{tid:02d}",
                })

        gate_cfg = self.runtime_config.get("gpu_gate") or {}
        return {
            "schema_version": "fics_demo_job_plan_v1",
            "job_id": job_id,
            "created_at": _now_iso(),
            "model_execution_order": [model_key],
            "models": model_specs,
            "inference_config": {
                "dtype": "bfloat16",
                "batch_size": 1,
                "do_sample": False,
                "repetition_penalty": 1.0,
                "max_new_tokens": 512,
                "qwen_min_pixels": 65536,
                "qwen_max_pixels": 200704,
                "deepseek_image_size": 384,
                "coordinate_protocol": "xyxy_norm1000",
                "prompt": FROZEN_DETECTION_PROMPT,
                "prompt_sha256": FROZEN_DETECTION_PROMPT_SHA256,
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
                "per_model_gpu_timeout_seconds": 2400.0,
                "total_gpu_timeout_seconds": 3600.0,
            },
            "boards": {
                board_id: {
                    "board_id": board_id,
                    "image_id": 1,
                    "orig_wh": meta["orig_wh"],
                    "roi_offset_x": meta["roi_offset_x"],
                    "roi_offset_y": meta["roi_offset_y"],
                    "roi_width": meta["roi_width"],
                    "roi_height": meta["roi_height"],
                    "tile_count": meta["tile_count"],
                }
            },
            "queues": {model_key: queue_items},
        }

    def submit_job(self, *, upload_id: str, model_key: str, mode: str) -> Dict[str, Any]:
        """Acquire single-GPU-job lock, persist GT-free plan, and launch background execution."""
        mk, norm_mode = validate_model_and_mode(model_key, mode)
        udir = self._upload_dir(upload_id)
        meta = json.loads((udir / "upload_meta.json").read_text(encoding="utf-8"))

        with self._lock:
            if self._active_job_id is not None:
                raise BusyGpuJobError(
                    f"已有 GPU 任务正在运行 ({self._active_job_id})，请等待其完成或点击取消后再发起新任务。",
                    active_job_id=self._active_job_id,
                )
            job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            jdir = self.jobs_dir / job_id
            jdir.mkdir(parents=True, exist_ok=True)
            plan = self._build_single_job_plan(job_id=job_id, meta=meta, model_key=mk, mode=norm_mode)
            _write_json_atomic(jdir / "plan.json", plan)

            base_mode = "T" if norm_mode == "T_B_other" else norm_mode
            expected_base_calls = len(plan["queues"][mk])
            job_spec = {
                "job_id": job_id,
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
                "tile_count": meta["tile_count"],
                "model_key": mk,
                "model_display_name": MODEL_DISPLAY_NAMES[mk],
                "mode": norm_mode,
                "mode_display_name": MODE_DISPLAY_NAMES[norm_mode],
                "base_mode": base_mode,
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
            )
            calls_detail = self._summarize_raw_calls(raw_records, mk=mk, base_mode=base_mode)

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
                preds = list(recon["predictions"])
                result_payload = self._build_final_result_payload(
                    job_spec=job_spec,
                    plan=plan,
                    recon=recon,
                    kept_predictions=preds,
                    pre_verifier_candidates=None,
                    verifier_summary=None,
                    calls_detail=calls_detail,
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
            self._update_state(
                job_id,
                status="preparing_verifier",
                stage_label=f"正在裁剪候选框准备二次核验（共 {len(recon['predictions'])} 个 T 候选）",
                base_calls_completed=len(raw_records),
                verifier_calls_planned=len(recon["predictions"]),
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
                reconstructed_by_board=[recon],
                board_image_paths={board_id: pathlib.Path(job_spec["source_image_path"])},
                crops_dir=jdir / "verifier_crops",
                inference_spec=verifier_spec,
                source_provenance={
                    "job_id": job_id,
                    "board_id": board_id,
                    "model_key": mk,
                    "mode": "T",
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
                    stage_label=f"正在执行冻结二次核验 (B_other · {unique_calls} 个独立候选裁剪)",
                )
                if self._custom_verifier_runner is not None:
                    self._custom_verifier_runner(job_spec, manifest_path, cancel_ev, self)
                elif self._custom_job_runner is None:
                    v_cmd = [
                        self.python_executable,
                        str(_SCRIPTS_ROOT / "run_candidate_verifier.py"),
                        "--manifest", str(manifest_path),
                        "--output-dir", str(verifier_out_dir),
                        "--execute",
                    ]
                    rc_v = self._run_subprocess_cancellable(
                        job_id=job_id,
                        cmd=v_cmd,
                        log_path=jdir / "verifier_runner.log",
                        cancel_ev=cancel_ev,
                    )
                    if rc_v != 0:
                        raise RuntimeError(f"二次核验子进程退出码 {rc_v}")
                v_raw_p = verifier_out_dir / "verifier_raw_outputs.jsonl"
                if not v_raw_p.is_file():
                    raise RuntimeError("未找到二次核验输出 verifier_raw_outputs.jsonl。")
                verifier_rows = load_jsonl_records(v_raw_p)
            else:
                verifier_rows = []

            if cancel_ev.is_set():
                raise InterruptedError("任务已被用户取消。")

            exported = export_verified_predictions_gt_free(
                manifest=manifest,
                verifier_rows=verifier_rows,
                policy="B_other",
                output_path=jdir / "b_other_predictions_gt_free.json",
            )
            board_exp = exported["per_board"].get(board_id) or {"pred_record": {"objects": []}, "decisions": []}
            kept_objs = [
                {
                    "prediction_id": f"{board_id}_T_B_other_k{i:03d}",
                    "board_id": board_id,
                    "image_id": 1,
                    "label": o["label"],
                    "bbox_xyxy": o["bbox_xyxy"],
                    "score": float(o.get("score", 1.0)),
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
                    "verifier_label": d.get("strict_label"),
                    "kept": bool(d.get("retained")),
                    "decision_reason": d.get("reason"),
                }
                for d in board_exp.get("decisions") or []
            ]
            result_payload = self._build_final_result_payload(
                job_spec=job_spec,
                plan=plan,
                recon=recon,
                kept_predictions=kept_objs,
                pre_verifier_candidates=pre_candidates,
                verifier_summary=exported["summary"],
                calls_detail=calls_detail,
            )
            _write_json_atomic(jdir / "job_result.json", result_payload)
            self._update_state(
                job_id,
                status="completed",
                stage_label="切片推理与 B_other 冻结核验全部完成",
                finished_at_epoch=time.time(),
                verifier_calls_completed=len(verifier_rows),
                technical_complete=True,
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
    def _summarize_raw_calls(raw_records: Sequence[Mapping[str, Any]], *, mk: str, base_mode: str) -> Dict[str, Any]:
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
            v2 = classify_call_termination_and_parse_v2(
                raw_text=raw_txt,
                width=in_w,
                height=in_h,
                generated_tokens=r.get("generated_tokens"),
                max_new_tokens=int(r.get("max_new_tokens") or 512),
                generated_token_ids=r.get("generated_token_ids"),
                eos_token_ids=eos_ids,
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
                "generated_tokens": r.get("generated_tokens"),
                "gpu_generate_ms": r.get("gpu_generate_ms"),
                "cuda_max_reserved_mib": r.get("cuda_max_reserved_mib"),
                "status": v2["status"],
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
    ) -> Dict[str, Any]:
        return {
            "schema_version": "fics_demo_job_result_v1",
            "job_id": job_spec["job_id"],
            "upload_id": job_spec["upload_id"],
            "board_id": job_spec["board_id"],
            "safe_filename": job_spec["safe_filename"],
            "source_image_sha256": job_spec["source_image_sha256"],
            "model_key": job_spec["model_key"],
            "model_display_name": job_spec["model_display_name"],
            "mode": job_spec["mode"],
            "mode_display_name": job_spec["mode_display_name"],
            "score_semantics_note": "score=1.0 仅为离散坐标标签输出的固定占位值，不代表模型概率置信度。",
            "gt_loaded": False,
            "post_hoc_eval": None,
            "orig_wh": job_spec["orig_wh"],
            "roi": {
                "offset_x": job_spec["roi_offset_x"],
                "offset_y": job_spec["roi_offset_y"],
                "width": job_spec["roi_width"],
                "height": job_spec["roi_height"],
                "tile_count": job_spec["tile_count"],
                "image_path": job_spec["roi_image_path"],
            },
            "technical_complete": True,
            "raw_global_predictions": recon.get("raw_global_predictions", len(kept_predictions)),
            "post_nms_candidates_count": recon.get("post_nms_predictions", len(kept_predictions)),
            "kept_predictions_count": len(kept_predictions),
            "class_counts": _count_classes(kept_predictions),
            "pre_verifier_class_counts": _count_classes(pre_verifier_candidates) if pre_verifier_candidates is not None else None,
            "predictions": list(kept_predictions),
            "pre_verifier_candidates": list(pre_verifier_candidates) if pre_verifier_candidates is not None else None,
            "verifier_summary": dict(verifier_summary) if verifier_summary is not None else None,
            "termination_v2": calls_detail["termination_v2"],
            "calls": calls_detail["calls"],
            "inference_config_sha256": {
                "prompt_sha256": plan["inference_config"]["prompt_sha256"],
            },
            "completed_at": _now_iso(),
        }

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Request safe cancellation for an active job and kill its subprocess tree."""
        self._job_dir(job_id)
        with self._lock:
            ev = self._cancel_events.get(job_id)
            proc = self._active_procs.get(job_id)
        if ev is not None:
            ev.set()
            self._update_state(job_id, cancel_requested=True, stage_label="正在安全终止推理子进程...")
        if proc is not None:
            terminate_process_tree(proc)
        return self.get_job_status(job_id)

    def list_jobs(self, limit: int = 30) -> List[Dict[str, Any]]:
        """List persisted jobs newest-first without failing on partial directories."""
        items: List[Dict[str, Any]] = []
        if not self.jobs_dir.is_dir():
            return items
        for d in sorted(self.jobs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not d.is_dir() or not (d / "job_state.json").is_file():
                continue
            try:
                items.append(self.get_job_status(d.name))
            except Exception:
                continue
            if len(items) >= int(limit):
                break
        return items

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        jdir = self._job_dir(job_id)
        spec = json.loads((jdir / "job_spec.json").read_text(encoding="utf-8"))
        state = json.loads((jdir / "job_state.json").read_text(encoding="utf-8"))
        mk = spec["model_key"]

        # Read real progress from jsonl files while active
        base_raw_p = jdir / "base_runs" / f"run_{mk}" / "raw_outputs_all.jsonl"
        ver_raw_p = jdir / "verifier_runs" / "verifier_raw_outputs.jsonl"
        base_done = len(load_jsonl_records(base_raw_p)) if base_raw_p.is_file() else int(state.get("base_calls_completed", 0))
        ver_done = len(load_jsonl_records(ver_raw_p)) if ver_raw_p.is_file() else int(state.get("verifier_calls_completed", 0))

        base_plan = int(state.get("base_calls_planned") or spec["expected_base_calls"] or 0)
        ver_plan = int(state.get("verifier_calls_planned") or 0)
        if ver_plan == 0 and (jdir / "verifier_manifest.json").is_file():
            try:
                man = json.loads((jdir / "verifier_manifest.json").read_text(encoding="utf-8"))
                ver_plan = len(man.get("unique_calls") or [])
            except Exception:
                pass

        end_t = state.get("finished_at_epoch") or time.time()
        elapsed_s = round(max(0.0, float(end_t) - float(state["started_at_epoch"])), 1)
        total_plan = base_plan + ver_plan
        total_done = min(total_plan, base_done + ver_done)
        eta_s: Optional[float] = None
        if state["status"] not in ("completed", "failed", "cancelled", "incomplete") and total_done > 0 and total_plan > total_done:
            eta_s = round((elapsed_s / total_done) * (total_plan - total_done), 1)

        gate_p = jdir / "base_runs" / f"run_{mk}" / "gpu_gate.json"
        gate_info = json.loads(gate_p.read_text(encoding="utf-8")) if gate_p.is_file() else None

        return {
            "job_id": job_id,
            "upload_id": spec["upload_id"],
            "board_id": spec["board_id"],
            "model_key": mk,
            "model_display_name": spec["model_display_name"],
            "mode": spec["mode"],
            "mode_display_name": spec["mode_display_name"],
            "status": state["status"],
            "stage_label": state["stage_label"],
            "cancel_requested": bool(state.get("cancel_requested")),
            "technical_complete": state.get("technical_complete"),
            "error_message": state.get("error_message"),
            "elapsed_seconds": elapsed_s,
            "eta_seconds": eta_s,
            "base_calls_planned": base_plan,
            "base_calls_completed": base_done,
            "verifier_calls_planned": ver_plan,
            "verifier_calls_completed": ver_done,
            "gpu_gate": gate_info,
            "has_result": (jdir / "job_result.json").is_file(),
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

        kept_eval = _match_preds_with_gt(gts, res["predictions"], iou_thresh=0.50)
        pre_eval = (
            _match_preds_with_gt(gts, res["pre_verifier_candidates"], iou_thresh=0.50)
            if res.get("pre_verifier_candidates") is not None else None
        )
        eval_bundle = {
            "gt_evaluated": True,
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
            "gts": kept_eval["gts"],
            "predictions_matched": kept_eval["predictions"],
            "pre_verifier_metrics": pre_eval["metrics"] if pre_eval else None,
            "evaluated_at": _now_iso(),
        }
        res["gt_loaded"] = True
        res["post_hoc_eval"] = eval_bundle
        _write_json_atomic(jdir / "job_result.json", res)
        return res

    def export_job(self, *, job_id: str, kind: str, view: str = "kept") -> Tuple[bytes, str, str]:
        """Generate downloadable standard JSON, annotated PNG overlay, or Markdown summary."""
        res = self.get_job_result(job_id)
        k = str(kind or "").strip().lower()
        if k == "json":
            data = json.dumps(res, indent=2, ensure_ascii=False).encode("utf-8")
            return data, "application/json; charset=utf-8", f"{job_id}_detections.json"
        if k == "overlay_png":
            roi_p = pathlib.Path(res["roi"]["image_path"])
            if not roi_p.is_file():
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
            lines = [
                f"# FICS-PCB 整板检测结果摘要 ({job_id})",
                "",
                f"- 上传文件名：`{res['safe_filename']}`",
                f"- 原图 SHA-256：`{res['source_image_sha256']}`",
                f"- 原图尺寸：`{res['orig_wh'][0]}x{res['orig_wh'][1]}` | ROI 偏移：`({res['roi']['offset_x']}, {res['roi']['offset_y']})` | ROI 尺寸：`{res['roi']['width']}x{res['roi']['height']}` | 切片数：`{res['roi']['tile_count']}`",
                f"- 模型：`{res['model_display_name']}` (`{res['model_key']}`)",
                f"- 模式：`{res['mode_display_name']}` (`{res['mode']}`)",
                f"- 技术完整性：`{res['technical_complete']}`",
                f"- 保留预测总数：**{res['kept_predictions_count']}**",
                "",
                "## 各类别检出计数（6 类）",
            ]
            for cls in FICS_SIX_CLASSES:
                lines.append(f"- `{cls}`: {res['class_counts'].get(cls, 0)}")
            if res.get("verifier_summary"):
                vs = res["verifier_summary"]
                lines.extend([
                    "",
                    "## 二次核验 (B_other) 过滤统计",
                    f"- 候选总数 (T Post-NMS)：{vs['total_candidates']}",
                    f"- 核验保留数：{vs['retained_count']}",
                    f"- 识别为 `other` 剔除：{vs['rejected_as_other']}",
                    f"- 异常/缺失过滤：{vs['format_failure_filtered'] + vs['technical_failure_filtered']}",
                ])
            lines.extend(["", "## 事后 GT 评价状态"])
            if res.get("post_hoc_eval"):
                phe = res["post_hoc_eval"]
                m = phe["metrics"]
                id_verified = bool(phe.get("image_identity_verified"))
                r_disp = m.get("recall_display") or ("N/A (GT=0)" if m.get("recall") is None else f"{m['recall']*100:.2f}%")
                f1_disp = m.get("f1_display") or ("N/A (GT=0)" if m.get("f1") is None else f"{m['f1']*100:.2f}%")
                lines.append(
                    f"- 已对齐事后 COCO GT (image_id={phe['coco_image_id']}, file_name={phe.get('coco_file_name')}, IoU>=0.50)："
                    f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, P={m['precision']*100:.2f}%, R={r_disp}, F1={f1_disp}"
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
                lines.append("- **未评价**（本次运行未载入 Ground Truth；全程严格遵循 GT-free 流程）")
            text = ("\n".join(lines) + "\n").encode("utf-8")
            return text, "text/markdown; charset=utf-8", f"{job_id}_summary.md"
        raise DemoError(f"不支持的导出格式：{kind!r}", status_code=400)
