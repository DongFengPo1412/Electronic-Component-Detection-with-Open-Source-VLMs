"""Minimal local HTTP replay showcase for verified FICS-PCB 6-class VLM runs.

Serves read-only historical replay results (`results_index.json` -> `replay_report`,
`gt_coco`, `frozen_plan`, `raw_jsonl`) with strict dataset-scope (`s18+s28` dev vs
`s29+s27` unobserved) isolation, GT/TP/FP/FN overlays, native crop zoom, v2 termination
diagnostics, raw text inspection, and SHA-256 provenance. Never runs GPU inference
and never falls back to other datasets or legacy WACV data when inputs are missing.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import pathlib
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = pathlib.Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

import evaluate as repo_eval  # noqa: E402
from pcb_vlm.evaluation.fics_replay import (  # noqa: E402
    build_image_hash_index,
    classify_call_termination_and_parse_v2,
    resolve_portable_image_path,
)

MODEL_ORDER = ("qwen2", "deepseek", "qwen3")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_and_verify_index_artifact(
    base_dir: pathlib.Path,
    entry: Dict[str, Any],
    label: str,
) -> pathlib.Path:
    raw_path = entry.get("path")
    if not raw_path:
        raise FileNotFoundError(f"Index entry {label!r} missing path.")
    cand = pathlib.Path(str(raw_path))
    resolved = cand if cand.is_absolute() else (base_dir / cand).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Indexed artifact {label!r} not found: {resolved}")
    expected_sha = entry.get("sha256")
    if not expected_sha or not isinstance(expected_sha, str):
        raise ValueError(f"Indexed artifact {label!r} missing declared sha256.")
    actual_sha = _sha256_file(resolved)
    if actual_sha.lower() != expected_sha.strip().lower():
        raise ValueError(
            f"SHA-256 mismatch for {label!r} ({resolved}): {actual_sha} != {expected_sha}"
        )
    return resolved


def _parse_kv_pairs(items: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" in item:
            k, v = item.split("=", 1)
            if k.strip() and v.strip():
                out.append((k.strip(), v.strip()))
    return out


def _match_board_tp_fp_fn(
    gts: List[Dict[str, Any]],
    preds: List[Dict[str, Any]],
    iou_thresh: float = 0.50,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Annotate GT (TP_GT/FN) and Pred (TP/FP) using scripts/evaluate.py canonical matcher."""
    pred_match: Dict[int, Tuple[int, float]] = {}
    gt_match: Dict[int, Tuple[int, float]] = {}

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

    for cat in sorted(set(gts_by_cat) & set(preds_by_cat)):
        gt_items = gts_by_cat[cat]
        pred_items = preds_by_cat[cat]
        match_res = repo_eval.match_category_predictions_one_to_one(
            [item[1] for item in gt_items],
            [item[1] for item in pred_items],
            iou_threshold=iou_thresh,
        )
        for p_local_idx, g_local_idx, ov in match_res["matched_pairs"]:
            pi = pred_items[p_local_idx][0]
            gi = gt_items[g_local_idx][0]
            pred_match[pi] = (int(gts[gi]["id"]), round(ov, 4))
            gt_match[gi] = (pi, round(ov, 4))

    gt_out: List[Dict[str, Any]] = []
    for gi, g in enumerate(gts):
        row = dict(g)
        if gi in gt_match:
            row["match_status"] = "TP_GT"
            row["matched_iou"] = gt_match[gi][1]
        else:
            row["match_status"] = "FN"
            row["matched_iou"] = None
        gt_out.append(row)

    pred_out: List[Dict[str, Any]] = []
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
        pred_out.append(row)
    return gt_out, pred_out


class ShowcaseStore:
    """Thread-safe read-only loader isolating datasets by `(dataset_key, board_id)`."""

    def __init__(
        self,
        index_path: pathlib.Path,
        *,
        path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
        image_roots: Optional[Sequence[pathlib.Path]] = None,
    ) -> None:
        self.index_path = pathlib.Path(index_path)
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.path_remaps = list(path_remaps or [])
        self.image_hash_index = build_image_hash_index(image_roots) if image_roots else None
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self._roi_cache: Dict[Tuple[str, str, int, Optional[str]], bytes] = {}
        self._load_all()

    def _load_all(self) -> None:
        base_dir = self.index_path.parent
        for ds_key, ds_meta in self.index["datasets"].items():
            verified_artifacts: Dict[str, pathlib.Path] = {}
            for art_key, art_entry in ds_meta["artifacts"].items():
                if art_entry.get("optional") and not art_entry.get("path"):
                    continue
                verified_artifacts[art_key] = _resolve_and_verify_index_artifact(
                    base_dir,
                    art_entry,
                    f"{ds_key}.artifacts.{art_key}",
                )

            rep_path = verified_artifacts["replay_report"]
            coco_path = verified_artifacts["gt_coco"]
            plan_path = verified_artifacts["frozen_plan"]

            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

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

            # Extract W-call ROI metadata and per-model raw call records
            raw_by_model_board: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for mk in MODEL_ORDER:
                raw_path = _resolve_and_verify_index_artifact(
                    base_dir,
                    ds_meta["raw_jsonl"][mk],
                    f"{ds_key}.raw_jsonl.{mk}",
                )
                eos_ids = (100001,) if mk == "deepseek" else (151645, 151643)
                for line in raw_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    bid = str(r["board_id"])
                    if bid in boards and r.get("mode") == "W" and boards[bid]["roi_image_path"] is None:
                        cand_p = resolve_portable_image_path(
                            r,
                            path_remaps=self.path_remaps,
                            image_hash_index=self.image_hash_index,
                        )
                        boards[bid]["roi_image_path"] = str(cand_p) if cand_p else r.get("image_path")
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
                    raw_by_model_board.setdefault((mk, bid), []).append({
                        "call_id": r["call_id"],
                        "mode": r["mode"],
                        "tile_id": r.get("tile_id"),
                        "roi_offset_x": int(r.get("roi_offset_x", 0)),
                        "roi_offset_y": int(r.get("roi_offset_y", 0)),
                        "tile_offset_x": int(r.get("tile_offset_x", 0)),
                        "tile_offset_y": int(r.get("tile_offset_y", 0)),
                        "width": int(r["orig_input_width"]),
                        "height": int(r["orig_input_height"]),
                        "generated_tokens": r.get("generated_tokens"),
                        "gpu_generate_ms": r.get("gpu_generate_ms"),
                        "cuda_max_reserved_mib": r.get("cuda_max_reserved_mib"),
                        "normal_eos": v2["normal_eos"],
                        "length_limited_without_eos": v2["length_limited_without_eos"],
                        "syntax_or_format_incomplete_after_eos": v2["syntax_or_format_incomplete_after_eos"],
                        "strict_schema_ok": v2["strict_schema_ok"],
                        "explicit_empty": v2["explicit_empty"],
                        "status": v2["status"],
                        "parsed_count": len(v2["objects"]),
                        "raw_output": str(r.get("raw_output") or ""),
                    })

            self.datasets[ds_key] = {
                "dataset_key": ds_key,
                "label": ds_meta["label"],
                "split_role": ds_meta["split_role"],
                "boards": boards,
                "replay": rep,
                "raw_by_model_board": raw_by_model_board,
                "provenance": {
                    k: {"path": v["path"], "sha256": v.get("sha256")}
                    for k, v in ds_meta["artifacts"].items()
                },
                "raw_provenance": {
                    mk: {"path": v["path"], "sha256": v.get("sha256")}
                    for mk, v in ds_meta["raw_jsonl"].items()
                },
            }

    def get_overview(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "notice": "历史真实推理回放（只读）：展示已完成且通过独立复核的 201 调用 (s18+s28) 与 306 调用 (s29+s27) 数据；集成后未重跑 GPU，缺失范围不回退旧数据。",
            "datasets": {},
        }
        for ds_key, ds in self.datasets.items():
            rep = ds["replay"]
            out["datasets"][ds_key] = {
                "label": ds["label"],
                "split_role": ds["split_role"],
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
                "models": {
                    mk: {
                        "termination_v2": rep["models"][mk]["termination_v2"],
                        "combined": {
                            mode: {
                                k: rep["models"][mk]["combined"][mode]["evaluate_detections_iou50"][k]
                                for k in ("tp", "fp", "fn", "total_gt", "total_predictions", "micro")
                            }
                            for mode in ("W", "T")
                        },
                    }
                    for mk in MODEL_ORDER
                },
            }
        return out

    def get_board_view(self, *, dataset_key: str, board_id: str, model_key: str, mode: str) -> Dict[str, Any]:
        if dataset_key not in self.datasets:
            raise KeyError(f"Unknown dataset scope: {dataset_key!r}")
        ds = self.datasets[dataset_key]
        if board_id not in ds["boards"]:
            raise KeyError(f"Board {board_id!r} not in dataset {dataset_key!r}")
        if model_key not in MODEL_ORDER:
            raise KeyError(f"Unsupported model key: {model_key!r}")
        if mode not in ("W", "T"):
            raise KeyError(f"Unsupported mode: {mode!r}")

        b = ds["boards"][board_id]
        mrep = ds["replay"]["models"][model_key]["per_board"][board_id][mode]
        ev = mrep["evaluate_detections_iou50"]
        preds_raw = list((mrep.get("pred_record") or {}).get("objects") or [])
        gts_matched, preds_matched = _match_board_tp_fp_fn(b["gts"], preds_raw, iou_thresh=0.50)

        calls = [c for c in ds["raw_by_model_board"].get((model_key, board_id), []) if c["mode"] == mode]
        roi_exists = bool(b["roi_image_path"] and pathlib.Path(b["roi_image_path"]).is_file())
        return {
            "dataset_key": dataset_key,
            "dataset_label": ds["label"],
            "board_id": board_id,
            "coco_image_id": b["coco_image_id"],
            "model_key": model_key,
            "mode": mode,
            "roi": {
                "offset_x": b["roi_offset_x"],
                "offset_y": b["roi_offset_y"],
                "width": b["roi_width"],
                "height": b["roi_height"],
                "image_available": roi_exists,
                "image_path": b["roi_image_path"],
                "sha256": b["roi_sha256"],
            },
            "metrics": {
                "tp": ev["tp"],
                "fp": ev["fp"],
                "fn": ev["fn"],
                "total_gt": ev["total_gt"],
                "total_predictions": ev["total_predictions"],
                "precision": ev["micro"]["precision"],
                "recall": ev["micro"]["recall"],
                "f1": ev["micro"]["f1"],
                "per_class": ev["per_class"],
            },
            "termination_v2": ds["replay"]["models"][model_key]["termination_v2"],
            "gts": gts_matched,
            "predictions": preds_matched,
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
        if dataset_key not in self.datasets:
            raise KeyError(f"Unknown dataset scope: {dataset_key!r}")
        ds = self.datasets[dataset_key]
        if board_id not in ds["boards"]:
            raise KeyError(f"Board {board_id!r} not in dataset {dataset_key!r}")
        b = ds["boards"][board_id]
        img_str = b.get("roi_image_path")
        if not img_str or not pathlib.Path(img_str).is_file():
            raise FileNotFoundError(f"ROI image missing for ({dataset_key}, {board_id}): {img_str!r}")
        expected_roi_sha = b.get("roi_sha256")
        if expected_roi_sha:
            actual_roi_sha = _sha256_file(pathlib.Path(img_str))
            if actual_roi_sha.lower() != str(expected_roi_sha).strip().lower():
                raise ValueError(
                    f"ROI image SHA-256 mismatch for ({dataset_key}, {board_id}): "
                    f"{actual_roi_sha} != {expected_roi_sha}"
                )

        cache_key = (dataset_key, board_id, int(max_width), crop)
        if cache_key in self._roi_cache:
            return self._roi_cache[cache_key]

        with Image.open(img_str) as im:
            rgb = im.convert("RGB")
            if crop:
                parts = [int(float(v)) for v in crop.split(",")]
                if len(parts) != 4:
                    raise ValueError(f"Expected x1,y1,x2,y2 for crop, got {crop!r}")
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
        self._roi_cache[cache_key] = data
        return data


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>FICS-PCB 真实推理回放展示台</title>
<style>
  :root { --bg: #0f172a; --panel: #1e293b; --line: #334155; --text: #f1f5f9; --muted: #94a3b8; --accent: #38bdf8; --warn: #f59e0b; --danger: #ef4444; --ok: #22c55e; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "Microsoft YaHei", system-ui, sans-serif; background: var(--bg); color: var(--text); }
  .banner { background: #172554; border-bottom: 1px solid #1d4ed8; padding: 10px 18px; font-size: 13px; color: #dbeafe; }
  .banner strong { color: #93c5fd; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 12px 18px; background: var(--panel); border-bottom: 1px solid var(--line); }
  .toolbar label { font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 4px; }
  select, button { background: #0f172a; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  button.active { border-color: var(--accent); background: #0c4a6e; }
  .checks { display: flex; flex-wrap: wrap; gap: 10px; margin-left: auto; font-size: 12px; align-items: center; }
  .checks label { flex-direction: row; color: var(--text); cursor: pointer; }
  .layout { display: grid; grid-template-columns: minmax(0, 1.85fr) minmax(320px, 1fr); gap: 14px; padding: 14px 18px; }
  @media (max-width: 1024px) { .layout { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px; overflow: hidden; }
  .viewport-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
  .canvas-wrap { position: relative; width: 100%; overflow: auto; max-height: 68vh; background: #020617; border: 1px solid var(--line); border-radius: 8px; }
  .stage { position: relative; display: inline-block; transform-origin: top left; }
  .stage img { display: block; max-width: none; }
  .stage svg { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }
  .metrics-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
  .kpi { background: #0f172a; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
  .kpi .lbl { font-size: 11px; color: var(--muted); }
  .kpi .val { font-size: 18px; font-weight: 700; margin-top: 2px; }
  .empty-box { padding: 16px; border: 1px dashed var(--warn); background: rgba(245,158,11,0.1); border-radius: 8px; color: #fde68a; font-size: 13px; margin: 10px 0; }
  .scroll-list { max-height: 220px; overflow: auto; border: 1px solid var(--line); border-radius: 6px; margin-top: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 5px 7px; border-bottom: 1px solid #263244; text-align: left; }
  tr:hover { background: #0f172a; cursor: pointer; }
  pre.raw-pre { white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere; max-height: 200px; overflow: auto; background: #020617; padding: 10px; border-radius: 6px; border: 1px solid var(--line); font-size: 12px; line-height: 1.45; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .b-tp { background: rgba(56,189,248,0.2); color: #7dd3fc; }
  .b-fp { background: rgba(239,68,68,0.2); color: #fca5a5; }
  .b-fn { background: rgba(245,158,11,0.2); color: #fde047; }
  .b-gt { background: rgba(34,197,94,0.2); color: #86efac; }
</style>
</head>
<body>
  <div class="banner" id="noticeBanner">
    <strong>历史真实推理回放（只读）</strong>：加载中…（严禁自动回退其他样本或伪造未运行结果）
  </div>

  <div class="toolbar">
    <label>数据集范围（严格隔离）
      <select id="selDataset"></select>
    </label>
    <label>板卡
      <select id="selBoard"></select>
    </label>
    <label>模型
      <select id="selModel">
        <option value="qwen3">Qwen3-VL-2B-Instruct</option>
        <option value="qwen2">Qwen2-VL-2B-Instruct</option>
        <option value="deepseek">deepseek-vl-1.3b-chat</option>
      </select>
    </label>
    <label>输入模式
      <select id="selMode">
        <option value="W">W（整 ROI）</option>
        <option value="T">T（切片 + NMS）</option>
      </select>
    </label>
    <label>缩放视角
      <select id="selZoom">
        <option value="fit">自适应视野</option>
        <option value="1">1.0x 原图比例</option>
        <option value="2">2.0x 细节放大</option>
      </select>
    </label>
    <button id="btnResetCrop" type="button">显示全 ROI</button>
    <div class="checks">
      <label><input type="checkbox" id="chkTP" checked /> <span class="badge b-tp">TP 命中</span></label>
      <label><input type="checkbox" id="chkFP" checked /> <span class="badge b-fp">FP 误报</span></label>
      <label><input type="checkbox" id="chkFN" checked /> <span class="badge b-fn">FN 漏检</span></label>
      <label><input type="checkbox" id="chkGT" /> <span class="badge b-gt">已命中 GT</span></label>
    </div>
  </div>

  <div class="layout">
    <div class="card">
      <div class="viewport-head">
        <div id="boardTitle" style="font-weight:700;font-size:14px;">-</div>
        <div id="cropStatus" style="font-size:12px;color:var(--muted);">点击右侧框可局部裁剪放大</div>
      </div>
      <div id="errorPanel" class="empty-box" style="display:none;"></div>
      <div class="canvas-wrap" id="canvasWrap">
        <div class="stage" id="stage">
          <img id="boardImg" alt="FICS-PCB ROI" />
          <svg id="overlaySvg" preserveAspectRatio="none"></svg>
        </div>
      </div>
      <div style="margin-top:10px;display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:12px;color:var(--muted);">原始调用选择器（查看单次 VLM 输出、EOS 终止与未过滤原始文本）</span>
        <select id="selCall" style="max-width:340px;"></select>
      </div>
      <div id="callMeta" style="font-size:12px;color:var(--muted);margin-top:6px;"></div>
      <pre id="rawOutputPre" class="raw-pre">-</pre>
    </div>

    <div style="display:flex;flex-direction:column;gap:12px;">
      <div class="card">
        <div style="font-weight:700;margin-bottom:8px;font-size:14px;">当前视角真实指标（IoU >= 0.50）</div>
        <div class="metrics-grid" id="kpiGrid"></div>
        <div id="v2SummaryBox" style="font-size:12px;color:var(--muted);line-height:1.5;"></div>
      </div>

      <div class="card">
        <div style="font-weight:700;margin-bottom:6px;font-size:14px;">预测与 GT 目标列表（点击即局部放大定位）</div>
        <div id="emptyItemsNotice" class="empty-box" style="display:none;">当前模型/模式下无有效预测框（0 预测进入全 GT 分母统计）。</div>
        <div class="scroll-list">
          <table>
            <thead><tr><th>状态</th><th>类别</th><th>IoU / ROI坐标</th></tr></thead>
            <tbody id="itemsBody"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div style="font-weight:700;margin-bottom:6px;font-size:14px;">数据来源与 SHA-256 溯源</div>
        <pre id="provPre" class="raw-pre" style="max-height:140px;"></pre>
      </div>
    </div>
  </div>

<script>
let OVERVIEW = null;
let CURRENT = null;
let CURRENT_CROP = null; // [x1,y1,x2,y2] in ROI pixel space

async function fetchJson(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}

function pct(v) { return (v === null || v === undefined) ? "N/A" : (v * 100).toFixed(2) + "%"; }

function showError(msg) {
  const ep = document.getElementById("errorPanel");
  ep.textContent = msg;
  ep.style.display = "block";
}
function hideError() {
  document.getElementById("errorPanel").style.display = "none";
}

function populateBoards() {
  const dsKey = document.getElementById("selDataset").value;
  const selB = document.getElementById("selBoard");
  selB.innerHTML = "";
  const ds = OVERVIEW.datasets[dsKey];
  if (!ds) return;
  for (const b of ds.boards) {
    const opt = document.createElement("option");
    opt.value = b.board_id;
    opt.textContent = `${b.board_id} (GT=${b.gt_count}, ${b.roi_width}x${b.roi_height})`;
    selB.appendChild(opt);
  }
}

async function loadBoardView() {
  hideError();
  CURRENT_CROP = null;
  const dsKey = document.getElementById("selDataset").value;
  const boardId = document.getElementById("selBoard").value;
  const modelKey = document.getElementById("selModel").value;
  const mode = document.getElementById("selMode").value;
  try {
    CURRENT = await fetchJson(`/api/board?dataset=${encodeURIComponent(dsKey)}&board=${encodeURIComponent(boardId)}&model=${encodeURIComponent(modelKey)}&mode=${encodeURIComponent(mode)}`);
    renderAll();
  } catch (err) {
    showError("加载失败（不回退其他样本）：" + err.message);
  }
}

function renderAll() {
  if (!CURRENT) return;
  document.getElementById("boardTitle").textContent =
    `${CURRENT.dataset_label} · 板卡 ${CURRENT.board_id} (COCO image_id=${CURRENT.coco_image_id}) · ${CURRENT.model_key} [${CURRENT.mode}]`;

  const m = CURRENT.metrics;
  const kpiHtml = [
    ["TP / FP / FN", `${m.tp} / ${m.fp} / ${m.fn}`],
    ["Precision", pct(m.precision)],
    ["Recall", pct(m.recall)],
    ["F1", pct(m.f1)],
    ["GT 总数", m.total_gt],
    ["Post-NMS 预测", m.total_predictions],
  ].map(([k, v]) => `<div class="kpi"><div class="lbl">${k}</div><div class="val">${v}</div></div>`).join("");
  document.getElementById("kpiGrid").innerHTML = kpiHtml;

  const v2 = CURRENT.termination_v2;
  document.getElementById("v2SummaryBox").innerHTML =
    `模型全量终止统计（本范围共 ${v2.completed_ok_calls}/${v2.total_calls_observed} 次）：正常 EOS=${v2.normal_eos}，达 512 上限无 EOS=${v2.length_limited_without_eos}，EOS 后格式异常=${v2.syntax_or_format_incomplete_after_eos}，严格 Schema=${v2.strict_schema_ok}，显式空数组=${v2.explicit_empty}。`;

  const selCall = document.getElementById("selCall");
  selCall.innerHTML = "";
  for (let i = 0; i < CURRENT.calls.length; i++) {
    const c = CURRENT.calls[i];
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${c.call_id} [${c.status}] (obj=${c.parsed_count}, tok=${c.generated_tokens})`;
    selCall.appendChild(opt);
  }
  renderSelectedCall();
  renderItemsTable();
  renderProvenance();
  renderCanvas();
}

function renderSelectedCall() {
  const idx = Number(document.getElementById("selCall").value || 0);
  const c = (CURRENT && CURRENT.calls[idx]) || null;
  if (!c) {
    document.getElementById("callMeta").textContent = "无调用记录";
    document.getElementById("rawOutputPre").textContent = "";
    return;
  }
  document.getElementById("callMeta").textContent =
    `call_id=${c.call_id} | 输入=${c.width}x${c.height} | tokens=${c.generated_tokens} | GPU=${c.gpu_generate_ms}ms | VRAM=${c.cuda_max_reserved_mib}MiB | normal_eos=${c.normal_eos} | len_limit=${c.length_limited_without_eos} | strict_schema=${c.strict_schema_ok}`;
  document.getElementById("rawOutputPre").textContent = c.raw_output || "(空输出)";
}

function renderItemsTable() {
  const tbody = document.getElementById("itemsBody");
  tbody.innerHTML = "";
  const rox = CURRENT.roi.offset_x, roy = CURRENT.roi.offset_y;
  const rows = [];
  for (const p of CURRENT.predictions) rows.push({ kind: p.match_status, cls: p.label, box: p.bbox_xyxy, iou: p.matched_iou });
  for (const g of CURRENT.gts) if (g.match_status === "FN") rows.push({ kind: "FN", cls: g.label, box: g.bbox_xyxy, iou: null });

  document.getElementById("emptyItemsNotice").style.display = CURRENT.predictions.length === 0 ? "block" : "none";
  for (const item of rows) {
    const tr = document.createElement("tr");
    const rx1 = Math.round(item.box[0] - rox), ry1 = Math.round(item.box[1] - roy);
    const rx2 = Math.round(item.box[2] - rox), ry2 = Math.round(item.box[3] - roy);
    const badgeCls = item.kind === "TP" ? "b-tp" : (item.kind === "FP" ? "b-fp" : "b-fn");
    tr.innerHTML = `<td><span class="badge ${badgeCls}">${item.kind}</span></td><td>${item.cls}</td><td>${item.iou ? "IoU=" + item.iou + " " : ""}[${rx1},${ry1},${rx2},${ry2}]</td>`;
    tr.onclick = () => {
      const pad = 260;
      CURRENT_CROP = [
        Math.max(0, rx1 - pad),
        Math.max(0, ry1 - pad),
        Math.min(CURRENT.roi.width, rx2 + pad),
        Math.min(CURRENT.roi.height, ry2 + pad),
      ];
      renderCanvas();
    };
    tbody.appendChild(tr);
  }
}

function renderProvenance() {
  const p = {
    frozen_plan_sha256: CURRENT.provenance.frozen_plan?.sha256,
    gt_coco_sha256: CURRENT.provenance.gt_coco?.sha256,
    replay_report_sha256: CURRENT.provenance.replay_report?.sha256,
    raw_jsonl_sha256: CURRENT.raw_provenance[CURRENT.model_key]?.sha256,
    roi_image_sha256: CURRENT.roi.sha256,
  };
  document.getElementById("provPre").textContent = JSON.stringify(p, null, 2);
}

function renderCanvas() {
  if (!CURRENT) return;
  const img = document.getElementById("boardImg");
  const svg = document.getElementById("overlaySvg");
  const wrap = document.getElementById("canvasWrap");
  const stage = document.getElementById("stage");
  const zoomVal = document.getElementById("selZoom").value;

  if (!CURRENT.roi.image_available) {
    showError("当前板卡 ROI 原图在本地路径不可用，页面拒绝回退占位图：" + CURRENT.roi.image_path);
    img.removeAttribute("src");
    svg.innerHTML = "";
    return;
  }

  const rox = CURRENT.roi.offset_x, roy = CURRENT.roi.offset_y;
  const vx1 = CURRENT_CROP ? CURRENT_CROP[0] : 0;
  const vy1 = CURRENT_CROP ? CURRENT_CROP[1] : 0;
  const vx2 = CURRENT_CROP ? CURRENT_CROP[2] : CURRENT.roi.width;
  const vy2 = CURRENT_CROP ? CURRENT_CROP[3] : CURRENT.roi.height;
  const vw = Math.max(1, vx2 - vx1), vh = Math.max(1, vy2 - vy1);

  document.getElementById("cropStatus").textContent = CURRENT_CROP
    ? `局部裁剪窗口: [${vx1}, ${vy1}, ${vx2}, ${vy2}] (px)`
    : `完整 ROI 视野: ${CURRENT.roi.width}x${CURRENT.roi.height} (offset=[${rox}, ${roy}])`;

  const cropParam = CURRENT_CROP ? `&crop=${vx1},${vy1},${vx2},${vy2}` : "";
  const reqMaxW = CURRENT_CROP ? 1200 : 1800;
  img.src = `/api/image?dataset=${encodeURIComponent(CURRENT.dataset_key)}&board=${encodeURIComponent(CURRENT.board_id)}&max_width=${reqMaxW}${cropParam}`;

  const dispW = zoomVal === "fit" ? Math.max(480, wrap.clientWidth - 4) : Math.min(2800, Math.round(vw * Number(zoomVal)));
  const dispH = Math.round(dispW * (vh / vw));
  img.style.width = dispW + "px";
  img.style.height = dispH + "px";
  stage.style.width = dispW + "px";
  stage.style.height = dispH + "px";
  svg.setAttribute("viewBox", `${vx1} ${vy1} ${vw} ${vh}`);

  const showTP = document.getElementById("chkTP").checked;
  const showFP = document.getElementById("chkFP").checked;
  const showFN = document.getElementById("chkFN").checked;
  const showGT = document.getElementById("chkGT").checked;
  const sw = Math.max(2, vw / 420);
  const fs = Math.max(12, Math.round(vw / 75));
  let elems = "";

  const drawBox = (b, color, dash, text) => {
    const x1 = b[0] - rox, y1 = b[1] - roy, x2 = b[2] - rox, y2 = b[3] - roy;
    const w = Math.max(2, x2 - x1), h = Math.max(2, y2 - y1);
    elems += `<rect x="${x1}" y="${y1}" width="${w}" height="${h}" fill="none" stroke="${color}" stroke-width="${sw}" ${dash ? 'stroke-dasharray="' + (sw * 3) + ' ' + (sw * 2) + '"' : ""} />`;
    elems += `<text x="${x1}" y="${Math.max(vy1 + fs, y1 - 4)}" fill="${color}" font-size="${fs}" font-weight="bold" style="paint-order:stroke;stroke:#020617;stroke-width:${Math.max(3, fs / 4)}px;">${text}</text>`;
  };

  for (const g of CURRENT.gts) {
    if (g.match_status === "TP_GT" && showGT) drawBox(g.bbox_xyxy, "#22c55e", false, `GT:${g.label}`);
    if (g.match_status === "FN" && showFN) drawBox(g.bbox_xyxy, "#facc15", true, `FN:${g.label}`);
  }
  for (const p of CURRENT.predictions) {
    if (p.match_status === "TP" && showTP) drawBox(p.bbox_xyxy, "#38bdf8", false, `TP:${p.label} (${p.matched_iou})`);
    if (p.match_status === "FP" && showFP) drawBox(p.bbox_xyxy, "#ef4444", true, `FP:${p.label}`);
  }
  svg.innerHTML = elems;
}

async function init() {
  OVERVIEW = await fetchJson("/api/overview");
  document.getElementById("noticeBanner").innerHTML = `<strong>历史真实推理回放（只读）</strong>：${OVERVIEW.notice}`;
  const selDs = document.getElementById("selDataset");
  for (const [k, ds] of Object.entries(OVERVIEW.datasets)) {
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = `${ds.label} (${k})`;
    selDs.appendChild(opt);
  }
  populateBoards();
  selDs.onchange = () => { populateBoards(); loadBoardView(); };
  document.getElementById("selBoard").onchange = loadBoardView;
  document.getElementById("selModel").onchange = loadBoardView;
  document.getElementById("selMode").onchange = loadBoardView;
  document.getElementById("selCall").onchange = renderSelectedCall;
  document.getElementById("selZoom").onchange = renderCanvas;
  document.getElementById("btnResetCrop").onclick = () => { CURRENT_CROP = null; renderCanvas(); };
  for (const id of ["chkTP", "chkFP", "chkFN", "chkGT"]) {
    document.getElementById(id).onchange = renderCanvas;
  }
  await loadBoardView();
}
init().catch(e => showError("初始化失败：" + e.message));
</script>
</body>
</html>
"""


class ShowcaseHandler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, store: ShowcaseStore, **kwargs: Any) -> None:
        self.store = store
        super().__init__(*args, **kwargs)

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/overview":
                self._send_json(HTTPStatus.OK, self.store.get_overview())
            elif parsed.path == "/api/board":
                ds = (qs.get("dataset") or [""])[0]
                board = (qs.get("board") or [""])[0]
                model = (qs.get("model") or [""])[0]
                mode = (qs.get("mode") or [""])[0]
                self._send_json(
                    HTTPStatus.OK,
                    self.store.get_board_view(dataset_key=ds, board_id=board, model_key=model, mode=mode),
                )
            elif parsed.path == "/api/image":
                ds = (qs.get("dataset") or [""])[0]
                board = (qs.get("board") or [""])[0]
                max_w = int((qs.get("max_width") or ["1600"])[0])
                crop = (qs.get("crop") or [None])[0]
                jpg = self.store.render_image_jpeg(dataset_key=ds, board_id=board, max_width=max_w, crop=crop)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile.write(jpg)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {parsed.path}"})
        except (KeyError, FileNotFoundError, ValueError) as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(exc)})


def create_server(
    *,
    index_path: pathlib.Path,
    host: str = "127.0.0.1",
    port: int = 7861,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_roots: Optional[Sequence[pathlib.Path]] = None,
) -> ThreadingHTTPServer:
    store = ShowcaseStore(index_path, path_remaps=path_remaps, image_roots=image_roots)
    handler = functools.partial(ShowcaseHandler, store=store)
    return ThreadingHTTPServer((host, port), handler)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FICS-PCB read-only real replay web showcase.")
    parser.add_argument("--index", required=True, help="Path to results_index.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--path-map", action="append", help="Repeatable OLD_PREFIX=NEW_PREFIX for relocated ROI images.")
    parser.add_argument("--image-root", action="append", help="Repeatable directory containing relocated images.")
    args = parser.parse_args(argv)

    srv = create_server(
        index_path=pathlib.Path(args.index),
        host=args.host,
        port=args.port,
        path_remaps=_parse_kv_pairs(args.path_map),
        image_roots=[pathlib.Path(p) for p in (args.image_root or [])],
    )
    print(f"FICS Showcase listening on http://{args.host}:{srv.server_address[1]}/ (read-only historical replay)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
