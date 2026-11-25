"""Pure CPU post-GT evaluator & IC high-conf FP root-cause diagnostic for Round 17 locked predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import evaluate as repo_eval  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    _prf1,
    box_in_composite_region,
    evaluate_composite_board,
    load_coco_gts_for_single_board,
)

DEFAULT_LOCK_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\eval\r17_pre_gt_lock.json")
DEFAULT_INVENTORY = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json")
DEFAULT_OUTPUT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\eval\r17_paired_eval_report.json")
DEFAULT_DIAG_PNG = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\eval\r17_ic_fp_before_after_diagnostic.png")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    ia = iw * ih
    if ia <= 0.0:
        return 0.0
    aa = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return ia / max(1e-9, aa + ab - ia)


def _find_ic_fps(gts: Sequence[Mapping[str, Any]], preds_025: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ic_gts = [{"bbox": g["bbox_xyxy"], "category": "ic"} for g in gts if g["label"] == "ic"]
    ic_preds = [p for p in preds_025 if p.get("label") == "ic"]
    # Greedy score-desc 1-to-1 matching identical to evaluate.py
    ic_sorted = sorted(ic_preds, key=lambda x: -float(x.get("score", 0.0)))
    used_gt: set[int] = set()
    fps: List[Dict[str, Any]] = []
    for p in ic_sorted:
        pb = p["bbox_xyxy"]
        best_i = -1
        best_iou = 0.50
        for gi, g in enumerate(ic_gts):
            if gi in used_gt:
                continue
            v = _iou(pb, g["bbox"])
            if v >= best_iou:
                best_iou = v
                best_i = gi
        if best_i >= 0:
            used_gt.add(best_i)
        else:
            # Classify failure cause against any GT (e.g., overlaps transistor GT or overlaps IC GT < 0.50 fragment)
            overlapped_gt_labels = [
                g["label"] for g in gts if _iou(pb, g["bbox_xyxy"]) >= 0.15
            ]
            if "ic" in overlapped_gt_labels:
                cause = "large_ic_fragment_or_offset"
            elif "transistor" in overlapped_gt_labels:
                cause = "transistor_to_ic_misclass"
            elif overlapped_gt_labels:
                cause = f"other_gt_misclass:{overlapped_gt_labels[0]}"
            else:
                cause = "unannotated_bg_header_connector_sticker"
            fps.append({
                "score": round(float(p.get("score", 0.0)), 4),
                "bbox_xyxy": [round(float(v), 1) for v in pb],
                "cause": cause,
            })
    return fps


def _render_ic_diag_png(
    *,
    items: Sequence[Mapping[str, Any]],
    out_path: pathlib.Path,
) -> None:
    cols = 4
    cell_w, cell_h = 300, 320
    pad = 12
    rows = max(1, (len(items) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * (cell_w + pad) + pad, rows * (cell_h + pad) + pad), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    img_cache: Dict[str, Image.Image] = {}
    try:
        for idx, it in enumerate(items[:12]):
            r, c = idx // cols, idx % cols
            ox, oy = pad + c * (cell_w + pad), pad + r * (cell_h + pad)
            tp = str(it["tif_path"])
            if tp not in img_cache:
                img_cache[tp] = Image.open(tp).convert("RGB")
            im = img_cache[tp]
            w, h = im.size
            x1, y1, x2, y2 = it["bbox_xyxy"]
            bw, bh = max(20.0, x2 - x1), max(20.0, y2 - y1)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            half = max(bw, bh) * 1.15
            rx1, ry1 = max(0, int(cx - half)), max(0, int(cy - half))
            rx2, ry2 = min(w, int(cx + half)), min(h, int(cy + half))
            crop = im.crop((rx1, ry1, rx2, ry2))
            cw, ch = crop.size
            scale = min(cell_w / max(1, cw), (cell_h - 36) / max(1, ch))
            rw, rh = max(1, int(cw * scale)), max(1, int(ch * scale))
            res = crop.resize((rw, rh), Image.Resampling.BILINEAR)
            d2 = ImageDraw.Draw(res)
            bx1, by1 = (x1 - rx1) * scale, (y1 - ry1) * scale
            bx2, by2 = (x2 - rx1) * scale, (y2 - ry1) * scale
            col = (248, 113, 113) if it["arm"] == "r15" else (250, 204, 21)
            d2.rectangle([bx1, by1, bx2, by2], outline=col, width=3)
            draw.text((ox, oy), f"{it['arm'].upper()} {it['board']} s={it['score']:.2f}", fill=(241, 245, 249), font=font)
            draw.text((ox, oy + 14), f"{it['cause'][:38]}", fill=(148, 163, 184), font=font)
            canvas.paste(res, (ox, oy + 32))
    finally:
        for im in img_cache.values():
            im.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate locked R17 paired predictions against COCO GT (CPU only).")
    parser.add_argument("--lock-file", type=pathlib.Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--inventory", type=pathlib.Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diag-png", type=pathlib.Path, default=DEFAULT_DIAG_PNG)
    args = parser.parse_args(argv)

    lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
    mismatches: List[Dict[str, str]] = []
    for lf in lock["locked_files"]:
        lp = pathlib.Path(str(lf["path"]))
        if not lp.is_file():
            mismatches.append({"role": lf["role"], "path": str(lp), "reason": "missing"})
            continue
        actual_sha = _sha256_file(lp)
        if actual_sha != str(lf["sha256"]).lower():
            mismatches.append({"role": lf["role"], "path": str(lp), "expected": lf["sha256"], "actual": actual_sha})
    if mismatches:
        raise RuntimeError(f"Pre-GT lock SHA-256 verification failed: {mismatches}")

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    coco_by_bkey = {
        b["physical_board"]: pathlib.Path(str(b["coco_path"]))
        for b in inv["boards"]
        if b.get("coco_path") and b["physical_board"] in ("s20", "s4", "s1")
    }

    per_board: Dict[str, Any] = {}
    ic_diag_summary: Dict[str, Any] = {}
    diag_gallery_items: List[Dict[str, Any]] = []

    for bkey, binfo in lock["boards"].items():
        gts, _ = load_coco_gts_for_single_board(coco_by_bkey[bkey], image_id=None)
        arms_eval: Dict[str, Any] = {}
        ic_fps_by_arm: Dict[str, List[Dict[str, Any]]] = {}
        for arm in ("fics_yolo11n_r15", "fics_yolo11n_r17_cand"):
            a_dir = pathlib.Path(str(binfo["arms"][arm]["artifacts_dir"]))
            preds_025 = json.loads((a_dir / "yolo_official_025.json").read_text(encoding="utf-8"))
            pool_005 = json.loads((a_dir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
            summary = json.loads((a_dir / "yolo_run_summary.json").read_text(encoding="utf-8"))
            ev = evaluate_composite_board(
                gts=gts,
                predictions_025=preds_025,
                low_conf_pool_005=pool_005,
                roi_bbox_half_open=binfo["roi_bbox_half_open"],
            )
            arms_eval[arm] = {
                "job_id": binfo["arms"][arm]["job_id"],
                "weights_sha256": summary.get("weights_sha256"),
                "timing_and_vram": summary.get("timing_and_vram"),
                "client_poll_wall_seconds": binfo["arms"][arm]["client_poll_wall_seconds"],
                **ev,
            }
            ic_fps_by_arm[arm] = _find_ic_fps(gts, preds_025)

        def _cause_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
            out: Dict[str, int] = {}
            for x in items:
                out[x["cause"]] = out.get(x["cause"], 0) + 1
            return out

        ic_diag_summary[bkey] = {
            "r15_ic_fp_total": len(ic_fps_by_arm["fics_yolo11n_r15"]),
            "r17_cand_ic_fp_total": len(ic_fps_by_arm["fics_yolo11n_r17_cand"]),
            "r15_causes": _cause_counts(ic_fps_by_arm["fics_yolo11n_r15"]),
            "r17_cand_causes": _cause_counts(ic_fps_by_arm["fics_yolo11n_r17_cand"]),
        }
        for x in ic_fps_by_arm["fics_yolo11n_r15"][:2]:
            diag_gallery_items.append({"arm": "r15", "board": bkey, "tif_path": binfo["tif_path"], **x})
        for x in ic_fps_by_arm["fics_yolo11n_r17_cand"][:2]:
            diag_gallery_items.append({"arm": "r17", "board": bkey, "tif_path": binfo["tif_path"], **x})

        q_evals: List[Dict[str, Any]] = []
        for qrec in binfo.get("r17_cand_queries") or []:
            q_doc = json.loads(pathlib.Path(str(qrec["query_file"])).read_text(encoding="utf-8"))
            target_set = set(q_doc["query_spec"]["target_classes"])
            reg_mode = str(q_doc["query_spec"]["region_mode"])
            orig_wh = q_doc["orig_wh"]
            scoped_gts = [
                g for g in gts
                if g["label"] in target_set
                and box_in_composite_region(g["bbox_xyxy"], orig_wh=orig_wh, region_mode=reg_mode)[0]
            ]
            tot_tp = tot_fp = tot_fn = 0
            for cls in sorted(target_set):
                cg = [{"bbox": g["bbox_xyxy"], "category": cls} for g in scoped_gts if g["label"] == cls]
                cp = [{"bbox": p["bbox_xyxy"], "category": cls, "score": float(p.get("score", 0.0))} for p in q_doc["predictions"] if p["label"] == cls]
                m = repo_eval.match_category_predictions_one_to_one(cg, cp, iou_threshold=0.50)
                tot_tp += m["tp"]; tot_fp += m["fp"]; tot_fn += m["fn"]
            qm = _prf1(tot_tp, tot_fp, tot_fn)
            q_evals.append({
                "query_id": q_doc["query_id"],
                "raw_query": q_doc["query_spec"]["raw_query"],
                "target_classes": sorted(target_set),
                "region_mode": reg_mode,
                "gt_count": len(scoped_gts),
                "predicted_count": int(q_doc["predicted_count"]),
                "count_abs_error": abs(int(q_doc["predicted_count"]) - len(scoped_gts)),
                **qm,
            })

        per_board[bkey] = {"gt_total": len(gts), "arms": arms_eval, "r17_cand_query_evaluations": q_evals}

    _render_ic_diag_png(items=diag_gallery_items, out_path=args.diag_png)

    def _combine(arm_key: str) -> Dict[str, Any]:
        bkeys = ("s20", "s4", "s1")
        tp = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["tp"]) for b in bkeys)
        fp = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["fp"]) for b in bkeys)
        fn = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["fn"]) for b in bkeys)
        gt_tot = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["gt_total"]) for b in bkeys)
        pred_tot = sum(int(per_board[b]["arms"][arm_key]["class_aware"]["pred_total"]) for b in bkeys)
        per_cls: Dict[str, Any] = {}
        for cls in ("capacitor", "resistor", "inductor", "transistor", "diode", "ic"):
            ctp = sum(int(per_board[b]["arms"][arm_key]["per_class"][cls]["tp"]) for b in bkeys)
            cfp = sum(int(per_board[b]["arms"][arm_key]["per_class"][cls]["fp"]) for b in bkeys)
            cfn = sum(int(per_board[b]["arms"][arm_key]["per_class"][cls]["fn"]) for b in bkeys)
            cgt = sum(int(per_board[b]["arms"][arm_key]["per_class"][cls]["gt_count"]) for b in bkeys)
            cpred = sum(int(per_board[b]["arms"][arm_key]["per_class"][cls]["pred_count"]) for b in bkeys)
            per_cls[cls] = {"gt_count": cgt, "pred_count": cpred, **_prf1(ctp, cfp, cfn)}
        res = _prf1(tp, fp, fn)
        res.update({"gt_total": gt_tot, "pred_total": pred_tot, "count_abs_error": abs(pred_tot - gt_tot), "per_class": per_cls})
        return res

    c_r15 = _combine("fics_yolo11n_r15")
    c_r17 = _combine("fics_yolo11n_r17_cand")

    r15_dq_tp = c_r15["per_class"]["diode"]["tp"] + c_r15["per_class"]["transistor"]["tp"]
    r17_dq_tp = c_r17["per_class"]["diode"]["tp"] + c_r17["per_class"]["transistor"]["tp"]
    ic_fp_r15 = c_r15["per_class"]["ic"]["fp"]
    ic_fp_r17 = c_r17["per_class"]["ic"]["fp"]
    ic_fp_drop_pct = ((ic_fp_r15 - ic_fp_r17) / max(1, ic_fp_r15)) * 100.0
    max_board_drop_pp = max(
        float(per_board[b]["arms"]["fics_yolo11n_r15"]["class_aware"]["f1_pct"])
        - float(per_board[b]["arms"]["fics_yolo11n_r17_cand"]["class_aware"]["f1_pct"])
        for b in ("s20", "s4", "s1")
    )
    gate_passed = (
        c_r17["f1_pct"] >= c_r15["f1_pct"]
        and (
            r17_dq_tp > r15_dq_tp
            or (ic_fp_drop_pct >= 15.0 and c_r17["per_class"]["ic"]["tp"] >= c_r15["per_class"]["ic"]["tp"])
        )
        and max_board_drop_pp <= 3.0
    )

    report = {
        "schema_version": "r17_paired_eval_report_v1",
        "pre_gt_lock_path": str(args.lock_file),
        "pre_gt_lock_mismatches": 0,
        "combined_s20_s4_s1": {
            "fics_yolo11n_r15": c_r15,
            "fics_yolo11n_r17_cand": c_r17,
            "delta_f1_pp": round(c_r17["f1_pct"] - c_r15["f1_pct"], 2),
            "diode_transistor_tp_r15": r15_dq_tp,
            "diode_transistor_tp_r17_cand": r17_dq_tp,
            "ic_fp_drop_pct": round(ic_fp_drop_pct, 2),
            "max_single_board_f1_drop_pp": round(max_board_drop_pp, 2),
        },
        "r17_dev_candidate_gate": {
            "passed": gate_passed,
            "note": "Default remains fics_yolo11n_r15 until Codex independent new-family blind acceptance on s12_front & s24_front.",
        },
        "ic_fp_root_cause_diagnostic": ic_diag_summary,
        "ic_fp_diagnostic_png": str(args.diag_png),
        "per_board": per_board,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "r15_f1_pct": c_r15["f1_pct"],
        "r17_cand_f1_pct": c_r17["f1_pct"],
        "r15_dq_tp": r15_dq_tp,
        "r17_dq_tp": r17_dq_tp,
        "ic_fp_r15": ic_fp_r15,
        "ic_fp_r17": ic_fp_r17,
        "gate_passed": gate_passed,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
