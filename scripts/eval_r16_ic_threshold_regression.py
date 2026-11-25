"""Round 16 Step 2B: Pre-GT locked IC threshold regression on S20/S4/S1 (dev) & S31 (public read-only).

Strict invariants:
- Reads frozen `selected_ic_threshold` from `r16_val_ic_calibration_frozen.json`.
- Uses the IDENTICAL locked R15 `yolo_low_conf_pool_005.json` candidate pools for both arms (0.25 vs calibrated IC).
- Writes `r16_ic_filter_pre_gt_lock.json` hashing BOTH pre-filter and post-filter prediction lists BEFORE reading any COCO GT.
- S31 is strictly evaluated as a public read-only regression check (never used for threshold selection).
- Records every deleted IC TP and renders local visual evidence galleries (`ic_deleted_tp_evidence.png` & `ic_high_conf_fp_gallery.png`).
"""

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
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    _prf1,
    compute_iou_xyxy,
    evaluate_composite_board,
    filter_fics6_pool_by_class_thresholds,
    load_coco_gts_for_single_board,
)
from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES  # noqa: E402

DEFAULT_CAL_FILE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration16-20261124\ic_calibration\r16_val_ic_calibration_frozen.json")
DEFAULT_OUT_DIR = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration16-20261124\ic_calibration")

BOARDS_META: Dict[str, Dict[str, Any]] = {
    "s20": {
        "role": "dev_regression",
        "pool": pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\live_demo\s20\fics_yolo11n_r15\yolo_low_conf_pool_005.json"),
        "coco": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s20_front_coco.json"),
        "tif": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif"),
    },
    "s4": {
        "role": "dev_regression",
        "pool": pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\live_demo\s4\fics_yolo11n_r15\yolo_low_conf_pool_005.json"),
        "coco": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s4_front_coco.json"),
        "tif": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\img\s4_front.tif"),
    },
    "s1": {
        "role": "dev_regression",
        "pool": pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\live_demo\s1\fics_yolo11n_r15\yolo_low_conf_pool_005.json"),
        "coco": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration11-20261123\dummy_placeholder_check")
        if False else pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s1_front_official_coco.json"),
        "tif": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s1\DSLR\img\s1_front.tif"),
    },
    "s31": {
        "role": "public_readonly_regression_not_for_tuning",
        "pool": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-codex-s31-20261124\live_ws\jobs\job_20261124_200546_0e81598b\base_runs\composite_yolo\yolo_low_conf_pool_005.json"),
        "coco": pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\data_audit\official_coco\s31_front_official_coco.json"),
        "tif": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s31\DSLR\img\s31_front.tif"),
    },
}


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for ch in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest().lower()


def _render_crop_gallery(items: Sequence[Mapping[str, Any]], out_path: pathlib.Path, title_note: str) -> None:
    cols = 4
    thumb = 240
    pad = 10
    hdr = 26
    if not items:
        im = Image.new("RGB", (640, 80), (15, 23, 42))
        d = ImageDraw.Draw(im)
        d.text((16, 30), f"{title_note}: None (0 items)", fill=(226, 232, 240), font=ImageFont.load_default())
        im.save(out_path, format="PNG")
        return

    rows = max(1, (len(items) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + hdr + pad) + pad), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Group opens by tif path
    by_tif: Dict[str, List[tuple[int, Mapping[str, Any]]]] = {}
    for idx, it in enumerate(items):
        by_tif.setdefault(str(it["tif_path"]), []).append((idx, it))

    for tif_str, entries in by_tif.items():
        with Image.open(tif_str) as im_raw:
            rgb = im_raw.convert("RGB")
            w, h = rgb.size
            for idx, it in entries:
                r = idx // cols
                c = idx % cols
                ox = pad + c * (thumb + pad)
                oy = pad + r * (thumb + hdr + pad)
                bx = [float(v) for v in it["bbox_xyxy"]]
                bw = max(16.0, bx[2] - bx[0])
                bh = max(16.0, bx[3] - bx[1])
                pad_px = max(48.0, 0.45 * max(bw, bh))
                cx1 = max(0, int(round(bx[0] - pad_px)))
                cy1 = max(0, int(round(bx[1] - pad_px)))
                cx2 = min(w, int(round(bx[2] + pad_px)))
                cy2 = min(h, int(round(bx[3] + pad_px)))
                crop = rgb.crop((cx1, cy1, cx2, cy2))
                cw, ch = crop.size
                scale = min(thumb / float(cw), thumb / float(ch))
                rw, rh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
                resized = crop.resize((rw, rh), Image.Resampling.BILINEAR)
                d_sub = ImageDraw.Draw(resized)
                lx1 = (bx[0] - cx1) * scale
                ly1 = (bx[1] - cy1) * scale
                lx2 = (bx[2] - cx1) * scale
                ly2 = (bx[3] - cy1) * scale
                d_sub.rectangle([lx1, ly1, lx2, ly2], outline=it.get("color", (248, 113, 113)), width=2)
                if it.get("gt_bbox_xyxy"):
                    gb = [float(v) for v in it["gt_bbox_xyxy"]]
                    d_sub.rectangle(
                        [(gb[0] - cx1) * scale, (gb[1] - cy1) * scale, (gb[2] - cx1) * scale, (gb[3] - cy1) * scale],
                        outline=(34, 197, 94),
                        width=2,
                    )
                draw.text((ox, oy), str(it["caption"])[:42], fill=(226, 232, 240), font=font)
                canvas.paste(resized, (ox, oy + hdr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-GT locked IC threshold regression on S20/S4/S1 and read-only S31.")
    parser.add_argument("--cal-file", type=pathlib.Path, default=DEFAULT_CAL_FILE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    cal = json.loads(args.cal_file.read_text(encoding="utf-8"))
    ic_thr = cal.get("selected_ic_threshold")
    if ic_thr is None:
        # Still run diagnostic using highest val FP reduction candidate if none strictly qualified
        best_diag = max(cal["candidates"], key=lambda c: (int(c["ic_fp_reduction_vs_025"]), -float(c["ic_threshold"])))
        ic_thr = float(best_diag["ic_threshold"])
        eval_mode = "diagnostic_only_no_val_qualified_threshold"
    else:
        ic_thr = float(ic_thr)
        eval_mode = "val_calibrated_ic_threshold"

    # STEP 1 (GT-free): Materialize & Pre-GT lock both 0.25 and calibrated prediction lists
    locked_lists_dir = args.out_dir / "locked_lists"
    locked_lists_dir.mkdir(parents=True, exist_ok=True)
    lock_entries: List[Dict[str, Any]] = []
    pools: Dict[str, List[Dict[str, Any]]] = {}
    arm_lists: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for bid, meta in BOARDS_META.items():
        pool_p = meta["pool"]
        assert pool_p.is_file(), f"Missing pool: {pool_p}"
        pool_sha = _sha256(pool_p)
        pool = json.loads(pool_p.read_text(encoding="utf-8"))
        pools[bid] = pool

        base_025 = filter_fics6_pool_by_class_thresholds(pool, default_conf=0.25, per_class_conf=None)
        cal_list = filter_fics6_pool_by_class_thresholds(pool, default_conf=0.25, per_class_conf={"ic": ic_thr})
        arm_lists[bid] = {"base_025": base_025, "ic_cal": cal_list}

        for arm_name, data in (("base_025", base_025), ("ic_cal", cal_list)):
            out_p = locked_lists_dir / f"{bid}_{arm_name}.json"
            out_p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            lock_entries.append({
                "board": bid,
                "arm": arm_name,
                "source_pool_path": str(pool_p),
                "source_pool_sha256": pool_sha,
                "locked_list_path": str(out_p),
                "locked_list_sha256": _sha256(out_p),
                "count": len(data),
            })

    pre_gt_lock = {
        "schema_version": "r16_ic_filter_pre_gt_lock_v1",
        "eval_mode": eval_mode,
        "selected_ic_threshold": ic_thr,
        "gt_loaded_before_lock": False,
        "entries": lock_entries,
    }
    lock_path = args.out_dir / "r16_ic_filter_pre_gt_lock.json"
    lock_path.write_text(json.dumps(pre_gt_lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Verify lock hashes (0 mismatches) before reading any GT
    for ent in lock_entries:
        assert _sha256(pathlib.Path(ent["locked_list_path"])) == ent["locked_list_sha256"]

    # STEP 2 (Post-GT): Score against official COCO GTs
    per_board: Dict[str, Any] = {}
    deleted_ic_tps: List[Dict[str, Any]] = []
    high_conf_ic_fps: List[Dict[str, Any]] = []

    for bid, meta in BOARDS_META.items():
        gts, _ = load_coco_gts_for_single_board(meta["coco"], image_id=None)
        ev_base = evaluate_composite_board(gts=gts, predictions_025=arm_lists[bid]["base_025"], low_conf_pool_005=pools[bid])
        ev_cal = evaluate_composite_board(gts=gts, predictions_025=arm_lists[bid]["ic_cal"], low_conf_pool_005=pools[bid])

        # Identify exact deleted IC TPs (matched in base_025, removed by ic >= ic_thr)
        cal_pids = {str(p["prediction_id"]) for p in arm_lists[bid]["ic_cal"]}
        base_by_pid = {str(p["prediction_id"]): p for p in arm_lists[bid]["base_025"]}
        gt_by_id = {int(g["id"]): g for g in gts}
        for mp in ev_base["matched_pairs"]:
            if mp["label"] == "ic" and str(mp["prediction_id"]) not in cal_pids:
                pred = base_by_pid[str(mp["prediction_id"])]
                gt = gt_by_id[int(mp["gt_id"])]
                deleted_ic_tps.append({
                    "board": bid,
                    "role": meta["role"],
                    "prediction_id": mp["prediction_id"],
                    "score": float(pred["score"]),
                    "iou": float(mp["iou"]),
                    "bbox_xyxy": pred["bbox_xyxy"],
                    "gt_id": mp["gt_id"],
                    "gt_bbox_xyxy": gt["bbox_xyxy"],
                    "tif_path": str(meta["tif"]),
                    "caption": f"{bid} DEL-TP {mp['prediction_id']} s={pred['score']:.2f}",
                    "color": (250, 204, 21),
                })

        # Collect surviving IC FPs in ic_cal sorted by descending score
        matched_cal_pids = {str(mp["prediction_id"]) for mp in ev_cal["matched_pairs"]}
        for p in arm_lists[bid]["ic_cal"]:
            if p["label"] == "ic" and str(p["prediction_id"]) not in matched_cal_pids:
                # Check if overlaps any GT of another class
                ov_other = None
                for g in gts:
                    ov = compute_iou_xyxy(p["bbox_xyxy"], g["bbox_xyxy"])
                    if ov >= 0.30:
                        ov_other = f"ov_{g['label']}({ov:.2f})"
                        break
                high_conf_ic_fps.append({
                    "board": bid,
                    "role": meta["role"],
                    "prediction_id": p["prediction_id"],
                    "score": float(p["score"]),
                    "bbox_xyxy": p["bbox_xyxy"],
                    "overlap_note": ov_other or "background_or_structure",
                    "tif_path": str(meta["tif"]),
                    "caption": f"{bid} IC-FP s={p['score']:.2f} {ov_other or 'bg'}",
                    "color": (248, 113, 113),
                })

        per_board[bid] = {
            "role": meta["role"],
            "gt_total": len(gts),
            "base_025": {
                "overall": ev_base["class_aware"],
                "per_class": ev_base["per_class"],
            },
            "ic_cal": {
                "ic_threshold": ic_thr,
                "overall": ev_cal["class_aware"],
                "per_class": ev_cal["per_class"],
            },
        }

    # Combine S20 + S4 + S1 (dev regression ONLY; S31 strictly excluded!)
    def _combine_dev(arm_key: str) -> Dict[str, Any]:
        dev_keys = ("s20", "s4", "s1")
        tp = sum(int(per_board[b][arm_key]["overall"]["tp"]) for b in dev_keys)
        fp = sum(int(per_board[b][arm_key]["overall"]["fp"]) for b in dev_keys)
        fn = sum(int(per_board[b][arm_key]["overall"]["fn"]) for b in dev_keys)
        ic_tp = sum(int(per_board[b][arm_key]["per_class"]["ic"]["tp"]) for b in dev_keys)
        ic_fp = sum(int(per_board[b][arm_key]["per_class"]["ic"]["fp"]) for b in dev_keys)
        ic_fn = sum(int(per_board[b][arm_key]["per_class"]["ic"]["fn"]) for b in dev_keys)
        ic_gt = sum(int(per_board[b][arm_key]["per_class"]["ic"]["gt_count"]) for b in dev_keys)
        ic_pred = sum(int(per_board[b][arm_key]["per_class"]["ic"]["pred_count"]) for b in dev_keys)
        return {
            "overall": _prf1(tp, fp, fn),
            "ic": {
                "gt_count": ic_gt,
                "pred_count": ic_pred,
                "signed_count_error": ic_pred - ic_gt,
                **_prf1(ic_tp, ic_fp, ic_fn),
            },
        }

    dev_base = _combine_dev("base_025")
    dev_cal = _combine_dev("ic_cal")
    ic_fp_drop = dev_base["ic"]["fp"] - dev_cal["ic"]["fp"]
    ic_fp_drop_pct = round((ic_fp_drop / dev_base["ic"]["fp"]) * 100.0, 2) if dev_base["ic"]["fp"] > 0 else 0.0
    ic_tp_drop = dev_base["ic"]["tp"] - dev_cal["ic"]["tp"]
    f1_non_decreasing = dev_cal["overall"]["f1_pct"] >= dev_base["overall"]["f1_pct"]
    dev_gate_passed = (not cal["no_promotable_ic_threshold"]) and (ic_fp_drop_pct >= 20.0) and (ic_tp_drop <= 1) and f1_non_decreasing

    high_conf_ic_fps.sort(key=lambda x: -float(x["score"]))
    top_fps_gallery = high_conf_ic_fps[:12]

    del_tp_png = args.out_dir / "ic_deleted_tp_evidence.png"
    fp_gal_png = args.out_dir / "ic_high_conf_fp_gallery.png"
    _render_crop_gallery(deleted_ic_tps[:12], del_tp_png, "Deleted IC TPs")
    _render_crop_gallery(top_fps_gallery, fp_gal_png, f"Top Surviving IC FPs (thr={ic_thr:.2f})")

    report = {
        "schema_version": "r16_ic_regression_report_v1",
        "pre_gt_lock_path": str(lock_path),
        "pre_gt_lock_mismatches": 0,
        "val_calibration_no_promotable_threshold": cal["no_promotable_ic_threshold"],
        "evaluated_ic_threshold": ic_thr,
        "eval_mode": eval_mode,
        "dev_s20_s4_s1_combined": {
            "base_025": dev_base,
            "ic_cal": dev_cal,
            "ic_fp_drop": ic_fp_drop,
            "ic_fp_drop_pct": ic_fp_drop_pct,
            "ic_tp_drop": ic_tp_drop,
            "overall_f1_delta_pp": round(dev_cal["overall"]["f1_pct"] - dev_base["overall"]["f1_pct"], 2),
        },
        "r16_dev_gate": {
            "rule": "val_qualified AND S20+S4+S1 IC FP drop >= 20% AND IC TP drop <= 1 AND overall Micro-F1 non-decreasing",
            "passed": dev_gate_passed,
            "status_note": (
                "PASSED dev gate -> keep as experimental preset pending new-family blind verification"
                if dev_gate_passed
                else "NOT PROMOTED -> keep uniform 0.25 as default"
            ),
        },
        "per_board": per_board,
        "deleted_ic_tps": deleted_ic_tps,
        "top_surviving_ic_fps": top_fps_gallery,
        "evidence_images": {
            "deleted_ic_tp_png": str(del_tp_png),
            "high_conf_ic_fp_gallery_png": str(fp_gal_png),
        },
    }
    rep_path = args.out_dir / "r16_ic_regression_report.json"
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "evaluated_ic_threshold": ic_thr,
        "dev_s20_s4_s1": report["dev_s20_s4_s1_combined"],
        "r16_dev_gate": report["r16_dev_gate"],
        "s31_readonly_base_vs_cal": {
            "base_025": per_board["s31"]["base_025"]["overall"],
            "ic_cal": per_board["s31"]["ic_cal"]["overall"],
            "base_ic": per_board["s31"]["base_025"]["per_class"]["ic"],
            "cal_ic": per_board["s31"]["ic_cal"]["per_class"]["ic"],
        },
        "deleted_ic_tp_count": len(deleted_ic_tps),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
