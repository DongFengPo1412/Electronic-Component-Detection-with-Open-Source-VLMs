"""Build physical-family isolated 4-class crop dataset (`ic`, `diode`, `transistor`, `other`) for Round 18.

Strict invariants:
- Strictly physical-family disjoint splits:
  - `r18_clf_train`: 10 boards / 7 families (`s26`, `s25`, `s21/s23`, `s22`, `s30`, `s8`, `s18/s27/s28/s29`).
  - `r18_dq_val`: 1 unseen R17 family (`s14`, `s15`).
  - `r15_frozen_val`: frozen R15 val_split (`s2`, `s3`) used only for proposal caching & abstention calibration.
- Hard-bans dev (`s20`, `s4`, `s1`), blind (`s12`, `s24`), sealed (`s31`, `s10`, `s16`, `s19`), overlap (`s9`),
  and family-excluded boards (`s17`, `s11`, `s5`, `s6`, `s7`).
- Never embeds class/coord hints in crop pixels or crop filenames (`crop_00001.png`).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import random
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import (  # noqa: E402
    FICS_RAW_TO_CANONICAL,
    file_sha256,
    parse_component_location,
    validate_box_within_image,
)
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    DEFAULT_ULTRALYTICS_OVERLAY,
    EXPECTED_FICS_R15_BEST_SHA256,
    run_yolo11n_class_aware_board,
)
from pcb_vlm.evaluation.crop_verifier_r18 import (  # noqa: E402
    R18_CLASSES,
    R18_TARGET_CLASSES,
    extract_square_context_crop,
)

R18_CACHE = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration18-20261125")
R18_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125")
DEFAULT_CROP_DIR = R18_CACHE / "crop_dataset_r18"
DEFAULT_PROPOSALS_DIR = R18_CACHE / "r15_proposals_cache"
DEFAULT_AUDIT_DIR = R18_ART / "data_audit"

R15_WEIGHTS_PATH = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\weights\fics_yolo11n_r15_best.pt"
)
R14_INVENTORY_PATH = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json"
)
R17_EXPANSION_AUDIT_PATH = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\data_audit\r17_data_expansion_audit.json"
)

BANNED_BOARDS = {
    "s20", "s4", "s1",
    "s12", "s24",
    "s31", "s10", "s16", "s19", "s9",
    "s17", "s11", "s5", "s6", "s7",
}

TRAIN_BOARDS = {"s26", "s25", "s21", "s23", "s22", "s30", "s8", "s18", "s27", "s28", "s29"}
DQ_VAL_BOARDS = {"s14", "s15"}
R15_FROZEN_VAL_BOARDS = {"s2", "s3"}


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    ia = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if ia <= 0.0:
        return 0.0
    aa = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    return ia / max(1e-9, aa + ab - ia)


def _read_gts(csv_path: pathlib.Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    raw_lower = {k.lower(): v for k, v in FICS_RAW_TO_CANONICAL.items()}
    out: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rt = str(row.get("component_type") or "").strip()
            canon = FICS_RAW_TO_CANONICAL.get(rt) or raw_lower.get(rt.lower())
            if not canon:
                continue
            _, box, _ = parse_component_location(row.get("component_location") or "")
            validate_box_within_image(box, img_w, img_h)
            x1, y1, x2, y2 = [float(v) for v in box]
            out.append({"label": canon, "bbox_xyxy": [x1, y1, x2, y2], "area": (x2 - x1) * (y2 - y1)})
    return out


def _render_audit_grid(records: Sequence[Mapping[str, Any]], out_path: pathlib.Path) -> None:
    cols, thumb, pad, hdr = 5, 192, 10, 20
    rows = max(1, (len(records) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + hdr + pad) + pad), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, r in enumerate(records):
        row_i, col_i = idx // cols, idx % cols
        ox, oy = pad + col_i * (thumb + pad), pad + row_i * (thumb + hdr + pad)
        with Image.open(r["crop_path"]) as im:
            res = im.convert("RGB").resize((thumb, thumb), Image.Resampling.BILINEAR)
        draw.text((ox, oy), f"{r['label']} ({r['source_kind'][:16]})", fill=(226, 232, 240), font=font)
        canvas.paste(res, (ox, oy + hdr))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare R18 isolated 4-class crop dataset & cache R15 proposals.")
    parser.add_argument("--crop-dir", type=pathlib.Path, default=DEFAULT_CROP_DIR)
    parser.add_argument("--proposals-dir", type=pathlib.Path, default=DEFAULT_PROPOSALS_DIR)
    parser.add_argument("--audit-dir", type=pathlib.Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    r14_inv = json.loads(R14_INVENTORY_PATH.read_text(encoding="utf-8"))
    r17_exp = json.loads(R17_EXPANSION_AUDIT_PATH.read_text(encoding="utf-8"))

    board_sides: List[Dict[str, Any]] = []
    for b in r14_inv["boards"]:
        if b["split"] in ("train_pool", "val_split"):
            board_sides.append({
                "board_side_id": b["board_side_id"],
                "physical_board": b["physical_board"],
                "tif_path": b["tif_path"],
                "csv_path": b["csv_path"],
            })
    for b in r17_exp["train_expansion_sides"]:
        board_sides.append({
            "board_side_id": b["board_side_id"],
            "physical_board": b["physical_board"],
            "tif_path": b["tif_path"],
            "csv_path": b["csv_path"],
        })

    if args.crop_dir.exists():
        shutil.rmtree(args.crop_dir)
    for sp in ("train", "val"):
        (args.crop_dir / sp).mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    all_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    proposal_cache_index: Dict[str, Any] = {}
    crop_counter = 0

    for bs in board_sides:
        stem = str(bs["board_side_id"])
        phys = str(bs["physical_board"])
        assert phys not in BANNED_BOARDS, f"Banned board in R18 dataset: {phys}"
        if phys in TRAIN_BOARDS:
            role = "train"
        elif phys in DQ_VAL_BOARDS:
            role = "val"
        elif phys in R15_FROZEN_VAL_BOARDS:
            role = "r15_frozen_val"
        else:
            raise RuntimeError(f"Unassigned board {phys}")

        tif_p = pathlib.Path(str(bs["tif_path"]))
        csv_p = pathlib.Path(str(bs["csv_path"]))
        prop_dir = args.proposals_dir / stem
        preds_025_p = prop_dir / "yolo_official_025.json"
        if not preds_025_p.is_file():
            inf = run_yolo11n_class_aware_board(
                image_path=tif_p,
                board_id=stem,
                weights_path=R15_WEIGHTS_PATH,
                expected_weights_sha256=EXPECTED_FICS_R15_BEST_SHA256,
                detector_preset="fics_yolo11n_r15",
                expected_class_schema="fics_6",
                overlay_dir=DEFAULT_ULTRALYTICS_OVERLAY,
                output_dir=prop_dir,
                tile_size=640,
                overlap_ratio=0.20,
                imgsz=640,
                conf_low=0.05,
                conf_official=0.25,
                nms_iou=0.50,
            )
            preds_025 = inf["official_predictions_fics6"]
        else:
            preds_025 = json.loads(preds_025_p.read_text(encoding="utf-8"))

        with Image.open(tif_p) as im_raw:
            rgb = im_raw.convert("RGB")
            img_w, img_h = rgb.size
            gts = _read_gts(csv_p, img_w, img_h)

            proposal_cache_index[stem] = {
                "board_side_id": stem,
                "physical_board": phys,
                "role": role,
                "tif_path": str(tif_p),
                "csv_path": str(csv_p),
                "preds_025_path": str(preds_025_p),
                "pred_count_025": len(preds_025),
            }

            if role == "r15_frozen_val":
                continue

            target_gts = [g for g in gts if g["label"] in R18_TARGET_CLASSES]
            non_target_gts = [g for g in gts if g["label"] not in R18_TARGET_CLASSES]

            def _save_crop(box: Sequence[float], label: str, source_kind: str) -> None:
                nonlocal crop_counter
                crop_counter += 1
                fname = f"crop_{crop_counter:05d}.png"
                dst = args.crop_dir / role / fname
                extract_square_context_crop(rgb, box).save(dst, format="PNG")
                all_records[role].append({
                    "crop_id": crop_counter,
                    "split": role,
                    "board_side_id": stem,
                    "physical_board": phys,
                    "label": label,
                    "source_kind": source_kind,
                    "crop_path": str(dst),
                    "bbox_xyxy": [round(float(v), 2) for v in box],
                })

            # 1) All confirmed GT target crops (ic, diode, transistor)
            for g in target_gts:
                _save_crop(g["bbox_xyxy"], g["label"], "official_gt")

            # 2) Confirmed non-target GT crops (prioritize large capacitors & inductors + stratified resistors)
            ind_caps = sorted(
                [g for g in non_target_gts if g["label"] in ("inductor", "capacitor")],
                key=lambda x: -x["area"],
            )
            res_list = [g for g in non_target_gts if g["label"] == "resistor"]
            rng.shuffle(res_list)
            keep_non_target = ind_caps[:8] + res_list[:4]
            for g in keep_non_target:
                _save_crop(g["bbox_xyxy"], "other", f"known_non_target_{g['label']}")

            # 3) Confirmed R15 target-class FPs (IoU == 0.0 with ALL ic/diode/transistor GTs)
            fp_ov_added = 0
            fp_bg_added = 0
            for p in preds_025:
                if p.get("label") not in R18_TARGET_CLASSES:
                    continue
                pb = p["bbox_xyxy"]
                if any(_iou(pb, tg["bbox_xyxy"]) > 0.0 for tg in target_gts):
                    continue
                if any(_iou(pb, ntg["bbox_xyxy"]) >= 0.30 for ntg in non_target_gts):
                    if fp_ov_added < 8:
                        _save_crop(pb, "other", "r15_fp_non_target_gt")
                        fp_ov_added += 1
                else:
                    if fp_bg_added < 10:
                        _save_crop(pb, "other", "r15_fp_background")
                        fp_bg_added += 1

        print(f"[CROP] {stem} ({role}): total_so_far={crop_counter}", flush=True)

    manifest = {
        "schema_version": "r18_crop_manifest_v1",
        "classes": list(R18_CLASSES),
        "train": all_records["train"],
        "val": all_records["val"],
        "proposal_cache_index": proposal_cache_index,
    }
    manifest_p = args.crop_dir / "crop_manifest.json"
    manifest_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Visual audit grid (train & val samples across all 4 classes + other sub-sources)
    grid_samples: List[Dict[str, Any]] = []
    for cls, sk_prefix in (
        ("ic", "official_gt"),
        ("diode", "official_gt"),
        ("transistor", "official_gt"),
        ("other", "known_non_target"),
        ("other", "r15_fp_"),
    ):
        matches = [r for r in all_records["train"] if r["label"] == cls and r["source_kind"].startswith(sk_prefix)]
        grid_samples.extend(matches[:4])
    _render_audit_grid(grid_samples, args.audit_dir / "r18_crop_train_audit_grid.png")

    summary = {
        "schema_version": "r18_crop_dataset_audit_v1",
        "manifest_path": str(manifest_p),
        "manifest_sha256": file_sha256(manifest_p).lower(),
        "train_counts": dict(collections.Counter(r["label"] for r in all_records["train"])),
        "train_source_counts": dict(collections.Counter(r["source_kind"] for r in all_records["train"])),
        "val_counts": dict(collections.Counter(r["label"] for r in all_records["val"])),
        "val_source_counts": dict(collections.Counter(r["source_kind"] for r in all_records["val"])),
        "train_physical_boards": sorted({r["physical_board"] for r in all_records["train"]}),
        "val_physical_boards": sorted({r["physical_board"] for r in all_records["val"]}),
        "banned_boards_checked": sorted(BANNED_BOARDS),
        "zero_cross_family_overlap": True,
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "r18_crop_dataset_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("train_counts", "val_counts", "train_source_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
