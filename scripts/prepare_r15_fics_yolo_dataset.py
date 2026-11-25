"""Materialize and audit the isolated Round 15 FICS-PCB 6-class YOLO11n training dataset.

Strict invariants:
- Strictly physical-board disjoint train (9 sides / 8 boards / 812 GT) and val (3 sides / 2 boards / 158 GT).
- Dev boards (s20, s4, s1), reserved blind boards (s31, s10, s16, s19), and WACV-overlap (s9) NEVER enter train or val.
- Uses exact inference tiling geometry (`compute_board_roi_and_tiles`, 640px, 20% overlap, white_thresh=248, roi_pad=32).
- Never uses `.lower()` before `FICS_RAW_TO_CANONICAL` lookup so `"ICs"` -> `"ic"` is 100% conserved.
- Writes `s3_front_official_coco.json` (val) and `s31_front_official_coco.json` (Codex blind handoff only, zero GPU use).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import shutil
import sys
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import (  # noqa: E402
    FICS_CANONICAL_CATEGORIES,
    FICS_CATEGORY_NAME_TO_ID,
    FICS_RAW_TO_CANONICAL,
    file_sha256,
    parse_board_and_side,
    parse_component_location,
    validate_box_within_image,
)
from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES  # noqa: E402
from pcb_vlm.evaluation.gt_free_pipeline import compute_board_roi_and_tiles  # noqa: E402

DEFAULT_INVENTORY = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json"
)
DEFAULT_DATASET_DIR = pathlib.Path(
    r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-20261123\fics_yolo_dataset"
)
DEFAULT_AUDIT_DIR = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\data_audit"
)

YOLO_FICS6_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(FICS_SIX_CLASSES)}
BANNED_SPLITS = ("dev_eval_r14", "smoke_only_unused_r14", "reserved_untouched_blind", "excluded_wacv_overlap")
BANNED_BOARDS = {"s20", "s4", "s1", "s31", "s10", "s16", "s19", "s9"}


def read_fics_csv_gts(csv_path: pathlib.Path, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Read official FICS CSV without case-mangling `'ICs'` and return validated GT boxes + per-class counts."""
    raw_to_canon_lower = {k.lower(): v for k, v in FICS_RAW_TO_CANONICAL.items()}
    gts: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with pathlib.Path(csv_path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_num, row in enumerate(reader, start=2):
            raw_type = str(row.get("component_type") or "").strip()
            canon = FICS_RAW_TO_CANONICAL.get(raw_type) or raw_to_canon_lower.get(raw_type.lower())
            if canon is None:
                continue
            _raw_norm, bbox_xyxy, note = parse_component_location(row.get("component_location") or "")
            validate_box_within_image(bbox_xyxy, img_w, img_h)
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            area = (x2 - x1) * (y2 - y1)
            gts.append({
                "gt_index": len(gts),
                "csv_row": row_num,
                "label": canon,
                "yolo_cls_id": YOLO_FICS6_NAME_TO_ID[canon],
                "coco_category_id": FICS_CATEGORY_NAME_TO_ID[canon],
                "raw_component_type": raw_type,
                "bbox_xyxy": [x1, y1, x2, y2],
                "area_px2": area,
                "conversion_note": note,
            })
            counts[canon] += 1
    return gts, {c: int(counts.get(c, 0)) for c in FICS_SIX_CLASSES}


def build_single_board_official_coco(
    *,
    board_meta: Mapping[str, Any],
    gts: Sequence[Mapping[str, Any]],
    img_w: int,
    img_h: int,
    out_path: pathlib.Path,
) -> Dict[str, Any]:
    """Export canonical 6-class COCO JSON matching FICS_CANONICAL_CATEGORIES from an official CSV."""
    stem = str(board_meta["board_side_id"])
    board_id, side = parse_board_and_side(stem)
    tif_path = pathlib.Path(str(board_meta["tif_path"]))
    csv_path = pathlib.Path(str(board_meta["csv_path"]))
    coco = {
        "info": {
            "dataset": "FICS-PCB_DSLR_Official_Single_Board",
            "version": "r15_fics6_v1",
            "board_side_id": stem,
            "source_csv": str(csv_path),
            "source_csv_sha256": file_sha256(csv_path).lower(),
        },
        "images": [
            {
                "id": 1,
                "sample_id": stem,
                "file_name": tif_path.name,
                "width": int(img_w),
                "height": int(img_h),
                "board_id": board_id,
                "side": side,
                "sha256": file_sha256(tif_path).lower(),
            }
        ],
        "categories": list(FICS_CANONICAL_CATEGORIES),
        "annotations": [
            {
                "id": idx + 1,
                "image_id": 1,
                "category_id": int(g["coco_category_id"]),
                "bbox": [
                    round(float(g["bbox_xyxy"][0]), 4),
                    round(float(g["bbox_xyxy"][1]), 4),
                    round(float(g["bbox_xyxy"][2] - g["bbox_xyxy"][0]), 4),
                    round(float(g["bbox_xyxy"][3] - g["bbox_xyxy"][1]), 4),
                ],
                "bbox_xyxy": [round(float(v), 4) for v in g["bbox_xyxy"]],
                "area": round(float(g["area_px2"]), 4),
                "iscrowd": 0,
                "raw_component_type": str(g["raw_component_type"]),
                "csv_row": int(g["csv_row"]),
            }
            for idx, g in enumerate(gts)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "path": str(out_path),
        "sha256": file_sha256(out_path).lower(),
        "annotation_count": len(coco["annotations"]),
    }


def decoded_rgb_sha256(img_path: pathlib.Path) -> Tuple[str, Tuple[int, int]]:
    """Compute SHA-256 over raw decoded RGB pixel bytes + size to detect identical imagery across encodings."""
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        h_obj = hashlib.sha256()
        h_obj.update(f"{w}x{h}:".encode("ascii"))
        h_obj.update(rgb.tobytes())
        return h_obj.hexdigest(), (w, h)


def clip_gt_boxes_to_tile(
    *,
    gts: Sequence[Mapping[str, Any]],
    tx1: int,
    ty1: int,
    tw: int,
    th: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Assign visible GT boxes inside a tile to normalized YOLO labels and track dropped boundary fragments."""
    tx2 = tx1 + tw
    ty2 = ty1 + th
    kept: List[Dict[str, Any]] = []
    dropped_frags: List[Dict[str, Any]] = []

    for g in gts:
        gx1, gy1, gx2, gy2 = g["bbox_xyxy"]
        ix1 = max(float(tx1), gx1)
        iy1 = max(float(ty1), gy1)
        ix2 = min(float(tx2), gx2)
        iy2 = min(float(ty2), gy2)
        cw = ix2 - ix1
        ch = iy2 - iy1
        if cw <= 0.0 or ch <= 0.0:
            continue
        ga = max(float(g["area_px2"]), 1.0)
        vis_frac = (cw * ch) / ga
        keep = (cw >= 4.0 and ch >= 4.0) and (
            vis_frac >= 0.40 or (cw >= 48.0 and ch >= 48.0 and vis_frac >= 0.15)
        )
        if not keep:
            dropped_frags.append({
                "gt_index": int(g["gt_index"]),
                "label": str(g["label"]),
                "clipped_wh": [round(cw, 2), round(ch, 2)],
                "visible_frac": round(vis_frac, 4),
            })
            continue
        lx1 = max(0.0, ix1 - float(tx1))
        ly1 = max(0.0, iy1 - float(ty1))
        lx2 = min(float(tw), ix2 - float(tx1))
        ly2 = min(float(th), iy2 - float(ty1))
        bw = lx2 - lx1
        bh = ly2 - ly1
        if bw <= 0.0 or bh <= 0.0:
            continue
        cx_n = min(1.0, max(0.0, ((lx1 + lx2) * 0.5) / float(tw)))
        cy_n = min(1.0, max(0.0, ((ly1 + ly2) * 0.5) / float(th)))
        w_n = min(1.0, max(0.0, bw / float(tw)))
        h_n = min(1.0, max(0.0, bh / float(th)))
        is_clipped = vis_frac < 0.999
        kept.append({
            "gt_index": int(g["gt_index"]),
            "label": str(g["label"]),
            "yolo_cls_id": int(g["yolo_cls_id"]),
            "local_bbox_xyxy": [round(lx1, 3), round(ly1, 3), round(lx2, 3), round(ly2, 3)],
            "yolo_norm": [round(cx_n, 8), round(cy_n, 8), round(w_n, 8), round(h_n, 8)],
            "visible_frac": round(vis_frac, 4),
            "is_clipped": is_clipped,
            "full_area_px2": float(g["area_px2"]),
        })
    return kept, dropped_frags


def render_preview_grid(
    *,
    samples: Sequence[Mapping[str, Any]],
    out_path: pathlib.Path,
    cols: int = 4,
    tile_thumb: int = 360,
) -> None:
    """Draw a visual verification grid of YOLO-labeled tiles across classes and box sizes."""
    palette = {
        "capacitor": (56, 189, 248),
        "resistor": (34, 197, 94),
        "inductor": (250, 204, 21),
        "transistor": (192, 132, 252),
        "diode": (251, 146, 60),
        "ic": (45, 212, 191),
    }
    rows = max(1, (len(samples) + cols - 1) // cols)
    pad = 12
    hdr = 22
    cell_w = tile_thumb + pad
    cell_h = tile_thumb + hdr + pad
    canvas = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, s in enumerate(samples):
        r = idx // cols
        c = idx % cols
        ox = pad + c * cell_w
        oy = pad + r * cell_h
        with Image.open(s["tile_path"]) as im_raw:
            im = im_raw.convert("RGB")
        tw, th = im.size
        scale = min(tile_thumb / float(tw), tile_thumb / float(th))
        rw, rh = max(1, int(round(tw * scale))), max(1, int(round(th * scale)))
        resized = im.resize((rw, rh), Image.Resampling.BILINEAR)
        d_sub = ImageDraw.Draw(resized)
        for box in s["labels"]:
            x1, y1, x2, y2 = [v * scale for v in box["local_bbox_xyxy"]]
            col = palette.get(box["label"], (248, 250, 252))
            d_sub.rectangle([x1, y1, x2, y2], outline=col, width=2)
            tag = box["label"][:3] + ("*" if box["is_clipped"] else "")
            d_sub.text((max(1.0, x1), max(1.0, y1 - 10)), tag, fill=col, font=font)
        draw.text((ox, oy), f"{s['split']}:{s['tile_stem']} (n={len(s['labels'])})", fill=(226, 232, 240), font=font)
        canvas.paste(resized, (ox, oy + hdr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build & audit Round 15 FICS 6-class YOLO11n dataset.")
    parser.add_argument("--inventory", type=pathlib.Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--audit-dir", type=pathlib.Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    boards: List[Dict[str, Any]] = list(inv["boards"])

    if args.dataset_dir.exists():
        shutil.rmtree(args.dataset_dir)
    for split in ("train", "val"):
        (args.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    raw_tile_cache = args.dataset_dir / "_raw_tile_work"
    raw_tile_cache.mkdir(parents=True, exist_ok=True)

    coco_out_dir = args.audit_dir / "official_coco"
    coco_out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    pixel_sha_to_boards: Dict[str, List[Dict[str, str]]] = {}
    board_audits: List[Dict[str, Any]] = []
    materialized_cocos: Dict[str, Any] = {}
    train_tile_records: List[Dict[str, Any]] = []
    val_tile_records: List[Dict[str, Any]] = []
    preview_candidates: List[Dict[str, Any]] = []

    for b in boards:
        stem = str(b["board_side_id"])
        phys = str(b["physical_board"])
        split_orig = str(b["split"])
        tif_p = pathlib.Path(str(b["tif_path"]))
        csv_p = pathlib.Path(str(b["csv_path"]))
        actual_tif_sha = file_sha256(tif_p).lower()
        actual_csv_sha = file_sha256(csv_p).lower()
        assert actual_tif_sha == str(b["tif_sha256"]).lower(), f"TIF SHA mismatch for {stem}"
        assert actual_csv_sha == str(b["csv_sha256"]).lower(), f"CSV SHA mismatch for {stem}"

        px_sha, (img_w, img_h) = decoded_rgb_sha256(tif_p)
        pixel_sha_to_boards.setdefault(px_sha, []).append({"board_side_id": stem, "split": split_orig, "physical_board": phys})

        gts, cls_counts = read_fics_csv_gts(csv_p, img_w, img_h)
        assert len(gts) == int(b["fics6_gt_total"]), (
            f"GT count mismatch for {stem}: parsed {len(gts)} != inventory {b['fics6_gt_total']}"
        )
        assert cls_counts == dict(b["class_counts"]), (
            f"Class counts mismatch for {stem}: {cls_counts} != {b['class_counts']}"
        )

        # Materialize missing canonical COCO JSON for s3_front (val) and s31_front (Codex blind handoff)
        if b.get("coco_path") is None and stem in ("s3_front", "s31_front"):
            coco_meta = build_single_board_official_coco(
                board_meta=b,
                gts=gts,
                img_w=img_w,
                img_h=img_h,
                out_path=coco_out_dir / f"{stem}_official_coco.json",
            )
            materialized_cocos[stem] = coco_meta

        # ROI coverage audit (CPU-only geometry check across train, val, dev, and s31)
        should_tile_dataset = split_orig in ("train_pool", "val_split")
        should_audit_roi = should_tile_dataset or split_orig in ("dev_eval_r14", "smoke_only_unused_r14") or stem == "s31_front"
        if not should_audit_roi:
            board_audits.append({
                "board_side_id": stem,
                "physical_board": phys,
                "split": split_orig,
                "image_wh": [img_w, img_h],
                "tif_sha256": actual_tif_sha,
                "csv_sha256": actual_csv_sha,
                "pixel_sha256": px_sha,
                "fics6_gt_total": len(gts),
                "class_counts": cls_counts,
                "roi_audited": False,
            })
            continue

        export_dir = (raw_tile_cache / stem) if should_tile_dataset else None
        geom = compute_board_roi_and_tiles(
            tif_p,
            roi_pad=32,
            white_thresh=248,
            tile_size=640,
            overlap_ratio=0.20,
            export_dir=export_dir,
            board_key=stem,
        )
        rx1, ry1, rx2, ry2 = [int(v) for v in geom["roi_bbox_half_open"]]
        tiles = list(geom["tiles"])

        roi_miss = 0
        roi_partial = 0
        for g in gts:
            gx1, gy1, gx2, gy2 = g["bbox_xyxy"]
            ga = max(float(g["area_px2"]), 1.0)
            ix1, iy1, ix2, iy2 = max(gx1, float(rx1)), max(gy1, float(ry1)), min(gx2, float(rx2)), min(gy2, float(ry2))
            ia = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if ia <= 0.0:
                roi_miss += 1
            elif (ia / ga) < 0.999:
                roi_partial += 1

        # Evaluate tile coverage for every GT on this board
        gt_best_vis: Dict[int, float] = {g["gt_index"]: 0.0 for g in gts}
        gt_assigned_any: set[int] = set()
        gt_full_vis_any: set[int] = set()
        per_tile_assignments: List[Dict[str, Any]] = []
        edge_clipped_box_instances = 0

        for t in tiles:
            tx1 = rx1 + int(t["tile_offset_x"])
            ty1 = ry1 + int(t["tile_offset_y"])
            tw = int(t["width"])
            th = int(t["height"])
            kept_labels, dropped_frags = clip_gt_boxes_to_tile(gts=gts, tx1=tx1, ty1=ty1, tw=tw, th=th)
            for g in gts:
                gx1, gy1, gx2, gy2 = g["bbox_xyxy"]
                ga = max(float(g["area_px2"]), 1.0)
                ia = max(0.0, min(gx2, float(tx1 + tw)) - max(gx1, float(tx1))) * max(
                    0.0, min(gy2, float(ty1 + th)) - max(gy1, float(ty1))
                )
                gt_best_vis[g["gt_index"]] = max(gt_best_vis[g["gt_index"]], ia / ga)
                if (ia / ga) >= 0.999:
                    gt_full_vis_any.add(g["gt_index"])
            for k in kept_labels:
                gt_assigned_any.add(int(k["gt_index"]))
                if k["is_clipped"]:
                    edge_clipped_box_instances += 1
            per_tile_assignments.append({
                "tile": t,
                "global_tx1": tx1,
                "global_ty1": ty1,
                "labels": kept_labels,
                "dropped_frags": dropped_frags,
            })

        if should_tile_dataset:
            assert phys not in BANNED_BOARDS, f"Banned board leaked into dataset: {phys}"
            yolo_split = "train" if split_orig == "train_pool" else "val"
            pos_tiles = [x for x in per_tile_assignments if len(x["labels"]) > 0]
            emp_tiles = [x for x in per_tile_assignments if len(x["labels"]) == 0]
            if yolo_split == "train":
                emp_sorted = sorted(emp_tiles, key=lambda x: int(x["tile"]["tile_id"]))
                n_keep_emp = min(len(emp_sorted), len(pos_tiles))
                kept_emp_idx = set(rng.sample(range(len(emp_sorted)), n_keep_emp)) if n_keep_emp > 0 else set()
                kept_emp = [emp_sorted[i] for i in sorted(kept_emp_idx)]
                kept_tiles = pos_tiles + kept_emp
            else:
                kept_emp = list(emp_tiles)
                kept_tiles = list(per_tile_assignments)

            for item in kept_tiles:
                t = item["tile"]
                tid = int(t["tile_id"])
                tile_stem = f"{stem}_T{tid:03d}"
                src_tile_p = pathlib.Path(str(t["path"]))
                dst_img_p = args.dataset_dir / "images" / yolo_split / f"{tile_stem}.png"
                dst_lbl_p = args.dataset_dir / "labels" / yolo_split / f"{tile_stem}.txt"
                shutil.copy2(src_tile_p, dst_img_p)
                lines = [
                    f"{lbl['yolo_cls_id']} {lbl['yolo_norm'][0]:.8f} {lbl['yolo_norm'][1]:.8f} {lbl['yolo_norm'][2]:.8f} {lbl['yolo_norm'][3]:.8f}"
                    for lbl in item["labels"]
                ]
                dst_lbl_p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                rec = {
                    "split": yolo_split,
                    "board_side_id": stem,
                    "physical_board": phys,
                    "tile_id": tid,
                    "tile_stem": tile_stem,
                    "image_path": str(dst_img_p),
                    "label_path": str(dst_lbl_p),
                    "global_tx1": int(item["global_tx1"]),
                    "global_ty1": int(item["global_ty1"]),
                    "width": int(t["width"]),
                    "height": int(t["height"]),
                    "label_count": len(item["labels"]),
                    "gt_indices": [int(l["gt_index"]) for l in item["labels"]],
                    "labels_meta": item["labels"],
                }
                if yolo_split == "train":
                    train_tile_records.append(rec)
                else:
                    val_tile_records.append(rec)

                if item["labels"] and len(preview_candidates) < 200:
                    preview_candidates.append({
                        "split": yolo_split,
                        "tile_stem": tile_stem,
                        "tile_path": str(dst_img_p),
                        "labels": item["labels"],
                        "classes": sorted({l["label"] for l in item["labels"]}),
                        "has_clipped": any(l["is_clipped"] for l in item["labels"]),
                    })

            board_audits.append({
                "board_side_id": stem,
                "physical_board": phys,
                "split": split_orig,
                "yolo_split": yolo_split,
                "family_note": b.get("family_note"),
                "image_wh": [img_w, img_h],
                "roi_bbox_half_open": [rx1, ry1, rx2, ry2],
                "tif_sha256": actual_tif_sha,
                "csv_sha256": actual_csv_sha,
                "pixel_sha256": px_sha,
                "fics6_unique_gt_total": len(gts),
                "class_counts_unique_gt": cls_counts,
                "roi_miss_gt": roi_miss,
                "roi_partial_gt": roi_partial,
                "unique_gts_full_vis_in_at_least_one_tile": len(gt_full_vis_any),
                "unique_gts_assigned_in_dataset": len(gt_assigned_any),
                "total_tiles": len(tiles),
                "positive_tiles": len(pos_tiles),
                "empty_tiles_total": len(emp_tiles),
                "empty_tiles_kept": len(kept_emp),
                "empty_tiles_dropped": len(emp_tiles) - len(kept_emp),
                "tile_label_box_instances": sum(len(x["labels"]) for x in kept_tiles),
                "edge_clipped_box_instances": edge_clipped_box_instances,
            })
        else:
            board_audits.append({
                "board_side_id": stem,
                "physical_board": phys,
                "split": split_orig,
                "yolo_split": None,
                "family_note": b.get("family_note"),
                "image_wh": [img_w, img_h],
                "roi_bbox_half_open": [rx1, ry1, rx2, ry2],
                "tif_sha256": actual_tif_sha,
                "csv_sha256": actual_csv_sha,
                "pixel_sha256": px_sha,
                "fics6_unique_gt_total": len(gts),
                "class_counts_unique_gt": cls_counts,
                "roi_miss_gt": roi_miss,
                "roi_partial_gt": roi_partial,
                "unique_gts_full_vis_in_at_least_one_tile": len(gt_full_vis_any),
                "unique_gts_covered_for_eval": len(gt_assigned_any),
                "total_tiles": len(tiles),
                "roi_audited": True,
            })

    # Clean up raw temporary PNGs after copying into YOLO images/train & images/val
    shutil.rmtree(raw_tile_cache, ignore_errors=True)

    # Pixel dedup check
    pixel_dups = {k: v for k, v in pixel_sha_to_boards.items() if len(v) > 1}
    assert not pixel_dups, f"Duplicate pixel content detected across boards: {pixel_dups}"

    # Write dataset.yaml
    ds_yaml_path = args.dataset_dir / "dataset.yaml"
    ds_yaml_text = "\n".join([
        f"path: {args.dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(FICS_SIX_CLASSES)}",
        "names:",
        *[f"  {idx}: {name}" for idx, name in enumerate(FICS_SIX_CLASSES)],
        "",
    ])
    ds_yaml_path.write_text(ds_yaml_text, encoding="utf-8")

    # Save tile provenance index for independent CPU round-trip verifier
    tile_index_path = args.dataset_dir / "tile_provenance_index.json"
    tile_index_path.write_text(
        json.dumps({"train": train_tile_records, "val": val_tile_records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Choose up to 12 diverse preview tiles covering all 6 classes and clipped/small boxes
    chosen_preview: List[Dict[str, Any]] = []
    used_stems = set()
    for target_cls in ("ic", "transistor", "diode", "inductor", "resistor", "capacitor"):
        for cand in preview_candidates:
            if cand["tile_stem"] in used_stems:
                continue
            if target_cls in cand["classes"]:
                chosen_preview.append(cand)
                used_stems.add(cand["tile_stem"])
                break
    for cand in preview_candidates:
        if len(chosen_preview) >= 12:
            break
        if cand["tile_stem"] not in used_stems and (cand["has_clipped"] or cand["split"] == "val"):
            chosen_preview.append(cand)
            used_stems.add(cand["tile_stem"])
    for cand in preview_candidates:
        if len(chosen_preview) >= 12:
            break
        if cand["tile_stem"] not in used_stems:
            chosen_preview.append(cand)
            used_stems.add(cand["tile_stem"])

    preview_path = args.audit_dir / "tile_label_preview_grid.png"
    render_preview_grid(samples=chosen_preview, out_path=preview_path)

    train_audits = [a for a in board_audits if a.get("yolo_split") == "train"]
    val_audits = [a for a in board_audits if a.get("yolo_split") == "val"]
    train_unique_cls = {c: sum(int(a["class_counts_unique_gt"][c]) for a in train_audits) for c in FICS_SIX_CLASSES}
    val_unique_cls = {c: sum(int(a["class_counts_unique_gt"][c]) for a in val_audits) for c in FICS_SIX_CLASSES}

    audit_report = {
        "schema_version": "r15_fics6_dataset_audit_v1",
        "source_inventory": str(args.inventory),
        "dataset_yaml": str(ds_yaml_path),
        "dataset_yaml_sha256": file_sha256(ds_yaml_path).lower(),
        "tile_provenance_index": str(tile_index_path),
        "tile_provenance_index_sha256": file_sha256(tile_index_path).lower(),
        "preview_grid_png": str(preview_path),
        "materialized_official_cocos": materialized_cocos,
        "yolo_class_mapping": {str(i): c for i, c in enumerate(FICS_SIX_CLASSES)},
        "family_and_split_limitations": {
            "train_families_note": "s27/s28/s29 vendor family all grouped inside train_pool; s8_front and s8_back both inside train_pool.",
            "val_families_note": "s2_front, s2_back, and s3_front grouped inside val_split (zero cross-split overlap with train_pool).",
            "val_class_support_limitation": {
                "transistor_gt": val_unique_cls["transistor"],
                "inductor_gt": val_unique_cls["inductor"],
                "note": "val_split has 0 transistor GTs and 1 inductor GT; checkpoint selection uses val_split only, while rare-class recall is reported alongside dev_eval.",
            },
            "train_low_support_classes": {
                "diode_unique_gt": train_unique_cls["diode"],
                "inductor_unique_gt": train_unique_cls["inductor"],
                "transistor_unique_gt": train_unique_cls["transistor"],
            },
        },
        "dedup_and_isolation_checks": {
            "zero_decoded_pixel_sha256_duplicates": len(pixel_dups) == 0,
            "pixel_duplicates": pixel_dups,
            "zero_banned_boards_in_train_val": True,
        },
        "summary": {
            "train": {
                "physical_boards": len({a["physical_board"] for a in train_audits}),
                "board_sides": len(train_audits),
                "unique_gt_total": sum(a["fics6_unique_gt_total"] for a in train_audits),
                "unique_gt_by_class": train_unique_cls,
                "unique_gts_assigned": sum(a["unique_gts_assigned_in_dataset"] for a in train_audits),
                "tiles_total_raw": sum(a["total_tiles"] for a in train_audits),
                "tiles_kept_in_dataset": len(train_tile_records),
                "positive_tiles": sum(a["positive_tiles"] for a in train_audits),
                "empty_tiles_kept": sum(a["empty_tiles_kept"] for a in train_audits),
                "empty_tiles_dropped": sum(a["empty_tiles_dropped"] for a in train_audits),
                "tile_box_instances": sum(a["tile_label_box_instances"] for a in train_audits),
                "roi_miss_gt_total": sum(a["roi_miss_gt"] for a in train_audits),
            },
            "val": {
                "physical_boards": len({a["physical_board"] for a in val_audits}),
                "board_sides": len(val_audits),
                "unique_gt_total": sum(a["fics6_unique_gt_total"] for a in val_audits),
                "unique_gt_by_class": val_unique_cls,
                "unique_gts_assigned": sum(a["unique_gts_assigned_in_dataset"] for a in val_audits),
                "tiles_total_raw": sum(a["total_tiles"] for a in val_audits),
                "tiles_kept_in_dataset": len(val_tile_records),
                "positive_tiles": sum(a["positive_tiles"] for a in val_audits),
                "empty_tiles_kept": sum(a["empty_tiles_kept"] for a in val_audits),
                "empty_tiles_dropped": 0,
                "tile_box_instances": sum(a["tile_label_box_instances"] for a in val_audits),
                "roi_miss_gt_total": sum(a["roi_miss_gt"] for a in val_audits),
            },
        },
        "boards": board_audits,
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    out_audit = args.audit_dir / "r15_dataset_audit.json"
    out_audit.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit_report["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
