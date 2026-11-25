"""Materialize and audit the isolated Round 17 FICS-PCB 6-class YOLO11n expanded dataset.

Strict invariants:
- Combines R15 train_pool (9 sides / 8 boards / 812 GT) + R17 audited official FICS expansion
  (s14 x4, s15 x4, s26 x4, s25_front, s23_front -> 14 sides / 5 boards) = 23 sides / 13 boards.
- Keeps val_split EXACTLY frozen from R15 (s2_front, s2_back, s3_front -> 3 sides / 2 boards / 158 GT / 379 tiles).
- Hard-bans dev (s20, s4, s1), sealed/blind (s31, s10, s16, s19, s12, s24), WACV overlap (s9),
  and family-excluded boards (s17, s11, s5, s6, s7).
- Uses identical inference tiling geometry (`compute_board_roi_and_tiles`, 640px, 20% overlap, white_thresh=248, roi_pad=32).
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

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import (  # noqa: E402
    FICS_CATEGORY_NAME_TO_ID,
    FICS_RAW_TO_CANONICAL,
    file_sha256,
    parse_component_location,
    validate_box_within_image,
)
from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES  # noqa: E402
from pcb_vlm.evaluation.gt_free_pipeline import compute_board_roi_and_tiles  # noqa: E402

DEFAULT_R14_INVENTORY = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json"
)
DEFAULT_R17_EXPANSION_AUDIT = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\data_audit\r17_data_expansion_audit.json"
)
DEFAULT_R17_BLIND_MANIFEST = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\codex_reserved_blind\r17_codex_blind_manifest.json"
)
DEFAULT_DATASET_DIR = pathlib.Path(
    r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration17-20261124\fics_yolo_dataset_r17"
)
DEFAULT_AUDIT_DIR = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\data_audit"
)

YOLO_FICS6_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(FICS_SIX_CLASSES)}
BANNED_BOARDS = {
    "s20", "s4", "s1",
    "s31", "s10", "s16", "s19", "s9",
    "s12", "s24", "s17", "s11", "s5", "s6", "s7",
}


def read_fics_csv_gts(csv_path: pathlib.Path, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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


def decoded_rgb_sha256(img_path: pathlib.Path) -> Tuple[str, Tuple[int, int]]:
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
        kept.append({
            "gt_index": int(g["gt_index"]),
            "label": str(g["label"]),
            "yolo_cls_id": int(g["yolo_cls_id"]),
            "local_bbox_xyxy": [round(lx1, 3), round(ly1, 3), round(lx2, 3), round(ly2, 3)],
            "yolo_norm": [round(cx_n, 8), round(cy_n, 8), round(w_n, 8), round(h_n, 8)],
            "visible_frac": round(vis_frac, 4),
            "is_clipped": vis_frac < 0.999,
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
        draw.text((ox, oy), f"{s['tile_stem']} (n={len(s['labels'])})", fill=(226, 232, 240), font=font)
        canvas.paste(resized, (ox, oy + hdr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build & audit Round 17 expanded FICS 6-class YOLO11n dataset.")
    parser.add_argument("--r14-inventory", type=pathlib.Path, default=DEFAULT_R14_INVENTORY)
    parser.add_argument("--r17-audit", type=pathlib.Path, default=DEFAULT_R17_EXPANSION_AUDIT)
    parser.add_argument("--r17-blind", type=pathlib.Path, default=DEFAULT_R17_BLIND_MANIFEST)
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--audit-dir", type=pathlib.Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    r14_inv = json.loads(args.r14_inventory.read_text(encoding="utf-8"))
    r17_exp = json.loads(args.r17_audit.read_text(encoding="utf-8"))
    r17_blind = json.loads(args.r17_blind.read_text(encoding="utf-8"))

    # Combine R14/R15 boards + R17 expansion boards (keep blind boards for pixel dedup check only)
    boards: List[Dict[str, Any]] = []
    for b in r14_inv["boards"]:
        boards.append(dict(b))
    for b in r17_exp["train_expansion_sides"]:
        boards.append({
            "board_side_id": b["board_side_id"],
            "physical_board": b["physical_board"],
            "split": "train_pool",
            "family_note": f"R17 expansion ({b['physical_family']}): {b['family_proof']}",
            "tif_path": b["tif_path"],
            "tif_sha256": b["tif_sha256"],
            "csv_path": b["csv_path"],
            "csv_sha256": b["csv_sha256"],
            "fics6_gt_total": b["fics6_gt_total"],
            "class_counts": b["class_counts"],
            "is_r17_new": True,
        })

    if args.dataset_dir.exists():
        shutil.rmtree(args.dataset_dir)
    for split in ("train", "val"):
        (args.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    raw_tile_cache = args.dataset_dir / "_raw_tile_work"
    raw_tile_cache.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    pixel_sha_to_boards: Dict[str, List[Dict[str, str]]] = {}
    for bb in r17_blind["boards"]:
        pixel_sha_to_boards.setdefault(str(bb["pixel_sha256"]), []).append(
            {"board_side_id": str(bb["board_side_id"]), "split": "codex_blind", "physical_board": str(bb["physical_board"])}
        )

    board_audits: List[Dict[str, Any]] = []
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
        pixel_sha_to_boards.setdefault(px_sha, []).append(
            {"board_side_id": stem, "split": split_orig, "physical_board": phys}
        )

        gts, cls_counts = read_fics_csv_gts(csv_p, img_w, img_h)
        assert len(gts) == int(b["fics6_gt_total"]), f"GT count mismatch for {stem}"
        assert cls_counts == dict(b["class_counts"]), f"Class counts mismatch for {stem}"

        should_tile_dataset = split_orig in ("train_pool", "val_split")
        should_audit_roi = should_tile_dataset or split_orig in ("dev_eval_r14", "smoke_only_unused_r14")
        if not should_audit_roi:
            board_audits.append({
                "board_side_id": stem,
                "physical_board": phys,
                "split": split_orig,
                "yolo_split": None,
                "image_wh": [img_w, img_h],
                "tif_sha256": actual_tif_sha,
                "csv_sha256": actual_csv_sha,
                "pixel_sha256": px_sha,
                "fics6_unique_gt_total": len(gts),
                "class_counts_unique_gt": cls_counts,
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
            ia = max(0.0, min(gx2, float(rx2)) - max(gx1, float(rx1))) * max(
                0.0, min(gy2, float(ry2)) - max(gy1, float(ry1))
            )
            if ia <= 0.0:
                roi_miss += 1
            elif (ia / ga) < 0.999:
                roi_partial += 1

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

                if b.get("is_r17_new") and item["labels"]:
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
                "is_r17_new": bool(b.get("is_r17_new")),
                "family_note": b.get("family_note"),
                "image_wh": [img_w, img_h],
                "roi_bbox_half_open": [rx1, ry1, rx2, ry2],
                "tif_sha256": actual_tif_sha,
                "csv_sha256": actual_csv_sha,
                "csv_path": str(csv_p),
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

    shutil.rmtree(raw_tile_cache, ignore_errors=True)

    pixel_dups = {k: v for k, v in pixel_sha_to_boards.items() if len(v) > 1}
    assert not pixel_dups, f"Duplicate pixel content detected across boards: {pixel_dups}"

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

    tile_index_path = args.dataset_dir / "tile_provenance_index.json"
    tile_index_path.write_text(
        json.dumps({"train": train_tile_records, "val": val_tile_records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    chosen_preview: List[Dict[str, Any]] = []
    used_stems = set()
    for target_cls in ("diode", "transistor", "ic", "inductor"):
        for cand in preview_candidates:
            if len(chosen_preview) >= 12:
                break
            if cand["tile_stem"] in used_stems:
                continue
            if target_cls in cand["classes"]:
                chosen_preview.append(cand)
                used_stems.add(cand["tile_stem"])
                if sum(1 for x in chosen_preview if target_cls in x["classes"]) >= 2:
                    break
    for cand in preview_candidates:
        if len(chosen_preview) >= 12:
            break
        if cand["tile_stem"] not in used_stems:
            chosen_preview.append(cand)
            used_stems.add(cand["tile_stem"])

    preview_path = args.audit_dir / "r17_tile_label_preview_grid.png"
    render_preview_grid(samples=chosen_preview, out_path=preview_path)

    train_audits = [a for a in board_audits if a.get("yolo_split") == "train"]
    val_audits = [a for a in board_audits if a.get("yolo_split") == "val"]
    train_unique_cls = {c: sum(int(a["class_counts_unique_gt"][c]) for a in train_audits) for c in FICS_SIX_CLASSES}
    val_unique_cls = {c: sum(int(a["class_counts_unique_gt"][c]) for a in val_audits) for c in FICS_SIX_CLASSES}

    audit_report = {
        "schema_version": "r17_fics6_dataset_audit_v1",
        "source_r14_inventory": str(args.r14_inventory),
        "source_r17_expansion_audit": str(args.r17_audit),
        "dataset_yaml": str(ds_yaml_path),
        "dataset_yaml_sha256": file_sha256(ds_yaml_path).lower(),
        "tile_provenance_index": str(tile_index_path),
        "tile_provenance_index_sha256": file_sha256(tile_index_path).lower(),
        "preview_grid_png": str(preview_path),
        "yolo_class_mapping": {str(i): c for i, c in enumerate(FICS_SIX_CLASSES)},
        "dedup_and_isolation_checks": {
            "zero_decoded_pixel_sha256_duplicates": len(pixel_dups) == 0,
            "pixel_duplicates": pixel_dups,
            "zero_banned_boards_in_train_val": True,
            "banned_boards": sorted(BANNED_BOARDS),
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
    out_audit = args.audit_dir / "r17_dataset_audit.json"
    out_audit.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit_report["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
