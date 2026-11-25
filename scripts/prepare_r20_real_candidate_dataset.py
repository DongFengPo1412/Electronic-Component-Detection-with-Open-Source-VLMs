"""Build Round 20 real-candidate dataset from frozen R15 + R19 A1 outputs on legal train/cal boards only."""

from __future__ import annotations

import collections
import csv
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw
import torch

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.data.fics_dslr import FICS_RAW_TO_CANONICAL, parse_component_location  # noqa: E402
from pcb_vlm.evaluation.composite_detector import compute_iou_xyxy  # noqa: E402
from pcb_vlm.evaluation.crop_verifier_r18 import (  # noqa: E402
    extract_square_context_crop,
    load_r18_verifier_checkpoint,
)
from pcb_vlm.evaluation.round19_postproc import (  # noqa: E402
    R19_A1_M_REJECT_OTHER,
    R19_A1_T_REJECT_OTHER,
    apply_r19_a1_ic_reject_only,
    compute_r19_crop_probs_for_boxes,
)

R18_MANIFEST = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration18-20261125\crop_dataset_r18\crop_manifest.json")
R18_VERIFIER_WEIGHTS = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\weights\r18_resnet18_crop_verifier_best.pt")
R18_VERIFIER_SHA = "c27119eac5b98d6f8e56a61b280da94ce30bbed4443bb0cfe57c380d46379ca3"

R20_CACHE = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration20-20261125")
R20_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration20-20261125")
FROZEN_PLAN_PATH = R20_ART / "frozen_plan_r20_pre_train.json"

SOURCE_WEIGHTS = {
    "official_r15_a1_ic_miscls_dq": 6.0,
    "low_conf_r15_ic_miscls_dq": 3.0,
    "official_r15_a1_true_ic": 2.5,
    "official_r15_a1_other": 2.0,
    "train_gt_crop_supplement": 1.0,
}


def _read_csv_gts(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    raw_lower = {k.lower(): v for k, v in FICS_RAW_TO_CANONICAL.items()}
    gts: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rt = str(row.get("component_type") or "").strip()
            canon = FICS_RAW_TO_CANONICAL.get(rt) or raw_lower.get(rt.lower())
            if not canon:
                continue
            _, box, _ = parse_component_location(row.get("component_location") or "")
            gts.append({"label": canon, "bbox_xyxy": [float(v) for v in box]})
    return gts


def _box_min_side(box: Sequence[float]) -> float:
    return min(max(0.0, float(box[2]) - float(box[0])), max(0.0, float(box[3]) - float(box[1])))


def _match_boxes_to_gt_strict(
    boxes: Sequence[Mapping[str, Any]],
    gts: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """One-to-one greedy match of IC candidates to ALL GT, discarding tiny (<8px) or multi-label conflicts."""
    discard_counts = {"tiny_box_lt8px": 0, "multi_label_gt_conflict": 0}
    sorted_boxes = sorted(boxes, key=lambda p: (-float(p.get("score", 0.0)), str(p.get("prediction_id") or "")))
    used_gt = set()
    matched: List[Dict[str, Any]] = []

    for p in sorted_boxes:
        box = [float(v) for v in p["bbox_xyxy"]]
        if _box_min_side(box) < 8.0:
            discard_counts["tiny_box_lt8px"] += 1
            continue

        # Check multi-label conflict: overlaps IoU >= 0.50 with >=2 distinct GT classes
        overlapping_classes = set()
        for g in gts:
            if compute_iou_xyxy(box, g["bbox_xyxy"]) >= 0.50:
                overlapping_classes.add(str(g["label"]))
        if len(overlapping_classes) >= 2:
            discard_counts["multi_label_gt_conflict"] += 1
            continue

        best_gi, best_iou = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in used_gt:
                continue
            iou = compute_iou_xyxy(box, g["bbox_xyxy"])
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_gi >= 0 and best_iou >= 0.50:
            used_gt.add(best_gi)
            gt_lbl = str(gts[best_gi]["label"])
            target_lbl = gt_lbl if gt_lbl in ("ic", "diode", "transistor") else "other"
            matched.append({
                "pred": p,
                "gt_label": gt_lbl,
                "target_label": target_lbl,
                "gt_iou": round(best_iou, 4),
            })
        else:
            matched.append({
                "pred": p,
                "gt_label": "background",
                "target_label": "other",
                "gt_iou": 0.0,
            })
    return matched, discard_counts


def _render_audit_grid(samples: Sequence[Mapping[str, Any]], out_path: pathlib.Path) -> None:
    cols = 4
    rows = max(1, (len(samples) + cols - 1) // cols)
    cell_w, cell_h = 240, 260
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 28, 34))
    draw = ImageDraw.Draw(canvas)
    for idx, s in enumerate(samples[: cols * rows]):
        r, col = divmod(idx, cols)
        x0, y0 = col * cell_w, r * cell_h
        with Image.open(s["crop_path"]) as im:
            canvas.paste(im.convert("RGB"), (x0 + 8, y0 + 32))
        draw.text((x0 + 8, y0 + 4), f"{s['split']} {s['stem']} -> {s['label']}", fill=(240, 240, 240))
        draw.text((x0 + 8, y0 + 16), f"{s['source']}", fill=(130, 220, 170))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    t0 = time.perf_counter()
    plan = json.loads(FROZEN_PLAN_PATH.read_text(encoding="utf-8"))
    split_tbl = plan["disjoint_family_split_table"]

    train_stems = {s: fam for fam, stems in split_tbl["train_families"].items() for s in stems}
    cal_primary_stems = {s: fam for fam, stems in split_tbl["calibration_primary"].items() for s in stems}
    cal_secondary_stems = {s: f"fam_{s.split('_')[0]}" for s in split_tbl["calibration_secondary_twin_views_audit_only"]}

    all_legal = set(train_stems) | set(cal_primary_stems) | set(cal_secondary_stems)
    manifest = json.loads(R18_MANIFEST.read_text(encoding="utf-8"))
    prop_idx = manifest["proposal_cache_index"]

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    r18_model, _, actual_sha = load_r18_verifier_checkpoint(
        R18_VERIFIER_WEIGHTS, expected_sha256=R18_VERIFIER_SHA, device=dev
    )
    assert actual_sha == R18_VERIFIER_SHA

    crop_dir = R20_CACHE / "dataset_r20" / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    train_records: List[Dict[str, Any]] = []
    cal_records: List[Dict[str, Any]] = []
    discard_totals = collections.Counter()
    support_by_family: Dict[str, Any] = collections.defaultdict(lambda: collections.Counter())

    for stem in sorted(all_legal):
        entry = prop_idx[stem]
        tif_path = pathlib.Path(str(entry["tif_path"]))
        csv_path = pathlib.Path(str(entry["csv_path"]))
        preds_025 = json.loads(pathlib.Path(str(entry["preds_025_path"])).read_text(encoding="utf-8"))
        pool_005 = json.loads((pathlib.Path(str(entry["preds_025_path"])).parent / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        gts = _read_csv_gts(csv_path)

        # Run frozen R19 A1 on official_025 IC boxes
        ic_official = [p for p in preds_025 if str(p.get("label") or "").lower() == "ic"]
        r18_probs = compute_r19_crop_probs_for_boxes(image_path=tif_path, boxes=ic_official, model=r18_model, device=dev)
        a1_out = apply_r19_a1_ic_reject_only(
            official_preds_025=preds_025,
            probs_by_pred_id=r18_probs,
            t_reject_other=R19_A1_T_REJECT_OTHER,
            m_reject_other=R19_A1_M_REJECT_OTHER,
        )
        a1_ic_boxes = [p for p in a1_out["predictions"] if str(p.get("label") or "").lower() == "ic"]
        matched_official, disc_off = _match_boxes_to_gt_strict(a1_ic_boxes, gts)
        for k, v in disc_off.items():
            discard_totals[k] += v

        if stem in train_stems:
            split = "train"
            fam = train_stems[stem]
        elif stem in cal_primary_stems:
            split = "cal_primary"
            fam = cal_primary_stems[stem]
        else:
            split = "cal_secondary"
            fam = cal_secondary_stems[stem]

        with Image.open(tif_path) as im_raw:
            rgb = im_raw.convert("RGB")

            # 1) Save all A1-surviving official IC candidates (both train & cal)
            for idx_m, item in enumerate(matched_official):
                p = item["pred"]
                pid = str(p.get("prediction_id") or f"{stem}_ic_{idx_m}")
                tgt = item["target_label"]
                if tgt in ("diode", "transistor"):
                    src = "official_r15_a1_ic_miscls_dq"
                elif tgt == "ic":
                    src = "official_r15_a1_true_ic"
                else:
                    src = "official_r15_a1_other"
                c_path = crop_dir / split / f"{stem}__{pid}__off.png"
                c_path.parent.mkdir(parents=True, exist_ok=True)
                extract_square_context_crop(rgb, p["bbox_xyxy"]).save(c_path)
                rec = {
                    "split": split,
                    "stem": stem,
                    "family": fam,
                    "prediction_id": pid,
                    "label": tgt,
                    "gt_label": item["gt_label"],
                    "gt_iou": item["gt_iou"],
                    "r15_score": float(p.get("score", 0.0)),
                    "source": src,
                    "sample_weight": SOURCE_WEIGHTS[src],
                    "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
                    "crop_path": str(c_path),
                    "r18_probs": dict(r18_probs[pid]),
                }
                if split == "train":
                    train_records.append(rec)
                else:
                    cal_records.append(rec)
                support_by_family[f"{split}:{fam}"][f"{src}:{tgt}"] += 1

            # 2) Train-only supplements: low-conf IC->D/Q + train GT crops
            if split == "train":
                low_ic = [
                    p for p in pool_005
                    if str(p.get("label") or "").lower() == "ic" and float(p.get("score", 0.0)) < 0.25
                ]
                matched_low, disc_low = _match_boxes_to_gt_strict(low_ic, gts)
                for k, v in disc_low.items():
                    discard_totals[k] += v
                for idx_l, item in enumerate(matched_low):
                    if item["target_label"] not in ("diode", "transistor"):
                        continue
                    p = item["pred"]
                    pid = str(p.get("prediction_id") or f"{stem}_low_{idx_l}")
                    c_path = crop_dir / "train" / f"{stem}__{pid}__low.png"
                    extract_square_context_crop(rgb, p["bbox_xyxy"]).save(c_path)
                    train_records.append({
                        "split": "train",
                        "stem": stem,
                        "family": fam,
                        "prediction_id": pid,
                        "label": item["target_label"],
                        "gt_label": item["gt_label"],
                        "gt_iou": item["gt_iou"],
                        "r15_score": float(p.get("score", 0.0)),
                        "source": "low_conf_r15_ic_miscls_dq",
                        "sample_weight": SOURCE_WEIGHTS["low_conf_r15_ic_miscls_dq"],
                        "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
                        "crop_path": str(c_path),
                    })
                    support_by_family[f"train:{fam}"][f"low_conf_r15_ic_miscls_dq:{item['target_label']}"] += 1

                # Train-only GT crops for ic/diode/transistor
                for gi, g in enumerate(gts):
                    if g["label"] not in ("ic", "diode", "transistor"):
                        continue
                    if _box_min_side(g["bbox_xyxy"]) < 8.0:
                        discard_totals["tiny_box_lt8px"] += 1
                        continue
                    c_path = crop_dir / "train" / f"{stem}__gt_{gi}__{g['label']}.png"
                    extract_square_context_crop(rgb, g["bbox_xyxy"]).save(c_path)
                    train_records.append({
                        "split": "train",
                        "stem": stem,
                        "family": fam,
                        "prediction_id": f"{stem}_gt_{gi}",
                        "label": g["label"],
                        "gt_label": g["label"],
                        "gt_iou": 1.0,
                        "r15_score": 1.0,
                        "source": "train_gt_crop_supplement",
                        "sample_weight": SOURCE_WEIGHTS["train_gt_crop_supplement"],
                        "bbox_xyxy": list(g["bbox_xyxy"]),
                        "crop_path": str(c_path),
                    })
                    support_by_family[f"train:{fam}"][f"train_gt_crop_supplement:{g['label']}"] += 1

    # Save manifest & audit
    manifest_out = R20_CACHE / "dataset_r20" / "r20_dataset_manifest.json"
    manifest_out.write_text(
        json.dumps({"train": train_records, "cal": cal_records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_by_cls = dict(collections.Counter(r["label"] for r in train_records))
    train_by_src = dict(collections.Counter(r["source"] for r in train_records))
    cal_prim_by_cls = dict(collections.Counter(r["label"] for r in cal_records if r["split"] == "cal_primary"))
    cal_sec_by_cls = dict(collections.Counter(r["label"] for r in cal_records if r["split"] == "cal_secondary"))

    audit_doc = {
        "schema_version": "r20_real_candidate_dataset_audit_v1",
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "train_total": len(train_records),
        "train_by_class": train_by_cls,
        "train_by_source": train_by_src,
        "cal_primary_total": sum(1 for r in cal_records if r["split"] == "cal_primary"),
        "cal_primary_by_class": cal_prim_by_cls,
        "cal_secondary_total": sum(1 for r in cal_records if r["split"] == "cal_secondary"),
        "cal_secondary_by_class": cal_sec_by_cls,
        "discard_counts": dict(discard_totals),
        "support_by_family": {k: dict(v) for k, v in sorted(support_by_family.items())},
        "zero_dev_or_blind_overlap": True,
    }
    (R20_ART / "data_audit" / "r20_real_candidate_dataset_audit.json").write_text(
        json.dumps(audit_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Render audit grid prioritizing real misclassified IC->DQ and cal primary candidates
    grid_samples: List[Mapping[str, Any]] = []
    grid_samples.extend([r for r in train_records if r["source"] == "official_r15_a1_ic_miscls_dq"][:4])
    grid_samples.extend([r for r in cal_records if r["split"] == "cal_primary" and r["label"] in ("diode", "transistor")][:4])
    grid_samples.extend([r for r in cal_records if r["split"] == "cal_primary" and r["label"] == "ic"][:4])
    grid_samples.extend([r for r in cal_records if r["split"] == "cal_primary" and r["label"] == "other"][:4])
    _render_audit_grid(grid_samples, R20_ART / "data_audit" / "r20_train_cal_crop_audit.png")

    print(json.dumps({
        "wall_seconds": audit_doc["wall_seconds"],
        "train_total": audit_doc["train_total"],
        "train_by_class": train_by_cls,
        "train_by_source": train_by_src,
        "cal_primary_total": audit_doc["cal_primary_total"],
        "cal_primary_by_class": cal_prim_by_cls,
        "discard_counts": dict(discard_totals),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
