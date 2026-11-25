"""Train lightweight class-balanced head on frozen ImageNet ResNet-18 features and run R20 calibration gate."""

from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.crop_verifier_r18 import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from pcb_vlm.evaluation.round20_ic_relabel import (  # noqa: E402
    R20_CLASSES,
    R20_CLASS_TO_IDX,
    R20LightweightHead,
    apply_r20_targeted_ic_relabel,
    build_frozen_r18_backbone,
    extract_dihedral_avg_features,
)

IMAGENET_CKPT = pathlib.Path(r"C:\Users\Liu\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth")
R18_MANIFEST = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration18-20261125\crop_dataset_r18\crop_manifest.json")
R20_CACHE = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration20-20261125")
R20_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration20-20261125")
DATASET_MANIFEST = R20_CACHE / "dataset_r20" / "r20_dataset_manifest.json"
FROZEN_PLAN = R20_ART / "frozen_plan_r20_pre_train.json"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def _extract_8way_dihedral_features(
    backbone: nn.Module,
    crop_paths: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Return [N, 8, 512] L2-normalized ResNet-18 features for all 8 dihedral symmetries."""
    feats: List[torch.Tensor] = []
    for i in range(0, len(crop_paths), 32):
        chunk_imgs = []
        for cp in crop_paths[i : i + 32]:
            with Image.open(cp) as im:
                chunk_imgs.append(im.convert("RGB"))
        sym_list: List[torch.Tensor] = []
        for flip in (False, True):
            for k in (0, 1, 2, 3):
                batch_t = []
                for img in chunk_imgs:
                    im_v = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else img
                    if k > 0:
                        im_v = im_v.rotate(90 * k)
                    t = TF.normalize(TF.to_tensor(im_v), mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
                    batch_t.append(t)
                xb = torch.stack(batch_t, dim=0).to(device)
                fb = F.normalize(backbone(xb), dim=-1).detach().cpu()
                sym_list.append(fb)
        # [B, 8, 512]
        feats.append(torch.stack(sym_list, dim=1))
    return torch.cat(feats, dim=0)


@torch.inference_mode()
def _extract_4rot_avg_features(
    backbone: nn.Module,
    crop_paths: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    crops = []
    for cp in crop_paths:
        with Image.open(cp) as im:
            crops.append(im.convert("RGB"))
    return extract_dihedral_avg_features(backbone, crops, device=device, batch_size=32)


def main() -> None:
    t0 = time.perf_counter()
    _seed_all(42)
    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    ds = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    train_recs = list(ds["train"])
    cal_recs = list(ds["cal"])

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = build_frozen_r18_backbone(imagenet_ckpt_path=IMAGENET_CKPT, device=dev)

    feat_cache = R20_CACHE / "features_cache" / "r20_features.pt"
    if feat_cache.is_file():
        fc = torch.load(str(feat_cache), map_location="cpu", weights_only=True)
        train_feats_8way = fc["train_feats_8way"]
        cal_feats_4avg = fc["cal_feats_4avg"]
    else:
        train_feats_8way = _extract_8way_dihedral_features(backbone, [r["crop_path"] for r in train_recs], dev)
        cal_feats_4avg = _extract_4rot_avg_features(backbone, [r["crop_path"] for r in cal_recs], dev)
        torch.save({"train_feats_8way": train_feats_8way, "cal_feats_4avg": cal_feats_4avg}, str(feat_cache))

    # Build balanced sample weights = inverse-sqrt(class_count) * source_priority_weight
    cls_counts = collections.Counter(r["label"] for r in train_recs)
    cls_w = {c: 1.0 / math.sqrt(max(1, cls_counts[c])) for c in R20_CLASSES}
    mean_cw = sum(cls_w.values()) / len(R20_CLASSES)
    cls_w = {c: w / mean_cw for c, w in cls_w.items()}

    y_train = torch.tensor([R20_CLASS_TO_IDX[r["label"]] for r in train_recs], dtype=torch.long)
    w_train = torch.tensor([float(r["sample_weight"]) * cls_w[r["label"]] for r in train_recs], dtype=torch.float32)
    w_train = w_train / w_train.mean()

    head = R20LightweightHead(in_dim=512, hidden_dim=128, num_classes=len(R20_CLASSES), dropout=0.30).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    y_dev = y_train.to(dev)
    w_dev = w_train.to(dev)
    loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.05)

    history: List[Dict[str, Any]] = []
    for ep in range(1, epochs + 1):
        head.train()
        # Sample one random dihedral view per item each epoch
        view_idx = torch.randint(0, 8, (len(train_recs),))
        xb = train_feats_8way[torch.arange(len(train_recs)), view_idx].to(dev)
        optimizer.zero_grad(set_to_none=True)
        logits = head(xb)
        per_item_loss = loss_fn(logits, y_dev)
        loss = (per_item_loss * w_dev).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if ep in (1, 20, 40, 60):
            history.append({"epoch": ep, "weighted_train_loss": round(float(loss.item()), 5)})

    head.eval()
    weights_out = R20_ART / "weights" / "r20_ic_relabel_head_best.pt"
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "r20_ic_relabel_head_v1",
            "classes": list(R20_CLASSES),
            "in_dim": 512,
            "hidden_dim": 128,
            "dropout": 0.30,
            "epochs": epochs,
            "seed": 42,
            "imagenet_backbone_sha256": file_sha256(IMAGENET_CKPT).lower(),
            "head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        },
        str(weights_out),
    )
    head_sha = file_sha256(weights_out).lower()

    train_summary = {
        "schema_version": "r20_head_train_summary_v1",
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "epochs": epochs,
        "seed": 42,
        "train_total": len(train_recs),
        "train_by_class": dict(cls_counts),
        "weights_path": str(weights_out),
        "weights_sha256": head_sha,
        "history": history,
    }
    (R20_ART / "training" / "r20_head_train_summary.json").write_text(
        json.dumps(train_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Score all calibration candidates once with R20 head
    with torch.inference_mode():
        cal_logits = head(cal_feats_4avg.to(dev))
        cal_probs_rows = torch.softmax(cal_logits, dim=-1).detach().cpu().tolist()

    r20_probs_by_stem: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
    r18_probs_by_stem: Dict[str, Dict[str, Dict[str, float]]] = collections.defaultdict(dict)
    gt_by_stem_pid: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    cal_stems_by_split: Dict[str, List[str]] = {"cal_primary": [], "cal_secondary": []}
    cal_rec_by_stem_pid: Dict[Tuple[str, str], Mapping[str, Any]] = {}

    for rec, prow in zip(cal_recs, cal_probs_rows):
        stem = str(rec["stem"])
        pid = str(rec["prediction_id"])
        split = str(rec["split"])
        if stem not in cal_stems_by_split[split]:
            cal_stems_by_split[split].append(stem)
        r20_probs_by_stem[stem][pid] = {c: round(float(p), 6) for c, p in zip(R20_CLASSES, prow)}
        r18_probs_by_stem[stem][pid] = dict(rec["r18_probs"])
        gt_by_stem_pid[stem][pid] = str(rec["label"])
        cal_rec_by_stem_pid[(stem, pid)] = rec

    # Load actual A1-retained predictions per cal stem (preserving untouched D/Q boxes for overlap abstention)
    r18_manifest = json.loads(R18_MANIFEST.read_text(encoding="utf-8"))
    prop_idx = r18_manifest["proposal_cache_index"]
    a1_preds_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for stem in cal_stems_by_split["cal_primary"] + cal_stems_by_split["cal_secondary"]:
        preds_025 = json.loads(pathlib.Path(str(prop_idx[stem]["preds_025_path"])).read_text(encoding="utf-8"))
        kept_ic_pids = set(r20_probs_by_stem[stem].keys())
        a1_preds_by_stem[stem] = [
            p for p in preds_025
            if str(p.get("label") or "").lower() != "ic" or str(p.get("prediction_id") or "") in kept_ic_pids
        ]

    def _eval_cal(stems: Sequence[str], probs_map: Mapping[str, Mapping[str, Mapping[str, float]]], t_rel: float, m_rel: float) -> Dict[str, Any]:
        true_ic_corrupted = 0
        miscls_dq_corrected = 0
        non_dq_flipped_to_dq = 0
        s2_s3_absent_transistor_fp = 0
        overlap_abstained = 0
        per_stem: Dict[str, Any] = {}

        for stem in stems:
            out = apply_r20_targeted_ic_relabel(
                a1_predictions=a1_preds_by_stem[stem],
                probs_by_pred_id=probs_map[stem],
                t_relabel=t_rel,
                m_relabel=m_rel,
            )
            overlap_abstained += out["summary"]["abstain_same_class_overlap"]
            s_corr = s_true_bad = s_other_bad = 0
            for rb in out["relabeled_boxes"]:
                pid = str(rb["prediction_id"])
                new_lbl = str(rb["new_label"])
                gt_tgt = gt_by_stem_pid[stem].get(pid, "other")
                if gt_tgt == new_lbl and gt_tgt in ("diode", "transistor"):
                    miscls_dq_corrected += 1
                    s_corr += 1
                else:
                    non_dq_flipped_to_dq += 1
                    if gt_tgt == "ic":
                        true_ic_corrupted += 1
                        s_true_bad += 1
                    else:
                        s_other_bad += 1
                    if stem.startswith(("s2_", "s3_")) and new_lbl == "transistor":
                        s2_s3_absent_transistor_fp += 1
            per_stem[stem] = {
                "relabeled_count": len(out["relabeled_boxes"]),
                "miscls_dq_corrected": s_corr,
                "true_ic_corrupted": s_true_bad,
                "non_dq_flipped": s_other_bad,
                "overlap_abstained": out["summary"]["abstain_same_class_overlap"],
            }
        return {
            "true_ic_corrupted": true_ic_corrupted,
            "miscls_dq_corrected": miscls_dq_corrected,
            "non_dq_flipped_to_dq": non_dq_flipped_to_dq,
            "net_dq_gain": miscls_dq_corrected - non_dq_flipped_to_dq,
            "s2_s3_absent_transistor_fp_added": s2_s3_absent_transistor_fp,
            "overlap_abstained": overlap_abstained,
            "per_stem": per_stem,
        }

    # Sweep 15-point grid for R20 + R18 comparison
    grid_rows: List[Dict[str, Any]] = []
    passing_rows: List[Dict[str, Any]] = []
    for cfg in plan["calibration_grid"]:
        t_rel = float(cfg["t_relabel"])
        m_rel = float(cfg["m_relabel"])
        r20_prim = _eval_cal(cal_stems_by_split["cal_primary"], r20_probs_by_stem, t_rel, m_rel)
        r20_sec = _eval_cal(cal_stems_by_split["cal_secondary"], r20_probs_by_stem, t_rel, m_rel)
        r18_prim = _eval_cal(cal_stems_by_split["cal_primary"], r18_probs_by_stem, t_rel, m_rel)

        passes = (
            r20_prim["true_ic_corrupted"] == 0
            and r20_prim["miscls_dq_corrected"] >= 1
            and r20_prim["s2_s3_absent_transistor_fp_added"] == 0
            and r20_prim["net_dq_gain"] >= 0
        )
        row = {
            "rule": cfg,
            "passes_gate": passes,
            "r20_cal_primary": r20_prim,
            "r20_cal_secondary_audit": r20_sec,
            "r18_baseline_cal_primary": r18_prim,
        }
        grid_rows.append(row)
        if passes:
            passing_rows.append(row)

    passing_rows.sort(
        key=lambda r: (
            r["r20_cal_primary"]["net_dq_gain"],
            r["r20_cal_primary"]["miscls_dq_corrected"],
            -r["r20_cal_primary"]["non_dq_flipped_to_dq"],
            float(r["rule"]["t_relabel"]),
            float(r["rule"]["m_relabel"]),
        ),
        reverse=True,
    )

    gate_passed = len(passing_rows) > 0
    selected = passing_rows[0] if gate_passed else None

    cal_report = {
        "schema_version": "r20_calibration_report_v1",
        "calibration_primary_stems": cal_stems_by_split["cal_primary"],
        "calibration_secondary_stems": cal_stems_by_split["cal_secondary"],
        "grid_results": grid_rows,
        "passing_count": len(passing_rows),
        "calibration_gate_passed": gate_passed,
        "selected": selected,
    }
    cal_path = R20_ART / "calibration" / "r20_calibration_report.json"
    cal_path.write_text(json.dumps(cal_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    recipe = {
        "schema_version": "r20_selected_recipe_frozen_v1",
        "calibration_report_path": str(cal_path),
        "calibration_report_sha256": file_sha256(cal_path),
        "r20_head_weights": str(weights_out),
        "r20_head_sha256": head_sha,
        "calibration_gate_passed": gate_passed,
        "proceed_to_s4_web": gate_passed,
        "selected_rule": selected["rule"] if selected else None,
    }
    (R20_ART / "calibration" / "r20_selected_recipe_frozen.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "train_wall_seconds": train_summary["wall_seconds"],
        "passing_count": len(passing_rows),
        "calibration_gate_passed": gate_passed,
        "selected_rule": recipe["selected_rule"],
        "selected_r20_primary": selected["r20_cal_primary"] if selected else None,
        "selected_r18_primary": selected["r18_baseline_cal_primary"] if selected else None,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
