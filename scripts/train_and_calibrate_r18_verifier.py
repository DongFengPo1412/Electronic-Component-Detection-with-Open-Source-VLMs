"""Train bounded ResNet-18 4-class crop verifier and calibrate abstention thresholds strictly on validation."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T

Image.MAX_IMAGE_PIXELS = None

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.data.fics_dslr import FICS_RAW_TO_CANONICAL, file_sha256, parse_component_location  # noqa: E402
from pcb_vlm.evaluation.composite_detector import _prf1, evaluate_composite_board  # noqa: E402
from pcb_vlm.evaluation.crop_verifier_r18 import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    R18_CLASSES,
    R18_CLASS_TO_IDX,
    R18_TARGET_CLASSES,
    apply_r18_verifier_to_predictions,
    build_r18_resnet18,
    load_r18_verifier_checkpoint,
)

R18_CACHE = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration18-20261125")
R18_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125")
DEFAULT_MANIFEST = R18_CACHE / "crop_dataset_r18" / "crop_manifest.json"
DEFAULT_IMAGENET_CKPT = pathlib.Path(r"C:\Users\Liu\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth")
R15_WEIGHTS_PATH = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\weights\fics_yolo11n_r15_best.pt")
R17_BLIND_MANIFEST = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124\codex_reserved_blind\r17_codex_blind_manifest.json")


class _CropDataset(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]], transform: Any) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        r = self.records[idx]
        with Image.open(r["crop_path"]) as im:
            rgb = im.convert("RGB")
            x = self.transform(rgb)
        return x, int(R18_CLASS_TO_IDX[str(r["label"])])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_verifier(manifest_path: pathlib.Path, imagenet_ckpt: pathlib.Path, seed: int = 42) -> Dict[str, Any]:
    t0 = time.perf_counter()
    _seed_all(seed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_recs = list(manifest["train"])
    val_recs = list(manifest["val"])

    train_tf = T.Compose([
        T.RandomApply([T.RandomChoice([
            T.RandomRotation((0, 0)),
            T.RandomRotation((90, 90)),
            T.RandomRotation((180, 180)),
            T.RandomRotation((270, 270)),
        ])], p=1.0),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])

    train_ds = _CropDataset(train_recs, train_tf)
    val_ds = _CropDataset(val_recs, val_tf)

    cls_counts = collections.Counter(r["label"] for r in train_recs)
    # Inverse-sqrt class weights to boost diode/transistor without destabilizing gradients
    weights_per_cls = {c: 1.0 / math.sqrt(max(1, cls_counts[c])) for c in R18_CLASSES}
    mean_w = sum(weights_per_cls.values()) / len(R18_CLASSES)
    weights_per_cls = {c: w / mean_w for c, w in weights_per_cls.items()}
    sample_weights = [weights_per_cls[r["label"]] for r in train_recs]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_recs), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_r18_resnet18(num_classes=len(R18_CLASSES), imagenet_ckpt_path=imagenet_ckpt).to(device)

    cls_weight_tensor = torch.tensor([weights_per_cls[c] for c in R18_CLASSES], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cls_weight_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_macro_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_val_preds: List[int] = []
    best_val_targets: List[int] = []
    history: List[Dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        n_tr = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item()) * xb.size(0)
            n_tr += int(xb.size(0))
        scheduler.step()

        model.eval()
        val_preds: List[int] = []
        val_targets: List[int] = []
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_targets.extend(yb.tolist())

        macro_f1 = float(f1_score(val_targets, val_preds, average="macro", zero_division=0))
        acc = sum(1 for a, b in zip(val_preds, val_targets) if a == b) / max(1, len(val_targets))
        history.append({"epoch": ep, "train_loss": round(tot_loss / max(1, n_tr), 4), "val_acc": round(acc, 4), "val_macro_f1": round(macro_f1, 4)})
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_preds = list(val_preds)
            best_val_targets = list(val_targets)

    assert best_state is not None
    weights_out = R18_ART / "weights" / "r18_resnet18_crop_verifier_best.pt"
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "r18_resnet18_crop_verifier_v1",
            "classes": list(R18_CLASSES),
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_macro_f1, 6),
            "imagenet_init_sha256": file_sha256(imagenet_ckpt).lower(),
            "state_dict": best_state,
        },
        str(weights_out),
    )
    weights_sha = file_sha256(weights_out).lower()

    cm = confusion_matrix(best_val_targets, best_val_preds, labels=list(range(len(R18_CLASSES)))).tolist()
    cls_rep = classification_report(
        best_val_targets, best_val_preds, labels=list(range(len(R18_CLASSES))), target_names=list(R18_CLASSES), output_dict=True, zero_division=0
    )
    # Check false acceptance of `other` into diode/transistor
    other_idx = R18_CLASS_TO_IDX["other"]
    other_rows = [p for t, p in zip(best_val_targets, best_val_preds) if t == other_idx]
    other_to_dq = sum(1 for p in other_rows if p in (R18_CLASS_TO_IDX["diode"], R18_CLASS_TO_IDX["transistor"]))

    summary = {
        "schema_version": "r18_verifier_train_summary_v1",
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "seed": seed,
        "epochs": epochs,
        "best_epoch_by_r18_dq_val_macro_f1": best_epoch,
        "best_val_macro_f1": round(best_macro_f1, 4),
        "train_counts": dict(cls_counts),
        "val_counts": dict(collections.Counter(r["label"] for r in val_recs)),
        "imagenet_ckpt_sha256": file_sha256(imagenet_ckpt).lower(),
        "weights_path": str(weights_out),
        "weights_sha256": weights_sha,
        "r18_dq_val_confusion_matrix_labels": list(R18_CLASSES),
        "r18_dq_val_confusion_matrix": cm,
        "r18_dq_val_classification_report": cls_rep,
        "r18_dq_val_other_total": len(other_rows),
        "r18_dq_val_other_false_accept_to_dq": other_to_dq,
        "epoch_history": history,
    }
    (R18_ART / "training" / "r18_verifier_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[TRAIN] best_ep={best_epoch}, val_macro_f1={best_macro_f1:.4f}, diode_rec={cls_rep['diode']['recall']:.3f}, trans_rec={cls_rep['transistor']['recall']:.3f}, other->DQ={other_to_dq}/{len(other_rows)}", flush=True)
    return summary


def calibrate_abstention(manifest_path: pathlib.Path, weights_path: pathlib.Path, weights_sha: str) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prop_idx = manifest["proposal_cache_index"]
    val_boards = [v for v in prop_idx.values() if v["role"] in ("val", "r15_frozen_val")]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_r18_verifier_checkpoint(weights_path, expected_sha256=weights_sha, device=device)

    # Precompute raw probabilities for every cached validation board once!
    board_data: List[Dict[str, Any]] = []
    for vb in val_boards:
        stem = str(vb["board_side_id"])
        gts = _read_csv_gts(pathlib.Path(str(vb["csv_path"])))
        preds_025 = json.loads(pathlib.Path(str(vb["preds_025_path"])).read_text(encoding="utf-8"))
        prop_dir = pathlib.Path(str(vb["preds_025_path"])).parent
        pool_005 = json.loads((prop_dir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        summary = json.loads((prop_dir / "yolo_run_summary.json").read_text(encoding="utf-8"))
        roi = summary["geometry"]["roi_bbox_half_open"]
        base_ev = evaluate_composite_board(gts=gts, predictions_025=preds_025, low_conf_pool_005=pool_005, roi_bbox_half_open=roi)
        # Pre-score target crops via helper at dummy high thresholds to capture probs once
        raw_res = apply_r18_verifier_to_predictions(
            image_path=pathlib.Path(str(vb["tif_path"])),
            predictions=preds_025,
            model=model,
            device=device,
            weights_sha256=weights_sha,
            t_relabel=0.99,
            m_relabel=0.99,
            t_reject_other=1.01,
        )
        probs_by_idx = {}
        # Map decisions_log back to indices
        t_indices = [i for i, p in enumerate(preds_025) if str(p.get("label") or "").lower() in R18_TARGET_CLASSES]
        for ti, dlog in zip(t_indices, raw_res["decisions_log"]):
            probs_by_idx[ti] = dlog["probs"]
        board_data.append({
            "stem": stem,
            "role": vb["role"],
            "gts": gts,
            "preds_025": preds_025,
            "pool_005": pool_005,
            "roi": roi,
            "base_ev": base_ev,
            "probs_by_idx": probs_by_idx,
        })

    # Fast pure-CPU threshold sweep over precomputed probabilities
    from pcb_vlm.evaluation.crop_verifier_r18 import decide_r18_action

    base_r15_tp = sum(b["base_ev"]["class_aware"]["tp"] for b in board_data if b["role"] == "r15_frozen_val")
    base_r15_fp = sum(b["base_ev"]["class_aware"]["fp"] for b in board_data if b["role"] == "r15_frozen_val")
    base_r15_fn = sum(b["base_ev"]["class_aware"]["fn"] for b in board_data if b["role"] == "r15_frozen_val")
    base_r15_f1 = _prf1(base_r15_tp, base_r15_fp, base_r15_fn)["f1_pct"]
    base_r15_ic_tp = sum(b["base_ev"]["per_class"]["ic"]["tp"] for b in board_data if b["role"] == "r15_frozen_val")
    base_dq_val_dq_tp = sum(
        b["base_ev"]["per_class"]["diode"]["tp"] + b["base_ev"]["per_class"]["transistor"]["tp"]
        for b in board_data if b["role"] == "val"
    )

    grid_results: List[Dict[str, Any]] = []
    for t_rel in (0.55, 0.65, 0.75, 0.85):
        for m_rel in (0.15, 0.25, 0.35):
            for t_rej in (0.75, 0.85, 0.90, 0.95, 1.01):
                r15_tp = r15_fp = r15_fn = r15_ic_tp = 0
                dq_tp = dq_fp = dq_fn = dq_val_dq_tp = 0
                for b in board_data:
                    mod_preds = []
                    for i, p in enumerate(b["preds_025"]):
                        if i not in b["probs_by_idx"]:
                            mod_preds.append(p)
                        else:
                            act = decide_r18_action(
                                orig_label=str(p["label"]).lower(),
                                probs_by_class=b["probs_by_idx"][i],
                                t_relabel=t_rel,
                                m_relabel=m_rel,
                                t_reject_other=t_rej,
                                m_reject_other=0.25,
                            )
                            if act["kept"]:
                                np = dict(p)
                                np["label"] = act["new_label"]
                                mod_preds.append(np)
                    ev = evaluate_composite_board(gts=b["gts"], predictions_025=mod_preds, low_conf_pool_005=b["pool_005"], roi_bbox_half_open=b["roi"])
                    if b["role"] == "r15_frozen_val":
                        r15_tp += ev["class_aware"]["tp"]
                        r15_fp += ev["class_aware"]["fp"]
                        r15_fn += ev["class_aware"]["fn"]
                        r15_ic_tp += ev["per_class"]["ic"]["tp"]
                    else:
                        dq_tp += ev["class_aware"]["tp"]
                        dq_fp += ev["class_aware"]["fp"]
                        dq_fn += ev["class_aware"]["fn"]
                        dq_val_dq_tp += ev["per_class"]["diode"]["tp"] + ev["per_class"]["transistor"]["tp"]

                r15_f1 = _prf1(r15_tp, r15_fp, r15_fn)["f1_pct"]
                comb_f1 = _prf1(r15_tp + dq_tp, r15_fp + dq_fp, r15_fn + dq_fn)["f1_pct"]
                qualifies = (
                    r15_f1 >= base_r15_f1
                    and r15_ic_tp >= (base_r15_ic_tp - 1)
                    and dq_val_dq_tp >= base_dq_val_dq_tp
                )
                grid_results.append({
                    "t_relabel": t_rel,
                    "m_relabel": m_rel,
                    "t_reject_other": t_rej,
                    "m_reject_other": 0.25,
                    "r15_frozen_val_f1_pct": r15_f1,
                    "r15_frozen_val_ic_tp": r15_ic_tp,
                    "r18_dq_val_dq_tp": dq_val_dq_tp,
                    "r18_dq_val_f1_pct": _prf1(dq_tp, dq_fp, dq_fn)["f1_pct"],
                    "combined_val_f1_pct": comb_f1,
                    "qualifies": qualifies,
                })

    qualified = [g for g in grid_results if g["qualifies"]]
    if qualified:
        selected = max(qualified, key=lambda x: (x["r18_dq_val_dq_tp"], x["combined_val_f1_pct"], x["t_relabel"]))
    else:
        selected = max(grid_results, key=lambda x: (x["r15_frozen_val_f1_pct"], x["r18_dq_val_dq_tp"]))

    cal_doc = {
        "schema_version": "r18_val_abstention_calibration_v1",
        "baseline_r15_frozen_val_f1_pct": base_r15_f1,
        "baseline_r15_frozen_val_ic_tp": base_r15_ic_tp,
        "baseline_r18_dq_val_dq_tp": base_dq_val_dq_tp,
        "qualified_count": len(qualified),
        "selected": selected,
        "grid_results": grid_results,
    }
    cal_out = R18_ART / "calibration" / "r18_val_abstention_calibration.json"
    cal_out.parent.mkdir(parents=True, exist_ok=True)
    cal_out.write_text(json.dumps(cal_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[CALIBRATION] qualified={len(qualified)}/{len(grid_results)}, selected={selected}", flush=True)
    return cal_doc


def write_pre_eval_freeze(weights_path: pathlib.Path, cal_path: pathlib.Path) -> None:
    files = [
        weights_path,
        cal_path,
        _SRC_ROOT / "pcb_vlm" / "evaluation" / "crop_verifier_r18.py",
        R15_WEIGHTS_PATH,
        R17_BLIND_MANIFEST,
    ]
    frozen = {
        "schema_version": "r18_pre_eval_freeze_v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "pre_registered_r18_promotion_gate": {
            "rule": "On S20+S4+S1 official 0.25 track: (1) diode+transistor TP gain >= +3, (2) combined Micro-F1 >= 66.32%, (3) combined IC FP <= 72 with IC TP drop <= 1, (4) capacitor/resistor/inductor 100% bit-for-bit identical, (5) S1 absent D+Q FP non-increasing or <= 1 each.",
            "purpose": "Only if passed does R18 candidate qualify for Codex independent blind evaluation on s12_front & s24_front.",
        },
        "frozen_files": [{"path": str(p), "sha256": file_sha256(p).lower(), "bytes": p.stat().st_size} for p in files],
    }
    (R18_ART / "frozen_plan_r18_pre_eval.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    tr_sum = train_verifier(DEFAULT_MANIFEST, DEFAULT_IMAGENET_CKPT, seed=42)
    cal_doc = calibrate_abstention(DEFAULT_MANIFEST, pathlib.Path(tr_sum["weights_path"]), tr_sum["weights_sha256"])
    write_pre_eval_freeze(pathlib.Path(tr_sum["weights_path"]), R18_ART / "calibration" / "r18_val_abstention_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
