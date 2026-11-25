"""Round 20 targeted IC misclassification relabeler (runs strictly after frozen R19 A1).

Hard invariants:
- Only inspects A1-retained boxes with `label == "ic"`.
- Never adds or removes boxes, never moves `bbox_xyxy`, never changes YOLO `score`.
- Never relabels to `other` or `capacitor`/`resistor`/`inductor`.
- Only relabels to `diode` or `transistor` when `top1_prob >= t_relabel` and `margin >= m_relabel`
  AND there is no existing `IoU >= 0.50` box already labeled with the same `top1_cls`.
- Preserves `capacitor`, `resistor`, `inductor` 100% bit-for-bit identical in coords, labels, scores, and relative order.
"""

from __future__ import annotations

import copy
import pathlib
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

from pcb_vlm.data.fics_dslr import file_sha256
from pcb_vlm.evaluation.composite_detector import compute_iou_xyxy
from pcb_vlm.evaluation.crop_verifier_r18 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    extract_square_context_crop,
)
from pcb_vlm.evaluation.round19_postproc import verify_r19_untouched_invariants

R20_CLASSES: Tuple[str, ...] = ("ic", "diode", "transistor", "other")
R20_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(R20_CLASSES)}
R20_DQ_CLASSES: Tuple[str, ...] = ("diode", "transistor")
R20_UNTOUCHED_CLASSES: Tuple[str, ...] = ("capacitor", "resistor", "inductor")
R20_OVERLAP_ABSTAIN_IOU: float = 0.50


class R20LightweightHead(nn.Module):
    """Lightweight class-balanced MLP head over frozen 512-dim ResNet-18 features."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 128, num_classes: int = 4, dropout: float = 0.30) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_frozen_r18_backbone(
    *,
    imagenet_ckpt_path: pathlib.Path,
    device: torch.device,
) -> nn.Module:
    """Load frozen offline ImageNet ResNet-18 feature extractor (512-dim output)."""
    ckpt_p = pathlib.Path(imagenet_ckpt_path)
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Offline ImageNet ResNet-18 checkpoint missing: {ckpt_p}")
    model = tv_models.resnet18(weights=None)
    sd = torch.load(str(ckpt_p), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_r20_head_checkpoint(
    *,
    head_weights_path: pathlib.Path,
    imagenet_ckpt_path: pathlib.Path,
    expected_sha256: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, R20LightweightHead, Dict[str, Any], str]:
    """Load frozen ImageNet backbone + trained R20 lightweight head and verify SHA-256."""
    wp = pathlib.Path(head_weights_path)
    if not wp.is_file():
        raise FileNotFoundError(f"R20 head weights missing: {wp}")
    actual_sha = file_sha256(wp).lower()
    if expected_sha256 and actual_sha != str(expected_sha256).strip().lower():
        raise ValueError(f"R20 head SHA mismatch: {actual_sha} != {expected_sha256}")

    dev = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    backbone = build_frozen_r18_backbone(imagenet_ckpt_path=imagenet_ckpt_path, device=dev)
    ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
    head = R20LightweightHead(
        in_dim=int(ckpt.get("in_dim", 512)),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        num_classes=len(R20_CLASSES),
        dropout=float(ckpt.get("dropout", 0.30)),
    )
    head.load_state_dict(ckpt["head_state_dict"], strict=True)
    head.to(dev)
    head.eval()
    return backbone, head, ckpt, actual_sha


def _crop_to_tensor(crop: Image.Image) -> torch.Tensor:
    return TF.normalize(TF.to_tensor(crop), mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))


@torch.inference_mode()
def extract_dihedral_avg_features(
    backbone: nn.Module,
    crops: Sequence[Image.Image],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract L2-normalized 512-dim ResNet-18 features averaged across 4 right-angle rotations."""
    if not crops:
        return torch.empty((0, 512), dtype=torch.float32)
    feats_out: List[torch.Tensor] = []
    for i in range(0, len(crops), max(1, int(batch_size))):
        chunk = crops[i : i + batch_size]
        rot_feats: List[torch.Tensor] = []
        for k in (0, 1, 2, 3):
            tensors = [
                _crop_to_tensor(c if k == 0 else c.rotate(90 * k))
                for c in chunk
            ]
            xb = torch.stack(tensors, dim=0).to(device)
            fb = F.normalize(backbone(xb), dim=-1)
            rot_feats.append(fb)
        avg_f = F.normalize(torch.stack(rot_feats, dim=0).mean(dim=0), dim=-1)
        feats_out.append(avg_f.detach().cpu())
    return torch.cat(feats_out, dim=0)


@torch.inference_mode()
def score_crops_r20(
    backbone: nn.Module,
    head: R20LightweightHead,
    crops: Sequence[Image.Image],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict[str, float]]:
    """Run deterministic dihedral-averaged R20 feature extraction + head softmax."""
    if not crops:
        return []
    feats = extract_dihedral_avg_features(backbone, crops, device=device, batch_size=batch_size).to(device)
    logits = head(feats)
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    return [{cls: round(float(p), 6) for cls, p in zip(R20_CLASSES, row)} for row in probs]


def _top1_top2_margin(probs_by_class: Mapping[str, float]) -> Tuple[str, float, str, float, float]:
    ranked = sorted(((str(c), float(p)) for c, p in probs_by_class.items()), key=lambda x: (-x[1], x[0]))
    top1_cls, top1_p = ranked[0]
    top2_cls, top2_p = ranked[1]
    return top1_cls, top1_p, top2_cls, top2_p, top1_p - top2_p


def _pred_key(pred: Mapping[str, Any], fallback_idx: int) -> str:
    pid = str(pred.get("prediction_id") or "").strip()
    if pid:
        return pid
    box = [round(float(v), 2) for v in pred["bbox_xyxy"]]
    return f"idx{fallback_idx}_{pred.get('label')}_{box[0]}_{box[1]}_{box[2]}_{box[3]}"


def apply_r20_targeted_ic_relabel(
    *,
    a1_predictions: Sequence[Mapping[str, Any]],
    probs_by_pred_id: Optional[Mapping[str, Mapping[str, float]]] = None,
    image_path: Optional[pathlib.Path] = None,
    backbone: Optional[nn.Module] = None,
    head: Optional[R20LightweightHead] = None,
    device: Optional[torch.device] = None,
    t_relabel: float = 0.80,
    m_relabel: float = 0.25,
    overlap_iou_threshold: float = R20_OVERLAP_ABSTAIN_IOU,
) -> Dict[str, Any]:
    """Apply targeted IC->D/Q relabeling strictly on A1-retained `ic` boxes."""
    t0 = time.perf_counter()
    probs_map: Dict[str, Mapping[str, float]] = dict(probs_by_pred_id or {})
    if probs_by_pred_id is None:
        if image_path is None or backbone is None or head is None or device is None:
            raise ValueError("Either probs_by_pred_id or (image_path, backbone, head, device) must be provided.")
        ic_items = [(idx, p) for idx, p in enumerate(a1_predictions) if str(p.get("label") or "").strip().lower() == "ic"]
        if ic_items:
            with Image.open(image_path) as im_raw:
                rgb = im_raw.convert("RGB")
                crops = [extract_square_context_crop(rgb, p["bbox_xyxy"]) for _, p in ic_items]
            scored = score_crops_r20(backbone, head, crops, device=device, batch_size=32)
            for (idx, p), pr in zip(ic_items, scored):
                probs_map[_pred_key(p, idx)] = pr

    out_preds = [copy.deepcopy(dict(p)) for p in a1_predictions]
    # Existing D/Q boxes in A1 must be protected from duplicate relabels
    existing_dq_by_cls: Dict[str, List[Tuple[str, List[float]]]] = {"diode": [], "transistor": []}
    ic_indices: List[int] = []
    for idx, p in enumerate(out_preds):
        lbl = str(p.get("label") or "").strip().lower()
        p.setdefault("orig_r15_label", lbl)
        p.setdefault("box_source", "r15_official_025")
        p.setdefault("is_low_conf_supplement", False)
        if lbl in R20_DQ_CLASSES:
            existing_dq_by_cls[lbl].append((_pred_key(p, idx), [float(v) for v in p["bbox_xyxy"]]))
        elif lbl == "ic":
            ic_indices.append(idx)

    # Deterministic processing order: highest YOLO score first, then prediction_id
    ic_indices.sort(key=lambda i: (-float(out_preds[i].get("score", 0.0)), _pred_key(out_preds[i], i)))

    decisions_log: List[Dict[str, Any]] = []
    relabeled_boxes: List[Dict[str, Any]] = []
    overlap_abstained: List[Dict[str, Any]] = []
    counts = {
        "ic_inspected": len(ic_indices),
        "relabeled_to_diode": 0,
        "relabeled_to_transistor": 0,
        "abstain_same_class_overlap": 0,
        "keep_ic_confirm": 0,
        "keep_ic_abstain": 0,
    }

    for idx in ic_indices:
        p = out_preds[idx]
        pid = _pred_key(p, idx)
        pr = probs_map.get(pid)
        if pr is None:
            raise KeyError(f"Missing R20 probabilities for IC box {pid!r}")
        top1_cls, top1_p, top2_cls, top2_p, margin = _top1_top2_margin(pr)

        decision = "keep_ic_confirm" if top1_cls == "ic" else "keep_ic_abstain"
        conflict_pid: Optional[str] = None
        conflict_iou: float = 0.0
        new_label = "ic"

        if top1_cls in R20_DQ_CLASSES and top1_p >= float(t_relabel) and margin >= float(m_relabel):
            best_iou = 0.0
            best_pid = ""
            for ex_pid, ex_box in existing_dq_by_cls[top1_cls]:
                iou = compute_iou_xyxy(p["bbox_xyxy"], ex_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pid = ex_pid
            if best_iou >= float(overlap_iou_threshold):
                decision = "abstain_same_class_overlap"
                conflict_pid = best_pid
                conflict_iou = round(best_iou, 6)
                counts["abstain_same_class_overlap"] += 1
                overlap_abstained.append({
                    "prediction_id": pid,
                    "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
                    "r15_score": float(p.get("score", 0.0)),
                    "proposed_label": top1_cls,
                    "top1_prob": round(top1_p, 6),
                    "margin": round(margin, 6),
                    "conflict_with_prediction_id": conflict_pid,
                    "conflict_iou": conflict_iou,
                })
            else:
                decision = f"relabel_ic_to_{top1_cls}"
                new_label = top1_cls
                p["label"] = top1_cls
                p["aligned_fics6_label"] = top1_cls
                p["orig_r15_label"] = str(p.get("orig_r15_label") or "ic")
                p["box_source"] = "r20_ic_targeted_relabel"
                p["is_low_conf_supplement"] = False
                p["r20_relabel_decision"] = decision
                p["r20_relabel_source"] = "r20_frozen_r18feat_head"
                p["r20_top1_prob"] = round(top1_p, 6)
                p["r20_margin"] = round(margin, 6)
                existing_dq_by_cls[top1_cls].append((pid, [float(v) for v in p["bbox_xyxy"]]))
                counts[f"relabeled_to_{top1_cls}"] += 1
                relabeled_boxes.append({
                    "prediction_id": pid,
                    "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
                    "r15_score": float(p.get("score", 0.0)),
                    "orig_label": "ic",
                    "new_label": top1_cls,
                    "top1_prob": round(top1_p, 6),
                    "margin": round(margin, 6),
                    "probs": {str(k): round(float(v), 6) for k, v in pr.items()},
                })
        else:
            counts[decision] += 1
            p["r20_relabel_decision"] = decision
            p["r20_relabel_source"] = "r20_frozen_r18feat_head"
            p["r20_top1_class"] = top1_cls
            p["r20_top1_prob"] = round(top1_p, 6)
            p["r20_margin"] = round(margin, 6)

        decisions_log.append({
            "prediction_id": pid,
            "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
            "r15_score": float(p.get("score", 0.0)),
            "orig_label": "ic",
            "new_label": new_label,
            "decision": decision,
            "top1_class": top1_cls,
            "top1_prob": round(top1_p, 6),
            "top2_class": top2_cls,
            "margin": round(margin, 6),
            "conflict_with_prediction_id": conflict_pid,
            "conflict_iou": conflict_iou if conflict_pid else None,
            "probs": {str(k): round(float(v), 6) for k, v in pr.items()},
        })

    verify_r19_untouched_invariants(a0_preds=a1_predictions, derived_preds=out_preds, arm_name="R20_relabel")

    return {
        "predictions": out_preds,
        "decisions_log": decisions_log,
        "relabeled_boxes": relabeled_boxes,
        "overlap_abstained": overlap_abstained,
        "summary": {
            "t_relabel": float(t_relabel),
            "m_relabel": float(m_relabel),
            "overlap_iou_threshold": float(overlap_iou_threshold),
            **counts,
            "wall_seconds": round(time.perf_counter() - t0, 4),
        },
    }
