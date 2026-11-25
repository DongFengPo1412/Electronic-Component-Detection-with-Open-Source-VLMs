"""Round 18 frozen-locator supervised 4-class crop verifier (`ic`, `diode`, `transistor`, `other`).

Hard invariants:
- Only inspects frozen R15 predictions with `label in {"ic", "diode", "transistor"}`.
- Never modifies `capacitor`, `resistor`, or `inductor` predictions (bit-for-bit identical).
- Never moves or modifies `bbox_xyxy` coordinates.
- Pure RGB context-padded square crops (zero text/box/coordinate leakage into pixels).
- Supports explicit abstention (`abstain_keep_orig`) when confidence or margin is below threshold.
"""

from __future__ import annotations

import copy
import pathlib
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

from pcb_vlm.data.fics_dslr import file_sha256

R18_CLASSES: Tuple[str, ...] = ("ic", "diode", "transistor", "other")
R18_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(R18_CLASSES)}
R18_TARGET_CLASSES: Tuple[str, ...] = ("ic", "diode", "transistor")
R18_UNTOUCHED_CLASSES: Tuple[str, ...] = ("capacitor", "resistor", "inductor")

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def extract_square_context_crop(
    image: Image.Image,
    bbox_xyxy: Sequence[float],
    *,
    target_size: int = 224,
    context_scale: float = 1.60,
    min_side: float = 32.0,
) -> Image.Image:
    """Extract a pure RGB square context crop centered on `bbox_xyxy` without drawing boxes or text."""
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    img_w, img_h = rgb.size
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(float(min_side), max(bw, bh) * float(context_scale))

    left = int(round(cx - 0.5 * side))
    top = int(round(cy - 0.5 * side))
    side_i = max(1, int(round(side)))
    right = left + side_i
    bottom = top + side_i

    if left >= 0 and top >= 0 and right <= img_w and bottom <= img_h:
        crop = rgb.crop((left, top, right, bottom))
    else:
        canvas = Image.new("RGB", (side_i, side_i), (128, 128, 128))
        ix1 = max(0, left)
        iy1 = max(0, top)
        ix2 = min(img_w, right)
        iy2 = min(img_h, bottom)
        if ix2 > ix1 and iy2 > iy1:
            sub = rgb.crop((ix1, iy1, ix2, iy2))
            canvas.paste(sub, (ix1 - left, iy1 - top))
        crop = canvas
    if crop.size != (target_size, target_size):
        crop = crop.resize((target_size, target_size), Image.Resampling.BICUBIC)
    return crop


def build_r18_resnet18(
    *,
    num_classes: int = len(R18_CLASSES),
    imagenet_ckpt_path: Optional[pathlib.Path] = None,
) -> nn.Module:
    """Build ResNet-18 classifier with optional offline ImageNet backbone weights."""
    model = tv_models.resnet18(weights=None)
    if imagenet_ckpt_path is not None:
        sd = torch.load(str(imagenet_ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, int(num_classes))
    return model


def load_r18_verifier_checkpoint(
    weights_path: pathlib.Path,
    *,
    expected_sha256: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any], str]:
    """Load trained R18 ResNet-18 crop verifier checkpoint and verify SHA-256."""
    wp = pathlib.Path(weights_path)
    if not wp.is_file():
        raise FileNotFoundError(f"R18 crop verifier weights missing: {wp}")
    actual_sha = file_sha256(wp).lower()
    if expected_sha256 and actual_sha != str(expected_sha256).strip().lower():
        raise ValueError(f"R18 crop verifier SHA mismatch: {actual_sha} != {expected_sha256}")

    ckpt = torch.load(str(wp), map_location="cpu", weights_only=False)
    classes = tuple(ckpt.get("classes") or R18_CLASSES)
    if classes != R18_CLASSES:
        raise ValueError(f"Unexpected R18 classes in checkpoint: {classes}")
    model = build_r18_resnet18(num_classes=len(classes), imagenet_ckpt_path=None)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    dev = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(dev)
    model.eval()
    return model, ckpt, actual_sha


def _crop_to_tensor(crop: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(crop)
    return TF.normalize(t, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))


@torch.inference_mode()
def score_crops_with_model(
    model: nn.Module,
    crops: Sequence[Image.Image],
    *,
    device: torch.device,
    batch_size: int = 32,
) -> List[Dict[str, float]]:
    """Run deterministic batched softmax inference on a list of 224x224 RGB crops."""
    if not crops:
        return []
    out: List[Dict[str, float]] = []
    for i in range(0, len(crops), max(1, int(batch_size))):
        batch = torch.stack([_crop_to_tensor(c) for c in crops[i : i + batch_size]], dim=0).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        for row in probs:
            out.append({cls: round(float(p), 6) for cls, p in zip(R18_CLASSES, row)})
    return out


def decide_r18_action(
    *,
    orig_label: str,
    probs_by_class: Mapping[str, float],
    t_relabel: float,
    m_relabel: float,
    t_reject_other: float,
    m_reject_other: float = 0.25,
) -> Dict[str, Any]:
    """Apply frozen abstention & relabeling/rejection policy to one target candidate."""
    ranked = sorted(((str(c), float(p)) for c, p in probs_by_class.items()), key=lambda x: -x[1])
    top1_cls, top1_p = ranked[0]
    top2_cls, top2_p = ranked[1]
    margin = top1_p - top2_p

    if top1_cls == "other" and top1_p >= float(t_reject_other) and margin >= float(m_reject_other):
        return {
            "decision": "reject_other",
            "new_label": None,
            "kept": False,
            "top1_class": top1_cls,
            "top1_prob": round(top1_p, 6),
            "top2_class": top2_cls,
            "margin": round(margin, 6),
        }
    if top1_cls in R18_TARGET_CLASSES and top1_cls != orig_label and top1_p >= float(t_relabel) and margin >= float(m_relabel):
        return {
            "decision": "relabel",
            "new_label": top1_cls,
            "kept": True,
            "top1_class": top1_cls,
            "top1_prob": round(top1_p, 6),
            "top2_class": top2_cls,
            "margin": round(margin, 6),
        }
    if top1_cls == orig_label and top1_p >= float(t_relabel) and margin >= float(m_relabel):
        return {
            "decision": "confirm_orig",
            "new_label": orig_label,
            "kept": True,
            "top1_class": top1_cls,
            "top1_prob": round(top1_p, 6),
            "top2_class": top2_cls,
            "margin": round(margin, 6),
        }
    return {
        "decision": "abstain_keep_orig",
        "new_label": orig_label,
        "kept": True,
        "top1_class": top1_cls,
        "top1_prob": round(top1_p, 6),
        "top2_class": top2_cls,
        "margin": round(margin, 6),
    }


def apply_r18_verifier_to_predictions(
    *,
    image_path: pathlib.Path,
    predictions: Sequence[Mapping[str, Any]],
    model: nn.Module,
    device: torch.device,
    weights_sha256: str,
    t_relabel: float,
    m_relabel: float,
    t_reject_other: float,
    m_reject_other: float = 0.25,
) -> Dict[str, Any]:
    """Verify ONLY `ic`/`diode`/`transistor` predictions; leave `capacitor`/`resistor`/`inductor` bit-for-bit untouched."""
    t0 = time.perf_counter()
    target_indices: List[int] = []
    crops: List[Image.Image] = []

    with Image.open(image_path) as im_raw:
        rgb = im_raw.convert("RGB")
        for idx, pred in enumerate(predictions):
            lbl = str(pred.get("label") or "").strip().lower()
            if lbl in R18_TARGET_CLASSES:
                target_indices.append(idx)
                crops.append(extract_square_context_crop(rgb, pred["bbox_xyxy"]))

    probs_list = score_crops_with_model(model, crops, device=device, batch_size=32)
    decisions_by_idx: Dict[int, Dict[str, Any]] = {}
    decisions_log: List[Dict[str, Any]] = []

    for idx, probs in zip(target_indices, probs_list):
        pred = predictions[idx]
        orig_lbl = str(pred["label"]).strip().lower()
        act = decide_r18_action(
            orig_label=orig_lbl,
            probs_by_class=probs,
            t_relabel=t_relabel,
            m_relabel=m_relabel,
            t_reject_other=t_reject_other,
            m_reject_other=m_reject_other,
        )
        decisions_by_idx[idx] = {"probs": probs, **act}
        decisions_log.append({
            "prediction_id": pred.get("prediction_id"),
            "bbox_xyxy": list(pred["bbox_xyxy"]),
            "r15_score": float(pred.get("score", 0.0)),
            "orig_r15_label": orig_lbl,
            "verified_label": act["new_label"],
            "decision": act["decision"],
            "kept": act["kept"],
            "top1_class": act["top1_class"],
            "top1_prob": act["top1_prob"],
            "margin": act["margin"],
            "probs": probs,
        })

    verified_predictions: List[Dict[str, Any]] = []
    decision_counts: Dict[str, int] = {
        "relabel": 0,
        "confirm_orig": 0,
        "abstain_keep_orig": 0,
        "reject_other": 0,
        "untouched_non_target": 0,
    }
    for idx, pred in enumerate(predictions):
        item = copy.deepcopy(dict(pred))
        if idx not in decisions_by_idx:
            item["orig_r15_label"] = item["label"]
            item["verifier_decision"] = "untouched_non_target"
            item["verifier_source"] = "frozen_r15_passthrough"
            verified_predictions.append(item)
            decision_counts["untouched_non_target"] += 1
            continue
        dec = decisions_by_idx[idx]
        decision_counts[dec["decision"]] += 1
        if not dec["kept"]:
            continue
        item["orig_r15_label"] = str(pred["label"])
        item["label"] = str(dec["new_label"])
        item["aligned_fics6_label"] = str(dec["new_label"])
        item["verifier_decision"] = dec["decision"]
        item["verifier_source"] = "r18_resnet18_crop_verifier"
        item["verifier_top1_prob"] = dec["top1_prob"]
        item["verifier_margin"] = dec["margin"]
        item["vlm_intervened"] = False
        verified_predictions.append(item)

    wall_s = round(time.perf_counter() - t0, 4)
    return {
        "verified_predictions": verified_predictions,
        "decisions_log": decisions_log,
        "summary": {
            "crop_verifier_weights_sha256": weights_sha256,
            "thresholds": {
                "t_relabel": float(t_relabel),
                "m_relabel": float(m_relabel),
                "t_reject_other": float(t_reject_other),
                "m_reject_other": float(m_reject_other),
            },
            "verifier_calls": len(target_indices),
            "verifier_wall_seconds": wall_s,
            "input_prediction_count": len(predictions),
            "output_prediction_count": len(verified_predictions),
            "decision_counts": decision_counts,
        },
    }
