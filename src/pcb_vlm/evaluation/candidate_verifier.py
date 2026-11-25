"""GT-free second-stage candidate crop verifier manifest, strict parser, and filter policies."""

from __future__ import annotations

import hashlib
import io
import json
import math
import pathlib
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageOps

from pcb_vlm.data.fics_dslr import file_sha256
from pcb_vlm.evaluation.fics_replay import strip_eos_and_outer_fence

TARGET_SIX_CLASSES = ("capacitor", "resistor", "inductor", "transistor", "diode", "ic")
ALLOWED_SEVEN_LABELS = TARGET_SIX_CLASSES + ("other",)
FORBIDDEN_LEAK_TERMS = ("ground_truth", "orig_pred_label", "true_positive", "false_positive", "gt", "tp", "fp")

DEFAULT_VERIFIER_PROMPT = (
    'Examine this cropped PCB region centered on a single candidate object. '
    'Classify the object into exactly one label from: '
    '["capacitor", "resistor", "inductor", "transistor", "diode", "ic", "other"]. '
    'Use "other" if the crop shows only bare PCB/background/silkscreen/trace artifacts, '
    'no identifiable target component, or cannot be reliably classified. '
    'Return ONLY one JSON object: {"label": "<class>"}'
)


def parse_single_label_json(
    raw_text: str,
    *,
    allowed_labels: Sequence[str] = ALLOWED_SEVEN_LABELS,
) -> Dict[str, Any]:
    """Parse single-label verifier JSON with strict contract separated from compat regex."""
    cleaned, stripped_eos = strip_eos_and_outer_fence(raw_text)
    allowed = tuple(str(x).strip().lower() for x in allowed_labels)
    strict_ok = False
    strict_label: Optional[str] = None
    compat_regex_label: Optional[str] = None
    parse_error: Optional[str] = None

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and set(obj.keys()) == {"label"}:
            raw_lbl = obj["label"]
            if isinstance(raw_lbl, str):
                lbl = raw_lbl.strip().lower()
                if lbl in allowed:
                    strict_ok = True
                    strict_label = lbl
                else:
                    parse_error = f"unknown_label:{lbl}"
            else:
                parse_error = f"non_string_label:{type(raw_lbl).__name__}"
        else:
            parse_error = "unexpected_json_structure"
    except Exception as exc:
        parse_error = repr(exc)

    if not strict_ok:
        match = re.search(r'"label"\s*:\s*"([a-zA-Z_]+)"', cleaned)
        if match:
            cand = match.group(1).strip().lower()
            if cand in allowed:
                compat_regex_label = cand

    return {
        "cleaned_text": cleaned,
        "stripped_eos": stripped_eos,
        "strict_json_ok": strict_ok,
        "strict_label": strict_label,
        "compat_regex_label": compat_regex_label,
        "compat_effective_label": strict_label if strict_ok else compat_regex_label,
        "parse_error": parse_error,
    }


def _dedup_signature(crop_sha256: str, spec: Mapping[str, Any]) -> str:
    payload = {
        "crop_sha256": str(crop_sha256).lower(),
        "prompt_sha256": str(spec["prompt_sha256"]).lower(),
        "model_revision": str(spec["model_revision"]),
        "min_pixels": int(spec["min_pixels"]),
        "max_pixels": int(spec["max_pixels"]),
        "dtype": str(spec["dtype"]),
        "do_sample": bool(spec["do_sample"]),
        "repetition_penalty": float(spec["repetition_penalty"]),
        "max_new_tokens": int(spec["max_new_tokens"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_candidate_verifier_manifest(
    *,
    source_predictions: Sequence[Mapping[str, Any]],
    board_image_paths: Mapping[str, pathlib.Path],
    crops_dir: pathlib.Path,
    inference_spec: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    pad_fraction: float = 0.10,
    min_edge_upscale: int = 224,
) -> Dict[str, Any]:
    """Build GT-free neutral crop manifest with byte-SHA deduplication for all source predictions."""
    crops_dir.mkdir(parents=True, exist_ok=True)

    loaded_boards: Dict[str, Image.Image] = {}
    board_file_meta: Dict[str, Dict[str, Any]] = {}
    for bid in sorted(board_image_paths.keys()):
        img_p = pathlib.Path(board_image_paths[bid])
        with Image.open(img_p) as raw_im:
            im = ImageOps.exif_transpose(raw_im).convert("RGB")
        loaded_boards[bid] = im
        board_file_meta[bid] = {
            "path": str(img_p),
            "sha256": file_sha256(img_p),
            "width": im.size[0],
            "height": im.size[1],
        }

    spec_dict = dict(inference_spec)
    prompt_sha_actual = hashlib.sha256(str(spec_dict["prompt"]).encode("utf-8")).hexdigest().lower()
    if prompt_sha_actual != str(spec_dict["prompt_sha256"]).lower():
        raise ValueError("inference_spec prompt_sha256 does not match prompt text")

    predictions_out: List[Dict[str, Any]] = []
    unique_calls_out: List[Dict[str, Any]] = []
    sig_to_call: Dict[str, Dict[str, Any]] = {}
    seen_pred_ids = set()

    for idx, src in enumerate(source_predictions):
        pred_id = str(src["prediction_id"])
        if pred_id in seen_pred_ids:
            raise ValueError(f"Duplicate prediction_id: {pred_id}")
        seen_pred_ids.add(pred_id)

        bid = str(src["board_id"])
        if bid not in loaded_boards:
            raise KeyError(f"Missing board image path for board_id={bid!r}")
        src_im = loaded_boards[bid]
        iw, ih = src_im.size

        box = [float(v) for v in src["bbox_xyxy"]]
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0.0 or bh <= 0.0:
            predictions_out.append({
                "prediction_id": pred_id,
                "board_id": bid,
                "image_id": src["image_id"],
                "orig_order": int(src.get("orig_order", idx)),
                "orig_pred_label": str(src["label"]).strip().lower(),
                "orig_pred_bbox_xyxy": box,
                "score": float(src.get("score", 1.0)),
                "tile_id": src.get("tile_id"),
                "call_id": src.get("call_id"),
                "crop_valid": False,
                "invalid_reason": "non_positive_bbox_dimensions",
                "crop_box_padded_xyxy": None,
                "crop_sha256": None,
                "unique_call_id": None,
            })
            continue

        px = float(pad_fraction) * max(1.0, bw)
        py = float(pad_fraction) * max(1.0, bh)
        cx1 = max(0, int(math.floor(x1 - px)))
        cy1 = max(0, int(math.floor(y1 - py)))
        cx2 = min(iw, int(math.ceil(x2 + px)))
        cy2 = min(ih, int(math.ceil(y2 + py)))

        if cx2 <= cx1 or cy2 <= cy1:
            predictions_out.append({
                "prediction_id": pred_id,
                "board_id": bid,
                "image_id": src["image_id"],
                "orig_order": int(src.get("orig_order", idx)),
                "orig_pred_label": str(src["label"]).strip().lower(),
                "orig_pred_bbox_xyxy": box,
                "score": float(src.get("score", 1.0)),
                "tile_id": src.get("tile_id"),
                "call_id": src.get("call_id"),
                "crop_valid": False,
                "invalid_reason": "empty_crop_after_clipping",
                "crop_box_padded_xyxy": [cx1, cy1, cx2, cy2],
                "crop_sha256": None,
                "unique_call_id": None,
            })
            continue

        crop_im = src_im.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop_im.size
        upscaled = False
        if min(cw, ch) < int(min_edge_upscale):
            scale = float(min_edge_upscale) / float(min(cw, ch))
            nw = max(int(min_edge_upscale), int(math.ceil(cw * scale)))
            nh = max(int(min_edge_upscale), int(math.ceil(ch * scale)))
            crop_im = crop_im.resize((nw, nh), resample=Image.Resampling.BICUBIC)
            upscaled = True

        buf = io.BytesIO()
        crop_im.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        crop_sha = hashlib.sha256(png_bytes).hexdigest().lower()
        dedup_sig = _dedup_signature(crop_sha, spec_dict)

        if dedup_sig not in sig_to_call:
            uidx = len(unique_calls_out) + 1
            call_id = f"candB_u{uidx:04d}"
            neutral_name = f"crop_{uidx:04d}.png"
            out_path = crops_dir / neutral_name
            out_path.write_bytes(png_bytes)
            disk_sha = file_sha256(out_path).lower()
            if disk_sha != crop_sha:
                raise RuntimeError(f"Crop write verification failed for {out_path}")
            u_rec = {
                "call_id": call_id,
                "sample_index": uidx,
                "neutral_filename": neutral_name,
                "image_path": str(out_path),
                "image_sha256": disk_sha,
                "width": crop_im.size[0],
                "height": crop_im.size[1],
            }
            sig_to_call[dedup_sig] = u_rec
            unique_calls_out.append(u_rec)
        else:
            u_rec = sig_to_call[dedup_sig]

        predictions_out.append({
            "prediction_id": pred_id,
            "board_id": bid,
            "image_id": src["image_id"],
            "orig_order": int(src.get("orig_order", idx)),
            "orig_pred_label": str(src["label"]).strip().lower(),
            "orig_pred_bbox_xyxy": box,
            "score": float(src.get("score", 1.0)),
            "tile_id": src.get("tile_id"),
            "call_id": src.get("call_id"),
            "crop_valid": True,
            "orig_crop_width": cw,
            "orig_crop_height": ch,
            "upscaled_min_edge_224": upscaled,
            "crop_box_padded_xyxy": [cx1, cy1, cx2, cy2],
            "crop_sha256": crop_sha,
            "dedup_signature_sha256": dedup_sig,
            "unique_call_id": u_rec["call_id"],
            "neutral_filename": u_rec["neutral_filename"],
        })

    return {
        "schema_version": "candidate_verifier_manifest_v1",
        "crop_spec": {
            "pad_fraction": float(pad_fraction),
            "min_edge_upscale": int(min_edge_upscale),
            "resample": "BICUBIC",
            "integer_rounding": "floor(min), ceil(max), clamped to image bounds",
            "dedup_key": "exact_crop_png_sha256_plus_prompt_model_pixel_decoding_config",
        },
        "source_provenance": dict(source_provenance),
        "board_images": board_file_meta,
        "inference_spec": spec_dict,
        "counts": {
            "total_predictions": len(predictions_out),
            "valid_crops": sum(1 for p in predictions_out if p["crop_valid"]),
            "invalid_crops": sum(1 for p in predictions_out if not p["crop_valid"]),
            "unique_calls": len(unique_calls_out),
            "deduplicated_reused_predictions": sum(1 for p in predictions_out if p["crop_valid"]) - len(unique_calls_out),
        },
        "unique_calls": unique_calls_out,
        "predictions": predictions_out,
    }


def verify_verifier_manifest_integrity(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify prompt/image SHA-256 integrity, neutrality, and complete prediction->call mapping."""
    spec = manifest["inference_spec"]
    prompt = str(spec["prompt"])
    prompt_sha_exp = str(spec["prompt_sha256"]).lower()
    prompt_sha_act = hashlib.sha256(prompt.encode("utf-8")).hexdigest().lower()
    if prompt_sha_act != prompt_sha_exp:
        raise ValueError(f"Prompt SHA mismatch: {prompt_sha_act} != {prompt_sha_exp}")

    prompt_lower = prompt.lower()
    for term in FORBIDDEN_LEAK_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", prompt_lower):
            raise ValueError(f"Forbidden leakage token {term!r} found in prompt")

    calls_by_id: Dict[str, Dict[str, Any]] = {}
    for entry in manifest["unique_calls"]:
        cid = str(entry["call_id"])
        if cid in calls_by_id:
            raise ValueError(f"Duplicate unique call_id: {cid}")
        fname_lower = str(entry["neutral_filename"]).lower()
        for bad in ("gt", "tp", "fp", "capacitor", "resistor", "inductor", "transistor", "diode", "ic"):
            if re.search(rf"(?:^|[_\W]){bad}(?:$|[_\W])", fname_lower):
                raise ValueError(f"Forbidden semantic token {bad!r} in neutral filename {entry['neutral_filename']!r}")
        for forbidden_key in ("board_id", "orig_pred_label", "matched_gt_id"):
            if forbidden_key in entry and entry[forbidden_key] is not None:
                raise ValueError(f"Unique call {cid} leaked non-neutral field {forbidden_key}")
        img_p = pathlib.Path(entry["image_path"])
        actual_sha = file_sha256(img_p).lower()
        if actual_sha != str(entry["image_sha256"]).lower():
            raise ValueError(f"Image SHA mismatch for {img_p}: {actual_sha} != {entry['image_sha256']}")
        calls_by_id[cid] = dict(entry)

    seen_pred_ids = set()
    for pred in manifest["predictions"]:
        pid = str(pred["prediction_id"])
        if pid in seen_pred_ids:
            raise ValueError(f"Duplicate prediction_id: {pid}")
        seen_pred_ids.add(pid)
        if pred["crop_valid"]:
            ucid = str(pred["unique_call_id"])
            if ucid not in calls_by_id:
                raise ValueError(f"Prediction {pid} references missing unique_call_id {ucid}")
            if str(pred["crop_sha256"]).lower() != str(calls_by_id[ucid]["image_sha256"]).lower():
                raise ValueError(f"Crop SHA mismatch between prediction {pid} and call {ucid}")

    return {
        "prompt_sha256": prompt_sha_act,
        "unique_calls": len(calls_by_id),
        "total_predictions": len(manifest["predictions"]),
    }


def _build_legacy_counts(total_candidates: int) -> Dict[str, int]:
    return {
        "total_candidates": total_candidates,
        "retained_count": 0,
        "filtered_count": 0,
        "invalid_crop_filtered": 0,
        "rejected_as_other": 0,
        "cross_target_retained_same_orig_class": 0,
        "cross_target_rejected": 0,
        "exact_same_class_retained": 0,
        "format_failure_filtered": 0,
        "technical_failure_filtered": 0,
        "missing_record_count": 0,
    }


def _build_strict_counts(total_candidates: int) -> Dict[str, int]:
    counts = _build_legacy_counts(total_candidates)
    counts.update({
        "invalid_crop_kept_anomaly": 0,
        "missing_record_kept_anomaly": 0,
        "technical_failure_kept_anomaly": 0,
        "truncated_kept_anomaly": 0,
        "format_failure_kept_anomaly": 0,
        "total_anomalies_kept": 0,
    })
    return counts


def _make_kept_item(
    pred: Mapping[str, Any],
    *,
    final_label: str,
    include_trace_fields: bool,
    technical_anomaly: bool = False,
    anomaly_reason: Optional[str] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "label": final_label,
        "bbox_xyxy": [float(v) for v in pred["orig_pred_bbox_xyxy"]],
        "score": float(pred.get("score", 1.0)),
    }
    if include_trace_fields:
        item = {
            "prediction_id": pred["prediction_id"],
            "label": item["label"],
            "bbox_xyxy": item["bbox_xyxy"],
            "score": item["score"],
            "orig_order": pred.get("orig_order"),
            "tile_id": pred.get("tile_id"),
            "call_id": pred.get("call_id"),
            "unique_call_id": pred.get("unique_call_id"),
        }
        if technical_anomaly:
            item["technical_anomaly"] = True
            item["anomaly_reason"] = anomaly_reason
        if extra_fields:
            item.update(extra_fields)
    return item


def _evaluate_single_stage_candidate(
    pred: Mapping[str, Any],
    records_by_call_id: Mapping[str, Mapping[str, Any]],
    target_set: set[str],
) -> Dict[str, Any]:
    """Classify a single candidate's verifier call outcome cleanly without mutating predictions."""
    orig_lbl = str(pred["orig_pred_label"]).strip().lower()
    if not pred.get("crop_valid"):
        return {
            "kind": "invalid_crop",
            "orig_lbl": orig_lbl,
            "ucid": None,
            "strict_json_ok": False,
            "strict_lbl": None,
            "is_truncated": False,
            "technical_incomplete": True,
        }
    ucid = str(pred["unique_call_id"])
    rec = records_by_call_id.get(ucid)
    if rec is None:
        return {
            "kind": "missing_record",
            "orig_lbl": orig_lbl,
            "ucid": ucid,
            "strict_json_ok": False,
            "strict_lbl": None,
            "is_truncated": False,
            "technical_incomplete": True,
        }
    if not rec.get("ok"):
        return {
            "kind": "technical_failure",
            "orig_lbl": orig_lbl,
            "ucid": ucid,
            "strict_json_ok": False,
            "strict_lbl": None,
            "is_truncated": False,
            "technical_incomplete": True,
        }
    strict_ok = bool(rec.get("strict_json_ok"))
    strict_lbl = rec.get("strict_label") if strict_ok else None
    if not strict_ok or strict_lbl is None:
        return {
            "kind": "format_or_truncated",
            "orig_lbl": orig_lbl,
            "ucid": ucid,
            "strict_json_ok": False,
            "strict_lbl": None,
            "is_truncated": rec.get("has_normal_eos") is False,
            "technical_incomplete": False,
        }
    if strict_lbl == "other":
        return {
            "kind": "other",
            "orig_lbl": orig_lbl,
            "ucid": ucid,
            "strict_json_ok": True,
            "strict_lbl": "other",
            "is_truncated": False,
            "technical_incomplete": False,
        }
    if strict_lbl not in target_set:
        return {
            "kind": "unknown_label",
            "orig_lbl": orig_lbl,
            "ucid": ucid,
            "strict_json_ok": True,
            "strict_lbl": strict_lbl,
            "is_truncated": False,
            "technical_incomplete": False,
        }
    return {
        "kind": "target",
        "orig_lbl": orig_lbl,
        "ucid": ucid,
        "strict_json_ok": True,
        "strict_lbl": strict_lbl,
        "is_truncated": False,
        "technical_incomplete": False,
    }


def apply_verifier_filter_policy(
    *,
    predictions: Sequence[Mapping[str, Any]],
    verification_records_by_call_id: Mapping[str, Mapping[str, Any]],
    policy: str,
    target_classes: Sequence[str] = TARGET_SIX_CLASSES,
) -> Dict[str, Any]:
    """Apply frozen `B_other`, `B_other_strict_keep_anomaly`, `B_exact`, or `YOLO_PROPOSAL_ASSIGN_QWEN3_LABEL` policy without modifying box coordinates."""
    if policy not in ("B_other", "B_other_strict_keep_anomaly", "B_exact", "YOLO_PROPOSAL_ASSIGN_QWEN3_LABEL"):
        raise ValueError(f"Unsupported verifier filter policy: {policy!r}")

    keep_anomaly_strict = policy == "B_other_strict_keep_anomaly"
    target_set = {str(c).strip().lower() for c in target_classes}
    kept_by_board: Dict[str, List[Dict[str, Any]]] = {}
    outcome_counts = (
        _build_strict_counts(len(predictions))
        if keep_anomaly_strict
        else _build_legacy_counts(len(predictions))
    )
    decision_rows: List[Dict[str, Any]] = []

    def _append_kept(
        pred: Mapping[str, Any],
        *,
        final_label: str,
        technical_anomaly: bool = False,
        anomaly_reason: Optional[str] = None,
    ) -> None:
        bid = str(pred["board_id"])
        kept_by_board[bid].append(
            _make_kept_item(
                pred,
                final_label=final_label,
                include_trace_fields=keep_anomaly_strict,
                technical_anomaly=technical_anomaly,
                anomaly_reason=anomaly_reason,
            )
        )

    for pred in predictions:
        bid = str(pred["board_id"])
        kept_by_board.setdefault(bid, [])
        ev = _evaluate_single_stage_candidate(pred, verification_records_by_call_id, target_set)
        kind = ev["kind"]
        orig_lbl = ev["orig_lbl"]
        ucid = ev["ucid"]

        if kind == "invalid_crop":
            if keep_anomaly_strict:
                outcome_counts["retained_count"] += 1
                outcome_counts["invalid_crop_kept_anomaly"] += 1
                outcome_counts["total_anomalies_kept"] += 1
                _append_kept(pred, final_label=orig_lbl, technical_anomaly=True, anomaly_reason="invalid_crop_kept_anomaly")
                decision_rows.append({
                    "prediction_id": pred["prediction_id"],
                    "board_id": bid,
                    "orig_pred_label": orig_lbl,
                    "retained": True,
                    "reason": "invalid_crop_kept_anomaly",
                    "technical_anomaly": True,
                    "anomaly_reason": "invalid_crop_kept_anomaly",
                    "strict_json_ok": False,
                    "strict_label": None,
                })
            else:
                outcome_counts["filtered_count"] += 1
                outcome_counts["invalid_crop_filtered"] += 1
                decision_rows.append({
                    "prediction_id": pred["prediction_id"],
                    "board_id": bid,
                    "orig_pred_label": orig_lbl,
                    "retained": False,
                    "reason": "invalid_crop",
                    "strict_json_ok": False,
                    "strict_label": None,
                })
            continue

        if kind in ("missing_record", "technical_failure"):
            if kind == "missing_record":
                outcome_counts["missing_record_count"] += 1
            if keep_anomaly_strict:
                anom_reason = "missing_record_kept_anomaly" if kind == "missing_record" else "technical_failure_kept_anomaly"
                outcome_counts[anom_reason] += 1
                outcome_counts["total_anomalies_kept"] += 1
                outcome_counts["retained_count"] += 1
                _append_kept(pred, final_label=orig_lbl, technical_anomaly=True, anomaly_reason=anom_reason)
                decision_rows.append({
                    "prediction_id": pred["prediction_id"],
                    "board_id": bid,
                    "orig_pred_label": orig_lbl,
                    "unique_call_id": ucid,
                    "retained": True,
                    "reason": anom_reason,
                    "technical_anomaly": True,
                    "anomaly_reason": anom_reason,
                    "strict_json_ok": False,
                    "strict_label": None,
                })
            else:
                outcome_counts["filtered_count"] += 1
                outcome_counts["technical_failure_filtered"] += 1
                decision_rows.append({
                    "prediction_id": pred["prediction_id"],
                    "board_id": bid,
                    "orig_pred_label": orig_lbl,
                    "unique_call_id": ucid,
                    "retained": False,
                    "reason": "technical_failure_or_missing",
                    "strict_json_ok": False,
                    "strict_label": None,
                })
            continue

        if kind == "format_or_truncated":
            if keep_anomaly_strict:
                anom_reason = "truncated_kept_anomaly" if ev["is_truncated"] else "format_failure_kept_anomaly"
                outcome_counts[anom_reason] += 1
                outcome_counts["total_anomalies_kept"] += 1
                outcome_counts["retained_count"] += 1
                _append_kept(pred, final_label=orig_lbl, technical_anomaly=True, anomaly_reason=anom_reason)
                decision_rows.append({
                    "prediction_id": pred["prediction_id"],
                    "board_id": bid,
                    "orig_pred_label": orig_lbl,
                    "unique_call_id": ucid,
                    "retained": True,
                    "reason": anom_reason,
                    "technical_anomaly": True,
                    "anomaly_reason": anom_reason,
                    "strict_json_ok": False,
                    "strict_label": None,
                })
            else:
                outcome_counts["filtered_count"] += 1
                outcome_counts["format_failure_filtered"] += 1
                decision_rows.append({
                    "prediction_id": pred["prediction_id"],
                    "board_id": bid,
                    "orig_pred_label": orig_lbl,
                    "unique_call_id": ucid,
                    "retained": False,
                    "reason": "format_failure",
                    "strict_json_ok": False,
                    "strict_label": None,
                })
            continue

        strict_lbl = ev["strict_lbl"]
        if kind in ("other", "unknown_label"):
            if kind == "unknown_label" and keep_anomaly_strict:
                outcome_counts["retained_count"] += 1
                outcome_counts["format_failure_kept_anomaly"] += 1
                outcome_counts["total_anomalies_kept"] += 1
                _append_kept(pred, final_label=orig_lbl, technical_anomaly=True, anomaly_reason="unknown_label_kept_anomaly")
                decision_rows.append({
                    "prediction_id": pred["prediction_id"],
                    "board_id": bid,
                    "orig_pred_label": orig_lbl,
                    "unique_call_id": ucid,
                    "retained": True,
                    "reason": "unknown_label_kept_anomaly",
                    "technical_anomaly": True,
                    "anomaly_reason": "unknown_label_kept_anomaly",
                    "strict_json_ok": True,
                    "strict_label": strict_lbl,
                })
                continue
            outcome_counts["filtered_count"] += 1
            outcome_counts["rejected_as_other"] += 1
            drow: Dict[str, Any] = {
                "prediction_id": pred["prediction_id"],
                "board_id": bid,
                "orig_pred_label": orig_lbl,
                "unique_call_id": ucid,
                "retained": False,
                "reason": "rejected_as_other",
                "strict_json_ok": True,
                "strict_label": strict_lbl,
            }
            if keep_anomaly_strict:
                drow["technical_anomaly"] = False
            decision_rows.append(drow)
            continue

        same_class = strict_lbl == orig_lbl
        if policy == "YOLO_PROPOSAL_ASSIGN_QWEN3_LABEL":
            retain = True
            reason = "retained_assigned_verifier_class"
            final_label = strict_lbl
            outcome_counts["exact_same_class_retained"] += 1
        elif policy in ("B_other", "B_other_strict_keep_anomaly"):
            retain = True
            reason = "retained_exact_same" if same_class else "retained_cross_target_keep_orig_label"
            final_label = orig_lbl
            if same_class:
                outcome_counts["exact_same_class_retained"] += 1
            else:
                outcome_counts["cross_target_retained_same_orig_class"] += 1
        else:
            retain = same_class
            reason = "retained_exact_same" if same_class else "rejected_cross_target_mismatch"
            final_label = orig_lbl
            if same_class:
                outcome_counts["exact_same_class_retained"] += 1
            else:
                outcome_counts["cross_target_rejected"] += 1

        if retain:
            outcome_counts["retained_count"] += 1
            _append_kept(pred, final_label=final_label, technical_anomaly=False)
        else:
            outcome_counts["filtered_count"] += 1

        drow_t: Dict[str, Any] = {
            "prediction_id": pred["prediction_id"],
            "board_id": bid,
            "orig_pred_label": orig_lbl,
            "unique_call_id": ucid,
            "retained": retain,
            "reason": reason,
            "strict_json_ok": True,
            "strict_label": strict_lbl,
        }
        if keep_anomaly_strict:
            drow_t["technical_anomaly"] = False
        decision_rows.append(drow_t)

    if keep_anomaly_strict:
        tech_complete = (
            outcome_counts["technical_failure_kept_anomaly"] == 0
            and outcome_counts["missing_record_count"] == 0
            and outcome_counts["invalid_crop_kept_anomaly"] == 0
        )
    else:
        tech_complete = (
            outcome_counts["technical_failure_filtered"] == 0
            and outcome_counts["missing_record_count"] == 0
        )

    return {
        "policy": policy,
        "technical_complete": tech_complete,
        "counts": outcome_counts,
        "kept_predictions_by_board": kept_by_board,
        "decisions": decision_rows,
    }


def apply_dual_crop_consensus_policy(
    *,
    stage1_predictions: Sequence[Mapping[str, Any]],
    stage1_records_by_call_id: Mapping[str, Mapping[str, Any]],
    stage2_predictions_by_pid: Mapping[str, Mapping[str, Any]],
    stage2_records_by_call_id: Mapping[str, Mapping[str, Any]],
    target_classes: Sequence[str] = TARGET_SIX_CLASSES,
) -> Dict[str, Any]:
    """Apply Round 7 conservative `B_other_dual_crop_consensus`: reject ONLY when BOTH 10% and 40% crops output strict 'other'."""
    target_set = {str(c).strip().lower() for c in target_classes}
    kept_by_board: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {
        "total_candidates": len(stage1_predictions),
        "retained_count": 0,
        "filtered_count": 0,
        "rejected_as_other": 0,
        "stage1_retained_count": 0,
        "stage1_exact_same_class_retained": 0,
        "stage1_cross_target_retained_same_orig_class": 0,
        "stage1_rejected_other_count": 0,
        "stage2_required_count": 0,
        "stage2_evaluated_count": 0,
        "rescued_by_stage2_target_count": 0,
        "rescued_stage2_exact_same": 0,
        "rescued_stage2_cross_target": 0,
        "rejected_dual_crop_other_consensus": 0,
        "exact_same_class_retained": 0,
        "cross_target_retained_same_orig_class": 0,
        "format_failure_filtered": 0,
        "technical_failure_filtered": 0,
        "missing_record_count": 0,
        "stage1_invalid_crop_kept_anomaly": 0,
        "stage1_missing_record_kept_anomaly": 0,
        "stage1_technical_failure_kept_anomaly": 0,
        "stage1_truncated_kept_anomaly": 0,
        "stage1_format_failure_kept_anomaly": 0,
        "stage2_invalid_crop_kept_anomaly": 0,
        "stage2_missing_record_kept_anomaly": 0,
        "stage2_technical_failure_kept_anomaly": 0,
        "stage2_truncated_kept_anomaly": 0,
        "stage2_format_failure_kept_anomaly": 0,
        "invalid_crop_kept_anomaly": 0,
        "missing_record_kept_anomaly": 0,
        "technical_failure_kept_anomaly": 0,
        "truncated_kept_anomaly": 0,
        "format_failure_kept_anomaly": 0,
        "total_anomalies_kept": 0,
    }
    decision_rows: List[Dict[str, Any]] = []
    tech_incomplete = False

    def _record_anomaly(
        pred: Mapping[str, Any],
        *,
        stage_prefix: str,
        agg_key: str,
        reason: str,
        s1_ucid: Optional[str],
        s1_lbl: Optional[str],
        s1_ok: bool,
        s2_ucid: Optional[str],
        s2_lbl: Optional[str],
        s2_ok: bool,
        is_tech_incomplete: bool,
    ) -> None:
        nonlocal tech_incomplete
        if is_tech_incomplete:
            tech_incomplete = True
        bid = str(pred["board_id"])
        orig_lbl = str(pred["orig_pred_label"]).strip().lower()
        counts["retained_count"] += 1
        counts[f"{stage_prefix}_{agg_key}"] += 1
        counts[agg_key] += 1
        counts["total_anomalies_kept"] += 1
        if agg_key == "missing_record_kept_anomaly":
            counts["missing_record_count"] += 1
        kept_by_board[bid].append(
            _make_kept_item(
                pred,
                final_label=orig_lbl,
                include_trace_fields=True,
                technical_anomaly=True,
                anomaly_reason=reason,
                extra_fields={
                    "stage1_unique_call_id": s1_ucid,
                    "stage2_unique_call_id": s2_ucid,
                    "stage1_strict_label": s1_lbl,
                    "stage2_strict_label": s2_lbl,
                    "retention_source": f"{stage_prefix}_anomaly",
                },
            )
        )
        decision_rows.append({
            "prediction_id": pred["prediction_id"],
            "board_id": bid,
            "orig_order": pred.get("orig_order"),
            "orig_pred_label": orig_lbl,
            "unique_call_id": s2_ucid or s1_ucid,
            "stage1_unique_call_id": s1_ucid,
            "stage1_strict_json_ok": s1_ok,
            "stage1_strict_label": s1_lbl,
            "stage2_unique_call_id": s2_ucid,
            "stage2_strict_json_ok": s2_ok,
            "stage2_strict_label": s2_lbl,
            "retained": True,
            "reason": reason,
            "technical_anomaly": True,
            "anomaly_reason": reason,
            "strict_json_ok": s2_ok if s2_ucid is not None else s1_ok,
            "strict_label": s2_lbl if s2_ucid is not None else s1_lbl,
        })

    for pred in stage1_predictions:
        bid = str(pred["board_id"])
        pid = str(pred["prediction_id"])
        kept_by_board.setdefault(bid, [])
        ev1 = _evaluate_single_stage_candidate(pred, stage1_records_by_call_id, target_set)
        k1, orig_lbl, u1 = ev1["kind"], ev1["orig_lbl"], ev1["ucid"]

        if k1 == "invalid_crop":
            _record_anomaly(pred, stage_prefix="stage1", agg_key="invalid_crop_kept_anomaly", reason="stage1_invalid_crop_kept_anomaly", s1_ucid=None, s1_lbl=None, s1_ok=False, s2_ucid=None, s2_lbl=None, s2_ok=False, is_tech_incomplete=True)
            continue
        if k1 == "missing_record":
            _record_anomaly(pred, stage_prefix="stage1", agg_key="missing_record_kept_anomaly", reason="stage1_missing_record_kept_anomaly", s1_ucid=u1, s1_lbl=None, s1_ok=False, s2_ucid=None, s2_lbl=None, s2_ok=False, is_tech_incomplete=True)
            continue
        if k1 == "technical_failure":
            _record_anomaly(pred, stage_prefix="stage1", agg_key="technical_failure_kept_anomaly", reason="stage1_technical_failure_kept_anomaly", s1_ucid=u1, s1_lbl=None, s1_ok=False, s2_ucid=None, s2_lbl=None, s2_ok=False, is_tech_incomplete=True)
            continue
        if k1 == "format_or_truncated":
            agg = "truncated_kept_anomaly" if ev1["is_truncated"] else "format_failure_kept_anomaly"
            _record_anomaly(pred, stage_prefix="stage1", agg_key=agg, reason=f"stage1_{agg}", s1_ucid=u1, s1_lbl=None, s1_ok=False, s2_ucid=None, s2_lbl=None, s2_ok=False, is_tech_incomplete=False)
            continue
        if k1 == "unknown_label":
            _record_anomaly(pred, stage_prefix="stage1", agg_key="format_failure_kept_anomaly", reason="stage1_unknown_label_kept_anomaly", s1_ucid=u1, s1_lbl=ev1["strict_lbl"], s1_ok=True, s2_ucid=None, s2_lbl=None, s2_ok=False, is_tech_incomplete=False)
            continue

        if k1 == "target":
            same1 = ev1["strict_lbl"] == orig_lbl
            counts["retained_count"] += 1
            counts["stage1_retained_count"] += 1
            if same1:
                counts["stage1_exact_same_class_retained"] += 1
                counts["exact_same_class_retained"] += 1
                reason = "stage1_retained_exact_same"
            else:
                counts["stage1_cross_target_retained_same_orig_class"] += 1
                counts["cross_target_retained_same_orig_class"] += 1
                reason = "stage1_retained_cross_target_keep_orig_label"
            kept_by_board[bid].append(
                _make_kept_item(
                    pred,
                    final_label=orig_lbl,
                    include_trace_fields=True,
                    technical_anomaly=False,
                    extra_fields={
                        "stage1_unique_call_id": u1,
                        "stage2_unique_call_id": None,
                        "stage1_strict_label": ev1["strict_lbl"],
                        "stage2_strict_label": None,
                        "retention_source": "stage1_target",
                    },
                )
            )
            decision_rows.append({
                "prediction_id": pid,
                "board_id": bid,
                "orig_order": pred.get("orig_order"),
                "orig_pred_label": orig_lbl,
                "unique_call_id": u1,
                "stage1_unique_call_id": u1,
                "stage1_strict_json_ok": True,
                "stage1_strict_label": ev1["strict_lbl"],
                "stage2_unique_call_id": None,
                "stage2_strict_json_ok": None,
                "stage2_strict_label": None,
                "retained": True,
                "reason": reason,
                "technical_anomaly": False,
                "strict_json_ok": True,
                "strict_label": ev1["strict_lbl"],
            })
            continue

        # Stage-1 strict "other" -> must check Stage-2 (40% wider context crop)
        counts["stage1_rejected_other_count"] += 1
        counts["stage2_required_count"] += 1
        pred2 = stage2_predictions_by_pid.get(pid)
        if pred2 is None:
            _record_anomaly(pred, stage_prefix="stage2", agg_key="missing_record_kept_anomaly", reason="stage2_missing_manifest_entry_kept_anomaly", s1_ucid=u1, s1_lbl="other", s1_ok=True, s2_ucid=None, s2_lbl=None, s2_ok=False, is_tech_incomplete=True)
            continue

        counts["stage2_evaluated_count"] += 1
        ev2 = _evaluate_single_stage_candidate(pred2, stage2_records_by_call_id, target_set)
        k2, u2 = ev2["kind"], ev2["ucid"]

        if k2 == "invalid_crop":
            _record_anomaly(pred, stage_prefix="stage2", agg_key="invalid_crop_kept_anomaly", reason="stage2_invalid_crop_kept_anomaly", s1_ucid=u1, s1_lbl="other", s1_ok=True, s2_ucid=None, s2_lbl=None, s2_ok=False, is_tech_incomplete=True)
            continue
        if k2 == "missing_record":
            _record_anomaly(pred, stage_prefix="stage2", agg_key="missing_record_kept_anomaly", reason="stage2_missing_record_kept_anomaly", s1_ucid=u1, s1_lbl="other", s1_ok=True, s2_ucid=u2, s2_lbl=None, s2_ok=False, is_tech_incomplete=True)
            continue
        if k2 == "technical_failure":
            _record_anomaly(pred, stage_prefix="stage2", agg_key="technical_failure_kept_anomaly", reason="stage2_technical_failure_kept_anomaly", s1_ucid=u1, s1_lbl="other", s1_ok=True, s2_ucid=u2, s2_lbl=None, s2_ok=False, is_tech_incomplete=True)
            continue
        if k2 == "format_or_truncated":
            agg = "truncated_kept_anomaly" if ev2["is_truncated"] else "format_failure_kept_anomaly"
            _record_anomaly(pred, stage_prefix="stage2", agg_key=agg, reason=f"stage2_{agg}", s1_ucid=u1, s1_lbl="other", s1_ok=True, s2_ucid=u2, s2_lbl=None, s2_ok=False, is_tech_incomplete=False)
            continue
        if k2 == "unknown_label":
            _record_anomaly(pred, stage_prefix="stage2", agg_key="format_failure_kept_anomaly", reason="stage2_unknown_label_kept_anomaly", s1_ucid=u1, s1_lbl="other", s1_ok=True, s2_ucid=u2, s2_lbl=ev2["strict_lbl"], s2_ok=True, is_tech_incomplete=False)
            continue

        if k2 == "other":
            counts["filtered_count"] += 1
            counts["rejected_as_other"] += 1
            counts["rejected_dual_crop_other_consensus"] += 1
            decision_rows.append({
                "prediction_id": pid,
                "board_id": bid,
                "orig_order": pred.get("orig_order"),
                "orig_pred_label": orig_lbl,
                "unique_call_id": u2,
                "stage1_unique_call_id": u1,
                "stage1_strict_json_ok": True,
                "stage1_strict_label": "other",
                "stage2_unique_call_id": u2,
                "stage2_strict_json_ok": True,
                "stage2_strict_label": "other",
                "retained": False,
                "reason": "rejected_dual_crop_other_consensus",
                "technical_anomaly": False,
                "strict_json_ok": True,
                "strict_label": "other",
            })
            continue

        # k2 == "target" -> Stage-2 rescued original R5 box & label!
        same2 = ev2["strict_lbl"] == orig_lbl
        counts["retained_count"] += 1
        counts["rescued_by_stage2_target_count"] += 1
        if same2:
            counts["rescued_stage2_exact_same"] += 1
            counts["exact_same_class_retained"] += 1
            reason = "rescued_stage2_exact_same"
        else:
            counts["rescued_stage2_cross_target"] += 1
            counts["cross_target_retained_same_orig_class"] += 1
            reason = "rescued_stage2_cross_target_keep_orig_label"
        kept_by_board[bid].append(
            _make_kept_item(
                pred,
                final_label=orig_lbl,
                include_trace_fields=True,
                technical_anomaly=False,
                extra_fields={
                    "stage1_unique_call_id": u1,
                    "stage2_unique_call_id": u2,
                    "stage1_strict_label": "other",
                    "stage2_strict_label": ev2["strict_lbl"],
                    "retention_source": "stage2_rescued_target",
                },
            )
        )
        decision_rows.append({
            "prediction_id": pid,
            "board_id": bid,
            "orig_order": pred.get("orig_order"),
            "orig_pred_label": orig_lbl,
            "unique_call_id": u2,
            "stage1_unique_call_id": u1,
            "stage1_strict_json_ok": True,
            "stage1_strict_label": "other",
            "stage2_unique_call_id": u2,
            "stage2_strict_json_ok": True,
            "stage2_strict_label": ev2["strict_lbl"],
            "retained": True,
            "reason": reason,
            "technical_anomaly": False,
            "strict_json_ok": True,
            "strict_label": ev2["strict_lbl"],
        })

    return {
        "policy": "B_other_dual_crop_consensus",
        "technical_complete": not tech_incomplete,
        "counts": counts,
        "kept_predictions_by_board": kept_by_board,
        "decisions": decision_rows,
    }
