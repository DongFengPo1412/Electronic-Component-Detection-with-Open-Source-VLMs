"""In-repo CPU evaluation replay for FICS-PCB DSLR VLM outputs (v2 termination semantics).

Design guarantees:
1. FICS 6-class config (`configs/fics_six_classes.yaml`) and COCO category tables are verified
   against `FICS_CANONICAL_CATEGORIES` and never mutate or conflate with legacy WACV 11-class defaults,
   while checking alignment with the shared `parse_response()` canonical allow-list.
2. Separates `length_limited_without_eos` (hit `max_new_tokens` without trailing EOS) from
   `syntax_or_format_incomplete_after_eos` (normal EOS present, JSON/contract incomplete),
   and keeps JSON/schema validity orthogonal to token-budget exhaustion.
3. Distinguishes `strict_schema_ok` (finite numeric xyxy coords, valid label & ordering) from
   compatible non-strict extraction (`parse_response`), and reserves `explicit_empty` strictly
   for exact valid empty array `[]`.
4. Enforces frozen-plan / manifest identity binding across model identity, queue call specs,
   geometry offsets/dimensions, three-way image SHA-256 + optional on-disk re-hashing,
   runtime prompt SHA-256 verification, token length & EOS/finish_reason consistency, and
   trusted GT COCO identity & category compatibility.
5. Maps local ROI/tile detections to full board coordinates and applies NMS strictly per board
   (never cross-board); validates unique COCO `image_id`s and compatible category tables.
6. Never silently scores incomplete, missing, or technically failed runs as completed 0-score runs.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image

from pcb_vlm.config import get_canonical_categories
from pcb_vlm.data.fics_dslr import FICS_CANONICAL_CATEGORIES, file_sha256
from pcb_vlm.geometry.coordinate_protocol import PROTOCOLS, generation_finished, parse_response, to_pixels
from pcb_vlm.geometry.tiling import map_tile_box_to_global, nms_merge_predictions

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import evaluate as repo_eval  # noqa: E402

FICS_SIX_CLASSES: Tuple[str, ...] = tuple(c["name"] for c in FICS_CANONICAL_CATEGORIES)
DEFAULT_FICS_CONFIG_PATH = _REPO_ROOT / "configs" / "fics_six_classes.yaml"
EOS_TEXT_TOKENS = ("<|im_end|>", "<|endoftext|>", "</s>", "<｜end▁of▁sentence｜>", "<|end▁of▁sentence|>")
_COMPLETE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\r?\n?(.*?)\r?\n?```\s*$", re.DOTALL | re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable|sorry|apologize|not possible|do not have|don't have|"
    r"incapable|cannot detect|cannot locate|unable to provide|unable to detect|"
    r"as an ai|i am unable|i cannot|won't be able)\b",
    re.IGNORECASE,
)


def _is_int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_valid_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value.strip()))


def sha256_text(text: str) -> str:
    """Return lowercase UTF-8 SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_category_maps(
    categories: Any,
    *,
    context: str,
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Validate a COCO/config category list and return strict (id->name, name->id) dicts."""
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{context}: missing or empty categories table")
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for entry in categories:
        if not isinstance(entry, dict):
            raise ValueError(f"{context}: invalid category entry {entry!r}")
        cid = entry.get("id")
        cname = entry.get("name")
        if not _is_int_not_bool(cid) or not isinstance(cname, str) or not cname.strip():
            raise ValueError(f"{context}: invalid category id/name in {entry!r}")
        norm_name = cname.strip().lower()
        if cid in id_to_name and id_to_name[cid] != norm_name:
            raise ValueError(
                f"{context}: duplicate conflicting category_id={cid}: {id_to_name[cid]!r} vs {norm_name!r}"
            )
        if cid in id_to_name:
            raise ValueError(f"{context}: duplicate category_id={cid}")
        if norm_name in name_to_id:
            raise ValueError(
                f"{context}: duplicate category name {norm_name!r} for ids {name_to_id[norm_name]} and {cid}"
            )
        id_to_name[cid] = norm_name
        name_to_id[norm_name] = cid
    return id_to_name, name_to_id


def load_fics_six_class_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and verify the isolated FICS 6-class evaluation configuration YAML."""
    path = Path(config_path) if config_path is not None else DEFAULT_FICS_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "classes" not in data or "protocol" not in data:
        raise ValueError(f"Invalid FICS 6-class config at {path}")

    cfg_id_to_name, _ = _canonical_category_maps(data["classes"], context=f"FICS config {path}")
    canon_id_to_name, _ = _canonical_category_maps(FICS_CANONICAL_CATEGORIES, context="FICS_CANONICAL_CATEGORIES")
    if cfg_id_to_name != canon_id_to_name:
        raise ValueError(
            f"FICS config classes mismatch canonical FICS 6-class mapping: {cfg_id_to_name} vs {canon_id_to_name}"
        )

    parser_allowed = {c.strip().lower() for c in get_canonical_categories()}
    missing_in_parser = sorted(set(cfg_id_to_name.values()) - parser_allowed)
    if missing_in_parser:
        raise ValueError(
            f"FICS config classes missing from shared coordinate_protocol allow-list: {missing_in_parser}"
        )

    proto_cfg = data["protocol"]
    if not isinstance(proto_cfg, dict):
        raise ValueError(f"Invalid protocol section in FICS config at {path}")
    coord_proto = proto_cfg.get("coordinate_protocol")
    if coord_proto not in PROTOCOLS:
        raise ValueError(f"Unsupported coordinate_protocol in FICS config: {coord_proto!r}")
    for thresh_key in ("nms_iou_threshold", "eval_iou_threshold"):
        val = proto_cfg.get(thresh_key)
        if not _is_finite_number(val) or not (0.0 < float(val) <= 1.0):
            raise ValueError(f"Invalid {thresh_key} in FICS config: {val!r}")
    max_toks = proto_cfg.get("max_new_tokens_practical")
    if not _is_int_not_bool(max_toks) or max_toks <= 0:
        raise ValueError(f"Invalid max_new_tokens_practical in FICS config: {max_toks!r}")

    prompt_text = proto_cfg.get("prompt")
    declared_prompt_sha = proto_cfg.get("prompt_sha256")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError(f"Missing prompt text in FICS config at {path}")
    if not _is_valid_sha256_hex(declared_prompt_sha):
        raise ValueError(f"Missing or malformed prompt_sha256 in FICS config at {path}: {declared_prompt_sha!r}")
    computed_prompt_sha = sha256_text(prompt_text)
    if computed_prompt_sha != declared_prompt_sha.lower():
        raise ValueError(
            f"FICS config prompt text SHA-256 mismatch at {path}: computed {computed_prompt_sha} != declared {declared_prompt_sha}"
        )
    return data


def strip_eos_and_outer_fence(raw_text: str) -> Tuple[str, bool]:
    """Strip known EOS surface tokens and an outer complete Markdown fence."""
    text = raw_text.strip()
    for tok in EOS_TEXT_TOKENS:
        text = text.replace(tok, "").strip()
    match = _COMPLETE_FENCE_RE.match(text)
    if match:
        return match.group(1).strip(), True
    return text, False


def _reject_json_constant(val: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {val!r}")


def validate_strict_json_contract(
    cleaned_text: str,
    allowed_classes: Sequence[str] = FICS_SIX_CLASSES,
    protocol: str = "xyxy_norm1000",
) -> Dict[str, Any]:
    """Validate strict JSON array syntax and `{bbox_2d, label}` finite-numeric contract."""
    try:
        parsed = json.loads(cleaned_text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        return {
            "complete_json_syntax_ok": False,
            "complete_json_error": f"JSONDecodeError: {exc.msg} (pos {exc.pos})",
            "strict_schema_ok": False,
            "parsed_array_len": None,
            "malformed_object_items": 0,
            "schema_violation_details": [],
        }
    except ValueError as exc:
        return {
            "complete_json_syntax_ok": False,
            "complete_json_error": str(exc),
            "strict_schema_ok": False,
            "parsed_array_len": None,
            "malformed_object_items": 0,
            "schema_violation_details": ["nonfinite_json_constant"],
        }
    if not isinstance(parsed, list):
        return {
            "complete_json_syntax_ok": False,
            "complete_json_error": f"Expected JSON list, got {type(parsed).__name__}",
            "strict_schema_ok": False,
            "parsed_array_len": None,
            "malformed_object_items": 0,
            "schema_violation_details": [],
        }

    malformed = 0
    violations: List[str] = []
    allowed_set = {c.strip().lower() for c in allowed_classes}
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            return {
                "complete_json_syntax_ok": False,
                "complete_json_error": f"Element {idx} is {type(item).__name__}, expected dict",
                "strict_schema_ok": False,
                "parsed_array_len": len(parsed),
                "malformed_object_items": len(parsed),
                "schema_violation_details": [f"item_{idx}_not_dict"],
            }
        bbox = item.get("bbox_2d")
        lbl = item.get("label")
        if not isinstance(lbl, str) or lbl.strip().lower() not in allowed_set:
            malformed += 1
            violations.append(f"item_{idx}_invalid_label:{lbl!r}")
            continue
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_finite_number(v) for v in bbox):
            malformed += 1
            violations.append(f"item_{idx}_non_numeric_or_nonfinite_bbox")
            continue
        try:
            coords = [float(v) for v in bbox]
            to_pixels(coords, 1000, 1000, protocol=protocol)
        except Exception as exc:
            malformed += 1
            violations.append(f"item_{idx}_invalid_{protocol}:{exc}")

    return {
        "complete_json_syntax_ok": True,
        "complete_json_error": None,
        "strict_schema_ok": malformed == 0,
        "parsed_array_len": len(parsed),
        "malformed_object_items": malformed,
        "schema_violation_details": violations[:10],
    }


def _apply_path_remaps(raw_path: str, path_remaps: Sequence[Tuple[str, str]]) -> Optional[Path]:
    """Attempt cross-machine path prefix remapping preserving relative subpath separators."""
    norm_raw = raw_path.replace("\\", "/")
    norm_raw_lower = norm_raw.lower()
    for old_prefix, new_prefix in path_remaps:
        old_clean = old_prefix.replace("\\", "/").rstrip("/")
        if not old_clean:
            continue
        if norm_raw_lower == old_clean.lower():
            return Path(new_prefix)
        if norm_raw_lower.startswith(old_clean.lower() + "/"):
            rel = norm_raw[len(old_clean) + 1 :]
            return Path(new_prefix) / Path(rel)
    return None


def resolve_portable_image_path(
    row: Dict[str, Any],
    *,
    expected_call_spec: Optional[Dict[str, Any]] = None,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_hash_index: Optional[Dict[str, Path]] = None,
) -> Optional[Path]:
    """Resolve an input image path cleanly via local path, explicit prefix remap, or SHA-256 index."""
    target_sha = str(row.get("expected_sha256") or row.get("input_sha256") or "").lower()
    if image_hash_index and target_sha and target_sha in image_hash_index:
        cand = Path(image_hash_index[target_sha])
        if cand.is_file():
            return cand

    raw_candidates: List[str] = []
    if row.get("image_path"):
        raw_candidates.append(str(row["image_path"]))
    if expected_call_spec and expected_call_spec.get("path"):
        sp = str(expected_call_spec["path"])
        if sp not in raw_candidates:
            raw_candidates.append(sp)

    if path_remaps:
        for raw_p in raw_candidates:
            remapped = _apply_path_remaps(raw_p, path_remaps)
            if remapped is not None and remapped.is_file():
                return remapped

    for raw_p in raw_candidates:
        cand = Path(raw_p)
        if cand.is_file():
            return cand

    if path_remaps and raw_candidates:
        remapped = _apply_path_remaps(raw_candidates[0], path_remaps)
        if remapped is not None:
            return remapped
    return Path(raw_candidates[0]) if raw_candidates else None


def build_image_hash_index(directories: Sequence[Path]) -> Dict[str, Path]:
    """Index PNG/TIF/JPG image files under given directories by lowercase SHA-256 for portable verification."""
    index: Dict[str, Path] = {}
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    for root in directories:
        rpath = Path(root)
        if not rpath.exists():
            continue
        if rpath.is_file():
            index[file_sha256(rpath).lower()] = rpath
            continue
        for cand in sorted(rpath.rglob("*")):
            if cand.is_file() and cand.suffix.lower() in exts:
                sha = file_sha256(cand).lower()
                index.setdefault(sha, cand)
    return index


def validate_call_record_integrity(
    row: Dict[str, Any],
    *,
    expected_model_key: Optional[str] = None,
    expected_model_spec: Optional[Dict[str, Any]] = None,
    expected_call_spec: Optional[Dict[str, Any]] = None,
    expected_prompt_sha256: Optional[str] = None,
    expected_protocol: str = "xyxy_norm1000",
    expected_max_new_tokens: Optional[int] = None,
    expected_eos_token_ids: Optional[Sequence[int]] = None,
    verify_disk_hashes: bool = False,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_hash_index: Optional[Dict[str, Path]] = None,
) -> bool:
    """Strictly verify record provenance, geometry, hashes, token budget, and optional disk image bytes.

    Returns True when an on-disk image file was hashed and dimension-verified.
    """
    cid = row.get("call_id")
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError(f"Record missing valid 'call_id': {row!r}")

    for req_str in ("board_id", "mode", "model_key", "model_repo_id", "coordinate_protocol"):
        val = row.get(req_str)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Call {cid!r} missing required provenance field {req_str!r}")

    if expected_model_key is not None and row["model_key"] != expected_model_key:
        raise ValueError(
            f"Call {cid!r} model_key mismatch: expected {expected_model_key!r}, got {row['model_key']!r}"
        )

    if expected_model_spec is not None:
        exp_repo = expected_model_spec.get("repo_id")
        if exp_repo is not None and row["model_repo_id"] != exp_repo:
            raise ValueError(
                f"Call {cid!r} model_repo_id mismatch: expected {exp_repo!r}, got {row['model_repo_id']!r}"
            )
        exp_rev = expected_model_spec.get("revision")
        got_rev = row.get("model_revision")
        if exp_rev is not None or got_rev is not None:
            if got_rev != exp_rev:
                raise ValueError(
                    f"Call {cid!r} model_revision mismatch: expected {exp_rev!r}, got {got_rev!r}"
                )

    got_proto = row["coordinate_protocol"]
    if got_proto != expected_protocol:
        raise ValueError(
            f"Call {cid!r} coordinate_protocol mismatch: expected {expected_protocol!r}, got {got_proto!r}"
        )

    got_prompt_sha = row.get("prompt_sha256")
    if not _is_valid_sha256_hex(got_prompt_sha):
        raise ValueError(f"Call {cid!r} missing or malformed prompt_sha256: {got_prompt_sha!r}")
    if expected_prompt_sha256 is not None and got_prompt_sha.lower() != expected_prompt_sha256.lower():
        raise ValueError(
            f"Call {cid!r} prompt_sha256 mismatch: expected {expected_prompt_sha256}, got {got_prompt_sha!r}"
        )

    exp_sha = row.get("expected_sha256")
    inp_sha = row.get("input_sha256")
    if not _is_valid_sha256_hex(exp_sha):
        raise ValueError(f"Call {cid!r} missing or malformed expected_sha256: {exp_sha!r}")

    is_explicit_pre_input_failure = (
        row.get("ok") is False
        and ("input_sha256" in row)
        and (inp_sha is None)
        and bool(row.get("is_timeout") or row.get("is_oom") or row.get("watchdog_killed"))
        and (
            row.get("input_verification_status") == "unverified_pre_input_failure"
            or row.get("source") == "supervisor_watchdog"
        )
    )
    if not is_explicit_pre_input_failure:
        if not _is_valid_sha256_hex(inp_sha):
            raise ValueError(f"Call {cid!r} missing or malformed input_sha256: {inp_sha!r}")
        if exp_sha.lower() != inp_sha.lower():
            raise ValueError(f"Call {cid!r} input_sha256 != expected_sha256 ({inp_sha} vs {exp_sha})")
    elif row.get("input_verification_status") not in (None, "unverified_pre_input_failure"):
        raise ValueError(
            f"Call {cid!r} claims input_verification_status={row.get('input_verification_status')!r} with null input_sha256"
        )

    for dim_key in ("orig_input_width", "orig_input_height", "max_new_tokens"):
        if not _is_int_not_bool(row.get(dim_key)) or int(row[dim_key]) <= 0:
            raise ValueError(f"Call {cid!r} invalid positive integer field {dim_key}={row.get(dim_key)!r}")
    for off_key in ("roi_offset_x", "roi_offset_y", "tile_offset_x", "tile_offset_y"):
        if not _is_int_not_bool(row.get(off_key)) or int(row[off_key]) < 0:
            raise ValueError(f"Call {cid!r} invalid non-negative integer field {off_key}={row.get(off_key)!r}")

    if expected_max_new_tokens is not None and int(row["max_new_tokens"]) != int(expected_max_new_tokens):
        raise ValueError(
            f"Call {cid!r} max_new_tokens mismatch: expected {expected_max_new_tokens}, got {row['max_new_tokens']!r}"
        )

    if expected_call_spec is not None:
        if row["board_id"] != expected_call_spec["board_id"]:
            raise ValueError(
                f"Call {cid!r} board_id mismatch: expected {expected_call_spec['board_id']!r}, got {row['board_id']!r}"
            )
        if row["mode"] != expected_call_spec["mode"]:
            raise ValueError(
                f"Call {cid!r} mode mismatch: expected {expected_call_spec['mode']!r}, got {row['mode']!r}"
            )
        if "image_id" in expected_call_spec and row.get("image_id") != expected_call_spec["image_id"]:
            raise ValueError(
                f"Call {cid!r} image_id mismatch: expected {expected_call_spec['image_id']!r}, got {row.get('image_id')!r}"
            )
        if "tile_id" in expected_call_spec and row.get("tile_id") != expected_call_spec["tile_id"]:
            raise ValueError(
                f"Call {cid!r} tile_id mismatch: expected {expected_call_spec['tile_id']!r}, got {row.get('tile_id')!r}"
            )
        for row_k, spec_k in (
            ("orig_input_width", "width"),
            ("orig_input_height", "height"),
            ("roi_offset_x", "roi_offset_x"),
            ("roi_offset_y", "roi_offset_y"),
            ("tile_offset_x", "tile_offset_x"),
            ("tile_offset_y", "tile_offset_y"),
        ):
            if spec_k in expected_call_spec and int(row[row_k]) != int(expected_call_spec[spec_k]):
                raise ValueError(
                    f"Call {cid!r} {row_k} mismatch vs frozen spec: expected {expected_call_spec[spec_k]}, got {row[row_k]}"
                )
        frozen_sha = expected_call_spec.get("sha256")
        if not _is_valid_sha256_hex(frozen_sha):
            raise ValueError(f"Frozen call spec for {cid!r} missing valid sha256: {frozen_sha!r}")
        if exp_sha.lower() != frozen_sha.lower():
            raise ValueError(
                f"Call {cid!r} expected_sha256 mismatch vs frozen queue: frozen={frozen_sha}, expected={exp_sha}"
            )
        if not is_explicit_pre_input_failure and inp_sha.lower() != frozen_sha.lower():
            raise ValueError(
                f"Call {cid!r} SHA-256 mismatch vs frozen queue: frozen={frozen_sha}, expected={exp_sha}, input={inp_sha}"
            )

    if row.get("ok"):
        gen_ids = row.get("generated_token_ids")
        if gen_ids is None:
            gen_ids = row.get("generated_ids")
        if not isinstance(gen_ids, list) or not all(_is_int_not_bool(t) for t in gen_ids):
            raise ValueError(f"Call {cid!r} ok=True requires integer list generated_token_ids")
        gen_toks = row.get("generated_tokens")
        if not _is_int_not_bool(gen_toks) or gen_toks < 0:
            raise ValueError(f"Call {cid!r} ok=True requires valid generated_tokens, got {gen_toks!r}")
        if len(gen_ids) != int(gen_toks):
            raise ValueError(
                f"Call {cid!r} generated_token_ids length mismatch: len={len(gen_ids)} vs generated_tokens={gen_toks}"
            )
        if len(gen_ids) > int(row["max_new_tokens"]):
            raise ValueError(
                f"Call {cid!r} generated length {len(gen_ids)} exceeds max_new_tokens {row['max_new_tokens']}"
            )

        eos_ids = list(expected_eos_token_ids) if expected_eos_token_ids is not None else [151645, 151643]
        normal_eos = bool(generation_finished(gen_ids, eos_ids))
        finish_reason = row.get("finish_reason")
        if finish_reason is not None:
            if finish_reason == "eos_token":
                if not normal_eos:
                    raise ValueError(
                        f"Call {cid!r} declares finish_reason='eos_token' but trailing token {gen_ids[-1] if gen_ids else None} is not in EOS {eos_ids}"
                    )
            elif finish_reason == "length":
                if normal_eos:
                    raise ValueError(f"Call {cid!r} declares finish_reason='length' despite trailing EOS token")
                if len(gen_ids) < int(row["max_new_tokens"]):
                    raise ValueError(
                        f"Call {cid!r} declares finish_reason='length' at {len(gen_ids)} < max_new_tokens {row['max_new_tokens']}"
                    )
            else:
                raise ValueError(f"Call {cid!r} has invalid finish_reason={finish_reason!r}")

        # Verify recorded visual preprocessing tensor metadata against model visual budget when present
        pw = row.get("processed_width_px")
        ph = row.get("processed_height_px")
        pa = row.get("processed_area_px")
        if pw is not None and ph is not None and pa is not None:
            if not (_is_int_not_bool(pw) and _is_int_not_bool(ph) and _is_int_not_bool(pa)):
                raise ValueError(f"Call {cid!r} has non-integer processed dimensions")
            if pw * ph != pa:
                raise ValueError(f"Call {cid!r} processed_width_px*height != area ({pw}*{ph} vs {pa})")
        if expected_model_spec and isinstance(expected_model_spec.get("visual_budget"), dict):
            vb = expected_model_spec["visual_budget"]
            if "native_square_size" in vb and pw is not None and ph is not None:
                ns = int(vb["native_square_size"])
                if pw != ns or ph != ns:
                    raise ValueError(f"Call {cid!r} processed size {(pw, ph)} != native_square_size {ns}")
                if row.get("merged_visual_tokens") is not None and int(row["merged_visual_tokens"]) != 576:
                    raise ValueError(
                        f"Call {cid!r} merged_visual_tokens={row['merged_visual_tokens']} != 576 for native 384x384 SigLIP"
                    )
            lo = vb.get("min_pixels", vb.get("shortest_edge"))
            hi = vb.get("max_pixels", vb.get("longest_edge"))
            if lo is not None and hi is not None and pa is not None:
                if not (int(lo) <= pa <= int(hi)):
                    raise ValueError(
                        f"Call {cid!r} processed_area_px={pa} outside frozen visual budget [{lo}, {hi}]"
                    )
            grid = row.get("image_grid_thw")
            if grid is not None and row.get("merged_visual_tokens") is not None:
                if not (isinstance(grid, list) and len(grid) == 3 and grid[0] == 1):
                    raise ValueError(f"Call {cid!r} invalid image_grid_thw={grid!r}")
                if (int(grid[1]) * int(grid[2])) // 4 != int(row["merged_visual_tokens"]):
                    raise ValueError(
                        f"Call {cid!r} image_grid_thw={grid} inconsistent with merged_visual_tokens={row['merged_visual_tokens']}"
                    )

    if verify_disk_hashes:
        img_p = resolve_portable_image_path(
            row,
            expected_call_spec=expected_call_spec,
            path_remaps=path_remaps,
            image_hash_index=image_hash_index,
        )
        if img_p is None or not img_p.is_file():
            raise FileNotFoundError(
                f"Call {cid!r} image file not found for --verify-disk-hashes (tried {row.get('image_path')!r}; pass --path-map or --image-root if relocated)"
            )
        disk_sha = file_sha256(img_p)
        if disk_sha.lower() != exp_sha.lower():
            raise ValueError(f"Call {cid!r} on-disk image SHA-256 mismatch ({disk_sha} vs {exp_sha})")
        with Image.open(img_p) as im:
            disk_w, disk_h = im.size
        if (disk_w, disk_h) != (int(row["orig_input_width"]), int(row["orig_input_height"])):
            raise ValueError(
                f"Call {cid!r} on-disk image dimensions {(disk_w, disk_h)} != record {(row['orig_input_width'], row['orig_input_height'])}"
            )
        return True

    return False


def validate_coco_unique_image_ids(coco_gt: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Enforce globally unique image_ids and positive dimensions across multi-board COCO ground truth."""
    images = coco_gt.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("COCO ground truth missing non-empty 'images' list")
    seen_ids: Dict[str, Dict[str, Any]] = {}
    seen_boards: Dict[str, str] = {}
    by_board: Dict[str, Dict[str, Any]] = {}
    for img in images:
        raw_id = img.get("id", img.get("image_id"))
        if raw_id is None or isinstance(raw_id, bool):
            raise ValueError(f"COCO image entry missing valid 'id': {img}")
        w, h = img.get("width"), img.get("height")
        if not (_is_int_not_bool(w) and _is_int_not_bool(h) and w > 0 and h > 0):
            raise ValueError(f"COCO image {raw_id!r} missing valid positive width/height: {img}")
        sid = str(raw_id)
        if sid in seen_ids:
            raise ValueError(
                f"Duplicate COCO image_id {raw_id!r} colliding between {seen_ids[sid].get('board_id') or seen_ids[sid].get('file_name')} "
                f"and {img.get('board_id') or img.get('file_name')}"
            )
        seen_ids[sid] = img
        bkey = str(img.get("board_id") or img.get("sample_id") or Path(str(img.get("file_name", sid))).stem)
        if bkey in seen_boards and seen_boards[bkey] != sid:
            raise ValueError(f"Board {bkey!r} mapped to multiple image_ids: {seen_boards[bkey]!r}, {sid!r}")
        seen_boards[bkey] = sid
        by_board[bkey] = img
    return by_board


def validate_coco_schema_and_categories(
    coco_gt: Dict[str, Any],
    *,
    expected_classes: Optional[Sequence[Dict[str, Any]]] = FICS_CANONICAL_CATEGORIES,
) -> Dict[str, Dict[str, Any]]:
    """Validate COCO categories, unique image IDs, and all annotation references."""
    by_board = validate_coco_unique_image_ids(coco_gt)
    coco_id_to_name, _ = _canonical_category_maps(coco_gt.get("categories"), context="Combined GT COCO")
    if expected_classes is not None:
        exp_id_to_name, _ = _canonical_category_maps(list(expected_classes), context="Expected FICS classes")
        if coco_id_to_name != exp_id_to_name:
            raise ValueError(
                f"GT COCO categories mismatch expected FICS 6-class taxonomy: {coco_id_to_name} vs {exp_id_to_name}"
            )

    valid_img_ids = {str(img.get("id", img.get("image_id"))) for img in coco_gt["images"]}
    valid_cat_ids = set(coco_id_to_name.keys())
    seen_ann_ids: set[int] = set()
    for ann in coco_gt.get("annotations") or []:
        ann_id = ann.get("id")
        if not _is_int_not_bool(ann_id):
            raise ValueError(f"GT COCO annotation missing integer 'id': {ann!r}")
        if ann_id in seen_ann_ids:
            raise ValueError(f"Duplicate GT COCO annotation id={ann_id}")
        seen_ann_ids.add(ann_id)
        if str(ann.get("image_id")) not in valid_img_ids:
            raise ValueError(f"GT COCO annotation id={ann_id} references unknown image_id={ann.get('image_id')!r}")
        cat_id = ann.get("category_id")
        if not _is_int_not_bool(cat_id) or cat_id not in valid_cat_ids:
            raise ValueError(f"GT COCO annotation id={ann_id} references unknown category_id={cat_id!r}")
    return by_board


def merge_board_coco_gts_unique_ids(
    board_cocos: Sequence[Tuple[str, Dict[str, Any], int]],
) -> Dict[str, Any]:
    """Merge per-board COCO dicts using strict identical category tables, unique image_ids, and disjoint annotation IDs."""
    if not board_cocos:
        raise ValueError("Cannot merge empty board COCO sequence")

    merged_images: List[Dict[str, Any]] = []
    merged_anns: List[Dict[str, Any]] = []
    reference_cats_list: Optional[List[Dict[str, Any]]] = None
    reference_id_to_name: Optional[Dict[int, str]] = None
    reference_name_to_id: Optional[Dict[str, int]] = None
    seen_new_ids: set[int] = set()
    seen_boards: set[str] = set()
    next_ann_id = 1

    for board_id, coco, unique_image_id in board_cocos:
        if board_id in seen_boards:
            raise ValueError(f"Duplicate board_id {board_id!r} in COCO merge")
        seen_boards.add(board_id)
        if not _is_int_not_bool(unique_image_id):
            raise ValueError(f"Invalid unique_image_id={unique_image_id!r} for board {board_id!r}")
        if unique_image_id in seen_new_ids:
            raise ValueError(f"Collision in assigned unique_image_id={unique_image_id} for board {board_id!r}")
        seen_new_ids.add(unique_image_id)

        id_to_name, name_to_id = _canonical_category_maps(
            coco.get("categories"), context=f"Board {board_id!r} COCO"
        )
        if reference_id_to_name is None:
            reference_id_to_name = id_to_name
            reference_name_to_id = name_to_id
            reference_cats_list = [dict(c) for c in coco["categories"]]
        else:
            if id_to_name != reference_id_to_name or name_to_id != reference_name_to_id:
                raise ValueError(
                    f"Incompatible COCO categories in board {board_id!r}: "
                    f"{id_to_name} vs reference {reference_id_to_name}"
                )

        imgs = coco.get("images") or []
        if len(imgs) != 1:
            raise ValueError(f"Expected single-image board COCO for {board_id!r}, got {len(imgs)} images")
        orig_id = imgs[0].get("id")
        new_img = dict(imgs[0])
        new_img["orig_image_id"] = orig_id
        new_img["id"] = unique_image_id
        new_img.setdefault("board_id", board_id)
        merged_images.append(new_img)

        seen_orig_ann_ids: set[Any] = set()
        for ann in coco.get("annotations") or []:
            ann_id = ann.get("id")
            if ann_id is None or isinstance(ann_id, bool):
                raise ValueError(f"Board {board_id!r} annotation missing valid 'id': {ann!r}")
            if ann_id in seen_orig_ann_ids:
                raise ValueError(f"Board {board_id!r} has duplicate annotation id={ann_id!r}")
            seen_orig_ann_ids.add(ann_id)
            if str(ann.get("image_id")) != str(orig_id):
                raise ValueError(
                    f"Board {board_id!r} annotation id={ann_id!r} references nonexistent image_id={ann.get('image_id')!r} (expected {orig_id!r})"
                )
            cat_id = ann.get("category_id")
            if not _is_int_not_bool(cat_id) or cat_id not in id_to_name:
                raise ValueError(
                    f"Board {board_id!r} annotation id={ann_id!r} references unknown category_id={cat_id!r}"
                )
            new_ann = dict(ann)
            new_ann["image_id"] = unique_image_id
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            merged_anns.append(new_ann)

    return {
        "images": merged_images,
        "categories": reference_cats_list or [],
        "annotations": merged_anns,
    }


def build_expected_spec_from_plan(
    plan: Dict[str, Any],
    *,
    models_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build complete per-model and per-call expected specs from `frozen_plan.json` or a replay manifest."""
    if not isinstance(plan, dict):
        raise ValueError("Expected plan/manifest dict")

    models_map = plan.get("models")
    if not isinstance(models_map, dict) or not models_map:
        raise ValueError("Plan missing non-empty 'models' mapping")

    planned_order = list(plan.get("model_execution_order") or models_map.keys())
    if not planned_order:
        raise ValueError("Plan missing 'model_execution_order' and 'models'")
    for mk in planned_order:
        if mk not in models_map:
            raise ValueError(f"Planned model {mk!r} missing from plan['models']")

    selected_models = list(models_filter) if models_filter is not None else list(planned_order)
    unknown_selected = [mk for mk in selected_models if mk not in models_map]
    if unknown_selected:
        raise ValueError(f"Selected models {unknown_selected} do not exist in plan models {planned_order}")

    inf_cfg = plan.get("inference_config")
    if not isinstance(inf_cfg, dict):
        raise ValueError("Plan missing 'inference_config'")
    plan_prompt = inf_cfg.get("prompt")
    plan_prompt_sha = inf_cfg.get("prompt_sha256")
    if not _is_valid_sha256_hex(plan_prompt_sha):
        raise ValueError(f"Plan missing valid inference_config.prompt_sha256: {plan_prompt_sha!r}")
    if plan_prompt is not None:
        if not isinstance(plan_prompt, str) or sha256_text(plan_prompt) != plan_prompt_sha.lower():
            raise ValueError("Plan inference_config.prompt SHA-256 does not match inference_config.prompt_sha256")
    coord_proto = inf_cfg.get("coordinate_protocol")
    if coord_proto not in PROTOCOLS:
        raise ValueError(f"Plan has unsupported coordinate_protocol={coord_proto!r}")
    max_new_tokens = inf_cfg.get("max_new_tokens")
    if not _is_int_not_bool(max_new_tokens) or max_new_tokens <= 0:
        raise ValueError(f"Plan has invalid inference_config.max_new_tokens={max_new_tokens!r}")

    queues = plan.get("queues") or {}
    call_specs_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ordered_ids_by_model: Dict[str, List[str]] = {}
    if "formal_67" in queues and isinstance(queues["formal_67"], list):
        base_queue = queues["formal_67"]
        if not base_queue:
            raise ValueError("Plan queues.formal_67 is empty")
        for mk in planned_order:
            ordered_ids: List[str] = []
            spec_map: Dict[str, Dict[str, Any]] = {}
            for q in base_queue:
                if not isinstance(q, dict) or "board_id" not in q or "tag" not in q:
                    raise ValueError(f"Invalid queue entry in formal_67: {q!r}")
                cid = q.get("call_id") or f"{mk}_formal_{q['board_id']}_{q['tag']}"
                if cid in spec_map:
                    raise ValueError(f"Duplicate planned call_id {cid!r} in formal_67")
                spec = dict(q)
                spec["call_id"] = cid
                spec_map[cid] = spec
                ordered_ids.append(cid)
            ordered_ids_by_model[mk] = ordered_ids
            call_specs_by_model[mk] = spec_map
    elif isinstance(queues, dict) and queues:
        for mk in planned_order:
            qlist = queues.get(mk)
            if not isinstance(qlist, list) or not qlist:
                raise ValueError(f"Plan missing queue list for planned model {mk!r}")
            ordered_ids = []
            spec_map = {}
            for q in qlist:
                if not isinstance(q, dict) or "call_id" not in q:
                    raise ValueError(f"Queue entry for model {mk!r} must be a full call spec dict with 'call_id': {q!r}")
                cid = str(q["call_id"])
                if cid in spec_map:
                    raise ValueError(f"Duplicate planned call_id {cid!r} for model {mk!r}")
                spec_map[cid] = dict(q)
                ordered_ids.append(cid)
            ordered_ids_by_model[mk] = ordered_ids
            call_specs_by_model[mk] = spec_map
    else:
        raise ValueError("Plan missing non-empty 'queues'")

    gt_identity = plan.get("gt_identity") if isinstance(plan.get("gt_identity"), dict) else {}
    expected_gt_sha = gt_identity.get("combined_coco_sha256") or plan.get("combined_coco_sha256")
    if expected_gt_sha is not None and not _is_valid_sha256_hex(expected_gt_sha):
        raise ValueError(f"Plan/manifest has malformed combined_coco_sha256: {expected_gt_sha!r}")

    return {
        "planned_models": planned_order,
        "selected_models": selected_models,
        "models": models_map,
        "prompt_sha256": plan_prompt_sha.lower(),
        "prompt": plan_prompt,
        "coordinate_protocol": coord_proto,
        "max_new_tokens": int(max_new_tokens),
        "boards": plan.get("boards") or {},
        "expected_call_ids_by_model": ordered_ids_by_model,
        "expected_call_specs_by_model": call_specs_by_model,
        "expected_gt_sha256": expected_gt_sha.lower() if expected_gt_sha else None,
    }


def classify_call_termination_and_parse_v2(
    *,
    raw_text: str,
    width: int,
    height: int,
    generated_tokens: Optional[int] = None,
    max_new_tokens: int = 512,
    generated_token_ids: Optional[Sequence[int]] = None,
    eos_token_ids: Sequence[int] = (151645, 151643),
    protocol: str = "xyxy_norm1000",
    allowed_classes: Sequence[str] = FICS_SIX_CLASSES,
) -> Dict[str, Any]:
    """Parse one raw VLM output using v2 termination semantics (EOS vs length limit orthogonal to JSON/schema)."""
    cleaned_text, had_complete_fence = strip_eos_and_outer_fence(raw_text)
    strict = validate_strict_json_contract(cleaned_text, allowed_classes=allowed_classes, protocol=protocol)

    normal_eos = bool(generation_finished(list(generated_token_ids) if generated_token_ids is not None else [], list(eos_token_ids)))
    gen_len = int(generated_tokens) if generated_tokens is not None else (len(generated_token_ids) if generated_token_ids is not None else 0)
    length_limited_without_eos = bool((not normal_eos) and gen_len >= int(max_new_tokens))

    parsed = parse_response(
        cleaned_text,
        width=width,
        height=height,
        protocol=protocol,
        allow_qwen_tokens=False,
    )

    allowed_set = {c.strip().lower() for c in allowed_classes}
    fics_objects: List[Dict[str, Any]] = []
    non_fics_objects: List[Dict[str, Any]] = []
    for obj in parsed["objects"]:
        lbl = str(obj["label"]).strip().lower()
        if lbl in allowed_set:
            fics_objects.append({
                "label": lbl,
                "bbox_xyxy": [round(float(v), 2) for v in obj["bbox_xyxy"]],
            })
        else:
            non_fics_objects.append(obj)

    syntax_or_format_incomplete_after_eos = bool(
        normal_eos and (not strict["complete_json_syntax_ok"] or not strict["strict_schema_ok"])
    )
    eos_incomplete_reason: Optional[str] = None
    if syntax_or_format_incomplete_after_eos:
        if not cleaned_text:
            eos_incomplete_reason = "empty_text_after_eos"
        elif _REFUSAL_RE.search(cleaned_text):
            eos_incomplete_reason = "natural_language_refusal_after_eos"
        elif strict["complete_json_syntax_ok"] and not strict["strict_schema_ok"]:
            eos_incomplete_reason = "complete_json_schema_violation_after_eos"
        elif "[" in cleaned_text or "{" in cleaned_text:
            eos_incomplete_reason = "malformed_json_after_eos"
        else:
            eos_incomplete_reason = "natural_language_or_non_json_after_eos"

    # Legacy v1 metric retained only for auditability against earlier reports
    unclosed_brace = cleaned_text.count("{") > cleaned_text.count("}")
    unclosed_fence = cleaned_text.lstrip().startswith("```")
    has_json_open = "[" in cleaned_text
    legacy_is_truncated_v1 = bool(
        parsed["truncated"]
        or unclosed_brace
        or unclosed_fence
        or (has_json_open and not strict["complete_json_syntax_ok"])
        or length_limited_without_eos
    )

    parse_status = parsed["status"]
    if length_limited_without_eos:
        if fics_objects:
            parse_status = "partial_success"
        elif parse_status == "no_objects_detected":
            parse_status = "parse_failed"
    elif (not strict["strict_schema_ok"]) and fics_objects:
        parse_status = "partial_success"
    elif non_fics_objects and not fics_objects and parse_status in ("success", "partial_success"):
        parse_status = "parse_filtered"

    # Explicit empty ONLY when cleaned output is valid literal empty JSON list []
    explicit_empty = bool(
        strict["complete_json_syntax_ok"]
        and strict["strict_schema_ok"]
        and strict.get("parsed_array_len") == 0
        and not length_limited_without_eos
    )
    partial_recovery = bool(not strict["strict_schema_ok"] and len(fics_objects) > 0)
    parse_failed = bool(parse_status in ("parse_failed", "parse_filtered") and len(fics_objects) == 0)

    if strict["strict_schema_ok"] and len(fics_objects) > 0 and not length_limited_without_eos:
        bucket = "strict_json_with_objects"
    elif explicit_empty:
        bucket = "explicit_no_objects"
    elif partial_recovery or (length_limited_without_eos and len(fics_objects) > 0):
        bucket = "compatible_non_json_or_partial_with_objects"
    else:
        bucket = "parse_failed_or_filtered_zero_objects"

    return {
        "objects": fics_objects,
        "non_fics_objects_filtered": non_fics_objects,
        "status": parse_status,
        "call_bucket": bucket,
        "normal_eos": normal_eos,
        "length_limited_without_eos": length_limited_without_eos,
        "is_truncated": length_limited_without_eos,  # v2 definition: true token-budget truncation only
        "complete_json_syntax": bool(strict["complete_json_syntax_ok"]),
        "complete_json_syntax_surface": bool(strict["complete_json_syntax_ok"]),
        "strict_schema_ok": bool(strict["strict_schema_ok"]),
        "syntax_or_format_incomplete_after_eos": syntax_or_format_incomplete_after_eos,
        "eos_incomplete_reason": eos_incomplete_reason,
        "partial_recovery": partial_recovery,
        "parse_failed": parse_failed,
        "explicit_empty": explicit_empty,
        "legacy_is_truncated_v1": legacy_is_truncated_v1,
        "had_complete_fence": had_complete_fence,
        "audit": parsed["audit"],
    }


def _build_board_unit_result(
    *,
    board_id: str,
    board_coco: Dict[str, Any],
    unit_records: List[Dict[str, Any]],
    expected_call_ids: Optional[Sequence[str]],
    mode: str,
    nms_iou_threshold: float,
    eval_iou_threshold: float,
    run_coco_ap: bool,
) -> Dict[str, Any]:
    """Evaluate a single `(board_id, mode)` unit; refuse silent 0-score on incomplete failures."""
    if expected_call_ids is not None:
        by_cid = {r["call_id"]: r for r in unit_records}
        missing = [cid for cid in expected_call_ids if cid not in by_cid]
        ordered = [by_cid[cid] for cid in expected_call_ids if cid in by_cid]
    else:
        missing = []
        ordered = list(unit_records)

    failed_rows = [
        r
        for r in ordered
        if (not r.get("ok")) or r.get("is_oom") or r.get("is_timeout") or r.get("watchdog_killed")
    ]
    failed = [r["call_id"] for r in failed_rows]
    if missing or failed or not ordered:
        return {
            "evaluation_status": "incomplete_technical_failure",
            "technical_complete": False,
            "missing_call_ids": missing,
            "failed_call_ids": failed,
            "failed_call_details": [
                {
                    "call_id": r["call_id"],
                    "source": r.get("source"),
                    "is_timeout": bool(r.get("is_timeout")),
                    "is_oom": bool(r.get("is_oom")),
                    "watchdog_killed": bool(r.get("watchdog_killed")),
                    "last_worker_stage": r.get("last_worker_stage"),
                    "input_verification_status": r.get("input_verification_status")
                    or ("verified_sha256" if _is_valid_sha256_hex(r.get("input_sha256")) else "unknown"),
                    "input_sha256_source": r.get("input_sha256_source"),
                    "error": r.get("error"),
                }
                for r in failed_rows
            ],
            "completed_calls": len(ordered) - len(failed),
            "planned_calls": len(expected_call_ids) if expected_call_ids is not None else len(ordered),
            "note": "Incomplete or failed run is excluded from completed-model zero-score claims.",
        }

    img_meta = board_coco["images"][0]
    orig_w = int(img_meta["width"])
    orig_h = int(img_meta["height"])
    coco_img_id = img_meta["id"]

    raw_global_preds: List[Dict[str, Any]] = []
    for rec in ordered:
        for obj in rec["v2"]["objects"]:
            gbox = map_tile_box_to_global(
                obj["bbox_xyxy"],
                offset_x=int(rec["tile_offset_x"]) + int(rec["roi_offset_x"]),
                offset_y=int(rec["tile_offset_y"]) + int(rec["roi_offset_y"]),
                img_w=orig_w,
                img_h=orig_h,
            )
            raw_global_preds.append({
                "board_id": board_id,
                "image_id": coco_img_id,
                "label": obj["label"],
                "bbox_xyxy": gbox,
                "score": 1.0,
            })

    # Enforce strict intra-board NMS (never across boards)
    assert len({p["board_id"] for p in raw_global_preds}) <= 1
    after_nms = nms_merge_predictions(raw_global_preds, iou_threshold=nms_iou_threshold)

    any_len_trunc = any(rec["v2"]["length_limited_without_eos"] for rec in ordered)
    if mode == "W":
        board_status = ordered[0]["v2"]["status"]
    elif after_nms:
        board_status = "partial_success" if any_len_trunc else "success"
    elif any(rec["v2"]["status"] == "no_objects_detected" for rec in ordered):
        board_status = "no_objects_detected"
    else:
        board_status = "parse_failed"

    pred_record = {
        "image_id": coco_img_id,
        "status": board_status,
        "is_truncated": any_len_trunc,
        "objects": [
            {"label": p["label"], "bbox_xyxy": p["bbox_xyxy"], "score": 1.0}
            for p in after_nms
        ],
    }
    pred_payload = {"predictions": [pred_record]}
    std_eval = repo_eval.evaluate_detections(board_coco, pred_payload, iou_threshold=eval_iou_threshold)
    out: Dict[str, Any] = {
        "evaluation_status": "completed",
        "technical_complete": True,
        "planned_calls": len(ordered),
        "completed_calls": len(ordered),
        "raw_global_predictions": len(raw_global_preds),
        "post_nms_predictions": len(after_nms),
        "pred_record": pred_record,
        "evaluate_detections_iou50": std_eval,
    }
    if run_coco_ap:
        out["standard_benchmark"] = repo_eval.evaluate_standard_benchmark(board_coco, pred_payload)
    return out


def _aggregate_termination_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    buckets = collections.Counter()
    eos_reasons = collections.Counter()
    ok_recs = [r for r in records if r.get("ok") and "v2" in r]
    for r in ok_recs:
        v2 = r["v2"]
        buckets[v2["call_bucket"]] += 1
        if v2["eos_incomplete_reason"]:
            eos_reasons[v2["eos_incomplete_reason"]] += 1
    return {
        "total_calls_observed": len(records),
        "completed_ok_calls": len(ok_recs),
        "failed_calls": sum(1 for r in records if not r.get("ok")),
        "normal_eos": sum(1 for r in ok_recs if r["v2"]["normal_eos"]),
        "length_limited_without_eos": sum(1 for r in ok_recs if r["v2"]["length_limited_without_eos"]),
        "complete_json_syntax": sum(1 for r in ok_recs if r["v2"]["complete_json_syntax"]),
        "strict_schema_ok": sum(1 for r in ok_recs if r["v2"]["strict_schema_ok"]),
        "syntax_or_format_incomplete_after_eos": sum(1 for r in ok_recs if r["v2"]["syntax_or_format_incomplete_after_eos"]),
        "partial_recovery": sum(1 for r in ok_recs if r["v2"]["partial_recovery"]),
        "parse_failed": sum(1 for r in ok_recs if r["v2"]["parse_failed"]),
        "explicit_empty": sum(1 for r in ok_recs if r["v2"]["explicit_empty"]),
        "legacy_is_truncated_v1": sum(1 for r in ok_recs if r["v2"]["legacy_is_truncated_v1"]),
        "call_buckets": dict(buckets),
        "eos_incomplete_reasons": dict(eos_reasons),
        "length_limited_without_eos_call_ids": sorted(
            r["call_id"] for r in ok_recs if r["v2"]["length_limited_without_eos"]
        ),
    }


def evaluate_fics_replay(
    *,
    combined_coco: Dict[str, Any],
    call_rows_by_model: Dict[str, List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]] = None,
    frozen_plan: Optional[Dict[str, Any]] = None,
    expected_prompt_sha256: Optional[str] = None,
    expected_gt_sha256: Optional[str] = None,
    actual_gt_sha256: Optional[str] = None,
    verify_disk_hashes: bool = False,
    planned_call_ids_by_model: Optional[Dict[str, Sequence[str]]] = None,
    selected_models: Optional[Sequence[str]] = None,
    diagnostic_mode: bool = False,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_hash_index: Optional[Dict[str, Path]] = None,
    run_coco_ap: bool = True,
) -> Dict[str, Any]:
    """Replay FICS raw model outputs end-to-end on CPU with full provenance, schema, and completeness checks."""
    cfg = config or load_fics_six_class_config()
    proto = str(cfg["protocol"]["coordinate_protocol"])
    nms_iou = float(cfg["protocol"]["nms_iou_threshold"])
    eval_iou = float(cfg["protocol"]["eval_iou_threshold"])
    max_new_tokens_cfg = int(cfg["protocol"]["max_new_tokens_practical"])
    eos_map = cfg.get("eos_token_ids_by_model", {})
    cfg_prompt_sha = str(cfg["protocol"]["prompt_sha256"]).lower()
    allowed_classes = tuple(str(c["name"]).strip().lower() for c in cfg["classes"])

    plan_spec = (
        build_expected_spec_from_plan(frozen_plan, models_filter=selected_models)
        if frozen_plan is not None
        else None
    )

    if expected_prompt_sha256 is not None:
        if not _is_valid_sha256_hex(expected_prompt_sha256):
            raise ValueError(f"Malformed expected_prompt_sha256: {expected_prompt_sha256!r}")
        expected_prompt_sha_norm = expected_prompt_sha256.lower()
        if expected_prompt_sha_norm != cfg_prompt_sha:
            raise ValueError(
                f"Expected prompt_sha256 {expected_prompt_sha_norm} disagrees with verified config prompt_sha256 {cfg_prompt_sha}"
            )
    else:
        expected_prompt_sha_norm = cfg_prompt_sha

    if plan_spec is not None:
        if plan_spec["prompt_sha256"] != expected_prompt_sha_norm:
            raise ValueError(
                f"Plan prompt_sha256 {plan_spec['prompt_sha256']} disagrees with config prompt_sha256 {expected_prompt_sha_norm}"
            )
        if plan_spec["coordinate_protocol"] != proto:
            raise ValueError(
                f"Plan coordinate_protocol {plan_spec['coordinate_protocol']!r} disagrees with config {proto!r}"
            )
        if plan_spec["max_new_tokens"] != max_new_tokens_cfg:
            raise ValueError(
                f"Plan max_new_tokens {plan_spec['max_new_tokens']} disagrees with config {max_new_tokens_cfg}"
            )

    trusted_gt_sha = expected_gt_sha256.lower() if expected_gt_sha256 else (plan_spec["expected_gt_sha256"] if plan_spec else None)
    if trusted_gt_sha is not None:
        if not _is_valid_sha256_hex(trusted_gt_sha):
            raise ValueError(f"Malformed expected_gt_sha256: {trusted_gt_sha!r}")
        if actual_gt_sha256 is None:
            raise ValueError("expected_gt_sha256 provided without actual_gt_sha256 of loaded GT COCO file")
        if actual_gt_sha256.lower() != trusted_gt_sha:
            raise ValueError(
                f"GT COCO SHA-256 mismatch: actual {actual_gt_sha256} != expected {trusted_gt_sha}"
            )

    if not diagnostic_mode:
        if plan_spec is None:
            raise ValueError("Formal replay requires frozen_plan / manifest (pass diagnostic_mode=True or --diagnostic for unplanned runs)")
        if trusted_gt_sha is None:
            raise ValueError("Formal replay requires trusted expected_gt_sha256 (or pass diagnostic_mode=True / --diagnostic)")
        if selected_models is not None and list(selected_models) != list(plan_spec["planned_models"]):
            raise ValueError(
                f"Narrowing models to {list(selected_models)} violates formal plan scope {plan_spec['planned_models']} (requires --diagnostic)"
            )

    board_img_map = validate_coco_schema_and_categories(combined_coco, expected_classes=cfg["classes"])
    board_cocos: Dict[str, Dict[str, Any]] = {}
    for bkey, img in board_img_map.items():
        iid = img["id"]
        anns = [a for a in combined_coco.get("annotations", []) if str(a["image_id"]) == str(iid)]
        if plan_spec and bkey in plan_spec["boards"]:
            pb = plan_spec["boards"][bkey]
            if str(pb.get("image_id")) != str(iid):
                raise ValueError(f"GT COCO board {bkey!r} image_id={iid!r} != frozen plan {pb.get('image_id')!r}")
            if pb.get("orig_wh") and [img["width"], img["height"]] != list(pb["orig_wh"]):
                raise ValueError(f"GT COCO board {bkey!r} size {[img['width'], img['height']]} != frozen plan {pb.get('orig_wh')}")
            if pb.get("gt_count") is not None and len(anns) != int(pb["gt_count"]):
                raise ValueError(f"GT COCO board {bkey!r} gt_count={len(anns)} != frozen plan {pb['gt_count']}")
        board_cocos[bkey] = {
            "images": [img],
            "categories": list(combined_coco["categories"]),
            "annotations": anns,
        }

    # Determine expected models & expected call specs/IDs
    if plan_spec is not None:
        planned_models = list(plan_spec["planned_models"])
        active_models = list(plan_spec["selected_models"])
        if planned_call_ids_by_model is not None:
            # Ensure caller-supplied ids agree with frozen plan if both provided
            for mk in active_models:
                if mk in planned_call_ids_by_model and list(planned_call_ids_by_model[mk]) != plan_spec["expected_call_ids_by_model"][mk]:
                    raise ValueError(f"Caller planned_call_ids_by_model[{mk!r}] conflicts with frozen plan queue")
    elif planned_call_ids_by_model is not None:
        planned_models = list(planned_call_ids_by_model.keys())
        active_models = list(selected_models) if selected_models is not None else list(planned_models)
    else:
        planned_models = list(call_rows_by_model.keys())
        active_models = list(selected_models) if selected_models is not None else list(call_rows_by_model.keys())

    # Reject unexpected model sources outside scope/plan
    expected_model_set = set(planned_models)
    if plan_spec is not None or planned_call_ids_by_model is not None:
        unexpected_models = [mk for mk in call_rows_by_model if mk not in expected_model_set]
        if unexpected_models:
            raise ValueError(f"Unexpected model inputs not in plan: {unexpected_models}")
        if selected_models is not None:
            unrequested = [mk for mk in call_rows_by_model if mk not in set(active_models)]
            if unrequested:
                raise ValueError(f"Model inputs {unrequested} outside requested scope {active_models}")

    models_out: Dict[str, Any] = {}
    disk_verified_count = 0
    missing_models = [mk for mk in planned_models if mk not in call_rows_by_model]

    for model_key in active_models:
        eos_ids = list(eos_map.get(model_key, [151645, 151643]))
        model_spec = plan_spec["models"][model_key] if plan_spec else None
        spec_by_cid = plan_spec["expected_call_specs_by_model"][model_key] if plan_spec else None
        if plan_spec is not None:
            planned_ids: Optional[List[str]] = list(plan_spec["expected_call_ids_by_model"][model_key])
        elif planned_call_ids_by_model and model_key in planned_call_ids_by_model:
            planned_ids = list(planned_call_ids_by_model[model_key])
        else:
            planned_ids = None

        if model_key not in call_rows_by_model:
            # Missing entire model directory/JSONL: record structured incomplete status (never omit or fill 0-score!)
            models_out[model_key] = {
                "model_technical_complete": False,
                "missing_entire_model_input": True,
                "termination_v2": _aggregate_termination_stats([]),
                "missing_call_ids": list(planned_ids) if planned_ids is not None else [],
                "per_board": {},
                "combined": {
                    mode: {
                        "evaluation_status": "incomplete_technical_failure",
                        "technical_complete": False,
                        "missing_planned_calls": list(planned_ids) if planned_ids is not None else [],
                        "failed_calls": [],
                        "note": f"Entire model input missing for planned model {model_key!r}; zero-score attribution forbidden.",
                    }
                    for mode in ("W", "T")
                },
            }
            continue

        rows = call_rows_by_model[model_key]
        seen_cids: set[str] = set()
        processed_rows: List[Dict[str, Any]] = []
        for row in rows:
            cid = row.get("call_id")
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError(f"Model {model_key!r} record missing valid 'call_id': {row!r}")
            if cid in seen_cids:
                raise ValueError(f"Duplicate call_id {cid!r} in model {model_key!r}")
            seen_cids.add(cid)

            if planned_ids is not None and cid not in set(planned_ids):
                raise ValueError(f"Unexpected call_id {cid!r} not in planned queue for model {model_key!r}")

            call_spec = spec_by_cid.get(cid) if spec_by_cid is not None else None
            verified_disk = validate_call_record_integrity(
                row,
                expected_model_key=model_key,
                expected_model_spec=model_spec,
                expected_call_spec=call_spec,
                expected_prompt_sha256=expected_prompt_sha_norm,
                expected_protocol=proto,
                expected_max_new_tokens=max_new_tokens_cfg if plan_spec is not None else None,
                expected_eos_token_ids=eos_ids,
                verify_disk_hashes=verify_disk_hashes,
                path_remaps=path_remaps,
                image_hash_index=image_hash_index,
            )
            if verified_disk:
                disk_verified_count += 1

            rec = dict(row)
            if rec.get("ok"):
                gen_ids = rec.get("generated_token_ids")
                if gen_ids is None:
                    gen_ids = rec.get("generated_ids")
                rec["v2"] = classify_call_termination_and_parse_v2(
                    raw_text=str(rec.get("raw_output") or ""),
                    width=int(rec["orig_input_width"]),
                    height=int(rec["orig_input_height"]),
                    generated_tokens=rec.get("generated_tokens"),
                    max_new_tokens=int(rec["max_new_tokens"]),
                    generated_token_ids=gen_ids,
                    eos_token_ids=eos_ids,
                    protocol=proto,
                    allowed_classes=allowed_classes,
                )
            processed_rows.append(rec)

        by_cid_map = {r["call_id"]: r for r in processed_rows}
        missing_planned = [cid for cid in planned_ids if cid not in by_cid_map] if planned_ids is not None else []
        failed_calls = [
            r["call_id"]
            for r in processed_rows
            if (not r.get("ok")) or r.get("is_oom") or r.get("is_timeout") or r.get("watchdog_killed")
        ]

        term_summary = _aggregate_termination_stats(processed_rows)
        per_board: Dict[str, Dict[str, Any]] = {}
        combined_modes: Dict[str, Any] = {}

        if spec_by_cid is not None:
            modes_present = sorted({str(spec["mode"]) for spec in spec_by_cid.values()})
        else:
            modes_present = sorted({str(r["mode"]) for r in processed_rows if r.get("mode")})
            if planned_ids:
                inferred_modes = set(modes_present)
                for cid in planned_ids:
                    if cid.endswith("_W"):
                        inferred_modes.add("W")
                    elif re.search(r"_T\d+$", cid):
                        inferred_modes.add("T")
                modes_present = sorted(inferred_modes) or ["W", "T"]

        for bkey, bcoco in board_cocos.items():
            per_board[bkey] = {}
            for mode in modes_present:
                sub = [r for r in processed_rows if r.get("board_id") == bkey and r.get("mode") == mode]
                if spec_by_cid is not None and planned_ids is not None:
                    unit_expected = [
                        cid
                        for cid in planned_ids
                        if spec_by_cid[cid]["board_id"] == bkey and spec_by_cid[cid]["mode"] == mode
                    ]
                elif planned_ids is not None:
                    unit_expected = [cid for cid in planned_ids if f"_{bkey}_{mode}" in cid]
                else:
                    unit_expected = None
                per_board[bkey][mode] = _build_board_unit_result(
                    board_id=bkey,
                    board_coco=bcoco,
                    unit_records=sub,
                    expected_call_ids=unit_expected,
                    mode=mode,
                    nms_iou_threshold=nms_iou,
                    eval_iou_threshold=eval_iou,
                    run_coco_ap=run_coco_ap,
                )

        all_units_complete = bool(
            board_cocos
            and modes_present
            and all(
                per_board[bkey][mode].get("technical_complete")
                for bkey in board_cocos
                for mode in modes_present
            )
        )
        model_complete = bool((not missing_planned) and (not failed_calls) and len(processed_rows) > 0 and all_units_complete)
        for mode in modes_present:
            unit_list = [per_board[bkey][mode] for bkey in board_cocos if mode in per_board[bkey]]
            if not model_complete or not unit_list or not all(u.get("technical_complete") for u in unit_list):
                combined_modes[mode] = {
                    "evaluation_status": "incomplete_technical_failure",
                    "technical_complete": False,
                    "missing_planned_calls": missing_planned,
                    "failed_calls": failed_calls,
                    "note": "Model incomplete across planned units; zero-score attribution forbidden.",
                }
                continue
            comb_preds = {"predictions": [u["pred_record"] for u in unit_list]}
            std = repo_eval.evaluate_detections(combined_coco, comb_preds, iou_threshold=eval_iou)
            comb_entry: Dict[str, Any] = {
                "evaluation_status": "completed",
                "technical_complete": True,
                "evaluate_detections_iou50": std,
            }
            if run_coco_ap:
                comb_entry["standard_benchmark"] = repo_eval.evaluate_standard_benchmark(combined_coco, comb_preds)
            combined_modes[mode] = comb_entry

        models_out[model_key] = {
            "model_technical_complete": model_complete,
            "missing_entire_model_input": False,
            "missing_call_ids": missing_planned,
            "failed_call_ids": failed_calls,
            "termination_v2": term_summary,
            "per_board": per_board,
            "combined": combined_modes,
        }

    plan_bound = plan_spec is not None
    is_full_planned_matrix = bool(plan_bound and (active_models == planned_models) and (not missing_models))
    all_selected_complete = bool(
        active_models
        and all(models_out.get(mk, {}).get("model_technical_complete") for mk in active_models)
    )
    gt_identity_verified = bool(trusted_gt_sha is not None and actual_gt_sha256 is not None and actual_gt_sha256.lower() == trusted_gt_sha)
    completeness_verified = bool(plan_bound and is_full_planned_matrix and all_selected_complete and gt_identity_verified)

    weight_provenance: Dict[str, Any] = {}
    if plan_spec is not None:
        for mk, mmeta in plan_spec["models"].items():
            weight_provenance[mk] = {
                "repo_id": mmeta.get("repo_id"),
                "revision": mmeta.get("revision"),
                "note": "Verified per-row against plan repo_id/revision metadata; raw JSONL records do not store model weight file SHA-256 digests.",
            }

    return {
        "schema_version": "fics_cpu_replay_v2",
        "replay_mode": "diagnostic" if diagnostic_mode else "formal",
        "unverified_diagnostic_mode": bool(diagnostic_mode or not (plan_bound and gt_identity_verified)),
        "completeness_verified": completeness_verified,
        "scoped_subset_complete": all_selected_complete,
        "gt_identity_verified": gt_identity_verified,
        "expected_gt_coco_sha256": trusted_gt_sha,
        "gt_coco_sha256": actual_gt_sha256,
        "disk_images_reverified": bool(verify_disk_hashes),
        "disk_images_verified_count": disk_verified_count if verify_disk_hashes else 0,
        "disk_verification_note": (
            f"Re-hashed and dimension-verified {disk_verified_count} on-disk input images via --verify-disk-hashes."
            if verify_disk_hashes
            else "On-disk input image bytes were NOT re-hashed (--verify-disk-hashes not set); verified frozen queue sha256 == row.expected_sha256 == row.input_sha256."
        ),
        "scope": {
            "planned_models": planned_models,
            "selected_models": active_models,
            "missing_models": missing_models,
            "is_full_planned_matrix": is_full_planned_matrix,
            "plan_bound": plan_bound,
            "unplanned_sources": not plan_bound,
        },
        "protocol_version": cfg.get("protocol_version"),
        "prompt_sha256": expected_prompt_sha_norm,
        "prompt_text_sha256_verified": True,
        "classes": list(allowed_classes),
        "weight_identity_provenance": weight_provenance,
        "coco_boards": {k: {"image_id": v["id"], "width": v["width"], "height": v["height"]} for k, v in board_img_map.items()},
        "models": models_out,
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def discover_raw_jsonl_sources(
    *,
    runs_dir: Optional[Path] = None,
    raw_jsonl_by_model: Optional[Dict[str, Path]] = None,
) -> Dict[str, Path]:
    """Discover per-model raw JSONL paths from explicit overrides or `runs_dir/run_<model>/raw_outputs_all.jsonl`."""
    sources: Dict[str, Path] = {}
    if raw_jsonl_by_model:
        for mk, p in raw_jsonl_by_model.items():
            sources[mk] = Path(p)
    elif runs_dir is not None:
        runs_root = Path(runs_dir)
        if not runs_root.is_dir():
            raise FileNotFoundError(f"runs_dir does not exist or is not a directory: {runs_root}")
        for sub in sorted(runs_root.iterdir()):
            if sub.is_dir() and sub.name.startswith("run_"):
                cand = sub / "raw_outputs_all.jsonl"
                if cand.is_file():
                    sources[sub.name[len("run_") :]] = cand
    if not sources:
        raise FileNotFoundError("No raw_outputs_all.jsonl inputs discovered or provided.")
    return sources


def replay_from_runs_dir(
    *,
    gt_coco_path: Path,
    runs_dir: Optional[Path] = None,
    raw_jsonl_by_model: Optional[Dict[str, Path]] = None,
    config_path: Optional[Path] = None,
    frozen_plan: Optional[Dict[str, Any]] = None,
    expected_prompt_sha256: Optional[str] = None,
    expected_gt_sha256: Optional[str] = None,
    verify_disk_hashes: bool = False,
    planned_call_ids_by_model: Optional[Dict[str, Sequence[str]]] = None,
    selected_models: Optional[Sequence[str]] = None,
    diagnostic_mode: bool = False,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_roots: Optional[Sequence[Path]] = None,
    run_coco_ap: bool = True,
) -> Dict[str, Any]:
    """Convenience entrypoint loading COCO JSON and per-model JSONL files from disk."""
    gt_path = Path(gt_coco_path)
    coco = json.loads(gt_path.read_text(encoding="utf-8"))
    actual_gt_sha = file_sha256(gt_path)
    cfg = load_fics_six_class_config(config_path)

    sources = discover_raw_jsonl_sources(runs_dir=runs_dir, raw_jsonl_by_model=raw_jsonl_by_model)
    call_rows_by_model = {mkey: _load_jsonl(p) for mkey, p in sources.items()}

    hash_index = build_image_hash_index(image_roots) if (verify_disk_hashes and image_roots) else None

    result = evaluate_fics_replay(
        combined_coco=coco,
        call_rows_by_model=call_rows_by_model,
        config=cfg,
        frozen_plan=frozen_plan,
        expected_prompt_sha256=expected_prompt_sha256,
        expected_gt_sha256=expected_gt_sha256,
        actual_gt_sha256=actual_gt_sha,
        verify_disk_hashes=verify_disk_hashes,
        planned_call_ids_by_model=planned_call_ids_by_model,
        selected_models=selected_models,
        diagnostic_mode=diagnostic_mode,
        path_remaps=path_remaps,
        image_hash_index=hash_index,
        run_coco_ap=run_coco_ap,
    )
    result["gt_coco_sha256"] = actual_gt_sha
    result["input_jsonl_sha256"] = {k: file_sha256(p) for k, p in sources.items()}
    return result
