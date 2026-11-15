"""In-repo CPU-only reproducible evaluation replay CLI for FICS-PCB DSLR VLM outputs.

Note:
    This entrypoint provides reproducible CPU parsing, strict manifest/provenance binding,
    v2 termination accounting, board-isolated NMS, anti-overwrite output checks, and
    COCO/standard metric replay. GPU inference remains in external runner scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.fics_replay import (  # noqa: E402
    DEFAULT_FICS_CONFIG_PATH,
    discover_raw_jsonl_sources,
    replay_from_runs_dir,
)


def parse_raw_jsonl_args(items: Optional[List[str]]) -> Dict[str, Path]:
    """Parse repeatable `MODEL=PATH` arguments, rejecting duplicates or empty entries."""
    out: Dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected MODEL=PATH for --raw-jsonl, got {item!r}")
        k, v = item.split("=", 1)
        mkey = k.strip()
        pstr = v.strip()
        if not mkey or not pstr:
            raise ValueError(f"Invalid empty model or path in --raw-jsonl: {item!r}")
        if mkey in out:
            raise ValueError(f"Duplicate --raw-jsonl model key {mkey!r}")
        out[mkey] = Path(pstr)
    return out


def parse_path_map_args(items: Optional[List[str]]) -> List[Tuple[str, str]]:
    """Parse repeatable `OLD_PREFIX=NEW_PREFIX` mappings for relocated input images."""
    remaps: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected OLD_PREFIX=NEW_PREFIX for --path-map, got {item!r}")
        old_p, new_p = item.split("=", 1)
        if not old_p.strip() or not new_p.strip():
            raise ValueError(f"Invalid empty prefix in --path-map: {item!r}")
        remaps.append((old_p.strip(), new_p.strip()))
    return remaps


def parse_selected_models(items: Optional[List[str]]) -> Optional[List[str]]:
    """Parse `--models` values (comma-separated or repeatable), rejecting empty/duplicate keys."""
    if not items:
        return None
    selected: List[str] = []
    seen: set[str] = set()
    for item in items:
        for part in item.split(","):
            mk = part.strip()
            if not mk:
                raise ValueError(f"Empty model name in --models argument: {item!r}")
            if mk in seen:
                raise ValueError(f"Duplicate model {mk!r} in --models argument")
            seen.add(mk)
            selected.append(mk)
    return selected


def _path_identities(path: Path) -> Tuple[set[str], Optional[Tuple[int, int]]]:
    """Return normalized candidate path strings and `(st_dev, st_ino)` if path exists."""
    norms = {
        os.path.normcase(os.path.abspath(str(path))),
        os.path.normcase(str(path.resolve(strict=False))),
        os.path.normcase(str(path.parent.resolve(strict=False) / path.name)),
    }
    dev_ino: Optional[Tuple[int, int]] = None
    try:
        if path.exists() or path.is_symlink():
            st = os.stat(path)
            if st.st_ino != 0:
                dev_ino = (int(st.st_dev), int(st.st_ino))
    except OSError:
        dev_ino = None
    return norms, dev_ino


def ensure_safe_output_path(out_path: Path, input_paths: Sequence[Path]) -> None:
    """Refuse writing --output over any input file (across case, junctions, hardlinks) or pre-existing file."""
    out_norms, out_dev_ino = _path_identities(out_path)
    for inp in input_paths:
        if inp is None:
            continue
        inp_norms, inp_dev_ino = _path_identities(inp)
        if out_norms & inp_norms:
            raise ValueError(f"Refusing to overwrite input path with --output: {out_path} conflicts with input {inp}")
        if out_dev_ino is not None and inp_dev_ino is not None and out_dev_ino == inp_dev_ino:
            raise ValueError(f"Refusing to overwrite input inode (hardlink/junction) with --output: {out_path} -> {inp}")
        try:
            if (out_path.exists() or out_path.is_symlink()) and (inp.exists() or inp.is_symlink()):
                if os.path.samefile(out_path, inp):
                    raise ValueError(f"Refusing to overwrite samefile input with --output: {out_path} == {inp}")
        except OSError:
            pass

    if out_path.exists() or out_path.is_symlink() or out_path.resolve(strict=False).exists():
        raise FileExistsError(
            f"Refusing to overwrite pre-existing output file (no silent overwrite in default mode): {out_path}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible CPU evaluation replay for FICS-PCB DSLR 6-class VLM outputs (v2 termination accounting)."
    )
    parser.add_argument("--gt-coco", required=True, help="Path to combined FICS COCO ground-truth JSON with unique image_ids.")
    parser.add_argument("--runs-dir", help="Directory containing run_<model>/raw_outputs_all.jsonl subdirectories.")
    parser.add_argument(
        "--raw-jsonl",
        action="append",
        help="Explicit MODEL=PATH pair (repeatable; overrides --runs-dir discovery; rejects duplicate MODEL keys).",
    )
    parser.add_argument("--config", help="Optional path to FICS 6-class YAML config (defaults to configs/fics_six_classes.yaml).")
    parser.add_argument("--plan", help="Path to frozen_plan.json containing models, inference_config, boards, and queues.")
    parser.add_argument(
        "--replay-manifest",
        help="Optional clean replay identity manifest JSON carrying expected GT SHA-256, frozen_plan SHA-256, or path remaps.",
    )
    parser.add_argument(
        "--expected-gt-sha256",
        help="Trusted expected SHA-256 of --gt-coco (required in formal mode unless supplied via --replay-manifest or plan).",
    )
    parser.add_argument("--expected-prompt-sha256", help="Expected prompt SHA-256 checksum (checked against YAML & plan).")
    parser.add_argument(
        "--models",
        action="append",
        help="Optional subset of models (requires --diagnostic if narrowing the frozen plan).",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Allow unplanned, unverified-GT, or single-model subset diagnostic replay (explicitly marked non-formal).",
    )
    parser.add_argument("--verify-disk-hashes", action="store_true", help="Re-hash and verify input image dimensions on disk.")
    parser.add_argument(
        "--path-map",
        action="append",
        help="Repeatable OLD_PREFIX=NEW_PREFIX path mapping for relocated input images.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        help="Repeatable directory of relocated input images indexed by SHA-256 for portable --verify-disk-hashes.",
    )
    parser.add_argument("--no-coco-ap", action="store_true", help="Skip pycocotools AP calculation (IoU P/R/F1 only).")
    parser.add_argument("--output", required=True, help="New non-existing output JSON report path.")
    args = parser.parse_args(argv)

    try:
        raw_map = parse_raw_jsonl_args(args.raw_jsonl)
        path_remaps = parse_path_map_args(args.path_map)
        selected_models = parse_selected_models(args.models)

        gt_path = Path(args.gt_coco)
        cfg_path = Path(args.config) if args.config else DEFAULT_FICS_CONFIG_PATH
        plan_path = Path(args.plan) if args.plan else None
        manifest_path = Path(args.replay_manifest) if args.replay_manifest else None
        out_path = Path(args.output)

        discovered_sources = discover_raw_jsonl_sources(
            runs_dir=Path(args.runs_dir) if args.runs_dir else None,
            raw_jsonl_by_model=raw_map or None,
        )

        input_paths: List[Path] = [gt_path, cfg_path]
        if args.runs_dir:
            input_paths.append(Path(args.runs_dir))
        if plan_path is not None:
            input_paths.append(plan_path)
        if manifest_path is not None:
            input_paths.append(manifest_path)
        input_paths.extend(discovered_sources.values())

        # Check anti-overwrite collisions & pre-existing --output BEFORE creating any dir or running heavy work
        ensure_safe_output_path(out_path, input_paths)

        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None
        plan_data = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path else None

        if manifest_data is not None:
            if not isinstance(manifest_data, dict):
                raise ValueError(f"Replay manifest must be a JSON object: {manifest_path}")
            src_plan_sha = manifest_data.get("source_frozen_plan_sha256")
            if src_plan_sha and plan_path is not None:
                actual_plan_sha = file_sha256(plan_path)
                if actual_plan_sha.lower() != str(src_plan_sha).lower():
                    raise ValueError(
                        f"Frozen plan SHA-256 mismatch vs manifest: {actual_plan_sha} != {src_plan_sha}"
                    )
            if plan_data is None and manifest_data.get("queues"):
                plan_data = manifest_data
            if manifest_data.get("path_remaps"):
                for item in manifest_data["path_remaps"]:
                    if isinstance(item, dict) and "from" in item and "to" in item:
                        path_remaps.append((str(item["from"]), str(item["to"])))
                    elif isinstance(item, list) and len(item) == 2:
                        path_remaps.append((str(item[0]), str(item[1])))

        expected_gt_sha = args.expected_gt_sha256
        if expected_gt_sha is None and manifest_data is not None:
            gt_id_block = manifest_data.get("gt_identity") if isinstance(manifest_data.get("gt_identity"), dict) else {}
            expected_gt_sha = gt_id_block.get("combined_coco_sha256") or manifest_data.get("combined_coco_sha256")

        expected_prompt_sha = args.expected_prompt_sha256
        if expected_prompt_sha is None and manifest_data is not None:
            expected_prompt_sha = manifest_data.get("expected_prompt_sha256")

        image_roots = [Path(p) for p in (args.image_root or [])]

        result = replay_from_runs_dir(
            gt_coco_path=gt_path,
            runs_dir=Path(args.runs_dir) if args.runs_dir else None,
            raw_jsonl_by_model=raw_map or None,
            config_path=cfg_path,
            frozen_plan=plan_data,
            expected_prompt_sha256=expected_prompt_sha,
            expected_gt_sha256=expected_gt_sha,
            verify_disk_hashes=args.verify_disk_hashes,
            selected_models=selected_models,
            diagnostic_mode=args.diagnostic,
            path_remaps=path_remaps or None,
            image_roots=image_roots or None,
            run_coco_ap=not args.no_coco_ap,
        )
        if manifest_path is not None:
            result["replay_manifest_sha256"] = file_sha256(manifest_path)

        payload_text = json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Re-verify output non-existence and open exclusively (`x`) after full serialized report is ready
    try:
        ensure_safe_output_path(out_path, input_paths)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("x", encoding="utf-8") as handle:
            handle.write(payload_text)
    except Exception as exc:
        print(f"ERROR writing output {out_path}: {exc}", file=sys.stderr)
        return 1

    if args.diagnostic:
        if not result["scoped_subset_complete"]:
            print(
                f"[INCOMPLETE DIAGNOSTIC] FICS diagnostic replay incomplete -> {out_path} "
                f"(scope={result['scope']['selected_models']}, missing_models={result['scope']['missing_models']})"
            )
            return 1
        print(
            f"FICS CPU diagnostic replay complete (NOT full-matrix formal verification; "
            f"scope={result['scope']['selected_models']}, gt_verified={result['gt_identity_verified']}) -> {out_path}"
        )
        return 0

    if not result["completeness_verified"]:
        print(
            f"[INCOMPLETE FORMAL] FICS formal replay incomplete -> {out_path} "
            f"(missing_models={result['scope']['missing_models']}, gt_verified={result['gt_identity_verified']})"
        )
        return 1

    print(f"FICS formal CPU replay complete -> {out_path}")
    for mkey, mv in result["models"].items():
        t = mv["termination_v2"]
        print(
            f"  [{mkey}] complete={mv['model_technical_complete']} calls={t['completed_ok_calls']}/{t['total_calls_observed']} "
            f"normal_eos={t['normal_eos']} len_limit={t['length_limited_without_eos']} eos_fmt_incomp={t['syntax_or_format_incomplete_after_eos']}"
        )
        for mode in ("W", "T"):
            if mode in mv["combined"] and mv["combined"][mode].get("technical_complete"):
                ev = mv["combined"][mode]["evaluate_detections_iou50"]
                print(
                    f"    combined {mode}: TP={ev['tp']} FP={ev['fp']} FN={ev['fn']} "
                    f"P={ev['micro']['precision']*100:.2f}% R={ev['micro']['recall']*100:.2f}% F1={ev['micro']['f1']*100:.2f}%"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
