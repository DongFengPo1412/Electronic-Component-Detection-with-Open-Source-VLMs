"""Unified repository CLI for sequential multi-model FICS-PCB VLM inference.

This script integrates the isolated worker + fail-fast GPU gate + watchdog
supervisor into the repository without hardcoded external cache paths.
By default `--dry-run` validates the frozen plan, runtime config, queue layout,
and output path safety without touching any GPU. Real GPU runs always execute
the dual VRAM gate before worker launch.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Optional, Tuple

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.inference.fics_matrix import run_sequential_matrix  # noqa: E402


def _parse_path_maps(items: Optional[List[str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected OLD_PREFIX=NEW_PREFIX for --path-map, got {item!r}")
        old_p, new_p = item.split("=", 1)
        if not old_p.strip() or not new_p.strip():
            raise ValueError(f"Invalid empty prefix in --path-map: {item!r}")
        out.append((old_p.strip(), new_p.strip()))
    return out


def _parse_models(items: Optional[List[str]]) -> Optional[List[str]]:
    if not items:
        return None
    out: List[str] = []
    seen = set()
    for item in items:
        for part in item.split(","):
            mk = part.strip()
            if not mk:
                raise ValueError(f"Empty model key in --models: {item!r}")
            if mk in seen:
                raise ValueError(f"Duplicate model key in --models: {mk!r}")
            seen.add(mk)
            out.append(mk)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sequential subprocess-isolated multi-model FICS-PCB inference runner."
    )
    parser.add_argument("--plan", required=True, help="Path to frozen plan JSON containing models, queues, and inference_config.")
    parser.add_argument("--output-dir", required=True, help="New clean output directory for run_<model>/ artifacts.")
    parser.add_argument("--runtime-config", help="Optional local runtime YAML overriding model local_dir, overlay_dir, or hf_home.")
    parser.add_argument("--models", action="append", help="Optional model subset (comma-separated or repeatable).")
    parser.add_argument("--path-map", action="append", help="Repeatable OLD_PREFIX=NEW_PREFIX remap for relocated image paths.")
    parser.add_argument("--image-root", action="append", help="Repeatable directory of relocated images indexed by SHA-256.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only validation of plan, queue layout, runtime config, and output safety (default mode when --execute is omitted).",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the GPU gate check and sequential subprocess inference workers.",
    )
    parser.add_argument("--verify-input-hashes", action="store_true", help="Verify existence and SHA-256 of planned input images during validation.")
    parser.add_argument("--load-timeout", type=float, help="Optional override for model_load_timeout_seconds.")
    parser.add_argument("--per-call-timeout", type=float, help="Optional override for per_call_timeout_seconds.")
    parser.add_argument("--per-model-timeout", type=float, help="Optional override for per_model_gpu_timeout_seconds.")
    parser.add_argument("--total-timeout", type=float, help="Optional override for total_gpu_timeout_seconds.")
    args = parser.parse_args(argv)

    is_dry_run = not bool(args.execute)
    try:
        res = run_sequential_matrix(
            plan_path=pathlib.Path(args.plan),
            output_dir=pathlib.Path(args.output_dir),
            runtime_config_path=pathlib.Path(args.runtime_config) if args.runtime_config else None,
            models_filter=_parse_models(args.models),
            path_remaps=_parse_path_maps(args.path_map),
            image_roots=[pathlib.Path(p) for p in (args.image_root or [])],
            dry_run=is_dry_run,
            verify_input_hashes=bool(args.verify_input_hashes),
            load_timeout_s=args.load_timeout,
            per_call_timeout_s=args.per_call_timeout,
            per_model_timeout_s=args.per_model_timeout,
            total_timeout_s=args.total_timeout,
        )
        res["execution_mode"] = (
            "gpu_execute"
            if args.execute
            else ("cpu_dry_run" if args.dry_run else "cpu_dry_run_default")
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return int(res.get("exit_code", 1))
    except Exception as exc:
        print(json.dumps({"error": repr(exc), "exit_code": 2}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

