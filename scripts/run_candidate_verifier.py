"""GT-free CLI entry point for supervised second-stage candidate crop verification."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional, Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.inference.candidate_verifier_runner import (  # noqa: E402
    run_verifier_supervised,
    run_verifier_worker,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GT-free supervised Candidate B crop verifier.")
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--load-timeout", type=float, default=180.0)
    parser.add_argument("--per-call-timeout", type=float, default=90.0)
    parser.add_argument("--stage-timeout", type=float, default=3600.0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.worker:
        return run_verifier_worker(args.manifest)

    if args.output_dir is None:
        parser.error("--output-dir is required unless --worker is set")

    res = run_verifier_supervised(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        load_timeout_s=args.load_timeout,
        per_call_timeout_s=args.per_call_timeout,
        stage_timeout_s=args.stage_timeout,
        execute=args.execute,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if not args.execute:
        return 0
    return 0 if res.get("stage_status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
