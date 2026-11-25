"""CLI entry point for GT-free Stage A YOLO11n tiled candidate proposal generation."""

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

from pcb_vlm.inference.yolo_proposal_runner import run_yolo_proposal_locator  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GT-free Stage A YOLO11n tiled candidate proposal locator.")
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    res = run_yolo_proposal_locator(
        plan_path=args.plan,
        output_dir=args.output_dir,
        execute=args.execute,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        return 0
    return 0 if res.get("stage_status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
