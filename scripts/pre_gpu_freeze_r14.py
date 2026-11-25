"""Freeze Round 14 source code, runtime config, YOLO weights, and input hashes before any GPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any, Dict, Optional, Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze Round 14 code & inputs prior to GPU runs.")
    parser.add_argument("--runtime-config", type=pathlib.Path, required=True)
    parser.add_argument("--inventory", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    py_hashes: Dict[str, str] = {}
    for sub in ("src", "scripts", "tests"):
        root = _REPO_ROOT / sub
        for fp in sorted(root.rglob("*.py")):
            rel = fp.relative_to(_REPO_ROOT).as_posix()
            py_hashes[rel] = file_sha256(fp).lower()

    inputs_to_hash = {
        "runtime_config_pre_plan": str(args.runtime_config),
        "board_inventory": str(args.inventory),
        "yolo11n_best_pt": r"C:\Users\Liu\.codex\cache\tasks\pcb-yolo-v3-r7-20261104\training\weights\best.pt",
        "s20_tif": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif",
        "s20_coco": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s20_front_coco.json",
        "s4_tif": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\img\s4_front.tif",
        "s4_coco": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s4_front_coco.json",
        "s1_tif": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s1\DSLR\img\s1_front.tif",
        "s20_T15_png": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration13-20261123\tmp_work\demo_input\s20_T15.png",
        "s20_T15_coco": r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration12-20261123\tmp_work\demo_input\s20_T15_coco.json",
        "r5_s20_audit": r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration05-codex-s4-20261122\round5_independent_s20_and_sft_audit.json",
        "r5_s4_audit": r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration05-codex-s4-20261122\s4_round5_independent_audit.json",
    }
    input_hashes = {k: {"path": v, "sha256": file_sha256(pathlib.Path(v)).lower()} for k, v in inputs_to_hash.items()}

    doc = {
        "schema_version": "r14_frozen_plan_v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "software_copy_root": str(_REPO_ROOT),
        "global_gpu_budget_seconds": 900.0,
        "reserved_untouched_boards": ["s31", "s10", "s16", "s19"],
        "excluded_wacv_overlap_boards": ["s9"],
        "code_file_count": len(py_hashes),
        "code_sha256s": py_hashes,
        "inputs": input_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "frozen_at": doc["frozen_at"],
        "code_file_count": len(py_hashes),
        "yolo11n_best_pt": input_hashes["yolo11n_best_pt"]["sha256"],
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
