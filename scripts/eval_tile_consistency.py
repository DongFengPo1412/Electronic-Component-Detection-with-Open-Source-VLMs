"""Offline overlapping-tile consistency filtering (Candidate A) CLI.

Evaluates Candidate A (GT-free overlapping-tile consistency filtering of tiled `T` predictions)
against baseline `T` post-NMS performance across one or more FICS-PCB cohorts, and can emit
the frozen `method_A_spec.json` specification prior to evaluation.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.fics_replay import (  # noqa: E402
    DEFAULT_FICS_CONFIG_PATH,
    load_fics_six_class_config,
    merge_board_coco_gts_unique_ids,
)
from pcb_vlm.evaluation.tile_consistency import (  # noqa: E402
    DEFAULT_METHOD_A_SPEC,
    evaluate_tile_consistency_vs_baseline,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _parse_cohort_arg(spec_str: str) -> Tuple[Path, Path, Optional[Path]]:
    parts = [p.strip() for p in spec_str.split("::")]
    if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
        raise ValueError(
            f"Expected RAW_JSONL::COCO_JSON[::PLAN_JSON] for --cohort, got {spec_str!r}"
        )
    raw_p = Path(parts[0])
    coco_p = Path(parts[1])
    plan_p = Path(parts[2]) if len(parts) == 3 and parts[2] else None
    return raw_p, coco_p, plan_p


def _extract_roi_sizes_from_plan(plan: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    boards = plan.get("boards") or {}
    for bid, binfo in boards.items():
        if isinstance(binfo, dict) and binfo.get("roi_width") and binfo.get("roi_height"):
            out[str(bid)] = (int(binfo["roi_width"]), int(binfo["roi_height"]))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Candidate A overlapping-tile consistency filter.")
    parser.add_argument(
        "--write-spec",
        type=Path,
        default=None,
        help="Write default method_A_spec.json to this path.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Optional path to method_A_spec.json; defaults to DEFAULT_METHOD_A_SPEC.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="Cohort spec formatted as RAW_JSONL::COCO_JSON[::PLAN_JSON]. Repeatable.",
    )
    parser.add_argument(
        "--model-key",
        default="qwen3",
        help="Model key to filter from EOS/config settings (default: qwen3).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FICS_CONFIG_PATH,
        help="Path to configs/fics_six_classes.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = DEFAULT_METHOD_A_SPEC
    if args.spec is not None:
        with args.spec.open("r", encoding="utf-8") as handle:
            spec = json.load(handle)

    if args.write_spec is not None:
        args.write_spec.parent.mkdir(parents=True, exist_ok=True)
        with args.write_spec.open("w", encoding="utf-8") as handle:
            json.dump(spec, handle, indent=2, ensure_ascii=False)
        print(f"Wrote Candidate A spec: {args.write_spec}")
        if not args.cohort:
            return 0

    if not args.cohort:
        raise ValueError("Provide at least one --cohort RAW_JSONL::COCO_JSON[::PLAN_JSON] or --write-spec")

    cfg = load_fics_six_class_config(args.config)
    eos_ids = tuple(int(x) for x in cfg.get("eos_token_ids_by_model", {}).get(args.model_key, [151645, 151643]))
    max_new_tokens = int(cfg["protocol"]["max_new_tokens_practical"])
    eval_iou = float(cfg["protocol"]["eval_iou_threshold"])

    input_paths: List[Path] = []
    board_records_all: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    roi_sizes_all: Dict[str, Tuple[int, int]] = {}
    board_cocos_to_merge: List[Tuple[str, Dict[str, Any], int]] = []
    next_uid = 1

    for c_str in args.cohort:
        raw_p, coco_p, plan_p = _parse_cohort_arg(c_str)
        input_paths.extend([raw_p, coco_p])
        rows = _load_jsonl(raw_p)
        t_rows = [r for r in rows if str(r.get("mode")) == "T"]
        for r in t_rows:
            board_records_all[str(r["board_id"])].append(r)

        with coco_p.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        cats = coco["categories"]
        anns = coco.get("annotations", [])
        for img in coco["images"]:
            bid = str(img.get("board_id") or img.get("sample_id") or Path(str(img["file_name"])).stem)
            if bid not in board_records_all:
                continue
            iid = img["id"]
            sub_coco = {
                "images": [dict(img)],
                "categories": cats,
                "annotations": [dict(a) for a in anns if str(a["image_id"]) == str(iid)],
            }
            board_cocos_to_merge.append((bid, sub_coco, next_uid))
            next_uid += 1

        if plan_p is not None:
            input_paths.append(plan_p)
            with plan_p.open("r", encoding="utf-8") as handle:
                plan = json.load(handle)
            roi_sizes_all.update(_extract_roi_sizes_from_plan(plan))

    if args.output is not None:
        out_res = args.output.resolve()
        for inp in input_paths:
            if out_res == inp.resolve():
                raise ValueError(f"Refusing to overwrite input file {inp}")

    merged_coco = merge_board_coco_gts_unique_ids(board_cocos_to_merge)
    report = evaluate_tile_consistency_vs_baseline(
        board_records_by_board=dict(board_records_all),
        combined_coco=merged_coco,
        spec=spec,
        roi_sizes_by_board=roi_sizes_all or None,
        eos_token_ids=eos_ids,
        max_new_tokens=max_new_tokens,
        eval_iou_threshold=eval_iou,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"Wrote Candidate A report: {args.output}")
    else:
        print(json.dumps(report["micro_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
