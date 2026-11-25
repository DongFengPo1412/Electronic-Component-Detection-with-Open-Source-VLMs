"""Round 16 Step 2A: Frozen val_split-only IC confidence threshold calibration (CPU only).

Strict invariants:
- ONLY reads locked R15 whole-board predictions on `val_split` (`s2_front`, `s2_back`, `s3_front`).
- NEVER reads `S20`, `S4`, `S1`, or `S31` anywhere in this script.
- Pre-locked candidate grid: `[0.35, 0.45, 0.55, 0.65]` vs baseline `0.25` (other 5 classes fixed at `0.25`).
- Frozen selection rule:
    1) Qualifying set: `val_ic_tp(t) >= val_ic_tp(0.25)` AND `val_ic_fp(t) < val_ic_fp(0.25)`.
    2) Among qualifying candidates, maximize IC FP reduction; tie-break by smallest threshold `t`.
    3) If none qualify, return `selected_ic_threshold = None` (`no_promotable_ic_threshold = True`) and keep `0.25`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    _prf1,
    evaluate_composite_board,
    filter_fics6_pool_by_class_thresholds,
    load_coco_gts_for_single_board,
)

IC_CANDIDATE_THRESHOLDS: List[float] = [0.35, 0.45, 0.55, 0.65]
VAL_BOARDS = {
    "s2_front": {
        "pool": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-20261123\training_run\val_whole_board\s2_front\fics_yolo11n_r15\yolo_low_conf_pool_005.json"),
        "coco": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s2_front_official_coco.json"),
    },
    "s2_back": {
        "pool": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-20261123\training_run\val_whole_board\s2_back\fics_yolo11n_r15\yolo_low_conf_pool_005.json"),
        "coco": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\coco\s2_back_official_coco.json"),
    },
    "s3_front": {
        "pool": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-20261123\training_run\val_whole_board\s3_front\fics_yolo11n_r15\yolo_low_conf_pool_005.json"),
        "coco": pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\data_audit\official_coco\s3_front_official_coco.json"),
    },
}
DEFAULT_OUT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration16-20261124\ic_calibration\r16_val_ic_calibration_frozen.json")


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for ch in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest().lower()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate IC-only threshold on frozen val_split only.")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    input_hashes: Dict[str, Any] = {}
    pools: Dict[str, List[Dict[str, Any]]] = {}
    gts_by_board: Dict[str, List[Dict[str, Any]]] = {}
    for bid, paths in VAL_BOARDS.items():
        pool_p = paths["pool"]
        coco_p = paths["coco"]
        assert pool_p.is_file(), f"Missing val pool: {pool_p}"
        assert coco_p.is_file(), f"Missing val coco: {coco_p}"
        input_hashes[bid] = {
            "pool_path": str(pool_p),
            "pool_sha256": _sha256(pool_p),
            "coco_path": str(coco_p),
            "coco_sha256": _sha256(coco_p),
        }
        pools[bid] = json.loads(pool_p.read_text(encoding="utf-8"))
        gts_by_board[bid], _ = load_coco_gts_for_single_board(coco_p, image_id=None)

    def _eval_threshold(ic_thr: float) -> Dict[str, Any]:
        tot_tp = tot_fp = tot_fn = 0
        ic_tp = ic_fp = ic_fn = ic_gt = ic_pred = 0
        per_board: Dict[str, Any] = {}
        for bid in VAL_BOARDS:
            preds = filter_fics6_pool_by_class_thresholds(
                pools[bid],
                default_conf=0.25,
                per_class_conf={"ic": float(ic_thr)},
            )
            ev = evaluate_composite_board(
                gts=gts_by_board[bid],
                predictions_025=preds,
                low_conf_pool_005=pools[bid],
            )
            ca = ev["class_aware"]
            ic_m = ev["per_class"]["ic"]
            tot_tp += int(ca["tp"]); tot_fp += int(ca["fp"]); tot_fn += int(ca["fn"])
            ic_tp += int(ic_m["tp"]); ic_fp += int(ic_m["fp"]); ic_fn += int(ic_m["fn"])
            ic_gt += int(ic_m["gt_count"]); ic_pred += int(ic_m["pred_count"])
            per_board[bid] = {
                "overall": ca,
                "ic": ic_m,
            }
        overall_m = _prf1(tot_tp, tot_fp, tot_fn)
        ic_overall_m = _prf1(ic_tp, ic_fp, ic_fn)
        return {
            "ic_threshold": float(ic_thr),
            "overall_micro": overall_m,
            "ic_micro": {
                "gt_count": ic_gt,
                "pred_count": ic_pred,
                "signed_count_error": ic_pred - ic_gt,
                **ic_overall_m,
            },
            "per_board": per_board,
        }

    baseline_025 = _eval_threshold(0.25)
    base_ic_tp = int(baseline_025["ic_micro"]["tp"])
    base_ic_fp = int(baseline_025["ic_micro"]["fp"])

    candidates_eval: List[Dict[str, Any]] = []
    qualifying: List[Dict[str, Any]] = []
    for t_ic in IC_CANDIDATE_THRESHOLDS:
        res = _eval_threshold(t_ic)
        ic_tp = int(res["ic_micro"]["tp"])
        ic_fp = int(res["ic_micro"]["fp"])
        fp_red = base_ic_fp - ic_fp
        qualifies = (ic_tp >= base_ic_tp) and (ic_fp < base_ic_fp)
        res["ic_tp_delta_vs_025"] = ic_tp - base_ic_tp
        res["ic_fp_reduction_vs_025"] = fp_red
        res["ic_fp_reduction_pct"] = round((fp_red / base_ic_fp) * 100.0, 2) if base_ic_fp > 0 else 0.0
        res["qualifies_frozen_rule"] = qualifies
        candidates_eval.append(res)
        if qualifies:
            qualifying.append(res)

    if qualifying:
        chosen = max(qualifying, key=lambda r: (int(r["ic_fp_reduction_vs_025"]), -float(r["ic_threshold"])))
        selected_thr: Optional[float] = float(chosen["ic_threshold"])
        no_promotable = False
        decision_note = (
            f"Selected ic_threshold={selected_thr:.2f} strictly on val_split: "
            f"IC TP maintained ({chosen['ic_micro']['tp']} >= {base_ic_tp}) while IC FP dropped "
            f"{base_ic_fp} -> {chosen['ic_micro']['fp']} (-{chosen['ic_fp_reduction_vs_025']}, -{chosen['ic_fp_reduction_pct']}%)."
        )
    else:
        selected_thr = None
        no_promotable = True
        decision_note = "No candidate in [0.35, 0.45, 0.55, 0.65] preserved val_split IC TP while reducing IC FP; keep 0.25."

    out_doc = {
        "schema_version": "r16_val_ic_calibration_frozen_v1",
        "calibration_split": "val_split (s2_front, s2_back, s3_front; 158 GT, 20 IC GT)",
        "never_used_boards": ["s20", "s4", "s1", "s31", "s10", "s16", "s19", "s9"],
        "candidate_grid": IC_CANDIDATE_THRESHOLDS,
        "frozen_selection_rule": "val_ic_tp(t) >= val_ic_tp(0.25) AND val_ic_fp(t) < val_ic_fp(0.25); maximize FP reduction, tie-break smallest t",
        "input_hashes": input_hashes,
        "baseline_025": baseline_025,
        "candidates": candidates_eval,
        "selected_ic_threshold": selected_thr,
        "no_promotable_ic_threshold": no_promotable,
        "decision_note": decision_note,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_025_ic": baseline_025["ic_micro"],
        "candidates_summary": [
            {
                "t": c["ic_threshold"],
                "ic_tp": c["ic_micro"]["tp"],
                "ic_fp": c["ic_micro"]["fp"],
                "ic_fp_red_pct": c["ic_fp_reduction_pct"],
                "overall_f1_pct": c["overall_micro"]["f1_pct"],
                "qualifies": c["qualifies_frozen_rule"],
            }
            for c in candidates_eval
        ],
        "selected_ic_threshold": selected_thr,
        "no_promotable_ic_threshold": no_promotable,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
