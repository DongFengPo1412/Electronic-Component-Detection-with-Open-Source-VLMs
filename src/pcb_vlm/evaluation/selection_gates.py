"""Per-task-mode stop-loss checkpoint selection and early-abort gates (Round 10+).

Prevents composite scoring across mixed prompt modes from masking complete collapse
on any positive detection task mode (e.g., R9 step24/step48 returning [] on all
positive single-class S3 queries while scoring 100% on confirmed-absent queries).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


_MODES: tuple[str, ...] = ("six_class", "positive_single_class", "confirmed_absent")


def summarize_per_mode_eval(eval_doc: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Aggregate per-task-mode metrics separately across six_class, positive_single_class, and confirmed_absent."""
    rows: Sequence[Mapping[str, Any]] = eval_doc.get("per_query") or ()
    by_mode: Dict[str, List[Mapping[str, Any]]] = {m: [] for m in _MODES}
    for row in rows:
        kind = str(row.get("query_kind") or "").strip()
        if kind in by_mode:
            by_mode[kind].append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for mode in _MODES:
        mrows = by_mode[mode]
        total_q = len(mrows)
        known_pos_q = sum(1 for r in mrows if int(r.get("gt_count") or 0) > 0)
        gt_total = sum(int(r.get("gt_count") or 0) for r in mrows)
        formal_tp = sum(int(r.get("tp") or 0) for r in mrows)
        formal_fp = sum(int(r.get("fp") if r.get("fp") is not None else (r.get("fp_total") or 0)) for r in mrows)
        formal_fn = sum(int(r.get("fn") or 0) for r in mrows)
        parsed_total = sum(int(r.get("parsed_count") or 0) for r in mrows)
        strict_empty_cnt = sum(
            1
            for r in mrows
            if bool(r.get("explicit_empty")) and int(r.get("parsed_count") or 0) == 0 and str(r.get("status")) != "truncated"
        )
        truncated_cnt = sum(1 for r in mrows if str(r.get("status") or "") in ("truncated", "partial_success", "length_limited_without_eos"))
        failed_cnt = sum(1 for r in mrows if str(r.get("status") or "") in ("failed", "parse_failed", "error"))
        strict_schema_ok_cnt = sum(1 for r in mrows if bool(r.get("strict_schema_ok")))

        prec = (formal_tp / float(formal_tp + formal_fp)) if (formal_tp + formal_fp) > 0 else 0.0
        rec = (formal_tp / float(gt_total)) if gt_total > 0 else None
        f1 = (2.0 * prec * rec / (prec + rec)) if (rec is not None and (prec + rec) > 0.0) else (0.0 if gt_total > 0 else None)

        f1_vals = [float(r.get("f1") or 0.0) for r in mrows if int(r.get("gt_count") or 0) > 0]
        mae_vals = [float(r["count_abs_error"]) for r in mrows if r.get("count_abs_error") is not None]
        out[mode] = {
            "query_count": total_q,
            "known_positive_queries": known_pos_q,
            "gt_total": gt_total,
            "parsed_boxes_total": parsed_total,
            "formal_tp": formal_tp,
            "formal_fp": formal_fp,
            "formal_fn": formal_fn,
            "pooled_precision": round(prec, 4),
            "pooled_recall": round(rec, 4) if rec is not None else None,
            "pooled_f1": round(f1, 4) if f1 is not None else None,
            "macro_f1": round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else None,
            "count_mae": round(sum(mae_vals) / len(mae_vals), 4) if mae_vals else None,
            "strict_valid_empty_count": strict_empty_cnt,
            "strict_valid_empty_rate": round(strict_empty_cnt / total_q, 4) if total_q > 0 else None,
            "all_returned_empty": (total_q > 0 and parsed_total == 0),
            "truncated_count": truncated_cnt,
            "failed_count": failed_cnt,
            "strict_schema_ok_count": strict_schema_ok_cnt,
        }
    return out


def evaluate_checkpoint_selection_gate(
    eval_doc: Mapping[str, Any],
    *,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Audit candidate checkpoint per task mode BEFORE weighted selection scoring.

    Policy for future adapter selection:
    1. Audit `six_class`, `positive_single_class`, and `confirmed_absent` separately.
    2. Disqualify immediately if any positive mode with known positives outputs all empty `[]`
       or obtains zero formal IoU>=0.50 true positives (`formal_tp == 0`).
    3. Confirmed-absent excellence can NEVER compensate for positive-mode collapse.
    4. If NO candidate adapter clears every mode gate, halt before whole-board evaluation (`STOP_NO_WHOLE_BOARD`).
    """
    modes = summarize_per_mode_eval(eval_doc)
    disqualifying_reasons: List[str] = []

    for pos_mode in ("six_class", "positive_single_class"):
        ms = modes[pos_mode]
        if ms["known_positive_queries"] > 0:
            if ms["all_returned_empty"]:
                disqualifying_reasons.append(f"{pos_mode}_all_returned_empty")
            elif ms["formal_tp"] <= 0:
                disqualifying_reasons.append(f"{pos_mode}_zero_formal_tp")

    total_q = sum(m["query_count"] for m in modes.values())
    total_failed = sum(m["failed_count"] for m in modes.values())
    if total_q > 0 and total_failed > 0:
        disqualifying_reasons.append(f"technical_or_parse_failures ({total_failed}/{total_q})")

    abs_ms = modes["confirmed_absent"]
    absent_note = (
        f"confirmed_absent strict_valid_empty={abs_ms['strict_valid_empty_count']}/{abs_ms['query_count']}, "
        f"fp={abs_ms['formal_fp']} (never offsets positive collapse)"
    )

    eligible = len(disqualifying_reasons) == 0
    return {
        "tag": str(tag or eval_doc.get("tag") or "candidate"),
        "eligible": eligible,
        "decision": "PASS_MODE_GATES" if eligible else "DISQUALIFIED_STOP_LOSS",
        "disqualifying_reasons": disqualifying_reasons,
        "absent_audit_note": absent_note,
        "per_mode": modes,
        "policy_rule": "positive_modes_must_have_non_empty_and_formal_tp_gt_0_before_composite_ranking",
    }


def select_best_checkpoint_with_stoploss(
    candidate_evals: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply per-mode stop-loss gates across all candidate checkpoints before ranking."""
    audited = [
        evaluate_checkpoint_selection_gate(doc, tag=str(doc.get("tag") or f"ckpt_{i}"))
        for i, doc in enumerate(candidate_evals)
    ]
    eligible = [a for a in audited if a["eligible"]]
    if not eligible:
        return {
            "any_eligible": False,
            "selected_tag": None,
            "action": "STOP_NO_WHOLE_BOARD",
            "reason": "All candidate checkpoints failed per-task-mode stop-loss gates; do not launch whole-board evaluation.",
            "audited_candidates": audited,
        }

    score_by_tag = {
        str(doc.get("tag") or f"ckpt_{i}"): float(doc.get("composite_selection_score") or 0.0)
        for i, doc in enumerate(candidate_evals)
    }
    eligible.sort(key=lambda a: score_by_tag.get(a["tag"], 0.0), reverse=True)
    return {
        "any_eligible": True,
        "selected_tag": eligible[0]["tag"],
        "action": "PROCEED_WITH_SELECTED_CHECKPOINT",
        "reason": "Selected highest composite candidate among mode-gate-eligible checkpoints.",
        "audited_candidates": audited,
    }
