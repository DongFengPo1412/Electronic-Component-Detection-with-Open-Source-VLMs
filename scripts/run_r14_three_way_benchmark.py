"""Round 14 controlled 3-way benchmark on S20 and S4:
1. Locked Qwen3-VL-2B BF16 R5 baseline (768px tiled + B_other historical anchor)
2. YOLO11n direct class-aware (640px/20% overlap, imgsz=640, class-wise NMS 0.50, conf>=0.25 + conf>=0.05 pool saved)
3. YOLO11n + Bounded Qwen3 BF16 verification (<=48 hard candidates per board, single persistent VLM load)

Zero Ground Truth is read before pre_gt_lock.json is written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.candidate_verifier import (  # noqa: E402
    ALLOWED_SEVEN_LABELS,
    DEFAULT_VERIFIER_PROMPT,
)
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    DEFAULT_ULTRALYTICS_OVERLAY,
    EXPECTED_YOLO11N_BEST_SHA256,
    apply_bounded_vlm_verification,
    evaluate_composite_board,
    evaluate_three_way_comparison,
    load_coco_gts_for_single_board,
    query_nvidia_smi_used_mib,
    render_composite_query_overlay_png,
    run_yolo11n_class_aware_board,
    select_hard_candidates_for_vlm,
)
from pcb_vlm.evaluation.gt_free_pipeline import (  # noqa: E402
    compute_board_roi_and_tiles,
    load_jsonl_records,
    prepare_verifier_manifest_gt_free,
)

DEFAULT_BOARDS = {
    "s20": {
        "board_id": "s20_front",
        "image_id": 20,
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif",
        "tif_sha256": "3d485123be2cafdaddd984c975009ca55c92c5c2f31216b1946b55d517dc0b21",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s20_front_coco.json",
        "coco_sha256": "435245f2d06f444a38f91f1c1796c4df286a89372779cd2d3856681a60a0e798",
        "r5_audit_json": r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration05-codex-s4-20261122\round5_independent_s20_and_sft_audit.json",
        "r5_audit_key": "whole_board",
    },
    "s4": {
        "board_id": "s4_front",
        "image_id": 4,
        "tif_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\img\s4_front.tif",
        "tif_sha256": "fc0ea7b44bc74fcfa8dc5b59e3bc592d3f966d849098b6c907c3fd4327bb5906",
        "coco_path": r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\coco\s4_front_coco.json",
        "coco_sha256": "54abebbdd4a51976b504287cb65735e127204bcdc82809e208e012150fbada55",
        "r5_audit_json": r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration05-codex-s4-20261122\s4_round5_independent_audit.json",
        "r5_audit_key": "r5",
    },
}

DEFAULT_YOLO_WEIGHTS = r"C:\Users\Liu\.codex\cache\tasks\pcb-yolo-v3-r7-20261104\training\weights\best.pt"
DEFAULT_QWEN3_DIR = r"C:\Users\Liu\.codex\cache\tasks\pcb-qwen3-s18-20261114\models\Qwen--Qwen3-VL-2B-Instruct\89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_QWEN3_OVERLAY = r"C:\Users\Liu\.codex\cache\tasks\pcb-qwen3-s18-20261114\dependency-overlay"


def _write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_yolo_board_subprocess(
    *,
    python_exe: str,
    script_path: pathlib.Path,
    board_key: str,
    image_path: pathlib.Path,
    out_dir: pathlib.Path,
    weights_path: pathlib.Path,
) -> Dict[str, Any]:
    cmd = [
        python_exe,
        str(script_path),
        "--single-yolo-board", board_key,
        "--single-image-path", str(image_path),
        "--single-out-dir", str(out_dir),
        "--weights-path", str(weights_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300.0, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"YOLO subprocess failed for {board_key} (rc={res.returncode}):\n{res.stderr}\n{res.stdout}")
    return json.loads((out_dir / "yolo_run_summary.json").read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Round 14 S20/S4 3-way controlled composite evaluation.")
    parser.add_argument("--work-dir", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=False, default=None)
    parser.add_argument("--weights-path", type=pathlib.Path, default=pathlib.Path(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--qwen3-dir", type=pathlib.Path, default=pathlib.Path(DEFAULT_QWEN3_DIR))
    parser.add_argument("--qwen3-overlay", type=pathlib.Path, default=pathlib.Path(DEFAULT_QWEN3_OVERLAY))
    parser.add_argument("--max-vlm-candidates-per-board", type=int, default=48)
    parser.add_argument("--global-budget-seconds", type=float, default=900.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--post-gt-only", action="store_true")
    # Internal single-board YOLO subprocess mode to guarantee clean CUDA teardown before Qwen3
    parser.add_argument("--single-yolo-board", default=None)
    parser.add_argument("--single-image-path", type=pathlib.Path, default=None)
    parser.add_argument("--single-out-dir", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    if args.single_yolo_board:
        run_yolo11n_class_aware_board(
            image_path=args.single_image_path,
            board_id=DEFAULT_BOARDS[args.single_yolo_board]["board_id"],
            weights_path=args.weights_path,
            expected_weights_sha256=EXPECTED_YOLO11N_BEST_SHA256,
            overlay_dir=DEFAULT_ULTRALYTICS_OVERLAY,
            output_dir=args.single_out_dir,
            tile_size=640,
            overlap_ratio=0.20,
            imgsz=640,
            conf_low=0.05,
            conf_official=0.25,
            nms_iou=0.50,
        )
        return 0

    t_global0 = time.perf_counter()
    if args.work_dir is None or args.artifact_dir is None:
        raise ValueError("--work-dir and --artifact-dir are required")
    work_dir = pathlib.Path(args.work_dir).resolve()
    art_dir = pathlib.Path(args.artifact_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Pre-check TIFF & weight hashes (never opening COCO GT yet!)
    w_sha = file_sha256(args.weights_path).lower()
    if w_sha != EXPECTED_YOLO11N_BEST_SHA256:
        raise ValueError(f"YOLO11n weight hash mismatch: {w_sha}")

    board_geoms: Dict[str, Any] = {}
    for bk, bcfg in DEFAULT_BOARDS.items():
        tp = pathlib.Path(bcfg["tif_path"])
        t_sha = file_sha256(tp).lower()
        if t_sha != bcfg["tif_sha256"].lower():
            raise ValueError(f"{bk} TIFF hash mismatch: {t_sha}")
        if not args.execute:
            geom = compute_board_roi_and_tiles(
                tp, roi_pad=32, white_thresh=248, tile_size=640, overlap_ratio=0.20,
                export_dir=work_dir / f"dry_{bk}_tiles", board_key=bcfg["board_id"],
            )
            board_geoms[bk] = {"tif_sha256": t_sha, "orig_wh": geom["orig_wh"], "roi_wh": [geom["roi_width"], geom["roi_height"]], "tile_count_640": geom["tile_count"]}

    if not args.execute and not args.post_gt_only:
        print(json.dumps({
            "dry_run": True,
            "weights_sha256": w_sha,
            "max_vlm_candidates_per_board": int(args.max_vlm_candidates_per_board),
            "global_budget_seconds": float(args.global_budget_seconds),
            "boards": board_geoms,
        }, indent=2, ensure_ascii=False))
        return 0

    yolo_summaries: Dict[str, Any] = {}
    pools_005: Dict[str, List[Dict[str, Any]]] = {}
    officials_025: Dict[str, List[Dict[str, Any]]] = {}
    selected_by_board: Dict[str, Dict[str, Any]] = {}
    comp_results: Dict[str, Any] = {}

    if args.post_gt_only:
        lock_doc = json.loads((art_dir / "pre_gt_lock.json").read_text(encoding="utf-8"))
        pre_gt_hashes = dict(lock_doc["hashes"])
        total_stage1_gpu_wall_s = float(lock_doc["stage1_wall_seconds"])
        v_wall_s = float(lock_doc["vlm_verifier_wall_seconds"])
        ucalls = int(lock_doc["vlm_unique_calls"])
        for bk in DEFAULT_BOARDS:
            b_out = art_dir / f"{bk}_yolo"
            yolo_summaries[bk] = json.loads((b_out / "yolo_run_summary.json").read_text(encoding="utf-8"))
            pools_005[bk] = json.loads((b_out / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
            officials_025[bk] = json.loads((b_out / "yolo_official_025.json").read_text(encoding="utf-8"))
            selected_by_board[bk] = json.loads((b_out / "selected_hard_candidates_48.json").read_text(encoding="utf-8"))
            comp_results[bk] = json.loads((art_dir / f"{bk}_vlm_candidate_audit.json").read_text(encoding="utf-8"))
    else:
        # Stage A: Run YOLO11n class-aware on S20 and S4 sequentially (clean subprocess per board)
        locked_files: List[pathlib.Path] = []
        for bk, bcfg in DEFAULT_BOARDS.items():
            b_out = art_dir / f"{bk}_yolo"
            y_sum = _run_yolo_board_subprocess(
                python_exe=sys.executable,
                script_path=pathlib.Path(__file__).resolve(),
                board_key=bk,
                image_path=pathlib.Path(bcfg["tif_path"]),
                out_dir=b_out,
                weights_path=args.weights_path,
            )
            yolo_summaries[bk] = y_sum
            pool = json.loads((b_out / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
            off = json.loads((b_out / "yolo_official_025.json").read_text(encoding="utf-8"))
            pools_005[bk] = pool
            officials_025[bk] = off
            sel = select_hard_candidates_for_vlm(pool, max_candidates=int(args.max_vlm_candidates_per_board))
            selected_by_board[bk] = sel
            _write_json(b_out / "selected_hard_candidates_48.json", sel)
            locked_files.extend([
                b_out / "yolo_raw_outputs_all.jsonl",
                b_out / "yolo_low_conf_pool_005.json",
                b_out / "yolo_official_025.json",
                b_out / "yolo_raw_non_fics_extra.json",
                b_out / "yolo_run_summary.json",
                b_out / "selected_hard_candidates_48.json",
            ])

        # Stage B: ONE combined persistent Qwen3 BF16 verifier run across S20 + S4 (<= 96 candidates total)
        verifier_spec = {
            "model_key": "qwen3",
            "model_repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
            "model_local_dir": str(args.qwen3_dir),
            "overlay_dir": str(args.qwen3_overlay),
            "dtype": "bfloat16",
            "do_sample": False,
            "repetition_penalty": 1.0,
            "max_new_tokens": 32,
            "min_pixels": 65536,
            "max_pixels": 200704,
            "allowed_labels": list(ALLOWED_SEVEN_LABELS),
            "prompt": DEFAULT_VERIFIER_PROMPT,
            "prompt_sha256": hashlib.sha256(DEFAULT_VERIFIER_PROMPT.encode("utf-8")).hexdigest(),
        }
        combined_recon = [
            {
                "board_id": DEFAULT_BOARDS[bk]["board_id"],
                "image_id": DEFAULT_BOARDS[bk]["image_id"],
                "mode": "COMPOSITE_YOLO_QWEN3",
                "technical_complete": True,
                "predictions": selected_by_board[bk]["selected_candidates"],
            }
            for bk in DEFAULT_BOARDS
        ]
        manifest, _ = prepare_verifier_manifest_gt_free(
            reconstructed_by_board=combined_recon,
            board_image_paths={bcfg["board_id"]: pathlib.Path(bcfg["tif_path"]) for bcfg in DEFAULT_BOARDS.values()},
            crops_dir=work_dir / "verifier_crops",
            inference_spec=verifier_spec,
            source_provenance={"experiment": "r14_s20_s4_bounded_vlm", "max_per_board": int(args.max_vlm_candidates_per_board)},
            pad_fraction=0.25,
            min_edge_upscale=224,
        )
        manifest_p = art_dir / "combined_verifier_manifest.json"
        _write_json(manifest_p, manifest)
        locked_files.append(manifest_p)

        ver_out = art_dir / "verifier_runs"
        ucalls = len(manifest.get("unique_calls") or [])
        v_wall0 = time.perf_counter()
        if ucalls > 0:
            v_cmd = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "run_candidate_verifier.py"),
                "--manifest", str(manifest_p),
                "--output-dir", str(ver_out),
                "--stage-timeout", "700",
                "--per-call-timeout", "90",
                "--execute",
            ]
            v_proc = subprocess.run(v_cmd, capture_output=True, text=True, timeout=720.0, check=False)
            if v_proc.returncode != 0:
                raise RuntimeError(f"Verifier failed (rc={v_proc.returncode}):\n{v_proc.stderr}\n{v_proc.stdout}")
            v_rows = load_jsonl_records(ver_out / "verifier_raw_outputs.jsonl")
            locked_files.extend([ver_out / "verifier_raw_outputs.jsonl", ver_out / "verifier_session_summary.json"])
        else:
            v_rows = []
        v_wall_s = round(time.perf_counter() - v_wall0, 3)

        # Apply conservative bounded VLM decisions per board
        for bk, bcfg in DEFAULT_BOARDS.items():
            v_res = apply_bounded_vlm_verification(
                board_id=bcfg["board_id"],
                low_conf_pool_fics6=pools_005[bk],
                selected_candidates=selected_by_board[bk]["selected_candidates"],
                manifest=manifest,
                verifier_rows=v_rows,
                conf_official=0.25,
                rescue_low=0.12,
                nms_iou=0.50,
            )
            comp_results[bk] = v_res
            comp_p = art_dir / f"{bk}_composite_yolo_qwen3.json"
            aud_p = art_dir / f"{bk}_vlm_candidate_audit.json"
            _write_json(comp_p, v_res["final_predictions"])
            _write_json(aud_p, v_res)
            locked_files.extend([comp_p, aud_p])

        total_stage1_gpu_wall_s = round(time.perf_counter() - t_global0, 3)

        # PRE-GT LOCK: Hash EVERY raw output & formal prediction BEFORE touching any COCO GT!
        pre_gt_hashes = {str(p.relative_to(art_dir)): file_sha256(p).lower() for p in sorted(set(locked_files), key=lambda x: str(x))}
        lock_doc = {
            "schema_version": "r14_pre_gt_lock_v1",
            "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stage1_wall_seconds": total_stage1_gpu_wall_s,
            "vlm_verifier_wall_seconds": v_wall_s,
            "vlm_unique_calls": ucalls,
            "locked_file_count": len(pre_gt_hashes),
            "hashes": pre_gt_hashes,
        }
        _write_json(art_dir / "pre_gt_lock.json", lock_doc)

    # POST-GT EVALUATION (ONLY NOW read COCO GTs!)
    per_board_eval: Dict[str, Any] = {}
    for bk, bcfg in DEFAULT_BOARDS.items():
        coco_p = pathlib.Path(bcfg["coco_path"])
        actual_c_sha = file_sha256(coco_p).lower()
        if actual_c_sha != bcfg["coco_sha256"].lower():
            raise ValueError(f"{bk} COCO GT SHA mismatch: {actual_c_sha}")
        gts, coco_img = load_coco_gts_for_single_board(coco_p, image_id=None)

        r5_doc = json.loads(pathlib.Path(bcfg["r5_audit_json"]).read_text(encoding="utf-8"))
        r5_m = r5_doc[bcfg["r5_audit_key"]]["metrics"]
        r5_tp, r5_fp, r5_fn = int(r5_m["tp"]), int(r5_m["fp"]), int(r5_m["fn"])
        r5_prec = float(r5_m["precision"])
        r5_rec = float(r5_m["recall"])
        r5_f1 = float(r5_m["f1"])
        r5_eval = {
            "source_job_id": r5_doc[bcfg["r5_audit_key"]]["job_id"],
            "raw_to_nms": r5_doc[bcfg["r5_audit_key"]].get("raw_to_nms"),
            "class_aware": {
                "tp": r5_tp, "fp": r5_fp, "fn": r5_fn,
                "precision": round(r5_prec, 6), "recall": round(r5_rec, 6), "f1": round(r5_f1, 6),
                "precision_pct": round(r5_prec * 100.0, 2),
                "recall_pct": round(r5_rec * 100.0, 2),
                "f1_pct": round(r5_f1 * 100.0, 2),
                "gt_total": len(gts),
                "pred_total": r5_tp + r5_fp,
                "count_abs_error": abs((r5_tp + r5_fp) - len(gts)),
            },
            "per_class_tp_fp_fn": r5_m.get("per_class_tp_fp_fn"),
        }

        ev_yolo = evaluate_composite_board(
            gts=gts,
            predictions_025=officials_025[bk],
            low_conf_pool_005=pools_005[bk],
        )
        ev_comp = evaluate_composite_board(
            gts=gts,
            predictions_025=comp_results[bk]["final_predictions"],
            low_conf_pool_005=pools_005[bk],
        )

        # Render whole-board comparison overlays (YOLO Direct vs YOLO+Qwen3 Composite)
        roi_png = pathlib.Path(yolo_summaries[bk]["geometry"].get("roi_path") or "")
        if not roi_png.is_file():
            # Render using first-found ROI or compute on fly from TIF
            geom_tmp = compute_board_roi_and_tiles(
                pathlib.Path(bcfg["tif_path"]), roi_pad=32, white_thresh=248, tile_size=640, overlap_ratio=0.20,
                export_dir=work_dir / f"ov_{bk}", board_key=bcfg["board_id"],
            )
            roi_png = pathlib.Path(geom_tmp["roi_export"]["path"])
        ov_yolo = render_composite_query_overlay_png(
            roi_image_path=roi_png,
            roi_offset_x=int(yolo_summaries[bk]["geometry"]["roi_offset_x"]),
            roi_offset_y=int(yolo_summaries[bk]["geometry"]["roi_offset_y"]),
            orig_wh=yolo_summaries[bk]["geometry"]["orig_wh"],
            predictions=officials_025[bk],
        )
        (art_dir / f"{bk}_yolo_direct_overlay.png").write_bytes(ov_yolo)
        ov_comp = render_composite_query_overlay_png(
            roi_image_path=roi_png,
            roi_offset_x=int(yolo_summaries[bk]["geometry"]["roi_offset_x"]),
            roi_offset_y=int(yolo_summaries[bk]["geometry"]["roi_offset_y"]),
            orig_wh=yolo_summaries[bk]["geometry"]["orig_wh"],
            predictions=comp_results[bk]["final_predictions"],
        )
        (art_dir / f"{bk}_yolo_qwen3_composite_overlay.png").write_bytes(ov_comp)

        per_board_eval[bk] = {
            "board_id": bcfg["board_id"],
            "evaluation_scope": "dev_evaluation_not_blind_heldout",
            "tif_sha256": bcfg["tif_sha256"],
            "coco_sha256": actual_c_sha,
            "gt_count": len(gts),
            "yolo_summary": yolo_summaries[bk],
            "selection_summary": {
                "selected_count": selected_by_board[bk]["selected_count"],
                "reason_counts": selected_by_board[bk]["reason_counts"],
            },
            "vlm_decisions": comp_results[bk]["decision_counts"],
            "qwen3_r5": r5_eval,
            "yolo11n_direct": ev_yolo,
            "yolo11n_qwen3_composite": ev_comp,
        }

    three_way_agg = evaluate_three_way_comparison(per_board_results=per_board_eval)

    # Re-verify pre_gt_lock.json byte immutability (0 mismatches!)
    mismatches = []
    for rel, exp_h in pre_gt_hashes.items():
        act_h = file_sha256(art_dir / rel).lower()
        if act_h != exp_h:
            mismatches.append({"file": rel, "expected": exp_h, "actual": act_h})

    eval_out = {
        "schema_version": "r14_s20_s4_three_way_eval_v1",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage1_wall_seconds": total_stage1_gpu_wall_s,
        "vlm_verifier_wall_seconds": v_wall_s,
        "vlm_unique_calls": ucalls,
        "pre_gt_lock_verified": len(mismatches) == 0,
        "pre_gt_lock_mismatches": mismatches,
        "per_board": per_board_eval,
        "three_way_aggregate": three_way_agg,
    }
    _write_json(art_dir / "r14_s20_s4_three_way_eval.json", eval_out)
    print(json.dumps({
        "stage1_wall_seconds": total_stage1_gpu_wall_s,
        "pre_gt_lock_verified": len(mismatches) == 0,
        "combined_micro": three_way_agg["combined_micro"],
        "promotion_gate": three_way_agg["promotion_gate"],
    }, indent=2, ensure_ascii=False))
    return 0 if len(mismatches) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
