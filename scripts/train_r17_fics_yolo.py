"""Controlled Round 17 FICS-PCB 6-class YOLO11n training runner (smoke, train, val whole-board eval).

Strict invariants:
- Uses IDENTICAL initialization (frozen WACV YOLO11n best.pt SHA-256 5378f977...) and IDENTICAL
  FROZEN_TRAIN_HYPERPARAMS as R15 (`train_r15_fics_yolo.py`). Only the audited expanded training
  samples (`fics_yolo_dataset_r17`) differ.
- Training and checkpoint selection ONLY see `train` and frozen `val_split` (`s2_front`, `s2_back`, `s3_front`).
- NEVER reads dev boards (`s20`, `s4`, `s1`) or sealed/reserved blind boards (`s12`, `s24`, `s31`, `s10`, `s16`, `s19`, `s9`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

DEFAULT_OVERLAY_DIR = r"C:\Users\Liu\.codex\cache\tasks\pcb-train-prep-20261104\dependency-overlay"
if pathlib.Path(DEFAULT_OVERLAY_DIR).is_dir() and DEFAULT_OVERLAY_DIR not in sys.path:
    sys.path.insert(0, DEFAULT_OVERLAY_DIR)
os.environ["YOLO_OFFLINE"] = "true"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    EXPECTED_FICS_R15_BEST_SHA256,
    EXPECTED_YOLO11N_BEST_SHA256,
    YOLO_FICS6_ID_TO_RAW_NAME,
    evaluate_composite_board,
    load_coco_gts_for_single_board,
    query_nvidia_smi_used_mib,
    run_yolo11n_class_aware_board,
)
from pcb_vlm.evaluation.fics_replay import FICS_SIX_CLASSES  # noqa: E402
from pcb_vlm.inference.gpu_gate import probe_gpu_gate  # noqa: E402

DEFAULT_WACV_WEIGHTS = pathlib.Path(
    r"C:\Users\Liu\.codex\cache\tasks\pcb-yolo-v3-r7-20261104\training\weights\best.pt"
)
DEFAULT_R15_WEIGHTS = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\weights\fics_yolo11n_r15_best.pt"
)
DEFAULT_DATASET_YAML = pathlib.Path(
    r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration17-20261124\fics_yolo_dataset_r17\dataset.yaml"
)
DEFAULT_TRAIN_RUN_ROOT = pathlib.Path(
    r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration17-20261124\training_run"
)
DEFAULT_ART_ROOT = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration17-20261124"
)
DEFAULT_INVENTORY = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration14-20261123\r14_fics_adaptation_board_inventory.json"
)
DEFAULT_S3_COCO = pathlib.Path(
    r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\data_audit\official_coco\s3_front_official_coco.json"
)

FROZEN_TRAIN_HYPERPARAMS: Dict[str, Any] = {
    "epochs": 30,
    "patience": 12,
    "imgsz": 640,
    "batch": 8,
    "nbs": 16,
    "optimizer": "AdamW",
    "lr0": 0.0005,
    "lrf": 0.01,
    "cos_lr": True,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "amp": False,
    "device": 0,
    "workers": 2,
    "seed": 42,
    "deterministic": True,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "hsv_h": 0.015,
    "hsv_s": 0.2,
    "hsv_v": 0.2,
    "erasing": 0.0,
    "auto_augment": None,
    "close_mosaic": 0,
    "cache": False,
    "plots": True,
    "val": True,
    "save": True,
    "verbose": True,
    "wall_clock_cap_seconds": 3600.0,
    "soft_epoch_stop_seconds": 3300.0,
}


def build_fics6_initialized_from_wacv(
    *,
    wacv_weights_path: pathlib.Path,
    expected_wacv_sha256: str = EXPECTED_YOLO11N_BEST_SHA256,
    init_pt_out_path: pathlib.Path,
) -> Dict[str, Any]:
    if not wacv_weights_path.is_file():
        raise FileNotFoundError(f"WACV weights missing: {wacv_weights_path}")
    actual_sha = file_sha256(wacv_weights_path).lower()
    if expected_wacv_sha256 and actual_sha != expected_wacv_sha256.lower():
        raise ValueError(f"WACV weight SHA mismatch: {actual_sha} != {expected_wacv_sha256}")

    wacv_yolo = YOLO(str(wacv_weights_path))
    wacv_sd = wacv_yolo.model.state_dict()
    arch_yaml = dict(getattr(wacv_yolo.model, "yaml", {}) or {})
    arch_yaml["nc"] = len(FICS_SIX_CLASSES)
    arch_yaml_path = init_pt_out_path.parent / "yolo11n_fics6_arch.yaml"
    arch_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    arch_yaml_path.write_text(yaml.safe_dump(arch_yaml, sort_keys=False), encoding="utf-8")

    target_yolo = YOLO(str(arch_yaml_path))
    target_sd = target_yolo.model.state_dict()

    transferred_keys: List[str] = []
    reinitialized_keys: List[str] = []
    transferred_elements = 0
    reinitialized_elements = 0
    new_sd: Dict[str, torch.Tensor] = {}

    for k, t_tensor in target_sd.items():
        if k in wacv_sd and tuple(wacv_sd[k].shape) == tuple(t_tensor.shape):
            new_sd[k] = wacv_sd[k].detach().clone()
            transferred_keys.append(k)
            transferred_elements += int(t_tensor.numel())
        else:
            new_sd[k] = t_tensor.detach().clone()
            reinitialized_keys.append(k)
            reinitialized_elements += int(t_tensor.numel())

    target_yolo.model.load_state_dict(new_sd, strict=True)
    target_yolo.model.names = dict(YOLO_FICS6_ID_TO_RAW_NAME)
    target_yolo.model.nc = len(FICS_SIX_CLASSES)
    ckpt = {
        "model": target_yolo.model.float(),
        "train_args": {"nc": len(FICS_SIX_CLASSES), "names": dict(YOLO_FICS6_ID_TO_RAW_NAME)},
    }
    init_pt_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(init_pt_out_path))

    reloaded = YOLO(str(init_pt_out_path))
    norm_names = {int(k): str(v).strip().lower() for k, v in (reloaded.names or {}).items()}
    assert norm_names == YOLO_FICS6_ID_TO_RAW_NAME, f"Reloaded names mismatch: {norm_names}"

    return {
        "wacv_weights_path": str(wacv_weights_path),
        "wacv_weights_sha256": actual_sha,
        "init_weights_path": str(init_pt_out_path),
        "init_weights_sha256": file_sha256(init_pt_out_path).lower(),
        "transferred_tensor_count": len(transferred_keys),
        "transferred_param_elements": transferred_elements,
        "reinitialized_tensor_count": len(reinitialized_keys),
        "reinitialized_param_elements": reinitialized_elements,
        "reinitialized_tensor_names": reinitialized_keys,
        "reloaded_names": norm_names,
    }


def snapshot_tensor_delta(pt_before: pathlib.Path, pt_after: pathlib.Path) -> Dict[str, Any]:
    m1 = YOLO(str(pt_before)).model.state_dict()
    m2 = YOLO(str(pt_after)).model.state_dict()
    diff_sq = 0.0
    changed = 0
    total = 0
    for k, v1 in m1.items():
        if k in m2 and v1.is_floating_point():
            total += 1
            d = float((m2[k].float() - v1.float()).abs().max().item())
            if d > 0.0:
                changed += 1
            diff_sq += float((m2[k].float() - v1.float()).pow(2).sum().item())
    return {
        "float_tensors_checked": total,
        "float_tensors_changed": changed,
        "l2_delta": round(math.sqrt(diff_sq), 6),
    }


def _read_results_csv(csv_path: pathlib.Path) -> List[Dict[str, Any]]:
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            clean = {k.strip(): v.strip() for k, v in raw.items() if k is not None}
            parsed: Dict[str, Any] = {}
            for k, v in clean.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _make_smoke_dataset(full_dataset_yaml: pathlib.Path, smoke_ds_dir: pathlib.Path) -> pathlib.Path:
    cfg = yaml.safe_load(full_dataset_yaml.read_text(encoding="utf-8"))
    ds_root = pathlib.Path(cfg["path"])
    if smoke_ds_dir.exists():
        shutil.rmtree(smoke_ds_dir)
    for split in ("train", "val"):
        (smoke_ds_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (smoke_ds_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        lbls = sorted((ds_root / "labels" / split).glob("*.txt"))
        pos = [p for p in lbls if p.read_text(encoding="utf-8").strip()]
        chosen = (pos[:12] + [p for p in lbls if p not in pos[:12]])[:16]
        for lp in chosen:
            ip = ds_root / "images" / split / f"{lp.stem}.png"
            shutil.copy2(ip, smoke_ds_dir / "images" / split / ip.name)
            shutil.copy2(lp, smoke_ds_dir / "labels" / split / lp.name)
    out_yaml = smoke_ds_dir / "dataset.yaml"
    out_yaml.write_text(
        "\n".join([
            f"path: {smoke_ds_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            f"nc: {len(FICS_SIX_CLASSES)}",
            "names:",
            *[f"  {i}: {c}" for i, c in enumerate(FICS_SIX_CLASSES)],
            "",
        ]),
        encoding="utf-8",
    )
    return out_yaml


def run_smoke(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    gate = probe_gpu_gate(5500.0, 6144.0)
    if not bool(gate.get("passed")):
        raise RuntimeError(f"GPU VRAM gate failed: {gate}")

    smoke_root = args.train_run_root / "smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    init_pt = smoke_root / "fics6_init.pt"
    transfer_info = build_fics6_initialized_from_wacv(
        wacv_weights_path=args.wacv_weights,
        init_pt_out_path=init_pt,
    )
    smoke_yaml = _make_smoke_dataset(args.dataset_yaml, smoke_root / "smoke_ds")

    smi_before = query_nvidia_smi_used_mib(0)
    peak_smi = [smi_before or 0.0]

    def _poll_smi(_trainer: Any) -> None:
        v = query_nvidia_smi_used_mib(0)
        if v is not None:
            peak_smi[0] = max(peak_smi[0], v)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    model = YOLO(str(init_pt))
    model.add_callback("on_train_batch_end", _poll_smi)
    model.train(
        data=str(smoke_yaml),
        project=str(smoke_root),
        name="run",
        exist_ok=True,
        epochs=1,
        imgsz=640,
        batch=4,
        amp=False,
        device=0,
        workers=0,
        seed=42,
        deterministic=True,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        erasing=0.0,
        auto_augment=None,
        flipud=0.5,
        fliplr=0.5,
        cache=False,
        plots=False,
        val=True,
        save=True,
        verbose=True,
    )
    last_pt = smoke_root / "run" / "weights" / "last.pt"
    assert last_pt.is_file(), f"Smoke last.pt missing: {last_pt}"
    delta = snapshot_tensor_delta(init_pt, last_pt)
    assert delta["float_tensors_changed"] > 0 and delta["l2_delta"] > 0.0

    reloaded = YOLO(str(last_pt))
    reloaded_names = {int(k): str(v).strip().lower() for k, v in (reloaded.names or {}).items()}
    assert reloaded_names == YOLO_FICS6_ID_TO_RAW_NAME

    csv_rows = _read_results_csv(smoke_root / "run" / "results.csv")
    assert csv_rows
    last_row = csv_rows[-1]
    loss_vals = [float(last_row[k]) for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss") if k in last_row]
    assert len(loss_vals) == 3 and all(math.isfinite(v) for v in loss_vals)

    wall_s = round(time.perf_counter() - t0, 3)
    receipt = {
        "schema_version": "r17_train_smoke_receipt_v1",
        "passed": True,
        "smoke_wall_seconds": wall_s,
        "gpu_gate": gate,
        "transfer_audit": transfer_info,
        "param_update_check": delta,
        "losses": {
            "train/box_loss": last_row.get("train/box_loss"),
            "train/cls_loss": last_row.get("train/cls_loss"),
            "train/dfl_loss": last_row.get("train/dfl_loss"),
            "all_losses_finite": True,
        },
        "reloaded_model_names": reloaded_names,
        "vram": {
            "cuda_max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0), 2) if torch.cuda.is_available() else 0.0,
            "cuda_max_memory_reserved_mib": round(torch.cuda.max_memory_reserved(0) / (1024.0 * 1024.0), 2) if torch.cuda.is_available() else 0.0,
            "nvidia_smi_peak_mib": round(peak_smi[0], 2) if peak_smi[0] else None,
        },
    }
    out_p = args.art_root / "training" / "r17_train_smoke_receipt.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


def run_train(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    gate = probe_gpu_gate(5500.0, 6144.0)
    if not bool(gate.get("passed")):
        raise RuntimeError(f"GPU VRAM gate failed: {gate}")

    run_root = args.train_run_root / "fics6_r17_cand"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    init_pt = run_root / "fics6_init.pt"
    transfer_info = build_fics6_initialized_from_wacv(
        wacv_weights_path=args.wacv_weights,
        init_pt_out_path=init_pt,
    )

    smi_before = query_nvidia_smi_used_mib(0)
    peak_smi = [smi_before or 0.0]
    cap_triggered = [False]
    soft_cap_s = float(FROZEN_TRAIN_HYPERPARAMS["soft_epoch_stop_seconds"])

    def _on_fit_epoch_end(trainer: Any) -> None:
        v = query_nvidia_smi_used_mib(0)
        if v is not None:
            peak_smi[0] = max(peak_smi[0], v)
        if (time.perf_counter() - t0) >= soft_cap_s:
            cap_triggered[0] = True
            trainer.stop = True

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    model = YOLO(str(init_pt))
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    train_kwargs = {
        k: v for k, v in FROZEN_TRAIN_HYPERPARAMS.items()
        if k not in ("wall_clock_cap_seconds", "soft_epoch_stop_seconds")
    }
    model.train(
        data=str(args.dataset_yaml),
        project=str(run_root),
        name="run",
        exist_ok=True,
        **train_kwargs,
    )
    train_wall_s = round(time.perf_counter() - t0, 3)

    weights_dir = run_root / "run" / "weights"
    best_pt_src = weights_dir / "best.pt"
    last_pt_src = weights_dir / "last.pt"
    if not best_pt_src.is_file():
        raise FileNotFoundError(f"Trained best.pt missing: {best_pt_src}")

    best_pt_dst = args.art_root / "weights" / "fics_yolo11n_r17_cand_best.pt"
    best_pt_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt_src, best_pt_dst)
    best_sha = file_sha256(best_pt_dst).lower()

    reloaded = YOLO(str(best_pt_dst))
    reloaded_names = {int(k): str(v).strip().lower() for k, v in (reloaded.names or {}).items()}
    assert reloaded_names == YOLO_FICS6_ID_TO_RAW_NAME

    rows = _read_results_csv(run_root / "run" / "results.csv")
    best_row = max(
        rows,
        key=lambda r: (0.1 * float(r.get("metrics/mAP50(B)", 0.0)) + 0.9 * float(r.get("metrics/mAP50-95(B)", 0.0))),
    ) if rows else {}
    delta = snapshot_tensor_delta(init_pt, best_pt_dst)

    summary = {
        "schema_version": "r17_train_summary_v1",
        "training_wall_seconds": train_wall_s,
        "within_3600s_cap": train_wall_s <= float(FROZEN_TRAIN_HYPERPARAMS["wall_clock_cap_seconds"]),
        "soft_cap_triggered": cap_triggered[0],
        "epochs_completed": len(rows),
        "best_epoch_by_val_fitness": int(best_row.get("epoch", 0)) if best_row else None,
        "best_val_tile_metrics": best_row,
        "checkpoint_selection_policy": "frozen_r15_val_split_only_never_dev_or_blind",
        "recipe_deviation_from_r15": "none",
        "hyperparams": FROZEN_TRAIN_HYPERPARAMS,
        "transfer_audit": transfer_info,
        "param_update_vs_init": delta,
        "best_weights_path": str(best_pt_dst),
        "best_weights_sha256": best_sha,
        "last_weights_path": str(last_pt_src),
        "last_weights_sha256": file_sha256(last_pt_src).lower() if last_pt_src.is_file() else None,
        "reloaded_names": reloaded_names,
        "results_csv_path": str(run_root / "run" / "results.csv"),
        "vram": {
            "cuda_max_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0), 2) if torch.cuda.is_available() else 0.0,
            "cuda_max_memory_reserved_mib": round(torch.cuda.max_memory_reserved(0) / (1024.0 * 1024.0), 2) if torch.cuda.is_available() else 0.0,
            "nvidia_smi_before_mib": smi_before,
            "nvidia_smi_peak_mib": round(peak_smi[0], 2) if peak_smi[0] else None,
        },
        "epoch_history": rows,
    }
    out_p = args.art_root / "training" / "r17_train_summary.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "epoch_history"}, indent=2, ensure_ascii=False))
    return 0


def run_eval_val(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    val_boards = [b for b in inv["boards"] if b["split"] == "val_split"]
    assert len(val_boards) == 3

    r17_best = args.art_root / "weights" / "fics_yolo11n_r17_cand_best.pt"
    if not r17_best.is_file():
        raise FileNotFoundError(f"Trained R17 weights missing: {r17_best}")
    r17_sha = file_sha256(r17_best).lower()
    r15_sha = file_sha256(args.r15_weights).lower()
    assert r15_sha == EXPECTED_FICS_R15_BEST_SHA256

    eval_work = args.train_run_root / "val_whole_board"
    if eval_work.exists():
        shutil.rmtree(eval_work)

    per_board: Dict[str, Any] = {}
    for b in val_boards:
        bid = str(b["board_side_id"])
        img_p = pathlib.Path(str(b["tif_path"]))
        coco_p = pathlib.Path(str(b["coco_path"])) if b.get("coco_path") else DEFAULT_S3_COCO
        gts, _ = load_coco_gts_for_single_board(coco_p, image_id=None)

        arms = {}
        for arm_name, w_p, w_sha, dp in (
            ("fics_yolo11n_r15", args.r15_weights, r15_sha, "fics_yolo11n_r15"),
            ("fics_yolo11n_r17_cand", r17_best, r17_sha, "fics_yolo11n_r17_cand"),
        ):
            out_dir = eval_work / bid / arm_name
            inf = run_yolo11n_class_aware_board(
                image_path=img_p,
                board_id=bid,
                weights_path=w_p,
                expected_weights_sha256=w_sha,
                detector_preset=dp,
                expected_class_schema="fics_6",
                overlay_dir=DEFAULT_OVERLAY_DIR,
                output_dir=out_dir,
                tile_size=640,
                overlap_ratio=0.20,
                imgsz=640,
                conf_low=0.05,
                conf_official=0.25,
                nms_iou=0.50,
            )
            ev = evaluate_composite_board(
                gts=gts,
                predictions_025=inf["official_predictions_fics6"],
                low_conf_pool_005=inf["low_conf_pool_fics6"],
                roi_bbox_half_open=inf["summary"]["geometry"]["roi_bbox_half_open"],
            )
            arms[arm_name] = {"summary": inf["summary"], "metrics": ev}
        per_board[bid] = arms

    combined: Dict[str, Any] = {}
    for arm_name in ("fics_yolo11n_r15", "fics_yolo11n_r17_cand"):
        tp = sum(int(per_board[bid][arm_name]["metrics"]["class_aware"]["tp"]) for bid in per_board)
        fp = sum(int(per_board[bid][arm_name]["metrics"]["class_aware"]["fp"]) for bid in per_board)
        fn = sum(int(per_board[bid][arm_name]["metrics"]["class_aware"]["fn"]) for bid in per_board)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        combined[arm_name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision_pct": round(prec * 100.0, 2),
            "recall_pct": round(rec * 100.0, 2),
            "f1_pct": round(f1 * 100.0, 2),
        }

    val_eval = {
        "schema_version": "r17_val_whole_board_eval_v1",
        "val_inference_wall_seconds": round(time.perf_counter() - t0, 3),
        "combined_val_micro": combined,
        "per_board": {
            bid: {
                arm: {
                    "class_aware": per_board[bid][arm]["metrics"]["class_aware"],
                    "per_class": per_board[bid][arm]["metrics"]["per_class"],
                    "per_size_bin": per_board[bid][arm]["metrics"]["per_size_bin"],
                    "timing_and_vram": per_board[bid][arm]["summary"]["timing_and_vram"],
                }
                for arm in ("fics_yolo11n_r15", "fics_yolo11n_r17_cand")
            }
            for bid in per_board
        },
    }
    out_p = args.art_root / "training" / "r17_val_whole_board_eval.json"
    out_p.write_text(json.dumps(val_eval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(val_eval["combined_val_micro"], indent=2, ensure_ascii=False))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Round 17 FICS YOLO11n training and val evaluation.")
    parser.add_argument("--mode", choices=("smoke", "train", "eval_val"), required=True)
    parser.add_argument("--wacv-weights", type=pathlib.Path, default=DEFAULT_WACV_WEIGHTS)
    parser.add_argument("--r15-weights", type=pathlib.Path, default=DEFAULT_R15_WEIGHTS)
    parser.add_argument("--dataset-yaml", type=pathlib.Path, default=DEFAULT_DATASET_YAML)
    parser.add_argument("--train-run-root", type=pathlib.Path, default=DEFAULT_TRAIN_RUN_ROOT)
    parser.add_argument("--art-root", type=pathlib.Path, default=DEFAULT_ART_ROOT)
    parser.add_argument("--inventory", type=pathlib.Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args(argv)

    if args.mode == "smoke":
        return run_smoke(args)
    if args.mode == "train":
        return run_train(args)
    return run_eval_val(args)


if __name__ == "__main__":
    raise SystemExit(main())
