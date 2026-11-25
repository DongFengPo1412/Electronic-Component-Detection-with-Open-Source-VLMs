"""GT-free Round 21 448px calibration inference (serial GPU) & pre-GT lock."""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import time
from typing import Any, Dict, List

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for p in (_REPO_ROOT / "scripts", _SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pcb_vlm.data.fics_dslr import file_sha256  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    DEFAULT_ULTRALYTICS_OVERLAY,
    EXPECTED_FICS_R15_BEST_SHA256,
    run_yolo11n_class_aware_board,
)
from pcb_vlm.evaluation.crop_verifier_r18 import load_r18_verifier_checkpoint  # noqa: E402
from pcb_vlm.evaluation.round19_postproc import (  # noqa: E402
    R19_A1_M_REJECT_OTHER,
    R19_A1_T_REJECT_OTHER,
    apply_r19_a1_ic_reject_only,
    tag_r19_a0_predictions,
)
from pcb_vlm.evaluation.round21_multiscale_merge import (  # noqa: E402
    R21_SUPPLEMENT_IMGSZ,
    R21_SUPPLEMENT_OVERLAP_RATIO,
    R21_SUPPLEMENT_TILE_SIZE,
    merge_r21_448_dq_supplement,
)
from pcb_vlm.geometry.tiling import generate_tiles  # noqa: E402

R15_WEIGHTS = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\weights\fics_yolo11n_r15_best.pt")
R18_VERIFIER_WEIGHTS = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\weights\r18_resnet18_crop_verifier_best.pt")
R18_VERIFIER_SHA = "c27119eac5b98d6f8e56a61b280da94ce30bbed4443bb0cfe57c380d46379ca3"
R18_MANIFEST = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration18-20261125\crop_dataset_r18\crop_manifest.json")

R21_CACHE = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration21-20261125")
R21_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration21-20261125")

CAL_STEMS = ("s14_front1", "s15_front1", "s3_front")


def main() -> None:
    t_all0 = time.perf_counter()
    manifest = json.loads(R18_MANIFEST.read_text(encoding="utf-8"))
    prop_idx = manifest["proposal_cache_index"]

    print("=== Round 21 GPU Budget & Tiling Plan ===", flush=True)
    print("GPU Budget Limit: 600.0s (serial execution)", flush=True)
    for stem in CAL_STEMS:
        entry = prop_idx[stem]
        s640 = json.loads((pathlib.Path(str(entry["preds_025_path"])).parent / "yolo_run_summary.json").read_text(encoding="utf-8"))
        assert str(s640["weights_sha256"]).lower() == EXPECTED_FICS_R15_BEST_SHA256
        assert int(s640["geometry"]["tile_size"]) == 640
        roi = s640["geometry"]["roi_bbox_half_open"]
        rw, rh = roi[2] - roi[0], roi[3] - roi[1]
        t640 = len(generate_tiles(rw, rh, tile_size=640, overlap_ratio=0.20))
        t448 = len(generate_tiles(rw, rh, tile_size=R21_SUPPLEMENT_TILE_SIZE, overlap_ratio=R21_SUPPLEMENT_OVERLAP_RATIO))
        print(f"  {stem:12s} | ROI={rw}x{rh} | 640px tiles={t640} -> 448px tiles={t448}", flush=True)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    r18_model, _, _ = load_r18_verifier_checkpoint(R18_VERIFIER_WEIGHTS, expected_sha256=R18_VERIFIER_SHA, device=dev)

    locked_files: List[Dict[str, Any]] = []
    boards_meta: Dict[str, Any] = {}

    def _lock_copy(src: pathlib.Path, dst: pathlib.Path, role: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        locked_files.append({"role": role, "path": str(dst), "sha256": file_sha256(dst).lower(), "bytes": dst.stat().st_size})

    def _lock_json(dst: pathlib.Path, obj: Any, role: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        locked_files.append({"role": role, "path": str(dst), "sha256": file_sha256(dst).lower(), "bytes": dst.stat().st_size})

    gpu_wall_total = 0.0
    for stem in CAL_STEMS:
        entry = prop_idx[stem]
        tif_p = pathlib.Path(str(entry["tif_path"]))
        p640_dir = pathlib.Path(str(entry["preds_025_path"])).parent
        preds_640_025 = json.loads((p640_dir / "yolo_official_025.json").read_text(encoding="utf-8"))
        pool_640_005 = json.loads((p640_dir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        sum_640 = json.loads((p640_dir / "yolo_run_summary.json").read_text(encoding="utf-8"))

        run_dir = R21_CACHE / "cal_448_runs" / stem
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        res_448 = run_yolo11n_class_aware_board(
            image_path=tif_p,
            board_id=f"{stem}_448",
            weights_path=R15_WEIGHTS,
            expected_weights_sha256=EXPECTED_FICS_R15_BEST_SHA256,
            detector_preset="fics_yolo11n_r15",
            expected_class_schema="fics_6",
            overlay_dir=DEFAULT_ULTRALYTICS_OVERLAY,
            output_dir=run_dir,
            tile_size=R21_SUPPLEMENT_TILE_SIZE,
            overlap_ratio=R21_SUPPLEMENT_OVERLAP_RATIO,
            imgsz=R21_SUPPLEMENT_IMGSZ,
            conf_low=0.05,
            conf_official=0.25,
            nms_iou=0.50,
        )
        sum_448 = res_448["summary"]
        preds_448_025 = res_448["official_predictions_fics6"]
        pool_448_005 = res_448["low_conf_pool_fics6"]
        gpu_wall_total += float(sum_448["timing_and_vram"]["total_wall_seconds"])

        # Run frozen R19 A1 on 640px official_025 + R21 448px D/Q merge
        a0_preds = tag_r19_a0_predictions(preds_640_025)
        a1_out = apply_r19_a1_ic_reject_only(
            official_preds_025=preds_640_025,
            image_path=tif_p,
            model=r18_model,
            device=dev,
            t_reject_other=R19_A1_T_REJECT_OTHER,
            m_reject_other=R19_A1_M_REJECT_OTHER,
        )
        merged = merge_r21_448_dq_supplement(
            a0_preds_640=preds_640_025,
            a1_preds_640=a1_out["predictions"],
            preds_448_025=preds_448_025,
        )

        art_dir = R21_ART / "calibration" / stem
        _lock_copy(p640_dir / "yolo_official_025.json", art_dir / "yolo_640_official_025.json", f"{stem}_640_025")
        _lock_copy(p640_dir / "yolo_low_conf_pool_005.json", art_dir / "yolo_640_low_conf_pool_005.json", f"{stem}_640_005")
        _lock_copy(p640_dir / "yolo_run_summary.json", art_dir / "yolo_640_run_summary.json", f"{stem}_640_sum")
        _lock_json(art_dir / "yolo_448_official_025.json", preds_448_025, f"{stem}_448_025")
        _lock_json(art_dir / "yolo_448_low_conf_pool_005.json", pool_448_005, f"{stem}_448_005")
        _lock_json(art_dir / "yolo_448_run_summary.json", sum_448, f"{stem}_448_sum")
        _lock_json(art_dir / "arm_a0_predictions.json", a0_preds, f"{stem}_a0")
        _lock_json(art_dir / "arm_a1_predictions.json", a1_out["predictions"], f"{stem}_a1")
        _lock_json(art_dir / "arm_a0_plus_448_predictions.json", merged["a0_plus_448"], f"{stem}_a0_plus_448")
        _lock_json(art_dir / "arm_a1_plus_448_predictions.json", merged["a1_plus_448"], f"{stem}_a1_plus_448")
        _lock_json(art_dir / "r21_448_merge_audit.json", {
            "a1_summary": a1_out["summary"],
            "merge_summary": merged["summary"],
            "accepted_448_boxes": merged["accepted_448_boxes"],
            "conflict_suppressed": merged["conflict_suppressed"],
        }, f"{stem}_merge_audit")

        boards_meta[stem] = {
            "tif_path": str(tif_p),
            "tif_sha256": file_sha256(tif_p).lower(),
            "roi_640": sum_640["geometry"]["roi_bbox_half_open"],
            "roi_448": sum_448["geometry"]["roi_bbox_half_open"],
            "tiles_448": sum_448["geometry"]["tile_count"],
            "actual_input_shape_448": sum_448.get("actual_input_shape") or sum_448.get("operating_point", {}).get("actual_input_shape") or [1, 3, 640, 640],
            "wall_seconds_448": sum_448["timing_and_vram"]["total_wall_seconds"],
            "artifacts_dir": str(art_dir),
            "merge_summary": merged["summary"],
        }
        print(f"[LOCKED CAL] {stem}: 448_tiles={sum_448['geometry']['tile_count']}, 448_wall={sum_448['timing_and_vram']['total_wall_seconds']}s, accepted_448_dq={merged['summary']['accepted_448_count']}", flush=True)

    lock_doc = {
        "schema_version": "r21_cal_pre_gt_lock_v1",
        "gt_loaded_during_inference": False,
        "gpu_wall_seconds_total": round(gpu_wall_total, 3),
        "total_wall_seconds": round(time.perf_counter() - t_all0, 3),
        "boards": boards_meta,
        "locked_files": locked_files,
    }
    lock_path = R21_ART / "calibration" / "r21_cal_pre_gt_lock.json"
    lock_path.write_text(json.dumps(lock_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE R21 cal pre-GT lock: {len(locked_files)} files, gpu_wall={gpu_wall_total:.2f}s -> {lock_path}", flush=True)


if __name__ == "__main__":
    main()
