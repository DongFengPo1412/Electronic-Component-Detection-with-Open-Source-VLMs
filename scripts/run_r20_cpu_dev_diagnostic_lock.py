"""CPU-only stop-loss diagnostic replay & pre-GT lock on locked R19 S20/S4/S1 candidates (zero GPU)."""

from __future__ import annotations

import json
import pathlib
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
from pcb_vlm.evaluation.crop_verifier_r18 import load_r18_verifier_checkpoint  # noqa: E402
from pcb_vlm.evaluation.round19_postproc import (  # noqa: E402
    R19_A1_M_REJECT_OTHER,
    R19_A1_T_REJECT_OTHER,
    apply_r19_a1_ic_reject_only,
    tag_r19_a0_predictions,
)
from pcb_vlm.evaluation.round20_ic_relabel import (  # noqa: E402
    apply_r20_targeted_ic_relabel,
    load_r20_head_checkpoint,
)

R19_LIVE = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration19-20261125\live_demo")
R18_VERIFIER_WEIGHTS = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration18-20261125\weights\r18_resnet18_crop_verifier_best.pt")
R18_VERIFIER_SHA = "c27119eac5b98d6f8e56a61b280da94ce30bbed4443bb0cfe57c380d46379ca3"
IMAGENET_CKPT = pathlib.Path(r"C:\Users\Liu\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth")
R20_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration20-20261125")
R20_HEAD_WEIGHTS = R20_ART / "weights" / "r20_ic_relabel_head_best.pt"
R20_RECIPE = R20_ART / "calibration" / "r20_selected_recipe_frozen.json"

TIF_PATHS = {
    "s20": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s20\DSLR\img\s20_front.tif"),
    "s4": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-final-dev-expansion-20261116\extracted\s4\DSLR\img\s4_front.tif"),
    "s1": pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration08-20261122\extracted_new\s1\DSLR\img\s1_front.tif"),
}


def main() -> None:
    t0 = time.perf_counter()
    recipe = json.loads(R20_RECIPE.read_text(encoding="utf-8"))
    assert recipe["calibration_gate_passed"] is False, "Expected stop-loss path"

    cpu_dev = torch.device("cpu")
    r18_model, _, cv_sha = load_r18_verifier_checkpoint(
        R18_VERIFIER_WEIGHTS, expected_sha256=R18_VERIFIER_SHA, device=cpu_dev
    )
    r20_bb, r20_hd, _, r20_sha = load_r20_head_checkpoint(
        head_weights_path=R20_HEAD_WEIGHTS,
        imagenet_ckpt_path=IMAGENET_CKPT,
        expected_sha256=recipe["r20_head_sha256"],
        device=cpu_dev,
    )

    locked_files: List[Dict[str, Any]] = []
    boards_meta: Dict[str, Any] = {}

    def _lock_json(dst: pathlib.Path, obj: Any, role: str) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        locked_files.append({"role": role, "path": str(dst), "sha256": file_sha256(dst).lower(), "bytes": dst.stat().st_size})

    # Reference diagnostic operating point (t=0.80, m=0.25) for error analysis only
    for bkey in ("s20", "s4", "s1"):
        r19_bdir = R19_LIVE / bkey
        tif_p = TIF_PATHS[bkey]
        official_025 = json.loads((r19_bdir / "yolo_official_025.json").read_text(encoding="utf-8"))
        pool_005 = json.loads((r19_bdir / "yolo_low_conf_pool_005.json").read_text(encoding="utf-8"))
        yolo_summary = json.loads((r19_bdir / "yolo_run_summary.json").read_text(encoding="utf-8"))

        a0_preds = tag_r19_a0_predictions(official_025)
        a1_out = apply_r19_a1_ic_reject_only(
            official_preds_025=official_025,
            image_path=tif_p,
            model=r18_model,
            device=cpu_dev,
            t_reject_other=R19_A1_T_REJECT_OTHER,
            m_reject_other=R19_A1_M_REJECT_OTHER,
        )
        r20_out = apply_r20_targeted_ic_relabel(
            a1_predictions=a1_out["predictions"],
            image_path=tif_p,
            backbone=r20_bb,
            head=r20_hd,
            device=cpu_dev,
            t_relabel=0.80,
            m_relabel=0.25,
        )

        out_dir = R20_ART / "live_demo" / bkey
        _lock_json(out_dir / "yolo_official_025.json", official_025, f"{bkey}_r15_025")
        _lock_json(out_dir / "yolo_low_conf_pool_005.json", pool_005, f"{bkey}_pool_005")
        _lock_json(out_dir / "yolo_run_summary.json", yolo_summary, f"{bkey}_yolo_summary")
        _lock_json(out_dir / "r20_a0_predictions.json", a0_preds, f"{bkey}_a0")
        _lock_json(out_dir / "r20_a1_predictions.json", a1_out["predictions"], f"{bkey}_a1")
        _lock_json(out_dir / "r20_a1_relabel_diag_predictions.json", r20_out["predictions"], f"{bkey}_r20_diag")
        _lock_json(out_dir / "r20_relabel_diag_audit.json", {
            "diagnostic_only_stop_loss": True,
            "a1_weights_sha256": cv_sha,
            "r20_head_sha256": r20_sha,
            "a1_summary": a1_out["summary"],
            "r20_summary": r20_out["summary"],
            "relabeled_boxes": r20_out["relabeled_boxes"],
            "overlap_abstained": r20_out["overlap_abstained"],
            "decisions_log": r20_out["decisions_log"],
        }, f"{bkey}_r20_diag_audit")

        boards_meta[bkey] = {
            "tif_path": str(tif_p),
            "tif_sha256": file_sha256(tif_p).lower(),
            "roi_bbox_half_open": yolo_summary["geometry"]["roi_bbox_half_open"],
            "artifacts_dir": str(out_dir),
            "r20_summary": r20_out["summary"],
        }

    lock_doc = {
        "schema_version": "r20_pre_gt_lock_v1",
        "mode": "cpu_stop_loss_diagnostic_replay_only",
        "new_gpu_calls": 0,
        "gt_loaded_during_inference": False,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "boards": boards_meta,
        "locked_files": locked_files,
    }
    lock_path = R20_ART / "eval" / "r20_pre_gt_lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE CPU diagnostic lock: {len(locked_files)} files -> {lock_path}", flush=True)


if __name__ == "__main__":
    main()
