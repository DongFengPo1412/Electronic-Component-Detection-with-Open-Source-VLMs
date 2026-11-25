"""Write pre-GPU freeze manifest (`frozen_plan_r15_pre_gpu.json`) before any R15 GPU run."""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

REPO_ROOT = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-20261123\software_copy")
ART_ROOT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123")
DS_DIR = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration15-20261123\fics_yolo_dataset")
WACV_PT = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-yolo-v3-r7-20261104\training\weights\best.pt")

FILES_TO_FREEZE = [
    REPO_ROOT / "src" / "pcb_vlm" / "evaluation" / "composite_detector.py",
    REPO_ROOT / "src" / "pcb_vlm" / "evaluation" / "final_demo_service.py",
    REPO_ROOT / "scripts" / "fics_final_demo.py",
    REPO_ROOT / "scripts" / "run_composite_yolo_board.py",
    REPO_ROOT / "scripts" / "prepare_r15_fics_yolo_dataset.py",
    REPO_ROOT / "scripts" / "verify_r15_dataset_cpu.py",
    REPO_ROOT / "scripts" / "train_r15_fics_yolo.py",
    REPO_ROOT / "scripts" / "run_r15_web_paired_inference.py",
    REPO_ROOT / "scripts" / "eval_r15_paired_posthoc.py",
    REPO_ROOT / "tests" / "test_round15_fics_adaptation.py",
    DS_DIR / "dataset.yaml",
    ART_ROOT / "data_audit" / "r15_dataset_audit.json",
    ART_ROOT / "data_audit" / "r15_dataset_cpu_audit.json",
    ART_ROOT / "data_audit" / "official_coco" / "s3_front_official_coco.json",
    ART_ROOT / "data_audit" / "official_coco" / "s31_front_official_coco.json",
    WACV_PT,
]


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def main() -> int:
    entries = {}
    for p in FILES_TO_FREEZE:
        if not p.is_file():
            raise FileNotFoundError(f"Missing frozen file: {p}")
        entries[str(p)] = {"sha256": _sha256(p), "bytes": p.stat().st_size}

    manifest = {
        "schema_version": "r15_frozen_plan_pre_gpu_v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "main_repo_readonly": {
            "path": r"C:\workspace\Electronic-Component-Detection-with-Open-Source-VLMs",
            "expected_head": "4864b605dfc6c5d51e330e70290f98e4b7778fc1",
        },
        "pre_registered_promotion_gate": {
            "s20_s4_combined_micro_f1_min_pct": 37.57,
            "s20_min_tp": 29,
            "s4_min_tp": 28,
            "checkpoint_selection": "val_split_only_never_s20_s4_s1",
        },
        "reserved_untouched_blind_boards": ["s31", "s10", "s16", "s19"],
        "excluded_overlap_boards": ["s9"],
        "frozen_files": entries,
    }
    out = ART_ROOT / "frozen_plan_r15_pre_gpu.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing pre-GPU freeze manifest: {out}")
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Froze {len(entries)} files -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
