"""Write pre-live freeze manifest (`frozen_plan_r16_pre_live.json`) before starting port 8820 server."""

from __future__ import annotations

import hashlib
import json
import pathlib
import time

REPO_ROOT = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration16-20261124\software_copy")
ART_ROOT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration16-20261124")
FICS_R15_PT = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration15-20261123\weights\fics_yolo11n_r15_best.pt")

FILES_TO_FREEZE = [
    REPO_ROOT / "src" / "pcb_vlm" / "evaluation" / "composite_detector.py",
    REPO_ROOT / "src" / "pcb_vlm" / "evaluation" / "final_demo_service.py",
    REPO_ROOT / "scripts" / "fics_final_demo.py",
    REPO_ROOT / "scripts" / "run_composite_yolo_board.py",
    REPO_ROOT / "scripts" / "calibrate_r16_ic_threshold_val_only.py",
    REPO_ROOT / "scripts" / "eval_r16_ic_threshold_regression.py",
    REPO_ROOT / "scripts" / "audit_r16_weak_class_families.py",
    REPO_ROOT / "scripts" / "run_r16_cold_start_live_acceptance.py",
    REPO_ROOT / "tests" / "test_round16_default_and_ic_calibration.py",
    ART_ROOT / "runtime_config_r16.yaml",
    ART_ROOT / "r16_launch_config.json",
    ART_ROOT / "ic_calibration" / "r16_val_ic_calibration_frozen.json",
    FICS_R15_PT,
]


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for ch in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest().lower()


def main() -> int:
    entries = {str(p): {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in FILES_TO_FREEZE}
    doc = {
        "schema_version": "r16_frozen_plan_pre_live_v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "default_composite_detector_preset": "fics_yolo11n_r15",
        "expected_fics_r15_weights_sha256": "4d92fd1d8093ea7498c13b3aac6dd5a6e5c21b9965b96667270883aed571009e",
        "frozen_files": entries,
    }
    out = ART_ROOT / "frozen_plan_r16_pre_live.json"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Froze {len(entries)} files -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
