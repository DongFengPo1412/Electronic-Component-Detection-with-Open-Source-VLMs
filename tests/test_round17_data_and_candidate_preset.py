"""Round 17 regression tests: FICS R15 default preservation, isolated R17 candidate preset, and split isolation."""

from __future__ import annotations

import pathlib
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import prepare_r17_fics_yolo_dataset as r17_prep  # noqa: E402
from pcb_vlm.evaluation.composite_detector import (  # noqa: E402
    COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA,
    DEFAULT_COMPOSITE_DETECTOR_PRESET,
    SUPPORTED_COMPOSITE_DETECTOR_PRESETS,
)
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402


def test_r17_preserves_fics_r15_default_and_registers_candidate(tmp_path: pathlib.Path) -> None:
    assert DEFAULT_COMPOSITE_DETECTOR_PRESET == "fics_yolo11n_r15"
    assert "fics_yolo11n_r17_cand" in SUPPORTED_COMPOSITE_DETECTOR_PRESETS
    assert COMPOSITE_DETECTOR_PRESET_DEFAULT_SCHEMA["fics_yolo11n_r17_cand"] == "fics_6"

    cand_pt = tmp_path / "fics_yolo11n_r17_cand_best.pt"
    cand_pt.write_bytes(b"r17_cand_dummy_weights")
    r15_pt = tmp_path / "fics_yolo11n_r15_best.pt"
    r15_pt.write_bytes(b"r15_dummy_weights")

    rt_cfg_path = tmp_path / "rt.yaml"
    rt_cfg_path.write_text(
        yaml.safe_dump({
            "default_composite_detector_preset": "fics_yolo11n_r15",
            "models": {
                "yolo11n_fics_r15": {
                    "local_dir": str(r15_pt),
                    "weights_sha256": "4d92fd1d8093ea7498c13b3aac6dd5a6e5c21b9965b96667270883aed571009e",
                },
                "yolo11n_fics_r17_cand": {
                    "local_dir": str(cand_pt),
                    "weights_sha256": "deadbeef" * 8,
                },
            },
        }),
        encoding="utf-8",
    )

    svc = LiveDemoService(workspace_root=tmp_path / "ws", runtime_config_path=rt_cfg_path)
    defaults = svc.get_ui_defaults()
    assert defaults["composite_detector"]["default_detector_preset"] == "fics_yolo11n_r15"
    preset_ids = [p["preset_id"] for p in defaults["composite_detector"]["presets"]]
    assert "fics_yolo11n_r15" in preset_ids
    assert "wacv_yolo11n" in preset_ids
    assert "fics_yolo11n_r17_cand" in preset_ids

    spec_default = svc._resolve_composite_detector_spec(None)
    assert spec_default["detector_preset"] == "fics_yolo11n_r15"
    assert spec_default["expected_class_schema"] == "fics_6"

    spec_cand = svc._resolve_composite_detector_spec("fics_yolo11n_r17_cand")
    assert spec_cand["detector_preset"] == "fics_yolo11n_r17_cand"
    assert spec_cand["expected_class_schema"] == "fics_6"
    assert spec_cand["local_dir"] == str(cand_pt)
    assert spec_cand["per_class_conf"] is None


def test_r17_banned_boards_cover_all_dev_blind_and_sister_exclusions() -> None:
    required_banned = {
        "s20", "s4", "s1",
        "s12", "s24", "s17", "s11", "s5", "s6", "s7",
        "s31", "s10", "s16", "s19", "s9",
    }
    assert required_banned.issubset(r17_prep.BANNED_BOARDS)
