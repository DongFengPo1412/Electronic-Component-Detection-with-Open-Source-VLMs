import json
import pathlib
import socket
import sys

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fics_final_demo as demo  # noqa: E402

R27_ART = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-detection-iteration27-20261125")
LAUNCH_CFG = R27_ART / "configs" / "launch_config_r27_unified.json"
RUNTIME_CFG = R27_ART / "configs" / "runtime_config_r27_unified.yaml"


def test_r27_unified_launch_and_runtime_config():
    opts = demo.resolve_server_launch_options(["--config", str(LAUNCH_CFG), "--check-only"])
    assert opts["host"] == "127.0.0.1"
    assert opts["port"] == 8827
    assert opts["allow_port_fallback"] is False
    assert opts["check_only"] is True
    assert opts["runtime_config_path"] == RUNTIME_CFG
    assert opts["results_index_path"].is_file()


def test_r27_html_preset_defaults_and_a1_manual_option():
    html = demo.HTML_PAGE
    assert '<option value="fics_yolo11n_r15" selected>' in html
    assert '<option value="fics_yolo11n_r15_r19_a1_ic_reject">' in html
    assert "S16" in html
    sel_start = html.index('id="selCompositeDetectorPreset"')
    sel_end = html.index("</select>", sel_start)
    sel_block = html[sel_start:sel_end].lower()
    assert "w1" not in sel_block
    assert "w2" not in sel_block


def test_r27_preflight_blocks_port_conflict_and_missing_assets(tmp_path):
    bad_rt = tmp_path / "bad.yaml"
    bad_rt.write_text("models:\n  yolo11n_fics_r15:\n    weights: C:\\nonexistent\\bad.pt\n", encoding="utf-8")
    opts_missing = demo.resolve_server_launch_options(
        ["--config", str(LAUNCH_CFG), "--runtime-config", str(bad_rt), "--check-only"]
    )
    with pytest.raises(SystemExit) as exc_miss:
        demo.preflight_validate_launch(opts_missing)
    assert "Missing local model asset" in str(exc_miss.value)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        busy_port = s.getsockname()[1]
        s.listen(1)
        opts_port = demo.resolve_server_launch_options(
            ["--config", str(LAUNCH_CFG), "--port", str(busy_port), "--check-only"]
        )
        with pytest.raises(SystemExit) as exc_port:
            demo.preflight_validate_launch(opts_port)
        assert str(busy_port) in str(exc_port.value)
