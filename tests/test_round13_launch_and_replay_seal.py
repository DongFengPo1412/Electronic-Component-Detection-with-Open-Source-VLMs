"""Round 13 tests: CWD-independent launch config, runtime YAML results_index injection, and Tab 2 replay isolation."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import urllib.request

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for p in (str(_SRC_ROOT), str(_SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fics_final_demo  # noqa: E402


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def test_cwd_independent_launch_config_and_runtime_yaml_replay_injection(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx_file = pathlib.Path(r"C:\Users\Liu\.codex\artifacts\pcb-final-demo-integration-20261116\results_index.json")
    assert idx_file.is_file()

    ws = tmp_path / "ws_r13"
    rt_yaml = tmp_path / "rt.yaml"
    rt_yaml.write_text(
        f"results_index_path: {idx_file}\n"
        "default_model: qwen3\n"
        "default_mode: T\n"
        "default_preset: r5_qwen3_greedy_maxpx589824_tile768\n",
        encoding="utf-8",
    )
    launch_cfg = tmp_path / "launch.json"
    launch_cfg.write_text(
        json.dumps({
            "workspace_root": str(ws),
            "runtime_config": str(rt_yaml),
            "results_index": str(idx_file),
            "host": "127.0.0.1",
            "port": 8817,
            "allow_port_fallback": False,
        }),
        encoding="utf-8",
    )

    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    opts = fics_final_demo.resolve_server_launch_options(["--config", str(launch_cfg)])
    assert opts["workspace_root"] == ws.resolve()
    assert opts["runtime_config_path"] == rt_yaml.resolve()
    assert opts["results_index_path"] == idx_file.resolve()
    assert opts["port"] == 8817 and opts["allow_port_fallback"] is False

    # Even if results_index_path=None is passed directly to create_demo_server, runtime_config_path injects ResearchReplayStore!
    srv = fics_final_demo.create_demo_server(
        workspace_root=ws,
        runtime_config_path=rt_yaml,
        results_index_path=None,
        host="127.0.0.1",
        port=0,
    )
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"{base}/", timeout=15) as resp:
            html = resp.read().decode("utf-8")
        assert "8 板冻结归档与 S10 证据" not in html
        assert "S10" in html and "未接入本标签页" in html

        health = _get_json(f"{base}/api/health")
        assert health["research_available"] is True

        ov = _get_json(f"{base}/api/research/overview")
        assert ov["available"] is True
        assert "heldout_blind_s19_s16" in ov["datasets"]

        s19 = _get_json(f"{base}/api/research/board?dataset=heldout_blind_s19_s16&board=s19&model=qwen3&mode=T")
        s16 = _get_json(f"{base}/api/research/board?dataset=heldout_blind_s19_s16&board=s16&model=qwen3&mode=T")
        assert s19["board_id"] == "s19" and s16["board_id"] == "s16"
        assert s19["split_group"] == "heldout_blind" and s16["split_group"] == "heldout_blind"
        assert s19["metrics"]["total_gt"] == 67 and s16["metrics"]["total_gt"] == 40
        # Historical replay MUST NOT create any inference jobs!
        assert list((ws / "jobs").iterdir()) == []
        assert list((ws / "comparisons").iterdir()) == []
    finally:
        srv.shutdown()
        srv.server_close()
