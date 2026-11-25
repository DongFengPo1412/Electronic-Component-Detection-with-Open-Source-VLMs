import json
import pathlib
import sys

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fics_final_demo as demo  # noqa: E402


def test_r28_redesigned_workbench_tabs_and_static_assets():
    html = demo.HTML_PAGE
    assert ">检测<" in html
    assert ">三模型对比<" in html
    assert ">历史结果<" in html
    assert "上传 PCB 图片" in html
    assert "筛选已有预测（CPU · 0 GPU）" in html
    assert "预测数量" in html
    assert "/static_r28/r28_app.css" in html
    assert "/static_r28/r28_app.js" in html

    static_dir = SCRIPTS_DIR / "static_r28"
    assert (static_dir / "r28_app.css").is_file()
    assert (static_dir / "r28_app.js").is_file()


def test_r28_server_serves_static_assets_offline(tmp_path):
    import urllib.request
    srv = demo.create_demo_server(workspace_root=tmp_path / "ws", host="127.0.0.1", port=0, allow_port_fallback=True)
    import threading
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        css = urllib.request.urlopen(f"{base}/static_r28/r28_app.css", timeout=5).read().decode("utf-8")
        js = urllib.request.urlopen(f"{base}/static_r28/r28_app.js", timeout=5).read().decode("utf-8")
        assert "--primary" in css
        assert "handleFileUpload" in js
    finally:
        srv.shutdown()
        srv.demo_service.shutdown()
        srv.server_close()
