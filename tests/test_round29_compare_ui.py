import json
import pathlib
import sys
import threading
import urllib.request

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fics_final_demo as demo  # noqa: E402


def test_r29_compare_ui_has_slots_view_linked_zoom_and_gt_import():
    html = demo.HTML_PAGE
    assert 'id="selCmpResultView"' in html
    assert 'id="btnCmpZoomFit"' in html
    assert 'id="fileCmpGtCoco"' in html
    assert 'id="btnEvalCmpGt"' in html

    js = (SCRIPTS_DIR / "static_r28" / "r28_app.js").read_text(encoding="utf-8")
    assert "cmp.slots" in js
    assert "wall_elapsed_seconds" in js
    assert "format_non_compliant_zero_valid_boxes" in js
    assert "textContent" in js


def test_r29_replay_persisted_r28_comparison_api():
    ws = pathlib.Path(r"C:\Users\Liu\.codex\cache\tasks\pcb-detection-iteration29-20261126\live_ws")
    if not (ws / "comparisons" / "cmp_20261126_030920_5643df53" / "comparison.json").is_file():
        return
    srv = demo.create_demo_server(workspace_root=ws, host="127.0.0.1", port=0, allow_port_fallback=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        d = json.loads(urllib.request.urlopen(f"{base}/api/comparisons/cmp_20261126_030920_5643df53", timeout=5).read().decode("utf-8"))
        assert d["status"] == "completed"
        assert float(d["wall_elapsed_seconds"]) > 0
        by_mk = {s["model_key"]: s for s in d["slots"]}
        assert by_mk["qwen3"]["kept_predictions_count"] == 14
        assert by_mk["qwen2"]["kept_predictions_count"] == 6
        assert by_mk["deepseek"]["kept_predictions_count"] == 0
        assert by_mk["deepseek"]["outcome"]["zero_box_reason"] == "format_non_compliant_zero_valid_boxes"
    finally:
        srv.shutdown()
        srv.demo_service.shutdown()
        srv.server_close()
