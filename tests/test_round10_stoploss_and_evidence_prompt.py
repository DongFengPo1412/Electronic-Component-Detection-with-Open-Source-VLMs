"""Round 10 tests: per-mode stop-loss checkpoint gates, visible-evidence prompt variant, and R10 preset HTTP contracts."""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import yaml
from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for p in (str(_SRC_ROOT), str(_SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fics_final_demo  # noqa: E402
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    DemoError,
    LiveDemoService,
    resolve_model_preset_spec,
)
from pcb_vlm.evaluation.language_instruction import (  # noqa: E402
    build_single_class_detection_prompt,
    parse_pcb_language_instruction,
)
from pcb_vlm.evaluation.selection_gates import (  # noqa: E402
    evaluate_checkpoint_selection_gate,
    select_best_checkpoint_with_stoploss,
)


def _make_board_png(path: pathlib.Path, w: int = 1200, h: int = 900) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[40 : h - 40, 60 : w - 60] = [170, 50, 50]
    arr[140:230, 180:290] = [210, 180, 50]
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def _http_req(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _wait_job(base_url: str, job_id: str, timeout_iters: int = 200) -> dict:
    for _ in range(timeout_iters):
        _, _, raw = _http_req(f"{base_url}/api/jobs/{job_id}")
        js = json.loads(raw.decode("utf-8"))
        if js["status"] in ("completed", "failed", "incomplete", "cancelled"):
            return js
        time.sleep(0.03)
    raise AssertionError(f"Job {job_id} did not terminate in time")


def test_r10_stoploss_gate_disqualifies_single_class_collapse_despite_absent_100pct() -> None:
    # Mirrors R9 S3 failure: step24/step48 get 2 six_class TPs + 100% absent empty [],
    # but return [] across ALL positive single-class queries.
    collapsed = {
        "tag": "adapter_step48",
        "composite_selection_score": 0.305,
        "per_query": [
            {"query_kind": "six_class", "gt_count": 5, "tp": 2, "fp": 1, "fn": 3, "parsed_count": 3, "strict_schema_ok": True, "explicit_empty": False, "status": "success"},
            {"query_kind": "positive_single_class", "gt_count": 4, "tp": 0, "fp": 0, "fn": 4, "parsed_count": 0, "strict_schema_ok": True, "explicit_empty": True, "status": "success", "count_abs_error": 4, "f1": 0.0},
            {"query_kind": "positive_single_class", "gt_count": 3, "tp": 0, "fp": 0, "fn": 3, "parsed_count": 0, "strict_schema_ok": True, "explicit_empty": True, "status": "success", "count_abs_error": 3, "f1": 0.0},
            {"query_kind": "confirmed_absent", "gt_count": 0, "tp": 0, "fp": 0, "fn": 0, "fp_total": 0, "parsed_count": 0, "strict_schema_ok": True, "explicit_empty": True, "status": "success"},
        ],
    }
    healthy = {
        "tag": "nf4_base_ref",
        "composite_selection_score": 0.0327,
        "per_query": [
            {"query_kind": "six_class", "gt_count": 5, "tp": 3, "fp": 5, "fn": 2, "parsed_count": 8, "strict_schema_ok": False, "explicit_empty": False, "status": "partial_success"},
            {"query_kind": "positive_single_class", "gt_count": 4, "tp": 2, "fp": 3, "fn": 2, "parsed_count": 5, "strict_schema_ok": True, "explicit_empty": False, "status": "success", "count_abs_error": 1, "f1": 0.44},
            {"query_kind": "confirmed_absent", "gt_count": 0, "tp": 0, "fp": 0, "fn": 0, "fp_total": 2, "parsed_count": 2, "strict_schema_ok": True, "explicit_empty": False, "status": "success"},
        ],
    }
    audit_bad = evaluate_checkpoint_selection_gate(collapsed)
    assert audit_bad["eligible"] is False
    assert audit_bad["decision"] == "DISQUALIFIED_STOP_LOSS"
    assert "positive_single_class_all_returned_empty" in audit_bad["disqualifying_reasons"]

    audit_ok = evaluate_checkpoint_selection_gate(healthy)
    assert audit_ok["eligible"] is True

    # If ONLY collapsed checkpoints exist, stop before whole-board runs!
    sel_stop = select_best_checkpoint_with_stoploss([collapsed])
    assert sel_stop["any_eligible"] is False
    assert sel_stop["selected_tag"] is None
    assert sel_stop["action"] == "STOP_NO_WHOLE_BOARD"


def test_r10_visible_evidence_prompt_and_preset_contract() -> None:
    p_base, sha_base = build_single_class_detection_prompt("resistor", prompt_variant="r8_base")
    p_ev, sha_ev = build_single_class_detection_prompt("resistor", prompt_variant="r10_visible_evidence")
    assert sha_base != sha_ev
    assert "actual physical component body is clearly visible" in p_ev
    assert "bare solder pads" in p_ev and "silkscreen text or outlines" in p_ev
    assert "Do NOT output any component other than resistor" in p_ev

    # CH and EN synonymous instructions under R10 variant normalize to identical prompt SHA-256
    c1 = parse_pcb_language_instruction("只找电阻并计数", prompt_variant="r10_visible_evidence")
    c2 = parse_pcb_language_instruction("Locate all resistors and count them", prompt_variant="r10_visible_evidence")
    assert c1["ok"] and c2["ok"]
    assert c1["prompt_variant"] == "r10_visible_evidence"
    assert c1["prompt_sha256"] == c2["prompt_sha256"] == sha_ev

    spec = resolve_model_preset_spec("qwen3", "r10_qwen3_bf16_tile768_evidence_prompt")
    assert spec["preset_id"] == "r10_qwen3_bf16_tile768_evidence_prompt"
    assert spec["instruction_prompt_variant"] == "r10_visible_evidence"
    assert spec["tile_size"] == 768
    assert spec["qwen_max_pixels"] == 589824
    assert spec["explicit_do_sample_false"] is True
    assert spec.get("quantization") is None  # BF16 matches R5/R8

    for other_mk in ("qwen2", "deepseek", "yolo11n"):
        with pytest.raises(DemoError) as exc_info:
            resolve_model_preset_spec(other_mk, "r10_evidence_prompt")
        assert exc_info.value.status_code == 400


def test_r10_http_control_vs_evidence_prompt_propagation_and_ui(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "b_r10.png")
    rt_p = tmp_path / "rt_r10.yaml"
    rt_p.write_text(yaml.safe_dump({"models": {"qwen3": {"local_dir": str(tmp_path)}}}), encoding="utf-8")
    captured: list[dict] = []

    def _mock_runner(job_spec, cancel_ev, svc):
        jdir = svc.jobs_dir / job_spec["job_id"]
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        captured.append({"job_spec": job_spec, "plan": plan})
        run_dir = jdir / "base_runs" / f"run_{job_spec['model_key']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for q in plan["queues"][job_spec["model_key"]]:
            if cancel_ev.is_set():
                return
            objs = [{"bbox_2d": [100, 100, 200, 200], "label": "resistor"}] if int(q["call_index"]) == 0 else []
            lines.append(json.dumps({
                "call_id": q["call_id"],
                "board_id": q["board_id"],
                "tag": q["tag"],
                "mode": q["mode"],
                "image_id": q["image_id"],
                "call_index": q["call_index"],
                "tile_id": q["tile_id"],
                "ok": True,
                "orig_input_width": q["width"],
                "orig_input_height": q["height"],
                "roi_offset_x": q["roi_offset_x"],
                "roi_offset_y": q["roi_offset_y"],
                "tile_offset_x": q["tile_offset_x"],
                "tile_offset_y": q["tile_offset_y"],
                "planned_qwen_min_pixels": 65536,
                "planned_qwen_max_pixels": 589824,
                "processed_width_px": q["width"],
                "processed_height_px": q["height"],
                "processed_area_px": q["width"] * q["height"],
                "image_grid_thw": [1, 48, 48],
                "merged_visual_tokens": 576,
                "max_new_tokens": 512,
                "generated_tokens": 10,
                "generated_token_ids": [151645],
                "coordinate_protocol": "xyxy_norm1000",
                "prompt_sha256": plan["inference_config"]["prompt_sha256"],
                "gpu_generate_ms": 12.0,
                "cuda_max_reserved_mib": 2100.0,
                "raw_output": json.dumps(objs) + "<|im_end|>",
            }))
        (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    svc = LiveDemoService(workspace_root=tmp_path / "ws_r10", runtime_config_path=rt_p, job_runner=_mock_runner)
    srv = fics_final_demo.create_demo_server(workspace_root=svc.workspace_root, service=svc, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        st_h, _, raw_h = _http_req(f"{base}/")
        assert st_h == 200
        assert 'value="r10_qwen3_bf16_tile768_evidence_prompt"' in raw_h.decode("utf-8")

        st_u, _, raw_u = _http_req(f"{base}/api/upload", method="POST", body=png_bytes, headers={"X-Filename": "b_r10.png"})
        assert st_u == 200
        uid = json.loads(raw_u.decode("utf-8"))["upload_id"]

        for preset_id, exp_var in (
            ("r5_qwen3_greedy_maxpx589824_tile768", "r8_base"),
            ("r10_qwen3_bf16_tile768_evidence_prompt", "r10_visible_evidence"),
        ):
            st_j, _, raw_j = _http_req(
                f"{base}/api/jobs/instruct",
                method="POST",
                body=json.dumps({
                    "upload_id": uid,
                    "model_key": "qwen3",
                    "mode": "W",
                    "preset": preset_id,
                    "user_instruction": "只找电阻并计数",
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            assert st_j == 202
            jid = json.loads(raw_j.decode("utf-8"))["job_id"]
            js = _wait_job(base, jid)
            assert js["status"] == "completed"
            _, _, res_raw = _http_req(f"{base}/api/jobs/{jid}/result")
            res = json.loads(res_raw.decode("utf-8"))
            assert res["instruction_spec"]["prompt_variant"] == exp_var

        assert len(captured) == 2
        assert captured[0]["plan"]["inference_config"]["prompt_sha256"] != captured[1]["plan"]["inference_config"]["prompt_sha256"]
        assert "bare solder pads" in captured[1]["plan"]["inference_config"]["prompt"]
        assert "bare solder pads" not in captured[0]["plan"]["inference_config"]["prompt"]

        # Unsupported spatial / out-of-scope rejection check under R10 preset
        st_bad, _, _ = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({
                "upload_id": uid,
                "model_key": "qwen3",
                "mode": "W",
                "preset": "r10_qwen3_bf16_tile768_evidence_prompt",
                "user_instruction": "只找右上角电阻",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_bad == 400
    finally:
        srv.shutdown()
        srv.server_close()
