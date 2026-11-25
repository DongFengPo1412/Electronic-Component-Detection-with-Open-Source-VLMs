"""CPU, synthetic resolution/coordinate, and HTTP tests for Round 4 high-res budgets and DeepSeek diagnostic."""

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
from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for p in (str(_SRC_ROOT), str(_SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fics_final_demo  # noqa: E402
from pcb_vlm.evaluation.candidate_verifier import parse_single_label_json  # noqa: E402
from pcb_vlm.evaluation.fics_replay import classify_call_termination_and_parse_v2  # noqa: E402
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    FROZEN_DETECTION_PROMPT_SHA256,
    R3_QWEN2_NORM1_PROMPT_SHA256,
    DemoError,
    LiveDemoService,
    resolve_model_preset_spec,
    _write_json_atomic,
)
from pcb_vlm.evaluation.gt_free_pipeline import reconstruct_board_predictions_gt_free  # noqa: E402


def _make_board_png(path: pathlib.Path, w: int = 1280, h: int = 1024) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[60 : h - 60, 80 : w - 80] = [40, 115, 68]
    arr[200:300, 260:420] = [210, 180, 50]
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def _http_req(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_r4_preset_resolution_backward_compat_and_plan_propagation(tmp_path: pathlib.Path) -> None:
    """Verify baseline and r3_candidate stay locked at 200,704 while r4_highres bumps only Qwen3/Qwen2 max_pixels."""
    assert resolve_model_preset_spec("qwen3", "baseline")["qwen_max_pixels"] == 200704
    assert resolve_model_preset_spec("qwen3", "r3_candidate")["qwen_max_pixels"] == 200704
    assert resolve_model_preset_spec("qwen2", "r3_candidate")["qwen_max_pixels"] == 200704

    r4_q3 = resolve_model_preset_spec("qwen3", "r4_highres")
    assert r4_q3["preset_id"] == "r4_qwen3_greedy_maxpx589824"
    assert r4_q3["qwen_min_pixels"] == 65536
    assert r4_q3["qwen_max_pixels"] == 589824
    assert r4_q3["coordinate_protocol"] == "xyxy_norm1000"
    assert r4_q3["prompt_sha256"] == FROZEN_DETECTION_PROMPT_SHA256
    assert r4_q3["explicit_do_sample_false"] is True

    r4_q2 = resolve_model_preset_spec("qwen2", "r4_highres")
    assert r4_q2["preset_id"] == "r4_qwen2_norm1_maxpx451584"
    assert r4_q2["qwen_min_pixels"] == 65536
    assert r4_q2["qwen_max_pixels"] == 451584
    assert r4_q2["coordinate_protocol"] == "xyxy_norm1"
    assert r4_q2["prompt_sha256"] == R3_QWEN2_NORM1_PROMPT_SHA256
    assert r4_q2["explicit_do_sample_false"] is False

    # DeepSeek does not accept dynamic max_pixels preset
    with pytest.raises(DemoError):
        resolve_model_preset_spec("deepseek", "r4_highres")

    # Plan propagation
    img_bytes = _make_board_png(tmp_path / "b.png", 1200, 1000)
    svc = LiveDemoService(workspace_root=tmp_path / "ws", job_runner=lambda *a: None)
    up = svc.register_upload(raw_bytes=img_bytes, raw_filename="b.png")
    meta = json.loads((svc.uploads_dir / up["upload_id"] / "upload_meta.json").read_text(encoding="utf-8"))

    p3 = svc._build_single_job_plan(job_id="j_q3", meta=meta, model_key="qwen3", mode="T", preset="r4_highres")
    assert p3["inference_config"]["qwen_min_pixels"] == 65536
    assert p3["inference_config"]["qwen_max_pixels"] == 589824
    assert p3["models"]["qwen3"]["visual_budget"] == {"shortest_edge": 65536, "longest_edge": 589824}

    p2 = svc._build_single_job_plan(job_id="j_q2", meta=meta, model_key="qwen2", mode="T", preset="r4_qwen2_norm1_maxpx451584")
    assert p2["inference_config"]["qwen_min_pixels"] == 65536
    assert p2["inference_config"]["qwen_max_pixels"] == 451584
    assert p2["models"]["qwen2"]["visual_budget"] == {"min_pixels": 65536, "max_pixels": 451584}


def test_r4_highres_synthetic_box_mapping_preserves_1024_tile_coords() -> None:
    """Even when processor resizes 1024x1024 tile to 768x768 or 672x672, coords map back via original tile WH."""
    v2_q3 = classify_call_termination_and_parse_v2(
        raw_text=json.dumps([{"bbox_2d": [250, 250, 500, 500], "label": "capacitor"}]) + "<|im_end|>",
        width=1024,
        height=1024,
        generated_tokens=20,
        max_new_tokens=512,
        generated_token_ids=[151645],
        eos_token_ids=(151645, 151643),
        protocol="xyxy_norm1000",
    )
    assert v2_q3["objects"][0]["bbox_xyxy"] == [256.0, 256.0, 512.0, 512.0]

    v2_q2 = classify_call_termination_and_parse_v2(
        raw_text=json.dumps([{"bbox_2d": [0.25, 0.25, 0.50, 0.50], "label": "capacitor"}]) + "<|im_end|>",
        width=1024,
        height=1024,
        generated_tokens=20,
        max_new_tokens=512,
        generated_token_ids=[151645],
        eos_token_ids=(151645, 151643),
        protocol="xyxy_norm1",
    )
    assert v2_q2["objects"][0]["bbox_xyxy"] == [256.0, 256.0, 512.0, 512.0]

    recon = reconstruct_board_predictions_gt_free(
        board_id="syn_r4",
        image_id=1,
        orig_wh=(8256, 5504),
        raw_records=[{
            "call_id": "qwen2_syn_T00",
            "board_id": "syn_r4",
            "mode": "T",
            "call_index": 0,
            "tile_id": 0,
            "ok": True,
            "orig_input_width": 1024,
            "orig_input_height": 1024,
            "roi_offset_x": 1906,
            "roi_offset_y": 1278,
            "tile_offset_x": 819,
            "tile_offset_y": 0,
            "max_new_tokens": 512,
            "generated_tokens": 20,
            "generated_token_ids": [151645],
            "coordinate_protocol": "xyxy_norm1",
            "raw_output": json.dumps([{"bbox_2d": [0.25, 0.25, 0.50, 0.50], "label": "capacitor"}]) + "<|im_end|>",
        }],
        mode="T",
        expected_call_ids=["qwen2_syn_T00"],
        coordinate_protocol="xyxy_norm1",
    )
    assert recon["predictions"][0]["bbox_xyxy"] == [2981.0, 1534.0, 3237.0, 1790.0]

    # DeepSeek single-label diagnostic parser sanity check
    p_other = parse_single_label_json('{"label": "other"}<｜end▁of▁sentence｜>')
    assert p_other["strict_json_ok"] is True and p_other["strict_label"] == "other"


def test_http_r4_highres_planned_vs_actual_telemetry_and_single_model_preset_cmp(tmp_path: pathlib.Path) -> None:
    """End-to-end HTTP test for r4_highres qwen3+qwen2, deepseek rejection, and single-model preset in /api/comparisons."""
    raw_bytes = _make_board_png(tmp_path / "small_r4.png", 1200, 1000)

    def fake_runner(job_spec: dict, _cancel_ev: threading.Event, svc: LiveDemoService) -> None:
        mk = job_spec["model_key"]
        jdir = svc.jobs_dir / job_spec["job_id"]
        run_dir = jdir / "base_runs" / f"run_{mk}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(run_dir / "gpu_gate.json", {"passed": True, "smi_free_mib": 7000.0})
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        proto = str(job_spec["coordinate_protocol"])
        max_px = int(plan["inference_config"]["qwen_max_pixels"])
        if mk == "qwen3":
            assert max_px == 589824
            pw, ph, area, vis_toks, grid = 768, 768, 589824, 576, [1, 48, 48]
            raw_txt = json.dumps([{"bbox_2d": [250, 250, 450, 450], "label": "capacitor"}]) + "<|im_end|>"
        else:
            assert max_px == 451584
            pw, ph, area, vis_toks, grid = 672, 672, 451584, 576, [1, 48, 48]
            raw_txt = json.dumps([{"bbox_2d": [0.25, 0.25, 0.45, 0.45], "label": "capacitor"}]) + "<|im_end|>"
        lines = []
        for q in plan["queues"][mk]:
            lines.append(json.dumps({
                "call_id": q["call_id"],
                "board_id": q["board_id"],
                "mode": q["mode"],
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
                "planned_qwen_max_pixels": max_px,
                "processed_width_px": pw,
                "processed_height_px": ph,
                "processed_area_px": area,
                "image_grid_thw": grid,
                "merged_visual_tokens": vis_toks,
                "max_new_tokens": 512,
                "generated_tokens": 20,
                "generated_token_ids": [151645],
                "coordinate_protocol": proto,
                "prompt_sha256": plan["inference_config"]["prompt_sha256"],
                "effective_generation_probe": {
                    "declared_do_sample": False,
                    "explicit_do_sample_kwarg_passed": bool(job_spec["explicit_do_sample_false"]),
                    "effective_do_sample": False,
                    "effective_generation_mode": "GenerationMode.GREEDY_SEARCH",
                },
                "gpu_generate_ms": 50.0,
                "cuda_max_reserved_mib": 2400.0,
                "raw_output": raw_txt,
            }))
        (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    svc = LiveDemoService(workspace_root=tmp_path / "ws_http", job_runner=fake_runner)
    srv = fics_final_demo.create_demo_server(workspace_root=tmp_path / "ws_http", service=svc, host="127.0.0.1", port=0)
    base_url = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        code, _, html = _http_req(f"{base_url}/")
        assert code == 200 and 'value="r4_highres"' in html.decode("utf-8")

        c_up, _, b_up = _http_req(f"{base_url}/api/upload", method="POST", body=raw_bytes, headers={"X-Filename": "small_r4.png"})
        assert c_up == 200
        up = json.loads(b_up.decode("utf-8"))

        # DeepSeek rejected for r4_highres
        c_bad, _, _ = _http_req(
            f"{base_url}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "model_key": "deepseek", "mode": "T", "preset": "r4_highres"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_bad == 400

        # Launch r4_highres comparison for qwen3 + qwen2
        c_cmp, _, b_cmp = _http_req(
            f"{base_url}/api/comparisons",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "model_keys": ["qwen3", "qwen2"], "mode": "T", "preset": "r4_highres"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_cmp == 202
        cmp_id = json.loads(b_cmp.decode("utf-8"))["comparison_id"]

        done = None
        for _ in range(200):
            _, _, b_st = _http_req(f"{base_url}/api/comparisons/{cmp_id}")
            st = json.loads(b_st.decode("utf-8"))
            if st["status"] in ("completed", "failed"):
                done = st
                break
            time.sleep(0.05)
        assert done is not None and done["status"] == "completed"
        by_mk = {s["model_key"]: s for s in done["slots"]}
        assert by_mk["qwen3"]["planned_visual_budget"]["qwen_max_pixels"] == 589824
        assert by_mk["qwen3"]["actual_visual_telemetry_summary"]["processed_area_px"] == 589824
        assert by_mk["qwen2"]["planned_visual_budget"]["qwen_max_pixels"] == 451584
        assert by_mk["qwen2"]["actual_visual_telemetry_summary"]["processed_area_px"] == 451584
        assert by_mk["qwen3"]["predictions"][0]["bbox_xyxy"] == by_mk["qwen2"]["predictions"][0]["bbox_xyxy"]

        # Verify single-model concrete preset ID accepted by /api/comparisons
        c_cmp1, _, b_cmp1 = _http_req(
            f"{base_url}/api/comparisons",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "model_keys": ["qwen3"], "mode": "T", "preset": "r4_qwen3_greedy_maxpx589824"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_cmp1 == 202
        cmp1_id = json.loads(b_cmp1.decode("utf-8"))["comparison_id"]
        for _ in range(200):
            _, _, b_s1 = _http_req(f"{base_url}/api/comparisons/{cmp1_id}")
            s1 = json.loads(b_s1.decode("utf-8"))
            if s1["status"] == "completed":
                break
            time.sleep(0.05)
        assert s1["status"] == "completed"
    finally:
        svc.shutdown()
        srv.shutdown()
        srv.server_close()
