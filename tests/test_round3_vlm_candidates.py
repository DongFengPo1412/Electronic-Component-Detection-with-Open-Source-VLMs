"""CPU, synthetic coordinate-protocol, generation-probe, and HTTP regression tests for Round 3."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
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
from pcb_vlm.evaluation.fics_replay import classify_call_termination_and_parse_v2  # noqa: E402
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    FROZEN_DETECTION_PROMPT,
    FROZEN_DETECTION_PROMPT_SHA256,
    R3_DEEPSEEK_SHORT_PROMPT,
    R3_DEEPSEEK_SHORT_PROMPT_SHA256,
    R3_QWEN2_NORM1_PROMPT,
    R3_QWEN2_NORM1_PROMPT_SHA256,
    LiveDemoService,
    resolve_model_preset_spec,
    _write_json_atomic,
)
from pcb_vlm.evaluation.gt_free_pipeline import reconstruct_board_predictions_gt_free  # noqa: E402
from pcb_vlm.inference.fics_matrix import build_formal_calls_for_model  # noqa: E402


def _make_board_png(path: pathlib.Path, w: int = 1280, h: int = 1024) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[60 : h - 60, 80 : w - 80, 0] = 35
    arr[60 : h - 60, 80 : w - 80, 1] = 120
    arr[60 : h - 60, 80 : w - 80, 2] = 70
    arr[200:280, 240:380] = [180, 160, 40]
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def _http_req(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _wait_terminal(base_url: str, job_id: str, timeout_s: float = 15.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        code, _, raw = _http_req(f"{base_url}/api/jobs/{job_id}")
        assert code == 200
        st = json.loads(raw.decode("utf-8"))
        if st["status"] in ("completed", "failed", "cancelled", "incomplete"):
            return st
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_s}s")


def test_qwen2_norm1_synthetic_scaling_offsets_and_strict_rejections() -> None:
    """Verify non-square xyxy_norm1 scaling + ROI/tile offset mapping and strict unit enforcement."""
    # Non-square synthetic tile (width=1024, height=768) with non-zero ROI and tile offsets
    width, height = 1024, 768
    roi_ox, roi_oy = 1906, 1278
    tile_ox, tile_oy = 819, 621
    full_w, full_h = 8256, 5504

    valid_raw = json.dumps([
        {"bbox_2d": [0.25, 0.50, 0.75, 0.75], "label": "capacitor"},
    ]) + "<|im_end|>"
    v2_valid = classify_call_termination_and_parse_v2(
        raw_text=valid_raw,
        width=width,
        height=height,
        generated_tokens=20,
        max_new_tokens=512,
        generated_token_ids=[151645],
        eos_token_ids=(151645, 151643),
        protocol="xyxy_norm1",
    )
    assert v2_valid["strict_schema_ok"] is True
    assert len(v2_valid["objects"]) == 1
    assert v2_valid["objects"][0]["bbox_xyxy"] == [256.0, 384.0, 768.0, 576.0]

    # Reject out-of-range (>1.0, <0.0) and degenerate (zero-area) boxes under xyxy_norm1 without unit guessing
    invalid_raw = json.dumps([
        {"bbox_2d": [100, 100, 200, 200], "label": "resistor"},
        {"bbox_2d": [-0.1, 0.2, 0.4, 0.5], "label": "capacitor"},
        {"bbox_2d": [0.5, 0.5, 0.5, 0.8], "label": "ic"},
    ]) + "<|im_end|>"
    v2_inv = classify_call_termination_and_parse_v2(
        raw_text=invalid_raw,
        width=width,
        height=height,
        generated_tokens=25,
        max_new_tokens=512,
        generated_token_ids=[151645],
        eos_token_ids=(151645, 151643),
        protocol="xyxy_norm1",
    )
    assert v2_inv["strict_schema_ok"] is False
    assert v2_inv["objects"] == []
    assert v2_inv["audit"]["invalid_geometry"] == 3

    # End-to-end reconstruction through gt_free_pipeline
    recon = reconstruct_board_predictions_gt_free(
        board_id="syn_board",
        image_id=1,
        orig_wh=(full_w, full_h),
        raw_records=[{
            "call_id": "qwen2_syn_T00",
            "board_id": "syn_board",
            "mode": "T",
            "call_index": 0,
            "tile_id": 0,
            "ok": True,
            "orig_input_width": width,
            "orig_input_height": height,
            "roi_offset_x": roi_ox,
            "roi_offset_y": roi_oy,
            "tile_offset_x": tile_ox,
            "tile_offset_y": tile_oy,
            "max_new_tokens": 512,
            "generated_tokens": 20,
            "generated_token_ids": [151645],
            "coordinate_protocol": "xyxy_norm1",
            "raw_output": valid_raw,
        }],
        mode="T",
        expected_call_ids=["qwen2_syn_T00"],
        coordinate_protocol="xyxy_norm1",
    )
    assert recon["technical_complete"] is True
    assert recon["coordinate_protocol"] == "xyxy_norm1"
    # [256 + 1906 + 819, 384 + 1278 + 621, 768 + 1906 + 819, 576 + 1278 + 621]
    assert recon["predictions"][0]["bbox_xyxy"] == [2981.0, 2283.0, 3493.0, 2475.0]


def test_preset_resolution_and_plan_propagation(tmp_path: pathlib.Path) -> None:
    """Verify baseline vs opt-in r3_candidate preset resolution across service and fics_matrix."""
    b_q3 = resolve_model_preset_spec("qwen3", "baseline")
    assert b_q3["coordinate_protocol"] == "xyxy_norm1000"
    assert b_q3["prompt_sha256"] == FROZEN_DETECTION_PROMPT_SHA256
    assert b_q3["explicit_do_sample_false"] is False

    c_q2 = resolve_model_preset_spec("qwen2", "r3_candidate")
    assert c_q2["preset_id"] == "r3_qwen2_norm1"
    assert c_q2["coordinate_protocol"] == "xyxy_norm1"
    assert c_q2["prompt_sha256"] == R3_QWEN2_NORM1_PROMPT_SHA256
    assert c_q2["explicit_do_sample_false"] is False

    c_ds = resolve_model_preset_spec("deepseek", "r3_candidate")
    assert c_ds["preset_id"] == "r3_deepseek_short_json"
    assert c_ds["coordinate_protocol"] == "xyxy_norm1000"
    assert c_ds["prompt_sha256"] == R3_DEEPSEEK_SHORT_PROMPT_SHA256
    assert c_ds["explicit_do_sample_false"] is False

    c_q3 = resolve_model_preset_spec("qwen3", "r3_candidate")
    assert c_q3["preset_id"] == "r3_qwen3_greedy"
    assert c_q3["coordinate_protocol"] == "xyxy_norm1000"
    assert c_q3["prompt_sha256"] == FROZEN_DETECTION_PROMPT_SHA256
    assert c_q3["explicit_do_sample_false"] is True

    # Build a synthetic upload & single-job plan for each preset and verify fics_matrix propagation
    img_bytes = _make_board_png(tmp_path / "board.png", 1200, 1000)
    svc = LiveDemoService(workspace_root=tmp_path / "ws", job_runner=lambda *a: None)
    up = svc.register_upload(raw_bytes=img_bytes, raw_filename="board.png")
    meta = json.loads((svc.uploads_dir / up["upload_id"] / "upload_meta.json").read_text(encoding="utf-8"))
    for mk, exp_proto, exp_dsf in (
        ("qwen2", "xyxy_norm1", False),
        ("deepseek", "xyxy_norm1000", False),
        ("qwen3", "xyxy_norm1000", True),
    ):
        plan = svc._build_single_job_plan(job_id=f"j_{mk}", meta=meta, model_key=mk, mode="T", preset="r3_candidate")
        calls = build_formal_calls_for_model(plan, mk)
        assert len(calls) == up["tile_count"]
        assert calls[0]["coordinate_protocol"] == exp_proto
        assert calls[0]["explicit_do_sample_false"] is exp_dsf


def test_http_small_image_baseline_vs_r3_candidate_and_gt_scoring(tmp_path: pathlib.Path) -> None:
    """End-to-end HTTP verification on a small out-of-set PCB image with preset switching, empty prediction, and GT scoring."""
    raw_bytes = _make_board_png(tmp_path / "small_test_pcb.png", 1000, 800)

    def fake_runner(job_spec: dict, _cancel_ev: threading.Event, svc: LiveDemoService) -> None:
        mk = job_spec["model_key"]
        jdir = svc.jobs_dir / job_spec["job_id"]
        run_dir = jdir / "base_runs" / f"run_{mk}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(run_dir / "gpu_gate.json", {"passed": True, "smi_free_mib": 7000.0})
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        proto = str(job_spec.get("coordinate_protocol") or "xyxy_norm1000")
        lines = []
        for idx, q in enumerate(plan["queues"][mk]):
            if mk == "qwen2":
                assert proto == "xyxy_norm1"
                # Normalized 0-1 box + an out-of-range 0-1000 box that must be rejected
                raw_txt = json.dumps([
                    {"bbox_2d": [0.20, 0.20, 0.35, 0.35], "label": "capacitor"},
                    {"bbox_2d": [200, 200, 350, 350], "label": "resistor"},
                ]) + "<|im_end|>"
                toks, ids = 30, [151645]
                probe = {"declared_do_sample": False, "explicit_do_sample_kwarg_passed": False, "effective_do_sample": False, "effective_generation_mode": "GenerationMode.GREEDY_SEARCH"}
            elif mk == "deepseek":
                assert proto == "xyxy_norm1000"
                raw_txt = "[]<｜end▁of▁sentence｜>"
                toks, ids = 3, [100001]
                probe = {"declared_do_sample": False, "explicit_do_sample_kwarg_passed": True, "effective_do_sample": False, "effective_generation_mode": "GenerationMode.GREEDY_SEARCH"}
            else:
                assert proto == "xyxy_norm1000"
                assert bool(job_spec.get("explicit_do_sample_false")) is True
                raw_txt = json.dumps([
                    {"bbox_2d": [200, 200, 350, 350], "label": "capacitor"},
                ]) + "<|im_end|>"
                toks, ids = 20, [151645]
                probe = {"declared_do_sample": False, "explicit_do_sample_kwarg_passed": True, "effective_do_sample": False, "effective_generation_mode": "GenerationMode.GREEDY_SEARCH"}
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
                "max_new_tokens": 512,
                "generated_tokens": toks,
                "generated_token_ids": ids,
                "coordinate_protocol": proto,
                "prompt_sha256": q.get("prompt_sha256") or plan["inference_config"]["prompt_sha256"],
                "effective_generation_probe": probe,
                "gpu_generate_ms": 40.0,
                "cuda_max_reserved_mib": 1900.0,
                "raw_output": raw_txt,
            }))
        (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    svc = LiveDemoService(workspace_root=tmp_path / "ws_http", job_runner=fake_runner)
    srv = fics_final_demo.create_demo_server(workspace_root=tmp_path / "ws_http", service=svc, host="127.0.0.1", port=0)
    port = srv.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        code, _, html = _http_req(f"{base_url}/")
        assert code == 200
        assert 'id="livePreset"' in html.decode("utf-8")

        c_up, _, b_up = _http_req(
            f"{base_url}/api/upload",
            method="POST",
            body=raw_bytes,
            headers={"X-Filename": "small_test_pcb.png"},
        )
        assert c_up == 200
        up = json.loads(b_up.decode("utf-8"))

        c_cmp, _, b_cmp = _http_req(
            f"{base_url}/api/comparisons",
            method="POST",
            body=json.dumps({
                "upload_id": up["upload_id"],
                "model_keys": ["qwen2", "deepseek", "qwen3"],
                "mode": "T",
                "preset": "r3_candidate",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_cmp == 202
        cmp_init = json.loads(b_cmp.decode("utf-8"))
        cmp_id = cmp_init["comparison_id"]

        deadline = time.time() + 15.0
        cmp_done = None
        last_cmp_st = None
        while time.time() < deadline:
            c_st, _, b_st = _http_req(f"{base_url}/api/comparisons/{cmp_id}")
            assert c_st == 200
            cmp_st = json.loads(b_st.decode("utf-8"))
            last_cmp_st = cmp_st
            if cmp_st["status"] in ("completed", "failed", "partial_completed", "cancelled", "partial_cancelled"):
                cmp_done = cmp_st
                break
            time.sleep(0.05)
        assert cmp_done is not None and cmp_done["status"] == "completed", json.dumps(last_cmp_st, ensure_ascii=False, indent=2)
        assert cmp_done["preset"] == "r3_candidate"
        by_mk = {s["model_key"]: s for s in cmp_done["slots"]}
        assert by_mk["qwen2"]["kept_predictions_count"] == 1
        assert by_mk["qwen2"]["coordinate_protocol"] == "xyxy_norm1"
        assert by_mk["deepseek"]["kept_predictions_count"] == 0
        assert by_mk["deepseek"]["outcome"]["zero_box_reason"] == "explicit_empty_array"
        assert by_mk["qwen3"]["kept_predictions_count"] == 1
        assert by_mk["qwen3"]["explicit_do_sample_false"] is True
        # Qwen2 [0.20, 0.20, 0.35, 0.35] on same tile matches Qwen3 [200, 200, 350, 350]
        q2_box = by_mk["qwen2"]["predictions"][0]["bbox_xyxy"]
        q3_box = by_mk["qwen3"]["predictions"][0]["bbox_xyxy"]
        assert q2_box == q3_box

        # Verify optional post-hoc GT scoring on Qwen2 job
        coco_gt = {
            "images": [{
                "id": 1,
                "file_name": "small_test_pcb.png",
                "width": up["orig_wh"][0],
                "height": up["orig_wh"][1],
                "sha256": up["source_image_sha256"],
            }],
            "categories": [
                {"id": 1, "name": "capacitor"},
                {"id": 2, "name": "resistor"},
                {"id": 3, "name": "inductor"},
                {"id": 4, "name": "transistor"},
                {"id": 5, "name": "diode"},
                {"id": 6, "name": "ic"},
            ],
            "annotations": [{
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [q2_box[0], q2_box[1], q2_box[2] - q2_box[0], q2_box[3] - q2_box[1]],
                "area": (q2_box[2] - q2_box[0]) * (q2_box[3] - q2_box[1]),
                "iscrowd": 0,
            }],
        }
        c_gt, _, b_gt = _http_req(
            f"{base_url}/api/jobs/{by_mk['qwen2']['job_id']}/evaluate_gt",
            method="POST",
            body=json.dumps({"coco_gt": coco_gt}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_gt == 200
        gt_res = json.loads(b_gt.decode("utf-8"))
        assert gt_res["post_hoc_eval"]["metrics"]["tp"] == 1
        assert gt_res["post_hoc_eval"]["metrics"]["fp"] == 0
        assert gt_res["post_hoc_eval"]["metrics"]["fn"] == 0

        # Export endpoints work cleanly
        for kind in ("json", "overlay_png", "summary_md"):
            c_ex, _, blob = _http_req(f"{base_url}/api/jobs/{by_mk['qwen2']['job_id']}/export/{kind}")
            assert c_ex == 200 and len(blob) > 0
    finally:
        svc.shutdown()
        srv.shutdown()
        srv.server_close()
