"""CPU, 768px tiling contract, synthetic global mapping, and HTTP tests for Round 5."""

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
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    FROZEN_DETECTION_PROMPT_SHA256,
    DemoError,
    LiveDemoService,
    _write_json_atomic,
    resolve_model_preset_spec,
)
from pcb_vlm.evaluation.gt_free_pipeline import reconstruct_board_predictions_gt_free  # noqa: E402
from pcb_vlm.geometry.tiling import generate_tiles  # noqa: E402


def _make_board_png(path: pathlib.Path, w: int = 1800, h: int = 1200) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[100 : h - 100, 120 : w - 120] = [40, 115, 68]
    # Add target boxes well inside the clamped edge and inside inner tiles
    arr[250:390, 320:480] = [210, 180, 50]
    arr[860:980, 1420:1580] = [50, 180, 210]
    Image.fromarray(arr).save(path)
    return path.read_bytes()


def _http_req(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_r5_tile768_preset_resolution_and_model_restrictions(tmp_path: pathlib.Path) -> None:
    """Verify Round 5 768px preset applies solely to qwen3, preserves R4 controls, and leaves 1024px intact."""
    r4_q3 = resolve_model_preset_spec("qwen3", "r4_highres")
    assert r4_q3["tile_size"] == 1024
    assert r4_q3["qwen_max_pixels"] == 589824

    for p_name in ("r5_tile768", "r5_qwen3_greedy_maxpx589824_tile768"):
        r5_q3 = resolve_model_preset_spec("qwen3", p_name)
        assert r5_q3["preset_group"] == "r5_tile768"
        assert r5_q3["preset_id"] == "r5_qwen3_greedy_maxpx589824_tile768"
        assert r5_q3["tile_size"] == 768
        assert r5_q3["tile_overlap_ratio"] == pytest.approx(0.20)
        assert r5_q3["qwen_min_pixels"] == 65536
        assert r5_q3["qwen_max_pixels"] == 589824
        assert r5_q3["coordinate_protocol"] == "xyxy_norm1000"
        assert r5_q3["prompt_sha256"] == FROZEN_DETECTION_PROMPT_SHA256
        assert r5_q3["explicit_do_sample_false"] is True

        for other_mk in ("qwen2", "deepseek", "yolo11n", "yolo11n_qwen3"):
            with pytest.raises(DemoError):
                resolve_model_preset_spec(other_mk, p_name)

    img_bytes = _make_board_png(tmp_path / "b_768.png", 1800, 1200)
    svc = LiveDemoService(workspace_root=tmp_path / "ws", job_runner=lambda *a: None)
    up = svc.register_upload(raw_bytes=img_bytes, raw_filename="b_768.png")
    meta = json.loads((svc.uploads_dir / up["upload_id"] / "upload_meta.json").read_text(encoding="utf-8"))
    orig_tile_count = meta["tile_count"]

    plan_r5 = svc._build_single_job_plan(
        job_id="j_r5_q3",
        meta=meta,
        model_key="qwen3",
        mode="T",
        preset="r5_qwen3_greedy_maxpx589824_tile768",
    )
    meta_after = json.loads((svc.uploads_dir / up["upload_id"] / "upload_meta.json").read_text(encoding="utf-8"))
    assert meta_after["tile_count"] == orig_tile_count  # 1024px metadata never clobbered
    assert plan_r5["inference_config"]["tile_size"] == 768
    assert plan_r5["inference_config"]["tile_overlap_ratio"] == pytest.approx(0.20)
    assert plan_r5["inference_config"]["qwen_min_pixels"] == 65536
    assert plan_r5["inference_config"]["qwen_max_pixels"] == 589824
    expected_tiles = generate_tiles(meta["roi_width"], meta["roi_height"], tile_size=768, overlap_ratio=0.20)
    assert len(plan_r5["queues"]["qwen3"]) == len(expected_tiles)
    assert all(q["width"] == 768 and q["height"] == 768 for q in plan_r5["queues"]["qwen3"])


def test_r5_tile768_s20_geometry_and_synthetic_global_mapping() -> None:
    """Verify S20 ROI (4036x3061) produces exact 35 flushed 768x768 tiles and maps synthetic boxes accurately."""
    tiles = generate_tiles(4036, 3061, tile_size=768, overlap_ratio=0.20)
    assert len(tiles) == 35
    assert all(t["width"] == 768 and t["height"] == 768 for t in tiles)
    assert max(t["bbox"][2] for t in tiles) == 4036
    assert max(t["bbox"][3] for t in tiles) == 3061
    # Check clamped edge tile (tile 34 at bottom-right corner)
    assert tiles[-1]["offset_x"] == 4036 - 768
    assert tiles[-1]["offset_y"] == 3061 - 768

    # Synthetic [250, 250, 500, 500] inside 768x768 tile -> [192.0, 192.0, 384.0, 384.0] local px + offsets
    raw_rec = {
        "call_id": "qwen3_demo_syn768_T34",
        "board_id": "syn768",
        "mode": "T",
        "call_index": 34,
        "tile_id": 34,
        "ok": True,
        "orig_input_width": tiles[-1]["width"],
        "orig_input_height": tiles[-1]["height"],
        "roi_offset_x": 1906,
        "roi_offset_y": 1278,
        "tile_offset_x": tiles[-1]["offset_x"],
        "tile_offset_y": tiles[-1]["offset_y"],
        "max_new_tokens": 512,
        "generated_tokens": 22,
        "generated_token_ids": [151645],
        "coordinate_protocol": "xyxy_norm1000",
        "raw_output": json.dumps([{"bbox_2d": [250, 250, 500, 500], "label": "capacitor"}]) + "<|im_end|>",
    }
    recon = reconstruct_board_predictions_gt_free(
        board_id="syn768",
        image_id=1,
        orig_wh=(8256, 5504),
        raw_records=[raw_rec],
        mode="T",
        expected_call_ids=["qwen3_demo_syn768_T34"],
        coordinate_protocol="xyxy_norm1000",
    )
    assert recon["technical_complete"] is True
    assert recon["predictions"][0]["bbox_xyxy"] == [
        round(1906 + 3268 + 0.25 * 768, 2),
        round(1278 + 2293 + 0.25 * 768, 2),
        round(1906 + 3268 + 0.50 * 768, 2),
        round(1278 + 2293 + 0.50 * 768, 2),
    ]


def test_http_r5_tile768_end_to_end_export_and_gt_eval(tmp_path: pathlib.Path) -> None:
    """Exercise HTTP upload -> r5 768px qwen3 job/comparison -> GT-free export -> post-hoc GT scoring."""
    raw_bytes = _make_board_png(tmp_path / "small_r5_768.png", 1800, 1200)

    def fake_runner(job_spec: dict, _cancel_ev: threading.Event, svc: LiveDemoService) -> None:
        mk = job_spec["model_key"]
        jdir = svc.jobs_dir / job_spec["job_id"]
        run_dir = jdir / "base_runs" / f"run_{mk}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(run_dir / "gpu_gate.json", {"passed": True, "smi_free_mib": 7000.0})
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        assert plan["inference_config"]["tile_size"] == 768
        lines = []
        for q in plan["queues"][mk]:
            assert q["width"] == 768 and q["height"] == 768
            # Compute global box inside tile 0 that hits synthetic GT [320, 250, 480, 390]
            if q["tile_id"] == 0:
                gx0 = q["roi_offset_x"] + q["tile_offset_x"]
                gy0 = q["roi_offset_y"] + q["tile_offset_y"]
                nx1 = round((320.0 - gx0) / 768.0 * 1000.0)
                ny1 = round((250.0 - gy0) / 768.0 * 1000.0)
                nx2 = round((480.0 - gx0) / 768.0 * 1000.0)
                ny2 = round((390.0 - gy0) / 768.0 * 1000.0)
                raw_txt = json.dumps([{"bbox_2d": [nx1, ny1, nx2, ny2], "label": "capacitor"}]) + "<|im_end|>"
            else:
                raw_txt = "[]<|im_end|>"
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
                "planned_qwen_max_pixels": 589824,
                "processed_width_px": 768,
                "processed_height_px": 768,
                "processed_area_px": 589824,
                "image_grid_thw": [1, 48, 48],
                "merged_visual_tokens": 576,
                "max_new_tokens": 512,
                "generated_tokens": 18,
                "generated_token_ids": [151645],
                "coordinate_protocol": "xyxy_norm1000",
                "prompt_sha256": plan["inference_config"]["prompt_sha256"],
                "effective_generation_probe": {
                    "declared_do_sample": False,
                    "explicit_do_sample_kwarg_passed": True,
                    "effective_do_sample": False,
                    "effective_generation_mode": "GenerationMode.GREEDY_SEARCH",
                },
                "gpu_generate_ms": 42.0,
                "cuda_max_reserved_mib": 2500.0,
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
        assert code == 200 and 'value="r5_qwen3_greedy_maxpx589824_tile768"' in html.decode("utf-8")

        c_up, _, b_up = _http_req(f"{base_url}/api/upload", method="POST", body=raw_bytes, headers={"X-Filename": "small_r5_768.png"})
        assert c_up == 200
        up = json.loads(b_up.decode("utf-8"))

        # Multi-model comparison with r5 single-model preset rejected
        c_bad_multi, _, _ = _http_req(
            f"{base_url}/api/comparisons",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "model_keys": ["qwen2", "qwen3"], "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_bad_multi == 400

        c_cmp, _, b_cmp = _http_req(
            f"{base_url}/api/comparisons",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "model_keys": ["qwen3"], "mode": "T", "preset": "r5_qwen3_greedy_maxpx589824_tile768"}).encode("utf-8"),
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
        slot = done["slots"][0]
        assert slot["preset"] == "r5_qwen3_greedy_maxpx589824_tile768"
        assert slot["planned_visual_budget"]["tile_size"] == 768
        assert slot["tile_count"] > up["tile_count"]  # 768px produces more tiles than 1024px on 1800x1200 ROI
        assert slot["kept_predictions_count"] == 1
        jid = slot["job_id"]

        c_gt, _, b_gt = _http_req(
            f"{base_url}/api/jobs/{jid}/evaluate_gt",
            method="POST",
            body=json.dumps({
                "coco_gt": {
                    "images": [{"id": 1, "file_name": "small_r5_768.png", "width": 1800, "height": 1200}],
                    "categories": [
                        {"id": 1, "name": "capacitor"}, {"id": 2, "name": "resistor"}, {"id": 3, "name": "inductor"},
                        {"id": 4, "name": "transistor"}, {"id": 5, "name": "diode"}, {"id": 6, "name": "ic"},
                    ],
                    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [320, 250, 160, 140], "area": 22400}],
                },
                "target_image_id": 1,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_gt == 200
        ev = json.loads(b_gt.decode("utf-8"))["post_hoc_eval"]["metrics"]
        assert ev["tp"] == 1 and ev["fp"] == 0 and ev["fn"] == 0
    finally:
        svc.shutdown()
        srv.shutdown()
        srv.server_close()
