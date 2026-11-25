"""Round 9 tests: NF4 Base vs QLoRA Adapter presets, metadata propagation, single-class UI GT scope, and regressions."""

from __future__ import annotations

import hashlib
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


def _make_board_png(path: pathlib.Path, w: int = 1200, h: int = 900) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[40 : h - 40, 60 : w - 60] = [170, 50, 50]
    arr[140:230, 180:290] = [210, 180, 50]
    arr[320:410, 420:510] = [50, 180, 210]
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


def _make_mock_runner(obs_records: list[dict] | None = None):
    def _runner(job_spec, cancel_ev, svc):
        jdir = svc.jobs_dir / job_spec["job_id"]
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        if obs_records is not None:
            obs_records.append({"job_spec": job_spec, "plan": plan})
        run_dir = jdir / "base_runs" / f"run_{job_spec['model_key']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        inst = job_spec.get("instruction_spec")
        for q in plan["queues"][job_spec["model_key"]]:
            if cancel_ev.is_set():
                return
            if int(q["call_index"]) != 0:
                objs = []
            elif inst:
                tgt = inst["target_class"]
                if tgt == "resistor":
                    objs = [{"bbox_2d": [100, 100, 220, 220], "label": "resistor"}]
                elif tgt == "capacitor":
                    objs = [{"bbox_2d": [300, 300, 420, 420], "label": "capacitor"}]
                else:
                    objs = []
            else:
                objs = [
                    {"bbox_2d": [100, 100, 220, 220], "label": "resistor"},
                    {"bbox_2d": [300, 300, 420, 420], "label": "capacitor"},
                ]
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
                "generated_tokens": 12,
                "generated_token_ids": [151645],
                "coordinate_protocol": "xyxy_norm1000",
                "prompt_sha256": plan["inference_config"]["prompt_sha256"],
                "effective_generation_probe": {
                    "declared_do_sample": False,
                    "explicit_do_sample_kwarg_passed": True,
                    "effective_do_sample": False,
                    "effective_generation_mode": "GenerationMode.GREEDY_SEARCH",
                },
                "gpu_generate_ms": 15.0,
                "cuda_max_reserved_mib": 2100.0,
                "quantization": job_spec.get("quantization", "bf16"),
                "adapter_id": job_spec.get("adapter_id"),
                "adapter_sha256": job_spec.get("adapter_sha256"),
                "raw_output": json.dumps(objs, ensure_ascii=False) + "<|im_end|>",
            }))
        (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return _runner


def test_r9_preset_resolution_and_model_guard() -> None:
    b = resolve_model_preset_spec("qwen3", "r9_qwen3_nf4_base_tile768")
    assert b["preset_id"] == "r9_qwen3_nf4_base_tile768"
    assert b["quantization"] == "nf4"
    assert b["use_configured_r9_adapter"] is False
    assert b["adapter_id"] == "base_nf4"
    assert b["tile_size"] == 768
    assert b["qwen_max_pixels"] == 589824
    assert b["explicit_do_sample_false"] is True

    a = resolve_model_preset_spec("qwen3", "r9_nf4_adapter")
    assert a["preset_id"] == "r9_qwen3_nf4_adapter_tile768"
    assert a["quantization"] == "nf4"
    assert a["use_configured_r9_adapter"] is True
    assert a["tile_size"] == 768

    for other_mk in ("qwen2", "deepseek", "yolo11n", "yolo11n_qwen3"):
        for rp in ("r9_qwen3_nf4_base_tile768", "r9_qwen3_nf4_adapter_tile768"):
            with pytest.raises(DemoError) as exc_info:
                resolve_model_preset_spec(other_mk, rp)
            assert exc_info.value.status_code == 400


def test_r9_missing_and_invalid_adapter_return_400_http(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "b.png")
    bad_dir = tmp_path / "incomplete_adapter"
    bad_dir.mkdir(parents=True, exist_ok=True)  # missing adapter_config.json

    for idx, rt_cfg in enumerate((
        {"models": {"qwen3": {"local_dir": str(tmp_path)}}},
        {"models": {"qwen3": {"local_dir": str(tmp_path), "r9_adapter_dir": str(bad_dir)}}},
    )):
        rt_p = tmp_path / f"rt_{idx}.yaml"
        rt_p.write_text(yaml.safe_dump(rt_cfg, sort_keys=False), encoding="utf-8")
        svc = LiveDemoService(workspace_root=tmp_path / f"ws_{time.time_ns()}", runtime_config_path=rt_p, job_runner=_make_mock_runner())
        srv = fics_final_demo.create_demo_server(workspace_root=svc.workspace_root, service=svc, host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            st_u, _, raw_u = _http_req(f"{base}/api/upload", method="POST", body=png_bytes, headers={"X-Filename": "board.png"})
            assert st_u == 200
            uid = json.loads(raw_u.decode("utf-8"))["upload_id"]
            st_j, _, raw_j = _http_req(
                f"{base}/api/jobs",
                method="POST",
                body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r9_qwen3_nf4_adapter_tile768"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            assert st_j == 400
            err = json.loads(raw_j.decode("utf-8"))
            assert "adapter" in err["error"].lower() or "适配器" in err["error"]
        finally:
            srv.shutdown()
            srv.server_close()


def test_r9_full_http_propagation_single_class_gt_and_ui_html(tmp_path: pathlib.Path) -> None:
    img_path = tmp_path / "board_r9.png"
    png_bytes = _make_board_png(img_path)
    img_sha = hashlib.sha256(png_bytes).hexdigest()

    adapter_dir = tmp_path / "adapter_step48"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 16}), encoding="utf-8")
    weights_bytes = b"mock-safetensors-r9"
    (adapter_dir / "adapter_model.safetensors").write_bytes(weights_bytes)
    expected_adapter_sha = hashlib.sha256(weights_bytes).hexdigest()

    rt_cfg = {
        "models": {
            "qwen3": {
                "local_dir": str(tmp_path),
                "r9_adapter_dir": str(adapter_dir),
                "r9_adapter_id": "r9_sft_step48_s3best",
            }
        }
    }
    rt_p = tmp_path / "rt_ok.yaml"
    rt_p.write_text(yaml.safe_dump(rt_cfg, sort_keys=False), encoding="utf-8")
    captured: list[dict] = []
    svc = LiveDemoService(workspace_root=tmp_path / "ws_ok", runtime_config_path=rt_p, job_runner=_make_mock_runner(captured))
    srv = fics_final_demo.create_demo_server(workspace_root=svc.workspace_root, service=svc, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # Check frontend HTML/JS contracts
        st_h, _, body_h = _http_req(f"{base}/")
        assert st_h == 200
        html = body_h.decode("utf-8")
        assert 'value="r9_qwen3_nf4_base_tile768"' in html
        assert 'value="r9_qwen3_nf4_adapter_tile768"' in html
        assert "量化 / 适配器" in html
        assert "目标类 GT (TP/FP/FN" in html
        assert "同板全 6 类参考口径" in html
        assert "phe.single_class_eval" in html

        st_u, _, raw_u = _http_req(f"{base}/api/upload", method="POST", body=png_bytes, headers={"X-Filename": "board_r9.png"})
        assert st_u == 200
        up = json.loads(raw_u.decode("utf-8"))
        uid = up["upload_id"]

        # 1. Run NF4 Adapter single-class query (CH + EN synonymous) and NF4 Base
        st_b, _, raw_b = _http_req(
            f"{base}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": uid, "model_key": "qwen3", "mode": "T", "preset": "r9_qwen3_nf4_base_tile768"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_b == 202
        jid_base = json.loads(raw_b.decode("utf-8"))["job_id"]
        js_base = _wait_job(base, jid_base)
        assert js_base["status"] == "completed"
        assert js_base["quantization"] == "nf4"
        assert js_base["adapter_id"] == "base_nf4"

        st_a, _, raw_a = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({
                "upload_id": uid,
                "model_key": "qwen3",
                "mode": "T",
                "preset": "r9_qwen3_nf4_adapter_tile768",
                "user_instruction": "只找电阻并计数",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_a == 202
        jid_ad_res = json.loads(raw_a.decode("utf-8"))["job_id"]
        js_ad = _wait_job(base, jid_ad_res)
        assert js_ad["status"] == "completed"
        assert js_ad["quantization"] == "nf4"
        assert js_ad["adapter_id"] == "r9_sft_step48_s3best"

        st_en, _, raw_en = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({
                "upload_id": uid,
                "model_key": "qwen3",
                "mode": "T",
                "preset": "r9_qwen3_nf4_adapter_tile768",
                "user_instruction": "Find only capacitors and count them",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_en == 202
        jid_en = json.loads(raw_en.decode("utf-8"))["job_id"]
        assert _wait_job(base, jid_en)["status"] == "completed"

        # Unsupported spatial/exclusion/out-of-scope HTTP 400 rejections
        for bad_q in ("只找左上角的电阻", "找除了电阻之外的元件", "只找连接器并计数"):
            st_bad, _, raw_bad = _http_req(
                f"{base}/api/jobs/instruct",
                method="POST",
                body=json.dumps({
                    "upload_id": uid,
                    "model_key": "qwen3",
                    "mode": "T",
                    "preset": "r9_qwen3_nf4_adapter_tile768",
                    "user_instruction": bad_q,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            assert st_bad == 400
            err_doc = json.loads(raw_bad.decode("utf-8"))
            assert err_doc["error"]

        # Verify plan/spec/result propagation for Adapter job
        assert len(captured) == 3
        ad_cap = captured[1]
        assert ad_cap["plan"]["inference_config"]["quantization"] == "nf4"
        assert ad_cap["plan"]["inference_config"]["adapter_id"] == "r9_sft_step48_s3best"
        assert ad_cap["plan"]["inference_config"]["adapter_path"] == str(adapter_dir)
        assert ad_cap["job_spec"]["adapter_sha256"] == expected_adapter_sha

        _, _, res_raw = _http_req(f"{base}/api/jobs/{jid_ad_res}/result")
        res_ad = json.loads(res_raw.decode("utf-8"))
        assert res_ad["quantization"] == "nf4"
        assert res_ad["adapter_id"] == "r9_sft_step48_s3best"
        assert res_ad["adapter_sha256"] == expected_adapter_sha
        assert res_ad["instruction_spec"]["target_class"] == "resistor"
        assert "resistor" in res_ad["instruction_spec"]["normalized_prompt"]
        roi_ox = int(res_ad["roi"]["offset_x"])
        roi_oy = int(res_ad["roi"]["offset_y"])

        # Submit matching COCO GT and verify class_agnostic_eval & single_class_eval
        tile_w, tile_h = 768, 768
        rx1 = roi_ox + int(round(100 * tile_w / 1000))
        ry1 = roi_oy + int(round(100 * tile_h / 1000))
        rx2 = roi_ox + int(round(220 * tile_w / 1000))
        ry2 = roi_oy + int(round(220 * tile_h / 1000))
        cx1 = roi_ox + int(round(300 * tile_w / 1000))
        cy1 = roi_oy + int(round(300 * tile_h / 1000))
        cx2 = roi_ox + int(round(420 * tile_w / 1000))
        cy2 = roi_oy + int(round(420 * tile_h / 1000))
        coco_gt = {
            "images": [{"id": 77, "file_name": "board_r9.png", "width": 1200, "height": 900, "sha256": img_sha}],
            "categories": [
                {"id": 1, "name": "capacitor"},
                {"id": 2, "name": "resistor"},
                {"id": 3, "name": "inductor"},
                {"id": 4, "name": "transistor"},
                {"id": 5, "name": "diode"},
                {"id": 6, "name": "ic"},
            ],
            "annotations": [
                {"id": 1, "image_id": 77, "category_id": 2, "bbox": [rx1, ry1, rx2 - rx1, ry2 - ry1]},
                {"id": 2, "image_id": 77, "category_id": 1, "bbox": [cx1, cy1, cx2 - cx1, cy2 - cy1]},
            ],
        }
        st_gt, _, raw_gt = _http_req(
            f"{base}/api/jobs/{jid_ad_res}/evaluate_gt",
            method="POST",
            body=json.dumps({"coco_gt": coco_gt}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_gt == 200
        ev_res = json.loads(raw_gt.decode("utf-8"))
        phe = ev_res["post_hoc_eval"]
        assert phe["class_agnostic_eval"]["tp"] == 1
        assert phe["class_agnostic_eval"]["fn"] == 1
        sce = phe["single_class_eval"]
        assert sce["target_class"] == "resistor"
        assert sce["target_class_cn"] == "电阻"
        assert sce["gt_count"] == 1
        assert sce["pred_count"] == 1
        assert sce["tp"] == 1 and sce["fp"] == 0 and sce["fn"] == 0
        assert phe["metrics"]["total_gt"] == 2

        st_md, _, raw_md = _http_req(f"{base}/api/jobs/{jid_ad_res}/export/summary_md")
        assert st_md == 200
        md_txt = raw_md.decode("utf-8")
        assert "r9_sft_step48_s3best" in md_txt
        assert "nf4" in md_txt
        assert "目标单类事后 GT 口径 (`resistor`" in md_txt
        assert "同板全 6 类参考口径 (GT=2" in md_txt
        assert "类无关定位基准" in md_txt
    finally:
        srv.shutdown()
        srv.server_close()


def test_r9_legacy_w_and_t_presets_unchanged(tmp_path: pathlib.Path) -> None:
    b_q3 = resolve_model_preset_spec("qwen3", "baseline")
    assert b_q3["tile_size"] == 1024
    assert b_q3.get("quantization") is None
    r5 = resolve_model_preset_spec("qwen3", "r5_qwen3_greedy_maxpx589824_tile768")
    assert r5["tile_size"] == 768
    r7 = resolve_model_preset_spec("qwen3", "r7_qwen3_tile768_dual_crop_consensus")
    assert r7["verifier_policy"] == "B_other_dual_crop_consensus"
