"""Round 12 tests: cold-start single-model Qwen3 BF16 R5 T+R8 defaults & 3-model fair baseline isolation."""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for p in (str(_SRC_ROOT), str(_SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fics_final_demo  # noqa: E402
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402


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


def _wait_cmp(base_url: str, cmp_id: str, timeout_iters: int = 200) -> dict:
    for _ in range(timeout_iters):
        _, _, raw = _http_req(f"{base_url}/api/comparisons/{cmp_id}")
        js = json.loads(raw.decode("utf-8"))
        if js["status"] in ("completed", "failed", "incomplete", "cancelled"):
            return js
        time.sleep(0.03)
    raise AssertionError(f"Comparison {cmp_id} did not terminate in time")


def test_cold_start_html_config_and_two_entry_isolation(tmp_path: pathlib.Path) -> None:
    rt_cfg = tmp_path / "rt.yaml"
    rt_cfg.write_text(
        "default_model: qwen3\n"
        "default_mode: T\n"
        "default_preset: r5_qwen3_greedy_maxpx589824_tile768\n"
        "default_instruction_prompt_variant: r8_base\n"
        "three_model_compare_default_mode: W\n"
        "three_model_compare_default_preset: baseline\n",
        encoding="utf-8",
    )

    img_p = tmp_path / "board_t15.png"
    arr = np.full((768, 768, 3), 255, dtype=np.uint8)
    arr[20:748, 20:748] = [170, 50, 50]
    arr[100:220, 120:260] = [210, 180, 50]
    arr[300:420, 420:540] = [50, 180, 210]
    Image.fromarray(arr).save(img_p)

    captured_plans: dict[str, dict] = {}

    def _mock_runner(job_spec: dict, cancel_ev, svc_ref: LiveDemoService) -> None:
        jdir = svc_ref.jobs_dir / job_spec["job_id"]
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        captured_plans[job_spec["job_id"]] = plan
        mk = job_spec["model_key"]
        run_dir = jdir / "base_runs" / f"run_{mk}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "gpu_gate.json").write_text(json.dumps({"passed": True, "smi_free_mib": 7000.0}), encoding="utf-8")
        q = plan["queues"][mk][0]
        (run_dir / "raw_outputs_all.jsonl").write_text(
            json.dumps({
                "call_id": q["call_id"], "board_id": q["board_id"], "tag": q["tag"], "mode": q["mode"],
                "image_id": q["image_id"], "call_index": q["call_index"], "tile_id": q["tile_id"], "ok": True,
                "orig_input_width": q["width"], "orig_input_height": q["height"],
                "roi_offset_x": q["roi_offset_x"], "roi_offset_y": q["roi_offset_y"],
                "tile_offset_x": q["tile_offset_x"], "tile_offset_y": q["tile_offset_y"],
                "max_new_tokens": 512, "generated_tokens": 5, "generated_token_ids": [151645],
                "coordinate_protocol": "xyxy_norm1000", "prompt_sha256": plan["inference_config"]["prompt_sha256"],
                "gpu_generate_ms": 10.0, "cuda_max_reserved_mib": 2000.0,
                "raw_output": "[]<|im_end|>",
            }) + "\n",
            encoding="utf-8",
        )

    svc = LiveDemoService(workspace_root=tmp_path / "ws_r12", runtime_config_path=rt_cfg, job_runner=_mock_runner)
    srv = fics_final_demo.create_demo_server(workspace_root=tmp_path / "ws_r12", service=svc, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # 1. Check cold-start HTML defaults
        st_h, _, raw_h = _http_req(f"{base}/")
        assert st_h == 200
        html = raw_h.decode("utf-8")
        assert '<option value="T" selected>' in html
        assert '<option value="r5_qwen3_greedy_maxpx589824_tile768" selected>' in html
        assert 'id="btnEntrySingleDefense"' in html
        assert 'id="btnEntryThreeModelCompare"' in html
        assert '1. 实时 GPU 推理' in html and '2. 已有六类结果筛选（CPU）' in html and '3. 事后 COCO 标注评价' in html and '4. 历史只读回放' in html

        # 2. Check GET /api/config reflects runtime_config
        st_c, _, raw_c = _http_req(f"{base}/api/config")
        assert st_c == 200
        cfg = json.loads(raw_c.decode("utf-8"))
        assert cfg["single_model_defense"]["model_key"] == "qwen3"
        assert cfg["single_model_defense"]["mode"] == "T"
        assert cfg["single_model_defense"]["preset"] == "r5_qwen3_greedy_maxpx589824_tile768"
        assert cfg["three_model_compare"]["mode"] == "W"
        assert cfg["three_model_compare"]["preset"] == "baseline"

        # 3. Upload board image once
        st_u, _, raw_u = _http_req(f"{base}/api/upload", method="POST", body=img_p.read_bytes(), headers={"X-Filename": "board_t15.png"})
        assert st_u == 200
        up_id = json.loads(raw_u.decode("utf-8"))["upload_id"]

        # 4. Omitting preset in Entry A picks up Qwen3 BF16 R5 T automatically
        st_j1, _, raw_j1 = _http_req(
            f"{base}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": up_id, "model_key": "qwen3", "mode": "T"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_j1 == 202
        j1_id = json.loads(raw_j1.decode("utf-8"))["job_id"]
        assert _wait_job(base, j1_id)["status"] == "completed"
        p1 = captured_plans[j1_id]["inference_config"]
        assert p1["preset_id"] == "r5_qwen3_greedy_maxpx589824_tile768"
        assert p1["tile_size"] == 768 and p1["qwen_max_pixels"] == 589824
        assert p1["explicit_do_sample_false"] is True and p1["quantization"] == "bf16" and p1["max_new_tokens"] == 512

        # 5. Single-class instruction also inherits R5 T + R8 prompt when preset omitted
        st_j2, _, raw_j2 = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({"upload_id": up_id, "model_key": "qwen3", "mode": "T", "user_instruction": "只找电阻并计数"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_j2 == 202
        j2_id = json.loads(raw_j2.decode("utf-8"))["job_id"]
        assert _wait_job(base, j2_id)["status"] == "completed"
        p2 = captured_plans[j2_id]["inference_config"]
        assert p2["preset_id"] == "r5_qwen3_greedy_maxpx589824_tile768"
        assert p2["instruction_prompt_variant"] == "r8_base"
        assert p2["prompt_sha256"] != p1["prompt_sha256"]

        # 6. Spatial restriction STILL rejected on /api/jobs/instruct (HTTP 400, 0 jobs created)
        before_jobs = len(list((tmp_path / "ws_r12" / "jobs").iterdir()))
        st_sp, _, raw_sp = _http_req(
            f"{base}/api/jobs/instruct",
            method="POST",
            body=json.dumps({"upload_id": up_id, "model_key": "qwen3", "mode": "T", "user_instruction": "只找电阻，限图像左半边并计数"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_sp == 400 and "spatial_restriction_unsupported" in raw_sp.decode("utf-8")
        assert len(list((tmp_path / "ws_r12" / "jobs").iterdir())) == before_jobs

        # 7. Three-model comparison refuses R5 preset (HTTP 400) and succeeds with baseline
        st_bad_cmp, _, _ = _http_req(
            f"{base}/api/comparisons",
            method="POST",
            body=json.dumps({
                "upload_id": up_id, "model_keys": ["qwen2", "deepseek", "qwen3"],
                "mode": "W", "preset": "r5_qwen3_greedy_maxpx589824_tile768",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_bad_cmp == 400

        st_cmp, _, raw_cmp = _http_req(
            f"{base}/api/comparisons",
            method="POST",
            body=json.dumps({
                "upload_id": up_id, "model_keys": ["qwen2", "deepseek", "qwen3"],
                "mode": "W", "preset": "baseline",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert st_cmp == 202
        cmp_id = json.loads(raw_cmp.decode("utf-8"))["comparison_id"]
        cmp_fin = _wait_cmp(base, cmp_id)
        assert cmp_fin["status"] == "completed"
        assert len(cmp_fin["slots"]) == 3
        for slot in cmp_fin["slots"]:
            cfg_m = captured_plans[slot["job_id"]]["inference_config"]
            assert cfg_m["preset_id"] == "baseline"
            assert cfg_m["tile_size"] == 1024 and cfg_m["qwen_max_pixels"] == 200704
    finally:
        srv.shutdown()
        srv.server_close()
