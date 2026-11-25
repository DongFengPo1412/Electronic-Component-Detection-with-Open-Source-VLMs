"""Round 11 unit & integration tests: spatial instruction control & process-group global deadline."""

from __future__ import annotations

import json
import pathlib
import subprocess
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
from pcb_vlm.evaluation.final_demo_service import LiveDemoService  # noqa: E402
from pcb_vlm.evaluation.language_instruction import (  # noqa: E402
    build_single_class_detection_prompt,
    parse_pcb_language_instruction,
)
from pcb_vlm.evaluation.spatial_experiment import (  # noqa: E402
    aggregate_spatial_scores,
    audit_spatial_predictions,
    build_spatial_detection_prompt,
    classify_box_half,
    evaluate_spatial_promotion_gates,
    parse_spatial_pcb_instruction,
    run_with_process_group_deadline,
    score_spatial_arm_against_gt,
)


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


def test_spatial_prompts_and_parser() -> None:
    p_un, sha_un = build_spatial_detection_prompt("resistor", "unconstrained")
    p_un_r8, sha_un_r8 = build_single_class_detection_prompt("resistor", prompt_variant="r8_base")
    assert p_un == p_un_r8 and sha_un == sha_un_r8

    p_l, sha_l = build_spatial_detection_prompt("resistor", "left_half")
    p_r, sha_r = build_spatial_detection_prompt("resistor", "right_half")
    assert len({sha_un, sha_l, sha_r}) == 3
    assert "LEFT HALF" in p_l and "x < 500" in p_l and "full input image" in p_l
    assert "RIGHT HALF" in p_r and "x >= 500" in p_r and "full input image" in p_r

    ok_l = parse_spatial_pcb_instruction("只找电阻，限图像左半边并计数")
    assert ok_l["ok"] is True
    assert ok_l["target_class"] == "resistor"
    assert ok_l["region_mode"] == "left_half"
    assert ok_l["execution_source"] == "direct_vlm_spatial"
    assert ok_l["input_cropped"] is False
    assert ok_l["cross_region_boxes_preserved"] is True
    assert ok_l["prompt_sha256"] == sha_l

    ok_r = parse_spatial_pcb_instruction("find capacitors in the right half and count")
    assert ok_r["ok"] is True and ok_r["target_class"] == "capacitor" and ok_r["region_mode"] == "right_half"

    # Existing mainline parser MUST continue rejecting spatial queries!
    main_rej = parse_pcb_language_instruction("只找电阻，限图像左半边并计数")
    assert main_rej["ok"] is False and main_rej["error_code"] == "spatial_restriction_unsupported"

    # Experimental bounded parser rejects unsupported qualifiers like quadrants
    quad_rej = parse_spatial_pcb_instruction("只找左上角电阻")
    assert quad_rej["ok"] is False and quad_rej["error_code"] == "spatial_qualifier_out_of_bounded_protocol"


def test_classify_box_half_audit_and_scoring_preserves_cross_region() -> None:
    # Exact boundary x = 384 on 768px image: < 384 left_half, >= 384 right_half
    assert classify_box_half([100, 100, 200, 200], 768) == "left_half"
    assert classify_box_half([380, 100, 387, 200], 768) == "left_half"   # cx = 383.5 < 384
    assert classify_box_half([384, 100, 384, 200], 768) == "right_half"  # cx = 384.0 >= 384
    assert classify_box_half([450, 100, 600, 200], 768) == "right_half"

    preds = [
        {"bbox_xyxy": [100, 100, 180, 180], "label": "resistor"},  # left TP
        {"bbox_xyxy": [500, 100, 580, 180], "label": "resistor"},  # cross-region violation for left_half
    ]
    gts = [{"bbox_xyxy": [102, 102, 178, 178]}]
    scored = score_spatial_arm_against_gt(
        same_class_preds=preds,
        unauthorized_preds=[{"label": "capacitor", "bbox_xyxy": [200, 200, 240, 240]}],
        gt_boxes_for_class=gts,
        image_width=768,
        region_mode="left_half",
    )
    assert scored["raw_same_class_pred_count"] == 2
    assert scored["in_region_pred_count"] == 1
    assert scored["cross_region_violation_count"] == 1
    assert scored["cross_region_violation_rate"] == 0.5
    assert scored["tp"] == 1 and scored["in_region_fp"] == 0 and scored["strict_fp"] == 1 and scored["fn"] == 0
    assert scored["in_region_f1"] == 1.0 and scored["strict_f1"] < 1.0
    assert scored["unauthorized_class_count"] == 1

    # Half-absent case (0 GT on right_half while left_half has GT)
    absent = score_spatial_arm_against_gt(
        same_class_preds=[],
        unauthorized_preds=[],
        gt_boxes_for_class=gts,
        image_width=768,
        region_mode="right_half",
    )
    assert absent["is_half_absent"] is True
    assert absent["strict_valid_empty"] is True and absent["half_absent_fp_total"] == 0

    # Promotion gates reject when strict_f1 or violation rate fails
    pooled = aggregate_spatial_scores([scored, absent])
    gates = evaluate_spatial_promotion_gates(
        direct_left_positive_tp=1,
        direct_right_positive_tp=1,
        direct_pooled=pooled,
        cpu_pooled={"tp": 1, "strict_f1": 1.0},
        per_eval_pairs=[(scored, scored)],
        distinct_fresh_calls_verified=True,
        auditable_cross_region_logged=True,
    )
    assert gates["admission_gates_passed"] is True
    assert gates["promote_spatial_ui"] is False
    assert gates["recommended_ui_action"] == "keep_main_single_class_scope_and_isolation"


def test_http_isolated_experimental_spatial_endpoint_preserves_main_contract(tmp_path: pathlib.Path) -> None:
    img_p = tmp_path / "tile_768.png"
    arr = np.full((768, 768, 3), 255, dtype=np.uint8)
    arr[20:748, 20:748] = [170, 50, 50]
    arr[100:200, 100:200] = [210, 180, 50]
    arr[100:200, 520:620] = [50, 180, 210]
    Image.fromarray(arr).save(img_p)

    def _mock_runner(job_spec: dict, cancel_ev, svc_ref: LiveDemoService) -> None:
        jdir = svc_ref.jobs_dir / job_spec["job_id"]
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        run_dir = jdir / "base_runs" / "run_qwen3"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "gpu_gate.json").write_text(json.dumps({"passed": True, "smi_free_mib": 7000.0}), encoding="utf-8")
        q = plan["queues"]["qwen3"][0]
        # Return 1 left-half resistor AND 1 right-half resistor (cross-region!) to confirm preservation
        objs = [
            {"bbox_2d": [100, 100, 200, 200], "label": "resistor"},
            {"bbox_2d": [700, 100, 800, 200], "label": "resistor"},
        ]
        (run_dir / "raw_outputs_all.jsonl").write_text(
            json.dumps({
                "call_id": q["call_id"], "board_id": q["board_id"], "tag": q["tag"], "mode": q["mode"],
                "image_id": q["image_id"], "call_index": q["call_index"], "tile_id": q["tile_id"], "ok": True,
                "orig_input_width": 768, "orig_input_height": 768,
                "roi_offset_x": q["roi_offset_x"], "roi_offset_y": q["roi_offset_y"],
                "tile_offset_x": q["tile_offset_x"], "tile_offset_y": q["tile_offset_y"],
                "planned_qwen_min_pixels": 65536, "planned_qwen_max_pixels": 589824,
                "max_new_tokens": 512, "generated_tokens": 10, "generated_token_ids": [151645],
                "coordinate_protocol": "xyxy_norm1000", "prompt_sha256": plan["inference_config"]["prompt_sha256"],
                "gpu_generate_ms": 12.0, "cuda_max_reserved_mib": 2000.0,
                "raw_output": json.dumps(objs) + "<|im_end|>",
            }) + "\n",
            encoding="utf-8",
        )

    svc = LiveDemoService(workspace_root=tmp_path / "ws_r11", job_runner=_mock_runner)
    srv = fics_final_demo.create_demo_server(workspace_root=tmp_path / "ws_r11", service=svc, host="127.0.0.1", port=0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        st_u, _, raw_u = _http_req(f"{base}/api/upload", method="POST", body=img_p.read_bytes(), headers={"X-Filename": "tile_768.png"})
        assert st_u == 200
        up_id = json.loads(raw_u.decode("utf-8"))["upload_id"]

        # 1. Existing /api/jobs/instruct MUST still reject spatial instructions with HTTP 400
        payload = json.dumps({
            "upload_id": up_id, "model_key": "qwen3", "mode": "T",
            "preset": "r5_qwen3_greedy_maxpx589824_tile768",
            "user_instruction": "只找电阻，限图像左半边并计数",
        }).encode("utf-8")
        st_main, _, raw_main = _http_req(f"{base}/api/jobs/instruct", method="POST", body=payload, headers={"Content-Type": "application/json"})
        assert st_main == 400
        assert "spatial_restriction_unsupported" in raw_main.decode("utf-8")

        # 2. Isolated /api/experimental/spatial_job accepts it & preserves cross-region predictions
        st_sp, _, raw_sp = _http_req(f"{base}/api/experimental/spatial_job", method="POST", body=payload, headers={"Content-Type": "application/json"})
        assert st_sp == 202
        job_id = json.loads(raw_sp.decode("utf-8"))["job_id"]
        js = _wait_job(base, job_id)
        assert js["status"] == "completed"
        st_res, _, raw_res = _http_req(f"{base}/api/jobs/{job_id}/result")
        assert st_res == 200
        res = json.loads(raw_res.decode("utf-8"))
        assert res["instruction_spec"]["experimental_spatial"] is True
        assert res["instruction_spec"]["input_cropped"] is False
        assert res["instruction_spec"]["region_mode"] == "left_half"
        assert res["kept_predictions_count"] == 2  # BOTH boxes preserved!
        assert res["spatial_compliance"]["in_region_count"] == 1
        assert res["spatial_compliance"]["cross_region_violation_count"] == 1
    finally:
        srv.shutdown()
        srv.server_close()


def test_run_with_process_group_deadline_kills_process_tree(tmp_path: pathlib.Path) -> None:
    pid_file = tmp_path / "child_pids.json"
    parent_script = tmp_path / "spawn_tree.py"
    parent_script.write_text(
        "import json, pathlib, subprocess, sys, time\n"
        f"p = pathlib.Path(r'{pid_file}')\n"
        "grand = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "p.write_text(json.dumps({'grand_pid': grand.pid}), encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + 1.5
    rec = run_with_process_group_deadline(
        [sys.executable, str(parent_script)],
        global_deadline_epoch=deadline,
        stdout_path=tmp_path / "out.log",
        stderr_path=tmp_path / "err.log",
        poll_interval_s=0.1,
    )
    assert rec["timed_out_global_deadline"] is True
    assert rec["elapsed_seconds"] < 8.0
    assert pid_file.is_file()
    grand_pid = json.loads(pid_file.read_text(encoding="utf-8"))["grand_pid"]
    # Confirm grandchild is dead as well (Windows taskkill /T or killpg)
    time.sleep(0.2)
    chk = subprocess.run(["tasklist", "/FI", f"PID eq {grand_pid}", "/NH"], capture_output=True, text=True, check=False)
    assert str(grand_pid) not in (chk.stdout or "")
