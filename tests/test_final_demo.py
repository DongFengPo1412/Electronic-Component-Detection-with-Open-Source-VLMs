"""Comprehensive CPU and fake-worker HTTP tests for the FICS Final Demo system."""

from __future__ import annotations

import hashlib
import json
import os
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
from pcb_vlm.evaluation.final_demo_service import (  # noqa: E402
    LiveDemoService,
    ResearchReplayStore,
    _write_json_atomic,
)


def _make_board_png(path: pathlib.Path, w: int = 1300, h: int = 1150) -> bytes:
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    yy, xx = np.mgrid[80 : h - 80, 90 : w - 90]
    arr[80 : h - 80, 90 : w - 90, 0] = (xx % 211).astype(np.uint8)
    arr[80 : h - 80, 90 : w - 90, 1] = (yy % 197).astype(np.uint8)
    arr[80 : h - 80, 90 : w - 90, 2] = 65
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


def _fake_runner_success(job_spec: dict, cancel_ev: threading.Event, svc: LiveDemoService) -> None:
    mk = job_spec["model_key"]
    base_mode = job_spec["base_mode"]
    jdir = svc.jobs_dir / job_spec["job_id"]
    run_dir = jdir / "base_runs" / f"run_{mk}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(run_dir / "gpu_gate.json", {"passed": True, "smi_free_mib": 6800.0})

    plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
    lines = []
    for idx, q in enumerate(plan["queues"][mk]):
        if cancel_ev.is_set():
            raise InterruptedError("cancelled")
        if base_mode == "W":
            raw_txt = json.dumps([
                {"bbox_2d": [100, 100, 220, 220], "label": "capacitor"},
                {"bbox_2d": [104, 104, 218, 218], "label": "capacitor"},
            ]) + "<|im_end|>"
        elif idx == 0:
            raw_txt = json.dumps([
                {"bbox_2d": [120, 120, 240, 240], "label": "resistor"},
                {"bbox_2d": [320, 320, 440, 440], "label": "ic"},
            ]) + "<|im_end|>"
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
            "max_new_tokens": 512,
            "generated_tokens": 12,
            "generated_token_ids": [151645],
            "gpu_generate_ms": 50.0,
            "cuda_max_reserved_mib": 2100.0,
            "raw_output": raw_txt,
        }))
    (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if job_spec["mode"] == "T_B_other":
        # Wait briefly for manifest to be created by main thread? Actually runner runs BEFORE manifest creation,
        # so for fake runner test we write verifier outputs cleanly after detecting manifest OR hook via second stage!
        pass


def _fake_verifier_runner(job_spec: dict, manifest_path: pathlib.Path, cancel_ev: threading.Event, svc: LiveDemoService) -> None:
    jdir = svc.jobs_dir / job_spec["job_id"]
    v_dir = jdir / "verifier_runs"
    v_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    vlines = []
    for i, u in enumerate(man.get("unique_calls") or []):
        lbl = "resistor" if i == 0 else "other"
        vlines.append(json.dumps({
            "call_id": u["call_id"],
            "ok": True,
            "raw_text": json.dumps({"label": lbl}) + "<|im_end|>",
            "raw_output": json.dumps({"label": lbl}) + "<|im_end|>",
            "generated_tokens": 6,
            "has_normal_eos": True,
        }))
    (v_dir / "verifier_raw_outputs.jsonl").write_text("\n".join(vlines) + ("\n" if vlines else ""), encoding="utf-8")


class _TwoStageFakeRunner:
    """Deterministic fake base runner for CPU tests."""

    def __init__(self, *, mode_behavior: str = "normal") -> None:
        self.mode_behavior = mode_behavior

    def __call__(self, job_spec: dict, cancel_ev: threading.Event, svc: LiveDemoService) -> None:
        mk = job_spec["model_key"]
        jdir = svc.jobs_dir / job_spec["job_id"]
        run_dir = jdir / "base_runs" / f"run_{mk}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(run_dir / "gpu_gate.json", {"passed": True, "smi_free_mib": 6900.0})
        plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
        q_items = plan["queues"][mk]
        if self.mode_behavior == "incomplete" and len(q_items) > 1:
            q_items = q_items[:-1]  # intentionally drop last tile!

        lines = []
        for idx, q in enumerate(q_items):
            if self.mode_behavior == "empty":
                raw_txt = "[]<|im_end|>"
            elif q["mode"] == "W":
                raw_txt = json.dumps([
                    {"bbox_2d": [100, 100, 220, 220], "label": "capacitor"},
                    {"bbox_2d": [104, 104, 218, 218], "label": "capacitor"},
                ]) + "<|im_end|>"
            elif idx == 0:
                raw_txt = json.dumps([
                    {"bbox_2d": [120, 120, 240, 240], "label": "resistor"},
                    {"bbox_2d": [320, 320, 440, 440], "label": "ic"},
                ]) + "<|im_end|>"
            else:
                raw_txt = "[]<|im_end|>"
            lines.append(json.dumps({
                "call_id": q["call_id"],
                "board_id": q["board_id"],
                "mode": q["mode"],
                "call_index": q["call_index"],
                "ok": True,
                "orig_input_width": q["width"],
                "orig_input_height": q["height"],
                "roi_offset_x": q["roi_offset_x"],
                "roi_offset_y": q["roi_offset_y"],
                "tile_offset_x": q["tile_offset_x"],
                "tile_offset_y": q["tile_offset_y"],
                "max_new_tokens": 512,
                "generated_tokens": 10,
                "generated_token_ids": [151645],
                "gpu_generate_ms": 20.0,
                "cuda_max_reserved_mib": 2000.0,
                "raw_output": raw_txt,
            }))
        (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_gt_free_upload_validation_modes_w_t_b_other_exports_and_posthoc_gt(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "src_board.png")
    svc = LiveDemoService(
        workspace_root=tmp_path / "ws",
        job_runner=_TwoStageFakeRunner(),
        verifier_runner=_fake_verifier_runner,
    )
    srv = fics_final_demo.create_demo_server(
        workspace_root=tmp_path / "ws",
        host="127.0.0.1",
        port=0,
        allow_port_fallback=False,
        service=svc,
    )
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        # 1. Unsafe filename sanitized + GT-free ROI/tile preflight
        code, _, raw = _http_req(
            f"{base}/api/upload",
            method="POST",
            body=png_bytes,
            headers={"X-Filename": "../../../evil\\my board!!.png"},
        )
        assert code == 200
        up = json.loads(raw.decode("utf-8"))
        assert ".." not in up["safe_filename"]
        assert up["gt_required"] is False
        assert up["orig_wh"] == [1300, 1150]
        assert up["tile_count"] >= 4

        # 2. Disallow T_B_other for qwen2 and deepseek
        for bad_mk in ("qwen2", "deepseek"):
            c2, _, r2 = _http_req(
                f"{base}/api/jobs",
                method="POST",
                body=json.dumps({"upload_id": up["upload_id"], "model_key": bad_mk, "mode": "T_B_other"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            assert c2 == 400
            assert "Qwen3" in json.loads(r2.decode("utf-8"))["error"]

        # 3. Mode W preserves overlapping boxes (no extra NMS)
        c3, _, r3 = _http_req(
            f"{base}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "model_key": "qwen2", "mode": "W"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c3 == 202
        jw = json.loads(r3.decode("utf-8"))["job_id"]
        st_w = _wait_terminal(base, jw)
        assert st_w["status"] == "completed"
        _, _, res_w_raw = _http_req(f"{base}/api/jobs/{jw}/result")
        res_w = json.loads(res_w_raw.decode("utf-8"))
        assert res_w["gt_loaded"] is False
        assert res_w["kept_predictions_count"] == 2  # both overlapping W boxes preserved!

        # 4. Mode T_B_other on qwen3 filters 'other' cleanly & exports JSON/PNG/MD
        c4, _, r4 = _http_req(
            f"{base}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": up["upload_id"], "model_key": "qwen3", "mode": "T_B_other"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c4 == 202
        jb = json.loads(r4.decode("utf-8"))["job_id"]
        st_b = _wait_terminal(base, jb)
        assert st_b["status"] == "completed", f"Job failed: {st_b}"
        _, _, res_b_raw = _http_req(f"{base}/api/jobs/{jb}/result")
        res_b = json.loads(res_b_raw.decode("utf-8"))
        assert len(res_b["pre_verifier_candidates"]) == 2
        assert res_b["kept_predictions_count"] == 1
        assert res_b["verifier_summary"]["rejected_as_other"] == 1
        cl, _, jl = _http_req(f"{base}/api/jobs")
        assert cl == 200 and any(it["job_id"] == jb for it in json.loads(jl.decode("utf-8"))["jobs"])

        # Exports
        ce, _, je = _http_req(f"{base}/api/jobs/{jb}/export/json")
        assert ce == 200 and json.loads(je.decode("utf-8"))["job_id"] == jb
        cp, _, png_out = _http_req(f"{base}/api/jobs/{jb}/export/overlay_png?view=with_rejected")
        assert cp == 200 and png_out.startswith(b"\x89PNG")
        cm, _, md_out = _http_req(f"{base}/api/jobs/{jb}/export/summary_md")
        assert cm == 200 and "未评价" in md_out.decode("utf-8")

        # 5. Optional post-hoc GT validation: mismatch rejected (400), valid accepted (200)
        cats = [
            {"id": 1, "name": "capacitor"},
            {"id": 2, "name": "resistor"},
            {"id": 3, "name": "inductor"},
            {"id": 4, "name": "transistor"},
            {"id": 5, "name": "diode"},
            {"id": 6, "name": "ic"},
        ]
        bad_dim_coco = {
            "images": [{"id": 10, "file_name": "wrong.png", "width": 800, "height": 600}],
            "categories": cats,
            "annotations": [],
        }
        cg_bad, _, _ = _http_req(
            f"{base}/api/jobs/{jb}/evaluate_gt",
            method="POST",
            body=json.dumps({"coco_gt": bad_dim_coco}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert cg_bad == 400

        kept_box = res_b["predictions"][0]["bbox_xyxy"]
        good_coco = {
            "images": [{"id": 10, "file_name": up["safe_filename"], "width": 1300, "height": 1150}],
            "categories": cats,
            "annotations": [{
                "id": 1,
                "image_id": 10,
                "category_id": 2,
                "bbox": [kept_box[0], kept_box[1], kept_box[2] - kept_box[0], kept_box[3] - kept_box[1]],
            }],
        }
        cg_ok, _, rg_ok = _http_req(
            f"{base}/api/jobs/{jb}/evaluate_gt",
            method="POST",
            body=json.dumps({"coco_gt": good_coco}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert cg_ok == 200
        ev = json.loads(rg_ok.decode("utf-8"))
        assert ev["gt_loaded"] is True
        assert ev["post_hoc_eval"]["metrics"]["tp"] == 1
        assert ev["post_hoc_eval"]["pre_verifier_metrics"]["fp"] == 1
    finally:
        svc.shutdown()
        srv.shutdown()
        srv.server_close()


def test_single_gpu_lock_and_safe_process_tree_cancellation(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "board_cancel.png")
    child_pid_file = tmp_path / "child_pid.txt"
    started_ev = threading.Event()

    def _slow_subprocess_runner(job_spec: dict, cancel_ev: threading.Event, svc: LiveDemoService) -> None:
        child_code = (
            "import pathlib, subprocess, sys, time; "
            f"c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(c.pid), encoding='utf-8'); "
            "c.wait()"
        )
        started_ev.set()
        svc._run_subprocess_cancellable(
            job_id=job_spec["job_id"],
            cmd=[sys.executable, "-c", child_code],
            log_path=svc.jobs_dir / job_spec["job_id"] / "slow.log",
            cancel_ev=cancel_ev,
        )

    svc = LiveDemoService(workspace_root=tmp_path / "ws_cancel", job_runner=_slow_subprocess_runner)
    up = svc.register_upload(raw_bytes=png_bytes, raw_filename="board_cancel.png")
    st1 = svc.submit_job(upload_id=up["upload_id"], model_key="qwen3", mode="T")
    jid = st1["job_id"]

    # Wait until nested child PID is alive
    assert started_ev.wait(timeout=5.0)
    for _ in range(50):
        if child_pid_file.is_file():
            break
        time.sleep(0.05)
    assert child_pid_file.is_file()
    child_pid = int(child_pid_file.read_text(encoding="utf-8").strip())

    # Second concurrent job MUST be rejected cleanly with 409 lock!
    with pytest.raises(Exception) as exc_info:
        svc.submit_job(upload_id=up["upload_id"], model_key="qwen2", mode="W")
    assert getattr(exc_info.value, "status_code", None) == 409

    # Safe cancellation terminates both parent & nested child process
    svc.cancel_job(jid)
    svc.shutdown()
    assert svc.get_job_status(jid)["status"] == "cancelled"

    time.sleep(0.3)
    # Verify child PID no longer exists cleanly
    alive = True
    try:
        os.kill(child_pid, 0)
    except OSError:
        alive = False
    except SystemError:
        alive = False
    assert alive is False


def test_incomplete_and_explicit_empty_distinction(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "board_e.png")

    svc_empty = LiveDemoService(workspace_root=tmp_path / "ws_e", job_runner=_TwoStageFakeRunner(mode_behavior="empty"))
    up_e = svc_empty.register_upload(raw_bytes=png_bytes, raw_filename="e.png")
    st_e = svc_empty.submit_job(upload_id=up_e["upload_id"], model_key="qwen3", mode="T_B_other")
    for _ in range(100):
        cur = svc_empty.get_job_status(st_e["job_id"])
        if cur["status"] in ("completed", "failed", "incomplete"):
            break
        time.sleep(0.05)
    assert cur["status"] == "completed"
    res_e = svc_empty.get_job_result(st_e["job_id"])
    assert res_e["kept_predictions_count"] == 0
    assert res_e["termination_v2"]["explicit_empty"] == res_e["termination_v2"]["total_calls"]

    svc_inc = LiveDemoService(workspace_root=tmp_path / "ws_i", job_runner=_TwoStageFakeRunner(mode_behavior="incomplete"))
    up_i = svc_inc.register_upload(raw_bytes=png_bytes, raw_filename="i.png")
    st_i = svc_inc.submit_job(upload_id=up_i["upload_id"], model_key="qwen3", mode="T")
    for _ in range(100):
        cur_i = svc_inc.get_job_status(st_i["job_id"])
        if cur_i["status"] in ("completed", "failed", "incomplete"):
            break
        time.sleep(0.05)
    assert cur_i["status"] == "incomplete"
    assert cur_i["technical_complete"] is False


def test_research_replay_store_real_eight_boards_if_configured() -> None:
    idx_env = os.environ.get("FICS_RESULTS_INDEX")
    if not idx_env:
        pytest.skip("Set FICS_RESULTS_INDEX to run full 8-board research replay integrity check")
    store = ResearchReplayStore(pathlib.Path(idx_env))
    ov = store.get_overview()
    assert ov["available"] is True
    assert set(ov["datasets"].keys()) == {
        "dev_two_board_s18_s28",
        "unobserved_two_board_s29_s27",
        "dev_expansion_s4_s20",
        "heldout_blind_s19_s16",
    }
    h = ov["datasets"]["heldout_blind_s19_s16"]
    assert h["split_group"] == "heldout_blind"
    assert h["call_counts_per_model"] == {"W": 2, "T": 140, "batch_total": 142, "B_other_crops": 508, "qwen3_T_plus_B_other_total": 648}
    assert h["models"]["qwen3"]["combined"]["T_B_other"]["tp"] == 10
    assert h["models"]["qwen3"]["combined"]["T_B_other"]["fp"] == 140
    assert h["models"]["qwen3"]["combined"]["T_B_other"]["fn"] == 97


def test_concurrent_state_updates_and_cancellation_terminal_rules(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "board_conc.png")
    svc = LiveDemoService(workspace_root=tmp_path / "ws_conc", job_runner=_TwoStageFakeRunner())
    up = svc.register_upload(raw_bytes=png_bytes, raw_filename="board_conc.png")

    # 1. Serialize concurrent _update_state calls under load with zero lost keys
    jdir = svc.jobs_dir / "job_20261116_000001_conc0001"
    jdir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(jdir / "job_state.json", {
        "job_id": jdir.name,
        "status": "running_base_inference",
        "stage_label": "init",
        "cancel_requested": False,
        "worker_finished": False,
    })

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _writer_cancel() -> None:
        try:
            barrier.wait(timeout=3.0)
            svc._update_state(jdir.name, cancel_requested=True)
        except Exception as exc:
            errors.append(exc)

    def _writer_finish() -> None:
        try:
            barrier.wait(timeout=3.0)
            svc._update_state(jdir.name, worker_finished=True)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_writer_cancel)
    t2 = threading.Thread(target=_writer_finish)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert not errors
    final_st = json.loads((jdir / "job_state.json").read_text(encoding="utf-8"))
    assert final_st["cancel_requested"] is True
    assert final_st["worker_finished"] is True
    assert not [p for p in jdir.iterdir() if p.name.endswith(".tmp")]

    # 2. Live interleaved worker thread + cancel thread must never deadlock and terminal state must stay immutable
    in_worker = threading.Event()
    allow_exit = threading.Event()

    def _interleaved_worker(job_spec: dict, cancel_ev: threading.Event, s: LiveDemoService) -> None:
        in_worker.set()
        for step in range(8):
            s._update_state(job_spec["job_id"], status="running_base_inference", stage_label=f"step-{step}", base_calls_completed=step)
            if cancel_ev.is_set():
                raise InterruptedError("cancelled mid-run")
            time.sleep(0.02)
        allow_exit.wait(timeout=2.0)

    svc2 = LiveDemoService(workspace_root=tmp_path / "ws_conc2", job_runner=_interleaved_worker)
    up2 = svc2.register_upload(raw_bytes=png_bytes, raw_filename="board_conc2.png")
    st2 = svc2.submit_job(upload_id=up2["upload_id"], model_key="qwen2", mode="W")
    assert in_worker.wait(timeout=3.0)
    svc2.cancel_job(st2["job_id"])
    allow_exit.set()
    svc2.shutdown()

    end_st = svc2.get_job_status(st2["job_id"])
    assert end_st["status"] == "cancelled"
    assert end_st["cancel_requested"] is True
    # Attempting to overwrite terminal state with running/completed must be ignored
    after_late = svc2._update_state(st2["job_id"], status="completed", stage_label="should_not_overwrite")
    assert after_late["status"] == "cancelled"


def test_posthoc_gt_identity_negative_and_positive_cases(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "s20_target.png")
    img_sha = hashlib.sha256(png_bytes).hexdigest()
    svc = LiveDemoService(workspace_root=tmp_path / "ws_gt_id", job_runner=_TwoStageFakeRunner())
    up = svc.register_upload(raw_bytes=png_bytes, raw_filename="s20_target.png")
    st = svc.submit_job(upload_id=up["upload_id"], model_key="qwen3", mode="T")
    for _ in range(100):
        cur = svc.get_job_status(st["job_id"])
        if cur["status"] == "completed":
            break
        time.sleep(0.05)
    assert cur["status"] == "completed"
    jid = st["job_id"]
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(["capacitor", "resistor", "inductor", "transistor", "diode", "ic"])]

    # Negative 1: same size, single image, different filename, no confirm -> REJECTED & on-disk job_result untouched
    neg1 = {"images": [{"id": 999, "file_name": "completely_different_board.tif", "width": 1300, "height": 1150}], "categories": cats, "annotations": []}
    with pytest.raises(Exception) as e1:
        svc.evaluate_job_with_coco(job_id=jid, coco_gt=neg1)
    assert getattr(e1.value, "status_code", None) == 400
    assert svc.get_job_result(jid)["gt_loaded"] is False

    # Negative 2: SHA conflict even if filename matches AND confirm_manual_mapping=True -> REJECTED
    neg2 = {"images": [{"id": 1, "file_name": "s20_target.png", "width": 1300, "height": 1150, "sha256": "0" * 64}], "categories": cats, "annotations": []}
    with pytest.raises(Exception) as e2:
        svc.evaluate_job_with_coco(job_id=jid, coco_gt=neg2, confirm_manual_mapping=True)
    assert getattr(e2.value, "status_code", None) == 400
    assert svc.get_job_result(jid)["gt_loaded"] is False

    # Negative 3: single image with empty/unknown filename -> REJECTED
    neg3 = {"images": [{"id": 1, "file_name": "", "width": 1300, "height": 1150}], "categories": cats, "annotations": []}
    with pytest.raises(Exception) as e3:
        svc.evaluate_job_with_coco(job_id=jid, coco_gt=neg3)
    assert getattr(e3.value, "status_code", None) == 400

    # Negative 4: explicit wrong target_image_id & unconfirmed explicit mismatched ID -> REJECTED
    neg4 = {"images": [{"id": 7, "file_name": "other_board.tif", "width": 1300, "height": 1150}], "categories": cats, "annotations": []}
    with pytest.raises(Exception) as e4a:
        svc.evaluate_job_with_coco(job_id=jid, coco_gt=neg4, target_image_id=999)
    assert getattr(e4a.value, "status_code", None) == 400
    with pytest.raises(Exception) as e4b:
        svc.evaluate_job_with_coco(job_id=jid, coco_gt=neg4, target_image_id=7)
    assert getattr(e4b.value, "status_code", None) == 400
    assert svc.get_job_result(jid)["gt_loaded"] is False

    # Positive 1: filename match without SHA -> accepted, image_identity_verified=False
    pos1 = {"images": [{"id": 1, "file_name": "s20_target.png", "width": 1300, "height": 1150}], "categories": cats, "annotations": []}
    res1 = svc.evaluate_job_with_coco(job_id=jid, coco_gt=pos1)
    assert res1["gt_loaded"] is True
    assert res1["post_hoc_eval"]["image_identity_verified"] is False
    assert res1["post_hoc_eval"]["mapping_source"] == "auto_filename_match"

    # Positive 2: SHA match -> image_identity_verified=True
    pos2 = {"images": [{"id": 5, "file_name": "renamed_but_sha_verified.tif", "width": 1300, "height": 1150, "sha256": img_sha}], "categories": cats, "annotations": []}
    res2 = svc.evaluate_job_with_coco(job_id=jid, coco_gt=pos2)
    assert res2["post_hoc_eval"]["image_identity_verified"] is True
    assert res2["post_hoc_eval"]["mapping_source"] == "auto_sha256"

    # Positive 3: renamed board without SHA + explicit confirm_manual_mapping=True -> accepted with manual record & image_identity_verified=False
    pos3 = {"images": [{"id": 8, "file_name": "archived_original_s20.tif", "width": 1300, "height": 1150}], "categories": cats, "annotations": []}
    res3 = svc.evaluate_job_with_coco(
        job_id=jid,
        coco_gt=pos3,
        confirm_manual_mapping=True,
        manual_mapping_reason="Renamed for blind upload",
    )
    phe3 = res3["post_hoc_eval"]
    assert phe3["image_identity_verified"] is False
    assert phe3["mapping_source"] == "user_confirmed_manual_mapping"
    assert phe3["manual_confirmation"]["confirmed"] is True
    assert phe3["manual_confirmation"]["reason"] == "Renamed for blind upload"


def test_zero_support_gt_recall_null_and_na_display(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "zero_sup.png")
    svc = LiveDemoService(workspace_root=tmp_path / "ws_zero", job_runner=_TwoStageFakeRunner())
    up = svc.register_upload(raw_bytes=png_bytes, raw_filename="zero_sup.png")
    st = svc.submit_job(upload_id=up["upload_id"], model_key="qwen2", mode="W")
    for _ in range(100):
        cur = svc.get_job_status(st["job_id"])
        if cur["status"] == "completed":
            break
        time.sleep(0.05)
    assert cur["status"] == "completed"

    # Only resistor GT present; capacitor has gt_count=0 AND 2 FP predictions
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(["capacitor", "resistor", "inductor", "transistor", "diode", "ic"])]
    coco = {
        "images": [{"id": 1, "file_name": "zero_sup.png", "width": 1300, "height": 1150}],
        "categories": cats,
        "annotations": [{"id": 10, "image_id": 1, "category_id": 2, "bbox": [50, 50, 40, 40]}],
    }
    res = svc.evaluate_job_with_coco(job_id=st["job_id"], coco_gt=coco)
    m = res["post_hoc_eval"]["metrics"]
    assert m["total_gt"] == 1
    assert m["recall"] == 0.0
    assert m["per_class"]["capacitor"]["gt_count"] == 0
    assert m["per_class"]["capacitor"]["fp"] == 2
    assert m["per_class"]["capacitor"]["recall"] is None
    assert m["per_class"]["capacitor"]["recall_display"] == "N/A (GT=0)"
    assert m["per_class"]["diode"]["recall"] is None
    assert m["per_class"]["diode"]["recall_display"] == "N/A (GT=0)"

    blob, _, _ = svc.export_job(job_id=st["job_id"], kind="summary_md")
    md_text = blob.decode("utf-8")
    assert "N/A (GT=0)" in md_text


def test_demo_server_config_resolution_and_cli_overrides(tmp_path: pathlib.Path) -> None:
    cfg_path = tmp_path / "demo_startup.json"
    cfg_path.write_text(json.dumps({
        "workspace_root": str(tmp_path / "ws_from_cfg"),
        "runtime_config": str(tmp_path / "rt.yaml"),
        "results_index": str(tmp_path / "idx.json"),
        "host": "127.0.0.1",
        "port": 7901,
        "allow_port_fallback": False,
    }), encoding="utf-8")

    opts1 = fics_final_demo.resolve_server_launch_options(["--config", str(cfg_path)])
    assert opts1["workspace_root"] == tmp_path / "ws_from_cfg"
    assert opts1["port"] == 7901
    assert opts1["allow_port_fallback"] is False

    # CLI flags explicitly override --config values
    opts2 = fics_final_demo.resolve_server_launch_options([
        "--config", str(cfg_path),
        "--workspace-root", str(tmp_path / "ws_override"),
        "--port", "7999",
    ])
    assert opts2["workspace_root"] == tmp_path / "ws_override"
    assert opts2["port"] == 7999

    # Model runtime YAML passed to --config must fail fast
    bad_cfg = tmp_path / "wrong_role.json"
    bad_cfg.write_text(json.dumps({"models": {"qwen3": {"local_dir": "/tmp"}}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        fics_final_demo.resolve_server_launch_options(["--config", str(bad_cfg)])

    # Verify checked-in generic example JSON parses cleanly without any private paths
    ex_opts = fics_final_demo.resolve_server_launch_options([
        "--config", str(_REPO_ROOT / "configs" / "fics_final_demo.example.json")
    ])
    assert ex_opts["port"] == 7862
