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


def test_upload_preflight_guards_and_orig_vs_roi_preview(tmp_path: pathlib.Path) -> None:
    svc = LiveDemoService(workspace_root=tmp_path / "ws_preflight", job_runner=_TwoStageFakeRunner())

    # 1. Empty bytes rejected
    with pytest.raises(Exception) as e_empty:
        svc.register_upload(raw_bytes=b"", raw_filename="empty.png")
    assert getattr(e_empty.value, "status_code", None) == 400

    # 2. Unsupported extension rejected
    with pytest.raises(Exception) as e_ext:
        svc.register_upload(raw_bytes=b"1234", raw_filename="bad.gif")
    assert getattr(e_ext.value, "status_code", None) == 400

    # 3. Corrupted image bytes rejected
    with pytest.raises(Exception) as e_corrupt:
        svc.register_upload(raw_bytes=b"not_an_image_payload", raw_filename="broken.png")
    assert getattr(e_corrupt.value, "status_code", None) == 400

    # 4. Blank all-white or solid uniform image rejected
    white_p = tmp_path / "blank_white.png"
    Image.fromarray(np.full((400, 400, 3), 255, dtype=np.uint8)).save(white_p)
    with pytest.raises(Exception) as e_white:
        svc.register_upload(raw_bytes=white_p.read_bytes(), raw_filename="blank_white.png")
    assert getattr(e_white.value, "status_code", None) == 400

    solid_p = tmp_path / "solid_dark.png"
    Image.fromarray(np.full((400, 400, 3), 40, dtype=np.uint8)).save(solid_p)
    with pytest.raises(Exception) as e_solid:
        svc.register_upload(raw_bytes=solid_p.read_bytes(), raw_filename="solid_dark.png")
    assert getattr(e_solid.value, "status_code", None) == 400

    # 5. Too small image rejected (< 32x32)
    tiny_p = tmp_path / "tiny.png"
    arr_t = np.zeros((16, 16, 3), dtype=np.uint8)
    arr_t[:, :8] = 200
    Image.fromarray(arr_t).save(tiny_p)
    with pytest.raises(Exception) as e_tiny:
        svc.register_upload(raw_bytes=tiny_p.read_bytes(), raw_filename="tiny.png")
    assert getattr(e_tiny.value, "status_code", None) == 400

    # 6. Multi-frame TIFF rejected
    multi_tif = tmp_path / "multi_frame.tif"
    f1 = Image.fromarray(np.zeros((120, 120, 3), dtype=np.uint8))
    f2 = Image.fromarray(np.full((120, 120, 3), 128, dtype=np.uint8))
    f1.save(multi_tif, save_all=True, append_images=[f2])
    with pytest.raises(Exception) as e_multi:
        svc.register_upload(raw_bytes=multi_tif.read_bytes(), raw_filename="multi_frame.tif")
    assert getattr(e_multi.value, "status_code", None) == 400

    # 7. Valid PCB upload exposes available_modes and renders both ROI and ORIG previews
    valid_bytes = _make_board_png(tmp_path / "good_board.png", w=1200, h=900)
    up = svc.register_upload(raw_bytes=valid_bytes, raw_filename="good_board.png")
    assert len(up["available_modes"]) == 3
    jpg_roi = svc.render_upload_roi_jpeg(upload_id=up["upload_id"], max_width=800, source="roi")
    jpg_orig = svc.render_upload_roi_jpeg(upload_id=up["upload_id"], max_width=800, source="orig")
    assert jpg_roi.startswith(b"\xff\xd8")
    assert jpg_orig.startswith(b"\xff\xd8")


def test_multi_model_serial_comparison_partial_failure_isolation_and_exports(tmp_path: pathlib.Path) -> None:
    png_a = _make_board_png(tmp_path / "board_a.png", w=1280, h=960)
    png_b = _make_board_png(tmp_path / "board_b.png", w=1100, h=850)

    active_counter = 0
    max_concurrent = 0
    counter_lock = threading.Lock()
    execution_order: list[str] = []

    def _serial_mixed_runner(job_spec: dict, cancel_ev: threading.Event, svc: LiveDemoService) -> None:
        nonlocal active_counter, max_concurrent
        mk = job_spec["model_key"]
        with counter_lock:
            active_counter += 1
            if active_counter > max_concurrent:
                max_concurrent = active_counter
            execution_order.append(f"{mk}:{job_spec['mode']}")
        try:
            time.sleep(0.04)
            if cancel_ev.is_set():
                raise InterruptedError("cancelled")
            if mk == "deepseek":
                raise RuntimeError("simulated deepseek runtime error")

            jdir = svc.jobs_dir / job_spec["job_id"]
            run_dir = jdir / "base_runs" / f"run_{mk}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(run_dir / "gpu_gate.json", {"passed": True, "smi_free_mib": 6400.0})
            plan = json.loads((jdir / "plan.json").read_text(encoding="utf-8"))
            lines = []
            for q in plan["queues"][mk]:
                if mk == "qwen2":
                    # Explicit empty output -> completed_zero_boxes
                    raw_txt = "[]<|im_end|>"
                else:
                    # Qwen3 produces valid boxes -> completed_with_boxes
                    raw_txt = json.dumps([
                        {"bbox_2d": [100, 120, 260, 280], "label": "capacitor"},
                        {"bbox_2d": [320, 180, 480, 340], "label": "ic"},
                    ]) + "<|im_end|>"
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
                    "generated_tokens": 12,
                    "generated_token_ids": [151645],
                    "gpu_generate_ms": 25.0,
                    "cuda_max_reserved_mib": 2100.0,
                    "raw_output": raw_txt,
                }))
            (run_dir / "raw_outputs_all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        finally:
            with counter_lock:
                active_counter -= 1

    svc = LiveDemoService(
        workspace_root=tmp_path / "ws_cmp",
        job_runner=_serial_mixed_runner,
        verifier_runner=_fake_verifier_runner,
    )
    srv = fics_final_demo.create_demo_server(
        workspace_root=tmp_path / "ws_cmp",
        host="127.0.0.1",
        port=0,
        allow_port_fallback=False,
        service=svc,
    )
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        _, _, up_a_raw = _http_req(f"{base}/api/upload", method="POST", body=png_a, headers={"X-Filename": "board_a.png"})
        up_a = json.loads(up_a_raw.decode("utf-8"))
        _, _, up_b_raw = _http_req(f"{base}/api/upload", method="POST", body=png_b, headers={"X-Filename": "board_b.png"})
        up_b = json.loads(up_b_raw.decode("utf-8"))

        # Disallow multi-model comparison in T_B_other baseline mode
        c_bad, _, _ = _http_req(
            f"{base}/api/comparisons",
            method="POST",
            body=json.dumps({"upload_id": up_a["upload_id"], "model_keys": ["qwen2", "qwen3"], "mode": "T_B_other"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_bad == 400

        # Submit 3-model W serial comparison with 1 failing model (deepseek), 1 zero-box model (qwen2), 1 box model (qwen3)
        c_ok, _, cmp_raw = _http_req(
            f"{base}/api/comparisons",
            method="POST",
            body=json.dumps({
                "upload_id": up_a["upload_id"],
                "model_keys": ["qwen2", "deepseek", "qwen3"],
                "mode": "W",
                "include_qwen3_b_other": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_ok == 202
        cmp_id = json.loads(cmp_raw.decode("utf-8"))["comparison_id"]

        # Poll comparison to completion
        deadline = time.time() + 15.0
        cmp_final = None
        while time.time() < deadline:
            _, _, r_c = _http_req(f"{base}/api/comparisons/{cmp_id}")
            st_c = json.loads(r_c.decode("utf-8"))
            if st_c["status"] != "running":
                cmp_final = st_c
                break
            time.sleep(0.04)
        assert cmp_final is not None
        assert max_concurrent == 1  # Strictly serial on GPU
        assert execution_order == ["qwen2:W", "deepseek:W", "qwen3:W"]
        assert cmp_final["status"] == "partial_completed"

        slots_by_mk = {s["model_key"]: s for s in cmp_final["slots"]}
        # Qwen2: completed with 0 boxes (explicit empty)
        assert slots_by_mk["qwen2"]["queue_status"] == "completed"
        assert slots_by_mk["qwen2"]["outcome"]["outcome_code"] == "completed_zero_boxes"
        assert slots_by_mk["qwen2"]["kept_predictions_count"] == 0
        # DeepSeek: failed -> remaining model (Qwen3) still ran and succeeded!
        assert slots_by_mk["deepseek"]["queue_status"] == "failed"
        assert slots_by_mk["deepseek"]["outcome"]["outcome_code"] == "failed"
        assert slots_by_mk["deepseek"]["kept_predictions_count"] is None
        assert slots_by_mk["deepseek"]["exports"] is None
        # Qwen3: completed with 2 boxes
        assert slots_by_mk["qwen3"]["queue_status"] == "completed"
        assert slots_by_mk["qwen3"]["outcome"]["outcome_code"] == "completed_with_boxes"
        assert slots_by_mk["qwen3"]["kept_predictions_count"] == 2
        assert slots_by_mk["qwen3"]["class_counts"]["capacitor"] == 1
        assert slots_by_mk["qwen3"]["class_counts"]["ic"] == 1

        # Export comparison markdown & JSON: failed model must NOT have fake empty score tables
        cm_code, _, md_bytes = _http_req(f"{base}/api/comparisons/{cmp_id}/export/markdown")
        assert cm_code == 200
        md_txt = md_bytes.decode("utf-8")
        assert up_a["source_image_sha256"] in md_txt
        assert "不等于电路板上没有元器件" in md_txt
        assert "—（未生成）" in md_txt

        cj_code, _, js_bytes = _http_req(f"{base}/api/comparisons/{cmp_id}/export/json")
        assert cj_code == 200
        js_obj = json.loads(js_bytes.decode("utf-8"))
        assert js_obj["comparison_id"] == cmp_id
        assert js_obj["slots"][1]["exports"] is None

        # Cross-upload isolation check: cannot bind a job from board_b into a comparison for board_a
        _, _, jb_raw = _http_req(
            f"{base}/api/jobs",
            method="POST",
            body=json.dumps({"upload_id": up_b["upload_id"], "model_key": "qwen3", "mode": "W"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        jb_id = json.loads(jb_raw.decode("utf-8"))["job_id"]
        _wait_terminal(base, jb_id)

        c_cross, _, _ = _http_req(
            f"{base}/api/comparisons/from_jobs",
            method="POST",
            body=json.dumps({
                "upload_id": up_a["upload_id"],
                "mode": "W",
                "job_ids": [slots_by_mk["qwen2"]["job_id"], jb_id],
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert c_cross == 400
    finally:
        svc.shutdown()
        srv.shutdown()
        srv.server_close()


def test_comparison_cancel_prevents_next_slot_dispatch_interleaving(tmp_path: pathlib.Path) -> None:
    """Controlled CPU interleaving: once cancel_comparison is accepted, undispatched next slot must never start."""
    board_p = tmp_path / "board_cancel_race.png"
    _make_board_png(board_p)

    dispatched_models: list[str] = []
    slot1_at_submit_barrier = threading.Event()
    release_slot1_submit = threading.Event()

    def _runner(job_spec: Dict[str, Any], cancel_ev: threading.Event, svc: LiveDemoService) -> None:
        mk = job_spec["model_key"]
        dispatched_models.append(mk)
        jdir = svc.jobs_dir / job_spec["job_id"]
        run_dir = jdir / "base_runs" / f"run_{mk}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "event": "result",
            "call_id": job_spec["expected_base_call_ids"][0],
            "board_id": job_spec["board_id"],
            "mode": "W",
            "image_id": 1,
            "tile_id": None,
            "orig_input_width": job_spec["roi_width"],
            "orig_input_height": job_spec["roi_height"],
            "roi_offset_x": job_spec["roi_offset_x"],
            "roi_offset_y": job_spec["roi_offset_y"],
            "tile_offset_x": 0,
            "tile_offset_y": 0,
            "generated_tokens": 24,
            "max_new_tokens": 512,
            "generated_token_ids": [1, 151645],
            "ok": True,
            "raw_output": '[{"bbox_2d": [100, 100, 200, 200], "label": "capacitor"}]',
        }
        (run_dir / "raw_outputs_all.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

    svc = LiveDemoService(workspace_root=tmp_path / "ws", job_runner=_runner)
    orig_submit = svc.submit_job

    def _interleaved_submit(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        if kwargs.get("model_key") == "deepseek" and kwargs.get("_from_comparison_id"):
            slot1_at_submit_barrier.set()
            release_slot1_submit.wait(timeout=5.0)
        return orig_submit(*args, **kwargs)

    svc.submit_job = _interleaved_submit  # type: ignore[method-assign]
    try:
        up = svc.register_upload(raw_bytes=board_p.read_bytes(), raw_filename=board_p.name)
        cmp_st = svc.submit_comparison(upload_id=up["upload_id"], model_keys=["qwen2", "deepseek", "qwen3"], mode="W")
        cid = cmp_st["comparison_id"]

        assert slot1_at_submit_barrier.wait(timeout=5.0)
        svc.cancel_comparison(cid)
        release_slot1_submit.set()

        deadline = time.time() + 5.0
        final_cmp = svc.get_comparison_status(cid)
        while final_cmp["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
            final_cmp = svc.get_comparison_status(cid)

        assert dispatched_models == ["qwen2"]
        assert final_cmp["status"] == "partial_cancelled"
        assert final_cmp["slots"][0]["queue_status"] == "completed"
        assert final_cmp["slots"][0]["has_result"] is True
        assert final_cmp["slots"][1]["queue_status"] == "cancelled"
        assert final_cmp["slots"][1]["job_id"] is None
        assert final_cmp["slots"][2]["queue_status"] == "cancelled"
        assert final_cmp["slots"][2]["job_id"] is None
    finally:
        release_slot1_submit.set()
        svc.shutdown()


def test_frontend_stale_async_responses_and_session_isolation(tmp_path: pathlib.Path) -> None:
    """Deterministic JS VM regressions for cross-upload poll, same-upload comparison switch, focus race,
    running-job restore/cancel, single-vs-comparison switch, and transient poll recovery."""
    import shutil
    import subprocess

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("node binary not available in PATH for frontend JS VM regression")

    demo_script_path = _REPO_ROOT / "scripts" / "fics_final_demo.py"
    js_runner = tmp_path / "fe_race_check.js"
    js_runner.write_text(
        r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const pySrc = fs.readFileSync(process.argv[2], 'utf8');
const m = pySrc.match(/<script>([\s\S]*?)<\/script>/);
assert(m, 'Missing <script> block in fics_final_demo.py');
const scriptCode = m[1];

function makeDeferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}
function makeFakeElement(id) {
  const classes = new Set();
  return {
    id, value: '', checked: false, disabled: false, textContent: '', innerHTML: '', src: '',
    clientWidth: 1000, files: [], style: { display: '', width: '', height: '' }, className: '',
    classList: { add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c) },
    onclick: null, onchange: null, attributes: {},
    setAttribute(k, v) { this.attributes[k] = v; },
    getAttribute(k) { return this.attributes[k]; },
    appendChild(child) { return child; },
    querySelector(sel) {
      if (!this._q) this._q = {};
      if (!this._q[sel]) this._q[sel] = makeFakeElement(id + ':' + sel);
      return this._q[sel];
    },
    querySelectorAll() { return []; },
  };
}
function createSandbox(fetchHandler) {
  const elements = {};
  const clsBoxes = ['capacitor','resistor','inductor','transistor','diode','ic'].map(c => {
    const el = makeFakeElement('chkCls_' + c);
    el.value = c; el.checked = true; return el;
  });
  const getEl = id => (elements[id] || (elements[id] = makeFakeElement(id)));
  getEl('liveMode').value = 'W';
  getEl('liveModel').value = 'qwen2';
  getEl('liveCoordSpace').value = 'roi';
  getEl('liveOverlayView').value = 'kept';
  getEl('liveZoom').value = 'fit';
  getEl('chkModel_qwen2').checked = true;
  getEl('btnCancelJob').disabled = true;

  let nextTimerId = 10;
  const activeTimers = new Map();
  const clearedTimers = [];
  const sandbox = {
    console, setTimeout, clearTimeout, Promise, Set, Map, Array, Object, Number, String, Boolean, Math, JSON, Date, Error,
    encodeURIComponent, AbortController: global.AbortController,
    window: { location: { href: '' } },
    document: {
      getElementById: getEl,
      createElement: tag => makeFakeElement(tag),
      querySelectorAll: sel => (sel === '.chk-cls' ? clsBoxes : []),
    },
    setInterval(fn, ms) { const id = ++nextTimerId; activeTimers.set(id, fn); return id; },
    clearInterval(id) { clearedTimers.push(id); activeTimers.delete(id); },
    fetch(url, opts) { return fetchHandler(url, opts || {}); },
  };
  vm.createContext(sandbox);
  vm.runInContext(scriptCode + `
;this.__state = () => ({ LIVE_UPLOAD, LIVE_JOB_ID, LIVE_RESULT, LIVE_COMPARISON_ID, LIVE_COMPARISON, LIVE_POLL_TIMER, LIVE_ACTIVE_SESSION });
this.__setUpload = (u) => { LIVE_UPLOAD = u; };
`, sandbox);
  return { sandbox, getEl, activeTimers, clearedTimers };
}
const jsonResp = (obj, ok = true, status = 200) => ({ ok, status, json: async () => obj });
const mkUp = (uid, sha) => ({
  upload_id: uid, safe_filename: uid + '.png', source_image_sha256: sha,
  orig_wh: [1000, 800], roi_bbox_half_open: [10, 10, 990, 790], roi_offset_x: 10, roi_offset_y: 10, roi_width: 980, roi_height: 780, tile_count: 1,
});
const mkCmp = (cid, uid, sha, status, count = 0, jobId = null) => ({
  comparison_id: cid, upload_id: uid, safe_filename: uid + '.png', source_image_sha256: sha,
  orig_wh: [1000, 800], roi: { offset_x: 10, offset_y: 10, width: 980, height: 780, tile_count: 1 },
  baseline_mode: 'W', baseline_mode_display_name: 'W', model_keys: ['qwen2'], status,
  wall_elapsed_seconds: 1.0, sum_job_elapsed_seconds: 1.0,
  slots: [{
    slot_index: 0, slot_role: 'baseline_comparison', model_key: 'qwen2', model_display_name: 'Qwen2-VL-2B-Instruct',
    mode: 'W', queue_status: status === 'running' ? 'running' : 'completed', job_id: jobId, elapsed_seconds: 1.0,
    has_result: status !== 'running' && Boolean(jobId), kept_predictions_count: status === 'running' ? null : count,
    class_counts: status === 'running' ? null : { capacitor: count, resistor: 0, inductor: 0, transistor: 0, diode: 0, ic: 0 },
    termination_v2: status === 'running' ? null : { total_calls: 1, normal_eos: 1, length_limited_without_eos: 0, explicit_empty: 0 },
    predictions: status === 'running' ? null : [], outcome: { outcome_code: status === 'running' ? 'running' : 'completed_with_boxes', outcome_label: status, outcome_note: '' },
    exports: status === 'running' ? null : { json: '/j', overlay_png: '/p', summary_md: '/m' },
  }],
});
const mkJobSt = (jid, uid, sha, status) => ({
  job_id: jid, upload_id: uid, safe_filename: uid + '.png', source_image_sha256: sha,
  model_key: 'qwen2', model_display_name: 'Qwen2-VL-2B-Instruct', mode: 'W', mode_display_name: 'W',
  status, stage_label: status, elapsed_seconds: 1.0, eta_seconds: null,
  base_calls_planned: 1, base_calls_completed: status === 'completed' ? 1 : 0, verifier_calls_planned: 0, verifier_calls_completed: 0,
  has_result: status === 'completed', outcome: { outcome_code: status === 'completed' ? 'completed_with_boxes' : 'running', outcome_label: status },
});
const mkJobRes = (jid, uid, sha, mk = 'qwen2', count = 1) => ({
  job_id: jid, upload_id: uid, safe_filename: uid + '.png', source_image_sha256: sha,
  orig_wh: [1000, 800], roi: { offset_x: 10, offset_y: 10, width: 980, height: 780, tile_count: 1 },
  model_key: mk, model_display_name: mk, mode: 'W', mode_display_name: 'W',
  kept_predictions_count: count, post_nms_candidates_count: count, score_semantics_note: 'placeholder',
  outcome: { outcome_code: 'completed_with_boxes', outcome_label: 'ok', outcome_note: '' },
  class_counts: { capacitor: count, resistor: 0, inductor: 0, transistor: 0, diode: 0, ic: 0 },
  predictions: [], termination_v2: { total_calls: 1, normal_eos: 1, length_limited_without_eos: 0, explicit_empty: 0 },
  calls: [{ call_id: 'c0', status: 'ok', parsed_count: count, width: 980, height: 780, generated_tokens: 10, gpu_generate_ms: 10, cuda_max_reserved_mib: 100, raw_output: '[]' }],
});

(async () => {
  // 1. Cross-upload stale poll must not stop new timer
  {
    const dOld = makeDeferred();
    const uOld = mkUp('up_old', 'sha_old'), uNew = mkUp('up_new', 'sha_new');
    const env = createSandbox((url, opts) => {
      if (url === '/api/comparisons' && opts.method === 'POST') {
        const b = JSON.parse(opts.body);
        return Promise.resolve(jsonResp(b.upload_id === 'up_old' ? mkCmp('cmp_old', 'up_old', 'sha_old', 'running') : mkCmp('cmp_new', 'up_new', 'sha_new', 'running')));
      }
      if (url === '/api/comparisons/cmp_old') return dOld.promise;
      return Promise.resolve(jsonResp({ jobs: [], comparisons: [] }));
    });
    env.sandbox.__setUpload(uOld);
    await env.getEl('btnStartCompare').onclick();
    const tOld = env.sandbox.__state().LIVE_POLL_TIMER;
    const pOld = env.activeTimers.get(tOld)();
    env.sandbox.clearLiveResultsForNewUpload();
    env.sandbox.__setUpload(uNew);
    await env.getEl('btnStartCompare').onclick();
    const tNew = env.sandbox.__state().LIVE_POLL_TIMER;
    dOld.resolve(jsonResp(mkCmp('cmp_old', 'up_old', 'sha_old', 'running')));
    await pOld;
    assert.strictEqual(env.sandbox.__state().LIVE_POLL_TIMER, tNew);
    assert(env.activeTimers.has(tNew));
    assert(!env.clearedTimers.includes(tNew));
  }

  // 2. Same upload, two comparisons: late old comparison must not overwrite new comparison view/timer/export
  {
    const dOld = makeDeferred();
    const u1 = mkUp('up_same', 'sha_same');
    const env = createSandbox((url, opts) => {
      if (url === '/api/comparisons' && opts.method === 'POST') return Promise.resolve(jsonResp(mkCmp('cmp_old', 'up_same', 'sha_same', 'running')));
      if (url === '/api/comparisons/cmp_old') return dOld.promise;
      if (url === '/api/comparisons/cmp_new') return Promise.resolve(jsonResp(mkCmp('cmp_new', 'up_same', 'sha_same', 'running')));
      if (url === '/api/uploads/up_same') return Promise.resolve(jsonResp(u1));
      if (url === '/api/jobs/job_old/result') return Promise.resolve(jsonResp(mkJobRes('job_old', 'up_same', 'sha_same', 'qwen2', 5)));
      return Promise.resolve(jsonResp({ jobs: [], comparisons: [] }));
    });
    env.sandbox.__setUpload(u1);
    await env.getEl('btnStartCompare').onclick();
    const tOld = env.sandbox.__state().LIVE_POLL_TIMER;
    const pOld = env.activeTimers.get(tOld)();
    env.getEl('selRecentCmp').value = 'cmp_new';
    await env.getEl('btnLoadRecentCmp').onclick();
    const tNew = env.sandbox.__state().LIVE_POLL_TIMER;
    dOld.resolve(jsonResp(mkCmp('cmp_old', 'up_same', 'sha_same', 'completed', 5, 'job_old')));
    await pOld;
    const st = env.sandbox.__state();
    assert.strictEqual(st.LIVE_COMPARISON_ID, 'cmp_new');
    assert.strictEqual(st.LIVE_COMPARISON.comparison_id, 'cmp_new');
    assert.strictEqual(st.LIVE_POLL_TIMER, tNew);
    assert.strictEqual(st.LIVE_RESULT, null);
    env.getEl('btnExpCmpMd').onclick();
    assert(env.sandbox.window.location.href.includes('/api/comparisons/cmp_new/export/markdown'));
  }

  // 3. Rapid focus switch + running job restore + transient poll recovery
  {
    const dSlow = makeDeferred(), dFast = makeDeferred();
    const u1 = mkUp('up_r', 'sha_r');
    let failCount = 0;
    const env = createSandbox((url, opts) => {
      if (url === '/api/jobs/job_slow/result') return dSlow.promise;
      if (url === '/api/jobs/job_fast/result') return dFast.promise;
      if (url === '/api/uploads/up_r') return Promise.resolve(jsonResp(u1));
      if (url === '/api/jobs/job_run') {
        failCount++;
        if (failCount === 1) return Promise.resolve(jsonResp(mkJobSt('job_run', 'up_r', 'sha_r', 'running_base_inference')));
        if (failCount <= 4) return Promise.reject(new Error('network blip'));
        return Promise.resolve(jsonResp(mkJobSt('job_run', 'up_r', 'sha_r', 'running_base_inference')));
      }
      return Promise.resolve(jsonResp({ jobs: [], comparisons: [] }));
    });
    env.sandbox.__setUpload(u1);
    const pSlow = env.sandbox.focusComparisonSlot('job_slow');
    const pFast = env.sandbox.focusComparisonSlot('job_fast');
    dFast.resolve(jsonResp(mkJobRes('job_fast', 'up_r', 'sha_r', 'qwen3', 14)));
    await pFast;
    dSlow.resolve(jsonResp(mkJobRes('job_slow', 'up_r', 'sha_r', 'qwen2', 5)));
    await pSlow;
    assert.strictEqual(env.sandbox.__state().LIVE_JOB_ID, 'job_fast');

    // Restore running job -> cancel enabled, transient failures show recovery bar & resume button restores polling
    env.getEl('selRecentJob').value = 'job_run';
    await env.getEl('btnLoadRecentJob').onclick();
    assert.strictEqual(env.getEl('btnCancelJob').disabled, false);
    assert.strictEqual(env.getEl('btnStartJob').disabled, true);
    const tRun = env.sandbox.__state().LIVE_POLL_TIMER;
    const tick = env.activeTimers.get(tRun);
    await tick(); // transient #1 -> warning, timer still alive
    assert.strictEqual(env.getEl('pollRecoveryBar').style.display, 'block');
    assert(env.activeTimers.has(tRun));
    await tick(); // #2
    await tick(); // #3 -> pauses timer cleanly, cancel still enabled
    assert.strictEqual(env.getEl('btnCancelJob').disabled, false);
    assert(!env.activeTimers.has(tRun));
    // Click resume -> restarts polling and succeeds
    env.getEl('btnResumePoll').onclick();
    await new Promise(r => setTimeout(r, 10));
    assert(env.sandbox.__state().LIVE_POLL_TIMER !== null);
  }

  // 4. Same-session overlapping pollComparison / pollJob out-of-order terminal lock, single-flight tick, and cancel lock
  {
    const d1 = makeDeferred(), d2 = makeDeferred();
    let callIdx = 0;
    const u1 = mkUp('up_same_poll', 'sha_same_poll');
    const env = createSandbox((url, opts) => {
      if (url === '/api/comparisons' && opts.method === 'POST') return Promise.resolve(jsonResp(mkCmp('cmp_1', 'up_same_poll', 'sha_same_poll', 'running')));
      if (url === '/api/comparisons/cmp_1') { callIdx++; return callIdx === 1 ? d1.promise : d2.promise; }
      if (url === '/api/jobs/job_1/result') return Promise.resolve(jsonResp(mkJobRes('job_1', 'up_same_poll', 'sha_same_poll', 'qwen2', 5)));
      return Promise.resolve(jsonResp({ jobs: [], comparisons: [] }));
    });
    const renderHist = [];
    const origUpdate = env.sandbox.updateComparisonProgressUi;
    env.sandbox.updateComparisonProgressUi = (c) => { renderHist.push(c.status); return origUpdate(c); };
    env.sandbox.__setUpload(u1);
    await env.getEl('btnStartCompare').onclick();
    renderHist.length = 0;
    const s = env.sandbox.__state().LIVE_ACTIVE_SESSION;
    const p1 = env.sandbox.pollComparison(s, s.timerId);
    // Second timer tick while p1 is in-flight must be skipped by single-flight guard
    env.activeTimers.get(s.timerId)();
    assert.strictEqual(callIdx, 1);
    // Overlapping manual/second poll p2 resolves 'completed' before p1 resolves stale 'running'
    const p2 = env.sandbox.pollComparison(s, s.timerId);
    d2.resolve(jsonResp(mkCmp('cmp_1', 'up_same_poll', 'sha_same_poll', 'completed', 5, 'job_1')));
    await p2;
    d1.resolve(jsonResp(mkCmp('cmp_1', 'up_same_poll', 'sha_same_poll', 'running')));
    await p1;
    assert.deepStrictEqual(renderHist, ['completed']);
    assert.strictEqual(env.sandbox.__state().LIVE_COMPARISON.status, 'completed');
    assert.strictEqual(env.sandbox.__state().LIVE_POLL_TIMER, null);
    assert.strictEqual(s.terminalLocked, true);
  }

  // 5. Same-session pollJob out-of-order terminal lock + read-only upload disables start buttons
  {
    const d1 = makeDeferred(), d2 = makeDeferred();
    let callIdx = 0;
    const u1 = mkUp('up_j_poll', 'sha_j_poll');
    const env = createSandbox((url, opts) => {
      if (url === '/api/jobs' && opts.method === 'POST') return Promise.resolve(jsonResp(mkJobSt('job_p1', 'up_j_poll', 'sha_j_poll', 'running_base_inference')));
      if (url === '/api/jobs/job_p1') { callIdx++; return callIdx === 1 ? d1.promise : d2.promise; }
      if (url === '/api/jobs/job_p1/result') return Promise.resolve(jsonResp(mkJobRes('job_p1', 'up_j_poll', 'sha_j_poll', 'qwen2', 5)));
      return Promise.resolve(jsonResp({ jobs: [], comparisons: [] }));
    });
    const renderHist = [];
    const origJobUi = env.sandbox.updateProgressUi;
    env.sandbox.updateProgressUi = (st) => { renderHist.push(st.status); return origJobUi(st); };
    env.sandbox.__setUpload(u1);
    await env.getEl('btnStartJob').onclick();
    renderHist.length = 0;
    const s = env.sandbox.__state().LIVE_ACTIVE_SESSION;
    const p1 = env.sandbox.pollJob(s, s.timerId);
    const p2 = env.sandbox.pollJob(s, s.timerId);
    d2.resolve(jsonResp(mkJobSt('job_p1', 'up_j_poll', 'sha_j_poll', 'completed')));
    await p2;
    d1.resolve(jsonResp(mkJobSt('job_p1', 'up_j_poll', 'sha_j_poll', 'running_base_inference')));
    await p1;
    assert.deepStrictEqual(renderHist, ['completed']);
    assert.strictEqual(env.getEl('stBadge_qwen2').textContent, 'completed');

    // Read-only upload summary disables btnStartCompare and btnStartJob
    const uRo = Object.assign(mkUp('up_ro', 'sha_ro'), { read_only_playback_only: true, playback_preview_notice: '只读示例' });
    env.sandbox.__setUpload(uRo);
    env.sandbox.syncActionButtonsForSession(env.sandbox.__state().LIVE_ACTIVE_SESSION);
    assert.strictEqual(env.getEl('btnStartCompare').disabled, true);
    assert.strictEqual(env.getEl('btnStartJob').disabled, true);
  }
})();
""",
        encoding="utf-8",
    )
    proc = subprocess.run([node_bin, str(js_runner), str(demo_script_path)], capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, f"Frontend JS VM regression failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


def test_read_only_playback_upload_rejects_both_jobs_and_comparisons_without_creating_dirs(tmp_path: pathlib.Path) -> None:
    png_bytes = _make_board_png(tmp_path / "board_ro.png")
    svc = LiveDemoService(workspace_root=tmp_path / "ws_ro", job_runner=_TwoStageFakeRunner())
    up = svc.register_upload(raw_bytes=png_bytes, raw_filename="board_ro.png")
    uid = up["upload_id"]
    meta_p = svc.uploads_dir / uid / "upload_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["read_only_playback_only"] = True
    meta["playback_preview_notice"] = "只读回放样例，不可重推。"
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    pub = svc.get_upload_summary(uid)
    assert pub["read_only_playback_only"] is True
    assert pub["playback_preview_notice"] == "只读回放样例，不可重推。"

    jobs_before = list(svc.jobs_dir.iterdir())
    cmps_before = list(svc.comparisons_dir.iterdir())

    with pytest.raises(Exception) as exc_job:
        svc.submit_job(upload_id=uid, model_key="qwen2", mode="W")
    assert getattr(exc_job.value, "status_code", None) == 400

    with pytest.raises(Exception) as exc_cmp:
        svc.submit_comparison(upload_id=uid, model_keys=["qwen2", "deepseek", "qwen3"], mode="W")
    assert getattr(exc_cmp.value, "status_code", None) == 400

    assert list(svc.jobs_dir.iterdir()) == jobs_before
    assert list(svc.comparisons_dir.iterdir()) == cmps_before


def test_portable_config_relative_to_anchor_resolves_from_unrelated_cwd(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    demo_script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "fics_final_demo.py"
    spec = importlib.util.spec_from_file_location("fics_final_demo_portable_test", demo_script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pkg_root = tmp_path / "copied package with space"
    cfg_dir = pkg_root / "software" / "configs"
    ws_dir = pkg_root / "evidence" / "demo_live_workspace"
    idx_file = pkg_root / "evidence" / "results_index.portable.json"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ws_dir.mkdir(parents=True, exist_ok=True)
    idx_file.write_text("{}", encoding="utf-8")

    cfg_file = cfg_dir / "fics_final_demo.portable.json"
    cfg_file.write_text(
        json.dumps({
            "relative_to": "software_dir",
            "workspace_root": "../evidence/demo_live_workspace",
            "runtime_config": None,
            "results_index": "../evidence/results_index.portable.json",
            "host": "127.0.0.1",
            "port": 7862,
        }),
        encoding="utf-8",
    )

    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(unrelated_cwd)

    opts = mod.resolve_server_launch_options(["--config", str(cfg_file)])
    assert opts["workspace_root"] == ws_dir.resolve()
    assert opts["results_index_path"] == idx_file.resolve()
