"""CPU unit and fake-worker integration tests for the integrated FICS matrix runner."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
for _p in (str(_SRC_ROOT), str(_SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_fics_matrix  # noqa: E402
from pcb_vlm.evaluation.fics_replay import sha256_text  # noqa: E402
from pcb_vlm.inference.fics_matrix import (  # noqa: E402
    build_formal_calls_for_model,
    determine_matrix_exit_code,
    run_sequential_matrix,
)

PROMPT_TEXT = (
    "Locate all visible electronic components belonging to these classes: "
    "capacitor, resistor, inductor, transistor, diode, ic. "
    "Return a JSON array of objects with keys bbox_2d and label. "
    "Each bbox_2d must be [xmin, ymin, xmax, ymax], normalized to [0, 1000] relative to this input image. "
    "Use exactly one of the listed class names for each object. "
    "If no target components are visible, return []. "
    "Do not include explanations."
)


def _make_sample_plan(tmp_dir: pathlib.Path, *, call_count: int = 3, board_id: str = "s99_front") -> pathlib.Path:
    img_dir = tmp_dir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    queue_entries = []
    for idx in range(call_count):
        mode = "W" if idx == 0 else "T"
        tag = f"{board_id}_{mode}_{idx:02d}"
        img_p = img_dir / f"{tag}.png"
        payload = f"fake-image-{tag}".encode("utf-8")
        img_p.write_bytes(payload)
        queue_entries.append({
            "call_index": idx,
            "call_id": f"custom_call_{idx}",
            "board_id": board_id,
            "tag": tag,
            "mode": mode,
            "image_id": 99,
            "tile_id": None if mode == "W" else f"tile_{idx:02d}",
            "path": str(img_p),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "width": 640,
            "height": 480,
            "roi_offset_x": 10,
            "roi_offset_y": 20,
            "tile_offset_x": 0 if mode == "W" else idx * 100,
            "tile_offset_y": 0,
        })

    plan = {
        "schema_version": "1.0",
        "model_execution_order": ["qwen2", "qwen3"],
        "models": {
            "qwen2": {"repo_id": "Qwen/Qwen2-VL-2B-Instruct", "local_dir": "/dummy/qwen2"},
            "qwen3": {"repo_id": "Qwen/Qwen3-VL-2B-Instruct", "local_dir": "/dummy/qwen3"},
        },
        "inference_config": {
            "dtype": "bfloat16",
            "do_sample": False,
            "repetition_penalty": 1.0,
            "max_new_tokens": 512,
            "qwen_min_pixels": 65536,
            "qwen_max_pixels": 200704,
            "coordinate_protocol": "xyxy_norm1000",
            "prompt": PROMPT_TEXT,
            "prompt_sha256": sha256_text(PROMPT_TEXT),
        },
        "resource_limits": {
            "gpu_gate_min_smi_free_mib": 5500.0,
            "gpu_gate_min_torch_free_mib": 6144.0,
            "model_load_timeout_seconds": 5.0,
            "per_call_timeout_seconds": 2.0,
            "per_model_gpu_timeout_seconds": 15.0,
            "total_gpu_timeout_seconds": 30.0,
        },
        "boards": {
            board_id: {"image_id": 99, "roi_offset_x": 10, "roi_offset_y": 20},
        },
        "queues": {
            "qwen2": queue_entries,
            "qwen3": queue_entries,
        },
    }
    plan_path = tmp_dir / "custom_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan_path


def _write_fake_worker(tmp_dir: pathlib.Path) -> pathlib.Path:
    worker = tmp_dir / "fake_worker.py"
    worker.write_text(
        """
import json, sys, time
mode = sys.argv[1]
print(json.dumps({"event": "loaded", "model_key": sys.argv[2]}), flush=True)
for line in sys.stdin:
    s = line.strip()
    if not s:
        continue
    req = json.loads(s)
    if req.get("cmd") == "quit":
        break
    if mode == "hang_second" and req["call_index"] == 1:
        print(json.dumps({"event": "stage_event", "stage": "generation_started", "call_id": req["call_id"], "data": {}}), flush=True)
        time.sleep(10.0)
    elif mode == "hang_after_verified" and req["call_index"] == 1:
        print(json.dumps({
            "event": "stage_event",
            "stage": "input_verified",
            "call_id": req["call_id"],
            "data": {"input_sha256": req["expected_sha256"]},
        }), flush=True)
        print(json.dumps({"event": "stage_event", "stage": "generation_started", "call_id": req["call_id"], "data": {}}), flush=True)
        time.sleep(10.0)
    elif mode == "crash_second" and req["call_index"] == 1:
        sys.exit(3)
    print(json.dumps({
        "event": "result",
        "source": "fake_worker",
        "call_index": req["call_index"],
        "call_id": req["call_id"],
        "board_id": req["board_id"],
        "mode": req["mode"],
        "image_id": req["image_id"],
        "tile_id": req["tile_id"],
        "image_path": req["image_path"],
        "input_sha256": req["expected_sha256"],
        "expected_sha256": req["expected_sha256"],
        "orig_input_width": req["width"],
        "orig_input_height": req["height"],
        "roi_offset_x": req["roi_offset_x"],
        "roi_offset_y": req["roi_offset_y"],
        "tile_offset_x": req["tile_offset_x"],
        "tile_offset_y": req["tile_offset_y"],
        "max_new_tokens": 512,
        "prompt_sha256": req["prompt_sha256"],
        "coordinate_protocol": "xyxy_norm1000",
        "model_key": sys.argv[2],
        "model_repo_id": "Qwen/Qwen2-VL-2B-Instruct" if sys.argv[2] == "qwen2" else "Qwen/Qwen3-VL-2B-Instruct",
        "model_revision": None,
        "generated_tokens": 2,
        "generated_token_ids": [1, 151645],
        "finish_reason": "eos_token",
        "ok": True,
        "raw_output": "[]",
    }), flush=True)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return worker


class TestFicsMatrixRunner(unittest.TestCase):
    def test_dry_run_arbitrary_board_and_variable_queue_with_path_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=4, board_id="custom_mk6_board")
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
            # Relocate images and rewrite plan paths to simulate historical cache locations
            reloc_dir = tpath / "relocated_imgs"
            reloc_dir.mkdir()
            for p in (tpath / "imgs").iterdir():
                p.replace(reloc_dir / p.name)
            (tpath / "imgs").rmdir()
            for mk in ("qwen2", "qwen3"):
                for q in plan_data["queues"][mk]:
                    q["path"] = "Z:/old/cache/imgs/" + pathlib.Path(q["path"]).name
            plan_path.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

            calls = build_formal_calls_for_model(
                plan_data,
                "qwen2",
                path_remaps=[("Z:/old/cache/imgs", str(reloc_dir))],
            )
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[0]["board_id"], "custom_mk6_board")
            self.assertTrue(pathlib.Path(calls[0]["image_path"]).is_file())

            out_dir = tpath / "dry_out"
            rc = run_fics_matrix.main([
                "--plan", str(plan_path),
                "--output-dir", str(out_dir),
                "--path-map", f"Z:/old/cache/imgs={reloc_dir}",
                "--dry-run",
                "--verify-input-hashes",
            ])
            self.assertEqual(rc, 0)
            self.assertFalse((out_dir / "matrix_session_summary.json").exists())

    def test_refuses_preexisting_output_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=2)
            out_dir = tpath / "existing_out"
            (out_dir / "run_qwen2").mkdir(parents=True)
            (out_dir / "run_qwen2" / "raw_outputs_all.jsonl").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_sequential_matrix(plan_path=plan_path, output_dir=out_dir, dry_run=True)

    def test_fail_fast_when_gpu_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=2)
            out_dir = tpath / "gate_fail_out"
            res = run_sequential_matrix(
                plan_path=plan_path,
                output_dir=out_dir,
                dry_run=False,
                gpu_gate_fn=lambda s, m: {"passed": False, "reason": "simulated_low_vram"},
            )
            self.assertEqual(res["exit_code"], 1)
            self.assertEqual(res["stage_summaries"]["qwen2"]["status"], "gate_failed")
            self.assertTrue((out_dir / "run_qwen2" / "gpu_gate.json").is_file())

    def test_fake_worker_complete_success_and_exit_code_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=3)
            worker = _write_fake_worker(tpath)
            out_dir = tpath / "ok_out"
            res = run_sequential_matrix(
                plan_path=plan_path,
                output_dir=out_dir,
                dry_run=False,
                gpu_gate_fn=lambda s, m: {"passed": True},
                worker_cmd_builder=lambda mk, _mc, _rp: [sys.executable, "-u", str(worker), "ok", mk],
            )
            self.assertEqual(res["exit_code"], 0)
            for mk in ("qwen2", "qwen3"):
                self.assertEqual(res["stage_summaries"][mk]["status"], "completed")
                self.assertEqual(res["stage_summaries"][mk]["ok_calls"], 3)

    def test_watchdog_timeout_and_worker_eof_record_provenance_and_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=3)
            worker = _write_fake_worker(tpath)

            res_to = run_sequential_matrix(
                plan_path=plan_path,
                output_dir=tpath / "timeout_out",
                models_filter=["qwen2"],
                dry_run=False,
                per_call_timeout_s=0.4,
                gpu_gate_fn=lambda s, m: {"passed": True},
                worker_cmd_builder=lambda mk, _mc, _rp: [sys.executable, "-u", str(worker), "hang_second", mk],
            )
            self.assertEqual(res_to["exit_code"], 1)
            st_to = res_to["stage_summaries"]["qwen2"]
            self.assertEqual(st_to["status"], "call_timeout")
            self.assertEqual(st_to["ok_calls"], 1)
            self.assertEqual(st_to["failed_calls"], 1)
            raw_to = [
                json.loads(line)
                for line in pathlib.Path(st_to["raw_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertIsNone(raw_to[1]["input_sha256"])
            self.assertEqual(raw_to[1]["input_verification_status"], "unverified_pre_input_failure")
            self.assertEqual(raw_to[1]["input_sha256_source"], "not_verified_pre_input_failure")

            res_verified_to = run_sequential_matrix(
                plan_path=plan_path,
                output_dir=tpath / "timeout_verified_out",
                models_filter=["qwen2"],
                dry_run=False,
                per_call_timeout_s=0.4,
                gpu_gate_fn=lambda s, m: {"passed": True},
                worker_cmd_builder=lambda mk, _mc, _rp: [sys.executable, "-u", str(worker), "hang_after_verified", mk],
            )
            self.assertEqual(res_verified_to["exit_code"], 1)
            st_vto = res_verified_to["stage_summaries"]["qwen2"]
            raw_vto = [
                json.loads(line)
                for line in pathlib.Path(st_vto["raw_jsonl"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(raw_vto[1]["input_sha256"], raw_vto[1]["expected_sha256"])
            self.assertEqual(raw_vto[1]["input_verification_status"], "verified_by_worker_before_failure")
            self.assertEqual(raw_vto[1]["input_sha256_source"], "worker_stage_input_verified")

            res_eof = run_sequential_matrix(
                plan_path=plan_path,
                output_dir=tpath / "eof_out",
                models_filter=["qwen2"],
                dry_run=False,
                gpu_gate_fn=lambda s, m: {"passed": True},
                worker_cmd_builder=lambda mk, _mc, _rp: [sys.executable, "-u", str(worker), "crash_second", mk],
            )
            self.assertEqual(res_eof["exit_code"], 1)
            self.assertEqual(res_eof["stage_summaries"]["qwen2"]["status"], "worker_eof")

    def test_default_cli_and_api_never_touch_gpu_gate_or_worker(self) -> None:
        from unittest import mock
        import io, contextlib
        _SCRIPTS_ROOT = _REPO_ROOT / "scripts"
        if str(_SCRIPTS_ROOT) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_ROOT))
        import run_fics_matrix as runner_cli

        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=2)

            with (
                mock.patch("pcb_vlm.inference.fics_matrix.probe_gpu_gate", side_effect=AssertionError("GPU gate must not run in default dry-run")),
                mock.patch("pcb_vlm.inference.fics_matrix.subprocess.Popen", side_effect=AssertionError("Worker must not launch in default dry-run")),
            ):
                res_api = run_sequential_matrix(plan_path=plan_path, output_dir=tpath / "api_default_out")
                self.assertTrue(res_api["dry_run"])
                self.assertEqual(res_api["exit_code"], 0)

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc_cli = runner_cli.main([
                        "--plan", str(plan_path),
                        "--output-dir", str(tpath / "cli_default_out"),
                    ])
                self.assertEqual(rc_cli, 0)
                cli_payload = json.loads(buf.getvalue())
                self.assertTrue(cli_payload["dry_run"])
                self.assertEqual(cli_payload["execution_mode"], "cpu_dry_run_default")

                # --dry-run and --execute must be mutually exclusive
                with self.assertRaises(SystemExit) as se:
                    runner_cli.main([
                        "--plan", str(plan_path),
                        "--output-dir", str(tpath / "cli_conflict_out"),
                        "--dry-run",
                        "--execute",
                    ])
                self.assertNotEqual(se.exception.code, 0)

    def test_apply_runtime_overrides_rejects_frozen_identity_and_config_changes(self) -> None:
        from pcb_vlm.inference.fics_matrix import apply_runtime_overrides

        with tempfile.TemporaryDirectory() as tmp:
            tpath = pathlib.Path(tmp)
            plan_path = _make_sample_plan(tpath, call_count=1)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            # Add frozen fields explicitly on qwen2
            plan["models"]["qwen2"].update({
                "revision": None,
                "attn_implementation": "sdpa",
                "visual_budget": {"min_pixels": 65536, "max_pixels": 200704},
            })

            # Localization overrides work cleanly
            ok_merged = apply_runtime_overrides(plan, {
                "models": {
                    "qwen2": {
                        "repo_id": plan["models"]["qwen2"]["repo_id"],
                        "revision": None,
                        "attn_implementation": "sdpa",
                        "local_dir": "/local/models/qwen2",
                        "overlay_dir": "/local/overlay",
                        "hf_home": "/local/hf",
                        "python_executable": "/local/bin/python",
                        "require_overlay": False,
                    }
                }
            })
            self.assertEqual(ok_merged["models"]["qwen2"]["local_dir"], "/local/models/qwen2")

            # Frozen identity/budget/top-level tamper attempts must fail
            bad_cfgs = [
                {"models": {"qwen2": {"repo_id": "Evil/OtherModel"}}},
                {"models": {"qwen2": {"revision": "tampered-rev"}}},
                {"models": {"qwen2": {"attn_implementation": "flash_attention_2"}}},
                {"models": {"qwen2": {"visual_budget": {"min_pixels": 1, "max_pixels": 2}}}},
                {"models": {"unknown_model": {"local_dir": "/x"}}},
                {"inference_config": {"max_new_tokens": 9999}},
                {"queues": {"qwen2": []}},
                {"boards": {"tampered": {}}},
            ]
            for bad in bad_cfgs:
                with self.assertRaises(ValueError, msg=f"Should reject runtime override: {bad}"):
                    apply_runtime_overrides(plan, bad)

    def test_determine_matrix_exit_code_contract(self) -> None:
        ok_summary = {"status": "completed", "planned_calls": 2, "ok_calls": 2, "failed_calls": 0}
        self.assertEqual(determine_matrix_exit_code({"qwen2": ok_summary}, ["qwen2"]), 0)
        self.assertEqual(determine_matrix_exit_code({"qwen2": {**ok_summary, "failed_calls": 1}}, ["qwen2"]), 1)
        self.assertEqual(determine_matrix_exit_code({"qwen2": ok_summary}, ["qwen2", "qwen3"]), 1)


if __name__ == "__main__":
    unittest.main()

