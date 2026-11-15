"""Supervised subprocess runner for GT-free second-stage candidate crop verification."""

from __future__ import annotations

import gc
import json
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from pcb_vlm.data.fics_dslr import file_sha256
from pcb_vlm.evaluation.candidate_verifier import (
    parse_single_label_json,
    verify_verifier_manifest_integrity,
)
from pcb_vlm.inference.fics_matrix import (
    _append_jsonl,
    _kill_and_wait_worker,
    _read_next_ipc_event,
    _stream_worker_stderr,
    _stream_worker_stdout,
)
from pcb_vlm.inference.gpu_gate import probe_gpu_gate

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _emit_ipc(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def run_verifier_worker(manifest_path: pathlib.Path) -> int:
    """Isolated GPU worker reading neutral unique-crop call_ids from stdin."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest["inference_spec"]
    prompt = str(spec["prompt"])
    prompt_sha = str(spec["prompt_sha256"]).lower()
    allowed_labels = tuple(spec["allowed_labels"])
    calls_by_id = {str(e["call_id"]): e for e in manifest["unique_calls"]}

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ov_dir = str(spec["overlay_dir"])
    if ov_dir not in sys.path:
        sys.path.insert(0, ov_dir)

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

    assert os.path.normcase(os.path.abspath(ov_dir)) in os.path.normcase(os.path.abspath(transformers.__file__)), (
        f"Expected overlay transformers under {ov_dir}, got {transformers.__file__}"
    )

    t_load0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(spec["model_local_dir"]),
        size={"shortest_edge": int(spec["min_pixels"]), "longest_edge": int(spec["max_pixels"])},
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(spec["model_local_dir"]),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda:0")
    model.eval()
    torch.cuda.synchronize(0)
    load_s = round(time.perf_counter() - t_load0, 3)

    eos_ids = [151645, 151643]
    base_gen_cfg = GenerationConfig.from_model_config(model.config)
    _emit_ipc({
        "event": "loaded",
        "load_seconds": load_s,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model_local_dir": str(spec["model_local_dir"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })

    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        req = json.loads(s)
        if req.get("cmd") == "shutdown":
            break
        entry = calls_by_id[str(req["call_id"])]
        _emit_ipc({
            "event": "call_started",
            "call_id": entry["call_id"],
            "sample_index": entry["sample_index"],
            "neutral_filename": entry["neutral_filename"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

        img_p = pathlib.Path(entry["image_path"])
        actual_sha = file_sha256(img_p).lower()
        if actual_sha != str(entry["image_sha256"]).lower():
            raise RuntimeError(f"Worker SHA mismatch for {img_p}: {actual_sha}")

        with Image.open(img_p) as im_raw:
            pil_img = im_raw.convert("RGB")
        assert pil_img.size == (entry["width"], entry["height"])

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }]
        text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_cpu = processor(text=[text_in], images=[pil_img], return_tensors="pt")
        patch_size = int(getattr(processor.image_processor, "patch_size", 16))
        merge_size = int(getattr(processor.image_processor, "merge_size", 2))
        grid = [int(x) for x in inputs_cpu["image_grid_thw"][0].tolist()]
        proc_w = grid[2] * patch_size
        proc_h = grid[1] * patch_size
        merged_vis_toks = int((grid[1] // merge_size) * (grid[2] // merge_size))

        inputs_cuda = inputs_cpu.to("cuda:0")
        del inputs_cpu
        in_toks = int(inputs_cuda["input_ids"].shape[1])

        call_gen_cfg = GenerationConfig.from_dict(base_gen_cfg.to_dict())
        call_gen_cfg.max_new_tokens = int(spec["max_new_tokens"])
        call_gen_cfg.do_sample = False
        call_gen_cfg.repetition_penalty = 1.0
        call_gen_cfg.eos_token_id = eos_ids
        call_gen_cfg.bos_token_id = None
        call_gen_cfg.pad_token_id = 151643
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(call_gen_cfg, attr):
                setattr(call_gen_cfg, attr, None)

        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = model.generate(**inputs_cuda, generation_config=call_gen_cfg, use_cache=True)
        torch.cuda.synchronize(0)
        gpu_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        gen_ids = out_ids[0][in_toks:].detach().cpu().tolist()
        raw_out = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
        has_eos = bool(gen_ids and gen_ids[-1] in eos_ids)
        parsed = parse_single_label_json(raw_out, allowed_labels=allowed_labels)
        peak_mib = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)

        del inputs_cuda, out_ids
        gc.collect()
        torch.cuda.empty_cache()

        rec = {
            "call_id": entry["call_id"],
            "sample_index": entry["sample_index"],
            "neutral_filename": entry["neutral_filename"],
            "input_sha256": actual_sha,
            "prompt_sha256": prompt_sha,
            "model_repo_id": spec["model_repo_id"],
            "model_revision": spec["model_revision"],
            "processed_width_px": proc_w,
            "processed_height_px": proc_h,
            "processed_area_px": proc_w * proc_h,
            "merged_visual_tokens": merged_vis_toks,
            "generated_tokens": len(gen_ids),
            "has_normal_eos": has_eos,
            "raw_output": raw_out,
            "strict_json_ok": parsed["strict_json_ok"],
            "strict_label": parsed["strict_label"],
            "compat_regex_label": parsed["compat_regex_label"],
            "parse_error": parsed["parse_error"],
            "gpu_generate_ms": gpu_ms,
            "cuda_max_allocated_mib": peak_mib,
            "ok": True,
        }
        _emit_ipc({
            "event": "call_finished",
            "call_id": entry["call_id"],
            "sample_index": entry["sample_index"],
            "ok": True,
            "record": rec,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
    return 0


def run_verifier_supervised(
    *,
    manifest_path: pathlib.Path,
    output_dir: pathlib.Path,
    gpu_gate_fn: Callable[[float, float], Dict[str, Any]] = probe_gpu_gate,
    worker_cmd: Optional[Sequence[str]] = None,
    load_timeout_s: float = 180.0,
    per_call_timeout_s: float = 90.0,
    stage_timeout_s: float = 3600.0,
    execute: bool = False,
) -> Dict[str, Any]:
    """Run all unique verifier calls under dual VRAM gate and hard parent timeouts."""
    t_stage0 = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = verify_verifier_manifest_integrity(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "verifier_raw_outputs.jsonl"
    stage_jsonl = output_dir / "verifier_stage_events.jsonl"
    gate_json = output_dir / "verifier_gpu_gate.json"
    summary_json = output_dir / "verifier_session_summary.json"
    stderr_log = output_dir / "verifier_worker_stderr.log"

    for p in (raw_jsonl, stage_jsonl, gate_json, summary_json):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite pre-existing verifier output: {p}")

    unique_calls = manifest["unique_calls"]
    if not execute:
        return {
            "dry_run": True,
            "verified": verified,
            "planned_unique_calls": len(unique_calls),
            "output_dir": str(output_dir),
        }

    gate = gpu_gate_fn(5500.0, 6144.0)
    gate_json.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not bool(gate.get("passed")):
        summary = {
            "stage_status": "gate_failed",
            "planned_unique_calls": len(unique_calls),
            "attempted": 0,
            "completed": 0,
            "failed_or_timeout": 0,
            "not_started": len(unique_calls),
            "wall_seconds": round(time.perf_counter() - t_stage0, 3),
            "gpu_gate": gate,
            "verified_inputs": verified,
        }
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary

    runner_script = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "run_candidate_verifier.py"
    cmd = list(worker_cmd) if worker_cmd is not None else [
        sys.executable,
        "-u",
        str(runner_script),
        "--worker",
        "--manifest",
        str(manifest_path),
    ]
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_SRC_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(_SRC_ROOT)

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )
    out_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    t_out = threading.Thread(target=_stream_worker_stdout, args=(proc, out_q), daemon=True)
    t_err = threading.Thread(target=_stream_worker_stderr, args=(proc, stderr_log), daemon=True)
    t_out.start()
    t_err.start()

    records: List[Dict[str, Any]] = []
    stage_status = "completed"
    abort_reason: Optional[str] = None
    load_seconds: Optional[float] = None
    stage_deadline = time.perf_counter() + float(stage_timeout_s)
    attempted = 0

    try:
        load_deadline = min(stage_deadline, time.perf_counter() + float(load_timeout_s))
        loaded_ok = False
        while True:
            kind, msg = _read_next_ipc_event(out_q, load_deadline)
            if kind == "msg" and msg is not None:
                _append_jsonl(stage_jsonl, msg)
                if msg.get("event") == "loaded":
                    loaded_ok = True
                    load_seconds = msg.get("load_seconds")
                    break
                continue
            kill_info = _kill_and_wait_worker(proc)
            stage_status = "load_timeout" if kind == "timeout" else "load_failed"
            abort_reason = f"Worker failed before loaded ({kind}): kill={kill_info}"
            _append_jsonl(stage_jsonl, {"event": "supervisor_load_aborted", "kind": kind, "kill_info": kill_info})
            break

        if loaded_ok:
            total_u = len(unique_calls)
            for entry in unique_calls:
                if time.perf_counter() >= stage_deadline:
                    kill_info = _kill_and_wait_worker(proc)
                    stage_status = "stage_timeout"
                    abort_reason = "Exceeded verifier overall stage timeout"
                    _append_jsonl(stage_jsonl, {"event": "supervisor_stage_timeout", "kill_info": kill_info})
                    break

                attempted += 1
                _append_jsonl(stage_jsonl, {
                    "event": "supervisor_call_dispatch",
                    "call_id": entry["call_id"],
                    "sample_index": entry["sample_index"],
                })
                assert proc.stdin is not None
                proc.stdin.write(json.dumps({"call_id": entry["call_id"]}, ensure_ascii=False) + "\n")
                proc.stdin.flush()

                call_deadline = min(stage_deadline, time.perf_counter() + float(per_call_timeout_s))
                got_finished = False
                while True:
                    kind, msg = _read_next_ipc_event(out_q, call_deadline)
                    if kind == "msg" and msg is not None:
                        if msg.get("event") == "call_finished":
                            rec = msg["record"]
                            _append_jsonl(stage_jsonl, {k: v for k, v in msg.items() if k != "record"})
                            _append_jsonl(raw_jsonl, rec)
                            records.append(rec)
                            got_finished = True
                            if entry["sample_index"] % 25 == 0 or entry["sample_index"] == total_u:
                                print(
                                    f"[{entry['sample_index']:03d}/{total_u}] {entry['neutral_filename']} "
                                    f"strict={rec['strict_json_ok']} lbl={rec['strict_label']} "
                                    f"{rec['gpu_generate_ms']:.0f}ms",
                                    flush=True,
                                )
                            break
                        _append_jsonl(stage_jsonl, msg)
                        continue
                    kill_info = _kill_and_wait_worker(proc)
                    is_to = kind == "timeout"
                    stage_status = "call_timeout" if is_to else "worker_crashed"
                    abort_reason = f"Worker terminated during call {entry['call_id']} ({kind}): kill={kill_info}"
                    fail_rec = {
                        "call_id": entry["call_id"],
                        "sample_index": entry["sample_index"],
                        "neutral_filename": entry["neutral_filename"],
                        "input_sha256": entry["image_sha256"],
                        "prompt_sha256": manifest["inference_spec"]["prompt_sha256"],
                        "ok": False,
                        "is_timeout": is_to,
                        "watchdog_killed": True,
                        "kill_info": kill_info,
                        "error": abort_reason,
                        "raw_output": None,
                        "strict_json_ok": False,
                        "strict_label": None,
                        "compat_regex_label": None,
                    }
                    _append_jsonl(raw_jsonl, fail_rec)
                    _append_jsonl(
                        stage_jsonl,
                        {"event": "supervisor_call_aborted", "call_id": entry["call_id"], "kind": kind, "kill_info": kill_info},
                    )
                    records.append(fail_rec)
                    break

                if not got_finished:
                    break
    finally:
        if proc.poll() is None:
            try:
                if stage_status == "completed" and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    proc.stdin.flush()
                    proc.wait(timeout=5.0)
            except Exception:
                pass
            if proc.poll() is None:
                _kill_and_wait_worker(proc)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

    ok_rows = [r for r in records if r.get("ok")]
    summary = {
        "stage_status": stage_status,
        "abort_reason": abort_reason,
        "planned_unique_calls": len(unique_calls),
        "attempted": attempted,
        "completed": len(ok_rows),
        "failed_or_timeout": sum(1 for r in records if not r.get("ok")),
        "not_started": max(0, len(unique_calls) - attempted),
        "strict_json_ok_count": sum(1 for r in ok_rows if r.get("strict_json_ok")),
        "compat_regex_recovered_count": sum(
            1 for r in ok_rows if not r.get("strict_json_ok") and r.get("compat_regex_label") is not None
        ),
        "normal_eos_count": sum(1 for r in ok_rows if r.get("has_normal_eos")),
        "non_eos_ok_count": sum(1 for r in ok_rows if not r.get("has_normal_eos")),
        "load_seconds": load_seconds,
        "wall_seconds": round(time.perf_counter() - t_stage0, 3),
        "total_gpu_generate_ms": round(sum(float(r.get("gpu_generate_ms") or 0.0) for r in ok_rows), 2),
        "max_cuda_allocated_mib": max((float(r.get("cuda_max_allocated_mib") or 0.0) for r in ok_rows), default=None),
        "gpu_gate": gate,
        "verified_inputs": verified,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
