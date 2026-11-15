"""Sequential subprocess-isolated multi-model inference supervisor for FICS-PCB.

Adapted into the repository from the verified multi-board matrix supervisor.
Generalizes queue/board handling, parameterizes runtime paths via config/CLI,
enforces fail-fast dual-gate VRAM checks before worker spawn, enforces anti-overwrite
output checks, and records watchdog/EOF provenance without silent zero exits.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from pcb_vlm.evaluation.fics_replay import (
    build_expected_spec_from_plan,
    build_image_hash_index,
    load_fics_six_class_config,
    resolve_portable_image_path,
)
from pcb_vlm.inference.gpu_gate import probe_gpu_gate, sha256_file

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
DEFAULT_RUNTIME_CONFIG_EXAMPLE = _REPO_ROOT / "configs" / "fics_local_runtime.example.yaml"


def load_runtime_config(path: Optional[pathlib.Path]) -> Dict[str, Any]:
    """Load optional local runtime YAML configuration."""
    if path is None:
        return {}
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Runtime config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config must be a YAML mapping: {p}")
    return data


def apply_runtime_overrides(
    plan: Dict[str, Any],
    runtime_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a deep-copied plan with localization-only fields merged from runtime config.

    Frozen experimental configuration (``inference_config``, ``queues``, ``boards``,
    ``resource_limits``, and model ``repo_id`` / ``revision`` / ``attn_implementation`` /
    ``visual_budget``) cannot be altered via runtime overrides.
    """
    out = copy.deepcopy(plan)
    if not runtime_cfg:
        return out
    if not isinstance(runtime_cfg, dict):
        raise ValueError("runtime_cfg must be a dict or None.")
    for frozen_top in ("inference_config", "queues", "boards", "resource_limits"):
        if frozen_top in runtime_cfg and runtime_cfg[frozen_top] != out.get(frozen_top):
            raise ValueError(
                f"Runtime config cannot override frozen plan field {frozen_top!r}."
            )
    rt_models = runtime_cfg.get("models") or {}
    if not isinstance(rt_models, dict):
        raise ValueError("runtime_cfg['models'] must be a dict.")
    models_map = out.setdefault("models", {})
    for mkey, rt_spec in rt_models.items():
        if not isinstance(rt_spec, dict):
            continue
        if mkey not in models_map:
            raise ValueError(f"Runtime config references unknown model key {mkey!r}.")
        target = models_map[mkey]
        for locked_field in ("repo_id", "revision", "attn_implementation", "visual_budget"):
            if locked_field in rt_spec and rt_spec[locked_field] != target.get(locked_field):
                raise ValueError(
                    f"Runtime config cannot change frozen {mkey}.{locked_field}: "
                    f"{rt_spec[locked_field]!r} != {target.get(locked_field)!r}"
                )
        for field in ("local_dir", "overlay_dir", "hf_home", "python_executable", "require_overlay"):
            if rt_spec.get(field) is not None:
                target[field] = rt_spec[field]
    return out


def _resolve_plan_image_path(
    q: Dict[str, Any],
    *,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_hash_index: Optional[Dict[str, pathlib.Path]] = None,
) -> pathlib.Path:
    row_like = {
        "image_path": q.get("path") or q.get("image_path"),
        "expected_sha256": q.get("sha256") or q.get("expected_sha256"),
    }
    resolved = resolve_portable_image_path(
        row_like,
        expected_call_spec={"path": row_like["image_path"]},
        path_remaps=path_remaps,
        image_hash_index=image_hash_index,
    )
    return resolved if resolved is not None else pathlib.Path(str(row_like["image_path"] or ""))


def build_formal_calls_for_model(
    plan: Dict[str, Any],
    model_key: str,
    *,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_hash_index: Optional[Dict[str, pathlib.Path]] = None,
) -> List[Dict[str, Any]]:
    """Build ordered inference call payloads for any board/queue layout inside `plan`."""
    spec = build_expected_spec_from_plan(plan, models_filter=[model_key])
    queues = plan.get("queues") or {}
    if model_key in queues and isinstance(queues[model_key], list):
        raw_q = queues[model_key]
    elif isinstance(queues.get("formal_67"), list):
        raw_q = queues["formal_67"]
    else:
        raise ValueError(f"Plan has no formal queue for model {model_key!r}")

    inf_cfg = plan["inference_config"]
    prompt = str(inf_cfg.get("prompt") or load_fics_six_class_config()["protocol"]["prompt"])
    prompt_sha = str(spec["prompt_sha256"])

    calls: List[Dict[str, Any]] = []
    for idx, q in enumerate(raw_q):
        cid = str(q.get("call_id") or f"{model_key}_formal_{q['board_id']}_{q['tag']}")
        board_id = str(q["board_id"])
        board_info = (plan.get("boards") or {}).get(board_id) or {}
        mode = str(q.get("mode") or "")
        tile_id = q.get("tile_id")
        if mode == "W":
            roi_ox = int(q.get("roi_offset_x", board_info.get("roi_offset_x", 0)))
            roi_oy = int(q.get("roi_offset_y", board_info.get("roi_offset_y", 0)))
            tile_ox = 0
            tile_oy = 0
        elif mode == "T":
            roi_ox = int(q.get("roi_offset_x", board_info.get("roi_offset_x", 0)))
            roi_oy = int(q.get("roi_offset_y", board_info.get("roi_offset_y", 0)))
            tile_ox = int(q.get("tile_offset_x", 0))
            tile_oy = int(q.get("tile_offset_y", 0))
        else:
            raise ValueError(f"Unexpected call mode {mode!r} in call {cid!r}")

        resolved_path = _resolve_plan_image_path(
            q,
            path_remaps=path_remaps,
            image_hash_index=image_hash_index,
        )
        calls.append({
            "call_index": int(q.get("call_index", idx)),
            "call_id": cid,
            "board_id": board_id,
            "mode": mode,
            "image_id": int(q.get("image_id", board_info.get("image_id", 0))),
            "tile_id": tile_id,
            "image_path": str(resolved_path),
            "original_plan_path": str(q.get("path") or q.get("image_path") or ""),
            "expected_sha256": str(q.get("sha256") or q.get("expected_sha256") or ""),
            "width": int(q["width"]),
            "height": int(q["height"]),
            "roi_offset_x": roi_ox,
            "roi_offset_y": roi_oy,
            "tile_offset_x": tile_ox,
            "tile_offset_y": tile_oy,
            "prompt": prompt,
            "prompt_sha256": prompt_sha,
        })
    return calls


def _norm_identities(path: pathlib.Path) -> Tuple[set[str], Optional[Tuple[int, int]]]:
    norms = {
        os.path.normcase(os.path.abspath(str(path))),
        os.path.normcase(str(path.resolve(strict=False))),
        os.path.normcase(str(path.parent.resolve(strict=False) / path.name)),
    }
    dev_ino: Optional[Tuple[int, int]] = None
    try:
        if path.exists() or path.is_symlink():
            st = os.stat(path)
            if st.st_ino != 0:
                dev_ino = (int(st.st_dev), int(st.st_ino))
    except OSError:
        dev_ino = None
    return norms, dev_ino


def ensure_safe_output_dir(
    out_dir: pathlib.Path,
    *,
    input_paths: Sequence[pathlib.Path],
    model_keys: Sequence[str],
) -> None:
    """Refuse output directories colliding with inputs or containing pre-existing run artifacts."""
    out_norms, out_dev_ino = _norm_identities(out_dir)
    for inp in input_paths:
        if inp is None:
            continue
        inp_norms, inp_dev_ino = _norm_identities(inp)
        if out_norms & inp_norms:
            raise ValueError(f"Refusing --output-dir matching input path: {out_dir} == {inp}")
        if out_dev_ino is not None and inp_dev_ino is not None and out_dev_ino == inp_dev_ino:
            raise ValueError(f"Refusing --output-dir sharing inode with input path: {out_dir} == {inp}")

    conflicts: List[pathlib.Path] = []
    if (out_dir / "matrix_session_summary.json").exists():
        conflicts.append(out_dir / "matrix_session_summary.json")
    for mk in model_keys:
        rdir = out_dir / f"run_{mk}"
        for fname in ("raw_outputs_all.jsonl", "session_summary.json", "gpu_gate.json", "stage_events.jsonl"):
            cand = rdir / fname
            if cand.exists():
                conflicts.append(cand)
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite pre-existing inference output artifacts: "
            + ", ".join(str(c) for c in conflicts[:5])
        )


def _append_jsonl(path: pathlib.Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
        handle.flush()


def _stream_worker_stdout(proc: subprocess.Popen[str], out_q: "queue.Queue[Optional[Dict[str, Any]]]") -> None:
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            s = line.strip()
            if not s:
                continue
            try:
                out_q.put(json.loads(s))
            except Exception as exc:
                out_q.put({"event": "stdout_parse_error", "raw_line": s, "error": repr(exc)})
    finally:
        out_q.put(None)


def _stream_worker_stderr(proc: subprocess.Popen[str], stderr_log: pathlib.Path) -> None:
    assert proc.stderr is not None
    with stderr_log.open("a", encoding="utf-8") as handle:
        for line in proc.stderr:
            handle.write(line)
            handle.flush()


def _kill_and_wait_worker(proc: subprocess.Popen[str]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    method = "already_exited"
    rc = proc.poll()
    if rc is None:
        try:
            proc.terminate()
            method = "terminate"
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                method = "kill"
            except Exception:
                pass
            if os.name == "nt":
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    method = "taskkill_T_F"
                except Exception:
                    pass
            try:
                proc.wait(timeout=5.0)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = None
    return {
        "pid": proc.pid,
        "kill_method": method,
        "returncode": rc,
        "kill_wall_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def _read_next_ipc_event(
    out_q: "queue.Queue[Optional[Dict[str, Any]]]",
    deadline: float,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    remaining = max(0.0, deadline - time.perf_counter())
    if remaining <= 0.0:
        return "timeout", None
    try:
        msg = out_q.get(timeout=remaining)
    except queue.Empty:
        return "timeout", None
    if msg is None:
        return "eof", None
    return "msg", msg


def _build_watchdog_fail_rec(
    *,
    model_key: str,
    model_cfg: Dict[str, Any],
    call: Dict[str, Any],
    error: str,
    last_stage: Optional[str],
    last_stage_data: Optional[Dict[str, Any]],
    call_wall_ms: float,
    is_timeout: bool,
    watchdog_killed: bool,
    kill_info: Optional[Dict[str, Any]],
    max_new_tokens: int,
    verified_input_sha256: Optional[str] = None,
    input_sha256_source: str = "not_verified_pre_input_failure",
    input_verification_status: str = "unverified_pre_input_failure",
) -> Dict[str, Any]:
    return {
        "event": "result",
        "source": "supervisor_watchdog",
        "call_index": int(call["call_index"]),
        "call_id": call["call_id"],
        "board_id": call["board_id"],
        "phase": "formal",
        "mode": call["mode"],
        "image_id": call["image_id"],
        "tile_id": call["tile_id"],
        "image_path": call["image_path"],
        "input_sha256": verified_input_sha256,
        "input_sha256_source": input_sha256_source,
        "input_verification_status": input_verification_status,
        "expected_sha256": call["expected_sha256"],
        "orig_input_width": int(call["width"]),
        "orig_input_height": int(call["height"]),
        "roi_offset_x": int(call["roi_offset_x"]),
        "roi_offset_y": int(call["roi_offset_y"]),
        "tile_offset_x": int(call["tile_offset_x"]),
        "tile_offset_y": int(call["tile_offset_y"]),
        "max_new_tokens": int(max_new_tokens),
        "prompt_sha256": call["prompt_sha256"],
        "coordinate_protocol": "xyxy_norm1000",
        "model_key": model_key,
        "model_repo_id": model_cfg.get("repo_id"),
        "model_revision": model_cfg.get("revision"),
        "extra_prefill_forward_executed": False,
        "ok": False,
        "is_oom": False,
        "is_timeout": bool(is_timeout),
        "watchdog_killed": bool(watchdog_killed),
        "last_worker_stage": last_stage,
        "last_worker_stage_data": last_stage_data,
        "kill_info": kill_info,
        "error": error,
        "raw_output": None,
        "call_wall_ms": round(call_wall_ms, 2),
        "call_finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _default_worker_cmd(
    *,
    model_key: str,
    model_cfg: Dict[str, Any],
    runtime_cfg: Optional[Dict[str, Any]],
    plan_path: pathlib.Path,
    runtime_config_path: Optional[pathlib.Path],
    path_remaps: Sequence[Tuple[str, str]],
    image_roots: Sequence[pathlib.Path],
) -> List[str]:
    py_exe = str(
        model_cfg.get("python_executable")
        or (runtime_cfg or {}).get("python_executable")
        or sys.executable
    )
    cmd = [
        py_exe,
        "-u",
        "-m",
        "pcb_vlm.inference.fics_worker",
        "--plan",
        str(plan_path),
        "--model-key",
        model_key,
    ]
    if runtime_config_path is not None:
        cmd.extend(["--runtime-config", str(runtime_config_path)])
    if model_cfg.get("overlay_dir"):
        cmd.extend(["--overlay-dir", str(model_cfg["overlay_dir"])])
    if model_cfg.get("hf_home"):
        cmd.extend(["--hf-home", str(model_cfg["hf_home"])])
    for old_p, new_p in path_remaps:
        cmd.extend(["--path-map", f"{old_p}={new_p}"])
    for r in image_roots:
        cmd.extend(["--image-root", str(r)])
    return cmd


def run_single_model_stage(
    *,
    model_key: str,
    plan: Dict[str, Any],
    runtime_cfg: Optional[Dict[str, Any]],
    plan_path: pathlib.Path,
    runtime_config_path: Optional[pathlib.Path],
    output_dir: pathlib.Path,
    path_remaps: Sequence[Tuple[str, str]],
    image_roots: Sequence[pathlib.Path],
    image_hash_index: Optional[Dict[str, pathlib.Path]],
    gpu_gate_fn: Callable[[float, float], Dict[str, Any]],
    worker_cmd_builder: Optional[Callable[[str, Dict[str, Any], pathlib.Path], List[str]]] = None,
    matrix_deadline: float,
    load_timeout_s: Optional[float] = None,
    per_call_timeout_s: Optional[float] = None,
    per_model_timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one model stage inside an isolated subprocess with hard GPU gate and watchdog."""
    run_dir = output_dir / f"run_{model_key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = run_dir / "raw_outputs_all.jsonl"
    stage_jsonl = run_dir / "stage_events.jsonl"
    gate_json = run_dir / "gpu_gate.json"
    summary_json = run_dir / "session_summary.json"
    stderr_log = run_dir / "worker_stderr.log"
    runtime_plan_json = run_dir / "runtime_plan_resolved.json"
    runtime_plan_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    limits = plan.get("resource_limits") or {}
    min_smi = float(limits.get("gpu_gate_min_smi_free_mib", 5500.0))
    min_torch = float(limits.get("gpu_gate_min_torch_free_mib", 6144.0))
    load_to = float(load_timeout_s if load_timeout_s is not None else limits.get("model_load_timeout_seconds", 180.0))
    call_to = float(per_call_timeout_s if per_call_timeout_s is not None else limits.get("per_call_timeout_seconds", 90.0))
    model_to = float(per_model_timeout_s if per_model_timeout_s is not None else limits.get("per_model_gpu_timeout_seconds", 3600.0))
    max_new_tokens = int(plan["inference_config"]["max_new_tokens"])

    calls = build_formal_calls_for_model(
        plan,
        model_key,
        path_remaps=path_remaps,
        image_hash_index=image_hash_index,
    )
    model_cfg = (plan.get("models") or {})[model_key]
    t_stage0 = time.perf_counter()

    gate_res = gpu_gate_fn(min_smi, min_torch)
    gate_json.write_text(json.dumps(gate_res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not gate_res.get("passed"):
        summary = {
            "model_key": model_key,
            "model_repo_id": model_cfg.get("repo_id"),
            "status": "gate_failed",
            "gate_result": gate_res,
            "planned_calls": len(calls),
            "executed_calls": 0,
            "ok_calls": 0,
            "failed_calls": 0,
            "wall_seconds": round(time.perf_counter() - t_stage0, 3),
        }
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return summary

    if worker_cmd_builder is not None:
        cmd = worker_cmd_builder(model_key, model_cfg, runtime_plan_json)
    else:
        cmd = _default_worker_cmd(
            model_key=model_key,
            model_cfg=model_cfg,
            runtime_cfg=runtime_cfg,
            plan_path=runtime_plan_json,
            runtime_config_path=runtime_config_path,
            path_remaps=path_remaps,
            image_roots=image_roots,
        )

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

    loaded_msg: Optional[Dict[str, Any]] = None
    executed_records: List[Dict[str, Any]] = []
    stage_status = "completed"
    abort_reason: Optional[str] = None
    model_deadline = min(matrix_deadline, time.perf_counter() + model_to)

    try:
        load_deadline = min(model_deadline, time.perf_counter() + load_to)
        while True:
            kind, msg = _read_next_ipc_event(out_q, load_deadline)
            if kind == "msg" and msg is not None:
                if msg.get("event") == "loaded":
                    loaded_msg = msg
                    _append_jsonl(stage_jsonl, {"event": "worker_loaded", "model_key": model_key, "data": msg})
                    break
                _append_jsonl(stage_jsonl, {"event": "pre_load_msg", "model_key": model_key, "data": msg})
                continue
            kill_info = _kill_and_wait_worker(proc)
            stage_status = "load_failed_or_timeout"
            abort_reason = f"Worker failed during load ({kind}): kill={kill_info}"
            _append_jsonl(stage_jsonl, {"event": "load_aborted", "kind": kind, "kill_info": kill_info})
            break

        if loaded_msg is not None:
            for call in calls:
                now = time.perf_counter()
                if now >= model_deadline:
                    kill_info = _kill_and_wait_worker(proc)
                    stage_status = "model_or_matrix_timeout"
                    abort_reason = "Exceeded model or total matrix timeout budget before call dispatch"
                    _append_jsonl(stage_jsonl, {"event": "stage_timeout", "call_id": call["call_id"], "kill_info": kill_info})
                    break

                _append_jsonl(stage_jsonl, {
                    "event": "supervisor_call_dispatch",
                    "model_key": model_key,
                    "call_index": call["call_index"],
                    "call_id": call["call_id"],
                    "board_id": call["board_id"],
                    "mode": call["mode"],
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })

                assert proc.stdin is not None
                try:
                    proc.stdin.write(json.dumps(call, ensure_ascii=False) + "\n")
                    proc.stdin.flush()
                except Exception as exc:
                    kill_info = _kill_and_wait_worker(proc)
                    fail_rec = _build_watchdog_fail_rec(
                        model_key=model_key,
                        model_cfg=model_cfg,
                        call=call,
                        error=f"Stdin dispatch failed: {exc!r}",
                        last_stage=None,
                        last_stage_data=None,
                        call_wall_ms=0.0,
                        is_timeout=False,
                        watchdog_killed=True,
                        kill_info=kill_info,
                        max_new_tokens=max_new_tokens,
                    )
                    _append_jsonl(raw_jsonl, fail_rec)
                    executed_records.append(fail_rec)
                    stage_status = "dispatch_failed"
                    abort_reason = fail_rec["error"]
                    break

                call_t0 = time.perf_counter()
                call_deadline = min(model_deadline, call_t0 + call_to)
                last_stage: Optional[str] = None
                last_stage_data: Optional[Dict[str, Any]] = None
                verified_input_sha256: Optional[str] = None
                input_sha256_source = "not_verified_pre_input_failure"
                input_verification_status = "unverified_pre_input_failure"
                got_result = False

                while True:
                    kind, msg = _read_next_ipc_event(out_q, call_deadline)
                    if kind == "msg" and msg is not None:
                        ev = msg.get("event")
                        if ev == "stage_event":
                            last_stage = msg.get("stage")
                            last_stage_data = msg.get("data")
                            if last_stage == "input_verified" and isinstance(last_stage_data, dict):
                                cand_sha = last_stage_data.get("input_sha256")
                                exp_sha = str(call.get("expected_sha256") or "").lower()
                                if (
                                    isinstance(cand_sha, str)
                                    and len(cand_sha) == 64
                                    and all(ch in "0123456789abcdefABCDEF" for ch in cand_sha)
                                ):
                                    cand_lower = cand_sha.lower()
                                    if not exp_sha or cand_lower == exp_sha:
                                        verified_input_sha256 = cand_lower
                                        input_sha256_source = "worker_stage_input_verified"
                                        input_verification_status = "verified_by_worker_before_failure"
                                    else:
                                        input_sha256_source = "worker_stage_input_mismatch"
                                        input_verification_status = "input_hash_mismatch_before_failure"
                            _append_jsonl(stage_jsonl, msg)
                            continue
                        if ev == "result":
                            _append_jsonl(raw_jsonl, msg)
                            executed_records.append(msg)
                            got_result = True
                            break
                        _append_jsonl(stage_jsonl, msg)
                        continue

                    kill_info = _kill_and_wait_worker(proc)
                    is_to = kind == "timeout"
                    stage_status = "call_timeout" if is_to else "worker_eof"
                    abort_reason = (
                        f"Call {call['call_id']} {'timed out' if is_to else 'worker EOF'} at stage={last_stage}"
                    )
                    fail_rec = _build_watchdog_fail_rec(
                        model_key=model_key,
                        model_cfg=model_cfg,
                        call=call,
                        error=abort_reason,
                        last_stage=last_stage,
                        last_stage_data=last_stage_data,
                        call_wall_ms=(time.perf_counter() - call_t0) * 1000.0,
                        is_timeout=is_to,
                        watchdog_killed=True,
                        kill_info=kill_info,
                        max_new_tokens=max_new_tokens,
                        verified_input_sha256=verified_input_sha256,
                        input_sha256_source=input_sha256_source,
                        input_verification_status=input_verification_status,
                    )
                    _append_jsonl(raw_jsonl, fail_rec)
                    executed_records.append(fail_rec)
                    _append_jsonl(stage_jsonl, {
                        "event": "watchdog_kill",
                        "call_id": call["call_id"],
                        "reason": abort_reason,
                        "kill_info": kill_info,
                    })
                    break

                if not got_result:
                    break
    finally:
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    proc.stdin.flush()
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                _kill_and_wait_worker(proc)
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

    ok_calls = sum(1 for r in executed_records if r.get("ok") is True)
    failed_calls = len(executed_records) - ok_calls
    if stage_status == "completed" and (len(executed_records) != len(calls) or failed_calls > 0):
        stage_status = "incomplete_or_failed_calls"

    summary = {
        "model_key": model_key,
        "model_repo_id": model_cfg.get("repo_id"),
        "status": stage_status,
        "abort_reason": abort_reason,
        "gate_result": gate_res,
        "worker_loaded": loaded_msg,
        "planned_calls": len(calls),
        "executed_calls": len(executed_records),
        "ok_calls": ok_calls,
        "failed_calls": failed_calls,
        "worker_returncode": proc.returncode,
        "raw_jsonl": str(raw_jsonl),
        "stage_events_jsonl": str(stage_jsonl),
        "wall_seconds": round(time.perf_counter() - t_stage0, 3),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def determine_matrix_exit_code(stage_summaries: Dict[str, Dict[str, Any]], planned_models: Sequence[str]) -> int:
    """Return 0 ONLY if every planned model completed every planned call cleanly with `ok=True`."""
    if not planned_models:
        return 1
    for mk in planned_models:
        st = stage_summaries.get(mk)
        if not st:
            return 1
        if st.get("status") != "completed":
            return 1
        planned = int(st.get("planned_calls") or 0)
        if planned <= 0:
            return 1
        if int(st.get("ok_calls") or 0) != planned or int(st.get("failed_calls") or 0) != 0:
            return 1
    return 0


def run_sequential_matrix(
    *,
    plan_path: pathlib.Path,
    output_dir: pathlib.Path,
    runtime_config_path: Optional[pathlib.Path] = None,
    models_filter: Optional[Sequence[str]] = None,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_roots: Optional[Sequence[pathlib.Path]] = None,
    dry_run: bool = True,
    verify_input_hashes: bool = False,
    gpu_gate_fn: Optional[Callable[[float, float], Dict[str, Any]]] = None,
    worker_cmd_builder: Optional[Callable[[str, Dict[str, Any], pathlib.Path], List[str]]] = None,
    load_timeout_s: Optional[float] = None,
    per_call_timeout_s: Optional[float] = None,
    per_model_timeout_s: Optional[float] = None,
    total_timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate and dry-run by default, or execute GPU inference when ``dry_run=False``."""
    raw_plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
    runtime_cfg = load_runtime_config(runtime_config_path)
    merged_plan = apply_runtime_overrides(raw_plan, runtime_cfg)

    all_remaps = list(path_remaps or [])
    for item in runtime_cfg.get("path_remaps") or []:
        if isinstance(item, str) and "=" in item:
            k, v = item.split("=", 1)
            all_remaps.append((k.strip(), v.strip()))

    all_roots = [pathlib.Path(p) for p in (image_roots or [])]
    for p in runtime_cfg.get("image_roots") or []:
        all_roots.append(pathlib.Path(str(p)))
    hash_index = build_image_hash_index(all_roots) if all_roots else None

    spec = build_expected_spec_from_plan(merged_plan, models_filter=models_filter)
    selected_models = list(spec["selected_models"])

    ensure_safe_output_dir(
        output_dir,
        input_paths=[plan_path] + ([runtime_config_path] if runtime_config_path is not None else []),
        model_keys=selected_models,
    )

    plan_inspect: Dict[str, Any] = {}
    for mk in selected_models:
        calls = build_formal_calls_for_model(
            merged_plan,
            mk,
            path_remaps=all_remaps,
            image_hash_index=hash_index,
        )
        missing_images: List[str] = []
        hash_mismatches: List[str] = []
        for c in calls:
            img_p = pathlib.Path(c["image_path"])
            if verify_input_hashes:
                if not img_p.is_file():
                    missing_images.append(c["call_id"])
                elif c["expected_sha256"] and sha256_file(img_p) != c["expected_sha256"]:
                    hash_mismatches.append(c["call_id"])
        if verify_input_hashes and (missing_images or hash_mismatches):
            raise RuntimeError(
                f"Input verification failed for model {mk!r}: "
                f"missing={missing_images[:5]}, sha_mismatches={hash_mismatches[:5]}"
            )
        plan_inspect[mk] = {
            "planned_calls": len(calls),
            "boards": sorted({c["board_id"] for c in calls}),
            "modes": sorted({c["mode"] for c in calls}),
            "local_dir": merged_plan["models"][mk].get("local_dir"),
        }

    if dry_run:
        return {
            "dry_run": True,
            "exit_code": 0,
            "plan_path": str(plan_path),
            "runtime_config_path": str(runtime_config_path) if runtime_config_path else None,
            "output_dir": str(output_dir),
            "selected_models": selected_models,
            "models": plan_inspect,
            "verify_input_hashes": verify_input_hashes,
        }

    gate_callable = gpu_gate_fn if gpu_gate_fn is not None else probe_gpu_gate
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = merged_plan.get("resource_limits") or {}
    total_to = float(total_timeout_s if total_timeout_s is not None else limits.get("total_gpu_timeout_seconds", 7200.0))
    matrix_deadline = time.perf_counter() + total_to
    t_matrix0 = time.perf_counter()

    stage_summaries: Dict[str, Dict[str, Any]] = {}
    for mk in selected_models:
        if time.perf_counter() >= matrix_deadline:
            stage_summaries[mk] = {
                "model_key": mk,
                "status": "skipped_matrix_timeout",
                "planned_calls": plan_inspect[mk]["planned_calls"],
                "executed_calls": 0,
                "ok_calls": 0,
                "failed_calls": 0,
            }
            continue
        stage_summaries[mk] = run_single_model_stage(
            model_key=mk,
            plan=merged_plan,
            runtime_cfg=runtime_cfg,
            plan_path=plan_path,
            runtime_config_path=runtime_config_path,
            output_dir=output_dir,
            path_remaps=all_remaps,
            image_roots=all_roots,
            image_hash_index=hash_index,
            gpu_gate_fn=gate_callable,
            worker_cmd_builder=worker_cmd_builder,
            matrix_deadline=matrix_deadline,
            load_timeout_s=load_timeout_s,
            per_call_timeout_s=per_call_timeout_s,
            per_model_timeout_s=per_model_timeout_s,
        )

    exit_code = determine_matrix_exit_code(stage_summaries, selected_models)
    matrix_summary = {
        "dry_run": False,
        "exit_code": exit_code,
        "plan_path": str(plan_path),
        "runtime_config_path": str(runtime_config_path) if runtime_config_path else None,
        "output_dir": str(output_dir),
        "selected_models": selected_models,
        "stage_summaries": stage_summaries,
        "wall_seconds": round(time.perf_counter() - t_matrix0, 3),
    }
    (output_dir / "matrix_session_summary.json").write_text(
        json.dumps(matrix_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return matrix_summary
