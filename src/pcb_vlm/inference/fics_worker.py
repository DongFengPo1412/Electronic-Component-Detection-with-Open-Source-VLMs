"""Isolated single-model GPU worker for FICS-PCB DSLR VLM evaluation.

Adapted cleanly into the repository from the verified multi-board GPU runner.
Zero external cache paths are hardcoded; runtime directories, model weights,
dependency overlays, and relocated image path remaps are supplied via CLI/config/env.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from pcb_vlm.evaluation.fics_replay import (
    FICS_SIX_CLASSES,
    classify_call_termination_and_parse_v2,
    load_fics_six_class_config,
    resolve_portable_image_path,
    sha256_text,
)
from pcb_vlm.inference.gpu_gate import sha256_file


def _parse_kv_pairs(items: Optional[Sequence[str]], flag_name: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected OLD=NEW for {flag_name}, got {item!r}")
        k, v = item.split("=", 1)
        if not k.strip() or not v.strip():
            raise ValueError(f"Invalid empty key/value in {flag_name}: {item!r}")
        pairs.append((k.strip(), v.strip()))
    return pairs


def _load_runtime_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = pathlib.Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Runtime config must be a YAML mapping: {p}")
    return data


def configure_runtime_paths(
    model_key: str,
    *,
    runtime_cfg: Optional[Dict[str, Any]] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
    overlay_dir: Optional[str] = None,
    hf_home: Optional[str] = None,
    temp_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Configure environment variables and `sys.path` BEFORE importing transformers."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rt_models = (runtime_cfg or {}).get("models") or {}
    rt_m = rt_models.get(model_key) or {}
    mc = model_cfg or {}

    resolved_tmp = temp_dir or (runtime_cfg or {}).get("temp_dir") or os.environ.get("PCB_VLM_TEMP_DIR")
    if resolved_tmp:
        tmp_p = pathlib.Path(resolved_tmp)
        tmp_p.mkdir(parents=True, exist_ok=True)
        os.environ["TEMP"] = str(tmp_p)
        os.environ["TMP"] = str(tmp_p)
        os.environ["TMPDIR"] = str(tmp_p)

    resolved_hf = (
        hf_home
        or rt_m.get("hf_home")
        or mc.get("hf_home")
        or os.environ.get(f"PCB_VLM_{model_key.upper()}_HF_HOME")
        or (runtime_cfg or {}).get("hf_home")
        or os.environ.get("PCB_VLM_HF_HOME")
    )
    if resolved_hf:
        os.environ["HF_HOME"] = str(resolved_hf)

    resolved_overlay = (
        overlay_dir
        or rt_m.get("overlay_dir")
        or mc.get("overlay_dir")
        or os.environ.get(f"PCB_VLM_{model_key.upper()}_OVERLAY_DIR")
    )

    if model_key == "qwen3" and resolved_overlay:
        ov_str = str(resolved_overlay)
        while ov_str in sys.path:
            sys.path.remove(ov_str)
        sys.path.insert(0, ov_str)
    elif resolved_overlay:
        ov_str = str(resolved_overlay)
        while ov_str in sys.path:
            sys.path.remove(ov_str)

    return {
        "hf_home": resolved_hf,
        "overlay_dir": resolved_overlay,
        "require_overlay": bool(rt_m.get("require_overlay", mc.get("overlay_used", False))),
        "temp_dir": resolved_tmp,
    }


def check_raw_logits_tensor_finite(raw_logits: Any) -> Dict[str, Any]:
    """Verify finiteness on raw pre-masking step-1 logits without retaining tensor references."""
    import torch

    if raw_logits is None:
        raise ValueError("raw_logits cannot be None")
    if raw_logits.ndim == 3:
        last_logits = raw_logits[:, -1, :]
    elif raw_logits.ndim == 2:
        last_logits = raw_logits
    else:
        raise ValueError(f"Unexpected logits ndim={raw_logits.ndim}")

    has_nan = bool(torch.isnan(last_logits).any().item())
    has_inf = bool(torch.isinf(last_logits).any().item())
    return {
        "is_finite": bool((not has_nan) and (not has_inf)),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "shape": [int(x) for x in last_logits.shape],
    }


def make_first_step_streamer(emit_fn: Callable[[str, Dict[str, Any]], None], t_gen0: float, hook_state: Dict[str, Any]):
    """Emit `first_step_seen` telemetry immediately once step 1 completes and CUDA synchronizes."""
    import torch
    from transformers.generation.streamers import BaseStreamer

    class FirstStepStreamer(BaseStreamer):
        def __init__(self) -> None:
            super().__init__()
            self.seen_prompt = False
            self.first_step_emitted = False
            self.token_steps_seen = 0

        def put(self, value: Any) -> None:
            if not self.seen_prompt:
                self.seen_prompt = True
                return
            self.token_steps_seen += 1
            if not self.first_step_emitted:
                self.first_step_emitted = True
                if torch.cuda.is_available():
                    torch.cuda.synchronize(0)
                ttft_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
                try:
                    first_tok = int(value.reshape(-1)[0].item())
                except Exception:
                    first_tok = None
                payload: Dict[str, Any] = {
                    "first_token_id": first_tok,
                    "time_to_first_token_ms": ttft_ms,
                    "first_step_raw_logits_finite": hook_state.get("is_finite"),
                    "first_step_raw_logits_source": hook_state.get("source"),
                    "first_step_logits_has_nan": hook_state.get("has_nan"),
                    "first_step_logits_has_inf": hook_state.get("has_inf"),
                }
                if torch.cuda.is_available():
                    payload["cuda_max_allocated_after_step1_mib"] = round(
                        torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2
                    )
                    payload["cuda_max_reserved_after_step1_mib"] = round(
                        torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2
                    )
                emit_fn("first_step_seen", payload)

        def end(self) -> None:
            return None

    return FirstStepStreamer()


def _attach_step1_lm_head_hook(lm_head_module: Any, hook_state: Dict[str, Any]):
    handle_box: List[Any] = [None]

    def _hook(_module: Any, _inp: Any, output: Any) -> None:
        if hook_state["hook_calls"] == 0:
            hook_state["hook_calls"] += 1
            res = check_raw_logits_tensor_finite(output)
            hook_state["is_finite"] = res["is_finite"]
            hook_state["has_nan"] = res["has_nan"]
            hook_state["has_inf"] = res["has_inf"]
            hook_state["shape"] = res["shape"]
            if handle_box[0] is not None:
                try:
                    handle_box[0].remove()
                except Exception:
                    pass
            if not res["is_finite"]:
                raise RuntimeError(
                    f"Non-finite raw logits detected at step 1: has_nan={res['has_nan']} has_inf={res['has_inf']}"
                )

    handle_box[0] = lm_head_module.register_forward_hook(_hook)
    return handle_box[0]


def _resolve_call_image_path(
    req: Dict[str, Any],
    *,
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_hash_index: Optional[Dict[str, pathlib.Path]] = None,
) -> pathlib.Path:
    row_like = {
        "image_path": req.get("image_path"),
        "expected_sha256": req.get("expected_sha256"),
    }
    resolved = resolve_portable_image_path(
        row_like,
        expected_call_spec={"path": req.get("image_path")},
        path_remaps=path_remaps,
        image_hash_index=image_hash_index,
    )
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"Could not resolve input image for call {req.get('call_id')!r}: {req.get('image_path')!r}")
    return resolved


def run_single_call(
    *,
    model_key: str,
    req: Dict[str, Any],
    model: Any,
    processor: Any,
    base_gen_cfg: Any,
    eos_ids: List[int],
    bos_id: Optional[int],
    pad_id: int,
    rep_pen: float,
    max_new_tokens: int,
    qwen_min_pixels: int,
    qwen_max_pixels: int,
    model_repo_id: str,
    model_revision: Optional[str],
    expected_prompt_sha256: str,
    emit_stage: Callable[[str, Dict[str, Any]], None],
    path_remaps: Optional[Sequence[Tuple[str, str]]] = None,
    image_hash_index: Optional[Dict[str, pathlib.Path]] = None,
) -> Dict[str, Any]:
    """Execute one deterministic single-image inference call and emit full provenance record."""
    import torch
    from PIL import Image

    t_call0 = time.perf_counter()
    call_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    img_path = _resolve_call_image_path(req, path_remaps=path_remaps, image_hash_index=image_hash_index)
    actual_sha = sha256_file(img_path)
    if actual_sha != req["expected_sha256"]:
        raise RuntimeError(f"Input SHA mismatch for {img_path}: {actual_sha} != {req['expected_sha256']}")
    if req["prompt_sha256"] != expected_prompt_sha256 or sha256_text(req["prompt"]) != expected_prompt_sha256:
        raise RuntimeError(f"Prompt SHA mismatch: {req['prompt_sha256']} != {expected_prompt_sha256}")

    with Image.open(img_path) as im_raw:
        pil_img = im_raw.convert("RGB")
    orig_w, orig_h = pil_img.size
    if (orig_w, orig_h) != (int(req["width"]), int(req["height"])):
        raise RuntimeError(f"Size mismatch {(orig_w, orig_h)} != {(req['width'], req['height'])}")

    emit_stage("input_verified", {
        "input_sha256": actual_sha,
        "resolved_image_path": str(img_path),
        "orig_width": orig_w,
        "orig_height": orig_h,
        "max_new_tokens": max_new_tokens,
    })

    rec: Dict[str, Any] = {
        "event": "result",
        "source": "worker",
        "call_index": int(req["call_index"]),
        "call_id": req["call_id"],
        "board_id": req["board_id"],
        "phase": "formal",
        "mode": req["mode"],
        "image_id": req["image_id"],
        "tile_id": req["tile_id"],
        "image_path": str(img_path),
        "input_sha256": actual_sha,
        "expected_sha256": req["expected_sha256"],
        "orig_input_width": orig_w,
        "orig_input_height": orig_h,
        "roi_offset_x": int(req["roi_offset_x"]),
        "roi_offset_y": int(req["roi_offset_y"]),
        "tile_offset_x": int(req["tile_offset_x"]),
        "tile_offset_y": int(req["tile_offset_y"]),
        "max_new_tokens": max_new_tokens,
        "prompt_sha256": req["prompt_sha256"],
        "coordinate_protocol": "xyxy_norm1000",
        "model_key": model_key,
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "call_started_at": call_started_at,
        "extra_prefill_forward_executed": False,
        "ok": False,
        "is_oom": False,
        "is_timeout": False,
        "watchdog_killed": False,
        "error": None,
        "raw_output": None,
    }

    inputs_cuda = None
    prepare_inputs = None
    inputs_embeds = None
    out_ids = None
    hook_handle = None
    hook_state: Dict[str, Any] = {
        "hook_calls": 0,
        "source": "lm_head_forward_hook_pre_masking",
        "is_finite": None,
        "has_nan": None,
        "has_inf": None,
        "shape": None,
    }

    try:
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)

        if model_key in ("qwen2", "qwen3"):
            t_prep0 = time.perf_counter()
            if model_key == "qwen2":
                from qwen_vl_utils import process_vision_info

                processor.image_processor.min_pixels = qwen_min_pixels
                processor.image_processor.max_pixels = qwen_max_pixels
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img, "min_pixels": qwen_min_pixels, "max_pixels": qwen_max_pixels},
                        {"type": "text", "text": req["prompt"]},
                    ],
                }]
                text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs_cpu = processor(
                    text=[text_in],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                patch_size = int(getattr(processor.image_processor, "patch_size", 14))
                merge_size = int(getattr(processor.image_processor, "merge_size", 2))
            else:
                if processor.image_processor.size != {
                    "shortest_edge": qwen_min_pixels,
                    "longest_edge": qwen_max_pixels,
                }:
                    raise RuntimeError(f"Unexpected Qwen3 image processor size: {processor.image_processor.size}")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": req["prompt"]},
                    ],
                }]
                text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs_cpu = processor(text=[text_in], images=[pil_img], return_tensors="pt")
                patch_size = int(getattr(processor.image_processor, "patch_size", 16))
                merge_size = int(getattr(processor.image_processor, "merge_size", 2))

            cpu_preprocess_ms = round((time.perf_counter() - t_prep0) * 1000.0, 2)
            grid = [int(x) for x in inputs_cpu["image_grid_thw"][0].tolist()]
            proc_w = int(grid[2]) * patch_size
            proc_h = int(grid[1]) * patch_size
            proc_area = proc_w * proc_h
            if proc_area > qwen_max_pixels:
                raise RuntimeError(f"{model_key} processed area {proc_area} exceeds max_pixels={qwen_max_pixels}")

            t_h2d0 = time.perf_counter()
            inputs_cuda = inputs_cpu.to("cuda:0")
            torch.cuda.synchronize(0)
            h2d_transfer_ms = round((time.perf_counter() - t_h2d0) * 1000.0, 2)
            del inputs_cpu

            input_toks = int(inputs_cuda["input_ids"].shape[1])
            merged_vis_toks = int((grid[1] // merge_size) * (grid[2] // merge_size))
            px_shape = [int(x) for x in inputs_cuda["pixel_values"].shape]

            proc_payload = {
                "cpu_preprocess_ms": cpu_preprocess_ms,
                "h2d_transfer_ms": h2d_transfer_ms,
                "processed_width_px": proc_w,
                "processed_height_px": proc_h,
                "processed_area_px": proc_area,
                "image_grid_thw": grid,
                "pixel_values_shape": px_shape,
                "input_tokens": input_toks,
                "merged_visual_tokens": merged_vis_toks,
            }
            rec.update(proc_payload)
            emit_stage("processor_done", proc_payload)

            from transformers import GenerationConfig

            call_gen_cfg = GenerationConfig.from_dict(base_gen_cfg.to_dict())
            call_gen_cfg.max_new_tokens = max_new_tokens
            call_gen_cfg.do_sample = False
            call_gen_cfg.repetition_penalty = rep_pen
            call_gen_cfg.eos_token_id = list(eos_ids)
            call_gen_cfg.bos_token_id = bos_id
            call_gen_cfg.pad_token_id = pad_id
            for attr in ("temperature", "top_p", "top_k"):
                if hasattr(call_gen_cfg, attr):
                    setattr(call_gen_cfg, attr, None)

            hook_handle = _attach_step1_lm_head_hook(model.lm_head, hook_state)
            t_gen0 = time.perf_counter()
            streamer = make_first_step_streamer(emit_stage, t_gen0, hook_state)
            emit_stage("generation_started", {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "repetition_penalty": rep_pen,
                "extra_prefill_forward_executed": False,
            })

            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs_cuda,
                    generation_config=call_gen_cfg,
                    use_cache=True,
                    streamer=streamer,
                )
            torch.cuda.synchronize(0)
            gpu_generate_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
            gen_ids = out_ids[0][input_toks:].detach().cpu().tolist()
            if model_key == "qwen3":
                raw_out = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
            else:
                raw_out = processor.decode(gen_ids, skip_special_tokens=False)
        else:
            t_prep0 = time.perf_counter()
            conv = [
                {"role": "User", "content": f"<image_placeholder>{req['prompt']}", "images": [pil_img]},
                {"role": "Assistant", "content": ""},
            ]
            prepare_inputs_cpu = processor(conversations=conv, images=[pil_img], force_batchify=True)
            cpu_preprocess_ms = round((time.perf_counter() - t_prep0) * 1000.0, 2)

            t_h2d0 = time.perf_counter()
            prepare_inputs = prepare_inputs_cpu.to("cuda:0")
            torch.cuda.synchronize(0)
            h2d_transfer_ms = round((time.perf_counter() - t_h2d0) * 1000.0, 2)
            del prepare_inputs_cpu

            px_shape = [int(x) for x in prepare_inputs.pixel_values.shape]
            max_dim = max(orig_w, orig_h)
            cw = max(int(orig_w / max_dim * 384), 14)
            ch = max(int(orig_h / max_dim * 384), 14)
            input_toks = int(prepare_inputs.input_ids.shape[1])
            merged_vis_toks = int(prepare_inputs.images_seq_mask.sum().item())
            proc_payload = {
                "cpu_preprocess_ms": cpu_preprocess_ms,
                "h2d_transfer_ms": h2d_transfer_ms,
                "processed_width_px": int(px_shape[-1]),
                "processed_height_px": int(px_shape[-2]),
                "processed_content_wh_before_pad": [cw, ch],
                "processed_area_px": int(px_shape[-1]) * int(px_shape[-2]),
                "image_grid_thw": None,
                "pixel_values_shape": px_shape,
                "input_tokens": input_toks,
                "merged_visual_tokens": merged_vis_toks,
            }
            rec.update(proc_payload)
            emit_stage("processor_done", proc_payload)

            gen_model = getattr(model, "language_model", model)
            hook_handle = _attach_step1_lm_head_hook(gen_model.lm_head, hook_state)
            t_gen0 = time.perf_counter()
            streamer = make_first_step_streamer(emit_stage, t_gen0, hook_state)
            emit_stage("generation_started", {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "repetition_penalty": rep_pen,
                "extra_prefill_forward_executed": False,
            })

            with torch.inference_mode():
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                out_ids = gen_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=pad_id,
                    bos_token_id=bos_id,
                    eos_token_id=eos_ids[0],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=rep_pen,
                    use_cache=True,
                    streamer=streamer,
                )
            torch.cuda.synchronize(0)
            gpu_generate_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
            gen_ids = out_ids[0].detach().cpu().tolist()
            raw_out = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)

        if hook_state["is_finite"] is None:
            raise RuntimeError("lm_head step-1 raw logits finiteness hook did not execute")

        v2 = classify_call_termination_and_parse_v2(
            raw_text=raw_out,
            width=orig_w,
            height=orig_h,
            generated_tokens=len(gen_ids),
            max_new_tokens=max_new_tokens,
            generated_token_ids=gen_ids,
            eos_token_ids=eos_ids,
            protocol="xyxy_norm1000",
            allowed_classes=FICS_SIX_CLASSES,
        )
        finish_reason = (
            "eos_token"
            if v2["normal_eos"]
            else ("length" if len(gen_ids) >= max_new_tokens else "unknown")
        )

        emit_stage("generation_finished", {
            "raw_output_chars": len(raw_out),
            "generated_tokens": len(gen_ids),
            "max_new_tokens": max_new_tokens,
            "has_normal_eos": bool(v2["normal_eos"]),
            "finish_reason": finish_reason,
            "is_truncated": bool(v2["is_truncated"]),
            "gpu_generate_ms": gpu_generate_ms,
        })

        rec.update({
            "ok": True,
            "raw_output": raw_out,
            "raw_output_chars": len(raw_out),
            "generated_tokens": len(gen_ids),
            "generated_token_ids": gen_ids,
            "finish_reason": finish_reason,
            "has_normal_eos": bool(v2["normal_eos"]),
            "is_truncated": bool(v2["is_truncated"]),
            "length_limited_without_eos": bool(v2["length_limited_without_eos"]),
            "complete_json_syntax_ok": bool(v2["complete_json_syntax"]),
            "strict_schema_ok": bool(v2["strict_schema_ok"]),
            "partial_objects_recovered": bool(v2["partial_recovery"]),
            "objects_local": v2["objects"],
            "non_fics_objects_filtered": v2["non_fics_objects_filtered"],
            "parse_status": v2["status"],
            "call_bucket": v2["call_bucket"],
            "parse_audit": v2["audit"],
            "smoke_logits_finite_verified": hook_state["is_finite"],
            "first_step_raw_logits_source": hook_state["source"],
            "first_step_raw_logits_shape": hook_state["shape"],
            "gpu_generate_ms": gpu_generate_ms,
            "latency_ms": gpu_generate_ms,
            "cuda_max_allocated_mib": round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2),
            "cuda_max_reserved_mib": round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2),
        })
    except Exception as exc:
        is_oom = "out of memory" in str(exc).lower()
        rec["ok"] = False
        rec["is_oom"] = is_oom
        rec["error"] = repr(exc)
        rec["raw_output"] = None
        rec["smoke_logits_finite_verified"] = hook_state.get("is_finite")
    finally:
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                pass
        del inputs_cuda, prepare_inputs, inputs_embeds, out_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rec["call_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rec["call_wall_ms"] = round((time.perf_counter() - t_call0) * 1000.0, 2)
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated single-model GPU worker for FICS-PCB VLM evaluation.")
    parser.add_argument("--plan", required=True, help="Path to frozen plan JSON.")
    parser.add_argument("--model-key", required=True, choices=["qwen2", "deepseek", "qwen3"])
    parser.add_argument("--runtime-config", help="Optional local runtime YAML overriding model_dir/overlay/hf_home.")
    parser.add_argument("--overlay-dir", help="Optional dependency overlay directory (primarily for qwen3).")
    parser.add_argument("--hf-home", help="Optional HF_HOME directory override.")
    parser.add_argument("--path-map", action="append", help="Repeatable OLD_PREFIX=NEW_PREFIX image remap.")
    parser.add_argument("--image-root", action="append", help="Repeatable directory of images indexed by SHA-256.")
    args = parser.parse_args(argv)

    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    runtime_cfg = _load_runtime_yaml(args.runtime_config)
    model_cfg = dict((plan.get("models") or {}).get(args.model_key) or {})
    rt_model_cfg = dict(((runtime_cfg.get("models") or {}).get(args.model_key)) or {})

    rt_applied = configure_runtime_paths(
        args.model_key,
        runtime_cfg=runtime_cfg,
        model_cfg=model_cfg,
        overlay_dir=args.overlay_dir,
        hf_home=args.hf_home,
    )

    fics_cfg = load_fics_six_class_config()
    expected_prompt_sha256 = str(fics_cfg["protocol"]["prompt_sha256"]).lower()
    model_dir = str(rt_model_cfg.get("local_dir") or model_cfg.get("local_dir") or "")
    if not model_dir:
        raise ValueError(f"Missing local_dir for model {args.model_key!r} in plan and runtime config")

    path_remaps = _parse_kv_pairs(args.path_map, "--path-map")
    for item in runtime_cfg.get("path_remaps") or []:
        if isinstance(item, str) and "=" in item:
            k, v = item.split("=", 1)
            path_remaps.append((k.strip(), v.strip()))

    image_roots = [pathlib.Path(p) for p in (args.image_root or [])]
    for p in runtime_cfg.get("image_roots") or []:
        image_roots.append(pathlib.Path(str(p)))
    image_hash_index: Optional[Dict[str, pathlib.Path]] = None
    if image_roots:
        from pcb_vlm.evaluation.fics_replay import build_image_hash_index

        image_hash_index = build_image_hash_index(image_roots)

    ipc_out = sys.__stdout__
    sys.stdout = sys.stderr

    inf_cfg = plan["inference_config"]
    max_new_tokens = int(inf_cfg["max_new_tokens"])
    rep_pen = float(inf_cfg["repetition_penalty"])
    qwen_min_px = int(inf_cfg["qwen_min_pixels"])
    qwen_max_px = int(inf_cfg["qwen_max_pixels"])

    import torch
    import transformers
    from transformers import GenerationConfig

    require_overlay = bool(rt_applied.get("require_overlay", False) and rt_applied.get("overlay_dir"))
    if args.model_key == "qwen3" and require_overlay:
        ov_norm = os.path.normcase(os.path.abspath(str(rt_applied["overlay_dir"])))
        tf_norm = os.path.normcase(os.path.abspath(transformers.__file__))
        if ov_norm not in tf_norm:
            raise RuntimeError(f"Expected Qwen3 overlay transformers under {ov_norm}, got {transformers.__file__}")

    t_load0 = time.perf_counter()
    if args.model_key == "qwen2":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
            min_pixels=qwen_min_px,
            max_pixels=qwen_max_px,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=str(rt_model_cfg.get("attn_implementation") or model_cfg.get("attn_implementation") or "sdpa"),
            device_map="cuda:0",
        )
        model.eval()
        eos_ids = [int(x) for x in fics_cfg["eos_token_ids_by_model"]["qwen2"]]
        bos_id = getattr(model.generation_config, "bos_token_id", None)
        pad_id = 151643
        base_gen_cfg = GenerationConfig.from_model_config(model.config)
        img_proc = processor.image_processor
        proc_meta = {
            "processor_class": type(processor).__name__,
            "image_processor_class": type(img_proc).__name__,
            "min_pixels": int(getattr(img_proc, "min_pixels", qwen_min_px)),
            "max_pixels": int(getattr(img_proc, "max_pixels", qwen_max_px)),
            "patch_size": int(getattr(img_proc, "patch_size", 14)),
            "merge_size": int(getattr(img_proc, "merge_size", 2)),
        }
    elif args.model_key == "qwen3":
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            model_dir,
            size={"shortest_edge": qwen_min_px, "longest_edge": qwen_max_px},
            local_files_only=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            attn_implementation=str(rt_model_cfg.get("attn_implementation") or model_cfg.get("attn_implementation") or "sdpa"),
            local_files_only=True,
        ).to("cuda:0")
        model.eval()
        eos_ids = [int(x) for x in fics_cfg["eos_token_ids_by_model"]["qwen3"]]
        bos_id = None
        pad_id = 151643
        base_gen_cfg = GenerationConfig.from_model_config(model.config)
        img_proc = processor.image_processor
        proc_meta = {
            "processor_class": type(processor).__name__,
            "image_processor_class": type(img_proc).__name__,
            "size": dict(img_proc.size),
            "patch_size": int(getattr(img_proc, "patch_size", 16)),
            "merge_size": int(getattr(img_proc, "merge_size", 2)),
        }
    else:
        from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor

        processor = VLChatProcessor.from_pretrained(model_dir, local_files_only=True)
        model = MultiModalityCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        model.eval()
        eos_ids = [int(processor.tokenizer.eos_token_id)]
        bos_id = int(processor.tokenizer.bos_token_id)
        pad_id = eos_ids[0]
        base_gen_cfg = None
        img_proc = processor.image_processor
        proc_meta = {
            "processor_class": type(processor).__name__,
            "image_processor_class": type(img_proc).__name__,
            "image_size": int(getattr(img_proc, "image_size", 384)),
            "min_size": int(getattr(img_proc, "min_size", 14)),
            "background_color": list(getattr(img_proc, "background_color", (127, 127, 127))),
            "num_image_tokens": 576,
        }

    torch.cuda.synchronize(0)
    load_seconds = round(time.perf_counter() - t_load0, 3)
    alloc_mib = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
    res_mib = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)

    ipc_out.write(json.dumps({
        "event": "loaded",
        "model_key": args.model_key,
        "load_seconds": load_seconds,
        "actual_model_path": model_dir,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_allocated_mib": alloc_mib,
        "cuda_reserved_mib": res_mib,
        "cleaned_runtime_generation_config": {
            "do_sample": False,
            "repetition_penalty": rep_pen,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": list(eos_ids),
            "bos_token_id": bos_id,
            "pad_token_id": pad_id,
        },
        "processor_meta": proc_meta,
    }, ensure_ascii=False) + "\n")
    ipc_out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break

        def emit_stage(stage_name: str, data: Dict[str, Any]) -> None:
            evt = {
                "event": "stage_event",
                "source": "worker",
                "model_key": args.model_key,
                "call_index": int(req["call_index"]),
                "call_id": req["call_id"],
                "stage": stage_name,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "mono_ms": round(time.perf_counter() * 1000.0, 2),
                "data": data,
            }
            ipc_out.write(json.dumps(evt, ensure_ascii=False) + "\n")
            ipc_out.flush()

        rec = run_single_call(
            model_key=args.model_key,
            req=req,
            model=model,
            processor=processor,
            base_gen_cfg=base_gen_cfg,
            eos_ids=eos_ids,
            bos_id=bos_id,
            pad_id=pad_id,
            rep_pen=rep_pen,
            max_new_tokens=max_new_tokens,
            qwen_min_pixels=qwen_min_px,
            qwen_max_pixels=qwen_max_px,
            model_repo_id=str(rt_model_cfg.get("repo_id") or model_cfg["repo_id"]),
            model_revision=rt_model_cfg.get("revision", model_cfg.get("revision")),
            expected_prompt_sha256=expected_prompt_sha256,
            emit_stage=emit_stage,
            path_remaps=path_remaps,
            image_hash_index=image_hash_index,
        )
        ipc_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ipc_out.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())

