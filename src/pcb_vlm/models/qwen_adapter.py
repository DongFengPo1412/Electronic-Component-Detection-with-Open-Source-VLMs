"""Qwen2-VL Model Adapter for Electronic Component Detection.

Implements:
1. Native Qwen2-VL grounding prompt and multi-segment parsing (canonical x-first: xmin, ymin, xmax, ymax).
2. Segment-based multi-pattern extraction (Token format, JSON format, bracketed quadruplets, coordinate pairs).
3. Explicit label preservation across all patterns (retains 'resistor' in 'resistor [100,100,300,300]').
4. Robust truncation detection and tri-state status reporting ('success', 'partial_success', 'no_objects_detected', 'parse_failed').
5. Unified 11 canonical electronic component classes.
"""

import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from pcb_vlm.geometry.coordinate_protocol import parse_response, generation_finished
from pcb_vlm.config import get_canonical_categories
DEFAULT_PCB_CLASSES = get_canonical_categories()


class QwenAdapter:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        default_qwen = os.environ.get(
            "PCB_VLM_QWEN_PATH",
            os.path.join(
                os.environ.get("PCB_VLM_CACHE_ROOT", r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024"),
                "models", "models", "Qwen--Qwen2-VL-2B-Instruct", "snapshots", "master",
            ),
        )
        known_aliases = {
            "Qwen/Qwen2-VL-2B-Instruct": default_qwen,
            "qwen2-vl-2b": default_qwen,
            "qwen": default_qwen,
        }

        requested = model_path or model_id or "Qwen/Qwen2-VL-2B-Instruct"
        self.requested_model = requested
        if model_path and model_path not in known_aliases and not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)

        if model_path and os.path.exists(model_path):
            target_path = model_path
        elif model_id and os.path.exists(model_id):
            target_path = model_id
        elif model_id and model_id in known_aliases:
            target_path = known_aliases[model_id]
        elif model_path and model_path in known_aliases:
            target_path = known_aliases[model_path]
        elif (not model_path and not model_id) or requested in known_aliases:
            target_path = known_aliases["Qwen/Qwen2-VL-2B-Instruct"]
        elif cache_dir and os.path.exists(os.path.join(cache_dir, "models", "Qwen--Qwen2-VL-2B-Instruct", "snapshots", "master")):
            target_path = os.path.join(cache_dir, "models", "Qwen--Qwen2-VL-2B-Instruct", "snapshots", "master")
        elif os.path.exists(r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master") and (model_id == "Qwen/Qwen2-VL-2B-Instruct" or not model_id):
            target_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\Qwen--Qwen2-VL-2B-Instruct\snapshots\master"
        else:
            raise FileNotFoundError(
                f"Requested model '{requested}' does not exist on disk and is not a recognized local alias! "
                f"Silent fallback to default models is strictly prohibited."
            )

        if adapter_path is not None and not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Requested adapter_path '{adapter_path}' does not exist on disk!")

        self.model_path = target_path
        self.model_id = target_path
        self.adapter_path = adapter_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir

        print(f"Loading Qwen2-VL base from {target_path} (requested: {requested}) on {device}...")
        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            target_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None
        )
        if device != "cuda":
            self.model.to(device)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(target_path, local_files_only=True)
        load_time = time.time() - t0
        print(f"Qwen2-VL successfully loaded in {load_time:.2f}s.")

        if adapter_path and os.path.exists(adapter_path):
            print(f"Attaching LoRA adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()
            print("LoRA adapter attached.")

    def load_model(self):
        """No-op if already loaded in __init__."""
        pass

    def build_prompt(self, target_classes: Optional[List[str]] = None) -> str:
        """Construct prompt with unified 11 classes and canonical x-first coordinate instruction."""
        if not target_classes:
            target_classes = DEFAULT_PCB_CLASSES
        classes_str = ", ".join(target_classes)
        return (
            f"Locate all {classes_str} on this PCB image. "
            f"Output each bounding box in normalized [0, 1000] coordinates: label <|box_start|>(xmin,ymin),(xmax,ymax)<|box_end|>. "
            f"If no target components are visible, output 'None'."
        )

    def parse_legacy_grounding_output_with_status(self, text: str, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], str, str, bool]:
        """Parse Qwen2-VL grounding tokens to pixel xyxy, returning explicit status and truncation flags.
        Canonical coordinate order: (xmin, ymin), (xmax, ymax).
        """
        raw_candidates = []

        def _to_pixel_box(c1: float, c2: float, c3: float, c4: float) -> Tuple[float, float, float, float]:
            if max(c1, c2, c3, c4) <= 1.05:
                x1 = c1 * img_w
                y1 = c2 * img_h
                x2 = c3 * img_w
                y2 = c4 * img_h
            else:
                x1 = (c1 / 1000.0) * img_w
                y1 = (c2 / 1000.0) * img_h
                x2 = (c3 / 1000.0) * img_w
                y2 = (c4 / 1000.0) * img_h

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))
            return round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)

        # 1. Official token format: <|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
        p1 = re.compile(r"<\|object_ref_start\|>\s*([^<]+?)\s*<\|object_ref_end\|>\s*<\|box_start\|>\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*<\|box_end\|>")
        for m in p1.finditer(text):
            lbl, c1, c2, c3, c4 = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            raw_candidates.append((m.start(), m.end(), lbl.strip().lower(), [c1, c2, c3, c4]))

        # 2. Box token with preceding label: label <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        p2 = re.compile(r"([a-zA-Z_\-]+)\s*<\|box_start\|>\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*<\|box_end\|>")
        for m in p2.finditer(text):
            lbl, c1, c2, c3, c4 = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            raw_candidates.append((m.start(), m.end(), lbl.strip().lower(), [c1, c2, c3, c4]))

        # 3. Python dict / JSON format: {'label': '...', 'bbox': [x1, y1, x2, y2], ...}
        p_dict1 = re.compile(
            r"\{[^{}]*?['\"](?:label|category)['\"]\s*:\s*['\"]([^'\"]+)['\"][^{}]*?['\"](?:bbox|bbox_2d|box)['\"]\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\][^{}]*?\}"
        )
        for m in p_dict1.finditer(text):
            lbl, c1, c2, c3, c4 = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            raw_candidates.append((m.start(), m.end(), lbl.strip().lower(), [c1, c2, c3, c4]))

        # 4. Python dict / JSON format with bbox before label
        p_dict2 = re.compile(
            r"\{[^{}]*?['\"](?:bbox|bbox_2d|box)['\"]\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\][^{}]*?['\"](?:label|category)['\"]\s*:\s*['\"]([^'\"]+)['\"][^{}]*?\}"
        )
        for m in p_dict2.finditer(text):
            c1, c2, c3, c4, lbl = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5)
            raw_candidates.append((m.start(), m.end(), lbl.strip().lower(), [c1, c2, c3, c4]))

        # 5. Bare box token without label: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
        p3 = re.compile(r"<\|box_start\|>\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*<\|box_end\|>")
        for m in p3.finditer(text):
            c1, c2, c3, c4 = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            raw_candidates.append((m.start(), m.end(), "unknown", [c1, c2, c3, c4]))

        # 6. Bracketed quadruplets with optional preceding label: [label] [c1, c2, c3, c4]
        p5 = re.compile(r'(?:\b([a-zA-Z_\-]+)\s*)?\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]')
        for m in p5.finditer(text):
            lbl = m.group(1).strip().lower() if m.group(1) else "unknown"
            c1, c2, c3, c4 = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            raw_candidates.append((m.start(), m.end(), lbl, [c1, c2, c3, c4]))

        # 7. Coordinate pairs with optional preceding label: [label] (c1, c2), (c3, c4)
        p6 = re.compile(r'(?:\b([a-zA-Z_\-]+)\s*)?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*,?\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)')
        for m in p6.finditer(text):
            lbl = m.group(1).strip().lower() if m.group(1) else "unknown"
            c1, c2, c3, c4 = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            raw_candidates.append((m.start(), m.end(), lbl, [c1, c2, c3, c4]))

        # Multi-segment non-overlapping span filtering
        raw_candidates.sort(key=lambda x: (x[0], -x[1]))
        accepted_spans = []
        detected = []

        for start_idx, end_idx, lbl, coords in raw_candidates:
            overlap = False
            for a_start, a_end in accepted_spans:
                if max(start_idx, a_start) < min(end_idx, a_end):
                    overlap = True
                    break
            if not overlap:
                accepted_spans.append((start_idx, end_idx))
                x1, y1, x2, y2 = _to_pixel_box(coords[0], coords[1], coords[2], coords[3])
                if x2 > x1 and y2 > y1:
                    detected.append({
                        "label": lbl,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "raw_norm_box": coords
                    })

        # Geometric validity and deduplication
        filtered_detected = []
        seen_boxes = []
        for d in detected:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            w = x2 - x1
            h = y2 - y1
            lbl = d["label"]
            if w <= 3 or h <= 3:
                continue
            if w >= 0.95 * img_w and h >= 0.95 * img_h:
                continue
            if lbl in ["resistor", "capacitor", "diode", "led", "inductor"]:
                aspect = w / (h + 1e-6)
                if aspect > 5.0 or aspect < 0.20:
                    continue
                if w > 0.6 * img_w and h > 0.6 * img_h:
                    continue

            is_dup = False
            for s in seen_boxes:
                if s["label"] == lbl:
                    sx1, sy1, sx2, sy2 = s["bbox_xyxy"]
                    ix1 = max(x1, sx1)
                    iy1 = max(y1, sy1)
                    ix2 = min(x2, sx2)
                    iy2 = min(y2, sy2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = w * h + (sx2 - sx1) * (sy2 - sy1) - inter
                        if union > 0 and (inter / union) > 0.85:
                            is_dup = True
                            break
            if not is_dup:
                seen_boxes.append(d)
                filtered_detected.append(d)

        # Truncation check
        clean_text = text.strip()
        lower_text = clean_text.lower()
        box_start_count = text.count("<|box_start|>")
        box_end_count = text.count("<|box_end|>")
        open_bracket_count = text.count("[")
        close_bracket_count = text.count("]")
        open_paren_count = text.count("(")
        close_paren_count = text.count(")")

        is_syntax_truncated = (
            box_start_count > box_end_count or
            open_bracket_count > close_bracket_count or
            open_paren_count > close_paren_count or
            bool(re.search(r"(?:<\|box_start\b[^>]*|\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)\s*,?\s*|\b(?:resistor|capacitor|ic|connector|button|led|diode|inductor|switch|clock|transistor)\s*\[?\s*|\d+(?:\.\d+)?\s*,\s*)$", clean_text))
        )

        is_clean_empty = (
            clean_text in ["None", "none", "[]", "None.", "none.", "None\n", "none\n"] or
            bool(re.match(r"^(?:none|no components?(?: visible| found| present| detected)?|no objects?(?: visible| found| present| detected)?|no targets?(?: visible| found| present| detected)?|not detected)\.?$", lower_text))
        )

        # Status determination with exact matching
        if filtered_detected:
            if is_syntax_truncated:
                status = "partial_success"
                status_detail = f"Successfully parsed {len(filtered_detected)} bounding boxes, but text was truncated at output limit"
            else:
                status = "success"
                status_detail = f"Successfully parsed {len(filtered_detected)} bounding boxes"
        elif is_clean_empty:
            status = "no_objects_detected"
            status_detail = "Model explicitly reported no components present"
        elif raw_candidates and not filtered_detected:
            status = "parse_filtered"
            status_detail = f"Found {len(raw_candidates)} candidates, but all were filtered out as degenerate/invalid"
        elif clean_text == "":
            status = "parse_failed"
            status_detail = "Empty output generated by model"
        elif is_syntax_truncated:
            status = "parse_failed"
            status_detail = f"Output sequence was truncated before completing any valid bounding box: '{clean_text[-60:]}'"
        else:
            status = "parse_failed"
            status_detail = f"Could not parse valid coordinates from output: '{clean_text[:80]}...'" if len(clean_text) > 80 else f"Could not parse valid coordinates from output: '{clean_text}'"

        return filtered_detected, status, status_detail, is_syntax_truncated

    def parse_grounding_output_with_status(self, text, img_w, img_h, coordinate_protocol="xyxy_norm1000", legacy=False):
        if legacy:
            self.last_parse_audit = {"protocol": "legacy_auto_heuristic", "formal_eligible": False}
            return self.parse_legacy_grounding_output_with_status(text, img_w, img_h)
        result = parse_response(text, img_w, img_h, coordinate_protocol, allow_qwen_tokens=True)
        self.last_parse_audit = result['audit']
        self.last_parse_truncated = result['truncated']
        values = (result['objects'], result['status'], result['detail'])
        return values + (result['truncated'],) if True else values

    def parse_grounding_output(self, text, img_w, img_h, coordinate_protocol="xyxy_norm1000", legacy=False):
        return self.parse_grounding_output_with_status(text, img_w, img_h, coordinate_protocol, legacy)[0]

    def predict_image(
        self,
        image_input: Any,
        prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        max_pixels: int = 1280 * 28 * 28,
        **kwargs
    ) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            image_path = getattr(image_input, "filename", "in_memory_pil_image")
        else:
            image_path = str(image_input)
            pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size

        if prompt is None:
            if kwargs.get('coordinate_protocol', 'xyxy_norm1000') != 'xyxy_norm1000':
                raise ValueError('A nondefault coordinate protocol requires an explicit matching prompt')
            prompt = self.build_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img, "max_pixels": max_pixels},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=kwargs.get("repetition_penalty", 1.25),
                use_cache=True
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram_mb = 0.0

        latency_ms = (time.time() - t0) * 1000

        # Decode output tokens excluding prompt
        input_len = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_len:]
        generated_tokens = int(len(generated_ids))
        has_eos = generation_finished(generated_ids.tolist(), self.model.generation_config.eos_token_id)
        is_token_limit_reached = (generated_tokens >= max_new_tokens and not has_eos)
        finish_reason = "eos_token" if has_eos else ("length" if is_token_limit_reached else "unknown")

        raw_output = self.processor.decode(generated_ids, skip_special_tokens=False)

        detected_objects, status, status_detail, is_syntax_truncated = self.parse_grounding_output_with_status(raw_output, w, h, kwargs.get("coordinate_protocol", "xyxy_norm1000"))
        has_normal_eos = has_eos
        is_truncated = is_token_limit_reached or is_syntax_truncated
        truncation_reason = None
        if is_token_limit_reached and not ("<|im_end|>" in raw_output):
            truncation_reason = f"Max new tokens budget reached ({generated_tokens}/{max_new_tokens})"
            if status == "success":
                status = "partial_success"
                status_detail = f"Successfully parsed {len(detected_objects)} bounding boxes, but generation was cut off by token budget ({generated_tokens}/{max_new_tokens})"
            elif status == "no_objects_detected":
                status = "parse_failed"
                status_detail = f"Generation reached token budget ({generated_tokens}/{max_new_tokens}) without completing a valid response"
        elif is_syntax_truncated:
            truncation_reason = "Syntax truncated (unclosed tags or coordinate delimiters)"
            if status == "success":
                status = "partial_success"
                status_detail = f"Successfully parsed {len(detected_objects)} bounding boxes, but text was truncated at output limit"

        return {
            "image_path": image_path,
            "width": w,
            "height": h,
            "prompt": prompt,
            "raw_output": raw_output,
            "objects": detected_objects,
            "parse_audit": self.last_parse_audit,
            "coordinate_protocol": kwargs.get("coordinate_protocol", "xyxy_norm1000"),
            "coordinate_reference": "input_image_or_tile",
            "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
            "num_detected": len(detected_objects),
            "status": status,
            "status_detail": status_detail,
            "is_truncated": is_truncated,
            "is_syntax_truncated": is_syntax_truncated,
            "has_normal_eos": has_normal_eos,
            "processed_image_grid_thw": inputs.image_grid_thw.tolist(),
            "generated_tokens": generated_tokens,
            "token_budget": max_new_tokens,
            "finish_reason": finish_reason,
            "truncation_reason": truncation_reason,
            "requested_model": self.requested_model,
            "actual_model_path": self.model_path,
            "latency_ms": round(latency_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "device": str(self.device)
        }
