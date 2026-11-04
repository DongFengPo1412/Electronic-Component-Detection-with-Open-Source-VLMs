"""DeepSeek-VL-1.3B Model Adapter for PCB Image Analysis.

Architecture & Scope Note:
    The deployed checkpoint is `deepseek-ai/deepseek-vl-1.3b-chat` (DeepSeek-VL v1 generation).
    Unlike DeepSeek-VL2 (which introduced native visual grounding with `<|ref|>` and `<|det|>` heads),
    DeepSeek-VL-1.3B is a general vision-language dialogue foundation model.
    When given explicit coordinate grounding instructions, if it generates natural language descriptions
    instead of coordinates, the parser accurately flags the output as `parse_failed` (format mismatch)
    rather than misrepresenting it as `no_objects_detected`.
"""

import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torch

from pcb_vlm.geometry.coordinate_protocol import parse_response, generation_finished
from pcb_vlm.config import get_canonical_categories
DEFAULT_PCB_CLASSES = get_canonical_categories()


class DeepSeekAdapter:
    def __init__(
        self,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        known_aliases = {
            "deepseek-ai/deepseek-vl-1.3b-chat": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master",
            "deepseek-vl-1.3b": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master",
            "deepseek": r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master"
        }
        requested = model_path or model_id or "deepseek-ai/deepseek-vl-1.3b-chat"
        self.requested_model = requested
        if model_path and model_path not in known_aliases and not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)

        if model_path and os.path.exists(model_path):
            target_path = model_path
        elif model_id and os.path.isdir(model_id):
            target_path = model_id
        elif model_id and model_id in known_aliases:
            target_path = known_aliases[model_id]
        elif requested in known_aliases:
            target_path = known_aliases[requested]
        elif os.path.exists(r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master") and (model_id == "deepseek-ai/deepseek-vl-1.3b-chat" or not model_id):
            target_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master"
        else:
            raise FileNotFoundError(
                f"Requested model '{requested}' does not exist on disk and is not a recognized local alias! "
                f"Silent fallback is strictly prohibited."
            )

        self.model_id = target_path
        self.model_path = target_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None

    def load_model(self):
        """Lazy load DeepSeek-VL weights to prevent VRAM collision with Qwen."""
        if self.model is not None:
            return

        model_path = self.model_path
        print(f"Loading DeepSeek-VL from {model_path} (requested: {self.requested_model}) on {self.device}...")
        t0 = time.time()
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
        self.processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = MultiModalityCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == "cuda" else None
        )

        if self.device != "cuda" and hasattr(self.model, "to"):
            self.model.to(self.device)

        self.model.eval()
        load_time = time.time() - t0
        print(f"DeepSeek-VL successfully loaded in {load_time:.2f}s.")

    def build_prompt(self, target_classes: Optional[List[str]] = None) -> str:
        """Construct prompt for DeepSeek-VL-1.3B with unified 11 PCB classes and explicit coordinates request."""
        if not target_classes:
            target_classes = DEFAULT_PCB_CLASSES
        classes_str = ", ".join(target_classes)
        return (
            f"Locate all {classes_str} on this PCB image. "
            f"Output each bounding box in normalized [0, 1000] coordinates: category [ymin, xmin, ymax, xmax]. "
            f"If no target components are visible, output 'None'."
        )

    def parse_legacy_grounding_output_with_status(self, text: str, img_w: int, img_h: int) -> Tuple[List[Dict[str, Any]], str, str]:
        """Parse grounding tokens or coordinate tuples to pixel xyxy, returning explicit status."""
        detected = []

        # 1. Native DeepSeek-VL2 grounding format:
        # <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], [x3, y3, x4, y4]]<|/det|>
        ref_blocks = re.findall(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>", text, re.DOTALL)
        for label, det_content in ref_blocks:
            clean_label = label.strip().lower()
            box_matches = re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", det_content)
            for c1_s, c2_s, c3_s, c4_s in box_matches:
                c1, c2, c3, c4 = float(c1_s), float(c2_s), float(c3_s), float(c4_s)
                # DeepSeek canonical coordinate scale [ymin, xmin, ymax, xmax] in [0, 1000]
                x1 = max(0.0, min(float(img_w), (c2 / 1000.0) * img_w))
                y1 = max(0.0, min(float(img_h), (c1 / 1000.0) * img_h))
                x2 = max(0.0, min(float(img_w), (c4 / 1000.0) * img_w))
                y2 = max(0.0, min(float(img_h), (c3 / 1000.0) * img_h))

                if x2 > x1 and y2 > y1:
                    detected.append({
                        "label": clean_label,
                        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "raw_norm_box": [c1, c2, c3, c4]
                    })

        # 2. Fallback regex for bracketed quadruplets: [category] [c1, c2, c3, c4]
        # Remove native grounding blocks so their inner brackets are not matched twice
        remaining_text = re.sub(r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", " ", text, flags=re.DOTALL)
        quad_pattern = r"(?:([a-zA-Z_\-]+)\s*)?\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"
        for label, c1_s, c2_s, c3_s, c4_s in re.findall(quad_pattern, remaining_text):
            clean_label = label.strip().lower() if label else "component"
            c1, c2, c3, c4 = float(c1_s), float(c2_s), float(c3_s), float(c4_s)
            x1 = max(0.0, min(float(img_w), (c2 / 1000.0) * img_w))
            y1 = max(0.0, min(float(img_h), (c1 / 1000.0) * img_h))
            x2 = max(0.0, min(float(img_w), (c4 / 1000.0) * img_w))
            y2 = max(0.0, min(float(img_h), (c3 / 1000.0) * img_h))
            if x2 > x1 and y2 > y1:
                detected.append({
                    "label": clean_label,
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "raw_norm_box": [c1, c2, c3, c4]
                })

        # Geometric validity and anti-repetition filter
        filtered_detected = []
        for d in detected:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            w = x2 - x1
            h = y2 - y1
            lbl = d["label"]
            if w <= 3 or h <= 3:
                continue
            if w >= 0.92 * img_w and h >= 0.92 * img_h:
                continue
            if lbl in ["resistor", "capacitor", "diode", "led", "inductor"]:
                aspect = w / (h + 1e-6)
                if aspect > 4.5 or aspect < 0.22:
                    continue
                if w > 0.5 * img_w and h > 0.5 * img_h:
                    continue
            filtered_detected.append(d)

        # Status determination
        clean_text = text.strip()
        lower_text = clean_text.lower()

        if filtered_detected:
            status = "success"
            status_detail = f"Successfully parsed {len(filtered_detected)} bounding boxes"
        elif clean_text != "" and (clean_text == "None" or clean_text == "[]" or bool(re.search(r"\b(?:none|no components?|no objects?|no targets?|not detected)\b", lower_text))):
            status = "no_objects_detected"
            status_detail = "Model explicitly reported no components present in the image"
        elif clean_text == "":
            status = "parse_failed"
            status_detail = "Empty response generated by model"
        else:
            # Model generated natural language / conversational descriptions without bounding box coordinates
            status = "parse_failed"
            status_detail = f"Model generated conversational text without valid bounding box coordinates (format mismatch): '{clean_text[:80]}...'"

        return filtered_detected, status, status_detail

    def parse_grounding_output_with_status(self, text, img_w, img_h, coordinate_protocol="yxyx_norm1000", legacy=False):
        if legacy:
            self.last_parse_audit = {"protocol": "legacy_auto_heuristic", "formal_eligible": False}
            return self.parse_legacy_grounding_output_with_status(text, img_w, img_h)
        result = parse_response(text, img_w, img_h, coordinate_protocol, allow_qwen_tokens=False)
        self.last_parse_audit = result['audit']
        self.last_parse_truncated = result['truncated']
        values = (result['objects'], result['status'], result['detail'])
        return values + (result['truncated'],) if False else values

    def parse_grounding_output(self, text, img_w, img_h, coordinate_protocol="yxyx_norm1000", legacy=False):
        return self.parse_grounding_output_with_status(text, img_w, img_h, coordinate_protocol, legacy)[0]

    def predict_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        max_pixels: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        self.load_model()

        image_input = image_path
        pil_img = image_input.convert("RGB") if isinstance(image_input, Image.Image) else Image.open(image_input).convert("RGB")
        image_path = str(getattr(image_input, "filename", "in_memory_pil_image")) if isinstance(image_input, Image.Image) else str(image_input)
        w, h = pil_img.size

        if prompt is None:
            if kwargs.get('coordinate_protocol', 'yxyx_norm1000') != 'yxyx_norm1000':
                raise ValueError('A nondefault coordinate protocol requires an explicit matching prompt')
            prompt = self.build_prompt()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [pil_img]
            },
            {"role": "Assistant", "content": ""}
        ]

        prepare_inputs = self.processor(
            conversations=conversation,
            images=[pil_img],
            force_batchify=True
        ).to(self.model.device)

        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            gen_model = getattr(self.model, "language_model", self.model)
            outputs = gen_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
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
        raw_output = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        detected_objects, status, status_detail = self.parse_grounding_output_with_status(raw_output, w, h, kwargs.get("coordinate_protocol", "yxyx_norm1000"))

        generated_ids = outputs[0].cpu().tolist()
        has_eos = generation_finished(generated_ids, self.processor.tokenizer.eos_token_id)
        is_syntax_truncated = getattr(self, 'last_parse_truncated', False)
        is_truncated = (len(generated_ids) >= max_new_tokens and not has_eos) or is_syntax_truncated
        if is_truncated and status == 'success': status = 'partial_success'
        if is_truncated and status == 'no_objects_detected': status = 'parse_failed'
        return {
            "has_normal_eos": has_eos,
            "generated_tokens": len(generated_ids),
            "is_truncated": is_truncated,
            "is_syntax_truncated": is_syntax_truncated,
            "finish_reason": "eos_token" if has_eos else ("length" if len(generated_ids) >= max_new_tokens else "unknown"),
            "native_grounding_verified": False,
            "max_pixels_applied": False,
            "image_preprocessing_policy": "VL1 processor fixed resize/pad; max_pixels is not an effective parameter",
            "processed_pixel_values_shape": list(prepare_inputs.pixel_values.shape),
            "model_family": "DeepSeek-VL-v1",
            "image_path": image_path,
            "width": w,
            "height": h,
            "prompt": prompt,
            "raw_output": raw_output,
            "objects": detected_objects,
            "parse_audit": self.last_parse_audit,
            "coordinate_protocol": kwargs.get("coordinate_protocol", "yxyx_norm1000"),
            "coordinate_reference": "input_image_or_tile",
            "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
            "num_detected": len(detected_objects),
            "status": status,
            "status_detail": status_detail,
            "latency_ms": round(latency_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "device": str(self.device)
        }
