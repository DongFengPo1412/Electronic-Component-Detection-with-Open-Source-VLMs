"""Qwen2-VL Model Adapter for Electronic Component Detection and Visual Grounding."""

import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torch


class QwenAdapter:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir

        # Determine model path
        if os.path.exists(model_id):
            model_path = model_id
        else:
            print(f"Resolving model {model_id} via ModelScope...")
            from modelscope import snapshot_download
            model_path = snapshot_download(model_id, cache_dir=self.cache_dir)
            print(f"Model path resolved: {model_path}")

        print(f"Loading Qwen2-VL from {model_path} on {device} ({torch_dtype})...")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        t0 = time.time()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None
        )
        if device != "cuda":
            self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_path)
        load_time = time.time() - t0
        print(f"Qwen2-VL successfully loaded in {load_time:.2f}s.")

    def build_prompt(self, target_classes: Optional[List[str]] = None) -> str:
        if not target_classes:
            target_classes = ["resistor", "capacitor", "ic", "connector", "led", "transistor", "diode", "inductor", "button"]
        classes_str = ", ".join(target_classes)
        return (
            f"Locate and detect all electronic components on this PCB image: {classes_str}. "
            f"For each detected component, provide its label and bounding box in the format: "
            f"label <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>."
        )

    def parse_grounding_output(self, text: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        """Parse Qwen2-VL grounding tokens <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|> to pixel xyxy."""
        detected = []

        # Standard Qwen format: label <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
        pattern = r"([a-zA-Z_\-]+)\s*<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>"
        matches = re.findall(pattern, text)

        for label, ymin_s, xmin_s, ymax_s, xmax_s in matches:
            ymin = float(ymin_s)
            xmin = float(xmin_s)
            ymax = float(ymax_s)
            xmax = float(xmax_s)

            # Convert from [0, 1000] scale to image pixels
            x1 = max(0.0, min(float(img_w), (xmin / 1000.0) * img_w))
            y1 = max(0.0, min(float(img_h), (ymin / 1000.0) * img_h))
            x2 = max(0.0, min(float(img_w), (xmax / 1000.0) * img_w))
            y2 = max(0.0, min(float(img_h), (ymax / 1000.0) * img_h))

            if x2 > x1 and y2 > y1:
                detected.append({
                    "label": label.strip().lower(),
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "raw_norm_box": [ymin, xmin, ymax, xmax]
                })

        # Fallback regex for bracketed formats: label [ymin, xmin, ymax, xmax]
        if not detected:
            bracket_pattern = r"([a-zA-Z_\-]+)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
            for label, c1, c2, c3, c4 in re.findall(bracket_pattern, text):
                c1, c2, c3, c4 = float(c1), float(c2), float(c3), float(c4)
                # If values <= 1000, treat as normalized ymin, xmin, ymax, xmax
                if max(c1, c2, c3, c4) <= 1000:
                    x1 = (c2 / 1000.0) * img_w
                    y1 = (c1 / 1000.0) * img_h
                    x2 = (c4 / 1000.0) * img_w
                    y2 = (c3 / 1000.0) * img_h
                else:
                    x1, y1, x2, y2 = c1, c2, c3, c4
                if x2 > x1 and y2 > y1:
                    detected.append({
                        "label": label.strip().lower(),
                        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "raw_norm_box": [c1, c2, c3, c4]
                    })

        return detected

    def predict_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
        max_pixels: int = 1280 * 28 * 28
    ) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size

        if prompt is None:
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

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram_mb = 0.0

        latency_ms = (time.time() - t0) * 1000

        # Trim input prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        detected_objects = self.parse_grounding_output(output_text, w, h)

        return {
            "image_path": os.path.abspath(image_path),
            "width": w,
            "height": h,
            "prompt": prompt,
            "raw_output": output_text,
            "objects": detected_objects,
            "num_detected": len(detected_objects),
            "latency_ms": round(latency_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "device": self.device
        }
