"""DeepSeek-VL2 Model Adapter for Electronic Component Detection and Visual Grounding."""

import os
import re
import time
from typing import Dict, List, Any, Optional
from PIL import Image
import torch


class DeepSeekAdapter:
    def __init__(
        self,
        model_id: str = "deepseek-ai/deepseek-vl2-tiny",
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None

    def load_model(self):
        """Lazy load DeepSeek-VL2 weights to prevent VRAM collision with Qwen."""
        if self.model is not None:
            return

        if os.path.exists(self.model_id):
            model_path = self.model_id
        else:
            print(f"Resolving model {self.model_id} via ModelScope...")
            from modelscope import snapshot_download
            model_path = snapshot_download(self.model_id, cache_dir=self.cache_dir)
            print(f"DeepSeek model path resolved: {model_path}")

        print(f"Loading DeepSeek-VL2 from {model_path} on {self.device}...")
        t0 = time.time()
        from transformers import AutoModelForCausalLM
        try:
            from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
            self.model = DeepseekVLV2ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device == "cuda" else None
            )
            self.processor = DeepseekVLV2Processor.from_pretrained(model_path)
        except ImportError:
            # Fallback to standard transformers with trust_remote_code
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                device_map=self.device if self.device == "cuda" else None
            )
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if self.device != "cuda" and hasattr(self.model, "to"):
            self.model.to(self.device)

        load_time = time.time() - t0
        print(f"DeepSeek-VL2 successfully loaded in {load_time:.2f}s.")

    def build_prompt(self, target_classes: Optional[List[str]] = None) -> str:
        if not target_classes:
            target_classes = ["ic", "capacitor", "resistor", "connector", "led", "transistor", "diode", "inductor", "button"]
        classes_str = ", ".join(target_classes)
        return f"<|ref|>Locate all {classes_str} on this PCB image.<|/ref|><|det|>"

    def parse_grounding_output(self, text: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        """Parse DeepSeek-VL2 grounding tokens <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> to pixel xyxy."""
        detected = []

        # 1. Native DeepSeek-VL2 grounding format:
        # <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], [x3, y3, x4, y4]]<|/det|>
        ref_blocks = re.findall(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>", text, re.DOTALL)
        for label, det_content in ref_blocks:
            clean_label = label.strip().lower()
            # Extract coordinate quadruplets [c1, c2, c3, c4]
            box_matches = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", det_content)
            for c1_s, c2_s, c3_s, c4_s in box_matches:
                c1, c2, c3, c4 = float(c1_s), float(c2_s), float(c3_s), float(c4_s)
                # DeepSeek-VL2 uses normalized [0, 1000] scale in [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
                # Canonical mapping: [ymin, xmin, ymax, xmax]
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

        # 2. Fallback regex for standard [c1, c2, c3, c4] bracketed quadruplets
        if not detected:
            quad_pattern = r"([a-zA-Z_\-]+)\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
            for label, c1_s, c2_s, c3_s, c4_s in re.findall(quad_pattern, text):
                c1, c2, c3, c4 = float(c1_s), float(c2_s), float(c3_s), float(c4_s)
                x1 = max(0.0, min(float(img_w), (c2 / 1000.0) * img_w))
                y1 = max(0.0, min(float(img_h), (c1 / 1000.0) * img_h))
                x2 = max(0.0, min(float(img_w), (c4 / 1000.0) * img_w))
                y2 = max(0.0, min(float(img_h), (c3 / 1000.0) * img_h))
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
        max_new_tokens: int = 1024
    ) -> Dict[str, Any]:
        self.load_model()

        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size

        if prompt is None:
            prompt = self.build_prompt()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        # Conversation input for DeepSeek-VL2
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt}",
                "images": [pil_img]
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = self.processor(
            conversations=conversation,
            images=[pil_img],
            force_batchify=True,
            system_prompt=""
        ).to(self.model.device)

        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram_mb = 0.0

        latency_ms = (time.time() - t0) * 1000
        raw_output = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
        detected_objects = self.parse_grounding_output(raw_output, w, h)

        return {
            "image_path": image_path,
            "width": w,
            "height": h,
            "prompt": prompt,
            "raw_output": raw_output,
            "objects": detected_objects,
            "num_detected": len(detected_objects),
            "latency_ms": round(latency_ms, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "device": str(self.device)
        }
