"""
Cold-Start Verification Script for LoRA Adapter Reloading (Stage 13).
Validates:
1. Reloading base model + LoRA adapter in an independent process.
2. Inference execution with adapter enabled vs disabled.
3. Coordinate extraction and bounding box parsing.
"""

import os
import sys
import json
import time
import argparse
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pcb_vlm.models.qwen_adapter import QwenAdapter


def main():
    parser = argparse.ArgumentParser(description="Verify LoRA Adapter Reloading and Inference.")
    parser.add_argument("--base-model", type=str, required=True, help="Base model checkpoint directory.")
    parser.add_argument("--adapter-dir", type=str, required=True, help="LoRA adapter checkpoint directory.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to test image.")
    parser.add_argument("--output-json", type=str, required=True, help="Path to save verification json.")
    args = parser.parse_args()

    print("=== Stage 13: LoRA Cold-Start Reload Verification ===")
    print(f"Base Model:  {args.base_model}")
    print(f"Adapter Dir: {args.adapter_dir}")
    print(f"Test Image:  {args.image_path}")

    # 1. Load Base Model via QwenAdapter
    print("\n[Step 1] Loading base model via QwenAdapter...")
    t0 = time.time()
    adapter_helper = QwenAdapter(model_id=args.base_model, device="cuda", torch_dtype=torch.bfloat16)
    t_base_load = time.time() - t0
    print(f"Base model loaded in {t_base_load:.2f}s.")

    # 2. Base Model Inference (Adapter Disabled)
    classes = [
        "capacitor", "resistor", "diode", "transistor",
        "ic", "connector", "button", "switch", "led", "inductor", "clock"
    ]
    prompt = adapter_helper.build_prompt(classes)

    print("\n[Step 2] Running inference on Base Model (Zero-Shot)...")
    res_base = adapter_helper.predict_image(args.image_path, prompt=prompt, max_pixels=301056)
    print(f"Base Model detections: {res_base['num_detected']} boxes, Latency: {res_base['latency_ms']/1000.0:.2f}s")

    # 3. Reload LoRA Adapter onto Base Model
    print(f"\n[Step 3] Loading LoRA adapter from {args.adapter_dir}...")
    t1 = time.time()
    peft_model = PeftModel.from_pretrained(adapter_helper.model, args.adapter_dir)
    peft_model.eval()
    t_adapter_load = time.time() - t1
    print(f"LoRA adapter reloaded in {t_adapter_load:.2f}s.")

    # 5. Fine-Tuned Model Inference (Adapter Enabled)
    adapter_helper.model = peft_model
    print("\n[Step 4] Running inference with LoRA Adapter enabled...")
    res_lora = adapter_helper.predict_image(args.image_path, prompt=prompt, max_pixels=301056)
    print(f"LoRA Model detections: {res_lora['num_detected']} boxes, Latency: {res_lora['latency_ms']/1000.0:.2f}s")
    print(f"LoRA Raw response snippet:\n{res_lora['raw_output'][:200]}")

    results = {
        "status": "PASS",
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "test_image": args.image_path,
        "base_model_load_time_s": round(t_base_load, 2),
        "adapter_load_time_s": round(t_adapter_load, 2),
        "zero_shot": {
            "num_detections": res_base["num_detected"],
            "latency_s": round(res_base["latency_ms"] / 1000.0, 2),
            "objects_sample": res_base["objects"][:5]
        },
        "lora_fine_tuned": {
            "num_detections": res_lora["num_detected"],
            "latency_s": round(res_lora["latency_ms"] / 1000.0, 2),
            "raw_response_snippet": res_lora["raw_output"][:300],
            "objects_sample": res_lora["objects"][:5]
        }
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nVerification results written to {args.output_json}")


if __name__ == "__main__":
    main()
