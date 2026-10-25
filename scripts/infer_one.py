"""Single-image inference script for PCB Electronic Component Detection using Qwen2-VL."""

import argparse
import json
import os
import sys
import cv2
import torch

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pcb_vlm.models.qwen_adapter import QwenAdapter


COLOR_PALETTE = {
    "capacitor": (0, 165, 255),   # Orange
    "resistor": (0, 255, 0),     # Green
    "ic": (0, 0, 255),           # Red
    "connector": (255, 255, 0),   # Cyan
    "led": (255, 0, 255),         # Magenta
    "transistor": (255, 140, 0),  # Dark Orange
    "diode": (147, 20, 255),      # Pink
    "inductor": (211, 0, 148),    # Purple
    "button": (0, 215, 255),      # Gold
    "switch": (128, 128, 0),      # Teal
    "clock": (180, 105, 255)      # Hot Pink
}


def draw_predictions(image_path: str, objects: list, output_viz_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image for visualization at {image_path}")
        return

    h, w = img.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_xyxy"]
        label = obj["label"]
        color = COLOR_PALETTE.get(label, (255, 255, 255))

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), max(18, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resize if image is huge for preview
    max_dim = 1600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    os.makedirs(os.path.dirname(output_viz_path), exist_ok=True)
    cv2.imwrite(output_viz_path, img)
    print(f"Visualization saved to: {output_viz_path}")


def main():
    parser = argparse.ArgumentParser(description="Run single image inference with Qwen2-VL.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID on Hugging Face.")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt string.")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save prediction JSON.")
    parser.add_argument("--output-viz", type=str, default=None, help="Path to save visual preview.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28, help="Max pixels for vision encoder.")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        sys.exit(1)

    print(f"--- Qwen2-VL Single Image Inference ---")
    print(f"Image: {args.image}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Max Pixels: {args.max_pixels}")

    adapter = QwenAdapter(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir
    )

    print("\nRunning inference...")
    result = adapter.predict_image(args.image, prompt=args.prompt, max_pixels=args.max_pixels)

    print("\n=== Inference Results ===")
    print(f"  Latency:      {result['latency_ms']:.2f} ms")
    print(f"  Peak VRAM:    {result['peak_vram_mb']:.2f} MB")
    print(f"  Detected:     {result['num_detected']} components")
    for idx, obj in enumerate(result["objects"][:15]):
        print(f"    [{idx+1}] {obj['label']:12s} bbox={obj['bbox_xyxy']}")
    if result["num_detected"] > 15:
        print(f"    ... and {result['num_detected'] - 15} more.")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nPrediction JSON saved to: {args.output_json}")

    if args.output_viz:
        draw_predictions(args.image, result["objects"], args.output_viz)


if __name__ == "__main__":
    main()
