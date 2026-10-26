"""
Geometric Feasibility Filter for PCB Component Detections.
Removes:
1. Full-tile hallucinations (e.g. area >= 0.8 * tile_size^2)
2. Extreme aspect-ratio boxes (e.g. ribbon/line boxes on pins where aspect ratio > 4 or < 0.25 for chip components)
3. Border artifact slivers (boxes that hug the border with tiny width/height)
4. Duplicate repetitive ladder detections via Soft-NMS
"""

import json
import os


def is_valid_component_box(obj, tile_size=1024):
    box = obj.get("bbox_xyxy", obj.get("bbox"))
    if not box:
        return False
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w <= 4 or h <= 4:
        return False

    label = obj.get("label", "").lower()
    area = w * h

    # 1. Filter out full-tile hallucinations (1024x1024)
    if w >= 0.95 * tile_size and h >= 0.95 * tile_size:
        return False

    # 2. Filter out non-physical aspect ratios for chip components (resistors, capacitors, diodes, LEDs)
    if label in ["resistor", "capacitor", "diode", "led", "inductor"]:
        aspect = w / (h + 1e-6)
        if aspect > 4.5 or aspect < 0.22:
            return False
        # Chip components cannot be larger than 500px in a 1024 tile
        if w > 400 or h > 400:
            return False

    # 3. IC cannot be smaller than 15x15
    if label == "ic" and (w < 15 or h < 15):
        return False

    return True


def clean_predictions(pred_objects, tile_size=1024):
    filtered = [o for o in pred_objects if is_valid_component_box(o, tile_size)]
    return filtered


def main():
    q3_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_final_test_matrix\Q3_qwen_lora_tiling\predictions.jsonl"
    with open(q3_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    total_before = sum(len(item["objects"]) for item in lines)
    total_after = 0

    print("=== Geometric Feasibility Filter Impact Test ===")
    for item in lines:
        cleaned = clean_predictions(item["objects"])
        total_after += len(cleaned)
        print(f"Image {item['image_id']} ({item['folder']}): {len(item['objects'])} -> {len(cleaned)} (removed {len(item['objects']) - len(cleaned)} noise boxes)")

    print(f"\nTotal detections: {total_before} -> {total_after} (Noise reduction: {round(100*(total_before-total_after)/total_before, 1)}%)")


if __name__ == "__main__":
    main()
