"""
Visual Demonstration Generator (Real Test Set Evaluation Display).
Generates high-resolution annotated PCB detection comparison images
for DuetWIFI_Top and STK644 from the actual Stage 16 test run.
"""

import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont

CATEGORY_COLORS = {
    "resistor": (59, 130, 246),    # Blue
    "capacitor": (16, 185, 129),   # Emerald
    "connector": (245, 158, 11),   # Amber
    "ic": (139, 92, 246),          # Purple
    "led": (239, 68, 68),          # Red
    "button": (236, 72, 153),      # Pink
    "switch": (20, 184, 166),      # Teal
    "diode": (99, 102, 241),       # Indigo
    "transistor": (249, 115, 22),  # Orange
    "inductor": (132, 204, 22),    # Lime
    "clock": (6, 182, 212)         # Cyan
}


def draw_boxes_on_image(img, objects, default_color=None, outline_width=3, draw_labels=True):
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for obj in objects:
        box = obj.get("bbox_xyxy", obj.get("bbox"))
        if not box:
            continue
        # Check if box is [x1, y1, x2, y2] or [x, y, w, h]
        if "bbox_xyxy" in obj:
            x1, y1, x2, y2 = box
        else:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h

        label = obj.get("label", obj.get("category_name", "object"))
        color = default_color if default_color else CATEGORY_COLORS.get(label, (56, 189, 248))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)

        if draw_labels:
            text = f"{label}"
            # Text background
            bbox_t = draw.textbbox((x1, max(0, y1 - 18)), text, font=small_font)
            draw.rectangle([bbox_t[0] - 2, bbox_t[1] - 2, bbox_t[2] + 2, bbox_t[3] + 2], fill=color)
            draw.text((x1, max(0, y1 - 18)), text, fill=(255, 255, 255), font=small_font)

    return draw_img


def main():
    coco_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\coco_annotations.json"
    q3_preds_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_final_test_matrix\Q3_qwen_lora_tiling\predictions.jsonl"
    q0_preds_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_final_test_matrix\Q0_qwen_base_whole\predictions.jsonl"
    out_dir = r"C:\Users\Liu\.gemini\antigravity-ide\brain\f9e12edc-4294-4175-b6e8-02ef4f9b3091"
    os.makedirs(out_dir, exist_ok=True)

    with open(coco_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(q3_preds_file, "r", encoding="utf-8") as f:
        q3_preds = {json.loads(line)["image_id"]: json.loads(line) for line in f if line.strip()}
    with open(q0_preds_file, "r", encoding="utf-8") as f:
        q0_preds = {json.loads(line)["image_id"]: json.loads(line) for line in f if line.strip()}

    # ==========================================
    # Visual 1: DuetWIFI_Top (Image 11) Full & Crop Comparison
    # ==========================================
    img_11_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\DuetWIFI_Top\DuetWIFI_Top.jpg"
    img_11 = Image.open(img_11_path).convert("RGB")
    gts_11 = [a for a in coco["annotations"] if a["image_id"] == 11]
    preds_q3_11 = q3_preds[11]["objects"]
    preds_q0_11 = q0_preds[11]["objects"]

    print(f"DuetWIFI_Top: GT={len(gts_11)}, Q0 Dets={len(preds_q0_11)}, Q3 Dets={len(preds_q3_11)}")

    # 1. Full image with Q3 detections (scaled down to max width 1600 for crisp web viewing)
    full_annotated = draw_boxes_on_image(img_11, preds_q3_11, outline_width=4, draw_labels=True)
    full_annotated.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    out_full = os.path.join(out_dir, "pcb_test_duetwifi_full_detected.png")
    full_annotated.save(out_full, "PNG")
    print(f"Saved full annotated image to {out_full}")

    # 2. Side-by-side Detail Crop (1000x1000 crop focusing on MCU, Connectors, Buttons, Resistors)
    # Crop region around center-top: x in [800, 2000], y in [400, 1600]
    crop_box = (1000, 500, 2200, 1700)
    w_crop = crop_box[2] - crop_box[0]
    h_crop = crop_box[3] - crop_box[1]

    raw_crop = img_11.crop(crop_box)

    # Filter GT inside crop
    crop_gts = []
    for g in gts_11:
        gx, gy, gw, gh = g["bbox"]
        if gx >= crop_box[0] and gy >= crop_box[1] and gx + gw <= crop_box[2] and gy + gh <= crop_box[3]:
            crop_gts.append({
                "bbox_xyxy": [gx - crop_box[0], gy - crop_box[1], gx + gw - crop_box[0], gy + gh - crop_box[1]],
                "label": g["category_name"]
            })

    # Filter Q3 predictions inside crop
    crop_q3 = []
    for p in preds_q3_11:
        px1, py1, px2, py2 = p["bbox_xyxy"]
        if px1 >= crop_box[0] and py1 >= crop_box[1] and px2 <= crop_box[2] and py2 <= crop_box[3]:
            crop_q3.append({
                "bbox_xyxy": [px1 - crop_box[0], py1 - crop_box[1], px2 - crop_box[0], py2 - crop_box[1]],
                "label": p["label"]
            })

    # Filter Q0 predictions inside crop
    crop_q0 = []
    for p in preds_q0_11:
        px1, py1, px2, py2 = p["bbox_xyxy"]
        if px1 >= crop_box[0] and py1 >= crop_box[1] and px2 <= crop_box[2] and py2 <= crop_box[3]:
            crop_q0.append({
                "bbox_xyxy": [px1 - crop_box[0], py1 - crop_box[1], px2 - crop_box[0], py2 - crop_box[1]],
                "label": p["label"]
            })

    gt_annotated_crop = draw_boxes_on_image(raw_crop, crop_gts, default_color=(16, 185, 129), outline_width=4, draw_labels=True)
    q0_annotated_crop = draw_boxes_on_image(raw_crop, crop_q0, default_color=(239, 68, 68), outline_width=4, draw_labels=True)
    q3_annotated_crop = draw_boxes_on_image(raw_crop, crop_q3, outline_width=4, draw_labels=True)

    # Combine into 3-Panel Side-by-Side:
    # Panel 1: Ground Truth (Green)
    # Panel 2: Q0 Zero-Shot Base + Whole Image (Red/Sparse)
    # Panel 3: Q3 Fine-Tuned LoRA + 1024 Tiling (Color-coded Components)
    target_panel_w = 640
    target_panel_h = int(h_crop * (target_panel_w / w_crop))

    p1 = gt_annotated_crop.resize((target_panel_w, target_panel_h), Image.Resampling.LANCZOS)
    p2 = q0_annotated_crop.resize((target_panel_w, target_panel_h), Image.Resampling.LANCZOS)
    p3 = q3_annotated_crop.resize((target_panel_w, target_panel_h), Image.Resampling.LANCZOS)

    header_h = 50
    triptych = Image.new("RGB", (target_panel_w * 3 + 40, target_panel_h + header_h + 30), (15, 23, 42))
    draw_trip = ImageDraw.Draw(triptych)
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        title_font = ImageFont.load_default()

    # Paste panels
    triptych.paste(p1, (10, header_h))
    triptych.paste(p2, (target_panel_w + 20, header_h))
    triptych.paste(p3, (target_panel_w * 2 + 30, header_h))

    # Add Titles
    draw_trip.text((20, 15), f"Panel A: Ground Truth Reference ({len(crop_gts)} GT Objects)", fill=(52, 211, 153), font=title_font)
    draw_trip.text((target_panel_w + 30, 15), f"Panel B: Q0 Zero-Shot Base (Detections={len(crop_q0)})", fill=(248, 113, 113), font=title_font)
    draw_trip.text((target_panel_w * 2 + 40, 15), f"Panel C: Q3 LoRA + 1024 Tiling (Detections={len(crop_q3)})", fill=(56, 189, 248), font=title_font)

    out_comp = os.path.join(out_dir, "pcb_test_detection_comparison.png")
    triptych.save(out_comp, "PNG")
    print(f"Saved side-by-side comparison to {out_comp}")

    # ==========================================
    # Visual 2: STK644 (Image 27) Dense SMT Area Detection
    # ==========================================
    img_27_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\STK644\STK644.jpg"
    img_27 = Image.open(img_27_path).convert("RGB")
    preds_q3_27 = q3_preds[27]["objects"]
    # Crop central area with IC, buttons, connectors: [1500, 1000, 4200, 2800]
    stk_crop_box = (1400, 1000, 4200, 2800)
    stk_crop = img_27.crop(stk_crop_box)
    crop_q3_27 = []
    for p in preds_q3_27:
        px1, py1, px2, py2 = p["bbox_xyxy"]
        if px1 >= stk_crop_box[0] and py1 >= stk_crop_box[1] and px2 <= stk_crop_box[2] and py2 <= stk_crop_box[3]:
            crop_q3_27.append({
                "bbox_xyxy": [px1 - stk_crop_box[0], py1 - stk_crop_box[1], px2 - stk_crop_box[0], py2 - stk_crop_box[1]],
                "label": p["label"]
            })
    stk_annotated = draw_boxes_on_image(stk_crop, crop_q3_27, outline_width=5, draw_labels=True)
    stk_annotated.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    out_stk = os.path.join(out_dir, "pcb_test_stk644_detection.png")
    stk_annotated.save(out_stk, "PNG")
    print(f"Saved STK644 detection to {out_stk}")


if __name__ == "__main__":
    main()
