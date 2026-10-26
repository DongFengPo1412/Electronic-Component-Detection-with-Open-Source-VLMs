"""
Extract and visualize exact True Positive detections from Stage 16 test matrix.
"""

import os
import json
from PIL import Image, ImageDraw, ImageFont


def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def main():
    coco_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\coco_annotations.json"
    preds_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage16_final_test_matrix\Q3_qwen_lora_tiling\predictions.jsonl"
    out_dir = r"C:\Users\Liu\.gemini\antigravity-ide\brain\f9e12edc-4294-4175-b6e8-02ef4f9b3091"
    os.makedirs(out_dir, exist_ok=True)

    with open(coco_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_map = {im["id"]: im for im in coco["images"]}
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    preds_by_img = {}
    with open(preds_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                preds_by_img[item["image_id"]] = item

    gts_by_img = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gts_by_img:
            gts_by_img[img_id] = []
        x, y, w, h = ann["bbox"]
        gts_by_img[img_id].append({
            "bbox_xyxy": [x, y, x + w, y + h],
            "category": ann["category_name"],
            "ann_id": ann["id"]
        })

    true_positives = []

    for img_id, pred_data in preds_by_img.items():
        if img_id not in gts_by_img:
            continue
        gts = gts_by_img[img_id]
        preds = pred_data.get("objects", [])

        # Match greedily
        matched_gt = set()
        for p in preds:
            p_box = p["bbox_xyxy"]
            p_lbl = p["label"]
            best_iou = 0.0
            best_gt_idx = -1

            for idx, g in enumerate(gts):
                if idx in matched_gt:
                    continue
                if g["category"] == p_lbl:
                    iou = compute_iou(p_box, g["bbox_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou >= 0.10 and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                true_positives.append({
                    "image_id": img_id,
                    "image_file": img_map[img_id]["file_name"],
                    "category": p_lbl,
                    "iou": best_iou,
                    "gt_box": gts[best_gt_idx]["bbox_xyxy"],
                    "pred_box": p_box
                })

    print(f"Total True Positives extracted: {len(true_positives)}")
    # Sort by IoU descending
    true_positives.sort(key=lambda x: x["iou"], reverse=True)

    for i, tp in enumerate(true_positives[:10]):
        print(f"TP #{i+1}: {tp['category']}, IoU={tp['iou']:.3f}, Img={tp['image_file']}")

    # Render a Showcase Grid of top True Positives across categories
    # Pick top connectors, button, IC, resistors
    showcase = []
    seen_cats = {}
    for tp in true_positives:
        c = tp["category"]
        seen_cats[c] = seen_cats.get(c, 0) + 1
        if seen_cats[c] <= 3:
            showcase.append(tp)
        if len(showcase) >= 6:
            break

    card_w, card_h = 450, 450
    cols = 3
    rows = (len(showcase) + cols - 1) // cols
    gallery = Image.new("RGB", (cols * card_w + (cols + 1) * 20, rows * card_h + (rows + 1) * 30 + 50), (15, 23, 42))
    draw_gal = ImageDraw.Draw(gallery)

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_label = ImageFont.truetype("arial.ttf", 16)
        font_sub = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = font_title
        font_sub = font_title

    draw_gal.text((20, 15), "Real Test Set True Positives Showcase (IoU Overlap >= 0.10)", fill=(241, 245, 249), font=font_title)

    raw_dir = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019"

    for idx, item in enumerate(showcase):
        r = idx // cols
        c = idx % cols
        x_card = 20 + c * (card_w + 20)
        y_card = 60 + r * (card_h + 30)

        # Load image using absolute_path
        img_p = img_map[item["image_id"]]["absolute_path"]
        im = Image.open(img_p).convert("RGB")

        # Determine context crop window around GT box
        gt = item["gt_box"]
        pred = item["pred_box"]
        pad_x = max(60, (gt[2] - gt[0]) * 0.6)
        pad_y = max(60, (gt[3] - gt[1]) * 0.6)

        cx1 = max(0, min(gt[0], pred[0]) - pad_x)
        cy1 = max(0, min(gt[1], pred[1]) - pad_y)
        cx2 = min(im.width, max(gt[2], pred[2]) + pad_x)
        cy2 = min(im.height, max(gt[3], pred[3]) + pad_y)

        crop = im.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop.size
        # Draw on crop
        d_crop = ImageDraw.Draw(crop)

        # Transform to crop coords
        g_box_c = [gt[0] - cx1, gt[1] - cy1, gt[2] - cx1, gt[3] - cy1]
        p_box_c = [pred[0] - cx1, pred[1] - cy1, pred[2] - cx1, pred[3] - cy1]

        # Draw GT in Emerald Green
        d_crop.rectangle(g_box_c, outline=(16, 185, 129), width=4)
        # Draw Pred in Sky Blue
        d_crop.rectangle(p_box_c, outline=(56, 189, 248), width=3)

        crop = crop.resize((card_w, card_h - 70), Image.Resampling.LANCZOS)
        gallery.paste(crop, (x_card, y_card))

        # Card header / caption
        board_name = img_map[item["image_id"]].get("board_group_id", "PCB")
        draw_gal.rectangle([x_card, y_card + card_h - 65, x_card + card_w, y_card + card_h], fill=(30, 41, 59))
        draw_gal.text((x_card + 10, y_card + card_h - 60), f"Class: {item['category'].upper()} | IoU: {item['iou']:.3f}", fill=(56, 189, 248), font=font_label)
        draw_gal.text((x_card + 10, y_card + card_h - 35), f"Board: {board_name} | Green=Ground Truth, Blue=VLM Pred", fill=(148, 163, 184), font=font_sub)

    out_gal = os.path.join(out_dir, "pcb_true_positives_gallery.png")
    gallery.save(out_gal, "PNG")
    print(f"Saved True Positives gallery to: {out_gal}")


if __name__ == "__main__":
    main()
