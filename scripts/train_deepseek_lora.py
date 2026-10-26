"""LoRA Fine-Tuning Script for DeepSeek-VL-1.3B on PCB Electronic Component Grounding."""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from deepseek_vl.models import VLChatProcessor


class PCBDeepSeekSFTDataset(Dataset):
    def __init__(self, data_file: str, processor: VLChatProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        with open(data_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        pil_img = Image.open(image_path).convert("RGB")

        user_content = item["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
        assistant_content = item["conversations"][1]["value"]

        # DeepSeek-VL format
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{user_content}",
                "images": [pil_img]
            },
            {
                "role": "Assistant",
                "content": assistant_content
            }
        ]

        prepare = self.processor(
            conversations=conversation,
            images=[pil_img],
            force_batchify=False
        )

        input_ids = prepare.input_ids
        pixel_values = prepare.pixel_values  # [1, 3, 384, 384]
        num_image_tokens = prepare.num_image_tokens

        # Build labels: mask User prompt, keep Assistant response
        labels = input_ids.clone()
        # Find where Assistant starts
        assistant_prefix = "\n\nAssistant: "
        prefix_ids = self.processor.tokenizer.encode(assistant_prefix, add_special_tokens=False)
        for i in range(len(input_ids) - len(prefix_ids)):
            if input_ids[i:i+len(prefix_ids)].tolist() == prefix_ids:
                labels[:i+len(prefix_ids)] = -100
                break

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "num_image_tokens": num_image_tokens
        }


def collate_deepseek(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)

    max_len = max(len(ids) for ids in input_ids)
    pad_token_id = 100001 # DeepSeek pad or eos

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for ids, labs in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(attention_masks),
        "pixel_values": pixel_values
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for DeepSeek-VL-1.3B.")
    parser.add_argument("--model-path", type=str, default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master")
    parser.add_argument("--data-file", type=str, default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\sft_data_v2\train_sft.json")
    parser.add_argument("--val-data-file", type=str, default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\sft_data_v2\val_sft.json")
    parser.add_argument("--output-dir", type=str, default=r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage14_deepseek_lora")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-qlora", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== DeepSeek-VL-1.3B LoRA SFT Training ===")
    print(f"Base Model:    {args.model_path}")
    print(f"Train Dataset: {args.data_file}")
    print(f"Val Dataset:   {args.val_data_file}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Max Steps:     {args.max_steps}")
    print(f"Device:        {args.device}")

    # Load processor
    print("Loading VLChatProcessor...")
    processor = VLChatProcessor.from_pretrained(args.model_path)

    # Load model
    print("Loading DeepSeek-VL model...")
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_cfg,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        print("Loaded DeepSeek-VL with 4-bit NormalFloat4 (QLoRA).")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device if args.device == "cuda" else None
        )

    # Freeze vision encoder and aligner, train language model with LoRA
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.aligner.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model.language_model = get_peft_model(model.language_model, peft_config)
    model.language_model.print_trainable_parameters()

    # Datasets
    train_dataset = PCBDeepSeekSFTDataset(args.data_file, processor)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_deepseek)

    val_loader = None
    if args.val_data_file and os.path.exists(args.val_data_file):
        val_dataset = PCBDeepSeekSFTDataset(args.val_data_file, processor)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_deepseek)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.language_model.parameters(), lr=1.5e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.max_steps * 0.05), num_training_steps=args.max_steps)

    grad_accum_steps = 4
    model.train()
    step = 0
    total_loss = 0.0
    history = []
    best_val_loss = float("inf")
    t_start = time.time()
    optimizer.zero_grad()

    print("\nStarting DeepSeek training loop...")
    for epoch in range(50):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            pixel_values = batch["pixel_values"].to(args.device, dtype=torch.bfloat16)

            bs, seq_len = input_ids.shape
            images_seq_mask = (input_ids == processor.image_id)
            n_images = pixel_values.shape[0] // bs
            n_img_tokens = processor.num_image_tokens
            images_emb_mask = torch.ones((bs, n_images, n_img_tokens), dtype=torch.bool, device=args.device)

            inputs_embeds = model.prepare_inputs_embeds(
                input_ids=input_ids,
                pixel_values=pixel_values.view(bs, n_images, 3, 384, 384),
                images_seq_mask=images_seq_mask,
                images_emb_mask=images_emb_mask
            )

            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            total_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.language_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                step_loss = total_loss / grad_accum_steps
                total_loss = 0.0
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                record = {
                    "step": step,
                    "loss": round(step_loss, 4),
                    "peak_vram_mb": round(peak_vram, 2),
                    "elapsed_s": round(time.time() - t_start, 2)
                }

                if val_loader is not None and (step % args.eval_steps == 0 or step >= args.max_steps):
                    model.eval()
                    val_loss_acc = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for v_i, v_b in enumerate(val_loader):
                            if v_i >= 15:
                                break
                            v_in = v_b["input_ids"].to(args.device)
                            v_lab = v_b["labels"].to(args.device)
                            v_att = v_b["attention_mask"].to(args.device)
                            v_pix = v_b["pixel_values"].to(args.device, dtype=torch.bfloat16)
                            v_bs = v_in.shape[0]
                            v_seq_mask = (v_in == processor.image_id)
                            v_emb_mask = torch.ones((v_bs, 1, processor.num_image_tokens), dtype=torch.bool, device=args.device)
                            v_embeds = model.prepare_inputs_embeds(
                                input_ids=v_in,
                                pixel_values=v_pix.view(v_bs, 1, 3, 384, 384),
                                images_seq_mask=v_seq_mask,
                                images_emb_mask=v_emb_mask
                            )
                            val_loss_acc += model.language_model(inputs_embeds=v_embeds, attention_mask=v_att, labels=v_lab).loss.item()
                            val_count += 1
                    mean_val_loss = val_loss_acc / max(1, val_count)
                    model.train()
                    record["val_loss"] = round(mean_val_loss, 4)
                    print(f"Step [{step:03d}/{args.max_steps}] Loss: {step_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Peak VRAM: {peak_vram:.2f} MB", flush=True)

                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        best_dir = os.path.join(args.output_dir, "best_adapter")
                        model.language_model.save_pretrained(best_dir)
                        processor.save_pretrained(best_dir)
                        print(f"*** New best DeepSeek LoRA adapter saved with Val Loss {mean_val_loss:.4f} -> {best_dir} ***", flush=True)
                else:
                    print(f"Step [{step:03d}/{args.max_steps}] Loss: {step_loss:.4f} | Peak VRAM: {peak_vram:.2f} MB | Elapsed: {record['elapsed_s']}s", flush=True)

                history.append(record)

                if step % 25 == 0 or step >= args.max_steps:
                    chk_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    model.language_model.save_pretrained(chk_dir)
                    processor.save_pretrained(chk_dir)

                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break

    # Save final
    final_dir = os.path.join(args.output_dir, "final_adapter")
    model.language_model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    best_dir = os.path.join(args.output_dir, "best_adapter")
    if not os.path.exists(best_dir):
        model.language_model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)

    with open(os.path.join(args.output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.output_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": "deepseek-vl-1.3b-chat",
            "total_steps": step,
            "best_val_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else None,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
            "total_time_s": round(time.time() - t_start, 2)
        }, f, indent=2)

    print("DeepSeek-VL LoRA training completed successfully!", flush=True)


if __name__ == "__main__":
    main()
