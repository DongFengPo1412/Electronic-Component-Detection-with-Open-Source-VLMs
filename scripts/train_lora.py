"""
LoRA Fine-Tuning Script for Qwen2-VL-2B on Electronic Component Detection.
Supports:
1. PEFT LoRA (r=16, alpha=32, target_modules: q, k, v, o, gate, up, down).
2. Gradient checkpointing for 8GB VRAM compatibility.
3. Train step iteration, loss logging, and adapter weight checkpointing.
4. Native Qwen2-VL multimodal input collation.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info


class PCBComponentSFTDataset(Dataset):
    def __init__(self, data_file: str, processor, max_pixels: int = 301056, max_objects: int = 60, max_response_len: int = 3500):
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.processor = processor
        self.max_pixels = max_pixels

        # Filter out extreme whole-board samples with 600+ objects that cause sequence explosion
        # Keep high-resolution tile samples with detailed SMT components
        self.data = [
            d for d in raw_data
            if d.get("num_objects", 0) <= max_objects and len(d.get("conversations", [{}, {}])[1].get("value", "")) <= max_response_len
        ]
        print(f"Loaded {len(self.data)}/{len(raw_data)} samples (max_objects<={max_objects}, max_chars<={max_response_len}).")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        conversations = item["conversations"]

        pil_img = Image.open(image_path).convert("RGB")

        user_text = conversations[0]["value"].replace("<image>\n", "")
        assistant_text = conversations[1]["value"]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img, "max_pixels": self.max_pixels},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        image_grid_thw = inputs["image_grid_thw"]

        # Labels: copy input_ids
        labels = input_ids.clone()

        # Mask user prompt tokens so loss is only computed on assistant response
        assistant_prefix = "<|im_start|>assistant\n"
        prefix_ids = self.processor.tokenizer.encode(assistant_prefix, add_special_tokens=False)

        for i in range(len(input_ids) - len(prefix_ids)):
            if input_ids[i:i+len(prefix_ids)].tolist() == prefix_ids:
                labels[:i+len(prefix_ids)] = -100
                break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels
        }


def collate_fn(batch):
    # Dynamic padding
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []

    pad_token_id = 151643 # default Qwen pad token or eos

    for ids, labs, att in zip(input_ids, labels, attention_mask):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), -100, dtype=torch.long)]))
        padded_attention_mask.append(torch.cat([att, torch.zeros((pad_len,), dtype=torch.long)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
        "attention_mask": torch.stack(padded_attention_mask),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0).to(torch.bfloat16),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Qwen2-VL-2B.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to base model weights.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to train_sft.json.")
    parser.add_argument("--val-data-file", type=str, default=None, help="Path to val_sft.json for evaluation.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument("--config", type=str, default="configs/lora.yaml", help="Path to lora.yaml.")
    parser.add_argument("--max-steps", type=int, default=20, help="Max training steps.")
    parser.add_argument("--eval-steps", type=int, default=20, help="Evaluation interval.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"=== Qwen2-VL-2B LoRA SFT Training ===")
    print(f"Base Model:    {args.model_path}")
    print(f"Train Dataset: {args.data_file}")
    print(f"Val Dataset:   {args.val_data_file}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Max Steps:     {args.max_steps}")
    print(f"Eval Steps:    {args.eval_steps}")
    print(f"Device:        {args.device}")

    # Load Base Model
    print("\nLoading base model and processor...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device if args.device == "cuda" else None
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Enable Gradient Checkpointing
    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    # Configure LoRA
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias=lora_cfg.get("bias", "none")
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.4f}%)")

    # Locate base weight and lora weight for parameter isolation verification
    sample_base_param = None
    sample_lora_param = None
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad and sample_lora_param is None:
            sample_lora_param = param
            sample_lora_name = name
        elif "base_layer" in name and not param.requires_grad and sample_base_param is None:
            sample_base_param = param
            sample_base_name = name

    initial_base_val = sample_base_param.data[:5].clone() if sample_base_param is not None else None
    initial_lora_val = sample_lora_param.data[:5].clone() if sample_lora_param is not None else None
    verification_passed = {}

    # Dataset & DataLoader
    dataset = PCBComponentSFTDataset(
        data_file=args.data_file,
        processor=processor,
        max_pixels=cfg["data"].get("image_max_pixels", 301056)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"].get("per_device_train_batch_size", 1),
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = None
    if args.val_data_file and os.path.exists(args.val_data_file):
        val_dataset = PCBComponentSFTDataset(
            data_file=args.val_data_file,
            processor=processor,
            max_pixels=cfg["data"].get("image_max_pixels", 301056)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        print(f"Validation dataset loaded: {len(val_dataset)} samples.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1.0e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01))
    )
    warmup_steps = int(args.max_steps * float(cfg["training"].get("warmup_ratio", 0.05)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_steps
    )

    grad_accum_steps = cfg["training"].get("gradient_accumulation_steps", 4)
    model.train()

    step = 0
    total_loss = 0.0
    history = []
    best_val_loss = float("inf")
    t_start = time.time()

    print("\nStarting training loop...")
    optimizer.zero_grad()

    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / grad_accum_steps
            loss.backward()

            total_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Verify gradients on step 1 (Stage 13 verification requirement)
                if step == 0:
                    lora_grad_nonzero = (sample_lora_param.grad is not None) and (sample_lora_param.grad.abs().sum().item() > 0)
                    base_grad_is_none = (sample_base_param.grad is None)
                    verification_passed["step_1_loss_finite"] = not torch.isnan(loss).item() and not torch.isinf(loss).item()
                    verification_passed["lora_grad_nonzero"] = lora_grad_nonzero
                    verification_passed["base_model_frozen"] = base_grad_is_none
                    print(f"\n[Stage 13 Verification] Loss finite: {verification_passed['step_1_loss_finite']}")
                    print(f"[Stage 13 Verification] LoRA ({sample_lora_name}) grad nonzero: {lora_grad_nonzero}")
                    print(f"[Stage 13 Verification] Base model ({sample_base_name}) frozen (grad is None): {base_grad_is_none}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()

                # Verify weight update on step 1
                if step == 0:
                    lora_weight_updated = not torch.equal(sample_lora_param.data[:5], initial_lora_val)
                    base_weight_untouched = torch.equal(sample_base_param.data[:5], initial_base_val)
                    verification_passed["lora_weight_updated"] = lora_weight_updated
                    verification_passed["base_weight_untouched"] = base_weight_untouched
                    print(f"[Stage 13 Verification] LoRA weight updated after step: {lora_weight_updated}")
                    print(f"[Stage 13 Verification] Base model weight strictly untouched: {base_weight_untouched}\n")

                optimizer.zero_grad()
                step += 1

                step_loss = total_loss / grad_accum_steps
                total_loss = 0.0
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
                curr_lr = scheduler.get_last_lr()[0] if scheduler else cfg["training"]["learning_rate"]

                record = {
                    "step": step,
                    "loss": round(step_loss, 4),
                    "lr": round(curr_lr, 7),
                    "peak_vram_mb": round(peak_vram, 2),
                    "time_elapsed_s": round(time.time() - t_start, 2)
                }

                # Periodic validation evaluation
                if val_loader is not None and (step % args.eval_steps == 0 or step >= args.max_steps):
                    model.eval()
                    val_loss_acc = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for v_i, v_b in enumerate(val_loader):
                            if v_i >= 15:
                                break
                            v_b = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in v_b.items()}
                            val_loss_acc += model(**v_b).loss.item()
                            val_batches += 1
                    mean_val_loss = val_loss_acc / max(1, val_batches)
                    model.train()
                    record["val_loss"] = round(mean_val_loss, 4)
                    print(f"Step [{step:03d}/{args.max_steps}] Loss: {step_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Peak VRAM: {peak_vram:.2f} MB")

                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        best_dir = os.path.join(args.output_dir, "best_adapter")
                        model.save_pretrained(best_dir)
                        processor.save_pretrained(best_dir)
                        print(f"*** New best LoRA adapter saved with Val Loss {mean_val_loss:.4f} -> {best_dir} ***")
                else:
                    print(f"Step [{step:03d}/{args.max_steps}] Loss: {step_loss:.4f} | Peak VRAM: {peak_vram:.2f} MB | Elapsed: {record['time_elapsed_s']}s")

                history.append(record)

                if step % cfg["training"].get("save_steps", 20) == 0 or step >= args.max_steps:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
                    print(f"Saved LoRA checkpoint to {save_dir}")

                if step >= args.max_steps:
                    break

        if step >= args.max_steps:
            break

    # Save verification report if present
    if verification_passed:
        verif_file = os.path.join(args.output_dir, "self_check_verification.json")
        with open(verif_file, "w", encoding="utf-8") as f:
            json.dump({
                "trainable_params": trainable_params,
                "all_params": all_params,
                "trainable_percent": round(100 * trainable_params / all_params, 4),
                "verifications": verification_passed
            }, f, indent=2)

    # Save final model & training log
    final_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nFinal adapter weights saved to {final_dir}")

    # If no val data or best_adapter wasn't created, copy final as best
    best_dir = os.path.join(args.output_dir, "best_adapter")
    if not os.path.exists(best_dir):
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)

    log_file = os.path.join(args.output_dir, "training_history.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history written to {log_file}")

    manifest_file = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "stage": 14,
            "total_steps": step,
            "epochs": round(step * grad_accum_steps / len(dataset), 2),
            "best_val_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else None,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
            "total_time_s": round(time.time() - t_start, 2),
            "config": cfg
        }, f, indent=2)
    print(f"Run manifest written to {manifest_file}")


if __name__ == "__main__":
    main()
