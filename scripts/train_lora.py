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
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pcb_vlm.data.resumable_sampler import ResumableBatchSampler
from pcb_vlm.provenance import validate_samples, model_identity, code_identity, dependencies, verify_adapter, source_identity
from pcb_vlm.config import get_canonical_categories

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from qwen_vl_utils import process_vision_info


class PCBComponentSFTDataset(Dataset):
    def __init__(self, data_file: str, processor, max_pixels: int = 301056, max_objects: int = 60, max_response_len: int = 3500, max_sequence_length: Optional[int] = None):
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.processor = processor
        self.max_pixels = max_pixels
        self.max_sequence_length = max_sequence_length

        # Filter out extreme whole-board samples with 600+ objects that cause sequence explosion
        # Keep high-resolution tile samples with detailed SMT components
        self.data = [
            d for d in raw_data
            if (max_objects is None or d.get("num_objects", 0) <= max_objects) and (max_response_len is None or len(d.get("conversations", [{}, {}])[1].get("value", "")) <= max_response_len)
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
        if self.max_sequence_length is not None and len(input_ids) > self.max_sequence_length:
            raise ValueError(f"Sample {item.get('id', idx)} exceeds training sequence budget: {len(input_ids)} > {self.max_sequence_length}; labels are never truncated")
        attention_mask = inputs["attention_mask"][0]
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        image_grid_thw = inputs["image_grid_thw"]

        # Labels: copy input_ids
        labels = input_ids.clone()

        # Mask user prompt tokens so loss is only computed on assistant response
        assistant_prefix = "<|im_start|>assistant\n"
        prefix_ids = self.processor.tokenizer.encode(assistant_prefix, add_special_tokens=False)

        found_prefix = False
        for i in range(len(input_ids) - len(prefix_ids)):
            if input_ids[i:i+len(prefix_ids)].tolist() == prefix_ids:
                labels[:i+len(prefix_ids)] = -100
                found_prefix = True
                break

        # Strict assertion: supervision tokens must be present
        assert found_prefix, f"Sample {item.get('id', idx)} failed to locate assistant prefix for loss masking!"
        valid_target_tokens = (labels != -100).sum().item()
        assert valid_target_tokens > 0, f"Sample {item.get('id', idx)} has zero valid supervision tokens! Masking failed."

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "labels": labels
        }


def collate_fn(batch, pad_token_id=151643):
    # Dynamic padding
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_labels = []
    padded_attention_mask = []

    # pad_token_id is supplied from the actual processor tokenizer in training.

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


def save_adapter_with_provenance_metadata(
    save_dir: str,
    model,
    processor,
    base_model_path: str,
    split_sha256: str,
    train_boards: List[str],
    train_sft_sha256: str,
    step: int,
    training_cfg: Dict[str, Any],
    trainer_state: Optional[Dict[str, Any]] = None
):
    """Save LoRA weights, processor, and independent adapter_metadata.json for automated benchmark provenance verification."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    if trainer_state is not None:
        torch.save(trainer_state, os.path.join(save_dir, "trainer_state.pt"))

    metadata = {
        **training_cfg["provenance"],
        "format_version": "1.0",
        "base_model_path": base_model_path,
        "base_model_version": "Qwen/Qwen2-VL-2B-Instruct",
        "split_sha256": split_sha256,
        "train_boards": train_boards,
        "train_sft_sha256": train_sft_sha256,
        "coordinate_protocol": "x-first: (xmin, ymin), (xmax, ymax), normalized [0, 1000]",
        "classes": get_canonical_categories(),
        "training_step": step,
        "training_config": training_cfg,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    meta_path = os.path.join(save_dir, "adapter_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved independent adapter provenance metadata to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for Qwen2-VL-2B.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to base model weights.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to train_sft.json.")
    parser.add_argument("--val-data-file", type=str, default=None, help="Path to val_sft.json for evaluation.")
    parser.add_argument("--splits-path", type=str, default=None, help="Optional path to splits.json.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save checkpoints.")
    parser.add_argument("--config", type=str, default="configs/lora.yaml", help="Path to lora.yaml.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max training steps.")
    parser.add_argument("--eval-steps", type=int, default=25, help="Evaluation interval.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-qlora", action="store_true", help="Use 4-bit BitsAndBytes QLoRA to save VRAM.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Directory containing checkpoint to resume from (e.g. checkpoint-100).")
    parser.add_argument("--stop-after-step", type=int, default=None, help="Bounded smoke stop, without changing scheduler horizon")
    args = parser.parse_args()
    effective_stop = min(args.max_steps, args.stop_after_step or args.max_steps)

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise FileExistsError("Use a fresh output directory; resume weights/state may be read from a different existing run")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Provenance Data Verification at Startup
    import hashlib
    with open(args.data_file, "rb") as f:
        train_sft_sha256 = hashlib.sha256(f.read()).hexdigest()

    with open(args.data_file, "r", encoding="utf-8") as f:
        train_raw_data = json.load(f)
    train_boards = sorted(list(set(item["board_folder"] for item in train_raw_data if "board_folder" in item)))

    # Locate canonical splits.json to capture split_sha256
    splits_candidates = [
        args.splits_path,
        r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations_v2_canonical\splits.json",
        "annotations_v2_canonical/splits.json"
    ]
    splits_file = next((p for p in splits_candidates if p and os.path.exists(p)), None)
    if splits_file:
        with open(splits_file, "rb") as f:
            split_sha256 = hashlib.sha256(f.read()).hexdigest()
        print(f"Verified splits provenance: {splits_file} (SHA-256: {split_sha256[:16]}...)")
    else:
        raise FileNotFoundError("Canonical splits.json is required for training")

    if args.splits_path and not os.path.isfile(args.splits_path):
        raise FileNotFoundError(args.splits_path)
    train_boards = validate_samples(train_raw_data, "train", splits_file)
    if args.val_data_file:
        with open(args.val_data_file, encoding="utf-8") as stream:
            validate_samples(json.load(stream), "val", splits_file)
    cfg["training_run"] = {"seed": cfg["training"].get("seed", 42), "scheduler_horizon": args.max_steps, "use_qlora": args.use_qlora, "max_sequence_length": cfg["data"].get("max_length"), "actual_data_file": args.data_file}
    cfg["provenance"] = {"train_sft_path": os.path.abspath(args.data_file),
        "base_model_sha256": model_identity(args.model_path),
        "git_commit": code_identity(), "source_tree_sha256": source_identity(), "dependencies": dependencies()}
    from pathlib import Path
    annotation_file = Path(splits_file).with_name('coco_annotations.json')
    cfg['provenance'].update({
        'annotations_sha256': hashlib.sha256(annotation_file.read_bytes()).hexdigest() if annotation_file.is_file() else None,
        'splits_path': os.path.abspath(splits_file),
        'val_sft_path': os.path.abspath(args.val_data_file) if args.val_data_file else None,
        'val_sft_sha256': hashlib.sha256(Path(args.val_data_file).read_bytes()).hexdigest() if args.val_data_file else None,
    })
    if args.resume_from_checkpoint:
        previous = verify_adapter(args.resume_from_checkpoint, splits_file, args.model_path)
        if previous['train_sft_sha256'] != train_sft_sha256:
            raise ValueError('Resume requires the exact original training SFT')
        for key in ('lora', 'training', 'data', 'training_run'):
            if previous['training_config'].get(key) != cfg.get(key):
                raise ValueError(f'Resume config mismatch: {key}')

    print(f"=== Qwen2-VL-2B LoRA SFT Training ===")
    print(f"Base Model:    {args.model_path}")
    print(f"Train Dataset: {args.data_file} (SHA-256: {train_sft_sha256[:16]}...)")
    print(f"Train Boards:  {train_boards}")
    print(f"Val Dataset:   {args.val_data_file}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Max Steps:     {args.max_steps}")
    print(f"Eval Steps:    {args.eval_steps}")
    print(f"Device:        {args.device}")
    print(f"Use QLoRA 4bit: {args.use_qlora}")
    print(f"Resume Checkpoint: {args.resume_from_checkpoint}")

    # Set all random seeds BEFORE LoRA model initialization to guarantee deterministic parameter initialization
    seed = int(cfg["training"].get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # Load Base Model
    print("\nLoading base model and processor...")
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            quantization_config=bnb_cfg,
            torch_dtype=torch.bfloat16,
            device_map={"": args.device},
            local_files_only=True
        )
        model = prepare_model_for_kbit_training(model)
        print("Loaded Qwen2-VL with 4-bit NormalFloat4 (QLoRA).")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map=args.device if args.device == "cuda" else None
        )
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)

    # Enable Gradient Checkpointing
    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    # Configure or resume LoRA
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise FileNotFoundError(f"Checkpoint directory not found: {args.resume_from_checkpoint}")
        print(f"Resuming LoRA model weights from: {args.resume_from_checkpoint}")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
        from safetensors.torch import load_file
        from peft import get_peft_model_state_dict
        saved_adapter = load_file(os.path.join(args.resume_from_checkpoint, 'adapter_model.safetensors'))
        loaded_adapter = get_peft_model_state_dict(model)
        if saved_adapter.keys() != loaded_adapter.keys() or not all(torch.equal(value.cpu(),loaded_adapter[key].detach().cpu()) for key,value in saved_adapter.items()):
            raise ValueError('Reloaded adapter tensors do not match saved checkpoint')
        print('Resume adapter tensor readback: exact match.')
    else:
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
    model.config.use_cache = False
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _, all_params = model.get_nb_trainable_parameters()  # Counts logical 4-bit parameters, not packed storage elements.
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
    seed = int(cfg["training"].get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = PCBComponentSFTDataset(
        data_file=args.data_file,
        processor=processor,
        max_pixels=cfg["data"].get("image_max_pixels", 301056),
        max_objects=cfg["data"].get("max_objects", 60),
        max_response_len=cfg["data"].get("max_response_len", 3500),
        max_sequence_length=cfg["data"].get("max_length")
    )
    batch_size = cfg["training"].get("per_device_train_batch_size", 1)
    batch_sampler = ResumableBatchSampler(
        dataset_len=len(dataset),
        batch_size=batch_size,
        seed=seed,
        epoch=0,
        start_batch_idx=0
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=__import__("functools").partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id)
    )

    val_loader = None
    if args.val_data_file and os.path.exists(args.val_data_file):
        val_dataset = PCBComponentSFTDataset(
            data_file=args.val_data_file,
            processor=processor,
            max_pixels=cfg["data"].get("image_max_pixels", 301056),
            max_objects=cfg["data"].get("max_objects", 60),
            max_response_len=cfg["data"].get("max_response_len", 3500),
            max_sequence_length=cfg["data"].get("max_length")
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=__import__("functools").partial(collate_fn, pad_token_id=processor.tokenizer.pad_token_id)
        )
        print(f"Validation dataset loaded: {len(val_dataset)} samples.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
    best_adapter_source = None
    total_batches_processed = 0
    start_epoch = 0
    start_batch_in_epoch = 0
    resumed_permutation = None
    trainer_state_payload = None

    if args.resume_from_checkpoint:
        state_file = os.path.join(args.resume_from_checkpoint, "trainer_state.pt")
        if not os.path.isfile(state_file):
            raise FileNotFoundError("Exact resume requires trainer_state.pt")
        if os.path.exists(state_file):
            print(f"Loading trainer state from {state_file}...")
            state = torch.load(state_file, map_location="cpu", weights_only=False)
            trainer_state_payload = state
            required_state = ('step','optimizer_state_dict','scheduler_state_dict','rng_state','epoch','batch_in_epoch','epoch_indices','total_batches_processed','python_rng_state','numpy_rng_state')
            if any(k not in state for k in required_state):
                raise ValueError('Incomplete resume state; refusing partial recovery')
            step = state.get("step", 0)
            best_val_loss = state.get("best_val_loss", float("inf"))
            best_adapter_source = state.get("best_adapter_source")
            total_batches_processed = state.get("total_batches_processed", step * grad_accum_steps)
            start_epoch = state.get("epoch", 0)
            start_batch_in_epoch = state.get("batch_in_epoch", 0)
            resumed_permutation = state.get("epoch_indices", None)
            if "optimizer_state_dict" in state and state["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(state["optimizer_state_dict"])
                print(f"Restored optimizer state (resuming from step {step}).")
            if "scheduler_state_dict" in state and state["scheduler_state_dict"] is not None and scheduler is not None:
                scheduler.load_state_dict(state["scheduler_state_dict"])
                print("Restored learning rate scheduler state.")
            if "rng_state" in state and state["rng_state"] is not None:
                torch.set_rng_state(state["rng_state"].cpu())
                if "python_rng_state" in state: random.setstate(state["python_rng_state"])
                if "numpy_rng_state" in state: np.random.set_state(state["numpy_rng_state"])
                print("Restored PyTorch CPU RNG state.")
            if "cuda_rng_state" in state and state["cuda_rng_state"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(state["cuda_rng_state"].cpu())
                print("Restored CUDA RNG state.")

            ckpt_parent = os.path.dirname(args.resume_from_checkpoint)
            history_file = os.path.join(ckpt_parent, "training_history.json")
            history_jsonl = os.path.join(ckpt_parent, "training_history.jsonl")
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        loaded_history = json.load(f)
                    if not isinstance(loaded_history, list):
                        raise ValueError("training_history.json must contain a list")
                    history = [h for h in loaded_history if h.get("step", 0) <= step]
                except Exception as exc:
                    print(f"Warning: failed to load {history_file} ({exc}); starting with empty history.")
                    history = []
            elif os.path.exists(history_jsonl):
                try:
                    with open(history_jsonl, "r", encoding="utf-8") as f:
                        loaded_history = [json.loads(line) for line in f if line.strip()]
                    history = [h for h in loaded_history if h.get("step", 0) <= step]
                except Exception as exc:
                    print(f"Warning: failed to load {history_jsonl} ({exc}); starting with empty history.")
                    history = []
        else:
            print(f"Warning: {state_file} not found. Resumed LoRA weights only; optimizer starts fresh.")

    start_step = step
    t_start = time.time()
    print(f"\nStarting training loop at step {step}/{args.max_steps} (resuming epoch {start_epoch}, batch {start_batch_in_epoch}, batches done: {total_batches_processed})...")
    optimizer.zero_grad()

    global_batch_counter = total_batches_processed
    accum_counter = 0
    update_started = time.time()
    window_indices = []
    window_lengths = []

    for epoch in range(start_epoch, 100):
        if step >= effective_stop:
            break
        if epoch == start_epoch and start_batch_in_epoch > 0:
            batch_sampler.set_epoch(epoch, start_batch_idx=start_batch_in_epoch, permutation=resumed_permutation)
        else:
            batch_sampler.set_epoch(epoch, start_batch_idx=0, permutation=None)

        current_batch_in_epoch = batch_sampler.start_batch_idx

        for batch in dataloader:
            global_batch_counter += 1
            accum_counter += 1
            current_batch_in_epoch += 1
            window_indices.extend(batch_sampler.get_all_batches()[current_batch_in_epoch - 1])
            window_lengths.append(int(batch["input_ids"].shape[1]))
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / grad_accum_steps
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; refusing training update")
            loss.backward()

            total_loss += loss.item() * grad_accum_steps

            all_batches_count = len(batch_sampler.get_all_batches())
            is_window_end = (accum_counter % grad_accum_steps == 0)
            is_epoch_end = (current_batch_in_epoch == all_batches_count)

            if is_window_end or is_epoch_end:
                # If window was incomplete (remainder batches < grad_accum_steps at epoch end), rescale gradients
                if accum_counter < grad_accum_steps:
                    scale_factor = float(grad_accum_steps) / float(accum_counter)
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad.mul_(scale_factor)
                    step_loss = total_loss / accum_counter
                else:
                    step_loss = total_loss / grad_accum_steps

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

                # Clip once below and log the true pre-clip norm.
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg["training"].get("max_grad_norm", 1.0)))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                scheduler.step()

                # Verify weight update across steps (ensures non-zero lr has taken effect)
                curr_lr = scheduler.get_last_lr()[0] if scheduler else cfg["training"]["learning_rate"]
                if not verification_passed.get("lora_weight_updated", False):
                    if not torch.equal(sample_lora_param.data[:5], initial_lora_val):
                        verification_passed["lora_weight_updated"] = True
                        print(f"[Training Self-Check] Verified LoRA weight updated at step {step + 1} (lr={curr_lr:.8f})")
                base_weight_untouched = torch.equal(sample_base_param.data[:5], initial_base_val)
                verification_passed["base_weight_untouched"] = base_weight_untouched

                optimizer.zero_grad()
                step += 1

                total_loss = 0.0
                accum_counter = 0
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                record = {
                    "optimizer_steps_this_process": step - (start_step if "start_step" in locals() else 0),
                    "step": step,
                    "sample_ids": [dataset.data[i]["id"] for i in window_indices],
                    "physical_board_ids": [dataset.data[i].get("physical_board_id", "unknown") for i in window_indices],
                    "sequence_token_lengths": list(window_lengths),
                    "microbatches_this_update": len(window_lengths),
                    "peak_reserved_vram_mb": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
                    "gradient_norm": float(grad_norm),
                    "step_wall_s": round(time.time() - update_started, 3),
                    "sequence_tokens": int(batch["input_ids"].shape[1]),
                    "loss": round(step_loss, 4),
                    "lr": round(curr_lr, 7),
                    "peak_vram_mb": round(peak_vram, 2),
                    "time_elapsed_s": round(time.time() - t_start, 2)
                }

                # Compute next epoch & batch offsets for exact continuation
                if current_batch_in_epoch >= all_batches_count:
                    save_epoch = epoch + 1
                    save_batch_offset = 0
                    save_perm = None
                else:
                    save_epoch = epoch
                    save_batch_offset = current_batch_in_epoch
                    save_perm = batch_sampler.get_permutation()

                trainer_state_payload = {
                    "step": step,
                    "epoch": save_epoch,
                    "batch_in_epoch": save_batch_offset,
                    "epoch_indices": save_perm,
                    "total_batches_processed": global_batch_counter,
                    "best_val_loss": best_val_loss,
                    "best_adapter_source": best_adapter_source,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "rng_state": torch.get_rng_state(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                }

                # Validation evaluation across full validation set
                if val_loader is not None and (step % args.eval_steps == 0 or step >= effective_stop):
                    model.eval()
                    val_loss_acc = 0.0
                    val_batches = 0
                    max_val_batches = cfg.get("training", {}).get("max_val_batches", None)
                    with torch.no_grad():
                        for v_i, v_b in enumerate(val_loader):
                            if max_val_batches is not None and v_i >= max_val_batches:
                                break
                            v_b = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in v_b.items()}
                            val_loss_acc += model(**v_b).loss.item()
                            val_batches += 1

                    if val_batches > 0:
                        mean_val_loss = val_loss_acc / val_batches
                        model.train()
                        record["val_loss"] = round(mean_val_loss, 4)
                        record["val_batches"] = val_batches
                        print(f"Step [{step:03d}/{args.max_steps}] Loss: {step_loss:.4f} | Full Val Loss ({val_batches} batches): {mean_val_loss:.4f} | Peak VRAM: {peak_vram:.2f} MB", flush=True)

                        if mean_val_loss < best_val_loss:
                            best_val_loss = mean_val_loss
                            best_dir = os.path.join(args.output_dir, "best_adapter")
                            best_adapter_source = os.path.abspath(best_dir)
                            trainer_state_payload["best_val_loss"] = best_val_loss
                            trainer_state_payload["best_adapter_source"] = best_adapter_source
                            save_adapter_with_provenance_metadata(
                                best_dir, model, processor, args.model_path, split_sha256,
                                train_boards, train_sft_sha256, step, cfg, trainer_state_payload
                            )
                            print(f"*** New best LoRA adapter saved with Val Loss {mean_val_loss:.4f} ({val_batches} batches) -> {best_dir} ***", flush=True)
                    else:
                        model.train()
                        print(f"Step [{step:03d}/{args.max_steps}] Warning: Validation set has 0 batches. Skipping best model update.", flush=True)
                else:
                    print(f"Step [{step:03d}/{args.max_steps}] Loss: {step_loss:.4f} | Peak VRAM: {peak_vram:.2f} MB | Elapsed: {record['time_elapsed_s']}s", flush=True)

                history.append(record)
                with open(os.path.join(args.output_dir, "training_history.jsonl"), "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                window_indices = []
                window_lengths = []
                update_started = time.time()

                if step % cfg["training"].get("save_steps", 10) == 0 or step >= effective_stop:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
                    save_adapter_with_provenance_metadata(
                        save_dir, model, processor, args.model_path, split_sha256,
                        train_boards, train_sft_sha256, step, cfg, trainer_state_payload
                    )
                    print(f"Saved LoRA checkpoint and resume state to {save_dir} (epoch {save_epoch}, batch_offset {save_batch_offset})", flush=True)

                if step >= effective_stop:
                    break

        if step >= effective_stop:
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
    save_adapter_with_provenance_metadata(
        final_dir, model, processor, args.model_path, split_sha256,
        train_boards, train_sft_sha256, step, cfg, trainer_state_payload
    )
    print(f"\nFinal adapter weights and metadata saved to {final_dir}")

    # A final checkpoint is never promoted to best without a measured improvement.
    # A resumed run can legitimately have no new local best_adapter directory.

    log_file = os.path.join(args.output_dir, "training_history.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history written to {log_file}")

    manifest_file = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "stage": 14,
            "total_steps": step,
            "epochs": round(global_batch_counter / len(dataset), 2),
            "actual_sample_ids": [r["id"] for r in dataset.data],
            "processed_sample_ids": [sid for row in history for sid in row.get("sample_ids", [])],
            "validation_sample_ids": [r["id"] for r in val_dataset.data] if val_loader else [],
            "use_qlora": args.use_qlora,
            "scheduler_horizon": args.max_steps,
            "start_step": start_step,
            "best_adapter_source": best_adapter_source,
            "best_val_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else None,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
            "total_time_s": round(time.time() - t_start, 2),
            "config": cfg
        }, f, indent=2)
    print(f"Run manifest written to {manifest_file}")


if __name__ == "__main__":
    main()
