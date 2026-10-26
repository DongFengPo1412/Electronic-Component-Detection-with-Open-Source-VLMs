"""
Protocol Freeze and Audit Script (Stage 15).
Verifies:
1. Zero test leakage (test boards strictly disjoint from train and val).
2. Protocol parameter locking (categories, prompt, tiling, matching thresholds).
3. Computes and exports final_protocol_hash.
4. Generates frozen_protocol.yaml and test execution plan.
"""

import os
import sys
import json
import hashlib
import yaml

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    splits_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\splits.json"
    coco_file = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\annotations\coco_annotations.json"
    lora_adapter_dir = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\runs\stage14_lora_training\best_adapter"

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # 1. Audit Data Leakage
    train_boards = set(splits["splits"]["train"]["boards"])
    val_boards = set(splits["splits"]["val"]["boards"])
    test_boards = set(splits["splits"]["test"]["boards"])

    leakage_train_test = train_boards.intersection(test_boards)
    leakage_val_test = val_boards.intersection(test_boards)

    assert len(leakage_train_test) == 0, f"Data leakage between train and test: {leakage_train_test}"
    assert len(leakage_val_test) == 0, f"Data leakage between val and test: {leakage_val_test}"
    print(f"Data leakage audit PASSED: Train={len(train_boards)}, Val={len(val_boards)}, Test={len(test_boards)} boards completely disjoint.")

    # 2. Frozen Protocol Configuration
    protocol = {
        "version": "1.0-frozen",
        "task": "electronic_component_detection",
        "categories": [
            "button", "capacitor", "clock", "connector", "diode",
            "ic", "inductor", "led", "resistor", "switch", "transistor"
        ],
        "prompt": "Detect the button, capacitor, clock, connector, diode, ic, inductor, led, resistor, switch, transistor in this image. Return the bounding box coordinates [ymin, xmin, ymax, xmax] for each detected object.",
        "generation": {
            "do_sample": False,
            "max_new_tokens": 1024,
            "torch_dtype": "bfloat16",
            "device": "cuda"
        },
        "tiling": {
            "tile_size": 1024,
            "overlap_ratio": 0.20,
            "nms_iou_threshold": 0.50
        },
        "evaluation": {
            "iou_thresholds": [0.50, 0.25, 0.10],
            "metrics": ["precision", "recall", "f1", "support", "latency_ms", "peak_vram_mb"]
        },
        "models": {
            "qwen_base": "C:\\Users\\Liu\\.codex\\cache\\tasks\\pcb-vlm-antigravity-20261024\\models\\models\\Qwen--Qwen2-VL-2B-Instruct\\snapshots\\master",
            "qwen_lora": lora_adapter_dir
        },
        "test_split": {
            "image_ids": splits["splits"]["test"]["image_ids"],
            "boards": splits["splits"]["test"]["boards"],
            "total_images": len(splits["splits"]["test"]["image_ids"])
        }
    }

    # 3. Compute Protocol SHA-256 Hash
    protocol_str = json.dumps(protocol, sort_keys=True)
    protocol_hash = hashlib.sha256(protocol_str.encode("utf-8")).hexdigest()
    protocol["final_protocol_hash"] = protocol_hash

    # Save to configs/frozen_protocol.yaml
    out_yaml = os.path.join(root_dir, "configs", "frozen_protocol.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(protocol, f, sort_keys=False, allow_unicode=True)
    print(f"Frozen protocol saved to {out_yaml}")
    print(f"Final Protocol SHA-256 Hash: {protocol_hash}")

    # Also save to cache
    cache_out = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\protocol\frozen_protocol.json"
    os.makedirs(os.path.dirname(cache_out), exist_ok=True)
    with open(cache_out, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2, ensure_ascii=False)
    print(f"Cache copy saved to {cache_out}")

if __name__ == "__main__":
    main()
