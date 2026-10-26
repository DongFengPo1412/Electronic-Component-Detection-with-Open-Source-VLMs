import sys
import os
from PIL import Image

sys.path.insert(0, "src")
from deepseek_vl.models import VLChatProcessor

model_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master"
test_img_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\sample_patch.jpg"

print("Loading VLChatProcessor...")
processor = VLChatProcessor.from_pretrained(model_path)
print("Processor loaded successfully!")

if os.path.exists(test_img_path):
    img = Image.open(test_img_path).convert("RGB")
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>Locate the resistors in this PCB image.",
            "images": [img]
        },
        {"role": "Assistant", "content": ""}
    ]
    prepare_inputs = processor(
        conversations=conversation,
        images=[img],
        force_batchify=True
    )
    print("Prepare inputs keys:", list(prepare_inputs.keys()))
    print("Input IDs shape:", prepare_inputs.input_ids.shape)
    print("Pixel values shape:", prepare_inputs.pixel_values.shape)
    print("Images seq mask shape:", prepare_inputs.images_seq_mask.shape)
    print("ALL PROCESSOR CHECKS PASSED!")
