import sys
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor

model_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master"
print("Loading VLChatProcessor...")
processor = VLChatProcessor.from_pretrained(model_path)
print("Loading MultiModalityCausalLM...")
model = MultiModalityCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model.eval()
print("DeepSeek-VL MultiModalityCausalLM successfully loaded into GPU!")
vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
print(f"Peak VRAM: {vram:.2f} MB")
