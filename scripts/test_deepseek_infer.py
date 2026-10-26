import sys
import os
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from pcb_vlm.models.deepseek_adapter import DeepSeekAdapter

test_image = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\datasets\raw\pcb_wacv_2019\pcb_wacv_2019\STK644\STK644.jpg"
model_path = r"C:\Users\Liu\.codex\cache\tasks\pcb-vlm-antigravity-20261024\models\models\deepseek-ai--deepseek-vl-1.3b-chat\snapshots\master"

adapter = DeepSeekAdapter(model_id=model_path, device="cuda")
adapter.load_model()
res = adapter.predict_image(test_image, max_new_tokens=128)
print("DeepSeek-VL predict_image completed!")
print(f"Latency: {res['latency_ms']} ms")
print(f"Num detected: {res['num_detected']}")
print(f"Raw output snippet: {repr(res['raw_output'][:200])}")
