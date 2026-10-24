"""
Hardware and Environment Diagnostics Script for Electronic-Component-Detection-with-Open-Source-VLMs
Checks: Python, CUDA, PyTorch, GPU VRAM, System RAM, Disk Space, Key Libraries.
"""

import sys
import os
import shutil
import platform
import psutil

def run_diagnostics():
    print("=" * 60)
    print("  PCB VLM Environment Diagnostics (Stage 06)")
    print("=" * 60)

    # 1. Platform & Python
    print(f"\n[1] Platform & Python:")
    print(f"  OS Platform: {platform.platform()}")
    print(f"  Python Version: {sys.version.split()[0]}")
    print(f"  Python Executable: {sys.executable}")

    # 2. System RAM & Disk
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    avail_gb = mem.available / (1024 ** 3)
    print(f"\n[2] System Resources:")
    print(f"  RAM: Total={total_gb:.1f} GB, Available={avail_gb:.1f} GB")

    total_d, used_d, free_d = shutil.disk_usage("C:\\")
    print(f"  Disk C: Total={total_d / (1024 ** 3):.1f} GB, Free={free_d / (1024 ** 3):.1f} GB")

    # 3. PyTorch & CUDA
    print(f"\n[3] Deep Learning Hardware (PyTorch & CUDA):")
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_avail}")
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU Device: {device_name}")
            print(f"  Total VRAM: {vram_total:.2f} GB")

            # Tensor matmul smoke test on CUDA
            x = torch.randn(256, 256, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print(f"  CUDA MatMul Test: PASSED (allocated={torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB)")
        else:
            print("  WARNING: CUDA is NOT available. Running on CPU mode.")
    except ImportError:
        print("  ERROR: PyTorch is not installed in current environment.")

    # 4. Multimodal & Vision Libraries
    print(f"\n[4] Core Multimodal & CV Libraries:")
    libs = [
        "torchvision",
        "transformers",
        "accelerate",
        "peft",
        "cv2",
        "PIL",
        "yaml",
        "pycocotools",
        "shapely"
    ]
    for lib in libs:
        try:
            m = __import__(lib)
            ver = getattr(m, "__version__", "available")
            print(f"  {lib:15s}: {ver}")
        except ImportError:
            print(f"  {lib:15s}: NOT INSTALLED")

    print("\n" + "=" * 60)
    print("  Diagnostics Completed.")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
