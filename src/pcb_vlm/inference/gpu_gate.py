"""Dual-gate free VRAM probe before spawning any isolated VLM worker subprocess."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict


def sha256_file(path: pathlib.Path) -> str:
    """Return lowercase hex SHA-256 digest of a file using 1 MiB streaming reads."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_gpu_gate(min_smi_mib: float = 5500.0, min_torch_mib: float = 6144.0) -> Dict[str, Any]:
    """Run dual GPU VRAM availability checks via `nvidia-smi` and `torch.cuda.mem_get_info`."""
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.perf_counter()
    out: Dict[str, Any] = {
        "started_at": started_at,
        "min_smi_free_mib": float(min_smi_mib),
        "min_torch_free_mib": float(min_torch_mib),
    }
    smi_free = 0.0
    torch_free = 0.0

    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        parts = [x.strip() for x in smi.splitlines()[0].split(",")]
        smi_free = float(parts[3])
        out["nvidia_smi_gpu"] = {
            "name": parts[0],
            "memory_total_mib": float(parts[1]),
            "memory_used_mib": float(parts[2]),
            "memory_free_mib": smi_free,
            "utilization_pct": float(parts[4]),
        }
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip().splitlines()
        out["nvidia_smi_compute_apps"] = [a.strip() for a in apps if a.strip()]
    except Exception as exc:
        out["nvidia_smi_error"] = repr(exc)

    try:
        import torch

        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            torch_free = round(free_b / (1024 * 1024), 2)
            out["torch_cuda_mem_get_info"] = {
                "free_mib": torch_free,
                "total_mib": round(total_b / (1024 * 1024), 2),
                "device_name": torch.cuda.get_device_name(0),
            }
    except Exception as exc:
        out["torch_error"] = repr(exc)

    out["passed"] = bool(
        smi_free >= float(min_smi_mib)
        and torch_free >= float(min_torch_mib)
        and "nvidia_smi_error" not in out
        and "torch_error" not in out
    )
    out["wall_seconds"] = round(time.perf_counter() - t0, 3)
    return out


def main() -> int:
    res = probe_gpu_gate()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

