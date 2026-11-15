"""FICS-PCB VLM sequential multi-model inference & GPU gate modules."""

from pcb_vlm.inference.fics_matrix import (
    DEFAULT_RUNTIME_CONFIG_EXAMPLE,
    build_formal_calls_for_model,
    determine_matrix_exit_code,
    ensure_safe_output_dir,
    load_runtime_config,
    run_sequential_matrix,
)
from pcb_vlm.inference.gpu_gate import probe_gpu_gate, sha256_file

__all__ = [
    "DEFAULT_RUNTIME_CONFIG_EXAMPLE",
    "build_formal_calls_for_model",
    "determine_matrix_exit_code",
    "ensure_safe_output_dir",
    "load_runtime_config",
    "probe_gpu_gate",
    "run_sequential_matrix",
    "sha256_file",
]

