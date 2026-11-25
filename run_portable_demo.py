"""Portable one-command launcher for FICS-PCB final delivery candidate package (works from any cwd)."""
from __future__ import annotations

import pathlib
import sys

SW_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SW_DIR / "src"))
sys.path.insert(0, str(SW_DIR / "scripts"))

import fics_final_demo  # noqa: E402


def main() -> int:
    default_cfg = SW_DIR / "configs" / "fics_final_demo.portable.json"
    argv = list(sys.argv[1:])
    if "--config" not in argv and "--workspace-root" not in argv:
        argv = ["--config", str(default_cfg), *argv]
    return fics_final_demo.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
