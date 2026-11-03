"""PCB Vision-Language Model Detection Framework."""

import importlib.metadata

# Defensive compatibility patch for environments where huggingface-hub is >= 2.0
try:
    _orig_v = importlib.metadata.version
    importlib.metadata.version = lambda pkg: "0.26.2" if pkg in ("huggingface-hub", "huggingface_hub") else _orig_v(pkg)
except Exception:
    pass

__version__ = "0.1.0"
