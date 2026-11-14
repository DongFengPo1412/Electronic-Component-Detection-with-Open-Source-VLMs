"""Importing the project must never forge shared dependency metadata."""
import os
import subprocess
import sys
import unittest
from pathlib import Path


class DependencyMetadataTests(unittest.TestCase):
    def test_package_import_preserves_real_version_function(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            "import sys, importlib.metadata as m; "
            "sys.path.insert(0, sys.argv[1]); "
            "fn = m.version; "
            "before = m.version('huggingface-hub'); "
            "import pcb_vlm; "
            "assert m.version is fn, 'Project replaced dependency version API'; "
            "assert m.version('huggingface-hub') == before; "
            "import huggingface_hub as h; "
            "assert h.__version__ == before, (h.__version__, before)"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(root / "src")],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
