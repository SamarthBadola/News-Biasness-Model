"""Environment and provenance details written next to every run."""
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def git_commit():
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def environment():
    import numpy
    import sklearn
    import torch
    import transformers
    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": numpy.__version__, "scikit_learn": sklearn.__version__, "torch_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(), "code_commit": git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
