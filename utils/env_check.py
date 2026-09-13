import importlib
import os
import subprocess
import sys


def _ensure_package(package, import_name=None):
    try:
        importlib.import_module(import_name or package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_dependencies():
    """Installs packages that aren't part of the base env but are required
    by the training pipeline (metrics, model-complexity profiling, I/O)."""
    _ensure_package("medpy")
    _ensure_package("ptflops")
    _ensure_package("fvcore")
    _ensure_package("scikit-image", "skimage")
    _ensure_package("nibabel")


def configure_cuda_allocator(setting):
    """Set PYTORCH_CUDA_ALLOC_CONF before the CUDA caching allocator starts.

    MUST be called before the first CUDA allocation — the allocator reads this
    variable once, at initialisation, and silently ignores later changes. That
    is why it lives here and is called at the top of main() rather than next to
    the training loop that needs it.

    An explicit setting already in the environment wins, so
    `PYTORCH_CUDA_ALLOC_CONF=... python train.py` still overrides config.py.
    """
    if not setting:
        return None
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if existing:
        print(f"PYTORCH_CUDA_ALLOC_CONF already set to {existing!r}; leaving it alone")
        return existing
    if "torch.cuda" in sys.modules and sys.modules["torch.cuda"].is_initialized():
        print("CUDA already initialised — PYTORCH_CUDA_ALLOC_CONF will NOT take effect")
        return None
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = setting
    print(f"PYTORCH_CUDA_ALLOC_CONF={setting}")
    return setting
