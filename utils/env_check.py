import importlib
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
