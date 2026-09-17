"""Filesystem layout. Override with LAB_ROOT / LAB_MODELS env vars."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("LAB_ROOT", Path(__file__).resolve().parents[2]))
CATALOG = ROOT / "catalog"
SUBSETS = ROOT / "subsets"  # committed eval subsets: which task ids a benchmark is pinned to
RUNS = ROOT / "runs"
STATE = ROOT / "state"
GPU_LOCK = STATE / "gpu.lock"
GPU_LOCK_HOLDER = STATE / "gpu.lock.json"
HOSTED_SH = STATE / "hosted.sh"
HOSTED_JSON = STATE / "hosted.json"
SETTINGS = ROOT / "settings.yaml"


def _default_models_dir() -> Path:
    nvme = Path("/mnt/models")
    return nvme if nvme.is_dir() else Path.home() / ".cache" / "lab-models"


MODELS = Path(os.environ.get("LAB_MODELS", _default_models_dir()))
REFS = MODELS / "refs"
DATASETS = MODELS / "datasets"

# flashinfer JIT-compiles kernels on the first boot of a config and finds the toolkit through CUDA_HOME, or
# else through `which nvcc`. /usr/bin/nvcc is a symlink and nvcc takes its root from the path it was called by,
# so through that symlink the root is /usr and cuda_runtime.h goes missing: always hand the engine the real root.
CUDA_HOME = Path(os.environ.get("LAB_CUDA_HOME", "/usr/local/cuda"))

LLAMA_CPP = Path(os.environ.get("LAB_LLAMA_CPP", Path.home() / "llama.cpp"))
# The syv-ai vLLM stack: a pinned clone with its own venv and models/. An upgrade is a new clone and a new symlink.
QWEN_SERVING = Path(os.environ.get("LAB_QWEN_SERVING", Path.home() / "qwen-serving"))
