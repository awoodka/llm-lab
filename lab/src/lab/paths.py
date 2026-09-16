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

LLAMA_CPP = Path(os.environ.get("LAB_LLAMA_CPP", Path.home() / "llama.cpp"))
