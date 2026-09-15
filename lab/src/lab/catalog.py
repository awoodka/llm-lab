"""Model/config catalog (git-tracked YAML).

catalog/bases/<base>.yaml
catalog/models/<model>/model.yaml
catalog/models/<model>/configs/<config>.yaml

A config is referenced as "<model>/<config>".
"""

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lab import paths
from lab.schema import BaseModelInfo, ModelInfo, canonical_hash


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    repo: str | None = None
    file: str | None = None  # file within the repo (GGUF) or None for a whole-repo snapshot (EXL3)
    revision: str | None = None
    path: str | None = None  # explicit local path; overrides HF resolution


class BaseSpec(Strict):
    slug: str
    name: str
    family: str | None = None
    params_b: float | None = None
    active_params_b: float | None = None
    arch: Literal["dense", "moe"] = "dense"
    hf_repo: str | None = None


class ModelSpec(Strict):
    slug: str
    name: str
    base: str
    engine: Literal["llama.cpp", "exllamav3"] = "llama.cpp"
    format: str = "gguf"
    quant: str
    bpw: float | None = None
    source: Source
    notes: str | None = None


class SpecDecode(Strict):
    type: str = "draft-simple"
    model: str  # model slug of the draft model
    gpu_layers: int = 999
    n_max: int = 8
    cache_type_k: str | None = None
    cache_type_v: str | None = None


class Params(Strict):
    """Engine-neutral knobs. Every knob is rendered explicitly so engine defaults never leak in."""

    ctx: int = 8192
    gpu_layers: int = 999
    flash_attn: Literal["on", "off", "auto"] = "on"
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    batch: int = 2048
    ubatch: int = 512
    threads: int = 6
    n_cpu_moe: int = 0
    parallel: int = 1
    load_mode: str = "mmap"
    cache_ram_mib: int = 0
    spec: SpecDecode | None = None
    reasoning_format: str | None = None
    reasoning_budget: int | None = None
    extra_args: list[str] = Field(default_factory=list)


class BenchSpec(Strict):
    n_prompt: int = 512
    n_gen: int = 128
    depths: list[int] = Field(default_factory=lambda: [0, 4096, 16384])
    reps: int = 5
    delay_s: int = 3
    http_prompt_lengths: list[int] = Field(default_factory=lambda: [512, 4096])


class ConfigSpec(Strict):
    slug: str
    name: str
    notes: str | None = None
    params: Params = Field(default_factory=Params)
    bench: BenchSpec = Field(default_factory=BenchSpec)
    eval_overrides: dict[str, Any] = Field(default_factory=dict)


SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", s.lower()).strip("-")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _dump_yaml(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(model.model_dump(mode="json", exclude_none=True), sort_keys=False))


# -- bases -------------------------------------------------------------------
def base_path(slug: str) -> Path:
    return paths.CATALOG / "bases" / f"{slug}.yaml"


def load_base(slug: str) -> BaseSpec:
    p = base_path(slug)
    if not p.is_file():
        raise KeyError(f"unknown base model {slug!r} (expected {p})")
    return BaseSpec.model_validate(_load_yaml(p))


def save_base(spec: BaseSpec) -> Path:
    p = base_path(spec.slug)
    _dump_yaml(p, spec)
    return p


# -- models ------------------------------------------------------------------
def model_dir(slug: str) -> Path:
    return paths.CATALOG / "models" / slug


def load_model(slug: str) -> ModelSpec:
    p = model_dir(slug) / "model.yaml"
    if not p.is_file():
        raise KeyError(f"unknown model {slug!r} (expected {p})")
    return ModelSpec.model_validate(_load_yaml(p))


def save_model(spec: ModelSpec) -> Path:
    p = model_dir(spec.slug) / "model.yaml"
    _dump_yaml(p, spec)
    return p


def list_models() -> list[ModelSpec]:
    root = paths.CATALOG / "models"
    if not root.is_dir():
        return []
    return [load_model(p.name) for p in sorted(root.iterdir()) if (p / "model.yaml").is_file()]


def resolve_model_path(spec: ModelSpec, download: bool = False) -> Path:
    """Local path to the weights (GGUF file or EXL3 directory)."""
    src = spec.source
    if src.path:
        return Path(src.path)
    if not src.repo:
        raise ValueError(f"model {spec.slug} has neither source.path nor source.repo")
    from huggingface_hub import hf_hub_download, snapshot_download

    kw = dict(repo_id=src.repo, revision=src.revision, local_files_only=not download)
    if src.file:
        return Path(hf_hub_download(filename=src.file, **kw))
    return Path(snapshot_download(**kw))


# -- configs -----------------------------------------------------------------
def config_path(model_slug: str, config_slug: str) -> Path:
    return model_dir(model_slug) / "configs" / f"{config_slug}.yaml"


def split_ref(ref: str) -> tuple[str, str]:
    if "/" not in ref:
        raise ValueError(f"config reference must be <model>/<config>, got {ref!r}")
    model_slug, config_slug = ref.split("/", 1)
    return model_slug, config_slug


def load_config(ref: str) -> tuple[ModelSpec, ConfigSpec]:
    model_slug, config_slug = split_ref(ref)
    model = load_model(model_slug)
    p = config_path(model_slug, config_slug)
    if not p.is_file():
        raise KeyError(f"unknown config {ref!r} (expected {p})")
    return model, ConfigSpec.model_validate(_load_yaml(p))


def save_config(model_slug: str, cfg: ConfigSpec) -> Path:
    p = config_path(model_slug, cfg.slug)
    _dump_yaml(p, cfg)
    return p


def list_configs(model_slug: str) -> list[ConfigSpec]:
    d = model_dir(model_slug) / "configs"
    if not d.is_dir():
        return []
    return [ConfigSpec.model_validate(_load_yaml(p)) for p in sorted(d.glob("*.yaml"))]


def config_hash(model: ModelSpec, cfg: ConfigSpec) -> str:
    """Identity of what was measured: weights (slug + revision) + every knob."""
    return canonical_hash(
        {"model": model.slug, "revision": model.source.revision, "params": cfg.params.model_dump(mode="json")}
    )


# -- conversions to the publish contract --------------------------------------
def base_info(spec: BaseSpec) -> BaseModelInfo:
    return BaseModelInfo(**spec.model_dump())


def model_info(spec: ModelSpec, file_size_bytes: int | None) -> ModelInfo:
    return ModelInfo(
        slug=spec.slug,
        name=spec.name,
        base=spec.base,
        engine=spec.engine,
        format=spec.format,
        quant=spec.quant,
        bpw=spec.bpw,
        file_size_bytes=file_size_bytes,
        source_repo=spec.source.repo,
        source_file=spec.source.file,
        source_revision=spec.source.revision,
        notes=spec.notes,
    )


def weights_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
