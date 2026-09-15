"""Run bundle = the local run.json AND the ingest contract sent to the web app.

Changing a field here means bumping SCHEMA_VERSION and updating web/src/contract.ts.
"""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


def canonical_hash(obj: Any) -> str:
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BaseModelInfo(BaseModel):
    slug: str
    name: str
    family: str | None = None
    params_b: float | None = None
    active_params_b: float | None = None
    arch: Literal["dense", "moe"] = "dense"
    hf_repo: str | None = None


class ModelInfo(BaseModel):
    slug: str
    name: str
    base: str
    engine: Literal["llama.cpp", "exllamav3"]
    format: str
    quant: str
    bpw: float | None = None
    file_size_bytes: int | None = None
    source_repo: str | None = None
    source_file: str | None = None
    source_revision: str | None = None
    notes: str | None = None


class ConfigInfo(BaseModel):
    slug: str
    name: str
    config_hash: str
    params: dict[str, Any]
    launch_command: str
    engine_files: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None


class HardwareSnapshot(BaseModel):
    gpu_name: str
    gpu_vram_mb: int
    driver: str
    cuda: str
    power_limit_w: float
    pcie: str
    cpu_model: str
    cpu_threads_visible: int
    ram_mb: int
    kernel: str
    os: str


class EngineBuild(BaseModel):
    engine: str
    version: str | None = None
    commit_sha: str | None = None
    build_flags: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Metric(BaseModel):
    key: str  # pp_tps tg_tps ttft_ms vram_peak_mb ram_peak_mb gpu_w_avg tokens_per_joule ...
    method: str  # llama-bench | http | telemetry | llama-perplexity
    n_prompt: int = 0
    n_gen: int = 0
    depth: int = 0
    concurrency: int = 1
    value: float
    stddev: float | None = None
    n: int | None = None
    unit: str
    samples: list[float] | None = None


class QualityRef(BaseModel):
    base: str
    ref_label: str
    dataset: str
    ctx: int
    chunks: int
    logits_sha: str | None = None


class EvalResult(BaseModel):
    task: str
    metric: str
    filter: str = "none"
    value: float
    stderr: float | None = None
    n_samples: int | None = None
    limit_n: int | None = None
    lm_eval_version: str | None = None
    gen_kwargs: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    id: str
    kind: Literal["speed", "quality", "evals"]
    tier: str | None = None
    status: Literal["running", "ok", "failed"] = "running"
    started_at: str
    duration_s: float | None = None
    lab_version: str
    cli_args: str
    throttled: bool = False
    notes: str | None = None
    error: str | None = None
    telemetry: list[dict[str, float]] = Field(default_factory=list)  # 1 Hz downsampled
    raw: dict[str, Any] = Field(default_factory=dict)
    published_at: str | None = None


class RunBundle(BaseModel):
    schema_version: int = SCHEMA_VERSION
    base_model: BaseModelInfo
    model: ModelInfo
    config: ConfigInfo
    hardware: HardwareSnapshot
    engine_build: EngineBuild
    run: RunRecord
    metrics: list[Metric] = Field(default_factory=list)
    quality_ref: QualityRef | None = None
    eval_results: list[EvalResult] = Field(default_factory=list)

    def content_sha(self) -> str:
        """Hash of the publishable content (excludes local-only publish bookkeeping)."""
        data = self.model_dump(mode="json")
        data["run"].pop("published_at", None)
        return canonical_hash(data)
