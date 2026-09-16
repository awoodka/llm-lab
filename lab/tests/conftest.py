import pytest

from lab.schema import (
    BaseModelInfo,
    ConfigInfo,
    EngineBuild,
    HardwareSnapshot,
    Metric,
    ModelInfo,
    RunBundle,
    RunRecord,
)

HARDWARE = HardwareSnapshot(
    gpu_name="RTX 3090", gpu_vram_mb=24576, driver="595.99.02", cuda="13.0", power_limit_w=250,
    pcie="Gen3 x16", cpu_model="Ryzen 5 3600", cpu_threads_visible=6, ram_mb=24576, kernel="7.0", os="Debian 13",
)

QWEN_LADDER = [
    Metric(key="pp_tps", method="llama-bench", n_prompt=512, value=949, unit="t/s"),
    Metric(key="tg_tps", method="llama-bench", n_gen=128, depth=0, value=31.8, unit="t/s"),
    Metric(key="tg_tps", method="llama-bench", n_gen=128, depth=4096, value=30.9, unit="t/s"),
    Metric(key="tg_tps", method="llama-bench", n_gen=128, depth=16384, value=28.2, unit="t/s"),
]


@pytest.fixture
def speed_bundle():
    """A published speed run for a 64k config, with the plan's Qwen depth ladder."""

    def make(*, metrics: list[Metric] | None = None, ctx: int = 65536, **run_kwargs) -> RunBundle:
        run = dict(
            id="run-1", kind="speed", status="ok", started_at="2026-09-15T01:00:00Z",
            lab_version="0.1.0", cli_args="lab bench", published_at="2026-09-15T02:00:00Z",
        )
        run.update(run_kwargs)
        return RunBundle(
            base_model=BaseModelInfo(slug="qwen3.8-27b", name="Qwen3.8 27B"),
            model=ModelInfo(slug="qwen3.8-27b-q4_k_m-gguf", name="Qwen3.8 27B Q4_K_M", base="qwen3.8-27b",
                            engine="llama.cpp", format="gguf", quant="Q4_K_M"),
            config=ConfigInfo(slug="64k-q8kv", name="64k q8_0 KV", config_hash="hash-a",
                              params={"ctx": ctx}, launch_command="llama-server -m x.gguf"),
            hardware=HARDWARE,
            engine_build=EngineBuild(engine="llama.cpp", version="10883", commit_sha="91f6a6cf3"),
            run=RunRecord(**run),
            metrics=QWEN_LADDER if metrics is None else metrics,
        )

    return make
