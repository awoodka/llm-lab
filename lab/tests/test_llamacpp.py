from pathlib import Path

import pytest

from lab import catalog, paths
from lab.bench.native_llamacpp import rows_to_metrics
from lab.catalog import BenchSpec, ConfigSpec, ModelSpec, Params, Source, SpecDecode
from lab.engines.llamacpp import LlamaCpp


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CATALOG", tmp_path / "catalog")


def _model(slug="m-q4_k_m-gguf", path="/models/m.gguf") -> ModelSpec:
    return ModelSpec(slug=slug, name="M", base="m", quant="Q4_K_M", source=Source(path=path))


def test_server_argv_sets_every_knob_explicitly():
    eng = LlamaCpp(root=Path("/llama"))
    cfg = ConfigSpec(slug="32k", name="32k", params=Params(ctx=32768, cache_type_k="q8_0", cache_type_v="q8_0"))
    argv = eng.server_argv(_model(), cfg, host="127.0.0.1", port=8081)
    assert argv == [
        "/llama/build/bin/llama-server", "-m", "/models/m.gguf",
        "--host", "127.0.0.1", "--port", "8081",
        "-a", "m-q4_k_m-gguf/32k",
        "--fit", "off", "-c", "32768", "-np", "1", "--cache-ram", "0",
        "-ngl", "999", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "2048", "-ub", "512",
        "-t", "6", "-ncmoe", "0", "-lm", "mmap",
        "--metrics",
    ]  # fmt: skip


def test_speculative_flags_and_display_command():
    draft = _model(slug="d-q8_0-gguf", path="/models/draft.gguf")
    catalog.save_model(draft)
    cfg = ConfigSpec(slug="spec", name="spec", params=Params(spec=SpecDecode(model=draft.slug, n_max=12, cache_type_k="q8_0")))
    eng = LlamaCpp(root=Path("/llama"))
    argv = eng.server_argv(_model(), cfg, host="127.0.0.1", port=8081)
    i = argv.index("--spec-type")
    assert argv[i : i + 10] == ["--spec-type", "draft-simple", "-md", "/models/draft.gguf", "-ngld", "999", "--spec-draft-n-max", "12", "-ctkd", "q8_0"]
    shown = eng.display_command(_model(), cfg)
    assert shown.startswith("llama-server -m m.gguf ") and "-md draft.gguf" in shown


def test_bench_depths_are_clipped_to_ctx():
    cfg = ConfigSpec(slug="c", name="c", params=Params(ctx=8192), bench=BenchSpec(depths=[0, 4096, 8000, 16384]))
    argv = LlamaCpp(root=Path("/llama")).bench_argv(Path("/models/m.gguf"), cfg)
    assert argv[argv.index("-d") + 1] == "0,4096"
    assert argv[argv.index("-o") + 1] == "jsonl"


def test_rows_to_metrics_power_windows():
    rows = [
        {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 4000.0, "stddev_ts": 10.0, "samples_ts": [3990.0, 4010.0]},
        {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 100.0, "stddev_ts": 1.0, "samples_ts": [99.0, 101.0]},
    ]
    samples = [{"t": t / 10, "power_w": 250.0 if t < 20 else 200.0, "util": 95.0} for t in range(40)]
    m = {(x.key, x.n_gen): x for x in rows_to_metrics(rows, [2.0, 4.0], samples)}
    assert m[("pp_tps", 0)].value == 4000.0 and m[("pp_tps", 0)].n == 2
    assert m[("tg_tps", 128)].stddev == 1.0
    assert m[("gpu_w_avg", 128)].value == 200.0
    assert m[("tokens_per_joule", 128)].value == 0.5
