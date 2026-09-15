import pytest

from lab import catalog, paths
from lab.catalog import ConfigSpec, ModelSpec, Params, Source
from lab.gpu.lock import LockBusy, gpu_lock, is_locked


@pytest.fixture(autouse=True)
def tmp_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(paths, "STATE", tmp_path / "state")
    monkeypatch.setattr(paths, "GPU_LOCK", tmp_path / "state" / "gpu.lock")
    monkeypatch.setattr(paths, "GPU_LOCK_HOLDER", tmp_path / "state" / "gpu.lock.json")


def _model() -> ModelSpec:
    return ModelSpec(slug="m-q4_k_m-gguf", name="M Q4", base="m", quant="Q4_K_M", source=Source(repo="r/m", file="m.gguf", revision="abc"))


def test_config_hash_tracks_params_and_revision():
    m = _model()
    c1 = ConfigSpec(slug="a", name="a")
    c2 = ConfigSpec(slug="b", name="b")  # same params, different slug/name → same measurement identity
    c3 = ConfigSpec(slug="a", name="a", params=Params(ctx=32768))
    assert catalog.config_hash(m, c1) == catalog.config_hash(m, c2)
    assert catalog.config_hash(m, c1) != catalog.config_hash(m, c3)
    m2 = m.model_copy(update={"source": Source(repo="r/m", file="m.gguf", revision="def")})
    assert catalog.config_hash(m, c1) != catalog.config_hash(m2, c1)


def test_config_roundtrip():
    m = _model()
    catalog.save_model(m)
    catalog.save_config(m.slug, ConfigSpec(slug="32k", name="32k", params=Params(ctx=32768, cache_type_k="q8_0")))
    model, cfg = catalog.load_config(f"{m.slug}/32k")
    assert model == m and cfg.params.ctx == 32768 and cfg.params.cache_type_k == "q8_0"


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        Params.model_validate({"ctx_size": 4096})


def test_gpu_lock_is_exclusive():
    assert not is_locked()
    with gpu_lock("test"):
        assert is_locked()
        with pytest.raises(LockBusy):
            with gpu_lock("second"):
                pass
    assert not is_locked()
