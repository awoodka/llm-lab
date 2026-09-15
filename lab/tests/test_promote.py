import json
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

from lab import catalog, cli, paths, publish
from lab.catalog import ConfigSpec, ModelSpec, Params, Source
from lab.gpu import hosted
from lab.gpu.power import PowerLimitTooHigh

OLD, NEW = "old-q4_k_m-gguf/c", "new-q4_k_m-gguf/c"


@pytest.fixture
def calls(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CATALOG", tmp_path / "catalog")
    monkeypatch.setattr(paths, "HOSTED_SH", tmp_path / "hosted.sh")
    monkeypatch.setattr(paths, "HOSTED_JSON", tmp_path / "hosted.json")
    monkeypatch.setattr(hosted, "PAUSED_BY", tmp_path / "paused_by.json")
    for ref in (OLD, NEW):
        slug = ref.split("/")[0]
        catalog.save_model(ModelSpec(slug=slug, name=slug, base="b", quant="Q4_K_M", source=Source(path=f"/models/{slug}.gguf")))
        catalog.save_config(slug, ConfigSpec(slug="c", name="c", params=Params()))
    paths.HOSTED_SH.write_text("#!/bin/sh\nexec old\n")
    paths.HOSTED_JSON.write_text(json.dumps({"ref": OLD, "config_hash": "old", "port": 8080}))

    calls = []

    @contextmanager
    def fake_lock(purpose, wait=False):
        yield

    monkeypatch.setattr(cli, "gpu_lock", fake_lock)
    monkeypatch.setattr(cli, "_install_unit", lambda: None)
    monkeypatch.setattr(cli, "check_power_limit", lambda: 250.0)
    monkeypatch.setattr(hosted, "hosted_active", lambda: True)
    monkeypatch.setattr(hosted, "stop_hosted", lambda: calls.append("stop"))
    monkeypatch.setattr(hosted, "wait_gpu_idle", lambda: None)
    monkeypatch.setattr(publish, "put_hosted", lambda config_hash: calls.append("put_hosted"))
    return calls


def _hosted_ref() -> str:
    return json.loads(paths.HOSTED_JSON.read_text())["ref"]


def _start_with(monkeypatch, calls, *healthy):
    answers = iter(healthy)
    monkeypatch.setattr(hosted, "start_hosted", lambda wait_health=True: calls.append(("start", _hosted_ref())) or next(answers))


def test_promote_switches_the_hosted_model_and_tells_the_site(calls, monkeypatch):
    _start_with(monkeypatch, calls, True)
    result = CliRunner().invoke(cli.app, ["promote", NEW])
    assert result.exit_code == 0, result.output
    assert calls == ["stop", ("start", NEW), "put_hosted"]
    assert "/models/new-q4_k_m-gguf.gguf" in paths.HOSTED_SH.read_text()


def test_failed_promote_rolls_back_to_the_previous_model(calls, monkeypatch):
    _start_with(monkeypatch, calls, False, True)
    result = CliRunner().invoke(cli.app, ["promote", NEW])
    assert result.exit_code == 1
    assert f"rolled back to {OLD}, which is healthy again" in result.output
    assert calls == ["stop", ("start", NEW), "stop", ("start", OLD)]
    assert _hosted_ref() == OLD and paths.HOSTED_SH.read_text() == "#!/bin/sh\nexec old\n"


def test_failed_re_promote_of_the_same_model_just_stops(calls, monkeypatch):
    _start_with(monkeypatch, calls, False)
    result = CliRunner().invoke(cli.app, ["promote", OLD])
    assert result.exit_code == 1
    assert "no previous model to roll back to" in result.output
    assert calls == ["stop", ("start", OLD), "stop"]


def test_promote_refuses_above_the_power_cap_without_touching_anything(calls, monkeypatch):
    def too_high():
        raise PowerLimitTooHigh("GPU power limit is 280 W, above the 250 W maximum")

    monkeypatch.setattr(cli, "check_power_limit", too_high)
    result = CliRunner().invoke(cli.app, ["promote", NEW])
    assert result.exit_code == 1 and "280 W" in result.output
    assert calls == [] and _hosted_ref() == OLD
