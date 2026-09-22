import json
import socket
import time
from contextlib import contextmanager

import pytest

from lab import paths, publish
from lab.gpu import hosted


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SETTINGS", tmp_path / "settings.yaml")
    monkeypatch.setattr(hosted, "PAUSED_BY", tmp_path / "paused_by.json")
    monkeypatch.setattr(paths, "HOSTED_JSON", tmp_path / "hosted.json")
    (tmp_path / "hosted.json").write_text(json.dumps({"ref": "qwen/64k", "port": 8080}))
    monkeypatch.delenv("LAB_WEB_URL", raising=False)
    monkeypatch.delenv("LAB_INGEST_TOKEN", raising=False)


def test_put_pause_without_settings_is_silent(capsys):
    assert publish.put_pause(True, "bench", "m/c") is False
    assert capsys.readouterr().err == ""


def test_put_pause_fails_fast_when_the_site_refuses(monkeypatch, capsys):
    monkeypatch.setenv("LAB_WEB_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LAB_INGEST_TOKEN", "t")
    assert publish.put_pause(True, "bench", "m/c") is False
    assert "site status not updated" in capsys.readouterr().err


def test_put_pause_gives_up_quickly_when_the_site_hangs(monkeypatch, capsys):
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()  # connections land in the backlog and never get an answer
        monkeypatch.setenv("LAB_WEB_URL", f"http://127.0.0.1:{server.getsockname()[1]}")
        monkeypatch.setenv("LAB_INGEST_TOKEN", "t")
        started = time.monotonic()
        assert publish.put_pause(True, "bench", "m/c") is False
        assert time.monotonic() - started < 2.5
    assert "site status not updated" in capsys.readouterr().err


def _fake_gpu(monkeypatch, calls, *, active=True, report_error=False):
    @contextmanager
    def fake_lock(purpose, wait=False):
        calls.append("lock")
        yield

    def fake_put_pause(paused, reason=None, ref=None):
        calls.append(("pause", paused, reason, ref))
        if report_error:
            raise RuntimeError("site exploded")
        return True

    monkeypatch.setattr(hosted, "gpu_lock", fake_lock)
    monkeypatch.setattr(hosted, "check_power_limit", lambda: calls.append("power"))
    monkeypatch.setattr(hosted, "hosted_active", lambda: active)
    monkeypatch.setattr(hosted, "stop_hosted", lambda: calls.append("stop"))
    monkeypatch.setattr(hosted, "start_hosted", lambda wait_health=True, timeout_s=600: calls.append("start") or True)
    monkeypatch.setattr(hosted, "wait_gpu_idle", lambda: calls.append("idle"))
    monkeypatch.setattr(hosted, "wait_cool", lambda: calls.append("cool"))
    monkeypatch.setattr(publish, "put_pause", fake_put_pause)


def test_exclusive_gpu_reports_the_pause_before_waiting_and_the_resume_after(monkeypatch):
    calls = []
    _fake_gpu(monkeypatch, calls)
    with hosted.exclusive_gpu("bench speed gemma/32k-q8kv"):
        calls.append("body")
    assert calls == [
        "lock", "power", "stop", ("pause", True, "bench", "gemma/32k-q8kv"), "idle", "cool", "body", "start",
        ("pause", True, "starting", "qwen/64k"),
    ]


def test_keep_paused_reports_the_pause_over_without_starting(monkeypatch):
    calls = []
    _fake_gpu(monkeypatch, calls)
    with hosted.exclusive_gpu("bench speed gemma/32k-q8kv", keep_paused=True):
        pass
    assert "start" not in calls and calls[-1] == ("pause", False, None, None)


def test_power_limit_refusal_leaves_the_hosted_model_alone(monkeypatch):
    from lab.gpu.power import PowerLimitTooHigh

    calls = []
    _fake_gpu(monkeypatch, calls)

    def too_high():
        calls.append("power")
        raise PowerLimitTooHigh("GPU power limit is 280 W")

    monkeypatch.setattr(hosted, "check_power_limit", too_high)
    with pytest.raises(PowerLimitTooHigh):
        with hosted.exclusive_gpu("bench speed gemma/32k-q8kv"):
            calls.append("body")
    assert calls == ["lock", "power"]


@pytest.mark.parametrize(("limit", "env", "ok"), [(250.0, None, True), (280.0, None, False), (280.0, "280", True)])
def test_check_power_limit(monkeypatch, limit, env, ok):
    from lab.gpu import power

    monkeypatch.setattr(power, "enforced_power_limit_w", lambda gpu_index=0: limit)
    if env:
        monkeypatch.setenv("LAB_MAX_POWER_W", env)
    else:
        monkeypatch.delenv("LAB_MAX_POWER_W", raising=False)
    if ok:
        assert power.check_power_limit() == limit
    else:
        with pytest.raises(power.PowerLimitTooHigh, match="280 W"):
            power.check_power_limit()


def test_exclusive_gpu_still_runs_and_resumes_when_reporting_fails(monkeypatch):
    calls = []
    _fake_gpu(monkeypatch, calls, report_error=True)
    with hosted.exclusive_gpu("serve gemma/32k-q8kv", cool=False):
        calls.append("body")
    assert calls.index("stop") < calls.index("body") < calls.index("start")


def test_nothing_is_reported_when_no_model_was_hosted(monkeypatch):
    calls = []
    _fake_gpu(monkeypatch, calls, active=False)
    with hosted.exclusive_gpu("bench speed gemma/32k-q8kv"):
        pass
    assert not [c for c in calls if isinstance(c, tuple)]


@pytest.mark.parametrize("payload", ['{"cmd": "llama-server -m /mnt/models/hf/x.gguf"}', '{"cwd": "/home/alex/llama.cpp"}'])
def test_publish_rejects_local_paths(payload):
    with pytest.raises(publish.PublishError, match="local path"):
        publish.check_no_local_paths(json.loads(payload))


def test_publish_allows_file_names_and_urls():
    publish.check_no_local_paths(
        {"cmd": "llama-server -m gemma-3-4b-it-Q4_K_M.gguf", "src": "https://huggingface.co/home/x", "repo": "ggml-org/gemma-3-4b-it-GGUF"}
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"env": {"MODEL_HEALTH_URL": "https://ai.example-tailnet.ts.net:8443/health"}},
        {"cmd": "curl --resolve ai:8443:100.64.0.1 https://ai:8443/v1/models"},
        {"note": "served from 100.127" + ".255.254"},  # split, so the repo hygiene scan passes this file
    ],
)
def test_publish_rejects_tailnet_addresses(payload):
    with pytest.raises(publish.PublishError, match="private network address"):
        publish.check_no_private_addresses(payload)


def test_publish_allows_public_and_loopback_addresses():
    publish.check_no_private_addresses(
        {"cmd": "vllm serve --host 127.0.0.1 --port 18020", "dns": "1.1.1.1", "not_cgnat": ["100.63.0.1", "100.128.0.1"],
         "site": "https://localinference.example.com", "version": "0.1.0+47747f3"}
    )
