import json
import subprocess
from pathlib import Path

import httpx
import pytest

from lab import catalog, paths, publish
from lab.catalog import ConfigSpec, ModelSpec, Params, Source, VllmParams
from lab.engines.base import Warmup, get_engine
from lab.engines.llamacpp import LlamaCppDialect
from lab.engines.vllm import CheckoutMismatch, Vllm, VllmDialect

SHA = "bae2023ffc98753d337d2d2041784a277599a4c4"
REF = "qwen3.8-27b-w4a16-autoround-fast/64k-dflash2"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A stand-in launcher repo: one commit, a patch series with a superseded patch, a venv with a tiny vllm package."""
    root = tmp_path / "qwen-serving"
    site = root / "venv/lib/python3.12/site-packages/vllm"
    site.mkdir(parents=True)
    (site / "a.py").write_text("x = 1\nnew line that the first patch added here\n")
    (site / "b.py").write_text("y = 2\nrewritten by the second patch, long enough to count\n")
    patches = root / "patches"
    patches.mkdir()
    (patches / "series").write_text("# order\nfirst.patch\nold.patch\ndflash2-backport.patch\nnew.patch\n")
    (patches / "first.patch").write_text(
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n x = 1\n+new line that the first patch added here\n"
    )
    (patches / "old.patch").write_text(
        "--- a/b.py\n+++ b/b.py\n@@ -1 +1,2 @@\n y = 2\n+the old patch's line, which the new one replaced\n"
    )
    (patches / "new.patch").write_text(
        "Supersedes: old.patch\n--- a/b.py\n+++ b/b.py\n@@ -1 +1,2 @@\n y = 2\n+rewritten by the second patch, long enough to count\n"
    )
    (patches / "_check_applied.py").write_text("import sys\nsys.exit(1)\n")
    (root / "venv/bin").mkdir(parents=True)
    (root / "venv/bin/python").symlink_to(Path(subprocess.run(["which", "python3"], capture_output=True, text=True).stdout.strip()))
    (root / "single-user").mkdir()
    (root / "single-user/start_qwen.sh").write_text("#!/bin/bash\n")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", "patches", "single-user")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "pin")
    return root


def _model() -> ModelSpec:
    return ModelSpec(
        slug="qwen3.8-27b-w4a16-autoround-fast", name="Q", base="qwen3.8-27b", engine="vllm", format="safetensors",
        quant="W4A16", source=Source(repo="dbirks/x", revision="r1", path="models/fast"), artifacts={"draft": "syvai/d@r2"},
    )


def _cfg(commit: str, **params) -> ConfigSpec:
    return ConfigSpec(slug="64k-dflash2", name="c", params=VllmParams(launcher="single-user/start_qwen.sh", launcher_commit=commit,
                                                                      draft="models/draft", **params))


def test_server_argv_renders_every_knob_on_localhost_without_a_key(checkout, monkeypatch):
    monkeypatch.setattr(paths, "QWEN_SERVING", checkout)
    eng = Vllm(root=checkout)
    argv = eng.server_argv(_model(), _cfg(_git(checkout, "rev-parse", "HEAD")), host="127.0.0.1", port=8080)
    assert argv[:3] == ["env", "-u", "VLLM_API_KEY"]
    assert argv[-2:] == ["/bin/bash", str(checkout / "single-user/start_qwen.sh")]
    env = dict(a.split("=", 1) for a in argv[3:-2])
    assert env == {
        "CUDA_HOME": str(paths.CUDA_HOME),
        "HOST": "127.0.0.1", "PORT": "8080", "MODEL": str(checkout / "models/fast"), "SPEC": "dflash2", "CTX": "fast",
        "MAX_LEN": "65536", "PREFIX_CACHE": "1", "KV_MEM": "5583457484", "MAX_SEQS": "8", "GPU_UTIL": "0.93",
        "DFLASH_TOKENS": "7", "LOOKUP": "1", "TOOLS": "1", "VISION": "0", "DRAFT": str(checkout / "models/draft"),
        "EXTRA_ARGS": "--served-model-name qwen3.8-27b-w4a16-autoround-fast/64k-dflash2 --load-format auto",
    }  # fmt: skip


def test_the_launcher_is_told_the_real_toolkit_root(monkeypatch):
    """flashinfer JIT-compiles at boot: without CUDA_HOME it follows `which nvcc` to /usr, which has no headers."""
    monkeypatch.setattr(paths, "CUDA_HOME", Path("/opt/cuda-13"))
    env = Vllm(root=Path("/q")).launch_env(_model(), _cfg(SHA), host="127.0.0.1", port=1, weights="w", draft=None)
    assert env["CUDA_HOME"] == "/opt/cuda-13"
    with pytest.raises(ValueError, match="may not set CUDA_HOME"):
        Vllm(root=Path("/q")).launch_env(_model(), _cfg(SHA, env={"CUDA_HOME": "/usr"}), host="127.0.0.1", port=1, weights="w", draft=None)


def test_prefix_cache_off_is_really_off():
    env = Vllm(root=Path("/q")).launch_env(_model(), _cfg(SHA, prefix_cache=False), host="127.0.0.1", port=1, weights="w", draft=None)
    assert env["PREFIX_CACHE"] == "0" and env["EXTRA_ARGS"].endswith("--no-enable-prefix-caching")


@pytest.mark.parametrize(("params", "match"), [
    (dict(env={"HOST": "0.0.0.0"}), "may not set HOST"),
    (dict(env={"VLLM_API_KEY": "x"}), "may not set VLLM_API_KEY"),
    (dict(extra_args=["--chat-template", "a b"]), "splits EXTRA_ARGS"),
])  # fmt: skip
def test_launch_env_refuses_what_the_launcher_would_mangle_or_expose(params, match):
    with pytest.raises(ValueError, match=match):
        Vllm(root=Path("/q")).launch_env(_model(), _cfg(SHA, **params), host="127.0.0.1", port=1, weights="w", draft=None)


def test_refuses_an_unpinned_checkout_or_a_key_file(checkout):
    eng = Vllm(root=checkout)
    with pytest.raises(CheckoutMismatch, match="pins bae2023ffc98"):
        eng.server_argv(_model(), _cfg(SHA), host="127.0.0.1", port=8080)
    (checkout / "api_key.txt").write_text("secret")
    with pytest.raises(CheckoutMismatch, match="api_key.txt"):
        eng.check_checkout(_cfg(_git(checkout, "rev-parse", "HEAD")).params)


def test_display_command_has_no_paths_and_passes_publish_checks():
    shown = Vllm(root=Path("/home/alex/qwen-serving")).display_command(_model(), _cfg(SHA))
    assert shown.startswith(f"CUDA_HOME={paths.CUDA_HOME} HOST=127.0.0.1 PORT=8080 MODEL=models/fast SPEC=dflash2 ")
    assert shown.endswith(" single-user/start_qwen.sh") and "DRAFT=models/draft" in shown
    publish.check_no_local_paths({"cmd": shown})
    publish.check_no_secrets({"cmd": shown})


@pytest.mark.parametrize("text", ["VLLM_API_KEY=abc", "--api-key abc", "Authorization: Bearer abcdef123456", "hf_" + "a" * 30])
def test_publish_refuses_credentials(text):
    with pytest.raises(publish.PublishError, match="credential"):
        publish.check_no_secrets({"cmd": text})


def test_publish_allows_words_that_merely_mention_keys():
    publish.check_no_secrets({"doc": "get_weather(api_key: str) — pass your API key; VLLM_API_KEY is unset"})


def test_params_union_picks_the_right_type_both_ways():
    llama = ConfigSpec.model_validate({"slug": "a", "name": "a", "params": {"ctx": 32768}})
    vllm = ConfigSpec.model_validate({"slug": "b", "name": "b", "params": {"launcher": "s.sh", "launcher_commit": SHA}})
    assert type(llama.params) is Params and type(vllm.params) is VllmParams
    assert type(ConfigSpec(slug="c", name="c").params) is Params
    with pytest.raises(ValueError):
        ConfigSpec.model_validate({"slug": "d", "name": "d", "params": {"launcher": "s.sh"}})


def test_load_config_refuses_an_engine_params_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CATALOG", tmp_path)
    catalog.save_model(_model())
    catalog.save_config(_model().slug, ConfigSpec(slug="wrong", name="w", params=Params()))
    with pytest.raises(ValueError, match="runs on vllm"):
        catalog.load_config(f"{_model().slug}/wrong")


def test_llamacpp_config_hash_is_unchanged_and_vllm_hashes_its_artifacts():
    assert catalog.config_hash(*catalog.load_config("qwen3.8-27b-q4_k_m-gguf/64k-q8kv")) == (
        "3c7251bf69b58420f30b9f7aeb21cb19753a94e7216986aa0e12684a4912b76a"
    )
    model, cfg = catalog.load_config(REF)
    assert type(cfg.params) is VllmParams and model.engine == "vllm"
    other = model.model_copy(update={"artifacts": {**model.artifacts, "draft": "syvai/other@r"}})
    assert catalog.config_hash(model, cfg) != catalog.config_hash(other, cfg)


def test_patch_status_follows_supersedes_and_skips_retired(checkout):
    applied, missing = Vllm(root=checkout).patch_status()
    assert (applied, missing) == (3, [])
    (checkout / "venv/lib/python3.12/site-packages/vllm/a.py").write_text("x = 1\n")
    assert Vllm(root=checkout).patch_status() == (2, ["first.patch"])


def test_build_info_records_the_checkout(checkout):
    b = Vllm(root=checkout).build_info()
    assert b.engine == "vllm" and b.commit_sha == _git(checkout, "rev-parse", "--short=9", "HEAD")
    assert b.extra["launcher_commit"] == _git(checkout, "rev-parse", "HEAD") and b.extra["dirty"] is False
    assert b.extra["patches_applied"] == 3 and b.extra["patches_missing"] == []
    assert len(b.extra["patch_series_sha256"]) == 64


def test_engines_carry_boot_timeouts_and_cache_busting_extras():
    llama, vllm = get_engine("llama.cpp"), get_engine("vllm")
    assert llama.boot_timeout_s < vllm.boot_timeout_s == 1500
    assert llama.request_extras() == {"cache_prompt": False}
    assert vllm.request_extras()["cache_salt"] != vllm.request_extras()["cache_salt"]


# -- the eval dialect --------------------------------------------------------------
def _dialect(tmp_path: Path, **params) -> VllmDialect:
    return VllmDialect(served_model=REF, weights_dir=tmp_path, params=_cfg(SHA, **params).params)


def test_the_prompt_is_counted_in_one_call_that_carries_tools_and_template_kwargs(tmp_path):
    """vLLM renders and counts in the same call, so what is counted is what the completion will see."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"count": 304, "max_model_len": 65536, "tokens": [1, 2]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        n = _dialect(tmp_path).count_prompt(client, "http://x", {
            "model": "ignored", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 99,
        })
    assert n == 304
    sent = seen[0]
    assert sent["model"] == REF and sent["add_generation_prompt"] is True
    assert sent["tools"] and sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert "add_special_tokens" not in sent, "the chat path's default is what both sides must agree on"
    assert "max_tokens" not in sent


def test_each_task_attempt_gets_its_own_cache_scope(tmp_path):
    """vLLM's prefix cache is global: without a per-task salt one task is charged for another's prefill."""
    d = _dialect(tmp_path)
    assert d.cache_scope("aime#0") == d.cache_scope("aime#0")
    assert d.cache_scope("aime#0") != d.cache_scope("gpqa#0"), "two tasks must not share a prefix"
    assert d.cache_scope("aime#0") != d.cache_scope("aime#1"), "a retry starts as cold as its budget claims"
    assert set(d.cache_scope("aime#0")) == {"cache_salt"}


def test_props_report_the_sampling_without_ever_naming_a_local_path(tmp_path):
    """/v1/models answers with the weights' path on disk, and a published run names files, never paths."""
    (tmp_path / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_k": 20, "top_p": 0.95, "bos_token_id": 1}))
    card = {"id": REF, "max_model_len": 65536, "root": "/home/alex/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound-fast"}

    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [card]}))) as client:
        props = _dialect(tmp_path).props(client, "http://x", Warmup(seconds=1.0, replies=2, tool_calls=True))

    assert props["sampling"] == {"temperature": 1.0, "top_k": 20, "top_p": 0.95}, "only what steers generation"
    assert props["n_ctx"] == 65536 and props["served_model"] == REF
    assert props["chat_template_caps"] == {"supports_tools": True, "supports_tool_calls": True}
    publish.check_no_local_paths({"run": {"raw": {"props": props}}})


def test_a_server_that_cannot_call_tools_is_reported_as_such(tmp_path):
    """A wrong tool-call parser answers with prose, which a tool benchmark would score as a bad model."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "I would call get_weather"}, "finish_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        warm = _dialect(tmp_path).warmup(client, "http://x", REF, tools=True)
    assert warm.tool_calls is False and warm.replies == 2

    def calls(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"id": "1"}]}, "finish_reason": "tool_calls"}]})

    with httpx.Client(transport=httpx.MockTransport(calls)) as client:
        assert _dialect(tmp_path).warmup(client, "http://x", REF, tools=True).tool_calls is True
        assert _dialect(tmp_path).warmup(client, "http://x", REF, tools=False).tool_calls is None


@pytest.mark.parametrize(("engine_name", "parallel", "expected"), [
    ("llama.cpp", 1, {}),
    ("llama.cpp", 4, {"parallel": 4, "ctx": 4 * 32768}),
    ("vllm", 1, {}),
    ("vllm", 4, {"parallel": 4}),
])  # fmt: skip
def test_serving_several_requests_at_once_relaunches_only_the_engine_that_needs_it(engine_name, parallel, expected, tmp_path):
    """llama.cpp splits one KV cache into slots; vLLM already has them, so a parallel eval is the same server."""
    if engine_name == "llama.cpp":
        cfg = ConfigSpec(slug="c", name="C", params=Params(ctx=32768))
        dialect = LlamaCppDialect(served_model="m")
    else:
        cfg = _cfg(SHA, ctx=32768, max_seqs=8)
        dialect = _dialect(tmp_path, ctx=32768, max_seqs=8)

    launch, overrides = dialect.eval_launch(cfg, parallel)
    assert overrides == expected
    assert cfg.params.ctx == 32768, "the catalog's config is never mutated"
    if engine_name == "vllm":
        assert launch is cfg, "nothing to relaunch: the eval runs on the config that serves chat"
    elif parallel > 1:
        assert launch.params.parallel == parallel and launch.params.ctx == 32768 * parallel


def test_more_requests_than_the_config_serves_is_refused(tmp_path):
    with pytest.raises(ValueError, match="max_seqs"):
        _dialect(tmp_path, max_seqs=8).eval_launch(_cfg(SHA, max_seqs=8), 16)
