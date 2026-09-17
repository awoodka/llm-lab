import hashlib
import json
from importlib import resources

import httpx
import pytest

from lab.bench import http_chat
from lab.bench.http_chat import BenchError, cohort_metrics, load_prompts, parse_metrics, stream_chat, strip_paths

URL = "http://server/v1/chat/completions"


def _sse(*chunks: dict | str) -> bytes:
    lines = [f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks]
    return "".join([": keep-alive\n\n", *lines]).encode()


def _delta(**delta) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}


def _client(body: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, content=body)))


@pytest.fixture
def clock(monkeypatch):
    """perf_counter that advances by a scripted step each call: request start, then one reading per chunk."""
    ticks = iter([])

    def script(*times):
        nonlocal ticks
        ticks = iter(times)

    monkeypatch.setattr(http_chat, "clock", lambda: next(ticks))
    return script


def test_timing_uses_the_servers_token_count_not_chunks(clock):
    # 3 content chunks carry 100 tokens (speculative decoding sends several per chunk).
    body = _sse(
        _delta(role="assistant", content=""),
        _delta(content="Hello"),
        _delta(content=" there, a longer piece"),
        {"choices": [{"index": 0, "delta": {"content": " end"}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}},
        "[DONE]",
    )
    # t0, then one tick per data chunk (role, 3 content, usage), then end
    clock(10.0, 10.1, 10.25, 10.75, 11.24, 11.25, 11.3)
    got = stream_chat(_client(body), URL, {})
    rec = got["record"]
    assert got["text"] == "Hello there, a longer piece end"
    assert rec["ttft_ms"] == 250.0
    assert rec["tpot_ms"] == pytest.approx((11.24 - 10.25) * 1000 / 99, abs=1e-3)
    assert rec["decode_tps"] == pytest.approx(99 / 0.99, abs=0.01)
    assert rec["e2e_s"] == pytest.approx(1.3) and rec["finish_reason"] == "length"


def test_llamacpp_timings_are_kept_for_the_cross_check(clock):
    body = _sse(_delta(content="a"), _delta(content="b"),
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 64},
                 "timings": {"cache_n": 0, "predicted_per_second": 31.25}})  # fmt: skip
    clock(0.0, 0.2, 2.2, 2.3, 2.4)
    assert stream_chat(_client(body), URL, {})["record"]["server_decode_tps"] == 31.25


@pytest.mark.parametrize(("chunks", "match"), [
    ((_delta(reasoning_content="hmm"), _delta(content="x"), {"usage": {"prompt_tokens": 1, "completion_tokens": 64}}), "reasoning"),
    ((_delta(reasoning="hmm"), _delta(content="x"), {"usage": {"prompt_tokens": 1, "completion_tokens": 64}}), "reasoning"),
    ((_delta(content="x"), _delta(content="y")), "usage chunk"),
    ((_delta(content="x"), _delta(content="y"), {"usage": {"prompt_tokens": 1, "completion_tokens": 12}}), "only 12"),
    ((_delta(content="x"), _delta(content="y"),
      {"usage": {"prompt_tokens": 9, "completion_tokens": 64, "prompt_tokens_details": {"cached_tokens": 8}}}), "8 prompt tokens"),
    ((_delta(content="x"), _delta(content="y"),
      {"usage": {"prompt_tokens": 9, "completion_tokens": 64}, "timings": {"cache_n": 4}}), "4 prompt tokens"),
    (({"error": {"message": "boom"}},), "mid-stream"),
])  # fmt: skip
def test_unmeasurable_replies_fail_the_run(chunks, match):
    with pytest.raises(BenchError, match=match):
        stream_chat(_client(_sse(*chunks, "[DONE]")), URL, {})


def test_http_errors_fail_the_run():
    with pytest.raises(BenchError, match="HTTP 400: bad"):
        stream_chat(_client(b"bad", status=400), URL, {})


def _rec(prompt_tokens, completion_tokens, ttft_ms, tpot_ms, e2e_s):
    return dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, ttft_ms=ttft_ms, tpot_ms=tpot_ms, e2e_s=e2e_s)


def test_cohort_metrics_arithmetic_and_dimensions():
    records = [_rec(100, 1024, 200.0, 8.0, 8.4), _rec(300, 512, 600.0, 10.0, 5.7)]
    counters = {"vllm:spec_decode_num_drafts_total": 500.0, "vllm:spec_decode_num_accepted_tokens_total": 1100.0}
    ms = {m.key: m for m in cohort_metrics("http-sampled", records, energy_j=2800.0, duration_s=14.0, counters=counters)}
    assert ms["decode_tps"].value == pytest.approx(1000 / 9.0, abs=1e-3)
    assert ms["decode_tps"].samples == [125.0, 100.0] and ms["decode_tps"].n == 2
    assert ms["tpot_ms"].value == 9.0 and ms["ttft_ms"].value == 400.0
    assert ms["prefill_tps"].value == pytest.approx(400 / 0.8)
    assert ms["e2e_tps"].value == pytest.approx(1536 / 14.1, abs=1e-3)
    assert ms["out_tokens_mean"].value == 768.0
    assert ms["gpu_w_avg"].value == 200.0 and ms["tokens_per_joule"].value == pytest.approx(1536 / 2800, abs=1e-3)
    assert ms["spec_accept_len"].value == pytest.approx(3.2) and ms["spec_accept_len"].unit == "tok/step"
    for m in ms.values():
        assert (m.method, m.n_prompt, m.n_gen, m.depth, m.concurrency) == ("http-sampled", 0, 1024, 0, 1)


def test_llamacpp_cohorts_have_no_accept_length():
    keys = {m.key for m in cohort_metrics("http-greedy", [_rec(10, 100, 50.0, 30.0, 3.0)] * 2, 100.0, 5.0, None)}
    assert "spec_accept_len" not in keys and "decode_tps" in keys


def test_parse_metrics_sums_label_sets_and_skips_comments():
    text = (
        "# HELP vllm:spec_decode_num_drafts_total drafts\n"
        'vllm:spec_decode_num_drafts_total{engine="0",model_name="m/c"} 10.0\n'
        'vllm:spec_decode_num_drafts_total{engine="1",model_name="m/c"} 5.0\n'
        "vllm:num_preemptions_total 0.0\n"
        "garbage line\n"
    )
    assert parse_metrics(text) == {"vllm:spec_decode_num_drafts_total": 15.0, "vllm:num_preemptions_total": 0.0}


def test_recorded_command_lines_lose_their_local_paths():
    arg = '{"method":"dflash","model":"/home/alex/qwen-serving/models/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7}'
    assert strip_paths(arg) == '{"method":"dflash","model":"Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7}'
    assert strip_paths("/home/alex/qwen-serving-bae2023/venv/bin/python") == "python"
    assert strip_paths("https://huggingface.co/home/x") == "https://huggingface.co/home/x"


def test_vendored_prompts_are_the_pinned_file():
    data = (resources.files("lab.bench") / "data" / "prompts_real.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == "27da4fd8b2dcc133b98cec54bc337c34f5acb1b6372cc2fea1d8a92da9c78650"
    assert len(load_prompts()) == 8
