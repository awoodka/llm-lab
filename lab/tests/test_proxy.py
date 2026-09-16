"""The allowance proxy against a stand-in llama-server.

The fake speaks the three endpoints the proxy uses and records what it was sent, so the tests can assert
what reached the model: the config's own sampling, and an answer sized by the allowance.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from lab.evals.allowance import Allowance, SpeedPoint
from lab.evals.drivers.base import Task
from lab.evals.proxy import AllowanceProxy
from lab.evals.registry import BY_KEY
from lab.evals.runner import EvalSession

QWEN = Allowance(ctx=65536, pp0=949, tg=[SpeedPoint(0, 31.8), SpeedPoint(4096, 30.9), SpeedPoint(16384, 28.2)], limit_s=300, speed_run_id="run-1")


class FakeServer:
    """A llama-server stand-in: one token per word, and an answer that stops on the size it was given.

    With `bos`, it behaves like Gemma 3: the rendered template has no BOS and tokenizing with special
    tokens adds one, as the server does to every chat prompt. `misreport` skews the count it reports.
    """

    def __init__(self):
        self.seen: list[dict] = []
        self.fail_with: int | None = None
        self.bos = False
        self.misreport = 0
        proxy_self = self

        def render(body) -> str:
            text = " ".join(m["content"] for m in body["messages"])
            if body.get("tools"):
                text += " " + " ".join(["tooldef"] * 10)
            return text

        def tokens(text: str, add_special: bool) -> list[int]:
            return ([2] if add_special and proxy_self.bos else []) + list(range(10, 10 + len(text.split())))

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _json(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                self._json(200, {"status": "ok"} if self.path == "/health" else {"data": []})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
                if self.path == "/apply-template":
                    return self._json(200, {"prompt": render(body)})
                if self.path == "/tokenize":
                    return self._json(200, {"tokens": tokens(body["content"], body.get("add_special", False))})
                proxy_self.seen.append(body)
                prompt = len(tokens(render(body), True)) + proxy_self.misreport
                if proxy_self.fail_with:
                    return self._json(proxy_self.fail_with, {"error": "upstream said no"})
                asked = body["max_tokens"]
                # Long allowances finish; short ones run out, which is what a length stop means.
                completion, finish = (12, "stop") if asked > 100 else (asked, "length")
                if body.get("stream"):
                    return self._stream(prompt, completion, finish)
                # Thinking arrives apart from the answer, as llama-server sends it with a reasoning format.
                message = {"role": "assistant", "reasoning_content": "maybe \\boxed{70}", "content": "\\boxed{12}"}
                return self._json(200, {
                    "id": "chatcmpl-1", "model": "fake",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "prompt_tokens_details": {"cached_tokens": 3}},
                    "timings": {"predicted_per_second": 30.5, "prompt_per_second": 900.0},
                })

            def _stream(self, prompt, completion, finish):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("connection", "close")
                self.end_headers()
                self.close_connection = True
                for piece in ({"choices": [{"delta": {"content": "ok"}}]},
                              {"choices": [{"delta": {}, "finish_reason": finish}]},
                              {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}):
                    self.wfile.write(f"data: {json.dumps(piece)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.http.server_address[1]}"

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()


@pytest.fixture
def upstream():
    server = FakeServer()
    yield server
    server.close()


@pytest.fixture
def proxy(upstream, tmp_path):
    p = AllowanceProxy(upstream.url, QWEN, log_path=tmp_path / "requests.jsonl")
    with p:
        yield p


def chat(proxy: AllowanceProxy, words: int, **extra) -> httpx.Response:
    body = {"model": "m", "messages": [{"role": "user", "content": " ".join(["word"] * words)}], **extra}
    headers = {k: v for k, v in [("x-lab-task", extra.pop("task", None))] if v}
    return httpx.post(f"{proxy.base_url}/chat/completions", json=body, headers=headers, timeout=30)


def test_the_answer_is_sized_by_the_allowance_and_sampling_is_left_alone(proxy, upstream):
    r = chat(proxy, words=8000, temperature=0.0, top_p=0.1, seed=7, max_tokens=99999)
    assert r.status_code == 200
    sent = upstream.seen[-1]
    assert sent["max_tokens"] == 8759, "the plan's vector for an 8,000-token prompt at 300 s"
    assert not {"temperature", "top_p", "seed"} & sent.keys(), "the config's launch flags decide sampling"
    assert sent["messages"][0]["role"] == "user"


def test_a_prompt_that_uses_up_the_limit_is_answered_as_unfinished_without_asking_the_model(proxy, upstream):
    r = chat(proxy, words=65536)  # no context left
    assert r.status_code == 200
    assert r.json()["choices"][0]["finish_reason"] == "length"
    assert r.json()["choices"][0]["message"]["content"] == ""
    assert upstream.seen == [], "nothing was sent to the model"
    assert proxy.stats.totals().length_stops == 1


def test_per_task_numbers_and_the_request_log_are_what_the_run_publishes(proxy, tmp_path):
    chat(proxy, words=100, task="gpqa:1")
    chat(proxy, words=100, task="gpqa:2")
    with proxy.task("gpqa:3"):
        chat(proxy, words=100)  # an agent that sets no header still lands on the right task

    assert set(proxy.stats.tasks) == {"gpqa:1", "gpqa:2", "gpqa:3"}
    one = proxy.stats.for_task("gpqa:1")
    assert one.requests == 1 and one.completion_tokens == 12 and one.prompt_tokens == 100
    assert one.allowance_tokens > 0 and one.seconds > 0
    assert proxy.stats.totals().requests == 3

    lines = [json.loads(x) for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    assert [x["task"] for x in lines] == ["gpqa:1", "gpqa:2", "gpqa:3"]
    assert lines[0]["prompt_tokens"] == 100
    assert lines[0]["finish_reason"] == "stop"
    assert lines[0]["cached_tokens"] == 3
    assert lines[0]["predicted_per_second"] == 30.5


def test_an_answer_that_runs_out_of_allowance_is_counted_as_a_length_stop(upstream, tmp_path):
    # 65k of context left, but only ~60 tokens of thinking time: the model gets cut off.
    tight = Allowance(ctx=65536, pp0=949, tg=[SpeedPoint(0, 31.8)], limit_s=2.0)
    with AllowanceProxy(upstream.url, tight, log_path=tmp_path / "r.jsonl") as proxy:
        r = chat(proxy, words=100, task="aime:1")
        assert r.json()["choices"][0]["finish_reason"] == "length"
        assert upstream.seen[-1]["max_tokens"] == 60
        assert proxy.stats.for_task("aime:1").length_stops == 1


def test_the_deep_tier_is_capped_by_context_only(upstream, tmp_path):
    with AllowanceProxy(upstream.url, Allowance(ctx=65536, limit_s=None), log_path=tmp_path / "r.jsonl") as proxy:
        chat(proxy, words=1000)
        assert upstream.seen[-1]["max_tokens"] == 64536


def test_tools_count_toward_the_prompt(proxy, upstream):
    plain = chat(proxy, words=100)
    with_tools = chat(proxy, words=100, tools=[{"type": "function", "function": {"name": "f"}}])
    assert plain.status_code == with_tools.status_code == 200
    assert upstream.seen[-1]["max_tokens"] < upstream.seen[0]["max_tokens"], "the tool definitions took room"
    assert upstream.seen[-1]["tools"], "tools still reach the model"


def test_streaming_relays_the_chunks_and_still_records_usage(proxy):
    with httpx.stream(
        "POST", f"{proxy.base_url}/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"x-lab-task": "lcb:7"}, timeout=30,
    ) as r:
        body = "".join(r.iter_text())
    assert "data: [DONE]" in body
    assert r.headers["x-lab-allowance"], "sized before the first byte, so a streaming harness can record it too"
    assert proxy.count_mismatches == 0
    stats = proxy.stats.for_task("lcb:7")
    assert stats.requests == 1 and stats.completion_tokens == 12 and stats.length_stops == 0


def test_an_upstream_failure_is_reported_and_recorded(proxy, upstream):
    upstream.fail_with = 503
    r = chat(proxy, words=100, task="bfcl:9")
    assert r.status_code == 503
    assert proxy.stats.for_task("bfcl:9").errors == 1


def test_health_and_models_pass_through(proxy):
    assert httpx.get(f"http://127.0.0.1:{proxy.port}/health", timeout=10).json() == {"status": "ok"}
    assert httpx.get(f"{proxy.base_url}/models", timeout=10).status_code == 200
    assert httpx.post(f"{proxy.base_url}/embeddings", json={}, timeout=10).status_code == 404


def test_a_bos_the_template_leaves_out_is_counted_the_way_the_server_counts_it(proxy, upstream):
    upstream.bos = True
    r = chat(proxy, words=8000)
    assert r.headers["x-lab-prompt-tokens"] == "8001", "Gemma's BOS is part of the prompt the model sees"
    assert int(r.headers["x-lab-allowance"]) == upstream.seen[-1]["max_tokens"]
    assert proxy.count_mismatches == 0


def test_a_count_the_server_disagrees_with_is_logged_not_trusted(proxy, upstream, tmp_path):
    upstream.misreport = 1
    chat(proxy, words=100, task="aime:2")
    assert proxy.count_mismatches == 1
    line = json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[-1])
    assert line["count_mismatch"] is True
    assert (line["prompt_tokens"], line["server_prompt_tokens"]) == (100, 101)


def test_the_runner_keeps_the_reply_it_graded_and_the_allowance_it_was_given(proxy, upstream):
    upstream.bos = True
    session = EvalSession(proxy, checkpoint=None, model_name="m", deadline=None, request_timeout=30)
    task = Task(id="aime_2025/I-1", messages=[{"role": "user", "content": " ".join(["word"] * 100)}], answer="70")
    record, reply = session.answer(BY_KEY["aime_2025"], task, 1)

    assert reply == {"reasoning_content": "maybe \\boxed{70}", "content": "\\boxed{12}"}, "the transcript holds both"
    assert (record.extracted, record.passed) == ("12", False), "only the answer is graded, never the thinking"
    assert record.prompt_tokens == 101
    assert record.allowance == upstream.seen[-1]["max_tokens"], "the size that was enforced, not a recomputation"
