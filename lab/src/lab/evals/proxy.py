"""One place where the thinking allowance is enforced: an OpenAI-compatible proxy in front of the
config's own server.

Harnesses talk to the proxy, never to llama-server, so every benchmark is sized the same way and every
request lands in one log. Two rules the proxy keeps:

- **It never changes how the model samples.** Sampling fields are stripped from each request, so what
  runs is what the config was launched with (`/props` records it in the run).
- **It sizes every answer.** `max_tokens` becomes the allowance for that prompt; thinking counts toward
  it, and a request that ends on `length` is an unfinished answer, which the benchmark scores as wrong.

Prompt tokens are counted the way the server itself counts them: render the chat template, then
tokenize it with special tokens added. `/apply-template` leaves out the BOS a model like Gemma 3 needs
and the server adds it back when it tokenizes, so the count must too. Verified against llama-server
b10883 on Qwen3.8 (no BOS) and Gemma 3 (BOS): computed count == reported `usage.prompt_tokens`. Every
answer is checked against that report, and a disagreement is logged and counted rather than trusted.

Each answer carries `x-lab-prompt-tokens` and `x-lab-allowance` headers, so a harness records the size
that was actually enforced.
"""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from lab.evals.allowance import Allowance

#: Dropped from every request: the config's launch flags decide how it samples, not the harness.
SAMPLING_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "typical_p", "typ_p", "tfs_z", "top_n_sigma",
    "repeat_penalty", "repeat_last_n", "penalize_nl", "presence_penalty", "frequency_penalty",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers",
    "xtc_probability", "xtc_threshold", "mirostat", "mirostat_tau", "mirostat_eta",
    "samplers", "seed", "min_keep", "logit_bias", "n_probs", "logprobs", "top_logprobs",
})
#: Dropped because the proxy sets the answer's size itself.
LENGTH_FIELDS = frozenset({"max_tokens", "max_completion_tokens", "n_predict"})
#: Passed to the template renderer so the count matches what the server will build.
TEMPLATE_FIELDS = ("messages", "tools", "tool_choice", "chat_template_kwargs", "add_generation_prompt")


@dataclass
class TaskStats:
    """What one task cost, for the per-benchmark operating numbers the run publishes."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    allowance_tokens: int = 0
    length_stops: int = 0
    errors: int = 0
    seconds: float = 0.0

    def add(self, *, prompt: int, completion: int, allowance: int, finish: str | None, seconds: float, error: bool) -> None:
        self.requests += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.allowance_tokens += allowance
        self.length_stops += finish == "length"
        self.errors += error
        self.seconds += seconds


@dataclass
class ProxyStats:
    tasks: dict[str, TaskStats] = field(default_factory=dict)

    def for_task(self, task: str) -> TaskStats:
        return self.tasks.setdefault(task, TaskStats())

    def totals(self) -> TaskStats:
        total = TaskStats()
        for t in self.tasks.values():
            for k, v in asdict(t).items():
                setattr(total, k, getattr(total, k) + v)
        return total


class AllowanceProxy:
    """Serves `/v1/chat/completions` in front of `upstream`, sizing every answer.

    Start it inside the session that owns the GPU; `base_url` is what harnesses are pointed at.
    """

    def __init__(self, upstream: str, allowance: Allowance, *, log_path: Path | None = None, host: str = "127.0.0.1", port: int = 0):
        self.upstream = upstream.rstrip("/")
        self.allowance = allowance
        self.stats = ProxyStats()
        self.current_task = "unknown"
        #: Answers whose server-reported prompt size differed from the proxy's count.
        self.count_mismatches = 0
        self._lock = threading.Lock()
        self._log = open(log_path, "a", buffering=1) if log_path else None  # noqa: SIM115 - closed in stop()
        # No read timeout: one answer may legitimately take the whole task limit.
        self._client = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0))
        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="allowance-proxy", daemon=True)

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> "AllowanceProxy":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._client.close()
        if self._log:
            self._log.close()

    def __enter__(self) -> "AllowanceProxy":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL, for harnesses that expect one."""
        host, port = self._server.server_address[0], self.port
        return f"http://{host}:{port}/v1"

    @contextmanager
    def task(self, task_id: str) -> Iterator[TaskStats]:
        """Attribute everything sent while this is open to `task_id`, for agents that can't set a header."""
        previous = self.current_task
        self.current_task = task_id
        try:
            yield self.stats.for_task(task_id)
        finally:
            self.current_task = previous

    # -- request handling ------------------------------------------------------
    def prompt_tokens(self, body: dict[str, Any]) -> int:
        """Count the prompt the way the server will build it: render the template, then tokenize."""
        rendered = self._client.post(
            f"{self.upstream}/apply-template",
            json={k: body[k] for k in TEMPLATE_FIELDS if k in body},
        )
        rendered.raise_for_status()
        tokens = self._client.post(
            f"{self.upstream}/tokenize",
            json={"content": rendered.json()["prompt"], "add_special": True},
        )
        tokens.raise_for_status()
        return len(tokens.json()["tokens"])

    @staticmethod
    def sized_body(body: dict[str, Any], allowance: int) -> dict[str, Any]:
        out = {k: v for k, v in body.items() if k not in SAMPLING_FIELDS and k not in LENGTH_FIELDS}
        out["max_tokens"] = allowance
        if out.get("stream"):
            out["stream_options"] = {**(out.get("stream_options") or {}), "include_usage": True}
        return out

    def record(self, task: str, entry: dict[str, Any]) -> None:
        with self._lock:
            reported = entry.get("server_prompt_tokens")
            if reported is not None and reported != entry.get("prompt_tokens"):
                entry["count_mismatch"] = True
                self.count_mismatches += 1
            self.stats.for_task(task).add(
                prompt=entry.get("prompt_tokens") or 0,
                completion=entry.get("completion_tokens") or 0,
                allowance=entry.get("allowance") or 0,
                finish=entry.get("finish_reason"),
                seconds=entry.get("seconds") or 0.0,
                error=bool(entry.get("error")),
            )
            if self._log:
                self._log.write(json.dumps(entry) + "\n")


def _no_allowance_response(model: str, prompt_tokens: int) -> dict[str, Any]:
    """The prompt alone used up the task's time: an empty answer that stopped on length, i.e. unfinished."""
    return {
        "id": "lab-no-allowance",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0, "total_tokens": prompt_tokens},
    }


def _usage_from_stream(chunks: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    usage: dict[str, Any] = {}
    finish: str | None = None
    for c in chunks:
        if c.get("usage"):
            usage = c["usage"]
        for choice in c.get("choices") or []:
            finish = choice.get("finish_reason") or finish
    return usage, finish


def _make_handler(proxy: AllowanceProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "lab-allowance-proxy"

        def log_message(self, *args) -> None:  # the JSONL log is the record; stderr stays quiet
            pass

        # -- plumbing ----------------------------------------------------------
        def _send(self, status: int, payload: Any, content_type: str = "application/json", headers: dict[str, str] | None = None) -> None:
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send(status, {"error": {"message": message, "type": "lab_proxy"}})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path not in ("/health", "/props", "/v1/models", "/models"):
                self._error(404, f"no route for GET {path}")
                return
            try:
                r = proxy._client.get(f"{proxy.upstream}{path}", timeout=30)
            except httpx.HTTPError as e:
                self._error(502, f"upstream unreachable: {e}")
                return
            self._send(r.status_code, r.content, r.headers.get("content-type", "application/json"))

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._error(404, f"no route for POST {path}; the proxy serves chat completions only")
                return
            task = self.headers.get("x-lab-task") or proxy.current_task
            started = time.monotonic()
            try:
                body = self._body()
            except json.JSONDecodeError as e:
                self._error(400, f"invalid JSON: {e}")
                return
            try:
                prompt_tokens = proxy.prompt_tokens(body)
            except httpx.HTTPError as e:
                proxy.record(task, {"task": task, "error": f"token count failed: {e}", "seconds": time.monotonic() - started})
                self._error(502, f"could not count the prompt: {e}")
                return

            allowance = proxy.allowance.for_prompt(prompt_tokens)
            sized_headers = {"x-lab-prompt-tokens": str(prompt_tokens), "x-lab-allowance": str(allowance)}
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task": task,
                "prompt_tokens": prompt_tokens,
                "allowance": allowance,
            }
            if allowance <= 0:
                entry.update(completion_tokens=0, finish_reason="length", seconds=time.monotonic() - started, no_allowance=True)
                proxy.record(task, entry)
                self._send(200, _no_allowance_response(body.get("model", "lab"), prompt_tokens), headers=sized_headers)
                return

            sized = proxy.sized_body(body, allowance)
            try:
                if sized.get("stream"):
                    self._stream(sized, entry, task, started, sized_headers)
                else:
                    self._complete(sized, entry, task, started, sized_headers)
            except httpx.HTTPError as e:
                entry.update(error=str(e), seconds=time.monotonic() - started)
                proxy.record(task, entry)
                self._error(502, f"upstream failed: {e}")

        # -- the two shapes of an answer ---------------------------------------
        def _complete(self, sized: dict[str, Any], entry: dict[str, Any], task: str, started: float, headers: dict[str, str]) -> None:
            r = proxy._client.post(f"{proxy.upstream}/v1/chat/completions", json=sized)
            entry["seconds"] = round(time.monotonic() - started, 3)
            if r.status_code >= 400:
                entry.update(error=f"upstream {r.status_code}", status=r.status_code)
                proxy.record(task, entry)
                self._send(r.status_code, r.content, r.headers.get("content-type", "application/json"))
                return
            data = r.json()
            usage = data.get("usage") or {}
            timings = data.get("timings") or {}
            entry.update(
                server_prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=(data.get("choices") or [{}])[0].get("finish_reason"),
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                predicted_per_second=timings.get("predicted_per_second"),
                prompt_per_second=timings.get("prompt_per_second"),
            )
            proxy.record(task, entry)
            self._send(200, data, headers=headers)

        def _stream(self, sized: dict[str, Any], entry: dict[str, Any], task: str, started: float, headers: dict[str, str]) -> None:
            """Relay the SSE bytes untouched, reading usage out of the final chunks as they pass."""
            chunks: list[dict[str, Any]] = []
            with proxy._client.stream("POST", f"{proxy.upstream}/v1/chat/completions", json=sized) as r:
                if r.status_code >= 400:
                    r.read()
                    entry.update(error=f"upstream {r.status_code}", status=r.status_code, seconds=round(time.monotonic() - started, 3))
                    proxy.record(task, entry)
                    self._send(r.status_code, r.content, r.headers.get("content-type", "application/json"))
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.close_connection = True
                for line in r.iter_lines():
                    payload = f"{line}\n\n".encode() if line else b"\n"
                    self.wfile.write(payload)
                    self.wfile.flush()
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        try:
                            chunks.append(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            pass
            usage, finish = _usage_from_stream(chunks)
            entry.update(
                server_prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens", 0),
                finish_reason=finish,
                seconds=round(time.monotonic() - started, 3),
                stream=True,
            )
            proxy.record(task, entry)

    return Handler
