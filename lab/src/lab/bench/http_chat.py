"""Engine-agnostic chat benchmark, protocol `chat-c1-v1` (adapted from syv-ai/qwen38-27b-rtx3090, Apache-2.0).

The same eight real chat prompts go to any engine's OpenAI-compatible server, one stream at a time:
a discarded warmup pass, then a pass at Qwen's default sampling and a greedy pass, each prompt capped at
1024 output tokens with thinking off.

Timing comes from the stream. TTFT is the first content delta. TPOT spreads the time between the first
and last content deltas over the server's own token count, never over chunks, because speculative
decoding sends several tokens per chunk. Decode speed is 1000 / mean TPOT, as in the source repo.

No request may be served from another's prompt cache: vLLM requests carry a fresh cache_salt and
llama.cpp requests set cache_prompt false. The run fails if a server still reports a cache hit, or
streams any reasoning, or answers with fewer than 32 tokens.
"""

import json
import re
import statistics
from collections import defaultdict
from importlib import resources
from pathlib import Path
from time import perf_counter as clock

import httpx

from lab import catalog
from lab.bench.common import finish_failed, finish_ok, start_run
from lab.engines.base import get_engine
from lab.engines.process import ServerProcess
from lab.gpu.hosted import exclusive_gpu
from lab.schema import Metric
from lab.store import RunDir
from lab.telemetry import CGROUP, Telemetry

PROTOCOL = "chat-c1-v1"
PORT = 8081
MAX_TOKENS = 1024
MIN_TOKENS = 32
SAMPLED = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
GREEDY = {"temperature": 0.0}
COHORTS = {"http-sampled": SAMPLED, "http-greedy": GREEDY}
COHORT_DIMS = dict(n_prompt=0, n_gen=MAX_TOKENS, depth=0, concurrency=1)
LOCAL_PATH_RE = re.compile(r"(?<![\w.:/-])/(?:home|mnt|srv|root)/[^\s\"',]*")
# Environment the server process actually ran with that shapes its speed; everything else stays local.
ENV_KEEP_RE = re.compile(r"^(VLLM_|PYTORCH_|FLASHINFER_|TORCH|CUDA_|NCCL_|KVARN_)")
VLLM_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:num_preemptions_total",
)
VLLM_BOOT_RE = {
    "weights_load_s": re.compile(r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds"),
    "init_engine_s": re.compile(r"init engine .*? took ([\d.]+) s"),
    "compile_s": re.compile(r"\(compilation: ([\d.]+) s\)"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
}


class BenchError(RuntimeError):
    pass


def load_prompts() -> list[str]:
    text = (resources.files("lab.bench") / "data" / "prompts_real.jsonl").read_text(encoding="utf-8")
    return [json.loads(line)["prompt"] for line in text.splitlines() if line.strip()]


def strip_paths(text: str) -> str:
    """Local paths become their last component, so a recorded command line can be published."""
    return LOCAL_PATH_RE.sub(lambda m: Path(m.group()).name, text)


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus text → {metric name: value summed over label sets}."""
    out: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        try:
            number = float(value)
        except ValueError:
            continue
        out[series.split("{", 1)[0]] += number
    return dict(out)


def _oom_kills() -> int:
    try:
        events = dict(line.split() for line in Path(f"{CGROUP}/memory.events").read_text().splitlines())
        return int(events.get("oom_kill", 0))
    except OSError:
        return 0


# -- one request -------------------------------------------------------------------
def stream_chat(client: httpx.Client, url: str, body: dict) -> dict:
    """Send one streaming chat completion and time it. Raises BenchError on anything that makes it unmeasurable."""
    t0 = clock()
    first = last = None
    parts: list[str] = []
    reasoning_chunks = 0
    usage = timings = finish = None
    with client.stream("POST", url, json=body) as r:
        if r.status_code != 200:
            r.read()
            raise BenchError(f"HTTP {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise BenchError(f"server error mid-stream: {chunk['error']}")
            now = clock()
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning_chunks += 1
                if delta.get("content"):
                    parts.append(delta["content"])
                    if first is None:
                        first = now
                    last = now
                finish = choice.get("finish_reason") or finish
            usage = chunk.get("usage") or usage
            timings = chunk.get("timings") or timings
    end = clock()

    if reasoning_chunks:
        raise BenchError(f"the server streamed {reasoning_chunks} reasoning chunks; thinking must be off for this benchmark")
    if not usage:
        raise BenchError("the stream ended without a usage chunk (stream_options.include_usage)")
    n = int(usage["completion_tokens"])
    if n < MIN_TOKENS or first is None:
        raise BenchError(f"only {n} completion tokens; a reply this short can't be timed")
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if timings:
        cached = max(cached, int(timings.get("cache_n") or 0))
    if cached:
        raise BenchError(f"{cached} prompt tokens came from the prompt cache; every request must be cold")
    tpot_ms = (last - first) * 1000 / (n - 1)
    record = {
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": n,
        "ttft_ms": round((first - t0) * 1000, 2),
        "tpot_ms": round(tpot_ms, 3),
        "decode_tps": round(1000 / tpot_ms, 2),
        "e2e_s": round(end - t0, 3),
        "finish_reason": finish,
    }
    if timings and timings.get("predicted_per_second"):
        record["server_decode_tps"] = round(float(timings["predicted_per_second"]), 2)
    return {"record": record, "text": "".join(parts)}


# -- metrics -----------------------------------------------------------------------
def _mean_sd(xs: list[float]) -> tuple[float, float | None]:
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def cohort_metrics(method: str, records: list[dict], energy_j: float | None, duration_s: float | None,
                   counters: dict[str, float] | None = None) -> list[Metric]:
    def m(key: str, value: float, unit: str, **kw) -> Metric:
        return Metric(key=key, method=method, value=round(value, 3), unit=unit, **COHORT_DIMS, **kw)

    tpots = [r["tpot_ms"] for r in records]
    decodes = [1000 / t for t in tpots]
    ttfts = [r["ttft_ms"] for r in records]
    tokens = sum(r["completion_tokens"] for r in records)
    ttft_mean, ttft_sd = _mean_sd(ttfts)
    tpot_mean, tpot_sd = _mean_sd(tpots)
    out = [
        m("decode_tps", 1000 / tpot_mean, "t/s", stddev=_mean_sd(decodes)[1], n=len(records), samples=[round(d, 2) for d in decodes]),
        m("tpot_ms", tpot_mean, "ms", stddev=tpot_sd, n=len(records), samples=tpots),
        m("ttft_ms", ttft_mean, "ms", stddev=ttft_sd, n=len(records), samples=ttfts),
        m("prefill_tps", sum(r["prompt_tokens"] for r in records) / (sum(ttfts) / 1000), "t/s", n=len(records)),
        m("e2e_tps", tokens / sum(r["e2e_s"] for r in records), "t/s", n=len(records)),
        m("out_tokens_mean", tokens / len(records), "tokens", n=len(records)),
    ]
    if energy_j and duration_s:
        out.append(m("gpu_w_avg", energy_j / duration_s, "W"))
        out.append(m("tokens_per_joule", tokens / energy_j, "tok/J"))
    if counters and counters.get("vllm:spec_decode_num_drafts_total"):
        steps = counters["vllm:spec_decode_num_drafts_total"]
        accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
        out.append(m("spec_accept_len", 1 + accepted / steps, "tok/step", n=int(steps)))
    return out


def _run_metric(key: str, value: float, unit: str) -> Metric:
    return Metric(key=key, method="http", value=value, unit=unit)


# -- the run -----------------------------------------------------------------------
def _compile_cache_state() -> dict:
    root = Path.home() / ".cache/vllm/torch_compile_cache"
    if not root.is_dir():
        return {"entries": 0, "bytes": 0}
    return {"entries": len(list(root.iterdir())), "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())}


def _boot_facts(log: Path) -> dict:
    text = log.read_text(errors="replace") if log.is_file() else ""
    facts = {}
    for key, rx in VLLM_BOOT_RE.items():
        if found := rx.findall(text):
            facts[key] = float(found[-1].replace(",", ""))
    return facts


def _scrape(base: str) -> dict[str, float]:
    parsed = parse_metrics(httpx.get(f"{base}/metrics", timeout=10).text)
    return {k: parsed.get(k, 0.0) for k in VLLM_COUNTERS}


def run_http(ref: str, *, wait: bool, keep_paused: bool, cli_args: str) -> RunDir:
    model, cfg = catalog.load_config(ref)
    engine = get_engine(model.engine, cfg)
    prompts = load_prompts()
    served_id = f"{model.slug}/{cfg.slug}"
    argv = engine.server_argv(model, cfg, host="127.0.0.1", port=PORT)
    weights = catalog.resolve_model_path(model)
    ctx = start_run(model, cfg, "speed", engine, weights, cli_args)
    ctx.bundle.run.raw["bench"] = {
        "protocol": PROTOCOL, "prompts": len(prompts), "max_tokens": MAX_TOKENS, "concurrency": 1,
        "cohorts": COHORTS, "thinking": False, "warmup": "all prompts once, discarded",
    }  # fmt: skip
    (ctx.rd.path / "commands.json").write_text(json.dumps({"server": argv}, indent=2))
    base = f"http://127.0.0.1:{PORT}"
    is_vllm = engine.name == "vllm"
    tel: Telemetry | None = None
    raw: dict = {"protocol": PROTOCOL, "requests": []}
    metrics: list[Metric] = []
    oom_before = _oom_kills()
    try:
        with exclusive_gpu(f"bench http {ref}", wait=wait, keep_paused=keep_paused):
            if is_vllm:
                raw["compile_cache_before"] = _compile_cache_state()
            tel = Telemetry().start()
            server = ServerProcess(argv, log_path=ctx.rd.logs / "server.log")
            tel.watch_pid = server.pid
            try:
                tel.phase("boot")
                load_s = server.wait_healthy(f"{base}/health", engine.boot_timeout_s)
                listed = httpx.get(f"{base}/v1/models", timeout=10).raise_for_status().json()["data"]
                ids = [m["id"] for m in listed]
                if ids != [served_id]:
                    raise BenchError(f"server lists {ids}, expected [{served_id!r}]")
                if is_vllm and listed[0].get("max_model_len") != cfg.params.ctx:
                    raise BenchError(f"server max_model_len {listed[0].get('max_model_len')} != config ctx {cfg.params.ctx}")
                raw["server"] = {
                    "cmdline": [strip_paths(a) for a in server.cmdline()],
                    "env": {k: strip_paths(v) for k, v in sorted(server.environ().items()) if ENV_KEEP_RE.match(k)},
                }
                url = f"{base}/v1/chat/completions"
                with httpx.Client(timeout=httpx.Timeout(10, read=300)) as client, open(ctx.rd.raw / "outputs.jsonl", "w") as outputs:

                    def one_pass(phase: str, sampling: dict) -> list[dict]:
                        records = []
                        for i, prompt in enumerate(prompts):
                            body = {
                                "model": served_id,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": MAX_TOKENS,
                                "stream": True,
                                "stream_options": {"include_usage": True},
                                "chat_template_kwargs": {"enable_thinking": False},
                                **sampling,
                                **engine.request_extras(),
                            }
                            got = stream_chat(client, url, body)
                            rec = {"phase": phase, "prompt": i, **got["record"]}
                            outputs.write(json.dumps({**rec, "text": got["text"]}) + "\n")
                            records.append(rec)
                        return records

                    tel.phase("warmup")
                    raw["requests"] += one_pass("warmup", SAMPLED)
                    for method, sampling in COHORTS.items():
                        before = _scrape(base) if is_vllm else None
                        tel.phase(method)
                        records = one_pass(method, sampling)
                        tel.end_phase()
                        counters = None
                        if before is not None:
                            after = _scrape(base)
                            counters = {k: after[k] - before[k] for k in VLLM_COUNTERS}
                            raw.setdefault("counters", {})[method] = counters
                            if counters["vllm:prefix_cache_hits_total"] > 0:
                                raise BenchError(f"{method}: vLLM reports {counters['vllm:prefix_cache_hits_total']:.0f} prefix-cache hit tokens")
                        phase = tel.phases[-1]
                        metrics += cohort_metrics(method, records, phase.energy_j, phase.duration_s, counters)
                        raw["requests"] += records
            finally:
                tel.stop()
                server.stop()
        if is_vllm:
            raw["boot"] = _boot_facts(ctx.rd.logs / "server.log")
        summary = tel.summary()
        oom = _oom_kills() - oom_before
        metrics += [
            _run_metric("load_s", round(load_s, 1), "s"),
            _run_metric("vram_peak_mb", round(summary["vram_peak_mb"] - summary["vram_baseline_mb"]), "MiB"),
            _run_metric("ram_peak_mb", round(summary["ram_peak_mb"]), "MiB"),
            _run_metric("oom_kills", oom, "count"),
        ]
        if oom:
            raise BenchError(f"{oom} OOM kill(s) in the container during the run")
        finish_ok(ctx, metrics, tel, raw)
        return ctx.rd
    except BaseException as e:
        ctx.bundle.run.raw.update(raw)
        finish_failed(ctx, e, tel)
        raise
