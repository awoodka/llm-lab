"""Run a tier of capability benchmarks against one config.

The shape of a session: take the GPU (chat pauses), start the config's own server exactly as it is
launched for chat, put the allowance proxy in front of it, then work through the tasks, checkpointing
each answer as it is graded. `--stop-at` ends a sitting cleanly and `--resume` picks it up, so a
15-hour tier can be run across several nights and still score as one run.

Sampling is never set here. The config's launch flags decide it, and what the server reports is copied
into the run so a published number can be traced back to how it was produced.
"""

import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import httpx

from lab import catalog
from lab.bench.common import finish_failed, finish_ok, lab_version, now_iso, start_run
from lab.catalog import ConfigSpec, ModelSpec
from lab.engines.base import get_engine
from lab.engines.process import ServerProcess
from lab.evals import registry
from lab.evals.allowance import DEEP_LIMIT_S, QUICK_LIMIT_S, Allowance, from_speed_bundle, latest_speed_bundle
from lab.evals.drivers import get_driver
from lab.evals.drivers.base import Subset, Task, check_against_pin
from lab.evals.proxy import AllowanceProxy
from lab.evals.registry import Benchmark
from lab.evals.session import Attempt, Checkpoint, counted, excluded_tasks, metrics, progress, score
from lab.gpu.hosted import exclusive_gpu
from lab.schema import EvalResult, Metric
from lab.store import RunDir, find_run
from lab.telemetry import Telemetry

EVAL_PORT = 8082
HEALTH_TIMEOUT_S = 900.0
#: Telemetry every 10 s, published once a minute: a tier runs for hours, so 10 Hz would be millions of rows.
TELEMETRY_INTERVAL_S = 10.0
TELEMETRY_PUBLISH_S = 60.0
#: An infrastructure failure gets this many extra goes before the task is excluded.
RETRIES = 2
#: Sampling as the server reports it, so a result can be traced back to how it was produced.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "typical_p", "repeat_penalty", "repeat_last_n",
                 "presence_penalty", "frequency_penalty", "mirostat", "seed", "samplers", "dry_multiplier")


class EvalError(RuntimeError):
    pass


class StopRequested(Exception):
    """--stop-at reached: end the sitting cleanly, keep the checkpoint, publish nothing."""


def parse_stop_at(value: str | None) -> float | None:
    """`HH:MM` local time, as a monotonic deadline. Tomorrow's if that time has already passed today."""
    if not value:
        return None
    if not re.fullmatch(r"\d{1,2}:\d{2}", value):
        raise ValueError(f"--stop-at wants HH:MM, got {value!r}")
    hour, minute = (int(x) for x in value.split(":"))
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return time.monotonic() + (target - now).total_seconds()


def _props(base_url: str) -> dict[str, Any]:
    """Sampling and build identity from the live server. Never the model path: runs publish file names."""
    try:
        d = httpx.get(f"{base_url}/props", timeout=30).json()
    except httpx.HTTPError as e:
        return {"error": str(e)}
    params = d.get("default_generation_settings", {}).get("params", {})
    return {
        "sampling": {k: params[k] for k in SAMPLING_KEYS if k in params},
        "n_ctx": d.get("default_generation_settings", {}).get("n_ctx") or d.get("n_ctx"),
        "total_slots": d.get("total_slots"),
        "build_info": d.get("build_info"),
    }


def _greedy(props: dict[str, Any]) -> bool:
    """A config that samples greedily gives the same answer every time, so extra attempts buy nothing."""
    s = props.get("sampling") or {}
    return s.get("temperature") == 0 or s.get("top_k") == 1


def _launch_config(cfg: ConfigSpec, parallel: int) -> ConfigSpec:
    """`--parallel N` serves N requests at once, each still getting the config's full context."""
    if parallel <= 1:
        return cfg
    params = replace(cfg.params) if hasattr(cfg.params, "__replace__") else cfg.params.model_copy()
    params.parallel = parallel
    params.ctx = cfg.params.ctx * parallel
    return cfg.model_copy(update={"params": params})


class EvalSession:
    """One sitting: the proxy, the checkpoint and the deadline that ends it."""

    def __init__(self, proxy: AllowanceProxy, checkpoint: Checkpoint, model_name: str, deadline: float | None, request_timeout: float):
        self.proxy = proxy
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.deadline = deadline
        self.request_timeout = request_timeout
        self.lock = threading.Lock()
        self.client = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=request_timeout, write=60.0, pool=60.0))
        self.stopping = False

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            self.stopping = True
            raise StopRequested

    def answer(self, bench: Benchmark, task: Task, attempt: int) -> tuple[Attempt, dict[str, Any]]:
        """One request through the proxy, graded, plus the reply it was graded on.

        Infrastructure failures are retried, then excluded. Only `content` is graded: a model's thinking
        (`reasoning_content`) is kept for the transcript but never read for an answer.
        """
        driver = get_driver(bench.key)
        record = Attempt(benchmark=bench.key, task=task.id, attempt=attempt)
        body = {"model": self.model_name, "messages": task.messages, **task.request}
        headers = {"x-lab-task": f"{bench.key}:{task.id}#{attempt}"}
        for tries_left in range(RETRIES, -1, -1):
            started = time.monotonic()
            try:
                r = self.client.post(f"{self.proxy.base_url}/chat/completions", json=body, headers=headers)
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                record.error = f"{type(e).__name__}: {e}"
                record.seconds += time.monotonic() - started
                if tries_left:
                    time.sleep(5)
                    continue
                record.excluded = True
                return record, {}
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            usage = data.get("usage") or {}
            outcome = driver.grade(task, text)
            record.error = None
            record.seconds += round(time.monotonic() - started, 2)
            record.passed = outcome.passed
            record.extracted = outcome.extracted
            record.detail = outcome.detail
            record.prompt_tokens = usage.get("prompt_tokens") or 0
            record.completion_tokens = usage.get("completion_tokens") or 0
            record.finish_reason = choice.get("finish_reason")
            # The size the proxy enforced, not a recomputation from the server's count.
            record.allowance = int(r.headers.get("x-lab-allowance") or self.proxy.allowance.for_prompt(record.prompt_tokens))
            # An answer that ran out of allowance is unfinished, which the benchmark scores as wrong.
            if record.finish_reason == "length":
                record.passed = False
            reply = {k: message[k] for k in ("reasoning_content", "content", "tool_calls") if message.get(k)}
            return record, reply
        raise AssertionError("unreachable")

    def run_benchmark(self, bench: Benchmark, subset: Subset, attempts: int, parallel: int, transcripts) -> None:
        jobs = [
            (task, attempt)
            for task in subset.tasks
            for attempt in range(1, attempts + 1)
            if not self.checkpoint.has(bench.key, task.id, attempt)
        ]
        excluded = excluded_tasks(self.checkpoint.for_benchmark(bench.key))
        if not jobs:
            return

        def work(job: tuple[Task, int]) -> None:
            task, attempt = job
            if task.id in excluded:
                return  # already out of the score; more attempts would only spend time
            self.check_deadline()
            record, reply = self.answer(bench, task, attempt)
            with self.lock:
                self.checkpoint.add(record)
                transcripts(bench, task, attempt, record, reply)
                if record.excluded:
                    excluded.add(task.id)
                    print(f"  excluded {task.id}: {record.error}", file=sys.stderr)

        if parallel <= 1:
            for job in jobs:
                work(job)
        else:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                for future in [pool.submit(work, job) for job in jobs]:
                    future.result()


def run_evals(
    ref: str,
    *,
    tier: str,
    benchmark_keys: list[str] | None = None,
    limit: int | None = None,
    attempts_override: int | None = None,
    parallel: int = 1,
    resume: str | None = None,
    stop_at: str | None = None,
    wait: bool = False,
    keep_paused: bool = False,
    published_speed_only: bool = True,
    cli_args: str = "lab eval",
) -> RunDir:
    model, cfg = catalog.load_config(ref)
    benches = registry.resolve(benchmark_keys, tier)  # type: ignore[arg-type]
    deadline = parse_stop_at(stop_at)

    # Everything that can fail without the GPU fails here: datasets, pins, and the speed run to size by.
    subsets: dict[str, Subset] = {}
    for bench in benches:
        driver = get_driver(bench.key)
        subset = driver.subset()
        check_against_pin(bench.key, subset)
        if limit:
            subset = Subset(name=f"{subset.name}[:{limit}]", tasks=subset.tasks[:limit], source=subset.source)
        subsets[bench.key] = subset
    allowance = _allowance_for(model, cfg, tier, published_speed_only)

    rd, bundle, resumed = _open_run(ref, model, cfg, tier, cli_args, resume)
    ctx = _ctx(rd, bundle, time.monotonic())
    engine = get_engine(model.engine)
    launch_cfg = _launch_config(cfg, parallel)
    argv = engine.server_argv(model, launch_cfg, host="127.0.0.1", port=EVAL_PORT)
    checkpoint = Checkpoint(rd.raw / "attempts.jsonl")
    session_started = time.monotonic()
    tel: Telemetry | None = None
    stopped_early = False

    print(f"{'resuming' if resumed else 'starting'} {rd.path.name}: {tier} tier, {', '.join(b.key for b in benches)}", file=sys.stderr)
    try:
        with exclusive_gpu(f"evals {ref}", wait=wait, keep_paused=keep_paused, cool=False):
            tel = Telemetry(interval_s=TELEMETRY_INTERVAL_S).start()
            server = ServerProcess(argv, log_path=rd.logs / "llama-server.log")
            try:
                tel.watch_pid = server.pid
                server.wait_healthy(f"http://127.0.0.1:{EVAL_PORT}/health", timeout_s=HEALTH_TIMEOUT_S)
                props = _props(f"http://127.0.0.1:{EVAL_PORT}")
                attempts_note = None
                with AllowanceProxy(f"http://127.0.0.1:{EVAL_PORT}", allowance, log_path=rd.logs / "requests.jsonl") as proxy:
                    limit_s = allowance.limit_s or DEEP_LIMIT_S
                    sess = EvalSession(proxy, checkpoint, f"{model.slug}/{cfg.slug}", deadline, request_timeout=limit_s * 4 + 120)
                    for bench in benches:
                        attempts = attempts_override or bench.attempts
                        if attempts > 1 and _greedy(props):
                            attempts_note = "greedy sampling: one attempt per task, since every attempt would be identical"
                            attempts = 1
                        tel.phase(bench.key)
                        try:
                            sess.run_benchmark(bench, subsets[bench.key], attempts, parallel, _transcript_writer(rd))
                        except StopRequested:
                            stopped_early = True
                            break
                        finally:
                            print("  " + progress(bench, checkpoint.for_benchmark(bench.key), len(subsets[bench.key].tasks)), file=sys.stderr)
                    stats = {task: vars(s) for task, s in proxy.stats.tasks.items()}
                    count_mismatches = proxy.count_mismatches
                    if count_mismatches:
                        print(f"  warning: {count_mismatches} answer(s) where the server counted the prompt differently "
                              f"from the proxy; see count_mismatch in {rd.logs / 'requests.jsonl'}", file=sys.stderr)
            finally:
                server.stop()
                tel.stop()
    except BaseException as e:
        checkpoint.close()
        if tel is not None:
            tel.stop()
        if not isinstance(e, KeyboardInterrupt):
            finish_failed(ctx, e, tel)
            raise
        stopped_early = True
        print(f"\ninterrupted; resume with: lab eval {ref} --tier {tier} --resume {rd.path.name}", file=sys.stderr)
        _save_progress(rd, bundle, checkpoint, session_started, benches, subsets)
        return rd

    checkpoint.close()
    bundle.run.raw.update(
        allowance=allowance.as_raw(),
        props=props,
        subsets={k: {"id": s.id, "name": s.name, "source": s.source, "n_tasks": len(s.tasks)} for k, s in subsets.items()},
        # Across sittings: an attempt re-asked after an interruption replaces its earlier entry.
        proxy_stats={**(bundle.run.raw.get("proxy_stats") or {}), **stats},
        proxy_count_mismatches=(bundle.run.raw.get("proxy_count_mismatches") or 0) + count_mismatches,
        parallel=parallel,
        limited=bool(limit),
        launch_overrides={"parallel": parallel, "ctx": launch_cfg.params.ctx} if parallel > 1 else {},
    )
    if attempts_note:
        bundle.run.raw["attempts_note"] = attempts_note
    _save_progress(rd, bundle, checkpoint, session_started, benches, subsets)

    if stopped_early or not _complete(checkpoint, benches, subsets, attempts_override):
        print(f"\nsitting over; resume with: lab eval {ref} --tier {tier} --resume {rd.path.name}", file=sys.stderr)
        return rd
    _finish(rd, bundle, checkpoint, benches, subsets, tel, tier, session_started)
    return rd


# -- pieces ------------------------------------------------------------------------------------
def _allowance_for(model: ModelSpec, cfg: ConfigSpec, tier: str, published_only: bool) -> Allowance:
    if tier == "deep":
        # Wall-clock limited per task: a request is capped only by the context it has left.
        return Allowance(ctx=cfg.params.ctx, limit_s=None)
    speed = latest_speed_bundle(catalog.config_hash(model, cfg), published_only=published_only)
    return from_speed_bundle(speed, limit_s=QUICK_LIMIT_S)


def _open_run(ref: str, model: ModelSpec, cfg: ConfigSpec, tier: str, cli_args: str, resume: str | None):
    if resume:
        rd = find_run(resume)
        bundle = rd.load()
        if bundle.run.kind != "evals" or bundle.run.tier != tier:
            raise EvalError(f"{rd.path.name} is a {bundle.run.kind}/{bundle.run.tier} run, not {tier}-tier evals")
        if bundle.config.config_hash != catalog.config_hash(model, cfg):
            raise EvalError(f"{rd.path.name} belongs to a different config; a resumed run must be the same one")
        if bundle.run.published_at:
            raise EvalError(f"{rd.path.name} is already published; start a new run instead")
        return rd, bundle, True
    weights = catalog.resolve_model_path(model)
    ctx = start_run(model, cfg, "evals", get_engine(model.engine), weights, cli_args, tier=tier)
    return ctx.rd, ctx.bundle, False


def _ctx(rd: RunDir, bundle, t0: float):
    from lab.bench.common import RunContext

    return RunContext(rd, bundle, t0)


def _transcript_writer(rd: RunDir):
    """Every answer is kept on disk: the plan's checkpoint is reading a few before anything is published.

    `messages` is what was sent and `reply` what came back, thinking included, so a grade can be checked
    by reading exactly what the grader read.
    """
    path = rd.raw / "transcripts.jsonl"

    def write(bench: Benchmark, task: Task, attempt: int, record: Attempt, reply: dict[str, Any]) -> None:
        with open(path, "a", buffering=1) as fh:
            fh.write(json.dumps({
                "benchmark": bench.key, "task": task.id, "attempt": attempt,
                "messages": task.messages, "reply": reply, "expected": task.answer,
                "extracted": record.extracted, "passed": record.passed,
                "finish_reason": record.finish_reason, "completion_tokens": record.completion_tokens,
            }) + "\n")

    return write


def _sessions(bundle, session_started: float) -> list[dict[str, Any]]:
    sessions = list(bundle.run.raw.get("sessions") or [])
    sessions.append({"started_at": now_iso(), "seconds": round(time.monotonic() - session_started, 1)})
    return sessions


def _save_progress(rd: RunDir, bundle, checkpoint: Checkpoint, session_started: float, benches, subsets) -> None:
    """Keep a stopped run readable: how far it got, and how much active time it has cost so far."""
    sessions = _sessions(bundle, session_started)
    bundle.run.raw["sessions"] = sessions
    bundle.run.raw["progress"] = {
        b.key: {
            "done": len({a.task for a in counted(checkpoint.for_benchmark(b.key))}),
            "planned": len(subsets[b.key].tasks),
        }
        for b in benches
    }
    bundle.run.duration_s = round(sum(s["seconds"] for s in sessions), 1)
    rd.save(bundle)


def _complete(checkpoint: Checkpoint, benches, subsets, attempts_override: int | None) -> bool:
    for b in benches:
        done = {a.task for a in checkpoint.for_benchmark(b.key)}
        if len(done) < len(subsets[b.key].tasks):
            return False
    return True


def _finish(rd: RunDir, bundle, checkpoint: Checkpoint, benches, subsets, tel, tier: str, session_started: float) -> None:
    results: list[EvalResult] = []
    all_metrics: list[Metric] = []
    version = lab_version()
    for b in benches:
        attempts = checkpoint.for_benchmark(b.key)
        subset = subsets[b.key]
        results.append(score(b, attempts, subset_id=subset.id, harness_version=version, n_planned=len(subset.tasks)))
        all_metrics += metrics(b, attempts, quick=tier == "quick")
    bundle.eval_results = results
    finish_ok(_ctx(rd, bundle, session_started), all_metrics, tel, telemetry_every_s=TELEMETRY_PUBLISH_S)
    # finish_ok timed this sitting; the run's duration is the active time of every sitting it took.
    bundle.run.duration_s = round(sum(s["seconds"] for s in bundle.run.raw.get("sessions") or []), 1)
    rd.save(bundle)
    for r in results:
        print(f"  {r.task}: {r.value:.1%} ± {(r.stderr or 0):.1%} over {r.n_tasks} tasks", file=sys.stderr)


def summarise(rd: RunDir) -> str:
    """One-line-per-benchmark summary of a finished or half-finished run."""
    b = rd.load()
    lines = [f"{rd.path.name}  {b.run.status}  {b.model.slug}/{b.config.slug}  tier={b.run.tier}"]
    for r in b.eval_results:
        lines.append(f"  {r.task:<20} {r.value:>7.1%} ± {(r.stderr or 0):.1%}  n={r.n_tasks}  {r.subset_id}")
    for key, p in (b.run.raw.get("progress") or {}).items():
        if not any(r.task == key for r in b.eval_results):
            lines.append(f"  {key:<20} {p['done']}/{p['planned']} tasks done")
    if b.run.duration_s:
        lines.append(f"  active time: {b.run.duration_s / 3600:.2f} h over {len(b.run.raw.get('sessions') or [])} sitting(s)")
    return "\n".join(lines)
