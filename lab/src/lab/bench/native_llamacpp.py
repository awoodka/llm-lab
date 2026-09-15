"""Engine-native speed benchmark via llama-bench (llama.cpp only).

Power per test: llama-bench prints one JSONL row as each test finishes, so each row's window is
[previous row, this row]. Mean GPU power over the busy samples in that window gives W, and
tokens/J = (t/s) / W. VRAM here reflects llama-bench's KV sizing (p+n+d), not the server's ctx —
the HTTP bench measures VRAM at real context.
"""

import json
import os
import shlex
import signal
import statistics
import subprocess

from lab import catalog
from lab.bench.common import finish_failed, finish_ok, start_run
from lab.engines.base import get_engine
from lab.gpu.hosted import exclusive_gpu
from lab.schema import Metric
from lab.store import RunDir
from lab.telemetry import Telemetry

BUSY_UTIL = 50.0
MIN_BUSY_SAMPLES = 3


def _run_llama_bench(argv: list[str], rd: RunDir, tel: Telemetry) -> tuple[list[dict], list[float]]:
    rows: list[dict] = []
    marks: list[float] = []
    with open(rd.raw / "llama-bench.jsonl", "w") as raw, open(rd.logs / "llama-bench.log", "w") as log:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)
        tel.watch_pid = proc.pid
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                raw.write(line)
                raw.flush()
                if line.lstrip().startswith("{"):
                    rows.append(json.loads(line))
                    marks.append(tel.elapsed())
            rc = proc.wait()
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            raise
    if rc != 0:
        tail = (rd.logs / "llama-bench.log").read_text().splitlines()[-15:]
        raise RuntimeError(f"llama-bench exited {rc}:\n" + "\n".join(tail))
    return rows, marks


def rows_to_metrics(rows: list[dict], marks: list[float], samples: list[dict[str, float]]) -> list[Metric]:
    metrics: list[Metric] = []
    prev = 0.0
    for row, mark in zip(rows, marks, strict=True):
        is_pp = row["n_gen"] == 0
        dims = dict(n_prompt=row["n_prompt"], n_gen=row["n_gen"], depth=row["n_depth"])
        metrics.append(
            Metric(
                key="pp_tps" if is_pp else "tg_tps",
                method="llama-bench",
                value=row["avg_ts"],
                stddev=row["stddev_ts"],
                n=len(row["samples_ts"]),
                unit="t/s",
                samples=row["samples_ts"],
                **dims,
            )
        )
        busy = [s["power_w"] for s in samples if prev <= s["t"] <= mark and s["util"] >= BUSY_UTIL]
        if len(busy) >= MIN_BUSY_SAMPLES:
            watts = statistics.fmean(busy)
            metrics.append(Metric(key="gpu_w_avg", method="llama-bench", value=round(watts, 1), n=len(busy), unit="W", **dims))
            metrics.append(
                Metric(key="tokens_per_joule", method="llama-bench", value=round(row["avg_ts"] / watts, 3), unit="tok/J", **dims)
            )
        prev = mark
    return metrics


def run_speed(ref: str, *, wait: bool, keep_paused: bool, reps: int | None, cli_args: str) -> RunDir:
    model, cfg = catalog.load_config(ref)
    if model.engine != "llama.cpp":
        raise ValueError(f"native speed bench is llama.cpp-only; {model.slug} uses {model.engine}")
    if reps:
        cfg.bench.reps = reps
    engine = get_engine(model.engine)
    weights = catalog.resolve_model_path(model)
    ctx = start_run(model, cfg, "speed", engine, weights, cli_args)
    argv = engine.bench_argv(weights, cfg)
    (ctx.rd.path / "commands.json").write_text(json.dumps({"llama-bench": shlex.join(argv)}, indent=2))
    tel: Telemetry | None = None
    try:
        with exclusive_gpu(f"bench speed {ref}", wait=wait, keep_paused=keep_paused):
            tel = Telemetry().start()
            try:
                tel.phase("llama-bench")
                rows, marks = _run_llama_bench(argv, ctx.rd, tel)
            finally:
                tel.stop()
        summary = tel.summary()
        metrics = rows_to_metrics(rows, marks, tel.samples)
        max_depth = max(r["n_depth"] for r in rows)
        metrics += [
            Metric(key="vram_peak_mb", method="llama-bench", depth=max_depth, unit="MiB",
                   value=round(summary["vram_peak_mb"] - summary["vram_baseline_mb"])),
            Metric(key="ram_peak_mb", method="llama-bench", depth=max_depth, unit="MiB",
                   value=round(summary["ram_peak_mb"])),
            Metric(key="proc_rss_peak_mb", method="llama-bench", depth=max_depth, unit="MiB",
                   value=round(summary["proc_rss_peak_mb"])),
        ]
        if rows and rows[0].get("build_commit"):
            ctx.bundle.engine_build.extra["bench_build_commit"] = rows[0]["build_commit"]
        finish_ok(ctx, metrics, tel)
        return ctx.rd
    except BaseException as e:
        finish_failed(ctx, e, tel)
        raise
