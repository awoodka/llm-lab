"""Checkpointing and scoring for an eval session.

A quick tier takes a night or more, so nothing is held only in memory: every attempt is appended to
`raw/attempts.jsonl` as it finishes, and `--resume` replays that file and skips what is already done.
The same records produce the published numbers, so a resumed run scores exactly like an unbroken one.
"""

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lab.evals.registry import Benchmark
from lab.schema import EvalResult, Metric

#: A benchmark with more of its tasks excluded than this can't be published: too much is missing to score.
MAX_EXCLUDED_FRACTION = 0.10


@dataclass
class Attempt:
    """One answer, graded. `excluded` marks an infrastructure failure, which is not a model failure."""

    benchmark: str
    task: str
    attempt: int
    passed: bool = False
    extracted: str | None = None
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    allowance: int = 0
    finish_reason: str | None = None
    error: str | None = None
    excluded: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.benchmark, self.task, self.attempt)


class Checkpoint:
    """Append-only record of finished attempts, read back on resume."""

    def __init__(self, path: Path):
        self.path = path
        self.attempts: dict[tuple[str, str, int], Attempt] = {}
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    a = Attempt(**json.loads(line))
                    self.attempts[a.key] = a
        self._fh = open(path, "a", buffering=1)  # noqa: SIM115 - closed in close()

    def has(self, benchmark: str, task: str, attempt: int) -> bool:
        return (benchmark, task, attempt) in self.attempts

    def add(self, a: Attempt) -> None:
        self.attempts[a.key] = a
        self._fh.write(json.dumps(asdict(a)) + "\n")

    def for_benchmark(self, key: str) -> list[Attempt]:
        return [a for a in self.attempts.values() if a.benchmark == key]

    def close(self) -> None:
        self._fh.close()


def excluded_tasks(attempts: list[Attempt]) -> set[str]:
    """Tasks with an attempt that kept failing for reasons the model isn't responsible for."""
    return {a.task for a in attempts if a.excluded}


def counted(attempts: list[Attempt]) -> list[Attempt]:
    """The attempts a score is built from: every attempt at a task that wasn't excluded.

    Worked out from the records each time rather than marked on them, so a resumed run, which reads the
    checkpoint back from disk, excludes exactly what an unbroken run would.
    """
    excluded = excluded_tasks(attempts)
    return [a for a in attempts if a.task not in excluded]


def task_scores(attempts: list[Attempt]) -> dict[str, float]:
    """Per task: the share of its attempts that passed. Excluded tasks don't count either way."""
    by_task: dict[str, list[Attempt]] = {}
    for a in counted(attempts):
        by_task.setdefault(a.task, []).append(a)
    return {task: statistics.fmean(a.passed for a in xs) for task, xs in by_task.items() if xs}


def score(bench: Benchmark, attempts: list[Attempt], *, subset_id: str, harness_version: str, n_planned: int, harness: str = "lab") -> EvalResult:
    """The benchmark's result, with its standard error taken over tasks."""
    scores = task_scores(attempts)
    if not scores:
        raise ValueError(f"{bench.key}: nothing to score")
    values = list(scores.values())
    excluded = n_planned - len(values)
    if excluded > MAX_EXCLUDED_FRACTION * n_planned:
        raise ValueError(
            f"{bench.key}: {excluded} of {n_planned} tasks were excluded (more than "
            f"{MAX_EXCLUDED_FRACTION:.0%}); fix the cause and resume rather than publish a partial benchmark"
        )
    value = statistics.fmean(values)
    stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    used = counted(attempts)
    return EvalResult(
        task=bench.key,
        metric=bench.metric,
        value=value,
        stderr=stderr,
        n_samples=len(used),
        n_tasks=len(values),
        attempts_per_task=round(len(used) / len(values)) if values else None,
        harness=harness,
        harness_version=harness_version,
        subset_id=subset_id,
    )


def metrics(bench: Benchmark, attempts: list[Attempt], *, quick: bool) -> list[Metric]:
    """What the run cost per task, published under the benchmark's key so the site can show it."""
    used = counted(attempts)
    tasks = {a.task for a in used}
    n = len(tasks) or 1
    out = [
        Metric(key="eval_seconds_per_task", method=bench.key, value=round(sum(a.seconds for a in used) / n, 2), unit="s", n=len(tasks)),
        Metric(key="eval_tokens_per_task", method=bench.key, value=round(sum(a.completion_tokens for a in used) / n, 1), unit="tokens", n=len(tasks)),
        Metric(key="eval_length_stops", method=bench.key, value=sum(a.finish_reason == "length" for a in used), unit="count"),
        Metric(key="eval_excluded", method=bench.key, value=len(excluded_tasks(attempts)), unit="count"),
    ]
    if quick and used:
        out.append(
            Metric(key="eval_allowance_tokens", method=bench.key, value=round(statistics.fmean(a.allowance for a in used), 1), unit="tokens")
        )
    return out


def progress(bench: Benchmark, attempts: list[Attempt], n_planned: int) -> str:
    done = len({a.task for a in counted(attempts)})
    scores = task_scores(attempts)
    solved = statistics.fmean(scores.values()) if scores else 0.0
    return f"{bench.key}: {done}/{n_planned} tasks, {solved:.1%} solved so far"
