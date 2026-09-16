"""The six benchmarks, by the keys the site scores them under (web/src/scoring.ts).

Changing a key, a category or a tier here means changing it there: the site groups, weights and ranks
published results by these exact strings.
"""

from dataclasses import dataclass
from typing import Literal

Category = Literal["coding", "agents", "reasoning"]
Tier = Literal["quick", "deep"]


@dataclass(frozen=True)
class Benchmark:
    key: str
    name: str
    category: Category
    tier: Tier
    #: Tasks in the pinned subset: the unit the standard error is over.
    n: int
    metric: str
    #: Samples per task. More than one only where the metric is a mean over attempts.
    attempts: int = 1
    #: Runs the model's own output as code, so it needs a sandbox rather than just a grader.
    sandboxed: bool = False


BENCHMARKS: list[Benchmark] = [
    Benchmark("livecodebench", "LiveCodeBench", "coding", "quick", 100, "pass@1", sandboxed=True),
    Benchmark("swebench_verified", "SWE-bench Verified", "coding", "deep", 30, "resolved", sandboxed=True),
    Benchmark("bfcl", "BFCL", "agents", "quick", 400, "accuracy"),
    Benchmark("terminal_bench", "Terminal-Bench", "agents", "deep", 30, "resolved", sandboxed=True),
    Benchmark("gpqa_diamond", "GPQA Diamond", "reasoning", "quick", 198, "accuracy"),
    Benchmark("aime_2025", "AIME 2025", "reasoning", "quick", 30, "mean accuracy", attempts=4),
]

BY_KEY = {b.key: b for b in BENCHMARKS}


def tier_benchmarks(tier: Tier) -> list[Benchmark]:
    return [b for b in BENCHMARKS if b.tier == tier]


def missing_from_tier(tier: str | None, scored: set[str]) -> list[str]:
    """The tier's benchmarks a run hasn't scored. The site scores a tier as one run, so it must have them all."""
    return [b.key for b in BENCHMARKS if b.tier == tier and b.key not in scored]


def resolve(keys: list[str] | None, tier: Tier) -> list[Benchmark]:
    """The benchmarks a session will run: the whole tier, or the named subset of it."""
    if not keys:
        return tier_benchmarks(tier)
    out = []
    for key in keys:
        if key not in BY_KEY:
            raise KeyError(f"unknown benchmark {key!r}; known: {', '.join(BY_KEY)}")
        b = BY_KEY[key]
        if b.tier != tier:
            raise KeyError(f"{key} is a {b.tier}-tier benchmark, not {tier}")
        out.append(b)
    return out
