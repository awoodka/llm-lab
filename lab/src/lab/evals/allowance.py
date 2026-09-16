"""The thinking allowance: how many tokens a config may spend on one task.

Speed counts only here. A faster config can think longer inside the same time limit, and a model that
finishes early gains nothing from being quicker. Both sides of the wire compute this: the lab enforces
it per request, the site shows it per config, and both pass the same vectors (web/src/scoring.ts).
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any

from lab.schema import RunBundle

QUICK_LIMIT_S = 300.0
DEEP_LIMIT_S = 1200.0
#: The prompt size the site quotes one allowance for, so configs are comparable at a glance.
HEADLINE_PROMPT_TOKENS = 8000

#: Speed keys, in the order they're preferred. `decode_tps`/`prefill_tps` come from the HTTP chat benchmark.
TG_KEYS = ("tg_tps", "decode_tps")
PP_KEYS = ("pp_tps", "prefill_tps")


class NoSpeedRun(RuntimeError):
    """A config can't be evaluated until `lab bench` has measured it."""


@dataclass(frozen=True, order=True)
class SpeedPoint:
    depth: int
    tps: float


def tg_at_depth(points: list[SpeedPoint], depth: int) -> float:
    """Generation speed at a context depth, from the depths a speed run actually measured.

    At or below the shallowest, that measurement; between two, linear; past the deepest, the last
    segment's slope continued, never below 1 t/s.
    """
    pts = sorted(points)
    if not pts:
        return 0.0
    if len(pts) == 1 or depth <= pts[0].depth:
        return pts[0].tps
    i = bisect_left([p.depth for p in pts], depth)
    if i < len(pts):
        a, b = pts[i - 1], pts[i]
        return a.tps + (depth - a.depth) / (b.depth - a.depth) * (b.tps - a.tps)
    a, b = pts[-2], pts[-1]
    slope = (b.tps - a.tps) / (b.depth - a.depth)
    return max(1.0, b.tps + (depth - b.depth) * slope)


@dataclass(frozen=True)
class Allowance:
    """How each request of one session is sized, and the record of where those numbers came from.

    `limit_s` is None for the deep tier, which is wall-clock limited per task: there a request is
    capped only by the context it has left.
    """

    ctx: int
    pp0: float = 0.0
    tg: list[SpeedPoint] = field(default_factory=list)
    limit_s: float | None = None
    speed_run_id: str | None = None

    def for_prompt(self, prompt_tokens: int, *, spent_s: float = 0.0, cached_tokens: int = 0) -> int:
        """Tokens this request may generate.

        A task that takes several requests (an agent, a multi-turn conversation) shares one time limit:
        `spent_s` is what its earlier requests cost, and `cached_tokens` is the part of this prompt the
        server already holds, which costs nothing to process again. For a single request both are 0,
        which is the plan's formula exactly.
        """
        context_left = self.ctx - prompt_tokens
        if context_left <= 0:
            return 0
        if self.limit_s is None:
            return context_left
        if self.pp0 <= 0 or not self.tg:
            return 0
        seconds_left = self.limit_s - spent_s - max(0, prompt_tokens - cached_tokens) / self.pp0
        if seconds_left <= 0:
            return 0
        return max(0, min(int(seconds_left * tg_at_depth(self.tg, prompt_tokens)), context_left))

    def cost_s(self, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
        """What one answered request took out of its task's limit, at the measured speeds."""
        if self.limit_s is None or self.pp0 <= 0 or not self.tg:
            return 0.0
        prompt_s = max(0, prompt_tokens - cached_tokens) / self.pp0
        return prompt_s + completion_tokens / tg_at_depth(self.tg, prompt_tokens)

    def as_raw(self) -> dict[str, Any]:
        """What the run publishes about its own sizing, so a result can be re-derived later."""
        return {
            "speed_run_id": self.speed_run_id,
            "pp0": self.pp0,
            "tg": [{"depth": p.depth, "tps": p.tps} for p in sorted(self.tg)],
            "ctx": self.ctx,
            "limit_s": self.limit_s,
            "headline_tokens": self.for_prompt(HEADLINE_PROMPT_TOKENS),
        }


def _points(bundle: RunBundle, keys: tuple[str, ...]) -> list[SpeedPoint]:
    """One point per depth, taking the first key that measured it (llama-bench before HTTP)."""
    by_depth: dict[int, SpeedPoint] = {}
    for key in keys:
        for m in bundle.metrics:
            if m.key == key and m.depth not in by_depth:
                by_depth[m.depth] = SpeedPoint(m.depth, m.value)
    return sorted(by_depth.values())


def from_speed_bundle(bundle: RunBundle, limit_s: float | None) -> Allowance:
    """Read a speed run's depth ladder. The run's own config supplies the context length."""
    tg = _points(bundle, TG_KEYS)
    pp = _points(bundle, PP_KEYS)
    ctx = int(bundle.config.params.get("ctx") or 0)
    if not ctx:
        raise NoSpeedRun(f"run {bundle.run.id} has no ctx in its config params")
    if limit_s is not None and (not tg or not pp):
        raise NoSpeedRun(f"run {bundle.run.id} has no generation/prompt speed to size an allowance with")
    return Allowance(ctx=ctx, pp0=pp[0].tps if pp else 0.0, tg=tg, limit_s=limit_s, speed_run_id=bundle.run.id)


def latest_speed_bundle(config_hash: str, *, published_only: bool = True) -> RunBundle:
    """The speed run an allowance may be built from: newest, not throttled, for this exact config.

    Published by default, because the site shows the same number next to the result. A throttled run
    is never used: its speeds are low, so every allowance built from it would be short.
    """
    from lab.store import list_runs

    best: RunBundle | None = None
    for rd in list_runs():
        b = rd.load()
        if b.run.kind != "speed" or b.run.status != "ok" or b.run.throttled:
            continue
        if b.config.config_hash != config_hash:
            continue
        if published_only and not b.run.published_at:
            continue
        if best is None or b.run.started_at > best.run.started_at:
            best = b
    if best is None:
        what = "published, non-throttled" if published_only else "non-throttled"
        raise NoSpeedRun(f"no {what} speed run for config {config_hash[:12]}; run `lab bench` (and publish it) first")
    return best
