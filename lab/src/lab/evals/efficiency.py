"""How a model spent its thinking in an eval run: `lab runs thinking` and `lab runs markers`.

Both read a run's raw/attempts.jsonl and raw/transcripts.jsonl and print aggregates and task ids only,
never message text: GPQA's authors ask that its questions stay off the web, and nothing printed here should
make it easy to paste one. `markers` refuses GPQA outright, since even a list of the words a model opens its
sentences with can carry a question's terms. tests/test_efficiency.py plants a canary in GPQA messages and
reasoning and checks it never reaches any output.

Reasoning tokens come from, in order: the attempt's own count (the server's usage, recorded since the
runner kept it), the transcript's usage (LiveCodeBench and BFCL, whose harnesses keep whole conversations),
or else the completion tokens, labelled as such. Marker words are counted case-sensitively on word
boundaries, per 1,000 whitespace-separated words of reasoning text.
"""

import csv
import io
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_WORDS = ("Wait", "Hmm", "Alternatively", "Actually", "Maybe")
#: The words the thinking-levers fork penalizes by default (THINK_PENALTY_WORDS).
PENALTY_SET = ("Wait", "Hmm", "Alternatively")
#: Benchmarks whose text may not leave the machine in any form, even as word lists.
NO_TEXT_BENCHMARKS = ("gpqa_diamond",)
BOOTSTRAP_SAMPLES = 2000
SENTENCE_START = re.compile(r"(?:^|(?<=[.!?:])\s+|\n+)\s*([A-Z][A-Za-z'-]*)")


@dataclass
class Row:
    """One attempt, as far as the run recorded it."""

    benchmark: str
    task: str
    attempt: int
    passed: bool
    excluded: bool
    finish_reason: str | None
    seconds: float
    completion_tokens: int
    reasoning_tokens: int
    reasoning_source: str  # "server", "transcript" or "completion"
    words: int | None = None  # None: the transcript kept no reasoning
    markers: dict[str, int] = field(default_factory=dict)
    post_answer_share: float | None = None


def reasoning_text(reply: dict[str, Any]) -> str | None:
    """The thinking a transcript kept, or None if it kept none (AIME/GPQA runs before the runner fix)."""
    transcript = reply.get("transcript")
    if isinstance(transcript, dict):   # a harness's whole conversation (LiveCodeBench, BFCL)
        msgs = [m for m in transcript.get("messages") or [] if m.get("role") == "assistant"]
    elif isinstance(transcript, list):   # request/reply pairs, one per turn
        msgs = [((turn.get("reply") or {}).get("choices") or [{}])[0].get("message") or {} for turn in transcript]
    else:   # a single answer (AIME, GPQA)
        return reply.get("reasoning") or reply.get("reasoning_content") or None
    parts = [m.get("reasoning") or m.get("reasoning_content") or "" for m in msgs]
    return "\n".join(p for p in parts if p) or None


def _transcript_reasoning_tokens(reply: dict[str, Any]) -> int | None:
    transcript = reply.get("transcript")
    if not isinstance(transcript, dict):
        return None
    usage = transcript.get("usage")
    if usage:
        return (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    requests = transcript.get("requests") or []
    counts = [((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") for r in requests]
    return sum(counts) if counts and all(c is not None for c in counts) else None


def post_answer_share(text: str, expected: str | None) -> float | None:
    """AIME only, a heuristic: the share of reasoning words after the first standalone mention of the
    expected answer. Answers below 10 are skipped, since a bare digit turns up everywhere."""
    if not text or not expected or not expected.isdigit() or int(expected) < 10:
        return None
    m = re.search(rf"(?<![\d.]){re.escape(expected)}(?!\d)", text)
    if not m:
        return None
    total = len(text.split())
    return len(text[m.end():].split()) / total if total else None


def load_rows(run_dir: Path, words: tuple[str, ...] = DEFAULT_WORDS) -> list[Row]:
    raw = run_dir / "raw"
    transcripts: dict[tuple[str, str, int], dict[str, Any]] = {}
    if (raw / "transcripts.jsonl").exists():
        for line in open(raw / "transcripts.jsonl"):
            if line.strip():
                t = json.loads(line)
                transcripts[(t["benchmark"], t["task"], t["attempt"])] = t
    patterns = {w: re.compile(rf"\b{re.escape(w)}\b") for w in words}
    rows = []
    for line in open(raw / "attempts.jsonl"):
        if not line.strip():
            continue
        a = json.loads(line)
        t = transcripts.get((a["benchmark"], a["task"], a["attempt"])) or {}
        reply = t.get("reply") or {}
        if a.get("reasoning_tokens") is not None:
            tokens, source = a["reasoning_tokens"], "server"
        elif (tt := _transcript_reasoning_tokens(reply)) is not None:
            tokens, source = tt, "transcript"
        else:
            tokens, source = a.get("completion_tokens") or 0, "completion"
        text = reasoning_text(reply)
        row = Row(benchmark=a["benchmark"], task=a["task"], attempt=a["attempt"], passed=bool(a.get("passed")),
                  excluded=bool(a.get("excluded")), finish_reason=a.get("finish_reason"), seconds=a.get("seconds") or 0.0,
                  completion_tokens=a.get("completion_tokens") or 0, reasoning_tokens=tokens, reasoning_source=source)
        if text is not None:
            row.words = len(text.split())
            row.markers = {w: len(p.findall(text)) for w, p in patterns.items()}
            if row.benchmark == "aime_2025" and row.passed:
                row.post_answer_share = post_answer_share(text, t.get("expected"))
        rows.append(row)
    return rows


def counted(rows: list[Row]) -> list[Row]:
    """Every attempt at a task that wasn't excluded (as lab.evals.session.counted)."""
    excluded = {r.task for r in rows if r.excluded}
    return [r for r in rows if r.task not in excluded]


def task_means(rows: list[Row], value) -> dict[str, float]:
    by_task: dict[str, list[float]] = {}
    for r in rows:
        by_task.setdefault(r.task, []).append(float(value(r)))
    return {t: statistics.fmean(v) for t, v in by_task.items()}


def bootstrap_ci(deltas: list[float], seed: int = 0) -> tuple[float, float] | None:
    if len(deltas) < 2:
        return None
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(statistics.fmean(rng.choices(deltas, k=n)) for _ in range(BOOTSTRAP_SAMPLES))
    return means[int(0.025 * BOOTSTRAP_SAMPLES)], means[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


def summarize(rows: list[Row], words: tuple[str, ...]) -> dict[str, Any]:
    used = counted(rows)
    scores = task_means(used, lambda r: r.passed)
    values = list(scores.values())
    texts = [r for r in used if r.words is not None]
    n_words = sum(r.words or 0 for r in texts)
    shares = [r.post_answer_share for r in used if r.post_answer_share is not None]
    sources = Counter(r.reasoning_source for r in used)
    seconds = sum(r.seconds for r in used)
    out = {
        "tasks": len(values),
        "attempts": len(used),
        "accuracy": statistics.fmean(values) if values else None,
        "stderr": statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None,
        "reasoning_tokens_mean": statistics.fmean(r.reasoning_tokens for r in used) if used else None,
        "reasoning_tokens_median": statistics.median(r.reasoning_tokens for r in used) if used else None,
        "reasoning_source": "+".join(sorted(sources)) if sources else None,
        "length_stops": sum(r.finish_reason == "length" for r in used),
        "length_stop_share": sum(r.finish_reason == "length" for r in used) / len(used) if used else None,
        # Completion tokens over wall seconds, prompts included: comparable between runs of one config.
        "tok_s": sum(r.completion_tokens for r in used) / seconds if seconds else None,
        "reasoning_kept": len(texts),
        "markers_per_1k_words": {w: 1000 * sum(r.markers.get(w, 0) for r in texts) / n_words for w in words} if n_words else None,
        "penalty_set_per_1k_words": (1000 * sum(r.markers.get(w, 0) for r in texts for w in PENALTY_SET if w in words) / n_words
                                     if n_words else None),
        "post_answer_share": statistics.fmean(shares) if shares else None,
        "post_answer_n": len(shares),
    }
    return out


def paired(rows: list[Row], base: list[Row]) -> dict[str, Any]:
    """Task-paired differences against a baseline, over the tasks both counted."""
    run_used, base_used = counted(rows), counted(base)
    acc, base_acc = task_means(run_used, lambda r: r.passed), task_means(base_used, lambda r: r.passed)
    tok, base_tok = task_means(run_used, lambda r: r.reasoning_tokens), task_means(base_used, lambda r: r.reasoning_tokens)
    common = sorted(set(acc) & set(base_acc))
    d_acc = [acc[t] - base_acc[t] for t in common]
    d_tok = [tok[t] - base_tok[t] for t in common]
    base_by = {(r.task, r.attempt): r for r in base_used}
    pairs = [(r, base_by[(r.task, r.attempt)]) for r in run_used if (r.task, r.attempt) in base_by]
    return {
        "common_tasks": len(common),
        "accuracy_delta": statistics.fmean(d_acc) if d_acc else None,
        "accuracy_delta_ci": bootstrap_ci(d_acc),
        "reasoning_tokens_delta": statistics.fmean(d_tok) if d_tok else None,
        "reasoning_tokens_delta_ci": bootstrap_ci(d_tok, seed=1),
        "reasoning_tokens_ratio": (sum(tok[t] for t in common) / sum(base_tok[t] for t in common)
                                   if common and sum(base_tok[t] for t in common) else None),
        "length_stops_delta": sum(r.finish_reason == "length" for r, _ in pairs) - sum(b.finish_reason == "length" for _, b in pairs),
        # Attempts the baseline passed and this run failed, net of the reverse: the pilot's "disaster" count.
        "net_new_failures": sum((not r.passed) and b.passed for r, b in pairs) - sum(r.passed and not b.passed for r, b in pairs),
        "paired_attempts": len(pairs),
    }


def thinking_report(runs: dict[str, list[Row]], baseline: tuple[str, list[Row]] | None, words: tuple[str, ...],
                    benchmarks: list[str] | None = None) -> list[dict[str, Any]]:
    """One record per (run, benchmark): the summary, plus paired deltas when a baseline is given."""
    out = []
    labelled = ([baseline] if baseline else []) + list(runs.items())
    for label, rows in labelled:
        for bench in sorted({r.benchmark for r in rows}):
            if benchmarks and bench not in benchmarks:
                continue
            bench_rows = [r for r in rows if r.benchmark == bench]
            rec = {"run": label, "benchmark": bench, **summarize(bench_rows, words)}
            if baseline and label != baseline[0]:
                rec.update({f"vs_baseline_{k}": v for k, v in paired(bench_rows, [r for r in baseline[1] if r.benchmark == bench]).items()})
            out.append(rec)
    return out


def _fmt(v: Any, pct: bool = False, digits: int = 1) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.{digits}f}"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render(records: list[dict[str, Any]], words: tuple[str, ...], fmt: str) -> str:
    if fmt == "csv":
        flat = []
        for rec in records:
            row = {k: v for k, v in rec.items() if k != "markers_per_1k_words"}
            for w in words:
                row[f"{w}_per_1k_words"] = (rec.get("markers_per_1k_words") or {}).get(w)
            flat.append({k: (json.dumps(v) if isinstance(v, (tuple, list)) else v) for k, v in row.items()})
        keys = list(dict.fromkeys(k for row in flat for k in row))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
        return buf.getvalue()

    has_base = any("vs_baseline_common_tasks" in r for r in records)
    head = ["run", "benchmark", "accuracy", "tasks", "reasoning tokens mean / median", "length stops",
            "markers /1k words (" + "+".join(PENALTY_SET) + ")", *[f"{w} /1k" for w in words], "tok/s", "post-answer share"]
    if has_base:
        head += ["Δ accuracy [95% CI]", "Δ reasoning tokens [95% CI]", "tokens vs base", "Δ length stops", "net new failures"]
    lines = []
    for rec in records:
        acc = "-" if rec["accuracy"] is None else f"{100 * rec['accuracy']:.1f} ± {100 * (rec['stderr'] or 0):.1f}"
        cells = [rec["run"], rec["benchmark"], acc, str(rec["tasks"]),
                 f"{_fmt(rec['reasoning_tokens_mean'], digits=0)} / {_fmt(rec['reasoning_tokens_median'], digits=0)} ({rec['reasoning_source']})",
                 f"{rec['length_stops']} ({_fmt(rec['length_stop_share'], pct=True)}%)",
                 _fmt(rec["penalty_set_per_1k_words"], digits=2),
                 *[_fmt((rec.get("markers_per_1k_words") or {}).get(w), digits=2) for w in words],
                 _fmt(rec["tok_s"]),
                 "-" if rec["post_answer_share"] is None else f"{100 * rec['post_answer_share']:.0f}% (n={rec['post_answer_n']})"]
        if has_base:
            if "vs_baseline_common_tasks" in rec:
                d, ci = rec["vs_baseline_accuracy_delta"], rec["vs_baseline_accuracy_delta_ci"]
                acc_d = "-" if d is None else f"{100 * d:+.1f}" + ("" if not ci else f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]")
                t, tci = rec["vs_baseline_reasoning_tokens_delta"], rec["vs_baseline_reasoning_tokens_delta_ci"]
                tok_d = "-" if t is None else f"{t:+.0f}" + ("" if not tci else f" [{tci[0]:+.0f}, {tci[1]:+.0f}]")
                ratio = rec["vs_baseline_reasoning_tokens_ratio"]
                cells += [acc_d, tok_d, "-" if ratio is None else f"{100 * (ratio - 1):+.1f}%",
                          f"{rec['vs_baseline_length_stops_delta']:+d}", f"{rec['vs_baseline_net_new_failures']:+d}"]
            else:
                cells += ["baseline"] + ["-"] * 4
        lines.append(cells)
    if fmt == "markdown":
        out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        out += ["| " + " | ".join(c) + " |" for c in lines]
        return "\n".join(out) + "\n"
    widths = [max(len(head[i]), *(len(row[i]) for row in lines)) for i in range(len(head))]
    fmt_row = lambda cells: "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))  # noqa: E731
    return "\n".join([fmt_row(head), fmt_row(["-" * w for w in widths]), *map(fmt_row, lines)]) + "\n"


def markers_report(run_dir: Path, benchmarks: list[str] | None, top: int = 30) -> str:
    """Which words the model opens its sentences with, per 1,000 reasoning words: over all counted attempts,
    and split into attempts that ran out of room (finish_reason length) and the ones that finished, pooled
    and as a per-attempt mean. The candidates for THINK_PENALTY_WORDS come from here."""
    raw = run_dir / "raw"
    wanted = set(benchmarks or [])
    refused = sorted(wanted & set(NO_TEXT_BENCHMARKS))
    if refused:
        raise ValueError(f"{', '.join(refused)}: no text from this benchmark leaves the machine, not even word lists")
    transcripts = {}
    for line in open(raw / "transcripts.jsonl"):
        if line.strip():
            t = json.loads(line)
            transcripts[(t["benchmark"], t["task"], t["attempt"])] = t
    groups: dict[tuple[str, str], list[tuple[Counter, int]]] = {}
    # The attempt records decide what finished and what was excluded; transcripts only carry the text.
    records = {(a["benchmark"], a["task"], a["attempt"]): a
               for a in map(json.loads, filter(str.strip, open(raw / "attempts.jsonl")))}
    excluded = {(b, task) for (b, task, _), a in records.items() if a.get("excluded")}
    for key, t in transcripts.items():
        bench = t["benchmark"]
        if bench in NO_TEXT_BENCHMARKS or (wanted and bench not in wanted) or (bench, t["task"]) in excluded:
            continue
        text = reasoning_text(t.get("reply") or {})
        if not text or key not in records:
            continue
        starts = Counter(SENTENCE_START.findall(text))
        n = len(text.split())
        for label in ("all", "cut off" if records[key].get("finish_reason") == "length" else "finished"):
            groups.setdefault((bench, label), []).append((starts, n))
    lines = []
    for (bench, label), items in sorted(groups.items()):
        total = Counter()
        for c, _ in items:
            total.update(c)
        n_words = sum(n for _, n in items)
        lines.append(f"{bench} ({label}): {len(items)} attempts, {n_words:,} reasoning words; sentence starts per 1k words, pooled:")
        lines.append("  " + ", ".join(f"{w} {1000 * c / n_words:.2f}" for w, c in total.most_common(top)))
        cand = [w for w in DEFAULT_WORDS]
        per_attempt = {w: statistics.fmean(1000 * c[w] / n for c, n in items if n) for w in cand}
        lines.append("  candidates, pooled / per-attempt mean: "
                     + ", ".join(f"{w} {1000 * total[w] / n_words:.2f} / {per_attempt[w]:.2f}" for w in cand))
    return "\n".join(lines) + "\n"
