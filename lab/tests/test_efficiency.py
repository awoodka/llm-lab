"""lab runs thinking / lab runs markers: the numbers, the pairing, and that no GPQA text ever gets out."""

import json

import pytest

from lab.evals import efficiency

CANARY = "ZQXJ-canary-benzylidene-7731"   # stands in for GPQA question text; must never be printed
WORDS = efficiency.DEFAULT_WORDS


def write_run(path, attempts, transcripts):
    raw = path / "raw"
    raw.mkdir(parents=True)
    (raw / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts))
    (raw / "transcripts.jsonl").write_text("".join(json.dumps(t) + "\n" for t in transcripts))
    return path


def attempt(bench, task, n, passed, **kw):
    return {"benchmark": bench, "task": task, "attempt": n, "passed": passed, "extracted": None, "seconds": 10.0,
            "prompt_tokens": 50, "completion_tokens": 1000, "allowance": 40000, "finish_reason": "stop",
            "error": None, "excluded": False, "detail": {}, **kw}


def gpqa_transcript(task, n, reasoning):
    return {"benchmark": "gpqa_diamond", "task": task, "attempt": n,
            "messages": [{"role": "user", "content": f"Question about {CANARY}. Options (A) (B) (C) (D)"}],
            "reply": {"reasoning": reasoning, "content": f"Answer: B because {CANARY}"}, "expected": "B",
            "extracted": "B", "passed": True, "finish_reason": "stop", "completion_tokens": 1000}


def lcb_transcript(task, n, reasoning, reasoning_tokens):
    return {"benchmark": "livecodebench", "task": task, "attempt": n, "messages": [],
            "reply": {"transcript": {"messages": [{"role": "user", "content": "code it"},
                                                  {"role": "assistant", "content": "def f(): ...", "reasoning": reasoning}],
                                     "usage": {"completion_tokens": 900, "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
                                     "finish_reason": "stop"}},
            "expected": "", "extracted": None, "passed": True, "finish_reason": "stop", "completion_tokens": 900}


@pytest.fixture
def runs(tmp_path):
    base = write_run(tmp_path / "20260917T230515_base", [
        attempt("gpqa_diamond", "g1", 1, True), attempt("gpqa_diamond", "g2", 1, False, finish_reason="length"),
        attempt("livecodebench", "l1", 1, True), attempt("livecodebench", "l2", 1, False, finish_reason="length"),
    ], [
        gpqa_transcript("g1", 1, f"Wait. {CANARY} Hmm, so B."), gpqa_transcript("g2", 1, f"{CANARY} Wait Wait Wait"),
        lcb_transcript("l1", 1, "Wait. Let's code. Actually fine.", 800), lcb_transcript("l2", 1, "Maybe. Wait. Maybe.", 30000),
    ])
    lever = write_run(tmp_path / "20260930T220000_lever", [
        attempt("gpqa_diamond", "g1", 1, True, reasoning_tokens=500), attempt("gpqa_diamond", "g2", 1, True, reasoning_tokens=700),
        attempt("livecodebench", "l1", 1, False, reasoning_tokens=600), attempt("livecodebench", "l2", 1, True, reasoning_tokens=9000),
    ], [
        gpqa_transcript("g1", 1, f"So B. {CANARY}"), gpqa_transcript("g2", 1, f"Hmm. {CANARY}"),
        lcb_transcript("l1", 1, "Let's code.", 600), lcb_transcript("l2", 1, "Code. Done.", 9000),
    ])
    return base, lever


def test_rows_take_reasoning_tokens_from_the_best_source_they_have(runs):
    base, lever = runs
    b = {(r.benchmark, r.task): r for r in efficiency.load_rows(base)}
    assert (b["livecodebench", "l1"].reasoning_tokens, b["livecodebench", "l1"].reasoning_source) == (800, "transcript")
    assert (b["gpqa_diamond", "g1"].reasoning_tokens, b["gpqa_diamond", "g1"].reasoning_source) == (1000, "completion")
    lv = {(r.benchmark, r.task): r for r in efficiency.load_rows(lever)}
    assert (lv["livecodebench", "l2"].reasoning_tokens, lv["livecodebench", "l2"].reasoning_source) == (9000, "server")
    assert b["livecodebench", "l2"].markers == {"Wait": 1, "Hmm": 0, "Alternatively": 0, "Actually": 0, "Maybe": 2}


def test_the_report_pairs_tasks_and_counts_new_failures(runs):
    base, lever = runs
    b_rows, l_rows = efficiency.load_rows(base), efficiency.load_rows(lever)
    records = efficiency.thinking_report({"lever": l_rows}, ("base", b_rows), WORDS)
    lcb = next(r for r in records if r["run"] == "lever" and r["benchmark"] == "livecodebench")
    assert lcb["accuracy"] == 0.5 and lcb["vs_baseline_accuracy_delta"] == 0.0
    assert lcb["vs_baseline_net_new_failures"] == 0, "l1 lost, l2 won"
    assert lcb["vs_baseline_length_stops_delta"] == -1
    assert lcb["vs_baseline_reasoning_tokens_delta"] == pytest.approx(((600 - 800) + (9000 - 30000)) / 2)
    assert lcb["vs_baseline_reasoning_tokens_ratio"] == pytest.approx(9600 / 30800)
    gpqa = next(r for r in records if r["run"] == "lever" and r["benchmark"] == "gpqa_diamond")
    assert gpqa["vs_baseline_accuracy_delta"] == 0.5 and gpqa["vs_baseline_net_new_failures"] == -1


def test_no_gpqa_text_reaches_any_output(runs):
    base, lever = runs
    records = efficiency.thinking_report({"lever": efficiency.load_rows(lever)}, ("base", efficiency.load_rows(base)), WORDS)
    for fmt in ("table", "markdown", "csv"):
        out = efficiency.render(records, WORDS, fmt)
        assert CANARY not in out and "benzylidene" not in out, fmt
        assert "gpqa_diamond" in out, "the aggregates are there"
    markers = efficiency.markers_report(base, None)
    assert CANARY not in markers and "gpqa" not in markers, "GPQA is skipped by default"
    with pytest.raises(ValueError, match="gpqa_diamond"):
        efficiency.markers_report(base, ["gpqa_diamond", "livecodebench"])


def test_markers_split_finished_from_cut_off_attempts(runs):
    base, _ = runs
    out = efficiency.markers_report(base, ["livecodebench"])
    assert "livecodebench (cut off): 1 attempts" in out and "livecodebench (finished): 1 attempts" in out


def test_the_aime_post_answer_share_is_a_bounded_heuristic():
    assert efficiency.post_answer_share("so 204 is it and we check 204 again", "204") == pytest.approx(7 / 9)
    assert efficiency.post_answer_share("the answer is 7", "7") is None, "single digits are everywhere"
    assert efficiency.post_answer_share("never said", "204") is None
    assert efficiency.post_answer_share("1204 is not it", "204") is None, "only a standalone number counts"
