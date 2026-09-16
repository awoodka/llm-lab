"""Scoring, checkpointing and the rules that decide when a tier may be published."""

import json

import pytest

from lab.evals import registry
from lab.evals.allowance import Allowance
from lab.evals.drivers.aime import Aime2025, extract_answer
from lab.evals.drivers.base import Subset, Task, check_against_pin
from lab.evals.registry import BY_KEY
from lab.evals.runner import parse_stop_at
from lab.evals.session import Attempt, Checkpoint, metrics, score, task_scores

AIME = BY_KEY["aime_2025"]
GPQA = BY_KEY["gpqa_diamond"]


def attempt(task: str, n: int, passed: bool, **kw) -> Attempt:
    return Attempt(benchmark=AIME.key, task=task, attempt=n, passed=passed, **kw)


def test_a_score_is_the_mean_over_tasks_of_the_mean_over_attempts():
    attempts = [
        attempt("t1", 1, True), attempt("t1", 2, True), attempt("t1", 3, False), attempt("t1", 4, False),
        attempt("t2", 1, True), attempt("t2", 2, True), attempt("t2", 3, True), attempt("t2", 4, True),
    ]
    assert task_scores(attempts) == {"t1": 0.5, "t2": 1.0}
    r = score(AIME, attempts, subset_id="aime_2025_full:abc", harness_version="0.1.0", n_planned=2)
    assert r.value == 0.75
    assert r.n_tasks == 2 and r.attempts_per_task == 4 and r.n_samples == 8
    assert r.stderr == pytest.approx(0.25)  # stdev([0.5, 1.0]) / sqrt(2)
    assert r.task == "aime_2025" and r.metric == "mean accuracy"
    assert r.subset_id == "aime_2025_full:abc" and r.harness == "lab"


def test_an_unfinished_answer_is_wrong_not_missing():
    # The runner sets passed=False on a length stop; the score must count the task, not drop it.
    attempts = [attempt("t1", 1, False, finish_reason="length"), attempt("t2", 1, True)]
    r = score(GPQA, attempts, subset_id="s:1", harness_version="v", n_planned=2)
    assert r.value == 0.5 and r.n_tasks == 2


def test_excluded_tasks_do_not_count_either_way_until_there_are_too_many():
    attempts = [attempt(f"t{i}", 1, True) for i in range(10)]
    attempts.append(attempt("t10", 1, False, excluded=True, error="ConnectError"))
    r = score(GPQA, attempts, subset_id="s:1", harness_version="v", n_planned=11)
    assert r.value == 1.0 and r.n_tasks == 10, "an infrastructure failure is not a model failure"

    with pytest.raises(ValueError, match="excluded"):
        score(GPQA, attempts[:5] + [attempt(f"x{i}", 1, False, excluded=True) for i in range(5)],
              subset_id="s:1", harness_version="v", n_planned=10)


def test_operating_numbers_are_per_task_and_keyed_by_benchmark():
    attempts = [
        attempt("t1", 1, True, seconds=10, completion_tokens=100, allowance=8000),
        attempt("t1", 2, False, seconds=20, completion_tokens=900, allowance=8000, finish_reason="length"),
        attempt("t2", 1, True, seconds=30, completion_tokens=200, allowance=7000),
    ]
    by_key = {m.key: m for m in metrics(AIME, attempts, quick=True)}
    assert by_key["eval_seconds_per_task"].value == 30.0, "60 s over 2 tasks"
    assert by_key["eval_tokens_per_task"].value == 600.0
    assert by_key["eval_length_stops"].value == 1
    assert by_key["eval_excluded"].value == 0
    assert by_key["eval_allowance_tokens"].value == pytest.approx(7666.7, abs=0.1)
    assert all(m.method == "aime_2025" for m in by_key.values())
    assert "eval_allowance_tokens" not in {m.key for m in metrics(AIME, attempts, quick=False)}


def test_a_checkpoint_survives_a_stopped_sitting(tmp_path):
    path = tmp_path / "attempts.jsonl"
    first = Checkpoint(path)
    first.add(attempt("t1", 1, True, seconds=4))
    first.add(attempt("t2", 1, False, error="ConnectError", excluded=True))
    first.close()

    resumed = Checkpoint(path)
    assert resumed.has("aime_2025", "t1", 1)
    assert not resumed.has("aime_2025", "t1", 2), "an unfinished attempt is run again"
    assert resumed.attempts[("aime_2025", "t1", 1)].seconds == 4
    resumed.close()
    assert len(path.read_text().splitlines()) == 2


def test_a_resumed_run_excludes_exactly_what_an_unbroken_one_would(tmp_path):
    # t0 answered once, then hit an infrastructure failure: the whole task is out, not just that attempt.
    records = [attempt("t0", 1, True), attempt("t0", 2, False, excluded=True, error="ConnectError")]
    records += [attempt(f"t{i}", n, n == 1) for i in range(1, 10) for n in (1, 2)]
    path = tmp_path / "attempts.jsonl"
    first = Checkpoint(path)
    for r in records:
        first.add(r)
    first.close()

    kw = dict(subset_id="s:1", harness_version="v", n_planned=10)
    unbroken = score(AIME, records, **kw)
    resumed = score(AIME, Checkpoint(path).for_benchmark(AIME.key), **kw)
    assert unbroken == resumed
    assert unbroken.n_tasks == 9 and unbroken.n_samples == 18 and unbroken.value == 0.5
    assert {m.key: m.value for m in metrics(AIME, records, quick=True)}["eval_excluded"] == 1


def test_stop_at_is_the_next_time_that_clock_reads(monkeypatch):
    assert parse_stop_at(None) is None
    soon = parse_stop_at("23:59")
    assert 0 < soon - __import__("time").monotonic() <= 24 * 3600
    with pytest.raises(ValueError, match="HH:MM"):
        parse_stop_at("7am")


def test_the_tier_decides_which_benchmarks_run():
    assert [b.key for b in registry.resolve(None, "quick")] == ["livecodebench", "bfcl", "gpqa_diamond", "aime_2025"]
    assert [b.key for b in registry.resolve(["aime_2025"], "quick")] == ["aime_2025"]
    with pytest.raises(KeyError, match="deep-tier"):
        registry.resolve(["swebench_verified"], "quick")
    with pytest.raises(KeyError, match="unknown"):
        registry.resolve(["mmlu"], "quick")


def test_aime_grading_reads_the_boxed_answer():
    task = Task(id="aime_2025/I-1", messages=[], answer="70")
    driver = Aime2025()
    assert driver.grade(task, "... so the answer is \\boxed{70}.").passed
    assert driver.grade(task, "\\boxed{70}\n\nWait, actually \\boxed{71}").passed is False, "the last box wins"
    assert driver.grade(task, "I think it is 70").passed, "a bare final number still counts"
    assert driver.grade(task, "no idea").extracted is None
    assert extract_answer("\\boxed{1,024}") == "1024"


def test_a_subset_that_drifted_from_its_pin_refuses_to_run(tmp_path, monkeypatch):
    from lab.evals.drivers import base

    monkeypatch.setattr(base.paths, "SUBSETS", tmp_path)
    subset = Subset(name="demo", tasks=[Task(id="a", messages=[], answer="1")])
    with pytest.raises(FileNotFoundError):
        check_against_pin("demo", subset)
    base.write_pin("demo", subset)
    check_against_pin("demo", subset)
    drifted = Subset(name="demo", tasks=[Task(id="b", messages=[], answer="1")])
    with pytest.raises(ValueError, match="no longer yields"):
        check_against_pin("demo", drifted)


def test_the_committed_aime_pin_matches_the_dataset():
    """The pin in the repo is what a run will use; if the dataset moved, every later run refuses."""
    subset = Aime2025().subset()
    assert len(subset.tasks) == 30
    check_against_pin("aime_2025", subset)
    assert json.loads((__import__("lab.paths", fromlist=["x"]).SUBSETS / "aime_2025.json").read_text())["task_ids"][0] == "aime_2025/I-1"


def test_the_bfcl_subset_is_stratified_interleaved_and_matches_its_pin():
    from lab.evals.drivers.bfcl import CATEGORIES, PER_CATEGORY, Bfcl
    from lab.evals.drivers.base import load_pin

    sizes = {"simple_python": 400, "multiple": 200, "parallel": 200, "parallel_multiple": 200, "multi_turn_base": 200}
    listing = [{"id": f"{c}_{i}", "meta": {"category": c}} for c, n in sizes.items() for i in range(n)]
    chosen = Bfcl().select(listing)
    assert len(chosen) == PER_CATEGORY * len(CATEGORIES)
    assert [t["meta"]["category"] for t in chosen[:5]] == list(CATEGORIES), "a --limit smoke test touches every category"
    assert [t["id"] for t in chosen] == load_pin("bfcl")["task_ids"], "the committed pin is what the seed draws"


def evals_bundle(speed_bundle, **raw):
    b = speed_bundle(metrics=[], id="evals-1", kind="evals", tier="quick", status="ok", published_at=None)
    b.run.raw.update(raw)
    return b


def test_a_resumed_run_keeps_its_benchmarks_and_adds_the_ones_asked_for(speed_bundle):
    from lab.evals.runner import _benchmarks

    prior = evals_bundle(speed_bundle, subsets={"aime_2025": {}})  # a run from before `benchmarks` was recorded
    assert [b.key for b in _benchmarks(["bfcl"], "quick", prior)] == ["aime_2025", "bfcl"]
    assert {b.key for b in _benchmarks(None, "quick", prior)} == {b.key for b in registry.tier_benchmarks("quick")}
    assert [b.key for b in _benchmarks(["bfcl"], "quick", None)] == ["bfcl"]


def test_a_resumed_run_is_sized_as_it_started_not_by_a_newer_speed_run(speed_bundle):
    from lab.evals.allowance import Allowance, SpeedPoint
    from lab.evals.runner import _allowance_for

    started = Allowance(ctx=32768, pp0=8196, tg=[SpeedPoint(0, 180), SpeedPoint(16384, 163)], limit_s=300, speed_run_id="old")
    prior = evals_bundle(speed_bundle, allowance=started.as_raw())
    again = _allowance_for(None, None, "quick", True, prior)
    assert again == started
    assert again.for_prompt(8000) == started.for_prompt(8000)


def test_resuming_on_another_engine_build_or_harness_is_refused(speed_bundle, tmp_path):
    from types import SimpleNamespace

    from lab.evals.runner import EvalError, _hold_run_to_its_start
    from lab.evals.sandbox import Image
    from lab.schema import EngineBuild
    from lab.store import RunDir

    rd = RunDir(tmp_path)
    image = Image("bfcl", "lab-bfcl:aaa", "sha256:aaa", "web")
    allowance = Allowance(ctx=65536, limit_s=None)
    same_build = SimpleNamespace(build_info=lambda: EngineBuild(engine="llama.cpp", version="10883", commit_sha="91f6a6cf3"))
    new_build = SimpleNamespace(build_info=lambda: EngineBuild(engine="llama.cpp", version="10990", commit_sha="0abc"))

    bundle = evals_bundle(speed_bundle)
    bundle.eval_results = [score(AIME, [attempt("t1", 1, True)], subset_id="s", harness_version="v", n_planned=1)]
    _hold_run_to_its_start(rd, bundle, same_build, [AIME, BY_KEY["bfcl"]], allowance, {"bfcl": image}, resumed=True)
    assert bundle.run.status == "running" and bundle.eval_results == [], "not publishable until it is scored again"
    assert bundle.run.raw["benchmarks"] == ["aime_2025", "bfcl"]
    assert bundle.run.raw["sandbox_images"]["bfcl"]["image_id"] == "sha256:aaa"

    with pytest.raises(EvalError, match="one build"):
        _hold_run_to_its_start(rd, bundle, new_build, [AIME], allowance, {}, resumed=True)
    rebuilt = Image("bfcl", "lab-bfcl:bbb", "sha256:bbb", "web")
    with pytest.raises(EvalError, match="one harness"):
        _hold_run_to_its_start(rd, bundle, same_build, [AIME], allowance, {"bfcl": rebuilt}, resumed=True)


def test_a_tier_is_published_whole_or_not_at_all(speed_bundle, tmp_path):
    from lab.publish import PublishError, publish
    from lab.store import RunDir

    rd = RunDir(tmp_path)
    bundle = evals_bundle(speed_bundle)
    bundle.eval_results = [score(AIME, [attempt("t1", 1, True)], subset_id="s", harness_version="v", n_planned=1)]
    rd.save(bundle)
    with pytest.raises(PublishError, match="gpqa_diamond") as e:
        publish(rd, dry_run=True)
    assert "--benchmarks livecodebench,bfcl,gpqa_diamond" in str(e.value)


def test_the_livecodebench_subset_is_the_newest_problems_in_date_order():
    from lab.evals.drivers.livecodebench import NEWEST, LiveCodeBench

    listing = [{"id": f"livecodebench/p{i}", "meta": {"contest_date": f"2025-{1 + i % 4:02d}-{1 + i % 28:02d}T00:00:00"}} for i in range(175)]
    chosen = LiveCodeBench().select(listing)
    dates = [t["meta"]["contest_date"] for t in chosen]
    assert len(chosen) == NEWEST and dates == sorted(dates)
    assert min(dates) >= max(t["meta"]["contest_date"] for t in listing if t not in chosen)
    assert LiveCodeBench().describe(chosen)["date_window"] == [dates[0], dates[-1]]
