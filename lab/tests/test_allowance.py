"""The plan's allowance vectors. The site's scoring.test.ts asserts the same numbers."""

import pytest

from lab.evals.allowance import (
    Allowance,
    NoSpeedRun,
    SpeedPoint,
    from_speed_bundle,
    tg_at_depth,
)

QWEN_TG = [SpeedPoint(0, 31.8), SpeedPoint(4096, 30.9), SpeedPoint(16384, 28.2)]
GEMMA_TG = [SpeedPoint(0, 180), SpeedPoint(4096, 175), SpeedPoint(16384, 163)]

VECTORS = [
    # pp0, tg ladder, ctx, prompt tokens, limit s, expected allowance
    (949, QWEN_TG, 65536, 8000, 300, 8759),
    (8196, GEMMA_TG, 32768, 8000, 300, 24768),  # capped by the context left
    (949, QWEN_TG, 65536, 20000, 300, 7644),  # past the deepest measurement
    (949, QWEN_TG, 65536, 8000, 5, 0),  # the prompt alone uses up the limit
    (949, [SpeedPoint(0, 31.8)], 65536, 8000, 300, 9271),  # a single measured depth
]


@pytest.mark.parametrize(("pp0", "tg", "ctx", "prompt", "limit_s", "expected"), VECTORS)
def test_allowance_vectors(pp0, tg, ctx, prompt, limit_s, expected):
    assert Allowance(ctx=ctx, pp0=pp0, tg=tg, limit_s=limit_s).for_prompt(prompt) == expected


def test_tg_at_depth_clamps_below_the_first_depth_and_never_goes_under_1():
    assert tg_at_depth(QWEN_TG, 0) == 31.8
    assert tg_at_depth(QWEN_TG, -100) == 31.8
    assert tg_at_depth(QWEN_TG, 4096) == 30.9
    assert tg_at_depth([SpeedPoint(0, 10), SpeedPoint(1000, 5)], 100_000) == 1.0
    assert tg_at_depth([], 0) == 0.0


def test_the_deep_tier_is_capped_only_by_the_context_left():
    deep = Allowance(ctx=65536, limit_s=None)
    assert deep.for_prompt(20_000) == 45_536
    assert deep.for_prompt(65_536) == 0
    assert deep.for_prompt(70_000) == 0


def test_a_config_with_no_speed_numbers_cannot_be_evaluated(speed_bundle):
    bundle = speed_bundle(metrics=[])
    with pytest.raises(NoSpeedRun):
        from_speed_bundle(bundle, limit_s=300)
    # The deep tier needs no speed run: the limit is wall-clock, so context is the only cap.
    assert from_speed_bundle(bundle, limit_s=None).for_prompt(1000) == 65536 - 1000


def test_from_speed_bundle_reads_the_depth_ladder_and_reports_its_source(speed_bundle):
    a = from_speed_bundle(speed_bundle(), limit_s=300)
    assert a.pp0 == 949
    assert a.tg == QWEN_TG
    assert a.for_prompt(8000) == 8759
    raw = a.as_raw()
    assert raw["speed_run_id"] == "run-1"
    assert raw["headline_tokens"] == 8759
    assert raw["tg"][-1] == {"depth": 16384, "tps": 28.2}


def test_http_chat_metrics_fill_in_when_llama_bench_did_not_measure_a_depth(speed_bundle):
    from lab.schema import Metric

    bundle = speed_bundle(
        metrics=[
            Metric(key="pp_tps", method="llama-bench", n_prompt=512, value=949, unit="t/s"),
            Metric(key="tg_tps", method="llama-bench", n_gen=128, value=31.8, unit="t/s"),
            Metric(key="decode_tps", method="http-sampled", depth=16384, value=27.0, unit="t/s"),
        ]
    )
    a = from_speed_bundle(bundle, limit_s=300)
    assert [p.depth for p in a.tg] == [0, 16384]
    assert a.tg[1].tps == 27.0
