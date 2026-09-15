import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BENCHMARKS, allowance, binomialStderr, issuesPerNight, quickScore, rank, score, tgAtDepth } from '../src/scoring.ts';
import type { BenchScore, Score } from '../src/scoring.ts';

/** The plan's thinking-allowance vectors (section 5), verified against its published outputs. */
const QWEN_TG = [
  { depth: 0, tps: 31.8 },
  { depth: 4096, tps: 30.9 },
  { depth: 16384, tps: 28.2 },
];

test('allowance: interpolates generation speed at the prompt depth', () => {
  assert.equal(allowance({ pp0: 949, tg: QWEN_TG, ctx: 65536, promptTokens: 8000, limitS: 300 }), 8759);
});

test('allowance: never exceeds the context left', () => {
  const tg = [
    { depth: 0, tps: 180 },
    { depth: 4096, tps: 175 },
    { depth: 16384, tps: 163 },
  ];
  assert.equal(allowance({ pp0: 8196, tg, ctx: 32768, promptTokens: 8000, limitS: 300 }), 24768);
});

test('allowance: extrapolates past the deepest measurement', () => {
  assert.equal(allowance({ pp0: 949, tg: QWEN_TG, ctx: 65536, promptTokens: 20000, limitS: 300 }), 7644);
});

test('allowance: zero when the prompt alone uses up the limit', () => {
  assert.equal(allowance({ pp0: 949, tg: QWEN_TG, ctx: 65536, promptTokens: 8000, limitS: 5 }), 0);
});

test('allowance: a single measured depth is used everywhere', () => {
  assert.equal(allowance({ pp0: 949, tg: [{ depth: 0, tps: 31.8 }], ctx: 65536, promptTokens: 8000, limitS: 300 }), 9271);
});

test('tgAtDepth: clamps below the first depth and never goes under 1 t/s', () => {
  assert.equal(tgAtDepth(QWEN_TG, 0), 31.8);
  assert.equal(tgAtDepth(QWEN_TG, -100), 31.8);
  assert.equal(tgAtDepth([{ depth: 0, tps: 10 }, { depth: 1000, tps: 2 }], 10_000_000), 1);
});

/**
 * The approved mockup's illustrative configs, scored with binomial standard errors. The expected
 * numbers come from the plan's table (section 6), which was verified against the mockup's own script.
 */
const MOCKUP: { name: string; scores: Record<string, number>; expected: { score?: [number, number]; quick: number; rankLabel?: string } }[] = [
  { name: '27B dense', scores: { livecodebench: 62, swebench_verified: 47, bfcl: 78, terminal_bench: 33, gpqa_diamond: 68, aime_2025: 71 }, expected: { score: [57.5, 3.1], quick: 68.0, rankLabel: '1' } },
  { name: '30B-A3B MoE', scores: { livecodebench: 56, swebench_verified: 41, bfcl: 75, terminal_bench: 29, gpqa_diamond: 61, aime_2025: 69 }, expected: { score: [52.5, 3.2], quick: 63.0, rankLabel: '=2' } },
  { name: '24B dense, coding-tuned', scores: { livecodebench: 59, swebench_verified: 53, bfcl: 71, terminal_bench: 37, gpqa_diamond: 44, aime_2025: 28 }, expected: { score: [50.7, 3.0], quick: 56.5, rankLabel: '=2' } },
  { name: '20B-A4B MoE', scores: { livecodebench: 51, swebench_verified: 31, bfcl: 63, terminal_bench: 22, gpqa_diamond: 57, aime_2025: 63 }, expected: { score: [44.7, 3.0], quick: 56.1, rankLabel: '4' } },
  { name: '32B dense', scores: { livecodebench: 60, bfcl: 74, gpqa_diamond: 66, aime_2025: 55 }, expected: { quick: 64.0 } },
  { name: '14B dense', scores: { livecodebench: 43, bfcl: 64, gpqa_diamond: 49, aime_2025: 41 }, expected: { quick: 48.9 } },
  { name: '4B dense', scores: { livecodebench: 24, bfcl: 46, gpqa_diamond: 31, aime_2025: 17 }, expected: { quick: 29.2 } },
];

function withStderr(scores: Record<string, number>): Record<string, BenchScore> {
  const out: Record<string, BenchScore> = {};
  for (const b of BENCHMARKS) {
    if (scores[b.key] != null) out[b.key] = { value: scores[b.key], stderr: binomialStderr(scores[b.key], b.n) };
  }
  return out;
}

test('score: reproduces the mockup scores, errors and quick scores', () => {
  for (const row of MOCKUP) {
    const scores = withStderr(row.scores);
    const full = score(scores);
    const quick = quickScore(scores);
    assert.ok(quick, `${row.name}: quick score`);
    assert.equal(Number(quick.value.toFixed(1)), row.expected.quick, `${row.name}: quick score`);
    if (row.expected.score) {
      assert.ok(full, `${row.name}: full score`);
      assert.equal(Number(full.value.toFixed(1)), row.expected.score[0], `${row.name}: score`);
      assert.equal(Number(full.stderr.toFixed(1)), row.expected.score[1], `${row.name}: score stderr`);
    } else {
      assert.equal(full, null, `${row.name}: incomplete benchmarks must not score`);
    }
  }
});

test('score: an uneven profile scores below an even one with the same weighted average', () => {
  const even = withStderr({ livecodebench: 60, swebench_verified: 60, bfcl: 60, terminal_bench: 60, gpqa_diamond: 60, aime_2025: 60 });
  assert.equal(Number(score(even)!.value.toFixed(3)), 60);
  // Weighted average is also 60 (0.5·70 + 0.3·60 + 0.2·35), but the weak category drags the score down.
  const uneven = withStderr({ livecodebench: 70, swebench_verified: 70, bfcl: 60, terminal_bench: 60, gpqa_diamond: 35, aime_2025: 35 });
  assert.ok(score(uneven)!.value < score(even)!.value, 'a weak category costs more than a strong one gains');
});

test('score: coding counts for more than reasoning', () => {
  const flat = { livecodebench: 50, swebench_verified: 50, bfcl: 50, terminal_bench: 50, gpqa_diamond: 50, aime_2025: 50 };
  const betterCoding = score(withStderr({ ...flat, livecodebench: 60, swebench_verified: 60 }))!.value;
  const betterReasoning = score(withStderr({ ...flat, gpqa_diamond: 60, aime_2025: 60 }))!.value;
  assert.ok(betterCoding > betterReasoning, 'the same gain is worth more in coding');
});

test('rank: neighbours within one combined standard error share a rank', () => {
  const rows = MOCKUP.filter((r) => r.expected.score).map((r) => ({ name: r.name, score: score(withStderr(r.scores)) as Score }));
  const ranked = rank(rows);
  assert.deepEqual(
    ranked.map((r) => [r.name, r.rankLabel]),
    MOCKUP.filter((r) => r.expected.score).map((r) => [r.name, r.expected.rankLabel]),
  );
  assert.deepEqual(ranked[1].tiedWith, ['24B dense, coding-tuned']);
});

test('issuesPerNight: derived from minutes per issue and the resolved rate', () => {
  assert.equal(issuesPerNight(14, 47), 16);
  assert.equal(issuesPerNight(0, 47), null);
});
