/**
 * Capability scoring: how well a config does the work, and how long it may think while doing it.
 *
 * Pure functions only, so they can be unit-tested against the plan's vectors. Benchmark values and
 * their standard errors arrive from the lab as fractions (0-1); everything here works in percent.
 */

export type CategoryKey = 'coding' | 'agents' | 'reasoning';

export const CATEGORIES: { key: CategoryKey; name: string; weight: number }[] = [
  { key: 'coding', name: 'Coding', weight: 0.5 },
  { key: 'agents', name: 'Agents & tools', weight: 0.3 },
  { key: 'reasoning', name: 'Reasoning', weight: 0.2 },
];

export type Tier = 'quick' | 'deep';

export type Benchmark = {
  key: string;
  name: string;
  category: CategoryKey;
  tier: Tier;
  /** Unit the standard error is over: problems, cases, questions or tasks. */
  n: number;
  size: string;
  metric: string;
};

/** The lab publishes eval results under these task keys. */
export const BENCHMARKS: Benchmark[] = [
  { key: 'livecodebench', name: 'LiveCodeBench', category: 'coding', tier: 'quick', n: 100, size: '100 problems', metric: 'pass@1' },
  { key: 'swebench_verified', name: 'SWE-bench Verified', category: 'coding', tier: 'deep', n: 30, size: '30 issues', metric: 'resolved' },
  { key: 'bfcl', name: 'BFCL', category: 'agents', tier: 'quick', n: 400, size: '400 cases', metric: 'accuracy' },
  { key: 'terminal_bench', name: 'Terminal-Bench', category: 'agents', tier: 'deep', n: 30, size: '30 tasks', metric: 'resolved' },
  { key: 'gpqa_diamond', name: 'GPQA Diamond', category: 'reasoning', tier: 'quick', n: 198, size: '198 questions', metric: 'accuracy' },
  { key: 'aime_2025', name: 'AIME 2025', category: 'reasoning', tier: 'quick', n: 30, size: '30 problems × 4 attempts', metric: 'mean accuracy' },
];

export const QUICK_BENCHMARKS = BENCHMARKS.filter((b) => b.tier === 'quick');
export const byKey = new Map(BENCHMARKS.map((b) => [b.key, b]));

/** One benchmark result in percent. */
export type BenchScore = { value: number; stderr: number };

export type CategoryScore = { key: CategoryKey; name: string; weight: number; mean: number; stderr: number; benchmarks: string[] };

export type Score = { value: number; stderr: number; categories: CategoryScore[] };

/** Binomial standard error in percent, for a benchmark without one of its own. */
export function binomialStderr(valuePercent: number, n: number): number {
  const p = valuePercent / 100;
  return Math.sqrt((p * (1 - p)) / n) * 100;
}

/**
 * Weighted geometric mean of the category means, with its standard error by the delta method.
 *
 * Geometric so that weakness in one category pulls the whole score down. Returns null unless every
 * benchmark in `benchmarks` has a score.
 */
export function score(scores: Record<string, BenchScore>, benchmarks: Benchmark[] = BENCHMARKS): Score | null {
  if (!benchmarks.every((b) => scores[b.key] != null)) return null;
  const categories: CategoryScore[] = [];
  for (const cat of CATEGORIES) {
    const bs = benchmarks.filter((b) => b.category === cat.key);
    if (bs.length === 0) continue;
    const mean = bs.reduce((sum, b) => sum + scores[b.key].value, 0) / bs.length;
    const stderr = Math.sqrt(bs.reduce((sum, b) => sum + scores[b.key].stderr ** 2, 0)) / bs.length;
    categories.push({ ...cat, mean, stderr, benchmarks: bs.map((b) => b.key) });
  }
  const totalWeight = categories.reduce((sum, c) => sum + c.weight, 0);
  // A category mean of 0 would make the log diverge; 1% is the floor the plan sets.
  const value = Math.exp(categories.reduce((sum, c) => sum + (c.weight / totalWeight) * Math.log(Math.max(c.mean, 1)), 0));
  const stderr = value * Math.sqrt(categories.reduce((sum, c) => sum + ((c.weight / totalWeight) * c.stderr / Math.max(c.mean, 1)) ** 2, 0));
  return { value, stderr, categories };
}

/** The same formula over the quick-tier benchmarks, so a config can be compared before its deep runs. */
export function quickScore(scores: Record<string, BenchScore>): Score | null {
  return score(scores, QUICK_BENCHMARKS);
}

export type Ranked<T> = T & { rank: number; rankLabel: string; tiedWith: string[] };

/**
 * Sort by score and label ties: neighbours closer than one combined standard error share a rank.
 * `name` only feeds the "too close to call" note.
 */
export function rank<T extends { score: Score; name: string }>(rows: T[]): Ranked<T>[] {
  const sorted = [...rows].sort((a, b) => b.score.value - a.score.value);
  const out: Ranked<T>[] = [];
  sorted.forEach((row, i) => {
    const prev = out[i - 1];
    const tied = !!prev && Math.abs(prev.score.value - row.score.value) < Math.hypot(prev.score.stderr, row.score.stderr);
    const rankNumber = tied ? prev.rank : i + 1;
    if (tied) {
      prev.rankLabel = `=${prev.rank}`;
      prev.tiedWith.push(row.name);
    }
    out.push({ ...row, rank: rankNumber, rankLabel: tied ? `=${rankNumber}` : String(rankNumber), tiedWith: tied ? [prev.name] : [] });
  });
  return out;
}

// -- thinking allowance --------------------------------------------------------------------------

export type SpeedPoint = { depth: number; tps: number };

/** Generation speed at a context depth, from a speed run's measured depths. */
export function tgAtDepth(points: SpeedPoint[], depth: number): number {
  const pts = [...points].sort((a, b) => a.depth - b.depth);
  if (pts.length === 0) return 0;
  if (pts.length === 1 || depth <= pts[0].depth) return pts[0].tps;
  for (let i = 1; i < pts.length; i++) {
    if (depth <= pts[i].depth) {
      const a = pts[i - 1];
      const b = pts[i];
      return a.tps + ((depth - a.depth) / (b.depth - a.depth)) * (b.tps - a.tps);
    }
  }
  // Past the deepest measurement, continue the last segment's slope, but never below 1 t/s.
  const a = pts[pts.length - 2];
  const b = pts[pts.length - 1];
  const slope = (b.tps - a.tps) / (b.depth - a.depth);
  return Math.max(1, b.tps + (depth - b.depth) * slope);
}

export type AllowanceInput = {
  /** Prompt-processing t/s at depth 0. */
  pp0: number;
  /** Generation t/s by depth, from the same speed run. */
  tg: SpeedPoint[];
  /** The config's context length. */
  ctx: number;
  /** This request's prompt, as the model sees it. */
  promptTokens: number;
  /** Wall-clock budget for the task. */
  limitS: number;
};

/**
 * How many tokens a config may spend on one task: what it can generate in the time left after
 * reading the prompt, capped by the context it has left. Thinking counts toward it.
 */
export function allowance({ pp0, tg, ctx, promptTokens, limitS }: AllowanceInput): number {
  if (pp0 <= 0) return 0;
  const secondsLeft = limitS - promptTokens / pp0;
  if (secondsLeft <= 0) return 0;
  const contextLeft = ctx - promptTokens;
  if (contextLeft <= 0) return 0;
  return Math.max(0, Math.min(Math.floor(secondsLeft * tgAtDepth(tg, promptTokens)), contextLeft));
}

/** Issues a config could fix overnight at its measured pace. */
export function issuesPerNight(meanMinutesPerIssue: number, resolvedRatePercent: number, nightMinutes = 480): number | null {
  if (!meanMinutesPerIssue || meanMinutesPerIssue <= 0) return null;
  return Math.round((nightMinutes / meanMinutesPerIssue) * (resolvedRatePercent / 100));
}
