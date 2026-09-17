import type { Db } from './db.ts';
import { allowance, binomialStderr, byKey, issuesPerNight, quickScore, score } from './scoring.ts';
import type { BenchScore, Score, SpeedPoint } from './scoring.ts';
import type { HostedRow, PauseRow } from './status.ts';

type Row = Record<string, any>;

export type MetricRow = {
  key: string;
  method: string;
  n_prompt: number;
  n_gen: number;
  depth: number;
  concurrency: number;
  value: number;
  stddev: number | null;
  n: number | null;
  unit: string;
};

export type EvalRow = {
  task: string;
  metric: string;
  filter: string;
  value: number;
  stderr: number | null;
  n_samples: number | null;
  limit_n: number | null;
  lm_eval_version: string | null;
  harness: string | null;
  harness_version: string | null;
  subset_id: string | null;
  n_tasks: number | null;
  attempts_per_task: number | null;
};

export type RunSummary = {
  id: string;
  kind: 'speed' | 'quality' | 'evals';
  tier: string | null;
  started_at: string;
  duration_s: number | null;
  throttled: boolean;
  cli_args: string | null;
  lab_version: string | null;
  engine: string;
  engine_version: string | null;
  commit_sha: string | null;
  driver: string;
  metrics: MetricRow[];
  evals: EvalRow[];
  telemetry: Record<string, number>[];
};

export type ConfigView = {
  id: number;
  slug: string;
  name: string;
  config_hash: string;
  params: Record<string, unknown>;
  launch_command: string;
  notes: string | null;
  /** Engine-native speed (llama-bench). */
  speed: RunSummary | null;
  /** Same-protocol chat benchmark over HTTP, comparable across engines. */
  chat: RunSummary | null;
  quality: RunSummary | null;
  evalsQuick: RunSummary | null;
  evalsDeep: RunSummary | null;
  history: RunSummary[];
};

export type ModelView = {
  id: number;
  slug: string;
  name: string;
  engine: string;
  format: string;
  quant: string;
  bpw: number | null;
  file_size_bytes: number | null;
  source_repo: string | null;
  source_file: string | null;
  source_revision: string | null;
  notes: string | null;
  base: { slug: string; name: string; family: string | null; params_b: number | null; active_params_b: number | null; arch: string };
  configs: ConfigView[];
};

const RUN_SQL = `
  SELECT r.id, r.kind, r.tier, r.started_at, r.duration_s, r.throttled, r.cli_args, r.lab_version, r.telemetry_json,
         e.engine, e.version AS engine_version, e.commit_sha, h.driver
  FROM runs r
  JOIN engine_builds e ON e.id = r.engine_build_id
  JOIN hardware_snapshots h ON h.id = r.hardware_id`;

function toRun(db: Db, r: Row, withTelemetry: boolean): RunSummary {
  const metrics = db
    .prepare('SELECT key, method, n_prompt, n_gen, depth, concurrency, value, stddev, n, unit FROM metrics WHERE run_id = ? ORDER BY key, depth, n_prompt, n_gen')
    .all(r.id) as MetricRow[];
  const evals = db
    .prepare(`SELECT task, metric, filter, value, stderr, n_samples, limit_n, lm_eval_version,
                     harness, harness_version, subset_id, n_tasks, attempts_per_task
              FROM eval_results WHERE run_id = ? ORDER BY task, metric`)
    .all(r.id) as EvalRow[];
  return {
    id: r.id, kind: r.kind, tier: r.tier, started_at: r.started_at, duration_s: r.duration_s, throttled: !!r.throttled,
    cli_args: r.cli_args, lab_version: r.lab_version, engine: r.engine, engine_version: r.engine_version,
    commit_sha: r.commit_sha, driver: r.driver, metrics, evals,
    telemetry: withTelemetry ? JSON.parse(r.telemetry_json) : [],
  };
}

/**
 * Latest run of a kind, preferring non-throttled runs.
 *
 * `tier` separates quick from deep evals; without it the newer tier would hide the other. `methodLike`
 * keeps runs measured different ways apart, so a chat-benchmark run never hides a llama-bench one.
 */
function latestRun(db: Db, configId: number, kind: string, opts: { tier?: string; methodLike?: string } = {}): RunSummary | null {
  const where = [`r.config_id = ?`, `r.kind = ?`];
  const args: unknown[] = [configId, kind];
  if (opts.tier !== undefined) {
    where.push('r.tier = ?');
    args.push(opts.tier);
  }
  if (opts.methodLike !== undefined) {
    where.push('EXISTS (SELECT 1 FROM metrics mt WHERE mt.run_id = r.id AND mt.method LIKE ?)');
    args.push(opts.methodLike);
  }
  const r = db
    .prepare(`${RUN_SQL} WHERE ${where.join(' AND ')} ORDER BY r.throttled ASC, r.started_at DESC LIMIT 1`)
    .get(...(args as [])) as Row | undefined;
  return r ? toRun(db, r, true) : null;
}

function loadConfigs(db: Db, modelId: number, withHistory: boolean): ConfigView[] {
  const rows = db.prepare('SELECT * FROM configs WHERE model_id = ? ORDER BY created_at, id').all(modelId) as Row[];
  return rows.map((c) => ({
    id: c.id, slug: c.slug, name: c.name, config_hash: c.config_hash, params: JSON.parse(c.params_json),
    launch_command: c.launch_command, notes: c.notes,
    speed: latestRun(db, c.id, 'speed', { methodLike: 'llama-bench' }),
    chat: latestRun(db, c.id, 'speed', { methodLike: 'http-%' }),
    quality: latestRun(db, c.id, 'quality'),
    evalsQuick: latestRun(db, c.id, 'evals', { tier: 'quick' }),
    evalsDeep: latestRun(db, c.id, 'evals', { tier: 'deep' }),
    history: withHistory
      ? (db.prepare(`${RUN_SQL} WHERE r.config_id = ? ORDER BY r.started_at DESC`).all(c.id) as Row[]).map((r) => toRun(db, r, false))
      : [],
  }));
}

function toModel(db: Db, m: Row, withHistory: boolean): ModelView {
  return {
    id: m.id, slug: m.slug, name: m.name, engine: m.engine, format: m.format, quant: m.quant, bpw: m.bpw,
    file_size_bytes: m.file_size_bytes, source_repo: m.source_repo, source_file: m.source_file,
    source_revision: m.source_revision, notes: m.notes,
    base: { slug: m.base_slug, name: m.base_name, family: m.family, params_b: m.params_b, active_params_b: m.active_params_b, arch: m.arch },
    configs: loadConfigs(db, m.id, withHistory),
  };
}

const MODEL_SQL = `
  SELECT m.*, b.slug AS base_slug, b.name AS base_name, b.family, b.params_b, b.active_params_b, b.arch
  FROM models m JOIN base_models b ON b.id = m.base_model_id`;

export function getModels(db: Db): ModelView[] {
  const rows = db.prepare(`${MODEL_SQL} WHERE EXISTS (SELECT 1 FROM configs c JOIN runs r ON r.config_id = c.id WHERE c.model_id = m.id) ORDER BY b.name, m.name`).all() as Row[];
  return rows.map((m) => toModel(db, m, false));
}

export function getModel(db: Db, slug: string): ModelView | null {
  const m = db.prepare(`${MODEL_SQL} WHERE m.slug = ?`).get(slug) as Row | undefined;
  return m ? toModel(db, m, true) : null;
}

export function getRun(db: Db, id: string) {
  const r = db.prepare(`${RUN_SQL} WHERE r.id = ?`).get(id) as Row | undefined;
  if (!r) return null;
  const extra = db.prepare(
    `SELECT r.raw_json, c.slug AS config_slug, c.name AS config_name, m.slug AS model_slug, m.name AS model_name
     FROM runs r JOIN configs c ON c.id = r.config_id JOIN models m ON m.id = c.model_id WHERE r.id = ?`,
  ).get(id) as Row;
  return { ...toRun(db, r, true), raw: JSON.parse(extra.raw_json), config_slug: extra.config_slug, config_name: extra.config_name, model_slug: extra.model_slug, model_name: extra.model_name };
}

export function getHosted(db: Db) {
  return (db.prepare(
    `SELECT h.since, c.slug AS config_slug, c.name AS config_name, m.slug AS model_slug, m.name AS model_name
     FROM hosted h LEFT JOIN configs c ON c.config_hash = h.config_hash LEFT JOIN models m ON m.id = c.model_id WHERE h.id = 1`,
  ).get() as HostedRow | undefined) ?? null;
}

/** Latest pause report from `lab` (benchmarks and `lab serve` pause the hosted model), with the model's name if published. */
export function getPause(db: Db) {
  return (db.prepare(
    `SELECT p.paused, p.reason, p.ref, p.since, m.slug AS model_slug, m.name AS model_name
     FROM hosted_pause p LEFT JOIN models m ON m.slug = substr(p.ref, 1, instr(p.ref, '/') - 1) WHERE p.id = 1`,
  ).get() as PauseRow | undefined) ?? null;
}

export function getHardware(db: Db) {
  return {
    hardware: db.prepare('SELECT DISTINCT gpu_name, gpu_vram_mb, driver, cuda, power_limit_w, cpu_model, cpu_threads_visible, ram_mb FROM hardware_snapshots h WHERE EXISTS (SELECT 1 FROM runs r WHERE r.hardware_id = h.id)').all() as Row[],
    builds: db.prepare('SELECT DISTINCT engine, commit_sha FROM engine_builds e WHERE EXISTS (SELECT 1 FROM runs r WHERE r.engine_build_id = e.id) ORDER BY engine').all() as Row[],
  };
}

// -- metric selection ----------------------------------------------------------
const METHOD_PREFERENCE = ['http', 'llama-bench'];

export function pick(run: RunSummary | null, key: string, dims: Partial<Pick<MetricRow, 'depth' | 'n_prompt' | 'n_gen' | 'method'>> = {}): MetricRow | undefined {
  if (!run) return undefined;
  const matches = run.metrics.filter(
    (m) => m.key === key && Object.entries(dims).every(([k, v]) => (m as Record<string, unknown>)[k] === v),
  );
  const rank = (m: MetricRow) => {
    const i = METHOD_PREFERENCE.indexOf(m.method);
    return i === -1 ? METHOD_PREFERENCE.length : i;
  };
  return matches.sort((a, b) => rank(a) - rank(b))[0];
}

export function depthsFor(run: RunSummary | null, key: string): number[] {
  return [...new Set((run?.metrics ?? []).filter((m) => m.key === key).map((m) => m.depth))].sort((a, b) => a - b);
}

export type Headline = {
  pp0?: MetricRow;
  tg0?: MetricRow;
  tgDeep?: MetricRow;
  vram?: MetricRow;
  tokJ?: MetricRow;
  watts?: MetricRow;
};

/** llama-bench numbers of a config. Every pick names the method, so a chat-benchmark number never stands in. */
export function headline(cfg: ConfigView): Headline {
  const s = cfg.speed;
  const lb = { method: 'llama-bench' };
  const tgDepths = depthsFor(s, 'tg_tps');
  const deepest = tgDepths.at(-1);
  return {
    pp0: pick(s, 'pp_tps', { ...lb, depth: 0 }),
    tg0: pick(s, 'tg_tps', { ...lb, depth: 0 }),
    tgDeep: deepest ? pick(s, 'tg_tps', { ...lb, depth: deepest }) : undefined,
    vram: pick(s, 'vram_peak_mb', lb),
    tokJ: pick(s, 'tokens_per_joule', { ...lb, depth: 0, n_prompt: 0 }),
    watts: pick(s, 'gpu_w_avg', { ...lb, depth: 0, n_prompt: 0 }),
  };
}

/** The chat benchmark's cohorts: the model's default sampling (what pages lead with) and greedy decoding. */
export const CHAT_COHORTS = ['http-sampled', 'http-greedy'] as const;

export type ChatHeadline = {
  /** Decode t/s under default sampling: the like-for-like number across engines. */
  decode?: MetricRow;
  greedy?: MetricRow;
  ttft?: MetricRow;
  tokJ?: MetricRow;
  watts?: MetricRow;
  load?: MetricRow;
  vram?: MetricRow;
  ram?: MetricRow;
};

/** Chat-benchmark numbers of a config (its latest HTTP run). */
export function chatHeadline(cfg: ConfigView): ChatHeadline {
  const r = cfg.chat;
  const sampled = { method: 'http-sampled' };
  return {
    decode: pick(r, 'decode_tps', sampled),
    greedy: pick(r, 'decode_tps', { method: 'http-greedy' }),
    ttft: pick(r, 'ttft_ms', sampled),
    tokJ: pick(r, 'tokens_per_joule', sampled),
    watts: pick(r, 'gpu_w_avg', sampled),
    load: pick(r, 'load_s', { method: 'http' }),
    vram: pick(r, 'vram_peak_mb', { method: 'http' }),
    ram: pick(r, 'ram_peak_mb', { method: 'http' }),
  };
}

// -- capability scoring --------------------------------------------------------------------
/** An 8,000-token prompt under the 5-minute quick-tier limit: the allowance the site shows. */
export const HEADLINE_PROMPT_TOKENS = 8000;
export const QUICK_LIMIT_S = 300;

export type ScoredConfig = {
  m: ModelView;
  cfg: ConfigView;
  /** Percent solved per benchmark key, from the latest run of each tier. */
  scores: Record<string, BenchScore>;
  score: Score | null;
  quick: Score | null;
  /** Generation t/s at `speedDepth`, from llama-bench or the chat benchmark. */
  speed?: MetricRow;
  promptSpeed?: MetricRow;
  vram?: MetricRow;
  allowance: number | null;
  allowanceCapped: boolean;
  issuesPerNight: number | null;
  throttled: boolean;
};

/** Eval results of one run, as percentages keyed by task. */
function evalScores(run: RunSummary | null): Record<string, BenchScore> {
  const out: Record<string, BenchScore> = {};
  for (const e of run?.evals ?? []) {
    const bench = byKey.get(e.task);
    if (!bench) continue;
    const value = e.value * 100;
    out[e.task] = { value, stderr: e.stderr != null ? e.stderr * 100 : binomialStderr(value, e.n_tasks ?? bench.n) };
  }
  return out;
}

/**
 * Generation speed by depth for the allowance, preferring llama-bench's depth ladder. Without one, the chat
 * benchmark's default-sampling decode speed stands in: it is how the config serves chat.
 */
function speedPoints(cfg: ConfigView): SpeedPoint[] {
  const [run, key, method] = cfg.speed ? [cfg.speed, 'tg_tps', 'llama-bench'] : [cfg.chat, 'decode_tps', 'http-sampled'];
  return (run?.metrics ?? [])
    .filter((m) => m.key === key && m.method === method)
    .map((m) => ({ depth: m.depth, tps: m.value }));
}

/**
 * Every config with capability results, scored and ranked.
 *
 * Quick-tier runs count even when throttled: their allowances were fixed in advance from an earlier
 * speed run. Deep-tier runs are wall-clock limited, so a throttled one is reported but never ranked.
 */
export function scoredConfigs(models: ModelView[]): ScoredConfig[] {
  const out: ScoredConfig[] = [];
  for (const m of models) {
    for (const cfg of m.configs) {
      const quickRun = cfg.evalsQuick;
      const deepRun = cfg.evalsDeep;
      if (!quickRun && !deepRun) continue;
      const deepUsable = deepRun && !deepRun.throttled ? deepRun : null;
      const scores = { ...evalScores(quickRun), ...evalScores(deepUsable) };
      const speedRun = cfg.speed ?? cfg.chat;
      const depths = depthsFor(cfg.speed, 'tg_tps');
      const deepest = depths.at(-1);
      const speed = deepest != null
        ? pick(cfg.speed, 'tg_tps', { method: 'llama-bench', depth: deepest })
        : pick(cfg.chat, 'decode_tps', { method: 'http-sampled' });
      const promptSpeed = cfg.speed
        ? pick(cfg.speed, 'pp_tps', { method: 'llama-bench', depth: 0 })
        : pick(cfg.chat, 'prefill_tps', { method: 'http-sampled' });
      const ctx = Number(cfg.params.ctx ?? 0);
      const points = speedPoints(cfg);
      const rawAllowance =
        promptSpeed && points.length && ctx
          ? allowance({ pp0: promptSpeed.value, tg: points, ctx, promptTokens: HEADLINE_PROMPT_TOKENS, limitS: QUICK_LIMIT_S })
          : null;
      const secondsPerIssue = pick(deepUsable, 'eval_seconds_per_task', { method: 'swebench_verified' });
      const resolved = scores.swebench_verified?.value;
      out.push({
        m,
        cfg,
        scores,
        score: score(scores),
        quick: quickScore(scores),
        speed,
        promptSpeed,
        vram: pick(speedRun, 'vram_peak_mb', { method: cfg.speed ? 'llama-bench' : 'http' }),
        allowance: rawAllowance,
        allowanceCapped: rawAllowance != null && rawAllowance === ctx - HEADLINE_PROMPT_TOKENS,
        issuesPerNight:
          secondsPerIssue && resolved != null ? issuesPerNight(secondsPerIssue.value / 60, resolved) : null,
        throttled: !!(quickRun?.throttled || deepRun?.throttled),
      });
    }
  }
  return out;
}

/** A config whose latest llama-bench run may feed headline numbers: it exists and wasn't throttled. */
export function isUsable(cfg: ConfigView): boolean {
  return !!cfg.speed && !cfg.speed.throttled;
}

/** The same for the chat benchmark. */
export function isChatUsable(cfg: ConfigView): boolean {
  return !!cfg.chat && !cfg.chat.throttled;
}

/** A config with any speed numbers: llama-bench, the chat benchmark, or both. */
export function hasSpeed(cfg: ConfigView): boolean {
  return !!(cfg.speed || cfg.chat);
}

/**
 * The config that represents a model: non-throttled first, then fastest chat generation where it was
 * measured, then fastest llama-bench generation at empty context.
 */
export function bestConfig(model: ModelView): ConfigView | undefined {
  const usable = (cfg: ConfigView) => Number(isUsable(cfg) || isChatUsable(cfg));
  const chat = (cfg: ConfigView) => chatHeadline(cfg).decode?.value ?? -1;
  const tg0 = (cfg: ConfigView) => headline(cfg).tg0?.value ?? -1;
  return [...model.configs].sort((a, b) => usable(b) - usable(a) || chat(b) - chat(a) || tg0(b) - tg0(a))[0];
}

export type Highlight = { m: ModelView; cfg: ConfigView; metric: MetricRow };

export type SiteSummary = {
  models: number;
  configs: number;
  /** Fastest chat generation: the chat benchmark under default sampling, comparable across engines. */
  fastestChat?: Highlight;
  /** Fastest llama-bench generation at empty context (llama.cpp only). */
  fastest?: Highlight;
  deepest?: Highlight;
  /** Tokens per joule, from the chat benchmark when any config has one, otherwise from llama-bench. */
  efficient?: Highlight;
};

/**
 * Homepage headline numbers. Throttled runs never count, and each highlight compares one method only:
 * chat-benchmark numbers with each other, llama-bench numbers with each other.
 */
export function siteSummary(models: ModelView[]): SiteSummary {
  const all = models.flatMap((m) => m.configs.map((cfg) => ({ m, cfg })));
  const best = (
    pool: { m: ModelView; cfg: ConfigView }[],
    metric: (x: { m: ModelView; cfg: ConfigView }) => MetricRow | undefined,
  ): Highlight | undefined =>
    pool
      .map((x) => ({ ...x, metric: metric(x) }))
      .filter((x): x is Highlight => x.metric !== undefined)
      .sort((a, b) => b.metric.value - a.metric.value)[0];
  const bench = all.filter((x) => isUsable(x.cfg));
  const chat = all.filter((x) => isChatUsable(x.cfg));
  const deepest = Math.max(0, ...bench.flatMap((x) => depthsFor(x.cfg.speed, 'tg_tps')));
  const efficientChat = best(chat, (x) => chatHeadline(x.cfg).tokJ);
  return {
    models: models.filter((m) => m.configs.some(hasSpeed)).length,
    configs: all.filter((x) => hasSpeed(x.cfg)).length,
    fastestChat: best(chat, (x) => chatHeadline(x.cfg).decode),
    fastest: best(bench, (x) => headline(x.cfg).tg0),
    deepest: deepest > 0 ? best(bench, (x) => pick(x.cfg.speed, 'tg_tps', { method: 'llama-bench', depth: deepest })) : undefined,
    efficient: efficientChat ?? best(bench, (x) => headline(x.cfg).tokJ),
  };
}
