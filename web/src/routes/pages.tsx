import { Hono, type Context } from 'hono';
import type { Db } from '../db.ts';
import { DASH, date, depthLabel, engineLabel, gb, methodLabel, num, testLabel } from '../format.ts';
import {
  CHAT_COHORTS, HEADLINE_PROMPT_TOKENS, bestConfig, chatHeadline, depthsFor, getHardware, getHosted, getModel, getModels,
  getPause, getRun, hasSpeed, headline, pick, scoredConfigs, siteSummary,
  type ChatHeadline, type ConfigView, type EvalRow, type Headline, type MetricRow, type ModelView, type RunSummary, type ScoredConfig,
} from '../queries.ts';
import { BENCHMARKS, CATEGORIES, byKey, rank } from '../scoring.ts';
import { siteConfig } from '../site.ts';
import { hostingView, type ProbeFn } from '../status.ts';
import { Chart, HardwareFooter, Layout, Scoreboard, Tile, type ChartSpec, type ScoreboardRow, type ScoreboardSpec } from '../views/layout.tsx';
import { Methodology } from '../views/methodology.tsx';
import { StatusLine } from '../views/status.tsx';

const MAX_SLOTS = 8;

const INTRO =
  "Local Inference is my lab for running open-weight language models on a single RTX 3090. I tune each model's settings (context length, KV-cache precision, offload), then rank them on the work I actually want done: writing code, driving tools, reasoning. Speed counts here because it buys thinking time, not because a quick answer is worth more \u2014 every answer has to finish inside a token allowance taken from that config's own measured speed. Every number comes from a scripted, hash-locked config, so results are reproducible.";

type Row = { m: ModelView; cfg: ConfigView; h: Headline; ch: ChatHeadline };

const toRow = (m: ModelView, cfg: ConfigView): Row => ({ m, cfg, h: headline(cfg), ch: chatHeadline(cfg) });

const COLUMNS: { key: string; label: string; numeric?: boolean; value: (r: Row) => number | string | null | undefined; show: (r: Row) => string }[] = [
  { key: 'model', label: 'Model', value: (r) => r.m.name, show: (r) => r.m.name },
  { key: 'engine', label: 'Engine', value: (r) => engineLabel(r.m.engine), show: (r) => engineLabel(r.m.engine) },
  { key: 'size', label: 'Size', numeric: true, value: (r) => r.m.file_size_bytes, show: (r) => gb(r.m.file_size_bytes) },
  { key: 'config', label: 'Config', value: (r) => r.cfg.name, show: (r) => r.cfg.name },
  { key: 'ctx', label: 'Ctx', numeric: true, value: (r) => r.cfg.params.ctx as number, show: (r) => depthLabel(Number(r.cfg.params.ctx ?? 0)) },
  { key: 'chat', label: 'Chat t/s', numeric: true, value: (r) => r.ch.decode?.value, show: (r) => num(r.ch.decode?.value) },
  { key: 'ttft', label: 'TTFT ms', numeric: true, value: (r) => r.ch.ttft?.value, show: (r) => num(r.ch.ttft?.value, 0) },
  { key: 'pp0', label: 'PP t/s', numeric: true, value: (r) => r.h.pp0?.value, show: (r) => num(r.h.pp0?.value) },
  { key: 'tg0', label: 'TG t/s', numeric: true, value: (r) => r.h.tg0?.value, show: (r) => num(r.h.tg0?.value) },
  { key: 'tgDeep', label: 'TG t/s deep', numeric: true, value: (r) => r.h.tgDeep?.value, show: (r) => (r.h.tgDeep ? `${num(r.h.tgDeep.value)} @${depthLabel(r.h.tgDeep.depth)}` : DASH) },
  { key: 'vram', label: 'VRAM', numeric: true, value: (r) => r.h.vram?.value, show: (r) => (r.h.vram ? `${num(r.h.vram.value / 1024, 1)} GB` : DASH) },
  { key: 'tokJ', label: 'Tok/J', numeric: true, value: (r) => r.h.tokJ?.value, show: (r) => num(r.h.tokJ?.value) },
];

/** How a run was measured, for history tables and run pages. */
function runKind(r: Pick<RunSummary, 'kind' | 'tier' | 'metrics'>): string {
  if (r.kind === 'speed') return r.metrics.some((m) => m.method.startsWith('http')) ? 'chat benchmark' : 'speed (llama-bench)';
  return `${r.kind}${r.tier ? ` (${r.tier})` : ''}`;
}

const sd = (m?: MetricRow) => (m?.stddev != null ? `± ${num(m.stddev)}` : undefined);

function modelUrl(m: ModelView, cfg?: ConfigView) {
  return `/m/${encodeURIComponent(m.slug)}${cfg ? `?c=${encodeURIComponent(cfg.slug)}` : ''}`;
}

function qs(current: Record<string, string>, patch: Record<string, string | undefined>) {
  const p = new URLSearchParams({ ...current });
  for (const [k, v] of Object.entries(patch)) (v === undefined || v === '' ? p.delete(k) : p.set(k, v));
  const s = p.toString();
  return s ? `?${s}` : '?';
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

const CODING_INTRO =
  'Every config ranked for coding work on one RTX 3090. The score weights coding 50%, agents & tools 30% and reasoning 20%, and combines them as a weighted geometric mean, so being weak in any one area pulls it down. Answers must finish within a thinking allowance taken from each config\u2019s own measured speed, so faster configs get more room to reason.';

// -- capability scoreboard -----------------------------------------------------------------
/** Benchmarks in the order the scoreboard and the table show them: by category, then registry order. */
const BENCH_ORDER = CATEGORIES.flatMap((cat) => BENCHMARKS.filter((b) => b.category === cat.key));

/** A category's first benchmark is drawn as a circle, its second as a diamond. */
const SHAPES = new Map<string, 'circle' | 'diamond'>(
  CATEGORIES.flatMap((cat) =>
    BENCHMARKS.filter((b) => b.category === cat.key).map((b, i): [string, 'circle' | 'diamond'] => [b.key, i === 0 ? 'circle' : 'diamond']),
  ),
);

const catName = (key: string) => CATEGORIES.find((c) => c.key === key)?.name ?? key;

function allowanceText(tokens: number | null, capped: boolean): string {
  if (tokens == null) return DASH;
  const text = tokens < 1000 ? `${num(tokens, 0)} tokens` : `${num(tokens / 1000, 1)}k tokens`;
  return capped ? `${text} (context limit)` : text;
}

/** A scoreboard row plus the config behind it, which only the server-rendered table needs. */
type BoardRow = ScoreboardRow & { s: ScoredConfig };
type BoardGroup = { title: string; rows: BoardRow[] };

/** Ranked configs first, then the ones still missing their deep tier, sorted by quick score. */
function boardGroups(scored: ScoredConfig[]): BoardGroup[] {
  const row = (s: ScoredConfig, rankLabel: string, tieNote?: string): BoardRow => ({
    s,
    name: s.m.name,
    config: s.cfg.name,
    rankLabel,
    tieNote,
    scores: s.scores,
    score: s.score,
    quick: s.quick!,
    speed: s.speed?.value ?? null,
    promptSpeed: s.promptSpeed?.value ?? null,
    allowance: s.allowance,
    allowanceText: allowanceText(s.allowance, s.allowanceCapped),
    issuesPerNight: s.issuesPerNight,
    vramGb: s.vram ? s.vram.value / 1024 : null,
  });
  // A config with no quick tier has nothing to compare on either axis, so it stays off the board.
  const shown = scored.filter((s) => s.quick);
  const ranked = rank(shown.filter((s) => s.score).map((s) => ({ ...s, name: s.m.name, score: s.score! })));
  const unranked = [...shown.filter((s) => !s.score)].sort((a, b) => b.quick!.value - a.quick!.value);
  return [
    {
      title: 'Ranked \u00b7 all six benchmarks',
      rows: ranked.map((r) =>
        row(r, r.rankLabel, r.tiedWith.length ? `Too close to call with ${r.tiedWith.join(' and ')}: the scores are within one combined standard error.` : undefined),
      ),
    },
    { title: 'Not ranked yet \u00b7 quick tier only', rows: unranked.map((s) => row(s, '\u2013')) },
  ].filter((g) => g.rows.length > 0);
}

/** The context depth every speed bar was measured at, when the rows agree on one. */
function sharedSpeedDepth(groups: BoardGroup[]): number | null {
  const depths = [...new Set(groups.flatMap((g) => g.rows.map((r) => r.s.speed?.depth)).filter((d): d is number => d != null))];
  return depths.length === 1 ? depths[0] : null;
}

const speedLabel = (depth: number | null) =>
  depth == null ? 'Generation, deepest context measured \u00b7 t/s'
    : depth === 0 ? 'Generation at empty context \u00b7 t/s'
      : `Generation at ${depthLabel(depth)} context \u00b7 t/s`;

function boardSpec(groups: BoardGroup[], depth: number | null): ScoreboardSpec {
  return {
    categories: CATEGORIES.map((c) => ({ key: c.key, name: c.name, weight: c.weight })),
    benchmarks: BENCH_ORDER.map((b) => ({ key: b.key, name: b.name, category: b.category, tier: b.tier, size: b.size, shape: SHAPES.get(b.key)! })),
    // The spec is inlined into the page, so each row carries only the numbers the drawing needs.
    groups: groups.map((g) => ({ title: g.title, rows: g.rows.map(({ s, ...rest }) => rest) })),
    speedLabel: speedLabel(depth),
  };
}

/** Branded 404, registered on the top-level pages app (see app.ts). */
export function notFoundPage(c: Context) {
  return c.html(
    <Layout title="Not found" tab="home" path={c.req.path}>
      <h1>Not found</h1>
      <p><a href="/">Homepage</a> · <a href="/benchmarks">All benchmarks</a></p>
    </Layout>,
    404,
  );
}

export function pageRoutes(db: Db, probe: ProbeFn) {
  const app = new Hono();
  const hosting = async () => hostingView(await probe(), getHosted(db), getPause(db));

  // -- homepage: intro, headline numbers, chat and benchmarks entries -------------------
  app.get('/', async (c) => {
    const models = getModels(db);
    const summary = siteSummary(models);
    const view = await hosting();
    const { chatUrl } = siteConfig();
    const modelLink = (x?: { m: ModelView; cfg: ConfigView }) => x && <a href={modelUrl(x.m, x.cfg)}>{x.m.name}</a>;
    const withEngine = (x: { m: ModelView; cfg: ConfigView }) => <>{modelLink(x)} · {engineLabel(x.m.engine)}</>;
    // The lead speed tile compares like with like: the chat benchmark across engines when there is one.
    const lead = summary.fastestChat
      ? { label: 'Fastest chat generation', x: summary.fastestChat }
      : summary.fastest && { label: 'Fastest generation', x: summary.fastest };

    // Capability leads once a config has all six benchmarks; until then the speed tiles stand.
    const ranked = scoredConfigs(models).filter((s) => s.score);
    const bestBy = (value: (s: ScoredConfig) => number | null | undefined) =>
      ranked
        .map((s) => ({ s, v: value(s) }))
        .filter((x): x is { s: ScoredConfig; v: number } => x.v != null)
        .sort((a, b) => b.v - a.v)[0];
    const bestScore = bestBy((s) => s.score!.value);
    const bestCategory = (key: string) => bestBy((s) => s.score!.categories.find((cat) => cat.key === key)?.mean);
    const bestNight = bestBy((s) => s.issuesPerNight);

    return c.html(
      <Layout title="Local Inference · open-weight LLMs on one RTX 3090" tab="home" path="/" footer={<HardwareFooter {...getHardware(db)} />}>
        <section class="intro">
          <h1>Local Inference</h1>
          <p class="lead">{INTRO}</p>
          <p><a href="/methodology">How it's measured →</a></p>
        </section>

        {bestScore ? (
          <section class="tiles" aria-label="Headline results">
            <Tile
              label="Best for coding work"
              value={num(bestScore.v, 1)}
              unit={`\u00b1 ${num(bestScore.s.score!.stderr, 1)}`}
              note={modelLink(bestScore.s)}
            />
            {CATEGORIES.filter((cat) => cat.key !== 'reasoning').map((cat) => {
              const b = bestCategory(cat.key);
              return b && <Tile label={`Best at ${cat.name.toLowerCase()}`} value={num(b.v, 1)} unit="% solved" note={modelLink(b.s)} />;
            })}
            {bestNight && <Tile label="Most issues fixed per night" value={num(bestNight.v, 0)} unit="issues" note={modelLink(bestNight.s)} />}
          </section>
        ) : lead ? (
          <section class="tiles" aria-label="Headline results">
            <Tile label={lead.label} value={num(lead.x.metric.value)} unit="t/s" note={withEngine(lead.x)} />
            {summary.deepest && (
              <Tile label={`Generation at ${depthLabel(summary.deepest.metric.depth)} context`} value={num(summary.deepest.metric.value)} unit="t/s" note={<>{modelLink(summary.deepest)} · llama-bench</>} />
            )}
            {summary.efficient && (
              <Tile
                label={summary.efficient.metric.method === 'llama-bench' ? 'Best efficiency' : 'Best chat efficiency'}
                value={num(summary.efficient.metric.value)}
                unit="tok/J"
                note={withEngine(summary.efficient)}
              />
            )}
            <Tile label="Benchmarked" value={String(summary.models)} unit={summary.models === 1 ? 'model' : 'models'} note={plural(summary.configs, 'config')} />
          </section>
        ) : (
          <section class="empty"><p>No published results yet.</p></section>
        )}

        <section class="cards">
          <article class="card">
            <h2>Chat</h2>
            <p>Talk to the model this machine is hosting right now, running entirely on the RTX 3090.</p>
            <StatusLine view={view} />
            <p class="muted small">Private: sign-in required.</p>
            {chatUrl && <a class="button primary" href={chatUrl}>Open chat ↗</a>}
          </article>
          <article class="card">
            <h2>Benchmarks</h2>
            <p>Every model and config ranked for coding work, and the speed behind that ranking: chat generation on every engine, generation as the context fills, VRAM, power and efficiency, with the exact launch command for each.</p>
            <a class="button primary" href="/benchmarks">View benchmarks →</a>
          </article>
        </section>
      </Layout>,
    );
  });

  // -- benchmarks: capability first, speed alongside --------------------------------------
  app.get('/benchmarks', async (c) => {
    const q = c.req.query();
    const models = getModels(db);
    const view = await hosting();
    const { chatUrl } = siteConfig();
    const engines = [...new Set(models.map((m) => m.engine))].sort();
    const bases = [...new Map(models.map((m) => [m.base.slug, m.base.name])).entries()].sort((a, b) => a[1].localeCompare(b[1]));
    const matches = (m: ModelView) => (!q.engine || m.engine === q.engine) && (!q.arch || m.base.arch === q.arch) && (!q.base || m.base.slug === q.base);

    const groups = boardGroups(scoredConfigs(models).filter((s) => matches(s.m)));
    const depth = sharedSpeedDepth(groups);
    const speedOnly = models.filter(matches).flatMap((m) => m.configs).filter((cfg) => hasSpeed(cfg) && !cfg.evalsQuick && !cfg.evalsDeep).length;
    // Capability is the default view, but an empty scoreboard helps nobody: fall back until it has rows.
    const tab = q.view === 'speed' || (q.view !== 'coding' && groups.length === 0) ? 'speed' : 'coding';

    let rows: Row[] = q.all
      ? models.flatMap((m) => m.configs.filter(hasSpeed).map((cfg) => toRow(m, cfg)))
      : models.flatMap((m) => {
          const cfg = bestConfig(m);
          return cfg && hasSpeed(cfg) ? [toRow(m, cfg)] : [];
        });
    rows = rows.filter((r) => matches(r.m));

    // Default order: chat generation, then llama-bench generation for configs without a chat benchmark.
    const sortCol = COLUMNS.find((col) => col.key === q.sort) ?? COLUMNS.find((col) => col.key === 'chat')!;
    const dir = q.dir === 'asc' ? 1 : q.dir === 'desc' ? -1 : sortCol.numeric ? -1 : 1;
    const tg0 = COLUMNS.find((col) => col.key === 'tg0')!;
    const compare = (col: typeof sortCol, d: number, a: Row, b: Row): number => {
      const va = col.value(a), vb = col.value(b);
      if (va == null || vb == null) return va == null && vb == null ? 0 : va == null ? 1 : -1;
      return (typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))) * d;
    };
    rows.sort((a, b) => compare(sortCol, dir, a, b) || compare(tg0, -1, a, b));

    const rowLabel = (r: Row) => (q.all ? `${r.m.name} · ${r.cfg.name}` : r.m.name);
    const byChat = rows.filter((r) => r.ch.decode).sort((a, b) => b.ch.decode!.value - a.ch.decode!.value);
    const chatBar: ChartSpec = {
      type: 'bar-h', unit: 't/s', xLabel: 'tokens / second',
      categories: byChat.map((r) => `${rowLabel(r)} · ${engineLabel(r.m.engine)}`), links: byChat.map((r) => modelUrl(r.m, r.cfg)),
      series: [{ label: 'Chat generation', slot: 1, data: byChat.map((r) => r.ch.decode!.value), stddev: byChat.map((r) => r.ch.decode!.stddev) }],
    };
    const withTg = rows.filter((r) => r.h.tg0);
    const byTg = [...withTg].sort((a, b) => b.h.tg0!.value - a.h.tg0!.value);
    const tgBar: ChartSpec = {
      type: 'bar-h', unit: 't/s', xLabel: 'tokens / second',
      categories: byTg.map(rowLabel), links: byTg.map((r) => modelUrl(r.m, r.cfg)),
      series: [{ label: 'Generation', slot: 1, data: byTg.map((r) => r.h.tg0!.value), stddev: byTg.map((r) => r.h.tg0!.stddev) }],
    };
    const sizeScatter: ChartSpec = {
      type: 'scatter', unit: 't/s', xLabel: 'weights (GB)', yLabel: 'generation t/s',
      links: withTg.map((r) => modelUrl(r.m, r.cfg)),
      series: [{ label: 'Models', slot: 1, data: withTg.filter((r) => r.m.file_size_bytes).map((r) => ({ x: r.m.file_size_bytes! / 1e9, y: r.h.tg0!.value, label: rowLabel(r) })) }],
    };

    return c.html(
      <Layout
        title="Benchmarks"
        tab="benchmarks"
        path="/benchmarks"
        description={
          tab === 'coding'
            ? 'Open-weight models ranked for the work they do on one RTX 3090: coding, agents and tools, and reasoning, each answered inside a thinking allowance earned from the config’s own measured speed.'
            : 'Every model and config benchmarked on one RTX 3090, side by side: generation and prompt speed, context depth, VRAM, power and tokens per joule.'
        }
        charts={tab === 'speed'}
        scoreboard={tab === 'coding'}
        footer={<HardwareFooter {...getHardware(db)} />}
      >
        <section class="banner">
          <StatusLine view={view} />
          {chatUrl && <a class="button" href={chatUrl}>Open chat ↗</a>}
        </section>
        <h1>Model benchmarks</h1>
        {models.length === 0 ? (
          <section class="empty">
            <p>No published results yet.</p>
          </section>
        ) : (
          <>
            <nav class="tabs" aria-label="Benchmark view">
              <a href={qs(q, { view: 'coding' })} aria-current={tab === 'coding' ? 'page' : undefined}>Coding work</a>
              <a href={qs(q, { view: 'speed' })} aria-current={tab === 'speed' ? 'page' : undefined}>Speed</a>
            </nav>
            <p class="meta">
              {tab === 'coding' ? CODING_INTRO : 'Local model performance, measured on one machine.'} <a href="/methodology">How it's measured</a>
            </p>

            <form class="filters" method="get">
              <label>Engine <select name="engine"><option value="">All</option>{engines.map((e) => <option value={e} selected={q.engine === e}>{engineLabel(e)}</option>)}</select></label>
              <label>Base model <select name="base"><option value="">All</option>{bases.map(([slug, name]) => <option value={slug} selected={q.base === slug}>{name}</option>)}</select></label>
              <label>Architecture <select name="arch"><option value="">All</option><option value="dense" selected={q.arch === 'dense'}>Dense</option><option value="moe" selected={q.arch === 'moe'}>MoE</option></select></label>
              {tab === 'speed'
                ? <label class="check"><input type="checkbox" name="all" value="1" checked={!!q.all} /> Show every config</label>
                : q.all && <input type="hidden" name="all" value={q.all} />}
              <input type="hidden" name="view" value={tab} />
              {q.sort && <input type="hidden" name="sort" value={q.sort} />}
              {q.dir && <input type="hidden" name="dir" value={q.dir} />}
              <button type="submit">Apply</button>
            </form>

            {tab === 'coding' ? (
              groups.length === 0 ? (
                <section class="empty">
                  <p>
                    No capability evals published yet. Every config gets the quick tier overnight; until then, the{' '}
                    <a href={qs(q, { view: 'speed' })}>Speed tab</a> has what each one delivers.
                  </p>
                </section>
              ) : (
                <>
                  <Scoreboard
                    id="scoreboard"
                    title="Coding work, every config"
                    subtitle="Rows are ranked by score. Each row has three lines, top to bottom: coding, agents & tools, reasoning. Hover or tap a mark for details, or tab through the rows. The table below lists every number."
                    spec={boardSpec(groups, depth)}
                  />

                  <h2>Capability benchmarks</h2>
                  <p class="meta">Percent of tasks solved. The score carries ± one standard error, and rows sharing a rank are too close to call.</p>
                  <CapabilityTable groups={groups} depth={depth} />
                  {speedOnly > 0 && (
                    <p class="muted small">
                      {plural(speedOnly, 'config')} {speedOnly === 1 ? 'has' : 'have'} speed numbers but no capability evals yet; they are on the{' '}
                      <a href={qs(q, { view: 'speed' })}>Speed tab</a>.
                    </p>
                  )}

                  <section class="notes">
                    <h2>How the score works</h2>
                    <ul>
                      <li><strong>Quick tier</strong>, every config: LiveCodeBench (100 problems), BFCL (400 cases), GPQA Diamond (198 questions) and AIME 2025 (30 problems × 4 attempts). Limit: 5 minutes per task.</li>
                      <li><strong>Deep tier</strong>, only for configs worth a night each: SWE-bench Verified (30 issues, solved by mini-SWE-agent) and Terminal-Bench (30 tasks). Limit: 20 minutes per task. Configs without it aren't ranked, because quick-tier scores aren't comparable with full ones.</li>
                      <li><strong>Thinking allowance</strong> = (time limit − time to read the prompt) × generation speed at that depth, capped by the config's context. The table shows it for an 8k-token prompt in a 5-minute task. An answer that doesn't finish inside it counts as wrong.</li>
                      <li><strong>Too close to call</strong>: scores within one combined standard error of each other share a rank, shown as "=2".</li>
                      <li><strong>Issues fixed per night</strong> = 8 hours ÷ average time per SWE-bench issue × fix rate.</li>
                    </ul>
                  </section>
                </>
              )
            ) : (
              <>
                <p class="muted small">
                  {q.all ? 'Every benchmarked config.' : 'One row per model, using its fastest config (chat generation where measured, otherwise llama-bench generation at empty context).'}{' '}
                  Chat t/s and TTFT come from the chat benchmark: the same eight real prompts sent to every engine through its HTTP
                  API, one at a time, so they compare across engines. PP, TG, VRAM and tok/J are means of repeated llama-bench runs,
                  which only llama.cpp has.
                </p>

                <div class="table-wrap">
                  <table class="data">
                    <thead>
                      <tr>
                        {COLUMNS.map((col) => {
                          const active = col.key === sortCol.key;
                          const nextDir = active ? (dir === -1 ? 'asc' : 'desc') : col.numeric ? 'desc' : 'asc';
                          return (
                            <th class={col.numeric ? 'num' : ''} aria-sort={active ? (dir === -1 ? 'descending' : 'ascending') : undefined}>
                              <a href={qs(q, { sort: col.key, dir: nextDir })}>{col.label}{active ? (dir === -1 ? ' ↓' : ' ↑') : ''}</a>
                            </th>
                          );
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => (
                        <tr>
                          {COLUMNS.map((col) =>
                            col.key === 'model' ? (
                              <td>
                                <a href={modelUrl(r.m, r.cfg)}>{r.m.name}</a>
                                {(r.cfg.speed?.throttled || r.cfg.chat?.throttled) && <span class="tag">throttled</span>}
                                <div class="muted small">{r.m.quant} · {r.m.base.arch}</div>
                              </td>
                            ) : (
                              <td class={col.numeric ? 'num' : ''}>{col.show(r)}</td>
                            ),
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {(byChat.length > 0 || withTg.length > 0) && (
                  <div class="grid-2">
                    {byChat.length > 0 && (
                      <Chart id="chart-chat" title="Chat generation speed" subtitle="Decode tokens per second in the chat benchmark, default sampling, every engine. Click a bar to open the model." spec={chatBar} height={Math.max(160, byChat.length * 36 + 70)} />
                    )}
                    {withTg.length > 0 && (
                      <Chart id="chart-tg" title="Generation speed (llama-bench)" subtitle="Tokens per second at empty context. Click a bar to open the model." spec={tgBar} height={Math.max(160, byTg.length * 36 + 70)} />
                    )}
                    {withTg.length > 0 && (
                      <Chart id="chart-size" title="Speed vs weight size" subtitle="llama-bench generation. Bigger weights mean more memory traffic per token." spec={sizeScatter} />
                    )}
                  </div>
                )}
              </>
            )}
          </>
        )}
      </Layout>,
    );
  });

  // -- methodology -------------------------------------------------------------------------
  app.get('/methodology', (c) =>
    c.html(
      <Layout
        title="How it's measured"
        tab="benchmarks"
        path="/methodology"
        description="How Local Inference benchmarks models on one RTX 3090: the chat benchmark every engine runs, llama-bench settings, explicit engine flags, GPU telemetry, power and efficiency, throttling and limits."
      >
        <Methodology {...getHardware(db)} />
      </Layout>,
    ),
  );

  // -- model page: switch between configs --------------------------------------
  app.get('/m/:slug', (c) => {
    const model = getModel(db, c.req.param('slug'));
    if (!model) return c.notFound();
    const scored = scoredConfigs([model]);
    const topRanked = scored.filter((s) => s.score).sort((a, b) => b.score!.value - a.score!.value)[0];
    const cfg = model.configs.find((x) => x.slug === c.req.query('c')) ?? topRanked?.cfg ?? bestConfig(model) ?? model.configs[0];
    if (!cfg) return c.notFound();
    const h = headline(cfg);
    const ch = chatHeadline(cfg);
    const emphasis = model.configs.length > MAX_SLOTS;
    const vllm = model.engine === 'vllm';

    const paramKeys = [...new Set(model.configs.flatMap((x) => Object.keys(x.params)))];
    const differs = (k: string) => new Set(model.configs.map((x) => JSON.stringify(x.params[k] ?? null))).size > 1;
    const showVal = (v: unknown) => (v == null ? DASH : typeof v === 'object' ? JSON.stringify(v) : String(v));

    const depthChart = (key: 'tg_tps' | 'pp_tps'): ChartSpec => {
      const depths = [...new Set(model.configs.flatMap((x) => depthsFor(x.speed, key)))].sort((a, b) => a - b);
      const shown = model.configs.filter((x) => x.speed);
      return {
        type: 'line', unit: 't/s', xLabel: 'context depth (tokens already in cache)', yLabel: 'tokens / second',
        categories: depths.map(depthLabel),
        series: shown.map((x) => {
          const pts = depths.map((d) => pick(x.speed, key, { depth: d }));
          return {
            label: x.name,
            slot: emphasis ? (x.id === cfg.id ? 1 : 'muted') : model.configs.indexOf(x) + 1,
            dim: !emphasis && x.id !== cfg.id,
            data: pts.map((p) => p?.value ?? null),
            stddev: pts.map((p) => p?.stddev ?? null),
          };
        }),
      };
    };

    const teleChart = (run: RunSummary, field: string, label: string, unit: string): ChartSpec => ({
      type: 'line', unit, xLabel: 'seconds into run', yLabel: label,
      series: [{ label, slot: 1, data: run.telemetry.map((s) => ({ x: s.t, y: field === 'vram_mb' ? s[field] / 1024 : s[field] })) }],
    });
    const teleCharts = (run: RunSummary, prefix: string) =>
      run.telemetry.length > 1 && (
        <div class="grid-2">
          <Chart id={`${prefix}-power`} title="GPU power during the run" spec={teleChart(run, 'power_w', 'GPU power', 'W')} height={200} />
          <Chart id={`${prefix}-vram`} title="VRAM during the run" spec={teleChart(run, 'vram_mb', 'VRAM', 'GB')} height={200} />
        </div>
      );

    // Every config of the same base model with a chat benchmark, on any engine: the like-for-like comparison.
    const sameBase = getModels(db)
      .filter((m) => m.base.slug === model.base.slug)
      .flatMap((m) => m.configs.map((x) => ({ m, x, ch: chatHeadline(x) })))
      .filter((r) => r.ch.decode)
      .sort((a, b) => b.ch.decode!.value - a.ch.decode!.value);
    const engineChart: ChartSpec = {
      type: 'bar-h', unit: 't/s', xLabel: 'decode tokens / second',
      categories: sameBase.map((r) => `${r.m.name} · ${r.x.name} · ${engineLabel(r.m.engine)}`),
      links: sameBase.map((r) => modelUrl(r.m, r.x)),
      series: CHAT_COHORTS.map((method, i) => {
        const pts = sameBase.map((r) => pick(r.x.chat, 'decode_tps', { method }));
        return { label: methodLabel(method), slot: i + 1, data: pts.map((p) => p?.value ?? null), stddev: pts.map((p) => p?.stddev ?? null) };
      }),
    };
    const anyAccept = CHAT_COHORTS.some((method) => pick(cfg.chat, 'spec_accept_len', { method }));

    const speedRows = (cfg.speed?.metrics ?? []).filter((m) => m.key === 'pp_tps' || m.key === 'tg_tps');

    return c.html(
      <Layout
        title={model.name}
        tab="benchmarks"
        path={modelUrl(model, cfg)}
        description={
          ch.decode
            ? `${model.name} (${cfg.name}, ${engineLabel(model.engine)}) on a single RTX 3090: ${num(ch.decode.value)} t/s chat generation. Launch command, settings, chat benchmark, power and efficiency.`
            : `${model.name} (${cfg.name}) on a single RTX 3090: ${num(h.tg0?.value)} t/s generation at empty context. Launch command, settings, speed by context depth, power and efficiency.`
        }
        charts
        footer={<HardwareFooter {...getHardware(db)} />}
      >
        <p class="crumb"><a href="/benchmarks">← All models</a></p>
        <h1>{model.name}</h1>
        <p class="meta">
          {model.base.name} · {model.base.arch === 'moe' ? `MoE ${num(model.base.params_b, 1)}B (${num(model.base.active_params_b, 1)}B active)` : model.base.params_b ? `${num(model.base.params_b, 1)}B dense` : 'dense'} ·{' '}
          {engineLabel(model.engine)} · {model.format.toUpperCase()} {model.quant} · {gb(model.file_size_bytes)}
          {model.source_repo && (
            <> · <a href={`https://huggingface.co/${model.source_repo}`}>{model.source_repo}</a>{model.source_revision && <span class="muted"> @{model.source_revision.slice(0, 7)}</span>}</>
          )}
        </p>
        {model.notes && <p>{model.notes}</p>}

        <nav class="tabs" aria-label="Benchmarked settings">
          {model.configs.map((x) => (
            <a href={`?c=${encodeURIComponent(x.slug)}`} aria-current={x.id === cfg.id ? 'page' : undefined}>{x.name}</a>
          ))}
        </nav>

        {cfg.chat ? (
          <section class="tiles">
            <Tile label="Chat generation" value={num(ch.decode?.value)} unit="t/s" note={['default sampling', sd(ch.decode)].filter(Boolean).join(' ')} />
            <Tile label="Chat generation, greedy" value={num(ch.greedy?.value)} unit="t/s" note={sd(ch.greedy)} />
            <Tile label="Time to first token" value={num(ch.ttft?.value, 0)} unit="ms" />
            <Tile label="Peak VRAM" value={ch.vram ? num(ch.vram.value / 1024, 1) : DASH} unit="GB" note={ch.vram ? (vllm ? 'preallocated at startup' : 'serving the chat benchmark') : undefined} />
            <Tile label="GPU power while generating" value={num(ch.watts?.value)} unit="W" />
            <Tile label="Energy efficiency" value={num(ch.tokJ?.value)} unit="tok/J" note="chat benchmark" />
          </section>
        ) : (
          <section class="tiles">
            <Tile label="Generation, empty context" value={num(h.tg0?.value)} unit="t/s" note={sd(h.tg0)} />
            <Tile label={h.tgDeep ? `Generation at ${depthLabel(h.tgDeep.depth)} context` : 'Generation, deep context'} value={num(h.tgDeep?.value)} unit="t/s" />
            <Tile label="Prompt processing" value={num(h.pp0?.value)} unit="t/s" />
            <Tile label="Peak VRAM" value={h.vram ? num(h.vram.value / 1024, 1) : DASH} unit="GB" note={h.vram ? 'during llama-bench' : undefined} />
            <Tile label="GPU power while generating" value={num(h.watts?.value)} unit="W" />
            <Tile label="Energy efficiency" value={num(h.tokJ?.value)} unit="tok/J" />
          </section>
        )}
        {(cfg.speed?.throttled || cfg.chat?.throttled) && <p class="warn">⚠ A speed run hit thermal or power-brake throttling; its numbers may be low.</p>}

        <section>
          <h2>Settings</h2>
          {cfg.notes && <p>{cfg.notes}</p>}
          <div class="launch">
            <pre id="launch-cmd">{cfg.launch_command}</pre>
            <button type="button" data-copy="launch-cmd">Copy</button>
          </div>
          <div class="table-wrap">
            <table class="data params">
              <thead><tr><th>Setting</th><th>Value</th></tr></thead>
              <tbody>
                {paramKeys.map((k) => (
                  <tr class={differs(k) ? 'diff' : ''}>
                    <td><code>{k}</code></td>
                    <td>{showVal(cfg.params[k])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {model.configs.length > 1 && <p class="muted small">Highlighted rows differ between this model's configs.</p>}
        </section>

        {cfg.chat && (
          <section>
            <h2>Chat benchmark</h2>
            <p class="muted small">
              Eight real chat prompts sent through the server's HTTP API, one at a time, up to 1,024 tokens each, thinking off.
              Every engine runs the same protocol, so these numbers compare across engines. <a href="/methodology#chat-benchmark">How it's measured</a>
            </p>
            <div class="table-wrap">
              <table class="data">
                <thead>
                  <tr>
                    <th>Sampling</th><th class="num">Decode t/s</th><th class="num">± sd</th><th class="num">TPOT ms</th><th class="num">TTFT ms</th>
                    <th class="num">Prefill t/s</th><th class="num">End-to-end t/s</th><th class="num">Output tokens</th><th class="num">GPU W</th><th class="num">Tok/J</th>
                    {anyAccept && <th class="num">Accepted / step</th>}
                  </tr>
                </thead>
                <tbody>
                  {CHAT_COHORTS.filter((method) => pick(cfg.chat, 'decode_tps', { method })).map((method) => {
                    const m = (key: string) => pick(cfg.chat, key, { method });
                    const decode = m('decode_tps');
                    return (
                      <tr>
                        <td>{methodLabel(method)}{decode?.n ? <span class="muted small"> · {decode.n} prompts</span> : ''}</td>
                        <td class="num">{num(decode?.value)}</td>
                        <td class="num">{num(decode?.stddev)}</td>
                        <td class="num">{num(m('tpot_ms')?.value)}</td>
                        <td class="num">{num(m('ttft_ms')?.value, 0)}</td>
                        <td class="num">{num(m('prefill_tps')?.value, 0)}</td>
                        <td class="num">{num(m('e2e_tps')?.value)}</td>
                        <td class="num">{num(m('out_tokens_mean')?.value, 0)}</td>
                        <td class="num">{num(m('gpu_w_avg')?.value, 0)}</td>
                        <td class="num">{num(m('tokens_per_joule')?.value)}</td>
                        {anyAccept && <td class="num">{num(m('spec_accept_len')?.value)}</td>}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p class="muted small">
              Load time {ch.load ? `${num(ch.load.value, 0)} s` : DASH} · Peak VRAM {ch.vram ? `${num(ch.vram.value / 1024, 1)} GB` : DASH}
              {vllm && ch.vram ? ' (preallocated: vLLM reserves its KV-cache pool at startup)' : ''} · Peak RAM{' '}
              {ch.ram ? `${num(ch.ram.value / 1024, 1)} GB` : DASH} · <a href={`/runs/${cfg.chat.id}`}>Run of {date(cfg.chat.started_at)}</a>
            </p>
            {sameBase.length > 1 && (
              <Chart
                id="chart-chat-engines"
                title={`Chat generation, every ${model.base.name} config`}
                subtitle="Same prompts and settings on every engine. Click a bar to open that config."
                spec={engineChart}
                height={Math.max(180, sameBase.length * 56 + 80)}
              />
            )}
            {!cfg.speed && teleCharts(cfg.chat, 'chart-chat')}
          </section>
        )}

        {cfg.speed && (
          <section>
            <h2>Speed by context depth</h2>
            <p class="muted small">llama.cpp's own benchmark, llama-bench, with no server in the loop.</p>
            <div class="grid-2">
              <Chart id="chart-tg-depth" title="Generation vs context depth" subtitle={model.configs.length > 1 ? 'All configs of this model; the selected one is solid.' : undefined} spec={depthChart('tg_tps')} />
              <Chart id="chart-pp-depth" title="Prompt processing vs context depth" spec={depthChart('pp_tps')} />
            </div>
            <div class="table-wrap">
              <table class="data">
                <thead><tr><th>Test</th><th class="num">Depth</th><th class="num">t/s</th><th class="num">± sd</th><th class="num">Reps</th><th class="num">GPU W</th><th class="num">Tok/J</th></tr></thead>
                <tbody>
                  {speedRows.map((m) => {
                    const dims = { depth: m.depth, n_prompt: m.n_prompt, n_gen: m.n_gen };
                    return (
                      <tr>
                        <td>{testLabel(m)} <span class="muted small">{m.method}</span></td>
                        <td class="num">{depthLabel(m.depth)}</td>
                        <td class="num">{num(m.value)}</td>
                        <td class="num">{num(m.stddev)}</td>
                        <td class="num">{m.n ?? DASH}</td>
                        <td class="num">{num(pick(cfg.speed, 'gpu_w_avg', dims)?.value, 0)}</td>
                        <td class="num">{num(pick(cfg.speed, 'tokens_per_joule', dims)?.value)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {teleCharts(cfg.speed, 'chart')}
          </section>
        )}

        <section>
          <h2>Quality</h2>
          {cfg.quality ? (
            <MetricTable metrics={cfg.quality.metrics} />
          ) : (
            <p class="muted">Not measured yet: perplexity and KL-divergence against a reference quant.</p>
          )}
        </section>

        <section>
          <h2>Capability</h2>
          <Capability cfg={cfg} scored={scored.find((s) => s.cfg.id === cfg.id)} />
        </section>

        <section>
          <h2>Run history</h2>
          <div class="table-wrap">
            <table class="data">
              <thead><tr><th>Date</th><th>Kind</th><th>Engine build</th><th>Driver</th><th class="num">Duration</th><th>Flags</th></tr></thead>
              <tbody>
                {cfg.history.map((r) => (
                  <tr>
                    <td><a href={`/runs/${r.id}`}>{date(r.started_at)}</a></td>
                    <td>{runKind(r)}</td>
                    <td><code>{r.engine}@{r.commit_sha ?? '?'}</code></td>
                    <td>{r.driver}</td>
                    <td class="num">{r.duration_s != null ? `${num(r.duration_s / 60, 1)} min` : DASH}</td>
                    <td>{r.throttled ? 'throttled' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </Layout>,
    );
  });

  // -- run audit page ------------------------------------------------------------
  app.get('/runs/:id', (c) => {
    const run = getRun(db, c.req.param('id'));
    if (!run) return c.notFound();
    return c.html(
      <Layout title={`Run ${run.id.slice(0, 8)}`} tab="benchmarks" path={`/runs/${run.id}`} description={`Raw published record of a ${run.kind} run of ${run.model_name} (${run.config_name}).`}>
        <p class="crumb"><a href={`/m/${run.model_slug}?c=${run.config_slug}`}>← {run.model_name} · {run.config_name}</a></p>
        <h1>Run {run.id.slice(0, 8)}</h1>
        <div class="table-wrap">
          <table class="data params">
            <tbody>
              <tr><td>Kind</td><td>{runKind(run)}</td></tr>
              <tr><td>Started</td><td>{date(run.started_at)} UTC</td></tr>
              <tr><td>Duration</td><td>{run.duration_s != null ? `${num(run.duration_s, 0)} s` : DASH}</td></tr>
              <tr><td>Engine</td><td><code>{run.engine}@{run.commit_sha}</code> (build {run.engine_version ?? '?'})</td></tr>
              <tr><td>Driver</td><td>{run.driver}</td></tr>
              <tr><td>Throttled</td><td>{run.throttled ? 'yes' : 'no'}</td></tr>
              <tr><td>Command</td><td><code>{run.cli_args}</code></td></tr>
              <tr><td>Lab version</td><td><code>{run.lab_version}</code></td></tr>
            </tbody>
          </table>
        </div>
        <h2>Metrics</h2>
        <MetricTable metrics={run.metrics} />
        {run.evals.length > 0 && (
          <>
            <h2>Eval results</h2>
            <EvalTable evals={run.evals} />
          </>
        )}
        <h2>Raw</h2>
        <pre class="raw">{JSON.stringify(run.raw, null, 2)}</pre>
      </Layout>,
    );
  });

  return app;
}

const MetricTable = (props: { metrics: { key: string; method: string; n_prompt: number; n_gen: number; depth: number; value: number; stddev: number | null; n: number | null; unit: string }[] }) => (
  <div class="table-wrap">
    <table class="data">
      <thead><tr><th>Metric</th><th>Method</th><th class="num">Prompt</th><th class="num">Gen</th><th class="num">Depth</th><th class="num">Value</th><th class="num">± sd</th><th>Unit</th></tr></thead>
      <tbody>
        {props.metrics.map((m) => (
          <tr>
            <td><code>{m.key}</code></td>
            <td>{methodLabel(m.method)}</td>
            <td class="num">{m.n_prompt || DASH}</td>
            <td class="num">{m.n_gen || DASH}</td>
            <td class="num">{depthLabel(m.depth)}</td>
            <td class="num">{num(m.value)}</td>
            <td class="num">{num(m.stddev)}</td>
            <td>{m.unit}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

/** The scoreboard's numbers as a table: the equivalent for readers without JavaScript. */
const CapabilityTable = (props: { groups: BoardGroup[]; depth: number | null }) => {
  const cells = (render: (b: (typeof BENCH_ORDER)[number], first: boolean) => unknown) =>
    CATEGORIES.flatMap((cat) => BENCHMARKS.filter((b) => b.category === cat.key).map((b, i) => render(b, i === 0)));
  return (
    <div class="table-wrap">
      <table class="data">
        <thead>
          <tr>
            <th rowspan={2} class="num">#</th>
            <th rowspan={2}>Model · config</th>
            <th rowspan={2} class="num">Score</th>
            {CATEGORIES.map((cat) => (
              <th colspan={BENCHMARKS.filter((b) => b.category === cat.key).length} class="grp grp-start">
                <span class="key" style={`--k: var(--cat-${cat.key})`}></span>{cat.name} · {Math.round(cat.weight * 100)}%
              </th>
            ))}
            <th colspan={3} class="grp grp-start">Speed</th>
          </tr>
          <tr>
            {cells((b, first) => <th class={first ? 'num grp-start' : 'num'}>{b.name}</th>)}
            <th class="num grp-start">{props.depth != null ? `Gen @${depthLabel(props.depth)}` : 'Generation'}</th>
            <th class="num">Thinking allowance</th>
            <th class="num">Fixed / night</th>
          </tr>
        </thead>
        <tbody>
          {props.groups.map((g) => (
            <>
              <tr class="group-row"><td colspan={12}>{g.title}</td></tr>
              {g.rows.map((r) => (
                <tr>
                  <td class="num rank">{r.rankLabel}</td>
                  <td>
                    <div class="model-name">
                      <a href={modelUrl(r.s.m, r.s.cfg)}>{r.name}</a>
                      {!r.score && <span class="tag">quick tier only</span>}
                      {r.s.throttled && <span class="tag">throttled</span>}
                    </div>
                    <div class="muted small">{r.config}</div>
                  </td>
                  <td class="num">
                    {r.score
                      ? <>{num(r.score.value, 1)}<span class="muted small"> ± {num(r.score.stderr, 1)}</span></>
                      : <span class="muted">quick {num(r.quick.value, 1)}</span>}
                  </td>
                  {cells((b, first) => {
                    const v = r.scores[b.key];
                    return (
                      <td class={`num${first ? ' grp-start' : ''}${v ? '' : ' muted'}`} title={v ? undefined : 'Not run yet'}>
                        {v ? num(v.value, 1) : DASH}
                      </td>
                    );
                  })}
                  <td class="num grp-start">
                    {r.speed != null ? `${num(r.speed)}${props.depth == null && r.s.speed ? ` @${depthLabel(r.s.speed.depth)}` : ''}` : DASH}
                  </td>
                  <td class="num">{r.allowanceText}</td>
                  <td class={`num${r.issuesPerNight == null ? ' muted' : ''}`}>{r.issuesPerNight ?? DASH}</td>
                </tr>
              ))}
            </>
          ))}
        </tbody>
      </table>
    </div>
  );
};

/** One config's capability: the score it earns, and every benchmark behind it. */
const Capability = (props: { cfg: ConfigView; scored?: ScoredConfig }) => {
  const s = props.scored;
  if (!s) {
    return (
      <p class="muted">
        Not measured yet. The quick tier (LiveCodeBench, BFCL, GPQA Diamond, AIME 2025) runs for every config;
        SWE-bench Verified and Terminal-Bench run for the configs worth a night each.
      </p>
    );
  }
  const total = s.score ?? s.quick;
  const runFor = (tier: string) => (tier === 'quick' ? props.cfg.evalsQuick : props.cfg.evalsDeep);
  return (
    <>
      <section class="tiles">
        <Tile
          label={s.score ? 'Coding-work score' : 'Quick-tier score'}
          value={total ? num(total.value, 1) : DASH}
          unit={total ? `± ${num(total.stderr, 1)}` : undefined}
          note={s.score ? 'all six benchmarks' : 'not ranked: the deep tier has not run'}
        />
        {(total?.categories ?? []).map((cat) => (
          <Tile label={cat.name} value={num(cat.mean, 1)} unit="% solved" note={`${Math.round(cat.weight * 100)}% of the score`} />
        ))}
        <Tile
          label="Thinking allowance"
          value={allowanceText(s.allowance, s.allowanceCapped)}
          note={`${depthLabel(HEADLINE_PROMPT_TOKENS)}-token prompt, 5-minute task`}
        />
      </section>
      <div class="table-wrap">
        <table class="data">
          <thead>
            <tr>
              <th>Benchmark</th>
              <th>Tier</th>
              <th class="num">Solved</th>
              <th class="num">Tasks</th>
              <th class="num">Allowance</th>
              <th class="num">Length stops</th>
              <th class="num">s / task</th>
              <th>Harness</th>
              <th>Run</th>
            </tr>
          </thead>
          <tbody>
            {BENCH_ORDER.map((b) => {
              const run = runFor(b.tier);
              const e = run?.evals.find((x) => x.task === b.key);
              const v = s.scores[b.key];
              const metric = (key: string) => (e ? pick(run ?? null, key, { method: b.key })?.value : undefined);
              return (
                <tr>
                  <td>
                    {b.name}
                    <div class="muted small">{catName(b.category)} · {b.metric} · {b.size}</div>
                  </td>
                  <td>{b.tier}</td>
                  <td class="num">{v ? <>{num(v.value, 1)}<span class="muted small"> ± {num(v.stderr, 1)}</span></> : DASH}</td>
                  <td class="num">{e?.n_tasks != null ? `${e.n_tasks}${e.attempts_per_task && e.attempts_per_task > 1 ? ` × ${e.attempts_per_task}` : ''}` : DASH}</td>
                  <td class="num">{num(metric('eval_allowance_tokens'), 0)}</td>
                  <td class="num">{num(metric('eval_length_stops'), 0)}</td>
                  <td class="num">{num(metric('eval_seconds_per_task'), 0)}</td>
                  <td>{e?.harness ?? DASH}{e?.harness_version ? <span class="muted small"> {e.harness_version}</span> : ''}</td>
                  <td>{e && run ? <a href={`/runs/${run.id}`}>{date(run.started_at)}</a> : DASH}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {(s.cfg.evalsQuick?.throttled || s.cfg.evalsDeep?.throttled) && (
        <p class="warn">⚠ An eval run hit throttling. Quick-tier allowances were fixed in advance, so those results still count; a throttled deep-tier run is shown but never ranked.</p>
      )}
    </>
  );
};

const EvalTable = (props: { evals: EvalRow[] }) => (
  <div class="table-wrap">
    <table class="data">
      <thead><tr><th>Task</th><th>Metric</th><th class="num">Value</th><th class="num">± se</th><th class="num">Tasks</th><th class="num">Attempts</th><th>Harness</th><th>Subset</th></tr></thead>
      <tbody>
        {props.evals.map((e) => (
          <tr>
            <td>{byKey.get(e.task)?.name ?? e.task}</td>
            <td>{e.metric}{e.filter && e.filter !== 'none' ? ` (${e.filter})` : ''}</td>
            <td class="num">{num(e.value * 100, 1)}%</td>
            <td class="num">{e.stderr != null ? `${num(e.stderr * 100, 1)}%` : DASH}</td>
            <td class="num">{e.n_tasks ?? e.n_samples ?? DASH}</td>
            <td class="num">{e.attempts_per_task ?? DASH}</td>
            <td>{e.harness ?? DASH}{e.harness_version ? ` ${e.harness_version}` : ''}</td>
            <td><code class="small">{e.subset_id ?? DASH}</code></td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);
