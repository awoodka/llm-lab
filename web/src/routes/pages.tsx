import { Hono, type Context } from 'hono';
import type { Db } from '../db.ts';
import { DASH, date, depthLabel, gb, num, testLabel } from '../format.ts';
import {
  bestConfig, depthsFor, getHardware, getHosted, getModel, getModels, getPause, getRun, headline, pick, siteSummary,
  type ConfigView, type Headline, type ModelView,
} from '../queries.ts';
import { siteConfig } from '../site.ts';
import { hostingView, type ProbeFn } from '../status.ts';
import { Chart, HardwareFooter, Layout, Tile, type ChartSpec } from '../views/layout.tsx';
import { Methodology } from '../views/methodology.tsx';
import { StatusLine } from '../views/status.tsx';

const MAX_SLOTS = 8;

const INTRO =
  "Local Inference is my lab for running open-weight language models on a single RTX 3090. I tune each model's settings (context length, KV-cache precision, offload) and measure what the card actually delivers: generation and prompt speed as context fills up, peak VRAM, GPU power and tokens per joule. Every number comes from a scripted, hash-locked config, so results are reproducible.";

type Row = { m: ModelView; cfg: ConfigView; h: Headline };

const COLUMNS: { key: string; label: string; numeric?: boolean; value: (r: Row) => number | string | null | undefined; show: (r: Row) => string }[] = [
  { key: 'model', label: 'Model', value: (r) => r.m.name, show: (r) => r.m.name },
  { key: 'engine', label: 'Engine', value: (r) => r.m.engine, show: (r) => r.m.engine },
  { key: 'size', label: 'Size', numeric: true, value: (r) => r.m.file_size_bytes, show: (r) => gb(r.m.file_size_bytes) },
  { key: 'config', label: 'Config', value: (r) => r.cfg.name, show: (r) => r.cfg.name },
  { key: 'ctx', label: 'Ctx', numeric: true, value: (r) => r.cfg.params.ctx as number, show: (r) => depthLabel(Number(r.cfg.params.ctx ?? 0)) },
  { key: 'pp0', label: 'PP t/s', numeric: true, value: (r) => r.h.pp0?.value, show: (r) => num(r.h.pp0?.value) },
  { key: 'tg0', label: 'TG t/s', numeric: true, value: (r) => r.h.tg0?.value, show: (r) => num(r.h.tg0?.value) },
  { key: 'tgDeep', label: 'TG t/s deep', numeric: true, value: (r) => r.h.tgDeep?.value, show: (r) => (r.h.tgDeep ? `${num(r.h.tgDeep.value)} @${depthLabel(r.h.tgDeep.depth)}` : DASH) },
  { key: 'vram', label: 'VRAM', numeric: true, value: (r) => r.h.vram?.value, show: (r) => (r.h.vram ? `${num(r.h.vram.value / 1024, 1)} GB` : DASH) },
  { key: 'tokJ', label: 'Tok/J', numeric: true, value: (r) => r.h.tokJ?.value, show: (r) => num(r.h.tokJ?.value) },
];

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

    return c.html(
      <Layout title="Local Inference · open-weight LLMs on one RTX 3090" tab="home" path="/" footer={<HardwareFooter {...getHardware(db)} />}>
        <section class="intro">
          <h1>Local Inference</h1>
          <p class="lead">{INTRO}</p>
          <p><a href="/methodology">How it's measured →</a></p>
        </section>

        {summary.fastest ? (
          <section class="tiles" aria-label="Headline results">
            <Tile label="Fastest generation" value={num(summary.fastest.metric.value)} unit="t/s" note={modelLink(summary.fastest)} />
            {summary.deepest && (
              <Tile label={`Generation at ${depthLabel(summary.deepest.metric.depth)} context`} value={num(summary.deepest.metric.value)} unit="t/s" note={modelLink(summary.deepest)} />
            )}
            {summary.efficient && <Tile label="Best efficiency" value={num(summary.efficient.metric.value)} unit="tok/J" note={modelLink(summary.efficient)} />}
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
            <p>Every model and config side by side: speed as the context fills, VRAM, power and efficiency, with the exact launch command for each.</p>
            <a class="button primary" href="/benchmarks">View benchmarks →</a>
          </article>
        </section>
      </Layout>,
    );
  });

  // -- benchmarks: compare every model ----------------------------------------------------
  app.get('/benchmarks', async (c) => {
    const q = c.req.query();
    const models = getModels(db);
    const view = await hosting();
    const { chatUrl } = siteConfig();
    const engines = [...new Set(models.map((m) => m.engine))].sort();
    const bases = [...new Map(models.map((m) => [m.base.slug, m.base.name])).entries()].sort((a, b) => a[1].localeCompare(b[1]));

    let rows: Row[] = q.all
      ? models.flatMap((m) => m.configs.filter((cfg) => cfg.speed).map((cfg) => ({ m, cfg, h: headline(cfg) })))
      : models.flatMap((m) => {
          const cfg = bestConfig(m);
          return cfg ? [{ m, cfg, h: headline(cfg) }] : [];
        });
    rows = rows.filter((r) => (!q.engine || r.m.engine === q.engine) && (!q.arch || r.m.base.arch === q.arch) && (!q.base || r.m.base.slug === q.base));

    const sortCol = COLUMNS.find((col) => col.key === q.sort) ?? COLUMNS.find((col) => col.key === 'tg0')!;
    const dir = q.dir === 'asc' ? 1 : q.dir === 'desc' ? -1 : sortCol.numeric ? -1 : 1;
    rows.sort((a, b) => {
      const va = sortCol.value(a), vb = sortCol.value(b);
      if (va == null) return 1;
      if (vb == null) return -1;
      return (typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))) * dir;
    });

    const rowLabel = (r: Row) => (q.all ? `${r.m.name} · ${r.cfg.name}` : r.m.name);
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
        description="Every model and config benchmarked on one RTX 3090, side by side: generation and prompt speed, context depth, VRAM, power and tokens per joule."
        charts
        footer={<HardwareFooter {...getHardware(db)} />}
      >
        <section class="banner">
          <StatusLine view={view} />
          {chatUrl && <a class="button" href={chatUrl}>Open chat ↗</a>}
        </section>
        <h1>Model benchmarks</h1>
        <p class="meta">Local model performance, measured on one machine. <a href="/methodology">How it's measured</a></p>
        {models.length === 0 ? (
          <section class="empty">
            <p>No published results yet.</p>
          </section>
        ) : (
          <>
            <form class="filters" method="get">
              <label>Engine <select name="engine"><option value="">All</option>{engines.map((e) => <option value={e} selected={q.engine === e}>{e}</option>)}</select></label>
              <label>Base model <select name="base"><option value="">All</option>{bases.map(([slug, name]) => <option value={slug} selected={q.base === slug}>{name}</option>)}</select></label>
              <label>Architecture <select name="arch"><option value="">All</option><option value="dense" selected={q.arch === 'dense'}>Dense</option><option value="moe" selected={q.arch === 'moe'}>MoE</option></select></label>
              <label class="check"><input type="checkbox" name="all" value="1" checked={!!q.all} /> Show every config</label>
              {q.sort && <input type="hidden" name="sort" value={q.sort} />}
              {q.dir && <input type="hidden" name="dir" value={q.dir} />}
              <button type="submit">Apply</button>
            </form>
            <p class="muted small">{q.all ? 'Every benchmarked config.' : 'One row per model, using its fastest config (generation speed, empty context).'} Speeds are means of repeated llama-bench runs on this machine.</p>

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
                            {r.cfg.speed?.throttled && <span class="tag">throttled</span>}
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

            {withTg.length > 0 && (
              <div class="grid-2">
                <Chart id="chart-tg" title="Generation speed" subtitle="Tokens per second at empty context. Click a bar to open the model." spec={tgBar} height={Math.max(160, byTg.length * 36 + 70)} />
                <Chart id="chart-size" title="Speed vs weight size" subtitle="Bigger weights mean more memory traffic per token." spec={sizeScatter} />
              </div>
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
        description="How Local Inference benchmarks models on one RTX 3090: llama-bench settings, explicit engine flags, GPU telemetry, power and efficiency, throttling and limits."
      >
        <Methodology {...getHardware(db)} />
      </Layout>,
    ),
  );

  // -- model page: switch between configs --------------------------------------
  app.get('/m/:slug', (c) => {
    const model = getModel(db, c.req.param('slug'));
    if (!model) return c.notFound();
    const cfg = model.configs.find((x) => x.slug === c.req.query('c')) ?? bestConfig(model) ?? model.configs[0];
    if (!cfg) return c.notFound();
    const h = headline(cfg);
    const emphasis = model.configs.length > MAX_SLOTS;

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

    const tele = cfg.speed?.telemetry ?? [];
    const teleChart = (field: string, label: string, unit: string): ChartSpec => ({
      type: 'line', unit, xLabel: 'seconds into run', yLabel: label,
      series: [{ label, slot: 1, data: tele.map((s) => ({ x: s.t, y: field === 'vram_mb' ? s[field] / 1024 : s[field] })) }],
    });

    const speedRows = (cfg.speed?.metrics ?? []).filter((m) => m.key === 'pp_tps' || m.key === 'tg_tps');

    return c.html(
      <Layout
        title={model.name}
        tab="benchmarks"
        path={modelUrl(model, cfg)}
        description={`${model.name} (${cfg.name}) on a single RTX 3090: ${num(h.tg0?.value)} t/s generation at empty context. Launch command, settings, speed by context depth, power and efficiency.`}
        charts
        footer={<HardwareFooter {...getHardware(db)} />}
      >
        <p class="crumb"><a href="/benchmarks">← All models</a></p>
        <h1>{model.name}</h1>
        <p class="meta">
          {model.base.name} · {model.base.arch === 'moe' ? `MoE ${num(model.base.params_b, 1)}B (${num(model.base.active_params_b, 1)}B active)` : model.base.params_b ? `${num(model.base.params_b, 1)}B dense` : 'dense'} ·{' '}
          {model.engine} · {model.format.toUpperCase()} {model.quant} · {gb(model.file_size_bytes)}
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

        <section class="tiles">
          <Tile label="Generation, empty context" value={num(h.tg0?.value)} unit="t/s" note={h.tg0?.stddev != null ? `± ${num(h.tg0.stddev)}` : undefined} />
          <Tile label={h.tgDeep ? `Generation at ${depthLabel(h.tgDeep.depth)} context` : 'Generation, deep context'} value={num(h.tgDeep?.value)} unit="t/s" />
          <Tile label="Prompt processing" value={num(h.pp0?.value)} unit="t/s" />
          <Tile label="Peak VRAM" value={h.vram ? num(h.vram.value / 1024, 1) : DASH} unit="GB" note={h.vram ? 'during llama-bench' : undefined} />
          <Tile label="GPU power while generating" value={num(h.watts?.value)} unit="W" />
          <Tile label="Energy efficiency" value={num(h.tokJ?.value)} unit="tok/J" />
        </section>
        {cfg.speed?.throttled && <p class="warn">⚠ This run hit thermal or power-brake throttling; numbers may be low.</p>}

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

        {cfg.speed && (
          <section>
            <h2>Speed</h2>
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
            {tele.length > 1 && (
              <div class="grid-2">
                <Chart id="chart-power" title="GPU power during the run" spec={teleChart('power_w', 'GPU power', 'W')} height={200} />
                <Chart id="chart-vram" title="VRAM during the run" spec={teleChart('vram_mb', 'VRAM', 'GB')} height={200} />
              </div>
            )}
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
          <h2>Task evals</h2>
          {cfg.evals ? <MetricTable metrics={cfg.evals.metrics} /> : <p class="muted">Not measured yet.</p>}
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
                    <td>{r.kind}{r.tier ? ` (${r.tier})` : ''}</td>
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
              <tr><td>Kind</td><td>{run.kind}{run.tier ? ` (${run.tier})` : ''}</td></tr>
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
            <td>{m.method}</td>
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
