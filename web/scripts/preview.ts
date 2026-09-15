/**
 * Local design preview: the real pages app, seeded with the mockup's invented models.
 *
 * For design checkpoints only — it never runs in production and writes nothing to the real database.
 * Every page it serves carries a banner saying the numbers are made up.
 *
 *   npm run preview -- --host 127.0.0.1 --port 8788
 */
import { randomUUID } from 'node:crypto';
import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { IngestBody } from '../src/contract.ts';
import { createPagesApp } from '../src/app.ts';
import { openDb, type Db } from '../src/db.ts';
import { ingest } from '../src/routes/api.ts';
import { BENCHMARKS, binomialStderr, byKey } from '../src/scoring.ts';

type Preview = {
  name: string;
  config: string;
  slug: string;
  arch: 'dense' | 'moe';
  paramsB: number;
  activeB?: number;
  quant: string;
  kv: string;
  ctx: number;
  /** Generation t/s at 16k, as the mockup lists it. */
  tg16k: number;
  pp: number;
  vram: number;
  sizeGb: number;
  /** Minutes per SWE-bench issue, for the deep tier. */
  issueMin?: number;
  /** Percent solved, by benchmark key. */
  scores: Record<string, number>;
};

/** The approved mockup's CONFIGS, as models the site can ingest. Invented numbers. */
const PREVIEWS: Preview[] = [
  { name: '27B dense', config: '64k ctx · q8_0 KV', slug: 'p27b', arch: 'dense', paramsB: 27, quant: 'Q4_K_M', kv: 'q8_0', ctx: 65536, tg16k: 28.2, pp: 949, vram: 16.9, sizeGb: 16.5, issueMin: 14,
    scores: { livecodebench: 62, swebench_verified: 47, bfcl: 78, terminal_bench: 33, gpqa_diamond: 68, aime_2025: 71 } },
  { name: '30B-A3B MoE', config: '64k ctx · q8_0 KV', slug: 'p30b-a3b', arch: 'moe', paramsB: 30, activeB: 3, quant: 'Q4_K_M', kv: 'q8_0', ctx: 65536, tg16k: 112, pp: 2900, vram: 19.8, sizeGb: 18.6, issueMin: 6,
    scores: { livecodebench: 56, swebench_verified: 41, bfcl: 75, terminal_bench: 29, gpqa_diamond: 61, aime_2025: 69 } },
  { name: '24B dense, coding-tuned', config: '64k ctx · q8_0 KV', slug: 'p24b-code', arch: 'dense', paramsB: 24, quant: 'Q5_K_M', kv: 'q8_0', ctx: 65536, tg16k: 36, pp: 1300, vram: 20.1, sizeGb: 17.1, issueMin: 12,
    scores: { livecodebench: 59, swebench_verified: 53, bfcl: 71, terminal_bench: 37, gpqa_diamond: 44, aime_2025: 28 } },
  { name: '20B-A4B MoE', config: '64k ctx · f16 KV', slug: 'p20b-a4b', arch: 'moe', paramsB: 20, activeB: 4, quant: 'MXFP4', kv: 'f16', ctx: 65536, tg16k: 138, pp: 3900, vram: 14.2, sizeGb: 11.7, issueMin: 5,
    scores: { livecodebench: 51, swebench_verified: 31, bfcl: 63, terminal_bench: 22, gpqa_diamond: 57, aime_2025: 63 } },
  { name: '32B dense', config: '32k ctx · q8_0 KV', slug: 'p32b', arch: 'dense', paramsB: 32, quant: 'IQ4_XS', kv: 'q8_0', ctx: 32768, tg16k: 24.5, pp: 780, vram: 21.6, sizeGb: 17.9,
    scores: { livecodebench: 60, bfcl: 74, gpqa_diamond: 66, aime_2025: 55 } },
  { name: '14B dense', config: '32k ctx · f16 KV', slug: 'p14b', arch: 'dense', paramsB: 14, quant: 'Q6_K', kv: 'f16', ctx: 32768, tg16k: 52, pp: 2100, vram: 15.3, sizeGb: 12.1,
    scores: { livecodebench: 43, bfcl: 64, gpqa_diamond: 49, aime_2025: 41 } },
  { name: '4B dense', config: '32k ctx · f16 KV', slug: 'p4b', arch: 'dense', paramsB: 4, quant: 'Q4_K_M', kv: 'f16', ctx: 32768, tg16k: 163, pp: 8196, vram: 3.6, sizeGb: 2.5,
    scores: { livecodebench: 24, bfcl: 46, gpqa_diamond: 31, aime_2025: 17 } },
];

const HARDWARE = {
  gpu_name: 'NVIDIA GeForce RTX 3090', gpu_vram_mb: 24576, driver: '595.99.02', cuda: '13.0', power_limit_w: 250,
  pcie: 'Gen3 x16', cpu_model: 'AMD Ryzen 5 3600', cpu_threads_visible: 6, ram_mb: 24576, kernel: '7.0.2-6-pve', os: 'Debian 13',
};
const ENGINE = { engine: 'llama.cpp', version: '10883', commit_sha: '91f6a6cf3', extra: {} };

/** The same shape every bundle shares: which model, config and machine a run belongs to. */
function subject(p: Preview) {
  return {
    base_model: { slug: `${p.slug}-base`, name: p.name, arch: p.arch, params_b: p.paramsB, active_params_b: p.activeB ?? null },
    model: {
      slug: `${p.slug}-${p.quant.toLowerCase()}-gguf`, name: `${p.name} ${p.quant}`, base: `${p.slug}-base`,
      engine: 'llama.cpp', format: 'gguf', quant: p.quant, file_size_bytes: Math.round(p.sizeGb * 1e9),
    },
    config: {
      slug: `${p.ctx / 1024}k-${p.kv}`, name: p.config, config_hash: `preview-${p.slug}`,
      params: { ctx: p.ctx, kv_type: p.kv, n_gpu_layers: 99, flash_attn: true },
      launch_command: `llama-server -m ${p.slug}.gguf -c ${p.ctx} -ngl 99 --cache-type-k ${p.kv}`,
    },
    hardware: HARDWARE,
    engine_build: ENGINE,
  };
}

/** A speed run with the depth ladder the allowance needs. */
function speedBundle(p: Preview) {
  const tg = (depth: number, tps: number) => ({ key: 'tg_tps', method: 'llama-bench', n_gen: 128, depth, value: tps, stddev: tps * 0.01, n: 3, unit: 't/s' });
  return IngestBody.parse({
    schema_version: 2,
    bundle_sha: `preview-${p.slug}-speed`,
    ...subject(p),
    run: {
      id: randomUUID(), kind: 'speed', status: 'ok', started_at: '2026-09-14T22:00:00Z', duration_s: 900,
      lab_version: '0.1.0', cli_args: `lab bench ${p.slug}`, throttled: false,
    },
    metrics: [
      { key: 'pp_tps', method: 'llama-bench', n_prompt: 512, value: p.pp, stddev: p.pp * 0.02, n: 3, unit: 't/s' },
      { key: 'pp_tps', method: 'llama-bench', n_prompt: 512, depth: 16384, value: p.pp * 0.82, stddev: p.pp * 0.02, n: 3, unit: 't/s' },
      tg(0, p.tg16k * 1.12), tg(4096, p.tg16k * 1.07), tg(16384, p.tg16k),
      { key: 'vram_peak_mb', method: 'nvml', value: Math.round(p.vram * 1024), unit: 'MiB' },
      { key: 'gpu_w_avg', method: 'nvml', value: 232, unit: 'W' },
      { key: 'tokens_per_joule', method: 'derived', value: p.tg16k / 232, unit: 'tok/J' },
    ],
  });
}

/** An evals run for one tier: the benchmark results plus the operating numbers behind them. */
function evalsBundle(p: Preview, tier: 'quick' | 'deep') {
  const benchmarks = BENCHMARKS.filter((b) => b.tier === tier && p.scores[b.key] != null);
  if (benchmarks.length === 0) return null;
  const allowance = Math.min(Math.floor((tier === 'quick' ? 300 : 1200) - 8000 / p.pp) * p.tg16k, p.ctx - 8000);
  return IngestBody.parse({
    schema_version: 2,
    bundle_sha: `preview-${p.slug}-${tier}`,
    ...subject(p),
    run: {
      id: randomUUID(), kind: 'evals', tier, status: 'ok', started_at: tier === 'quick' ? '2026-09-15T02:00:00Z' : '2026-09-16T02:00:00Z',
      duration_s: tier === 'quick' ? 54000 : 68000, lab_version: '0.1.0',
      cli_args: `lab eval ${p.slug} --tier ${tier}`, throttled: false,
      raw: { allowance: { pp0: p.pp, tg: [{ depth: 0, tps: p.tg16k * 1.12 }, { depth: 16384, tps: p.tg16k }] }, note: 'preview data' },
    },
    metrics: benchmarks.flatMap((b) => [
      { key: 'eval_seconds_per_task', method: b.key, value: b.tier === 'deep' ? (p.issueMin ?? 12) * 60 : 180, unit: 's' },
      { key: 'eval_tokens_per_task', method: b.key, value: Math.round(allowance * 0.6), unit: 'tokens' },
      ...(b.tier === 'quick' ? [{ key: 'eval_allowance_tokens', method: b.key, value: Math.round(allowance), unit: 'tokens' }] : []),
      { key: 'eval_length_stops', method: b.key, value: b.key === 'aime_2025' ? 3 : 1, unit: 'count' },
      { key: 'eval_excluded', method: b.key, value: 0, unit: 'count' },
    ]),
    eval_results: benchmarks.map((b) => ({
      task: b.key,
      metric: byKey.get(b.key)!.metric,
      value: p.scores[b.key] / 100,
      stderr: binomialStderr(p.scores[b.key], b.n) / 100,
      n_samples: b.n,
      harness: { gpqa_diamond: 'lm-eval', aime_2025: 'lm-eval', swebench_verified: 'mini-swe-agent', terminal_bench: 'terminal-bench' }[b.key] ?? b.key,
      harness_version: '1.2.0',
      subset_id: `${b.key}-2026-09:9f2c41`,
      n_tasks: b.n,
      attempts_per_task: b.key === 'aime_2025' ? 4 : 1,
    })),
  });
}

function seed(): Db {
  const db = openDb(':memory:');
  for (const p of PREVIEWS) {
    ingest(db, speedBundle(p), false);
    for (const tier of ['quick', 'deep'] as const) {
      const b = evalsBundle(p, tier);
      if (b) ingest(db, b, false);
    }
  }
  return db;
}

const BANNER =
  '<p class="warn">Design preview: every model, score and speed on this page is invented, copied from the approved mockup. Nothing here was measured.</p>';

const arg = (name: string, fallback: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i === -1 ? fallback : process.argv[i + 1];
};

const pages = createPagesApp(seed(), async () => 'up');
const app = new Hono();
app.all('*', async (c) => {
  const res = await pages.fetch(c.req.raw);
  if (!res.headers.get('content-type')?.includes('text/html')) return res;
  const html = (await res.text()).replace('<main>', `<main>${BANNER}`);
  return c.html(html, res.status as 200);
});

const hostname = arg('host', '127.0.0.1');
const port = Number(arg('port', '8788'));
process.env.CHAT_URL ??= 'https://chat.example.invalid';
serve({ fetch: app.fetch, hostname, port }, () => console.log(`preview listening on http://${hostname}:${port}`));
