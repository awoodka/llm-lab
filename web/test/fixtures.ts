import { IngestBody } from '../src/contract.ts';
import type { Probe, ProbeFn } from '../src/status.ts';

export const TOKEN = 'test-token-0123456789abcdef0123456789abcdef';

type Overrides = {
  runId?: string;
  sha?: string;
  configHash?: string;
  configSlug?: string;
  ctx?: number;
  throttled?: boolean;
  tg0?: number;
  /** Metric method: 'llama-bench' by default, 'http-sampled' for a chat-benchmark run. */
  method?: string;
  startedAt?: string;
  engine?: 'llama.cpp' | 'exllamav3' | 'vllm';
};

/** A minimal valid ingest bundle for one gemma speed run; tests override the parts they vary. */
export function bundle(overrides: Overrides = {}) {
  return IngestBody.parse({
    schema_version: 1,
    bundle_sha: overrides.sha ?? 'sha-1',
    base_model: GEMMA_BASE,
    model: { slug: 'gemma-3-4b-it-q4_k_m-gguf', name: 'Gemma 3 4B IT Q4_K_M', base: 'gemma-3-4b-it', engine: overrides.engine ?? 'llama.cpp', format: 'gguf', quant: 'Q4_K_M', file_size_bytes: 2483218432 },
    config: { slug: overrides.configSlug ?? 'default', name: overrides.configSlug ?? 'Default', config_hash: overrides.configHash ?? 'hash-a', params: { ctx: overrides.ctx ?? 8192 }, launch_command: 'llama-server -m x.gguf' },
    hardware: { gpu_name: 'RTX 3090', gpu_vram_mb: 24576, driver: '595.99.02', cuda: '13.2', power_limit_w: 280, pcie: 'Gen3 x16', cpu_model: 'Ryzen 5 3600', cpu_threads_visible: 6, ram_mb: 16384, kernel: '7.0', os: 'Debian 13' },
    engine_build: { engine: overrides.engine ?? 'llama.cpp', version: '10883', commit_sha: '91f6a6cf3', extra: { dirty: false } },
    run: { id: overrides.runId ?? '4f7c1f0e-8a3b-4c56-9d2e-0a1b2c3d4e5f', kind: 'speed', status: 'ok', started_at: overrides.startedAt ?? '2026-09-15T01:00:00Z', lab_version: '0.1.0', cli_args: 'lab bench', throttled: overrides.throttled ?? false },
    metrics: [
      { key: 'pp_tps', method: overrides.method ?? 'llama-bench', n_prompt: 512, value: 7840, stddev: 700, unit: 't/s' },
      { key: 'tg_tps', method: overrides.method ?? 'llama-bench', n_gen: 128, value: overrides.tg0 ?? 175.9, stddev: 0.5, unit: 't/s' },
      { key: 'tg_tps', method: overrides.method ?? 'llama-bench', n_gen: 128, depth: 4096, value: 160.2, stddev: 0.4, unit: 't/s' },
    ],
  });
}

type EvalOverrides = {
  runId?: string;
  sha?: string;
  tier?: 'quick' | 'deep';
  startedAt?: string;
  throttled?: boolean;
  configHash?: string;
  configSlug?: string;
  /** Task key to fraction solved, e.g. { livecodebench: 0.24 }. */
  scores?: Record<string, number>;
  /** Score the vLLM config instead of the gemma one: no llama-bench ladder, and BFCL in native tool mode. */
  vllm?: boolean;
};

const QUICK_SCORES = { livecodebench: 0.24, bfcl: 0.46, gpqa_diamond: 0.31, aime_2025: 0.17 };
const DEEP_SCORES = { swebench_verified: 0.1, terminal_bench: 0.07 };
const N_TASKS: Record<string, number> = { livecodebench: 100, bfcl: 400, gpqa_diamond: 198, aime_2025: 30, swebench_verified: 30, terminal_bench: 30 };
const HARNESS: Record<string, string> = {
  livecodebench: 'livecodebench', bfcl: 'bfcl', gpqa_diamond: 'lm-eval', aime_2025: 'lm-eval',
  swebench_verified: 'mini-swe-agent', terminal_bench: 'terminal-bench',
};

/** A v2 capability-evals bundle for the same gemma config, quick tier by default. */
export function evalsBundle(overrides: EvalOverrides = {}) {
  const tier = overrides.tier ?? 'quick';
  const scores = overrides.scores ?? (tier === 'quick' ? QUICK_SCORES : DEEP_SCORES);
  const base = overrides.vllm
    ? vllmBundle({ configHash: overrides.configHash, configSlug: overrides.configSlug })
    : bundle({ configHash: overrides.configHash, configSlug: overrides.configSlug });
  return IngestBody.parse({
    ...base,
    schema_version: 2,
    bundle_sha: overrides.sha ?? `evals-${tier}-1`,
    run: {
      id: overrides.runId ?? (tier === 'quick' ? '11111111-2222-4333-8444-555555555555' : '66666666-7777-4888-8999-aaaaaaaaaaaa'),
      kind: 'evals',
      tier,
      status: 'ok',
      started_at: overrides.startedAt ?? '2026-09-16T02:00:00Z',
      lab_version: '0.1.0',
      cli_args: `lab eval gemma-3-4b-it-q4_k_m-gguf/default --tier ${tier}`,
      throttled: overrides.throttled ?? false,
      raw: { allowance: { speed_run_id: base.run.id, pp0: 7840, tg: [{ depth: 0, tps: 175.9 }] } },
    },
    metrics: Object.keys(scores).map((task) => ({
      key: 'eval_seconds_per_task', method: task, value: 42, unit: 's',
    })),
    eval_results: Object.entries(scores).map(([task, value]) => ({
      task,
      metric: task === 'livecodebench' ? 'pass@1' : 'accuracy',
      value,
      stderr: Math.sqrt((value * (1 - value)) / N_TASKS[task]),
      n_samples: N_TASKS[task],
      harness: HARNESS[task],
      harness_version: '1.0.0',
      subset_id: `${task}-v1:abc123`,
      n_tasks: N_TASKS[task],
      attempts_per_task: task === 'aime_2025' ? 4 : 1,
      // Which mode BFCL ran in is part of what the number means, not decoration.
      ...(task === 'bfcl' ? { gen_kwargs: { mode: overrides.vllm ? 'FC' : 'prompting' } } : {}),
    })),
  });
}

type ChatOverrides = {
  runId?: string;
  sha?: string;
  startedAt?: string;
  throttled?: boolean;
  configHash?: string;
  configSlug?: string;
  /** Default-sampling and greedy decode t/s. */
  decode?: number;
  greedy?: number;
};

const GEMMA_BASE = { slug: 'gemma-3-4b-it', name: 'Gemma 3 4B IT', arch: 'dense', params_b: 3.88 };
const QWEN_BASE = { slug: 'qwen3.8-27b', name: 'Qwen3.8 27B', arch: 'dense', params_b: 27 };
export const VLLM_MODEL = 'qwen3.8-27b-w4a16-autoround-fast';
export const QWEN_GGUF_MODEL = 'qwen3.8-27b-q4_k_m-gguf';

/** The chat benchmark's metrics, as lab/src/lab/bench/http_chat.py publishes them. */
function chatMetrics(decode: number, greedy: number, vllm: boolean) {
  const cohort = (method: string, tps: number) =>
    [
      { key: 'decode_tps', value: tps, stddev: +(tps * 0.03).toFixed(2), n: 8, unit: 't/s', samples: Array(8).fill(tps) },
      { key: 'tpot_ms', value: +(1000 / tps).toFixed(3), unit: 'ms' },
      { key: 'ttft_ms', value: vllm ? 169 : 212, unit: 'ms' },
      { key: 'prefill_tps', value: 1450, unit: 't/s' },
      { key: 'e2e_tps', value: +(tps * 0.97).toFixed(2), unit: 't/s' },
      { key: 'out_tokens_mean', value: 981, unit: 'tokens' },
      { key: 'gpu_w_avg', value: 245, unit: 'W' },
      { key: 'tokens_per_joule', value: +(tps / 245).toFixed(3), unit: 'tok/J' },
      ...(vllm ? [{ key: 'spec_accept_len', value: 3.21, unit: 'tok/step' }] : []),
    ].map((m) => ({ ...m, method, n_prompt: 0, n_gen: 1024, depth: 0, concurrency: 1 }));
  const run = [
    { key: 'load_s', value: vllm ? 72 : 9, unit: 's' },
    { key: 'vram_peak_mb', value: vllm ? 23450 : 18740, unit: 'MiB' },
    { key: 'ram_peak_mb', value: 5120, unit: 'MiB' },
    { key: 'oom_kills', value: 0, unit: 'count' },
  ].map((m) => ({ ...m, method: 'http' }));
  return [...cohort('http-sampled', decode), ...cohort('http-greedy', greedy), ...run];
}

function chatRun(o: ChatOverrides, id: string, ref: string) {
  return {
    id: o.runId ?? id, kind: 'speed', status: 'ok', started_at: o.startedAt ?? '2026-09-17T03:00:00Z', lab_version: '0.1.0',
    cli_args: `lab bench ${ref} --http`, throttled: o.throttled ?? false, raw: { protocol: 'chat-c1-v1' },
  };
}

/** A chat-benchmark run on the gemma llama.cpp config that bundle() speed-tests. */
export function chatBundle(o: ChatOverrides = {}) {
  const base = bundle({ configHash: o.configHash, configSlug: o.configSlug });
  return IngestBody.parse({
    ...base,
    schema_version: 2,
    bundle_sha: o.sha ?? 'chat-gemma-1',
    run: chatRun(o, '7a7a7a7a-1111-4222-8333-444444444444', `gemma-3-4b-it-q4_k_m-gguf/${base.config.slug}`),
    metrics: chatMetrics(o.decode ?? 150.5, o.greedy ?? 152.5, false),
  });
}

/** Qwen3.8 27B on vLLM: a chat-benchmark run only, since llama-bench is llama.cpp's. */
export function vllmBundle(o: ChatOverrides = {}) {
  const slug = o.configSlug ?? '64k-dflash2';
  return IngestBody.parse({
    ...bundle(),
    schema_version: 2,
    bundle_sha: o.sha ?? 'chat-vllm-1',
    base_model: QWEN_BASE,
    model: {
      slug: VLLM_MODEL, name: 'Qwen3.8 27B W4A16 AutoRound (fast)', base: QWEN_BASE.slug, engine: 'vllm', format: 'safetensors',
      quant: 'W4A16', file_size_bytes: 19470000000, source_repo: 'dbirks/Qwen3.8-27B-W4A16-AutoRound',
    },
    config: {
      slug, name: slug, config_hash: o.configHash ?? 'hash-vllm', params: { launcher: 'single-user/start_qwen.sh', ctx: 65536, spec: 'dflash2' },
      launch_command: 'SPEC=dflash2 CTX=fast PREFIX_CACHE=1 single-user/start_qwen.sh',
    },
    engine_build: { engine: 'vllm', version: '0.28.0', commit_sha: 'bae2023ff', extra: { dirty: false } },
    run: {
      ...chatRun(o, '8b8b8b8b-1111-4222-8333-444444444444', `${VLLM_MODEL}/${slug}`),
      telemetry: [0, 1, 2].map((t) => ({ t, power_w: 240, vram_mb: 23450, temp_c: 60, util: 99 })),
    },
    metrics: chatMetrics(o.decode ?? 124.4, o.greedy ?? 136.8, true),
  });
}

/** Qwen3.8 27B Q4_K_M on llama.cpp with a chat-benchmark run: the same base model as vllmBundle(). */
export function qwenGgufChatBundle(o: ChatOverrides = {}) {
  const slug = o.configSlug ?? '64k-q8kv';
  return IngestBody.parse({
    ...bundle(),
    schema_version: 2,
    bundle_sha: o.sha ?? 'chat-qwen-gguf-1',
    base_model: QWEN_BASE,
    model: {
      slug: QWEN_GGUF_MODEL, name: 'Qwen3.8 27B Q4_K_M', base: QWEN_BASE.slug, engine: 'llama.cpp', format: 'gguf', quant: 'Q4_K_M',
      file_size_bytes: 17400000000,
    },
    config: { slug, name: slug, config_hash: o.configHash ?? 'hash-qwen-gguf', params: { ctx: 65536 }, launch_command: 'llama-server -m q.gguf' },
    run: chatRun(o, '9c9c9c9c-1111-4222-8333-444444444444', `${QWEN_GGUF_MODEL}/${slug}`),
    metrics: chatMetrics(o.decode ?? 32.7, o.greedy ?? 33.1, false),
  });
}

/** A model-health probe that always reports the same state. */
export const probe = (value: Probe): ProbeFn => async () => value;
