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
    base_model: { slug: 'gemma-3-4b-it', name: 'Gemma 3 4B IT', arch: 'dense', params_b: 3.88 },
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
  const base = bundle({ configHash: overrides.configHash, configSlug: overrides.configSlug });
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
    })),
  });
}

/** A model-health probe that always reports the same state. */
export const probe = (value: Probe): ProbeFn => async () => value;
