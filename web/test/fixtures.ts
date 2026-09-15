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
};

/** A minimal valid ingest bundle for one gemma speed run; tests override the parts they vary. */
export function bundle(overrides: Overrides = {}) {
  return IngestBody.parse({
    schema_version: 1,
    bundle_sha: overrides.sha ?? 'sha-1',
    base_model: { slug: 'gemma-3-4b-it', name: 'Gemma 3 4B IT', arch: 'dense', params_b: 3.88 },
    model: { slug: 'gemma-3-4b-it-q4_k_m-gguf', name: 'Gemma 3 4B IT Q4_K_M', base: 'gemma-3-4b-it', engine: 'llama.cpp', format: 'gguf', quant: 'Q4_K_M', file_size_bytes: 2483218432 },
    config: { slug: overrides.configSlug ?? 'default', name: overrides.configSlug ?? 'Default', config_hash: overrides.configHash ?? 'hash-a', params: { ctx: overrides.ctx ?? 8192 }, launch_command: 'llama-server -m x.gguf' },
    hardware: { gpu_name: 'RTX 3090', gpu_vram_mb: 24576, driver: '595.99.02', cuda: '13.2', power_limit_w: 280, pcie: 'Gen3 x16', cpu_model: 'Ryzen 5 3600', cpu_threads_visible: 6, ram_mb: 16384, kernel: '7.0', os: 'Debian 13' },
    engine_build: { engine: 'llama.cpp', version: '10883', commit_sha: '91f6a6cf3', extra: { dirty: false } },
    run: { id: overrides.runId ?? '4f7c1f0e-8a3b-4c56-9d2e-0a1b2c3d4e5f', kind: 'speed', status: 'ok', started_at: '2026-09-15T01:00:00Z', lab_version: '0.1.0', cli_args: 'lab bench', throttled: overrides.throttled ?? false },
    metrics: [
      { key: 'pp_tps', method: 'llama-bench', n_prompt: 512, value: 7840, stddev: 700, unit: 't/s' },
      { key: 'tg_tps', method: 'llama-bench', n_gen: 128, value: overrides.tg0 ?? 175.9, stddev: 0.5, unit: 't/s' },
      { key: 'tg_tps', method: 'llama-bench', n_gen: 128, depth: 4096, value: 160.2, stddev: 0.4, unit: 't/s' },
    ],
  });
}

/** A model-health probe that always reports the same state. */
export const probe = (value: Probe): ProbeFn => async () => value;
