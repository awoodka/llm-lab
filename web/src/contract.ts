// Mirror of lab/src/lab/schema.py (RunBundle). Bump SCHEMA_VERSION on both sides together.
import { z } from 'zod';

export const SCHEMA_VERSION = 1;

const num = z.number();
const optNum = z.number().nullish();
const optStr = z.string().nullish();

export const BaseModelInfo = z.object({
  slug: z.string(),
  name: z.string(),
  family: optStr,
  params_b: optNum,
  active_params_b: optNum,
  arch: z.enum(['dense', 'moe']),
  hf_repo: optStr,
});

export const ModelInfo = z.object({
  slug: z.string(),
  name: z.string(),
  base: z.string(),
  engine: z.enum(['llama.cpp', 'exllamav3']),
  format: z.string(),
  quant: z.string(),
  bpw: optNum,
  file_size_bytes: z.number().int().nullish(),
  source_repo: optStr,
  source_file: optStr,
  source_revision: optStr,
  notes: optStr,
});

export const ConfigInfo = z.object({
  slug: z.string(),
  name: z.string(),
  config_hash: z.string(),
  params: z.record(z.string(), z.unknown()),
  launch_command: z.string(),
  engine_files: z.record(z.string(), z.string()).default({}),
  notes: optStr,
});

export const HardwareSnapshot = z.object({
  gpu_name: z.string(),
  gpu_vram_mb: num,
  driver: z.string(),
  cuda: z.string(),
  power_limit_w: num,
  pcie: z.string(),
  cpu_model: z.string(),
  cpu_threads_visible: num,
  ram_mb: num,
  kernel: z.string(),
  os: z.string(),
});

export const EngineBuild = z.object({
  engine: z.string(),
  version: optStr,
  commit_sha: optStr,
  build_flags: optStr,
  extra: z.record(z.string(), z.unknown()).default({}),
});

export const Metric = z.object({
  key: z.string(),
  method: z.string(),
  n_prompt: z.number().int().default(0),
  n_gen: z.number().int().default(0),
  depth: z.number().int().default(0),
  concurrency: z.number().int().default(1),
  value: num,
  stddev: optNum,
  n: z.number().int().nullish(),
  unit: z.string(),
  samples: z.array(num).nullish(),
});

export const QualityRef = z.object({
  base: z.string(),
  ref_label: z.string(),
  dataset: z.string(),
  ctx: z.number().int(),
  chunks: z.number().int(),
  logits_sha: optStr,
});

export const EvalResult = z.object({
  task: z.string(),
  metric: z.string(),
  filter: z.string().default('none'),
  value: num,
  stderr: optNum,
  n_samples: z.number().int().nullish(),
  limit_n: z.number().int().nullish(),
  lm_eval_version: optStr,
  gen_kwargs: z.record(z.string(), z.unknown()).default({}),
});

export const RunRecord = z.object({
  id: z.string().uuid(),
  kind: z.enum(['speed', 'quality', 'evals']),
  tier: optStr,
  status: z.literal('ok'),
  started_at: z.string(),
  duration_s: optNum,
  lab_version: z.string(),
  cli_args: z.string(),
  throttled: z.boolean(),
  notes: optStr,
  error: optStr,
  telemetry: z.array(z.record(z.string(), num)).default([]),
  raw: z.record(z.string(), z.unknown()).default({}),
  published_at: optStr,
});

export const IngestBody = z.object({
  schema_version: z.literal(SCHEMA_VERSION),
  bundle_sha: z.string(),
  base_model: BaseModelInfo,
  model: ModelInfo,
  config: ConfigInfo,
  hardware: HardwareSnapshot,
  engine_build: EngineBuild,
  run: RunRecord,
  metrics: z.array(Metric),
  quality_ref: QualityRef.nullish(),
  eval_results: z.array(EvalResult).default([]),
});

export type IngestBody = z.infer<typeof IngestBody>;

export const HostedBody = z.object({
  config_hash: z.string().nullable(),
  /** Ignored: the chat link is site configuration (CHAT_URL). Still accepted from older lab versions. */
  chat_url: z.string().nullish(),
});

/** Mirror of lab/src/lab/publish.py put_pause(): benchmarks and `lab serve` pause the hosted model. */
export const PauseBody = z.object({
  paused: z.boolean(),
  reason: z.enum(['bench', 'serve']).nullable(),
  ref: z.string().max(200).regex(/^[\w.-]+\/[\w.-]+$/).nullable(),
});
