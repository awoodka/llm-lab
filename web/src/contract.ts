// Mirror of lab/src/lab/schema.py (RunBundle). Bump SCHEMA_VERSION on both sides together.
import { z } from 'zod';

export const SCHEMA_VERSION = 2;
/** v1 lab versions publish speed runs; evals runs must be v2. */
export const ACCEPTED_SCHEMA_VERSIONS = [1, 2] as const;

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
  engine: z.enum(['llama.cpp', 'exllamav3', 'vllm']),
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
  /** Fraction (0-1); the site shows and scores it as a percentage. */
  value: num,
  stderr: optNum,
  n_samples: z.number().int().nullish(),
  limit_n: z.number().int().nullish(),
  /** v1 field, kept for old rows; v2 runs send harness/harness_version instead. */
  lm_eval_version: optStr,
  gen_kwargs: z.record(z.string(), z.unknown()).default({}),
  // schema v2: which harness produced this, over which pinned subset
  harness: optStr,
  harness_version: optStr,
  subset_id: optStr,
  n_tasks: z.number().int().nullish(),
  attempts_per_task: z.number().int().nullish(),
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
  /** v1 lab versions still publish speed runs; evals runs are v2 (see the refinement below). */
  schema_version: z.union([z.literal(1), z.literal(2)]),
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
}).superRefine((b, ctx) => {
  if (b.run.kind !== 'evals') return;
  if (b.schema_version < 2) {
    ctx.addIssue({ code: 'custom', path: ['schema_version'], message: 'evals runs require schema_version 2' });
  }
  if (b.run.tier !== 'quick' && b.run.tier !== 'deep') {
    ctx.addIssue({ code: 'custom', path: ['run', 'tier'], message: "evals runs need tier 'quick' or 'deep'" });
  }
});

export type IngestBody = z.infer<typeof IngestBody>;

export const HostedBody = z.object({
  config_hash: z.string().nullable(),
  /** Ignored: the chat link is site configuration (CHAT_URL). Still accepted from older lab versions. */
  chat_url: z.string().nullish(),
});

/** Mirror of lab/src/lab/publish.py put_pause(): benchmarks, evals and `lab serve` pause the hosted model;
 * `starting` reports a model that is loading (vLLM takes minutes to boot). */
export const PauseBody = z.object({
  paused: z.boolean(),
  reason: z.enum(['bench', 'serve', 'evals', 'starting']).nullable(),
  ref: z.string().max(200).regex(/^[\w.-]+\/[\w.-]+$/).nullable(),
});
