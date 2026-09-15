PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS base_models (
  id INTEGER PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  family TEXT,
  params_b REAL,
  active_params_b REAL,
  arch TEXT NOT NULL CHECK (arch IN ('dense', 'moe')),
  hf_repo TEXT
);

CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  base_model_id INTEGER NOT NULL REFERENCES base_models(id),
  engine TEXT NOT NULL,
  format TEXT NOT NULL,
  quant TEXT NOT NULL,
  bpw REAL,
  file_size_bytes INTEGER,
  source_repo TEXT,
  source_file TEXT,
  source_revision TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS configs (
  id INTEGER PRIMARY KEY,
  model_id INTEGER NOT NULL REFERENCES models(id),
  slug TEXT NOT NULL,
  name TEXT NOT NULL,
  config_hash TEXT NOT NULL UNIQUE,
  params_json TEXT NOT NULL,
  launch_command TEXT NOT NULL,
  engine_files_json TEXT NOT NULL DEFAULT '{}',
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
  UNIQUE (model_id, slug)
);

CREATE TABLE IF NOT EXISTS hardware_snapshots (
  id INTEGER PRIMARY KEY,
  hash TEXT NOT NULL UNIQUE,
  gpu_name TEXT, gpu_vram_mb INTEGER, driver TEXT, cuda TEXT, power_limit_w REAL, pcie TEXT,
  cpu_model TEXT, cpu_threads_visible INTEGER, ram_mb INTEGER, kernel TEXT, os TEXT
);

CREATE TABLE IF NOT EXISTS engine_builds (
  id INTEGER PRIMARY KEY,
  hash TEXT NOT NULL UNIQUE,
  engine TEXT NOT NULL,
  version TEXT,
  commit_sha TEXT,
  build_flags TEXT,
  extra_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS quality_refs (
  id INTEGER PRIMARY KEY,
  base_model_id INTEGER NOT NULL REFERENCES base_models(id),
  ref_label TEXT NOT NULL,
  dataset TEXT NOT NULL,
  ctx INTEGER NOT NULL,
  chunks INTEGER NOT NULL,
  logits_sha TEXT,
  UNIQUE (base_model_id, ref_label, dataset, ctx, chunks)
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  config_id INTEGER NOT NULL REFERENCES configs(id),
  hardware_id INTEGER NOT NULL REFERENCES hardware_snapshots(id),
  engine_build_id INTEGER NOT NULL REFERENCES engine_builds(id),
  quality_ref_id INTEGER REFERENCES quality_refs(id),
  kind TEXT NOT NULL CHECK (kind IN ('speed', 'quality', 'evals')),
  tier TEXT,
  started_at TEXT NOT NULL,
  duration_s REAL,
  lab_version TEXT,
  cli_args TEXT,
  throttled INTEGER NOT NULL DEFAULT 0,
  notes TEXT,
  telemetry_json TEXT NOT NULL DEFAULT '[]',
  raw_json TEXT NOT NULL DEFAULT '{}',
  bundle_sha TEXT NOT NULL,
  published_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS runs_config ON runs(config_id, kind, started_at);

CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  key TEXT NOT NULL,
  method TEXT NOT NULL,
  n_prompt INTEGER NOT NULL DEFAULT 0,
  n_gen INTEGER NOT NULL DEFAULT 0,
  depth INTEGER NOT NULL DEFAULT 0,
  concurrency INTEGER NOT NULL DEFAULT 1,
  value REAL NOT NULL,
  stddev REAL,
  n INTEGER,
  unit TEXT NOT NULL,
  samples_json TEXT,
  PRIMARY KEY (run_id, key, method, n_prompt, n_gen, depth, concurrency)
);

CREATE TABLE IF NOT EXISTS eval_results (
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  task TEXT NOT NULL,
  metric TEXT NOT NULL,
  filter TEXT NOT NULL DEFAULT 'none',
  value REAL NOT NULL,
  stderr REAL,
  n_samples INTEGER,
  limit_n INTEGER,
  lm_eval_version TEXT,
  gen_kwargs_json TEXT NOT NULL DEFAULT '{}',
  -- schema v2: which harness produced this, and over which pinned subset
  harness TEXT,
  harness_version TEXT,
  subset_id TEXT,
  n_tasks INTEGER,
  attempts_per_task INTEGER,
  PRIMARY KEY (run_id, task, metric, filter)
);

CREATE TABLE IF NOT EXISTS hosted (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  config_hash TEXT,
  since TEXT,
  chat_url TEXT -- unused: the chat link is site configuration (CHAT_URL)
);

-- Why the hosted model is down: `lab` reports when a benchmark or `lab serve` pauses it, and when that ends.
CREATE TABLE IF NOT EXISTS hosted_pause (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  paused INTEGER NOT NULL,
  reason TEXT,
  ref TEXT,
  since TEXT NOT NULL
);

-- Latest non-throttled run per config and kind; pages build headline numbers from this.
CREATE VIEW IF NOT EXISTS latest_runs AS
SELECT r.*
FROM runs r
WHERE r.throttled = 0
  AND r.started_at = (
    SELECT MAX(r2.started_at) FROM runs r2
    WHERE r2.config_id = r.config_id AND r2.kind = r.kind AND r2.throttled = 0
  );
