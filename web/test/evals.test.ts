import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bundle, evalsBundle } from './fixtures.ts';
import { IngestBody } from '../src/contract.ts';
import { openDb } from '../src/db.ts';
import { ingest } from '../src/routes/api.ts';
import { getModel } from '../src/queries.ts';

const CONFIG = (db: ReturnType<typeof openDb>) => getModel(db, 'gemma-3-4b-it-q4_k_m-gguf')!.configs[0];

test('a v1 speed bundle is still accepted next to v2 evals', () => {
  const db = openDb(':memory:');
  assert.equal(ingest(db, bundle(), false).status, 'created');
  assert.equal(ingest(db, evalsBundle(), false).status, 'created');
  const cfg = CONFIG(db);
  assert.ok(cfg.speed, 'speed run');
  assert.ok(cfg.evalsQuick, 'quick evals run');
  assert.equal(cfg.evalsDeep, null);
});

test('eval results keep their harness and subset details', () => {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  ingest(db, evalsBundle(), false);
  const evals = CONFIG(db).evalsQuick!.evals;
  assert.equal(evals.length, 4);
  const lcb = evals.find((e) => e.task === 'livecodebench')!;
  assert.equal(lcb.harness, 'livecodebench');
  assert.equal(lcb.harness_version, '1.0.0');
  assert.equal(lcb.n_tasks, 100);
  assert.equal(lcb.subset_id, 'livecodebench-v1:abc123');
  assert.equal(evals.find((e) => e.task === 'aime_2025')!.attempts_per_task, 4);
});

test('quick and deep runs are kept per tier, not one latest evals run', () => {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  ingest(db, evalsBundle({ tier: 'quick', startedAt: '2026-09-16T02:00:00Z' }), false);
  ingest(db, evalsBundle({ tier: 'deep', startedAt: '2026-09-17T02:00:00Z' }), false);
  const cfg = CONFIG(db);
  assert.equal(cfg.evalsQuick?.tier, 'quick');
  assert.equal(cfg.evalsDeep?.tier, 'deep');
  assert.equal(cfg.evalsQuick?.evals.length, 4);
  assert.equal(cfg.evalsDeep?.evals.length, 2);
});

test('the newest run of a tier wins', () => {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  ingest(db, evalsBundle({ startedAt: '2026-09-16T02:00:00Z' }), false);
  ingest(db, evalsBundle({
    runId: '99999999-1111-4222-8333-444444444444', sha: 'evals-quick-2', startedAt: '2026-09-18T02:00:00Z',
    scores: { livecodebench: 0.31, bfcl: 0.5, gpqa_diamond: 0.33, aime_2025: 0.2 },
  }), false);
  const lcb = CONFIG(db).evalsQuick!.evals.find((e) => e.task === 'livecodebench')!;
  assert.equal(lcb.value, 0.31);
});

test('a throttled quick run still counts: its allowances were fixed in advance', () => {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  ingest(db, evalsBundle({ throttled: true }), false);
  const cfg = CONFIG(db);
  assert.ok(cfg.evalsQuick, 'throttled quick run is still returned');
  assert.equal(cfg.evalsQuick?.throttled, true);
});

test('an evals run needs schema v2 and a known tier', () => {
  const noTier = { ...evalsBundle(), run: { ...evalsBundle().run, tier: null } };
  assert.equal(IngestBody.safeParse(noTier).success, false);
  const wrongTier = { ...evalsBundle(), run: { ...evalsBundle().run, tier: 'overnight' } };
  assert.equal(IngestBody.safeParse(wrongTier).success, false);
  const v1Evals = { ...evalsBundle(), schema_version: 1 };
  assert.equal(IngestBody.safeParse(v1Evals).success, false);
  assert.equal(IngestBody.safeParse(evalsBundle()).success, true);
});

test('vllm is a known engine, and a chat-benchmark run never hides the llama-bench run', () => {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  ingest(db, bundle({
    runId: '22222222-3333-4444-8555-666666666666', sha: 'chat-1', method: 'http-sampled',
    startedAt: '2026-09-17T02:00:00Z', engine: 'vllm',
  }), false);
  const cfg = CONFIG(db);
  assert.equal(cfg.speed?.metrics[0].method, 'llama-bench');
  assert.equal(cfg.chat?.metrics[0].method, 'http-sampled');
  assert.notEqual(cfg.speed?.id, cfg.chat?.id);
});
