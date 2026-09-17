import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bundle } from './fixtures.ts';
import { openDb } from '../src/db.ts';
import { ingest } from '../src/routes/api.ts';
import { getModel, getModels } from '../src/queries.ts';

test('ingest is idempotent and guards changed content', () => {
  const db = openDb(':memory:');
  assert.equal(ingest(db, bundle(), false).status, 'created');
  assert.equal(ingest(db, bundle(), false).status, 'unchanged');
  assert.throws(() => ingest(db, bundle({ sha: 'sha-2' }), false), /different content/);
  assert.equal(ingest(db, bundle({ sha: 'sha-2' }), true).status, 'created');
  assert.equal((db.prepare('SELECT COUNT(*) AS n FROM metrics').get() as { n: number }).n, 3);
});

test('a config slug cannot be republished with different settings', () => {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  assert.throws(
    () => ingest(db, bundle({ runId: '5f7c1f0e-8a3b-4c56-9d2e-0a1b2c3d4e5f', configHash: 'hash-b', ctx: 32768 }), false),
    /different settings/,
  );
});

test('queries expose models, configs and headline metrics', () => {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  const models = getModels(db);
  assert.equal(models.length, 1);
  const model = getModel(db, 'gemma-3-4b-it-q4_k_m-gguf');
  assert.ok(model);
  assert.equal(model.configs[0].speed?.metrics.length, 3);
});

test('a llama.cpp config keeps its llama-bench run and its chat-benchmark run side by side', async () => {
  const { chatBundle } = await import('./fixtures.ts');
  const { chatHeadline, headline } = await import('../src/queries.ts');
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  assert.equal(ingest(db, chatBundle(), false).status, 'created');
  const cfg = getModel(db, 'gemma-3-4b-it-q4_k_m-gguf')!.configs[0];
  assert.equal(cfg.speed?.id, '4f7c1f0e-8a3b-4c56-9d2e-0a1b2c3d4e5f');
  assert.equal(cfg.chat?.id, '7a7a7a7a-1111-4222-8333-444444444444');
  const ch = chatHeadline(cfg);
  assert.equal(ch.decode?.value, 150.5);
  assert.equal(ch.decode?.method, 'http-sampled');
  assert.equal(ch.greedy?.value, 152.5);
  assert.equal(ch.vram?.method, 'http');
  assert.equal(headline(cfg).tg0?.value, 175.9, 'llama-bench numbers never come from the chat run');
});

test('a vLLM model with only a chat benchmark is listed and picks its chat numbers', async () => {
  const { vllmBundle, VLLM_MODEL } = await import('./fixtures.ts');
  const { bestConfig, chatHeadline, headline, siteSummary } = await import('../src/queries.ts');
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  ingest(db, vllmBundle(), false);
  const models = getModels(db);
  assert.equal(models.length, 2);
  const vllm = models.find((m) => m.slug === VLLM_MODEL)!;
  assert.equal(vllm.engine, 'vllm');
  const cfg = bestConfig(vllm)!;
  assert.equal(cfg.speed, null);
  assert.equal(chatHeadline(cfg).decode?.value, 124.4);
  assert.equal(chatHeadline(cfg).ttft?.value, 169);
  assert.equal(headline(cfg).tg0, undefined);
  const summary = siteSummary(models);
  assert.equal(summary.fastestChat?.m.slug, VLLM_MODEL);
  assert.equal(summary.fastest?.metric.method, 'llama-bench', 'the llama-bench highlight stays llama-bench');
  assert.equal(summary.efficient?.metric.method, 'http-sampled', 'efficiency compares chat runs once there are any');
  assert.equal(summary.models, 2);
  assert.equal(summary.configs, 2);
});
