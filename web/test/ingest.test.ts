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
