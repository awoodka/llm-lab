import assert from 'node:assert/strict';
import { before, test } from 'node:test';
import { createApiApp, createPagesApp } from '../src/app.ts';
import { openDb } from '../src/db.ts';
import { getPause } from '../src/queries.ts';
import { assertNoLeaks, bundle, probe, TOKEN } from './fixtures.ts';

const AUTH = { authorization: `Bearer ${TOKEN}`, 'content-type': 'application/json' };
const JSON_ONLY = { 'content-type': 'application/json' };

before(() => {
  process.env.INGEST_TOKEN = TOKEN;
});

function send(app: ReturnType<typeof createApiApp>, method: string, path: string, body: unknown, headers: Record<string, string> = AUTH) {
  return app.request(path, { method, headers, body: JSON.stringify(body) });
}

test('the API app ingests with the token, refuses without it, and serves no pages', async () => {
  const api = createApiApp(openDb(':memory:'));
  assert.equal((await send(api, 'POST', '/api/ingest', bundle(), JSON_ONLY)).status, 401);
  const ok = await send(api, 'POST', '/api/ingest', bundle());
  assert.equal(ok.status, 200);
  assert.equal((await ok.json()).status, 'created');
  const root = await api.request('/');
  assert.equal(root.status, 404);
  assert.doesNotMatch(await root.text(), /<html/);
});

test('pause reports are validated and drive the public status line', async () => {
  const db = openDb(':memory:');
  const api = createApiApp(db);
  await send(api, 'POST', '/api/ingest', bundle());
  const ref = 'gemma-3-4b-it-q4_k_m-gguf/default';

  assert.equal((await send(api, 'PUT', '/api/hosted/pause', { paused: true, reason: 'bench', ref }, JSON_ONLY)).status, 401);
  assert.equal((await send(api, 'PUT', '/api/hosted/pause', { paused: true, reason: 'party', ref })).status, 400);
  assert.equal((await send(api, 'PUT', '/api/hosted/pause', { paused: true, reason: 'bench', ref: '../../etc/passwd' })).status, 400);

  assert.equal((await send(api, 'PUT', '/api/hosted/pause', { paused: true, reason: 'bench', ref })).status, 200);
  assert.equal(getPause(db)?.model_name, 'Gemma 3 4B IT Q4_K_M');

  const offline = await (await createPagesApp(db, probe('offline')).request('/')).text();
  assert.match(offline, /Paused:<\/strong> benchmarking <a href="\/m\/gemma-3-4b-it-q4_k_m-gguf">Gemma 3 4B IT Q4_K_M<\/a> right now/);
  const up = await (await createPagesApp(db, probe('up')).request('/')).text();
  assert.doesNotMatch(up, /Paused/, 'a model that answers its health check is online, whatever lab reported');

  assert.equal((await send(api, 'PUT', '/api/hosted/pause', { paused: false, reason: null, ref: null })).status, 200);
  const resumed = await (await createPagesApp(db, probe('offline')).request('/')).text();
  assert.match(resumed, /<strong>Offline<\/strong>/);
});

test('the hosted model shows on the site, and an old lab chat_url is ignored', async () => {
  const db = openDb(':memory:');
  const api = createApiApp(db);
  await send(api, 'POST', '/api/ingest', bundle());
  const put = await send(api, 'PUT', '/api/hosted', { config_hash: 'hash-a', chat_url: 'https://web.example-tailnet.ts.net:8443' });
  assert.equal(put.status, 200);
  const html = await (await createPagesApp(db, probe('up')).request('/')).text();
  assert.match(html, /<strong>Online<\/strong> · <a href="\/m\/gemma-3-4b-it-q4_k_m-gguf\?c=default">Gemma 3 4B IT Q4_K_M · Default<\/a>/);
  assertNoLeaks(html, '/');
});
