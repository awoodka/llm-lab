import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import { createPagesApp } from '../src/app.ts';
import { openDb, type Db } from '../src/db.ts';
import { ingest } from '../src/routes/api.ts';
import { bundle, probe } from './fixtures.ts';

const CHAT = 'https://chat.example.test';
const MODEL = 'gemma-3-4b-it-q4_k_m-gguf';
const RUN = '4f7c1f0e-8a3b-4c56-9d2e-0a1b2c3d4e5f';

function seeded(): Db {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  return db;
}

async function get(app: ReturnType<typeof createPagesApp>, path: string) {
  const res = await app.request(path);
  return { status: res.status, html: await res.text() };
}

beforeEach(() => {
  process.env.CHAT_URL = CHAT;
  process.env.PUBLIC_ORIGIN = 'https://site.example.test';
});

afterEach(() => {
  delete process.env.CHAT_URL;
  delete process.env.PUBLIC_ORIGIN;
});

test('the homepage has the intro, headline numbers, live status and both entries', async () => {
  const { status, html } = await get(createPagesApp(seeded(), probe('up')), '/');
  assert.equal(status, 200);
  assert.match(html, /<h1>Local Inference<\/h1>/);
  assert.match(html, /Fastest generation[\s\S]*176/);
  assert.match(html, /Generation at 4k context[\s\S]*160/);
  assert.match(html, /<strong>Online<\/strong>/);
  assert.ok(html.includes(`href="${CHAT}">Open chat`), 'the chat entry goes straight to CHAT_URL');
  assert.match(html, /href="\/benchmarks">View benchmarks/);
  assert.match(html, /property="og:url" content="https:\/\/site\.example\.test\/"/);
  assert.doesNotMatch(html, /chart\.umd\.min\.js/, 'the homepage draws no charts');
});

test('benchmarks, methodology, model and run pages render', async () => {
  const app = createPagesApp(seeded(), probe('up'));

  const bench = await get(app, '/benchmarks');
  assert.equal(bench.status, 200);
  assert.match(bench.html, /aria-current="page">Benchmarks</);
  assert.match(bench.html, /Gemma 3 4B IT Q4_K_M/);
  assert.match(bench.html, /means of repeated llama-bench runs/);
  assert.match(bench.html, /chart\.umd\.min\.js/);

  const method = await get(app, '/methodology');
  assert.equal(method.status, 200);
  assert.match(method.html, /<h1>How it(&#39;|')s measured<\/h1>/);
  assert.match(method.html, /utilization was at least 50%/);

  const model = await get(app, `/m/${MODEL}`);
  assert.equal(model.status, 200);
  assert.match(model.html, /<a href="\/benchmarks">← All models<\/a>/);

  assert.equal((await get(app, `/runs/${RUN}`)).status, 200);
});

test('unknown pages get the branded 404, and the pages app has no API at all', async () => {
  const app = createPagesApp(seeded(), probe('up'));
  for (const path of ['/no-such-page', '/m/no-such-model', '/runs/no-such-run']) {
    const res = await get(app, path);
    assert.equal(res.status, 404, path);
    assert.match(res.html, /<h1>Not found<\/h1>/, path);
  }
  for (const [method, path] of [['GET', '/api/hosted'], ['POST', '/api/ingest'], ['PUT', '/api/hosted'], ['PUT', '/api/hosted/pause'], ['DELETE', `/api/runs/${RUN}`]]) {
    const res = await app.request(path, { method, headers: { authorization: 'Bearer anything', 'content-type': 'application/json' }, body: method === 'GET' ? undefined : '{}' });
    assert.equal(res.status, 404, `${method} ${path}`);
  }
});

test('public pages never show tailnet addresses or local paths', async () => {
  const app = createPagesApp(seeded(), probe('up'));
  for (const path of ['/', '/benchmarks', '/methodology', `/m/${MODEL}`, `/m/${MODEL}?c=default`, `/runs/${RUN}`, '/no-such-page']) {
    const { html } = await get(app, path);
    assert.doesNotMatch(html, /ts\.net/, path);
    assert.doesNotMatch(html, /(?<![\w.:/-])\/(home|mnt|srv|root)\//, path);
  }
});

test('throttled runs are tagged and never feed the headline numbers', async () => {
  const db = seeded();
  ingest(db, bundle({ runId: '5e8d2a1b-3c4d-4e5f-8a9b-0c1d2e3f4a5b', sha: 'sha-hot', configHash: 'hash-hot', configSlug: 'hot', throttled: true, tg0: 999 }), false);
  const app = createPagesApp(db, probe('up'));

  const home = await get(app, '/');
  assert.doesNotMatch(home.html, /999/);
  assert.match(home.html, /Fastest generation[\s\S]*176/);

  const best = await get(app, '/benchmarks');
  assert.doesNotMatch(best.html, /999/, 'the per-model row uses the non-throttled config');

  const all = await get(app, '/benchmarks?all=1');
  assert.match(all.html, /class="tag">throttled</);
});
