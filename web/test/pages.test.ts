import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import { createPagesApp } from '../src/app.ts';
import { openDb, type Db } from '../src/db.ts';
import { ingest } from '../src/routes/api.ts';
import { bundle, evalsBundle, probe } from './fixtures.ts';

const CHAT = 'https://chat.example.test';
const MODEL = 'gemma-3-4b-it-q4_k_m-gguf';
const RUN = '4f7c1f0e-8a3b-4c56-9d2e-0a1b2c3d4e5f';

function seeded(): Db {
  const db = openDb(':memory:');
  ingest(db, bundle(), false);
  return db;
}

const QUICK_RUN = '11111111-2222-4333-8444-555555555555';

/**
 * One config with all six benchmarks, one with the quick tier only, one with speed alone:
 * every state the scoreboard has to render.
 */
function scoredDb(): Db {
  const db = seeded();
  ingest(db, evalsBundle({ tier: 'quick' }), false);
  ingest(db, evalsBundle({ tier: 'deep' }), false);
  ingest(db, bundle({ runId: '2b2b2b2b-1111-4222-8333-444444444444', sha: 'sha-wide', configHash: 'hash-wide', configSlug: 'wide' }), false);
  ingest(db, evalsBundle({ configHash: 'hash-wide', configSlug: 'wide', runId: '3c3c3c3c-1111-4222-8333-444444444444', sha: 'evals-wide-quick', tier: 'quick' }), false);
  ingest(db, bundle({ runId: '4d4d4d4d-1111-4222-8333-444444444444', sha: 'sha-tight', configHash: 'hash-tight', configSlug: 'tight' }), false);
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


test('the coding-work tab ranks configs, and the table carries the same numbers', async () => {
  const app = createPagesApp(scoredDb(), probe('up'));
  const { status, html } = await get(app, '/benchmarks');
  assert.equal(status, 200);

  assert.match(html, /aria-current="page">Coding work</, 'capability is the default view once evals exist');
  assert.match(html, /scoreboard\.js/);
  assert.match(html, /data-scoreboard="scoreboard"/);
  assert.doesNotMatch(html, /chart\.umd\.min\.js/, 'the coding view draws no Chart.js charts');

  // The fixture's six benchmarks give 17 / 26.5 / 24 per category, so the weighted geometric mean is 20.8.
  assert.match(html, /Ranked . all six benchmarks/);
  assert.match(html, /20\.8/);
  assert.match(html, /Not ranked yet . quick tier only/);
  assert.match(html, /quick 29\.2/, 'the unranked config is listed by its quick-tier score');
  for (const name of ['LiveCodeBench', 'SWE-bench Verified', 'BFCL', 'Terminal-Bench', 'GPQA Diamond', 'AIME 2025']) {
    assert.match(html, new RegExp(name.replace('-', '.')), name);
  }
  assert.match(html, /1 config has speed numbers but no capability evals yet/);
  assert.match(html, /How the score works/);
});

test('the speed tab keeps the old table and charts', async () => {
  const app = createPagesApp(scoredDb(), probe('up'));
  const { html } = await get(app, '/benchmarks?view=speed');
  assert.match(html, /aria-current="page">Speed</);
  assert.match(html, /means of repeated llama-bench runs/);
  assert.match(html, /chart\.umd\.min\.js/);
  assert.doesNotMatch(html, /data-scoreboard/);
});

test('without evals the benchmarks page falls back to speed, and says so on the coding tab', async () => {
  const app = createPagesApp(seeded(), probe('up'));
  assert.match((await get(app, '/benchmarks')).html, /aria-current="page">Speed</);
  const coding = await get(app, '/benchmarks?view=coding');
  assert.match(coding.html, /No capability evals published yet/);
  assert.doesNotMatch(coding.html, /data-scoreboard/);
});

test('the homepage leads with the score once a config is ranked, and with speed until then', async () => {
  const speedOnly = await get(createPagesApp(seeded(), probe('up')), '/');
  assert.match(speedOnly.html, /Fastest generation/);
  assert.doesNotMatch(speedOnly.html, /Best for coding work/);

  const quickOnly = seeded();
  ingest(quickOnly, evalsBundle({ tier: 'quick' }), false);
  assert.match((await get(createPagesApp(quickOnly, probe('up')), '/')).html, /Fastest generation/, 'the quick tier alone does not rank a config');

  const ranked = await get(createPagesApp(scoredDb(), probe('up')), '/');
  assert.match(ranked.html, /Best for coding work[\s\S]*20\.8/);
  assert.match(ranked.html, /Best at coding/);
  assert.match(ranked.html, /Best at agents/);
  assert.match(ranked.html, /Most issues fixed per night/);
  assert.doesNotMatch(ranked.html, /Fastest generation/);
});

test('the model page opens on the ranked config and shows every benchmark behind its score', async () => {
  const app = createPagesApp(scoredDb(), probe('up'));
  const { html } = await get(app, `/m/${MODEL}`);
  assert.match(html, /<h2>Capability<\/h2>/);
  assert.match(html, /Coding-work score[\s\S]*20\.8/, 'it defaults to the ranked config');
  assert.match(html, /Agents &amp; tools/);
  assert.match(html, /mini-swe-agent/, 'the harness behind each benchmark');
  assert.match(html, /192 tokens \(context limit\)/, 'the thinking allowance, capped by this config context');
  assert.doesNotMatch(html, /Task evals/);

  const other = await get(app, `/m/${MODEL}?c=wide`);
  assert.match(other.html, /Quick-tier score/);
  assert.match(other.html, /not ranked: the deep tier has not run/);
});

test('an evals run page lists its results', async () => {
  const { html } = await get(createPagesApp(scoredDb(), probe('up')), `/runs/${QUICK_RUN}`);
  assert.match(html, /<h2>Eval results<\/h2>/);
  assert.match(html, /LiveCodeBench/);
  assert.match(html, /livecodebench-v1:abc123/, 'the pinned subset');
  assert.match(html, /24\.0%/);
});

test('the capability pages never show tailnet addresses or local paths either', async () => {
  const app = createPagesApp(scoredDb(), probe('up'));
  for (const path of ['/', '/benchmarks', '/benchmarks?view=speed', `/m/${MODEL}`, `/runs/${QUICK_RUN}`]) {
    const { html } = await get(app, path);
    assert.doesNotMatch(html, /ts\.net/, path);
    assert.doesNotMatch(html, /(?<![\w.:/-])\/(home|mnt|srv|root)\//, path);
  }
});

test('every script the pages load is actually served', async () => {
  const app = createPagesApp(scoredDb(), probe('up'));
  const pages = [await get(app, '/benchmarks'), await get(app, '/benchmarks?view=speed'), await get(app, `/m/${MODEL}`)];
  const srcs = new Set(pages.flatMap((p) => [...p.html.matchAll(/<script src="([^"]+)"/g)].map((m) => m[1])));
  assert.ok(srcs.has('/static/scoreboard.js'), 'the coding view loads the scoreboard script');
  for (const src of srcs) {
    assert.equal((await app.request(src)).status, 200, src);
  }
});
