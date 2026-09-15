import assert from 'node:assert/strict';
import { test } from 'node:test';
import { hostingView, modelProbe, PAUSE_FRESH_MS, PROBE_CACHE_MS, type PauseRow } from '../src/status.ts';

const HOSTED = { since: '2026-09-15T01:00:00Z', config_slug: 'default', config_name: 'Default', model_slug: 'gemma-3-4b-it-q4_k_m-gguf', model_name: 'Gemma 3 4B IT Q4_K_M' };
const NOW = Date.parse('2026-09-15T12:00:00.000Z');

function pause(ageMs: number, overrides: Partial<NonNullable<PauseRow>> = {}): PauseRow {
  return {
    paused: 1, reason: 'bench', ref: 'gemma-3-4b-it-q4_k_m-gguf/default', since: new Date(NOW - ageMs).toISOString(),
    model_slug: 'gemma-3-4b-it-q4_k_m-gguf', model_name: 'Gemma 3 4B IT Q4_K_M', ...overrides,
  };
}

test('a probe that finds the model up or loading wins over any pause report', () => {
  assert.equal(hostingView('up', HOSTED, pause(1000), NOW).state, 'online');
  assert.equal(hostingView('starting', HOSTED, pause(1000), NOW).state, 'starting');
});

test('an offline model is explained by a fresh pause report, and only a fresh one', () => {
  const v = hostingView('offline', HOSTED, pause(60_000), NOW);
  assert.equal(v.state, 'paused');
  assert.ok(v.state === 'paused' && v.reason === 'bench' && v.model.name === 'Gemma 3 4B IT Q4_K_M');
  assert.equal(hostingView('offline', HOSTED, pause(PAUSE_FRESH_MS + 1), NOW).state, 'offline');
  assert.equal(hostingView('offline', HOSTED, pause(1000, { paused: 0 }), NOW).state, 'offline');
  assert.equal(hostingView('offline', HOSTED, null, NOW).state, 'offline');
});

test('a paused model that was never published is named by its slug', () => {
  const v = hostingView('offline', HOSTED, pause(1000, { reason: 'serve', ref: 'qwen3-30b-a3b-q4_k_m-gguf/32k', model_slug: null, model_name: null }), NOW);
  assert.ok(v.state === 'paused' && v.reason === 'serve' && v.model.name === 'qwen3-30b-a3b-q4_k_m-gguf' && v.model.slug === null);
});

function fakeFetch(answer: number | 'error', counter: { calls: number }) {
  return (async () => {
    counter.calls++;
    if (answer === 'error') throw new Error('connection refused');
    return new Response('body', { status: answer });
  }) as unknown as typeof fetch;
}

test('the probe maps health answers and failures', async () => {
  const c = { calls: 0 };
  assert.equal(await modelProbe('http://model/health', fakeFetch(200, c))(), 'up');
  assert.equal(await modelProbe('http://model/health', fakeFetch(503, c))(), 'starting');
  assert.equal(await modelProbe('http://model/health', fakeFetch(502, c))(), 'offline');
  assert.equal(await modelProbe('http://model/health', fakeFetch('error', c))(), 'offline');
  assert.equal(await modelProbe(undefined)(), 'offline');
});

test('the probe caches its answer so public traffic cannot hammer the model', async () => {
  const c = { calls: 0 };
  let clock = 0;
  const p = modelProbe('http://model/health', fakeFetch(200, c), () => clock);
  await p();
  await p();
  assert.equal(c.calls, 1);
  clock += PROBE_CACHE_MS;
  await p();
  assert.equal(c.calls, 2);
});

test('concurrent requests share one in-flight probe', async () => {
  let calls = 0;
  const slow = (async () => {
    calls++;
    await new Promise((resolve) => setTimeout(resolve, 20));
    return new Response(null, { status: 200 });
  }) as unknown as typeof fetch;
  const p = modelProbe('http://model/health', slow);
  assert.deepEqual(await Promise.all([p(), p(), p()]), ['up', 'up', 'up']);
  assert.equal(calls, 1);
});
