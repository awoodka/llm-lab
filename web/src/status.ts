/**
 * Live status of the hosted chat model, for the homepage and the benchmarks banner.
 *
 * Liveness comes from probing the model's /health on ai (cached, so public traffic can't hammer it). `lab` only
 * reports why the model is down: benchmarks and `lab serve` pause it. A probe that finds the model up or loading
 * always wins, so a lost "resume" report can never show a false pause.
 */
export type Probe = 'up' | 'starting' | 'offline';
export type ProbeFn = () => Promise<Probe>;

export const PROBE_CACHE_MS = 15_000;
export const PROBE_TIMEOUT_MS = 2_000;
/** An older pause report no longer explains an offline model (e.g. lab died mid-benchmark). */
export const PAUSE_FRESH_MS = 6 * 60 * 60 * 1000;

export function modelProbe(url: string | undefined, fetchImpl: typeof fetch = fetch, now: () => number = Date.now): ProbeFn {
  if (!url) return async () => 'offline';
  let cached: { at: number; value: Probe } | undefined;
  let inflight: Promise<Probe> | undefined;
  return () => {
    if (cached && now() - cached.at < PROBE_CACHE_MS) return Promise.resolve(cached.value);
    inflight ??= fetchImpl(url, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) })
      .then(
        async (res): Promise<Probe> => {
          await res.body?.cancel().catch(() => {});
          // llama-server answers 503 while it loads the model; tailscale serve answers 502 when it isn't running.
          return res.ok ? 'up' : res.status === 503 ? 'starting' : 'offline';
        },
        (): Probe => 'offline',
      )
      .then((value) => {
        cached = { at: now(), value };
        inflight = undefined;
        return value;
      });
    return inflight;
  };
}

export type HostedRow = {
  since: string | null;
  config_slug: string | null;
  config_name: string | null;
  model_slug: string | null;
  model_name: string | null;
} | null;

export type PauseRow = {
  paused: number;
  reason: string | null;
  ref: string | null;
  since: string;
  model_slug: string | null;
  model_name: string | null;
} | null;

export type HostingView =
  | { state: 'online' | 'starting' | 'offline'; hosted: HostedRow }
  | { state: 'paused'; hosted: HostedRow; reason: 'bench' | 'serve'; model: { slug: string | null; name: string } };

export function hostingView(probe: Probe, hosted: HostedRow, pause: PauseRow, nowMs = Date.now()): HostingView {
  if (probe === 'up') return { state: 'online', hosted };
  if (probe === 'starting') return { state: 'starting', hosted };
  if (pause?.paused && (pause.reason === 'bench' || pause.reason === 'serve') && nowMs - Date.parse(pause.since) < PAUSE_FRESH_MS) {
    const refModel = pause.ref?.split('/')[0];
    return { state: 'paused', hosted, reason: pause.reason, model: { slug: pause.model_slug, name: pause.model_name ?? refModel ?? 'a model' } };
  }
  return { state: 'offline', hosted };
}
