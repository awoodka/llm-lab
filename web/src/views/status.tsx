import type { HostingView, PauseReason } from '../status.ts';

const DOT = { online: 'ok', starting: 'busy', paused: 'busy', offline: 'off' } as const;
const LABEL = { online: 'Online', starting: 'Starting', offline: 'Offline' } as const;
const PAUSE_TEXT: Record<PauseReason, string> = { bench: 'benchmarking', evals: 'running capability evals on', serve: 'testing' };

/** One line: a colored dot, the chat model's state, and which model. */
export const StatusLine = (props: { view: HostingView }) => {
  const v = props.view;
  const h = v.hosted;
  return (
    <span class="status">
      <span class={`dot ${DOT[v.state]}`} aria-hidden="true"></span>
      {v.state === 'paused' ? (
        <span>
          <strong>Paused:</strong> {PAUSE_TEXT[v.reason]} {v.model.slug ? <a href={`/m/${encodeURIComponent(v.model.slug)}`}>{v.model.name}</a> : v.model.name} right now
        </span>
      ) : (
        <span>
          <strong>{LABEL[v.state]}</strong>
          {h?.model_slug && (
            <> · <a href={`/m/${encodeURIComponent(h.model_slug)}?c=${encodeURIComponent(h.config_slug ?? '')}`}>{h.model_name} · {h.config_name}</a></>
          )}
        </span>
      )}
    </span>
  );
};
