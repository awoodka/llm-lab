import type { Child } from 'hono/jsx';
import { raw } from 'hono/html';
import { siteConfig } from '../site.ts';

/** Compact chart description; public/charts.js turns it into a Chart.js config using CSS color tokens. */
export type ChartSpec = {
  type: 'bar-h' | 'line' | 'scatter';
  unit: string;
  xLabel?: string;
  yLabel?: string;
  categories?: string[];
  links?: (string | null)[];
  series: {
    label: string;
    slot: number | 'muted';
    data: (number | null)[] | { x: number; y: number; label?: string }[];
    stddev?: (number | null)[];
    dim?: boolean;
  }[];
};

/**
 * Capability scoreboard description; public/scoreboard.js draws the SVG from it.
 * The server-rendered capability table carries the same numbers for readers without JavaScript.
 */
export type ScoreboardSpec = {
  categories: { key: string; name: string; weight: number }[];
  benchmarks: { key: string; name: string; category: string; tier: string; size: string; shape: 'circle' | 'diamond' }[];
  groups: { title: string; rows: ScoreboardRow[] }[];
  /** What the speed bars mean, e.g. "Generation at 16k · t/s". */
  speedLabel: string;
};

export type ScoreboardRow = {
  name: string;
  config: string;
  rankLabel: string;
  scores: Record<string, { value: number; stderr: number }>;
  score: { value: number; stderr: number; categories: { key: string; name: string; weight: number; mean: number }[] } | null;
  quick: { value: number; stderr: number; categories: { key: string; name: string; weight: number; mean: number }[] };
  tieNote?: string;
  speed?: number | null;
  promptSpeed?: number | null;
  allowance?: number | null;
  allowanceText?: string;
  issuesPerNight?: number | null;
  vramGb?: number | null;
};

export type Tab = 'home' | 'benchmarks';

const DEFAULT_DESCRIPTION =
  'Open-weight language models ranked on a single RTX 3090 for coding, tool use and reasoning, with the speed behind the ranking: generation and prompt speed as context grows, VRAM and tokens per joule.';

/** Page shell: header nav, Open Graph tags, and the chart scripts only on pages that draw charts. */
export const Layout = (props: {
  title: string;
  tab: Tab;
  /** Path of this page, for og:url. */
  path: string;
  children: Child;
  description?: string;
  footer?: Child;
  charts?: boolean;
  scoreboard?: boolean;
}) => {
  const { publicOrigin, chatUrl } = siteConfig();
  const title = props.title.includes('Local Inference') ? props.title : `${props.title} · Local Inference`;
  const description = props.description ?? DEFAULT_DESCRIPTION;
  return (
    <>
      {raw('<!doctype html>')}
      <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>{title}</title>
          <meta name="description" content={description} />
          <meta property="og:site_name" content="Local Inference" />
          <meta property="og:type" content="website" />
          <meta property="og:title" content={title} />
          <meta property="og:description" content={description} />
          {publicOrigin && <meta property="og:url" content={`${publicOrigin}${props.path}`} />}
          <link rel="icon" href="/static/favicon.svg" type="image/svg+xml" />
          <link rel="stylesheet" href="/static/style.css" />
        </head>
        <body>
          <header class="site">
            <a href="/" class="brand">Local Inference</a>
            <nav class="site-tabs" aria-label="Sections">
              <a href="/benchmarks" aria-current={props.tab === 'benchmarks' ? 'page' : undefined}>Benchmarks</a>
              {chatUrl && <a href={chatUrl}>Chat ↗</a>}
            </nav>
          </header>
          <main>{props.children}</main>
          {props.footer && <footer class="site">{props.footer}</footer>}
          {props.charts && (
            <>
              <script src="/static/chart.umd.min.js" defer></script>
              <script src="/static/charts.js" defer></script>
            </>
          )}
          {props.scoreboard && <script src="/static/scoreboard.js" defer></script>}
        </body>
      </html>
    </>
  );
};

export const Chart = (props: { id: string; title: string; subtitle?: string; spec: ChartSpec; height?: number }) => (
  <figure class="chart">
    <figcaption>
      <h3>{props.title}</h3>
      {props.subtitle && <p class="muted">{props.subtitle}</p>}
    </figcaption>
    <div class="chart-box" style={`height:${props.height ?? 280}px`}>
      <canvas data-chart={props.id} role="img" aria-label={props.title}></canvas>
    </div>
    <script
      type="application/json"
      id={props.id}
      dangerouslySetInnerHTML={{ __html: JSON.stringify(props.spec).replace(/</g, '\\u003c') }}
    />
  </figure>
);

const MARK = {
  circle: <circle r="5" />,
  diamond: <path d="M0 -6.5L6.5 0L0 6.5L-6.5 0Z" />,
};

/** The scoreboard figure: legend, the SVG the script draws, and the spec it reads. */
export const Scoreboard = (props: { id: string; title: string; subtitle?: string; spec: ScoreboardSpec }) => (
  <figure class="chart">
    <figcaption>
      <h3>{props.title}</h3>
      {props.subtitle && <p class="muted">{props.subtitle}</p>}
    </figcaption>
    <div class="legend">
      {props.spec.categories.map((cat) => (
        <span class="lg">
          <span class="lg-cat">{cat.name}</span>
          {props.spec.benchmarks
            .filter((b) => b.category === cat.key)
            .map((b) => (
              <>
                <svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true" class={`c-${cat.key}`}>
                  {MARK[b.shape]}
                </svg>
                <span>{b.name}</span>
              </>
            ))}
        </span>
      ))}
      <span class="lg">
        <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
          <rect x="2" y="0" width="12" height="16" rx="2" class="ci" />
          <rect x="6.5" y="0" width="3" height="16" rx="1.5" class="tick" />
        </svg>
        <span>Score ± 1 standard error</span>
      </span>
      <span class="lg">
        <svg width="20" height="12" viewBox="0 0 20 12" aria-hidden="true">
          <path d="M0 0H16Q20 0 20 4V8Q20 12 16 12H0Z" class="bar" />
        </svg>
        <span>{props.spec.speedLabel}</span>
      </span>
    </div>
    <div class="board" data-scoreboard={props.id}></div>
    <script
      type="application/json"
      id={props.id}
      dangerouslySetInnerHTML={{ __html: JSON.stringify(props.spec).replace(/</g, '\\u003c') }}
    />
  </figure>
);

export const Tile = (props: { label: string; value: string; unit?: string; note?: Child }) => (
  <div class="tile">
    <div class="tile-label">{props.label}</div>
    <div class="tile-value">
      {props.value}
      {props.unit && props.value !== '—' && <span class="tile-unit"> {props.unit}</span>}
    </div>
    {props.note && <div class="tile-note">{props.note}</div>}
  </div>
);

export const HardwareFooter = (props: { hardware: Record<string, any>[]; builds: Record<string, any>[] }) => (
  <>
    {props.hardware.map((h) => (
      <p>
        {h.gpu_name} ({Math.round(h.gpu_vram_mb / 1024)} GB, {h.power_limit_w} W limit, driver {h.driver}, CUDA {h.cuda}) · {h.cpu_model},{' '}
        {h.cpu_threads_visible} threads · {Math.round(h.ram_mb / 1024)} GB RAM
      </p>
    ))}
    {props.builds.length > 0 && (
      <p>Engine builds: {props.builds.map((b) => `${b.engine}@${b.commit_sha ?? '?'}`).join(', ')}</p>
    )}
  </>
);
