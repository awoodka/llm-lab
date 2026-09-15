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

export type Tab = 'home' | 'benchmarks';

const DEFAULT_DESCRIPTION =
  'Open-weight language models benchmarked on a single RTX 3090: generation and prompt speed as context grows, VRAM, GPU power and tokens per joule.';

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
