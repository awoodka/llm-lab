import { readFileSync } from 'node:fs';
import type { Hono } from 'hono';

const FILES: Record<string, [string, string]> = {
  '/static/style.css': ['../public/style.css', 'text/css; charset=utf-8'],
  '/static/charts.js': ['../public/charts.js', 'text/javascript; charset=utf-8'],
  '/static/scoreboard.js': ['../public/scoreboard.js', 'text/javascript; charset=utf-8'],
  '/static/favicon.svg': ['../public/favicon.svg', 'image/svg+xml'],
  '/static/chart.umd.min.js': ['../node_modules/chart.js/dist/chart.umd.min.js', 'text/javascript; charset=utf-8'],
};

export function staticRoutes(app: Hono) {
  const cache = new Map<string, Buffer>();
  const dev = process.env.NODE_ENV !== 'production';
  for (const [route, [rel, type]] of Object.entries(FILES)) {
    const url = new URL(rel, import.meta.url);
    app.get(route, (c) => {
      let body = dev ? undefined : cache.get(route);
      if (!body) {
        body = readFileSync(url);
        cache.set(route, body);
      }
      return c.body(new Uint8Array(body), 200, { 'content-type': type, 'cache-control': 'no-cache' });
    });
  }
}
