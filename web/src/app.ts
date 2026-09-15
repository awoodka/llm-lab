import { Hono } from 'hono';
import type { Db } from './db.ts';
import { apiRoutes } from './routes/api.ts';
import { notFoundPage, pageRoutes } from './routes/pages.tsx';
import { staticRoutes } from './static.ts';
import { modelProbe, type ProbeFn } from './status.ts';

/**
 * The public site: pages and static files only. Caddy proxies localinference.alexwoodka.com here. It has no /api
 * routes, so nothing reachable from the internet can write.
 */
export function createPagesApp(db: Db, probe: ProbeFn = modelProbe(process.env.MODEL_HEALTH_URL)) {
  const app = new Hono();
  staticRoutes(app);
  app.route('/', pageRoutes(db, probe));
  // Hono runs only the top-level app's not-found handler, so it has to live here.
  app.notFound(notFoundPage);
  return app;
}

/** The publishing API for `lab` on ai: published on 127.0.0.1 only and reached over the tailnet via `tailscale serve`. */
export function createApiApp(db: Db) {
  const app = new Hono();
  app.route('/api', apiRoutes(db));
  app.notFound((c) => c.json({ error: 'not found' }, 404));
  return app;
}
