import { serve } from '@hono/node-server';
import { createApiApp, createPagesApp } from './app.ts';
import { openDb } from './db.ts';

const db = openDb();
const hostname = process.env.HOST ?? '127.0.0.1';
const pagesPort = Number(process.env.PAGES_PORT ?? 3000);
const apiPort = Number(process.env.API_PORT ?? 3100);

serve({ fetch: createPagesApp(db).fetch, hostname, port: pagesPort }, (info) => console.log(`pages listening on http://${hostname}:${info.port}`));
serve({ fetch: createApiApp(db).fetch, hostname, port: apiPort }, (info) => console.log(`publishing API listening on http://${hostname}:${info.port}`));
