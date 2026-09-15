import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { DatabaseSync } from 'node:sqlite';

export type Db = DatabaseSync;

export function openDb(path = process.env.DB_PATH ?? 'data/lab.db'): Db {
  if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true });
  const db = new DatabaseSync(path);
  db.exec('PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;');
  db.exec(readFileSync(new URL('./db/schema.sql', import.meta.url), 'utf8'));
  migrate(db);
  return db;
}

/** schema.sql only creates missing tables, so existing databases need their new columns added here. */
export function migrate(db: Db): void {
  const { user_version } = db.prepare('PRAGMA user_version').get() as { user_version: number };
  if (user_version >= 2) return;
  tx(db, () => {
    const have = new Set((db.prepare("PRAGMA table_info('eval_results')").all() as { name: string }[]).map((c) => c.name));
    const v2Columns: [string, string][] = [
      ['harness', 'TEXT'],
      ['harness_version', 'TEXT'],
      ['subset_id', 'TEXT'],
      ['n_tasks', 'INTEGER'],
      ['attempts_per_task', 'INTEGER'],
    ];
    for (const [name, type] of v2Columns) {
      if (!have.has(name)) db.exec(`ALTER TABLE eval_results ADD COLUMN ${name} ${type}`);
    }
    db.exec('PRAGMA user_version = 2');
  });
}

export function tx<T>(db: Db, fn: () => T): T {
  db.exec('BEGIN IMMEDIATE');
  try {
    const result = fn();
    db.exec('COMMIT');
    return result;
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(',')}}`;
}

export function contentHash(value: unknown): string {
  return createHash('sha256').update(stableStringify(value)).digest('hex');
}
