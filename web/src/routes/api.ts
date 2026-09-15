import { timingSafeEqual } from 'node:crypto';
import { Hono } from 'hono';
import type { MiddlewareHandler } from 'hono';
import { HostedBody, IngestBody, PauseBody } from '../contract.ts';
import { contentHash, tx, type Db } from '../db.ts';

class Conflict extends Error {}

function requireToken(): MiddlewareHandler {
  const expected = Buffer.from(process.env.INGEST_TOKEN ?? '');
  return async (c, next) => {
    if (expected.length === 0) return c.json({ error: 'INGEST_TOKEN not configured on server' }, 503);
    const given = Buffer.from((c.req.header('authorization') ?? '').replace(/^Bearer\s+/i, ''));
    if (given.length !== expected.length || !timingSafeEqual(given, expected)) {
      return c.json({ error: 'unauthorized' }, 401);
    }
    await next();
  };
}

type Row = Record<string, unknown>;

export function ingest(db: Db, body: IngestBody, replace: boolean) {
  return tx(db, () => {
    const existing = db.prepare('SELECT bundle_sha FROM runs WHERE id = ?').get(body.run.id) as Row | undefined;
    if (existing) {
      if (existing.bundle_sha === body.bundle_sha) return { status: 'unchanged' as const };
      if (!replace) throw new Conflict(`run ${body.run.id} already published with different content (use --replace)`);
      db.prepare('DELETE FROM runs WHERE id = ?').run(body.run.id);
    }

    const b = body.base_model;
    db.prepare(
      `INSERT INTO base_models (slug, name, family, params_b, active_params_b, arch, hf_repo)
       VALUES (:slug, :name, :family, :params_b, :active_params_b, :arch, :hf_repo)
       ON CONFLICT (slug) DO UPDATE SET name = excluded.name, family = excluded.family, params_b = excluded.params_b,
         active_params_b = excluded.active_params_b, arch = excluded.arch, hf_repo = excluded.hf_repo`,
    ).run({ ...b, family: b.family ?? null, params_b: b.params_b ?? null, active_params_b: b.active_params_b ?? null, hf_repo: b.hf_repo ?? null });
    const baseId = (db.prepare('SELECT id FROM base_models WHERE slug = ?').get(b.slug) as Row).id;

    const m = body.model;
    db.prepare(
      `INSERT INTO models (slug, name, base_model_id, engine, format, quant, bpw, file_size_bytes, source_repo, source_file, source_revision, notes)
       VALUES (:slug, :name, :base_model_id, :engine, :format, :quant, :bpw, :file_size_bytes, :source_repo, :source_file, :source_revision, :notes)
       ON CONFLICT (slug) DO UPDATE SET name = excluded.name, base_model_id = excluded.base_model_id, engine = excluded.engine,
         format = excluded.format, quant = excluded.quant, bpw = excluded.bpw, file_size_bytes = excluded.file_size_bytes,
         source_repo = excluded.source_repo, source_file = excluded.source_file, source_revision = excluded.source_revision, notes = excluded.notes`,
    ).run({
      slug: m.slug, name: m.name, base_model_id: baseId as number, engine: m.engine, format: m.format, quant: m.quant,
      bpw: m.bpw ?? null, file_size_bytes: m.file_size_bytes ?? null, source_repo: m.source_repo ?? null,
      source_file: m.source_file ?? null, source_revision: m.source_revision ?? null, notes: m.notes ?? null,
    });
    const modelId = (db.prepare('SELECT id FROM models WHERE slug = ?').get(m.slug) as Row).id as number;

    const cfg = body.config;
    const bySlug = db.prepare('SELECT id, config_hash FROM configs WHERE model_id = ? AND slug = ?').get(modelId, cfg.slug) as Row | undefined;
    if (bySlug && bySlug.config_hash !== cfg.config_hash) {
      throw new Conflict(`config ${m.slug}/${cfg.slug} was published with different settings; create a new config slug`);
    }
    db.prepare(
      `INSERT INTO configs (model_id, slug, name, config_hash, params_json, launch_command, engine_files_json, notes)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT (config_hash) DO UPDATE SET name = excluded.name, notes = excluded.notes, launch_command = excluded.launch_command`,
    ).run(modelId, cfg.slug, cfg.name, cfg.config_hash, JSON.stringify(cfg.params), cfg.launch_command, JSON.stringify(cfg.engine_files), cfg.notes ?? null);
    const configRow = db.prepare('SELECT id, model_id FROM configs WHERE config_hash = ?').get(cfg.config_hash) as Row;
    if (configRow.model_id !== modelId) throw new Conflict(`config hash ${cfg.config_hash} belongs to a different model`);

    const hw = body.hardware;
    const hwHash = contentHash(hw);
    db.prepare(
      `INSERT OR IGNORE INTO hardware_snapshots (hash, gpu_name, gpu_vram_mb, driver, cuda, power_limit_w, pcie, cpu_model, cpu_threads_visible, ram_mb, kernel, os)
       VALUES (:hash, :gpu_name, :gpu_vram_mb, :driver, :cuda, :power_limit_w, :pcie, :cpu_model, :cpu_threads_visible, :ram_mb, :kernel, :os)`,
    ).run({ hash: hwHash, ...hw });
    const hwId = (db.prepare('SELECT id FROM hardware_snapshots WHERE hash = ?').get(hwHash) as Row).id as number;

    const eb = body.engine_build;
    // bench_build_commit etc. are per-run observations, not build identity
    const ebIdentity = { engine: eb.engine, version: eb.version, commit_sha: eb.commit_sha, build_flags: eb.build_flags, dirty: eb.extra.dirty ?? false };
    const ebHash = contentHash(ebIdentity);
    db.prepare(
      `INSERT OR IGNORE INTO engine_builds (hash, engine, version, commit_sha, build_flags, extra_json) VALUES (?, ?, ?, ?, ?, ?)`,
    ).run(ebHash, eb.engine, eb.version ?? null, eb.commit_sha ?? null, eb.build_flags ?? null, JSON.stringify(eb.extra));
    const ebId = (db.prepare('SELECT id FROM engine_builds WHERE hash = ?').get(ebHash) as Row).id as number;

    let refId: number | null = null;
    if (body.quality_ref) {
      const q = body.quality_ref;
      db.prepare(
        `INSERT OR IGNORE INTO quality_refs (base_model_id, ref_label, dataset, ctx, chunks, logits_sha) VALUES (?, ?, ?, ?, ?, ?)`,
      ).run(baseId as number, q.ref_label, q.dataset, q.ctx, q.chunks, q.logits_sha ?? null);
      refId = (db.prepare('SELECT id FROM quality_refs WHERE base_model_id = ? AND ref_label = ? AND dataset = ? AND ctx = ? AND chunks = ?')
        .get(baseId as number, q.ref_label, q.dataset, q.ctx, q.chunks) as Row).id as number;
    }

    const r = body.run;
    db.prepare(
      `INSERT INTO runs (id, config_id, hardware_id, engine_build_id, quality_ref_id, kind, tier, started_at, duration_s, lab_version,
         cli_args, throttled, notes, telemetry_json, raw_json, bundle_sha)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(r.id, configRow.id as number, hwId, ebId, refId, r.kind, r.tier ?? null, r.started_at, r.duration_s ?? null, r.lab_version,
      r.cli_args, r.throttled ? 1 : 0, r.notes ?? null, JSON.stringify(r.telemetry), JSON.stringify(r.raw), body.bundle_sha);

    const insMetric = db.prepare(
      `INSERT INTO metrics (run_id, key, method, n_prompt, n_gen, depth, concurrency, value, stddev, n, unit, samples_json)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    for (const x of body.metrics) {
      insMetric.run(r.id, x.key, x.method, x.n_prompt, x.n_gen, x.depth, x.concurrency, x.value, x.stddev ?? null, x.n ?? null, x.unit,
        x.samples ? JSON.stringify(x.samples) : null);
    }
    const insEval = db.prepare(
      `INSERT INTO eval_results (run_id, task, metric, filter, value, stderr, n_samples, limit_n, lm_eval_version, gen_kwargs_json)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    );
    for (const e of body.eval_results) {
      insEval.run(r.id, e.task, e.metric, e.filter, e.value, e.stderr ?? null, e.n_samples ?? null, e.limit_n ?? null,
        e.lm_eval_version ?? null, JSON.stringify(e.gen_kwargs));
    }
    return { status: 'created' as const };
  });
}

export function apiRoutes(db: Db) {
  const api = new Hono();
  api.use('*', requireToken());

  api.post('/ingest', async (c) => {
    const parsed = IngestBody.safeParse(await c.req.json().catch(() => null));
    if (!parsed.success) return c.json({ error: 'invalid bundle', issues: parsed.error.issues.slice(0, 20) }, 400);
    try {
      const result = ingest(db, parsed.data, c.req.query('replace') === '1');
      return c.json({ ...result, run_id: parsed.data.run.id, url: `/m/${parsed.data.model.slug}?c=${parsed.data.config.slug}` });
    } catch (err) {
      if (err instanceof Conflict) return c.json({ error: err.message }, 409);
      throw err;
    }
  });

  api.delete('/runs/:id', (c) => {
    const res = db.prepare('DELETE FROM runs WHERE id = ?').run(c.req.param('id'));
    return res.changes ? c.json({ deleted: true }) : c.json({ error: 'not found' }, 404);
  });

  api.put('/hosted', async (c) => {
    const parsed = HostedBody.safeParse(await c.req.json().catch(() => null));
    if (!parsed.success) return c.json({ error: 'invalid body' }, 400);
    db.prepare(
      `INSERT INTO hosted (id, config_hash, since) VALUES (1, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
       ON CONFLICT (id) DO UPDATE SET config_hash = excluded.config_hash, since = excluded.since`,
    ).run(parsed.data.config_hash);
    return c.json({ ok: true });
  });

  api.put('/hosted/pause', async (c) => {
    const parsed = PauseBody.safeParse(await c.req.json().catch(() => null));
    if (!parsed.success) return c.json({ error: 'invalid body', issues: parsed.error.issues.slice(0, 5) }, 400);
    const { paused, reason, ref } = parsed.data;
    db.prepare(
      `INSERT INTO hosted_pause (id, paused, reason, ref, since) VALUES (1, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
       ON CONFLICT (id) DO UPDATE SET paused = excluded.paused, reason = excluded.reason, ref = excluded.ref, since = excluded.since`,
    ).run(paused ? 1 : 0, reason, ref);
    return c.json({ ok: true });
  });

  return api;
}
