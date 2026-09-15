"""Run lifecycle shared by every benchmark kind: scaffold bundle → run → finish ok/failed."""

import importlib.metadata
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from lab import catalog, paths
from lab.catalog import ConfigSpec, ModelSpec
from lab.engines.base import Engine
from lab.schema import ConfigInfo, Metric, RunBundle, RunRecord
from lab.snapshot import hardware_snapshot
from lab.store import RunDir, new_run_dir
from lab.telemetry import Telemetry


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def lab_version() -> str:
    try:
        version = importlib.metadata.version("lab")
    except importlib.metadata.PackageNotFoundError:
        version = "0"
    try:
        sha = subprocess.run(
            ["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(paths.ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if sha:
            version += f"+{sha}{'.dirty' if dirty else ''}"
    except FileNotFoundError:
        pass
    return version


@dataclass
class RunContext:
    rd: RunDir
    bundle: RunBundle
    t0: float


def start_run(
    model: ModelSpec,
    cfg: ConfigSpec,
    kind: str,
    engine: Engine,
    weights: Path,
    cli_args: str,
    tier: str | None = None,
) -> RunContext:
    base = catalog.load_base(model.base)
    run_id, rd = new_run_dir(f"{model.slug}_{cfg.slug}", kind)
    bundle = RunBundle(
        base_model=catalog.base_info(base),
        model=catalog.model_info(model, catalog.weights_size(weights)),
        config=ConfigInfo(
            slug=cfg.slug,
            name=cfg.name,
            config_hash=catalog.config_hash(model, cfg),
            params=cfg.params.model_dump(mode="json"),
            launch_command=engine.display_command(model, cfg),
            engine_files=engine.render_files(model, cfg),
            notes=cfg.notes,
        ),
        hardware=hardware_snapshot(),
        engine_build=engine.build_info(),
        run=RunRecord(
            id=run_id,
            kind=kind,
            tier=tier,
            started_at=now_iso(),
            lab_version=lab_version(),
            cli_args=cli_args,
            raw={"bench": cfg.bench.model_dump(mode="json")},
        ),
    )
    rd.save(bundle)
    return RunContext(rd, bundle, time.monotonic())


def finish_ok(ctx: RunContext, metrics: list[Metric], tel: Telemetry | None, raw: dict | None = None) -> None:
    run = ctx.bundle.run
    ctx.bundle.metrics = metrics
    run.status = "ok"
    run.duration_s = round(time.monotonic() - ctx.t0, 2)
    if tel is not None:
        summary = tel.summary()
        run.throttled = summary["throttled"]
        run.telemetry = tel.downsampled()
        run.raw["telemetry_summary"] = summary
    if raw:
        run.raw.update(raw)
    ctx.rd.save(ctx.bundle)


def finish_failed(ctx: RunContext, err: BaseException, tel: Telemetry | None = None) -> None:
    run = ctx.bundle.run
    run.status = "failed"
    run.duration_s = round(time.monotonic() - ctx.t0, 2)
    run.error = "".join(traceback.format_exception_only(err)).strip()
    if tel is not None:
        run.telemetry = tel.downsampled()
    ctx.rd.save(ctx.bundle)
