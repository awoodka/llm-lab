import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
import typer
import yaml

from lab import catalog, paths, store
from lab.catalog import BaseSpec, ConfigSpec, ModelSpec, Source
from lab.engines.base import get_engine
from lab.gpu import hosted
from lab.gpu.lock import gpu_lock, is_locked, read_holder
from lab.gpu.power import PowerLimitTooHigh, check_power_limit, max_power_w

app = typer.Typer(no_args_is_help=True, help="Local LLM performance lab")
model_app = typer.Typer(no_args_is_help=True, help="Manage models (base model + quant)")
config_app = typer.Typer(no_args_is_help=True, help="Manage settings configs for a model")
runs_app = typer.Typer(no_args_is_help=True, help="Inspect local runs")
hosted_app = typer.Typer(no_args_is_help=True, help="Control the always-on hosted model")
gpu_app = typer.Typer(no_args_is_help=True, help="Run commands with the GPU to themselves")
app.add_typer(model_app, name="model")
app.add_typer(config_app, name="config")
app.add_typer(runs_app, name="runs")
app.add_typer(hosted_app, name="hosted")
app.add_typer(gpu_app, name="gpu")

HOSTED_PORT = 8080
SERVE_PORT = 8081
QUANT_RE = re.compile(r"(IQ\d_[A-Z]+(?:_[A-Z])?|Q\d_K_[SML]|Q\d_K|Q\d_\d|MXFP4|BF16|F16|F32)", re.I)


def _cli_args() -> str:
    return shlex.join(["lab", *sys.argv[1:]])


def _fail(msg: str) -> None:
    typer.secho(msg, fg="red", err=True)
    raise typer.Exit(1)


def _check_power() -> None:
    """Refuse GPU work up front, before the lock or the hosted model are touched, when the power cap isn't in force."""
    try:
        check_power_limit()
    except PowerLimitTooHigh as e:
        _fail(str(e))


@app.callback()
def _startup() -> None:
    hosted.recover_paused_hosted()


# -- gpu ---------------------------------------------------------------------
@gpu_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def gpu_run(
    ctx: typer.Context,
    label: str = typer.Option("manual", help="What the site names while chat is paused, e.g. qwen-serving/probe"),
    wait: bool = typer.Option(False, help="Queue behind a held GPU lock instead of failing"),
) -> None:
    """Run a command with the GPU to itself: chat pauses, the command runs in its own process group, chat resumes.

    Usage: lab gpu run --label x/y -- COMMAND ARGS... Ctrl-C or SIGTERM stops the command's whole group.
    """
    import signal

    from lab.engines.process import ServerProcess

    if not ctx.args:
        _fail("usage: lab gpu run [--label x/y] -- <command> [args...]")
    _check_power()

    def _interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt)
    rc = 130
    with hosted.exclusive_gpu(f"serve {label}", wait=wait, cool=False):
        server = ServerProcess(list(ctx.args))
        try:
            rc = server.wait()
        except KeyboardInterrupt:
            server.stop()
    raise typer.Exit(rc)


# -- doctor ------------------------------------------------------------------
@app.command()
def doctor() -> None:
    """Check GPU, lock, hosted model, llama.cpp build and disk."""
    import shutil

    from lab.snapshot import hardware_snapshot

    ok = True

    def line(good: bool, label: str, detail: str) -> None:
        nonlocal ok
        ok &= good
        typer.echo(f"{'✔' if good else '✘'} {label:<14} {detail}")

    hw = hardware_snapshot()
    cap = max_power_w()
    line(hw.power_limit_w <= cap + 0.5, "gpu",
         f"{hw.gpu_name} {hw.gpu_vram_mb} MiB, driver {hw.driver}, CUDA {hw.cuda}, limit {hw.power_limit_w:.0f} W (max {cap:.0f} W)")
    line(True, "host", f"{hw.cpu_model}, {hw.cpu_threads_visible} threads, {hw.ram_mb} MiB RAM")
    used = hosted.gpu_vram_used_mb()
    line(True, "vram", f"{used:.0f} MiB in use, {hosted.gpu_temp_c()} °C")
    line(not is_locked(), "gpu lock", f"held by {read_holder()}" if is_locked() else "free")
    state = hosted.hosted_state()
    line(True, "hosted", f"{'active' if hosted.hosted_active() else 'inactive'}; promoted={state['ref'] if state else None}")
    try:
        b = get_engine("llama.cpp").build_info()
        line(not b.extra.get("dirty"), "llama.cpp", f"{b.commit_sha} build {b.version} {'(DIRTY tree)' if b.extra.get('dirty') else ''}")
    except Exception as e:  # noqa: BLE001
        line(False, "llama.cpp", str(e))
    models_on_nvme = paths.MODELS == Path("/mnt/models")
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(paths.MODELS).free / 2**30
    line(models_on_nvme, "models dir", f"{paths.MODELS} ({free:.0f} GiB free){'' if models_on_nvme else ' — NVMe not mounted yet'}")
    line(os.environ.get("HF_HOME", "").startswith(str(paths.MODELS)), "HF_HOME", os.environ.get("HF_HOME", "(unset → ~/.cache/huggingface)"))
    try:
        from lab.publish import settings

        web_url = settings()[0]
        httpx.get(web_url, timeout=5)  # any HTTP response means the site is reachable
        line(True, "web", web_url)
    except Exception as e:  # noqa: BLE001
        line(False, "web", str(e))
    raise typer.Exit(0 if ok else 1)


# -- models ------------------------------------------------------------------
@model_app.command("add")
def model_add(
    repo: str = typer.Argument(..., help="HF repo, e.g. ggml-org/gemma-3-4b-it-GGUF"),
    file: str = typer.Argument(..., help="GGUF file within the repo"),
    base: str = typer.Option(..., help="Base model slug, e.g. gemma-3-4b-it"),
    quant: str | None = typer.Option(None, help="Quant label (guessed from file name)"),
    slug: str | None = typer.Option(None, help="Model slug (default: <base>-<quant>-gguf)"),
    name: str | None = typer.Option(None),
    revision: str | None = typer.Option(None),
) -> None:
    """Download a GGUF from the HF Hub and register it."""
    from huggingface_hub import hf_hub_download

    quant = quant or ((m := QUANT_RE.search(file)) and m.group(1).upper())
    if not quant:
        _fail("could not guess the quant from the file name; pass --quant")
    slug = slug or catalog.slugify(f"{base}-{quant}-gguf")
    path = Path(hf_hub_download(repo_id=repo, filename=file, revision=revision))
    resolved_rev = path.parent.name if path.parent.parent.name == "snapshots" else revision
    if not catalog.base_path(base).is_file():
        catalog.save_base(BaseSpec(slug=base, name=base, hf_repo=None))
        typer.echo(f"created base stub {catalog.base_path(base)} — fill in family/params_b/arch")
    size_gb = path.stat().st_size / 1e9
    spec = ModelSpec(
        slug=slug,
        name=name or f"{catalog.load_base(base).name} {quant}",
        base=base,
        quant=quant,
        source=Source(repo=repo, file=file, revision=resolved_rev),
    )
    typer.echo(f"saved {catalog.save_model(spec)} ({size_gb:.2f} GB at {path})")


@model_app.command("ls")
def model_ls() -> None:
    for m in catalog.list_models():
        cfgs = ", ".join(c.slug for c in catalog.list_configs(m.slug)) or "-"
        typer.echo(f"{m.slug:<40} {m.engine:<10} {m.quant:<8} configs: {cfgs}")


# -- configs -----------------------------------------------------------------
def _set_dotted(data: dict, key: str, value: str) -> None:
    parts = key.split(".")
    if parts[0] not in ("params", "bench", "eval_overrides", "name", "notes"):
        parts = ["params", *parts]
    cur = data
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(value)


@config_app.command("new")
def config_new(
    model: str,
    slug: str,
    name: str | None = typer.Option(None),
    from_: str | None = typer.Option(None, "--from", help="Copy an existing config slug of this model"),
    set_: list[str] = typer.Option([], "--set", help="key=value, e.g. ctx=32768, cache_type_k=q8_0, bench.depths=[0,8192]"),
) -> None:
    """Create a settings config. Keys without a prefix go to params."""
    catalog.load_model(model)
    if catalog.config_path(model, slug).exists():
        _fail(f"config {model}/{slug} already exists; configs are immutable once benchmarked — pick a new slug")
    data = catalog.load_config(f"{model}/{from_}")[1].model_dump(mode="json") if from_ else {}
    data.update(slug=slug, name=name or slug)
    for kv in set_:
        k, _, v = kv.partition("=")
        _set_dotted(data, k.strip(), v.strip())
    cfg = ConfigSpec.model_validate(data)
    typer.echo(f"saved {catalog.save_config(model, cfg)}")


@config_app.command("show")
def config_show(ref: str, cmd: bool = typer.Option(False, "--cmd", help="Only print the launch command")) -> None:
    model, cfg = catalog.load_config(ref)
    engine = get_engine(model.engine)
    if not cmd:
        typer.echo(yaml.safe_dump(cfg.model_dump(mode="json", exclude_none=True), sort_keys=False))
        typer.echo(f"config_hash: {catalog.config_hash(model, cfg)}")
    typer.echo(shlex.join(engine.server_argv(model, cfg, host="127.0.0.1", port=SERVE_PORT)))


# -- serve / bench -----------------------------------------------------------
@app.command()
def serve(
    ref: str,
    port: int = typer.Option(SERVE_PORT),
    host: str = typer.Option("127.0.0.1", help="Use 0.0.0.0 to reach it from the tailnet while tinkering"),
    wait: bool = typer.Option(False),
) -> None:
    """Run a config interactively in the foreground (pauses the hosted model; Ctrl-C resumes it)."""
    model, cfg = catalog.load_config(ref)
    argv = get_engine(model.engine).server_argv(model, cfg, host=host, port=port)
    typer.echo(shlex.join(argv))
    _check_power()
    with hosted.exclusive_gpu(f"serve {ref}", wait=wait, cool=False):
        proc = subprocess.Popen(argv)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()


@app.command()
def bench(
    ref: str,
    speed: bool = typer.Option(False, "--speed", help="llama-bench native speed (default if nothing selected)"),
    reps: int | None = typer.Option(None, help="Override bench.reps"),
    wait: bool = typer.Option(False, help="Queue behind a held GPU lock instead of failing"),
    keep_paused: bool = typer.Option(False, help="Leave the hosted model stopped afterwards"),
) -> None:
    """Benchmark a config and store a local run (publish separately)."""
    from lab.bench.native_llamacpp import run_speed

    _check_power()
    rd = run_speed(ref, wait=wait, keep_paused=keep_paused, reps=reps, cli_args=_cli_args())
    _print_run(rd)
    typer.echo(f"\nrun saved: {rd.path}\npublish with: lab publish {rd.path.name}")


# -- runs --------------------------------------------------------------------
def _print_run(rd: store.RunDir) -> None:
    b = rd.load()
    r = b.run
    typer.echo(f"{rd.path.name}  id={r.id}  status={r.status}  {r.duration_s}s  throttled={r.throttled}")
    typer.echo(f"  {b.model.slug} / {b.config.slug}  engine {b.engine_build.engine}@{b.engine_build.commit_sha}")
    if r.error:
        typer.secho(f"  error: {r.error}", fg="red")
    for m in b.metrics:
        dims = f"p{m.n_prompt} g{m.n_gen} d{m.depth}"
        sd = f" ± {m.stddev:.2f}" if m.stddev is not None else ""
        typer.echo(f"  {m.key:<18} {dims:<18} {m.value:>10.2f}{sd} {m.unit}")


@runs_app.command("ls")
def runs_ls(unpublished: bool = typer.Option(False)) -> None:
    for rd in store.list_runs():
        b = rd.load()
        if unpublished and b.run.published_at:
            continue
        pub = "published" if b.run.published_at else "local"
        typer.echo(f"{rd.path.name:<70} {b.run.status:<7} {pub}")


@runs_app.command("show")
def runs_show(ref: str) -> None:
    _print_run(store.find_run(ref))


# -- publish -----------------------------------------------------------------
@app.command()
def publish(
    refs: list[str],
    dry_run: bool = typer.Option(False),
    replace: bool = typer.Option(False, help="Overwrite a published run whose content changed"),
) -> None:
    """Publish local runs to the showcase site."""
    from lab.publish import PublishError, publish as do_publish

    for ref in refs:
        rd = store.find_run(ref)
        try:
            typer.echo(f"{rd.path.name}: {json.dumps(do_publish(rd, dry_run=dry_run, replace=replace))}")
        except PublishError as e:
            _fail(f"{rd.path.name}: {e}")


@app.command()
def unpublish(ref: str) -> None:
    from lab.publish import unpublish as do_unpublish

    rd = store.find_run(ref)
    do_unpublish(rd)
    typer.echo(f"unpublished {rd.path.name}")


# -- hosted ------------------------------------------------------------------
UNIT_TEMPLATE = """[Unit]
Description=lab hosted LLM (promoted config)

[Service]
Type=simple
ExecStartPre=/usr/bin/flock -n {lock} true
ExecStartPre={python} -m lab.gpu.power
ExecStart={script}
Restart=on-failure
RestartSec=30
TimeoutStopSec=60
KillMode=mixed

[Install]
WantedBy=default.target
"""


def _install_unit() -> None:
    unit = Path.home() / ".config/systemd/user" / hosted.UNIT
    content = UNIT_TEMPLATE.format(lock=paths.GPU_LOCK, script=paths.HOSTED_SH, python=sys.executable)
    if not unit.is_file() or unit.read_text() != content:
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", hosted.UNIT], check=True, capture_output=True)


def _rollback_promote(ref: str, previous: dict[Path, str]) -> None:
    """`ref` never became healthy: end the unit's restart loop and bring back the model hosted before it."""
    hosted.stop_hosted()
    logs = f"journalctl --user -u {hosted.UNIT} -n 50"
    prev_ref = json.loads(previous.get(paths.HOSTED_JSON, "{}")).get("ref")
    if not prev_ref or prev_ref == ref:
        _fail(f"{ref} failed to become healthy and there is no previous model to roll back to; the hosted model is stopped. See: {logs}")
    for p, text in previous.items():
        p.write_text(text)
    if paths.HOSTED_SH in previous:
        paths.HOSTED_SH.chmod(0o755)
    healthy = hosted.start_hosted(wait_health=True)
    _fail(f"{ref} failed to become healthy (see: {logs}); rolled back to {prev_ref}, "
          f"which is {'healthy again' if healthy else 'NOT healthy either'}")


@app.command()
def promote(ref: str) -> None:
    """Make a config the always-on hosted model. If it never becomes healthy, the previous one comes back."""
    model, cfg = catalog.load_config(ref)
    engine = get_engine(model.engine)
    argv = engine.server_argv(model, cfg, host="127.0.0.1", port=HOSTED_PORT)
    _check_power()
    previous = {p: p.read_text() for p in (paths.HOSTED_SH, paths.HOSTED_JSON) if p.is_file()}
    with gpu_lock(f"promote {ref}"):
        if hosted.hosted_active():
            hosted.stop_hosted()
        paths.HOSTED_SH.write_text(f"#!/bin/sh\nexec {shlex.join(argv)}\n")
        paths.HOSTED_SH.chmod(0o755)
        state = {"ref": ref, "config_hash": catalog.config_hash(model, cfg), "port": HOSTED_PORT,
                 "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        paths.HOSTED_JSON.write_text(json.dumps(state, indent=2))
        _install_unit()
        hosted.wait_gpu_idle()
    typer.echo(f"starting {ref} on 127.0.0.1:{HOSTED_PORT}…")
    if not hosted.start_hosted(wait_health=True):
        _rollback_promote(ref, previous)
    typer.echo("healthy")
    try:
        from lab.publish import put_hosted

        put_hosted(state["config_hash"])
    except Exception as e:  # noqa: BLE001
        typer.secho(f"note: site not updated ({e})", fg="yellow")


@hosted_app.command("status")
def hosted_status() -> None:
    typer.echo(json.dumps({"active": hosted.hosted_active(), "state": hosted.hosted_state(), "lock": read_holder()}, indent=2))


@hosted_app.command("start")
def hosted_start() -> None:
    if not hosted.start_hosted(wait_health=True):
        _fail("failed to start or not healthy")


@hosted_app.command("stop")
def hosted_stop() -> None:
    hosted.stop_hosted()
