"""The always-on hosted model (systemd user unit) and the pause/resume flow around GPU jobs.

Order matters: the hosted unit's ExecStartPre refuses to start while gpu.lock is held,
so the hosted model is restarted only *after* the lock is released.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pynvml

from lab import paths
from lab.gpu.lock import gpu_lock, is_locked
from lab.gpu.power import check_power_limit

UNIT = "lab-hosted.service"
PAUSED_BY = paths.STATE / "paused_by.json"
IDLE_VRAM_MB = 300


class GpuBusy(RuntimeError):
    pass


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def hosted_active() -> bool:
    try:
        return _systemctl("is-active", "--quiet", UNIT).returncode == 0
    except FileNotFoundError:
        return False


def hosted_state() -> dict | None:
    try:
        return json.loads(paths.HOSTED_JSON.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def stop_hosted() -> None:
    _systemctl("stop", UNIT)


def start_hosted(wait_health: bool = True, timeout_s: float = 600) -> bool:
    r = _systemctl("start", UNIT)
    if r.returncode != 0:
        print(f"warning: failed to start {UNIT}: {r.stderr.strip()}", file=sys.stderr)
        return False
    return wait_for_health(timeout_s) if wait_health else True


def wait_for_health(timeout_s: float = 600) -> bool:
    state = hosted_state()
    if not state:
        return False
    url = f"http://127.0.0.1:{state['port']}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if not hosted_active():
            return False
        time.sleep(2)
    return False


def gpu_vram_used_mb(gpu_index: int = 0) -> float:
    from lab.telemetry import vram_used_mb

    pynvml.nvmlInit()
    return vram_used_mb(pynvml.nvmlDeviceGetHandleByIndex(gpu_index))


def gpu_temp_c(gpu_index: int = 0) -> int:
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(gpu_index), pynvml.NVML_TEMPERATURE_GPU)


def wait_gpu_idle(timeout_s: float = 90) -> None:
    """PIDs are not visible across the LXC pid namespace, so idleness is judged by VRAM in use."""
    deadline = time.monotonic() + timeout_s
    while (used := gpu_vram_used_mb()) > IDLE_VRAM_MB:
        if time.monotonic() > deadline:
            raise GpuBusy(
                f"{used:.0f} MiB VRAM still in use after stopping the hosted model; "
                "another GPU process is running (e.g. a manual llama-server in tmux)"
            )
        time.sleep(1)


def wait_cool(max_temp_c: int = 45, timeout_s: float = 600) -> None:
    deadline = time.monotonic() + timeout_s
    while (t := gpu_temp_c()) > max_temp_c and time.monotonic() < deadline:
        print(f"cooling down: GPU {t} °C > {max_temp_c} °C", file=sys.stderr)
        time.sleep(10)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _report_pause(paused: bool, purpose: str = "") -> None:
    """Tell the site why the chat model is down, or that it's back.

    Never raises: a GPU job must not fail or stall because the site is unreachable (put_pause gives up after about a second).
    """
    try:
        from lab.publish import put_pause

        words = purpose.split()
        reason = words[0] if words and words[0] in ("bench", "serve") else None
        ref = next((w for w in words if "/" in w), None)
        put_pause(paused, reason if paused else None, ref if paused else None)
    except Exception:  # noqa: BLE001
        pass


def recover_paused_hosted() -> bool:
    """If a GPU job died without resuming the hosted model, resume it. Returns True if it acted."""
    try:
        info = json.loads(PAUSED_BY.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if _pid_alive(info.get("pid", -1)) or is_locked():
        return False
    PAUSED_BY.unlink(missing_ok=True)
    if not hosted_active():
        print(f"recovering: restarting hosted model paused by dead job {info}", file=sys.stderr)
        start_hosted(wait_health=False)
    _report_pause(False)
    return True


@contextmanager
def exclusive_gpu(purpose: str, wait: bool = False, keep_paused: bool = False, cool: bool = True) -> Iterator[None]:
    was_active = False
    try:
        with gpu_lock(purpose, wait=wait):
            # Before touching the hosted model, so a refusal never pauses chat.
            check_power_limit()
            was_active = hosted_active()
            if was_active:
                PAUSED_BY.write_text(json.dumps({"pid": os.getpid(), "purpose": purpose}))
                print("pausing hosted model…", file=sys.stderr)
                stop_hosted()
                # Before the idle and cool-down waits, so the report never lands inside a measured window.
                _report_pause(True, purpose)
            wait_gpu_idle()
            if cool:
                wait_cool()
            yield
    finally:
        if was_active:
            if not keep_paused:
                print("resuming hosted model…", file=sys.stderr)
                start_hosted(wait_health=False)
            _report_pause(False)
        PAUSED_BY.unlink(missing_ok=True)
