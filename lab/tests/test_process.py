import time
from pathlib import Path

import pytest

from lab.engines.process import ServerExited, ServerProcess


def _gone(pid: int, timeout_s: float = 5) -> bool:
    """True once `pid` no longer exists (or is only a zombie waiting for init to reap it)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = Path(f"/proc/{pid}/status")
        try:
            if "State:\tZ" in status.read_text():
                return True
        except FileNotFoundError:
            return True
        time.sleep(0.05)
    return False


def _wait_for(path: Path, timeout_s: float = 5) -> str:
    deadline = time.monotonic() + timeout_s
    while not (path.exists() and path.read_text().strip()):
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)
    return path.read_text().strip()


def test_stop_kills_children_the_launcher_forked(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    server = ServerProcess(["sh", "-c", f"sleep 300 & echo $! > {child_pid_file}; exec sleep 300"])
    child = int(_wait_for(child_pid_file))
    server.stop(timeout_s=5)
    assert _gone(server.pid) and _gone(child)


def test_stop_escalates_to_sigkill_when_sigterm_is_ignored(tmp_path):
    ready = tmp_path / "ready"
    server = ServerProcess(["sh", "-c", f"trap '' TERM; echo ok > {ready}; sleep 300; sleep 300"])
    _wait_for(ready)
    started = time.monotonic()
    server.stop(timeout_s=0.5)
    assert _gone(server.pid)
    assert time.monotonic() - started < 5


def test_wait_healthy_fails_fast_when_the_server_dies():
    server = ServerProcess(["sh", "-c", "exit 3"])
    with pytest.raises(ServerExited, match="code 3"):
        server.wait_healthy("http://127.0.0.1:9/health", timeout_s=30, interval_s=0.05)


def test_cmdline_follows_exec(tmp_path):
    ready = tmp_path / "ready"
    with ServerProcess(["sh", "-c", f"echo ok > {ready}; exec sleep 300"]) as server:
        _wait_for(ready)
        deadline = time.monotonic() + 5
        while server.cmdline()[:1] != ["sleep"] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.cmdline() == ["sleep", "300"]
