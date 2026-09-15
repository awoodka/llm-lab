"""A server started in its own process group, so stopping it stops everything it spawned.

Launch scripts fork: vLLM's API server starts an EngineCore child that holds the VRAM. Signalling only the
direct child would leave that process running, and the next GPU job would find the card busy.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import httpx


class ServerExited(RuntimeError):
    pass


class ServerProcess:
    def __init__(self, argv: list[str], *, log_path: Path | None = None, env: dict[str, str] | None = None):
        self.argv = argv
        self._log = open(log_path, "ab") if log_path else None  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(
            argv, stdout=self._log, stderr=subprocess.STDOUT if self._log else None, env=env, start_new_session=True
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait(self) -> int:
        return self.proc.wait()

    def wait_healthy(self, url: str, timeout_s: float, interval_s: float = 2.0) -> float:
        """Poll `url` until it answers 200 and return the seconds waited.

        Raises ServerExited as soon as the server dies, and TimeoutError after `timeout_s`.
        """
        started = time.monotonic()
        while True:
            if not self.alive():
                raise ServerExited(f"server exited with code {self.proc.returncode} before it became healthy")
            try:
                if httpx.get(url, timeout=2).status_code == 200:
                    return time.monotonic() - started
            except httpx.HTTPError:
                pass
            if time.monotonic() - started > timeout_s:
                raise TimeoutError(f"{url} was not healthy after {timeout_s:.0f} s")
            time.sleep(interval_s)

    def cmdline(self) -> list[str]:
        """The argv the process is actually running now (after any `exec` in a launch script)."""
        return [a.decode() for a in Path(f"/proc/{self.pid}/cmdline").read_bytes().split(b"\0") if a]

    def environ(self) -> dict[str, str]:
        pairs = (e.decode(errors="replace").partition("=") for e in Path(f"/proc/{self.pid}/environ").read_bytes().split(b"\0") if e)
        return {k: v for k, _, v in pairs}

    def _group_alive(self) -> bool:
        self.proc.poll()  # reap the leader first: an unreaped zombie still counts as a group member
        try:
            os.killpg(self.pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal_group(self, sig: int) -> None:
        try:
            os.killpg(self.pid, sig)
        except ProcessLookupError:
            pass

    def _wait_group_gone(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while self._group_alive():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        return True

    def stop(self, timeout_s: float = 60) -> int | None:
        """SIGTERM the whole process group; SIGKILL whatever is still there after `timeout_s`."""
        self._signal_group(signal.SIGTERM)
        if not self._wait_group_gone(timeout_s):
            self._signal_group(signal.SIGKILL)
            self._wait_group_gone(10)
        if self._log:
            self._log.close()
        return self.proc.poll()

    def __enter__(self) -> "ServerProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
