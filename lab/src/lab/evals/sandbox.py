"""Benchmarks whose harness runs the model's code live in a container on `web`, never on ai.

The container is the harness: it builds the prompts, grades the answers and runs whatever the model
wrote, with no network at all. It never reaches the model itself. It writes each chat request it wants
to its stdout; the lab sends that request through the allowance proxy and writes the answer back to
its stdin. So every request is still sized and logged in one place, ai installs no harness and runs no
benchmark code, and `web` only ever sees the conversation.

The channel is `ssh web docker run -i ...`, one JSON object per line:

    lab → box  {"op": "list"}                              box → lab  {"op": "tasks", "tasks": [...], "source": {...}}
    lab → box  {"op": "run", "task": ID, "attempt": N}
    box → lab  {"op": "chat", "id": K, "body": {...}}     lab → box  {"op": "reply", "id": K, "status": S, "body": {...}}
    box → lab  {"op": "log", "message": "..."}
    box → lab  {"op": "result", "task": ID, "attempt": N, "passed": B, "extracted": ..., "detail": {...}, "transcript": [...]}
    box → lab  {"op": "failed", "task": ID, "attempt": N, "message": "..."}     the harness broke, not the model

Images are built on `web` from `lab/sandbox/`, tagged with the hash of their build context, so a changed
harness is a new image and a run records exactly which one graded it.
"""

import hashlib
import io
import json
import os
import queue
import shlex
import subprocess
import sys
import tarfile
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lab import paths

SANDBOX_DIR = paths.ROOT / "sandbox"
#: Shared by every image: the container side of the channel.
COMMON = "common"
DEFAULT_HOST = "alex@web"

#: No network, nothing writable but a small /tmp, no privileges, bounded memory and processes.
#: `web` also runs the public site and chat, so a runaway benchmark must not starve them.
LIMITS = (
    "--network", "none",
    "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=512m",
    "--memory", "2g", "--memory-swap", "2g", "--cpus", "2", "--pids-limit", "256",
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
    "--user", "65534:65534",
)
#: With no message from the box for this long, it is considered hung. A chat request doesn't count:
#: while the lab is waiting on the model, the box is waiting on the lab.
IDLE_TIMEOUT_S = 15 * 60


class SandboxError(RuntimeError):
    """The sandbox failed, not the model: the task is retried, then excluded."""


def sandbox_host() -> str:
    if host := os.environ.get("LAB_SANDBOX_HOST"):
        return host
    if paths.SETTINGS.is_file():
        settings = yaml.safe_load(paths.SETTINGS.read_text()) or {}
        if host := settings.get("sandbox_host"):
            return str(host)
    return DEFAULT_HOST


def remote(host: str, argv: list[str]) -> list[str]:
    """An ssh command line for `argv` on `host`. ssh joins its arguments into one shell string, so quote them."""
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", host, shlex.join(argv)]


def _context_files(name: str) -> list[Path]:
    files = []
    for part in (COMMON, name):
        root = SANDBOX_DIR / part
        if not root.is_dir():
            raise SandboxError(f"no sandbox named {part!r} in {SANDBOX_DIR}")
        files += sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return files


def context_hash(name: str) -> str:
    h = hashlib.sha256()
    for f in _context_files(name):
        h.update(str(f.relative_to(SANDBOX_DIR)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _context_tar(name: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for f in _context_files(name):
            info = tar.gettarinfo(str(f), arcname=str(f.relative_to(SANDBOX_DIR)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(f, "rb") as fh:
                tar.addfile(info, fh)
    return buf.getvalue()


@dataclass(frozen=True)
class Image:
    name: str
    tag: str
    image_id: str
    host: str

    def as_raw(self) -> dict[str, str]:
        return {"tag": self.tag, "image_id": self.image_id, "host": self.host}


def _inspect(host: str, tag: str) -> str | None:
    r = subprocess.run(remote(host, ["docker", "image", "inspect", "--format", "{{.Id}}", tag]), capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def ensure_image(name: str, host: str | None = None, *, rebuild: bool = False) -> Image:
    """The image for sandbox `name`, built on the host if this exact context hasn't been built yet."""
    host = host or sandbox_host()
    tag = f"lab-{name}:{context_hash(name)[:12]}"
    if not rebuild and (image_id := _inspect(host, tag)):
        return Image(name, tag, image_id, host)
    print(f"building sandbox image {tag} on {host}…", file=sys.stderr)
    build = remote(host, ["docker", "build", "--quiet", "-t", tag, "-f", f"{name}/Dockerfile", "-"])
    r = subprocess.run(build, input=_context_tar(name), capture_output=True)
    if r.returncode != 0:
        raise SandboxError(f"building {tag} failed:\n{r.stderr.decode(errors='replace')[-4000:]}")
    image_id = _inspect(host, tag)
    if not image_id:
        raise SandboxError(f"built {tag} but can't find it on {host}")
    return Image(name, tag, image_id, host)


#: Answers a chat request: (status, body). The lab's implementation goes through the allowance proxy.
ChatFn = Callable[[dict[str, Any]], tuple[int, dict[str, Any]]]


class Box:
    """One long-lived harness container, and the loop that answers its chat requests.

    Tasks run one at a time. A box that dies, hangs or reports a harness failure raises SandboxError;
    the caller restarts it (`restart()`) before retrying.
    """

    def __init__(self, image: Image, args: list[str] | None = None, log_path: Path | None = None):
        self.image = image
        self.args = args or []
        self.log_path = log_path
        self.name = ""
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=50)

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> "Box":
        self.name = f"lab-{self.image.name}-{uuid.uuid4().hex[:8]}"
        argv = ["docker", "run", "-i", "--rm", "--name", self.name, *LIMITS, self.image.tag, *self.args]
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(
            remote(self.image.host, argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._read_stdout, args=(self._proc, self._lines), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()
        return self

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait()
        # A dropped ssh session can leave the container running on the host; make sure it's gone.
        subprocess.run(remote(self.image.host, ["docker", "rm", "-f", self.name]), capture_output=True)

    def restart(self) -> None:
        self.stop()
        self.start()

    def __enter__(self) -> "Box":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- the channel -----------------------------------------------------------
    def _read_stdout(self, proc: subprocess.Popen, lines: queue.Queue) -> None:
        assert proc.stdout
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr
        log = open(self.log_path, "a", buffering=1) if self.log_path else None  # noqa: SIM115 - closed below
        try:
            for line in proc.stderr:
                self._stderr.append(line.rstrip())
                if log:
                    log.write(line)
        finally:
            if log:
                log.close()

    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise SandboxError("the sandbox isn't running")
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except OSError as e:
            raise SandboxError(f"the sandbox stopped reading: {e}\n{self._tail()}") from e

    def _recv(self) -> dict[str, Any]:
        while True:
            try:
                line = self._lines.get(timeout=IDLE_TIMEOUT_S)
            except queue.Empty:
                raise SandboxError(f"no word from the sandbox in {IDLE_TIMEOUT_S // 60} minutes\n{self._tail()}") from None
            if line is None:
                raise SandboxError(f"the sandbox exited\n{self._tail()}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._stderr.append(f"(not protocol) {line.rstrip()[:200]}")
                continue
            if msg.get("op") == "log":
                print(f"    [{self.image.name}] {msg.get('message')}", file=sys.stderr)
                continue
            return msg

    def _tail(self) -> str:
        return "\n".join(self._stderr)

    def _expect(self, op: str, chat: ChatFn | None = None) -> dict[str, Any]:
        while True:
            msg = self._recv()
            kind = msg.get("op")
            if kind == op:
                return msg
            if kind == "chat" and chat is not None:
                status, body = chat(msg["body"])
                self._send({"op": "reply", "id": msg["id"], "status": status, "body": body})
                continue
            if kind == "failed":
                raise SandboxError(f"harness failure: {msg.get('message')}")
            raise SandboxError(f"unexpected message from the sandbox: {str(msg)[:300]}")

    # -- requests --------------------------------------------------------------
    def list_tasks(self) -> dict[str, Any]:
        """`{"tasks": [{"id": ..., "meta": {...}}, ...], "source": {...}}` as the harness reports them."""
        self._send({"op": "list"})
        return self._expect("tasks")

    def run(self, task: str, attempt: int, chat: ChatFn) -> dict[str, Any]:
        """Run one task to its verdict, answering the harness's chat requests with `chat`."""
        self._send({"op": "run", "task": task, "attempt": attempt})
        result = self._expect("result", chat)
        if result.get("task") != task or result.get("attempt") != attempt:
            raise SandboxError(f"the sandbox answered for {result.get('task')}#{result.get('attempt')}, not {task}#{attempt}")
        return result
