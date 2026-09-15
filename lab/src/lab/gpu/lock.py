"""Exclusive GPU lock (flock). Compatible with `flock -n gpu.lock` used by lab-hosted.service.

The kernel releases the lock if the holder dies, so a crashed bench never wedges the GPU.
"""

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from lab import paths


class LockBusy(RuntimeError):
    pass


def read_holder() -> dict | None:
    try:
        return json.loads(paths.GPU_LOCK_HOLDER.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def is_locked() -> bool:
    paths.STATE.mkdir(parents=True, exist_ok=True)
    fd = os.open(paths.GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


@contextmanager
def gpu_lock(purpose: str, wait: bool = False) -> Iterator[None]:
    paths.STATE.mkdir(parents=True, exist_ok=True)
    fd = os.open(paths.GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise LockBusy(f"GPU is locked by {read_holder() or 'unknown holder'} (use --wait to queue)")
        paths.GPU_LOCK_HOLDER.write_text(
            json.dumps({"pid": os.getpid(), "purpose": purpose, "since": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        )
        try:
            yield
        finally:
            paths.GPU_LOCK_HOLDER.unlink(missing_ok=True)
    finally:
        os.close(fd)
