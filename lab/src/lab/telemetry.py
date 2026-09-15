"""Background GPU/RAM sampler wrapped around every benchmark phase.

GPU power only: RAPL is unreadable in an unprivileged LXC. Energy comes from NVML's
cumulative counter (exact), not from integrating power samples.
"""

import os
import threading
import time
from dataclasses import dataclass, field

import pynvml

# Clock-event bits that mean real slowdown. SW power cap (0x04) is expected under the host's power cap
# (the host's power-limit service on pve) and is tracked separately rather than flagging the run.
THROTTLE_BITS = 0x08 | 0x20 | 0x40 | 0x80  # HW slowdown, SW thermal, HW thermal, HW power brake
POWER_CAP_BIT = 0x04

CGROUP = "/sys/fs/cgroup"


def vram_used_mb(h) -> float:
    """VRAM used by processes. NVML's v1 struct counts ~450 MiB of driver-reserved memory as used;
    v2 reports it separately (matches nvidia-smi)."""
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(h, version=pynvml.nvmlMemory_v2).used / 2**20
    except (pynvml.NVMLError, AttributeError, TypeError):
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20


def _clock_reasons(h) -> int:
    fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
    return fn(h)


def _gfx_clock_mhz(h) -> float | None:
    """Current graphics clock, to confirm the host's clock ceiling (nvidia-smi -lgc) held during a run."""
    try:
        return float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS))
    except pynvml.NVMLError:
        return None


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _proc_rss_kb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


@dataclass
class Phase:
    name: str
    t0: float
    e0_mj: int
    t1: float | None = None
    e1_mj: int | None = None

    @property
    def energy_j(self) -> float | None:
        return None if self.e1_mj is None else (self.e1_mj - self.e0_mj) / 1000

    @property
    def duration_s(self) -> float | None:
        return None if self.t1 is None else self.t1 - self.t0


@dataclass
class Telemetry:
    interval_s: float = 0.1
    gpu_index: int = 0
    watch_pid: int | None = None
    samples: list[dict[str, float]] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)

    def __post_init__(self):
        pynvml.nvmlInit()
        self._h = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_fd: int | None = None
        self.vram_baseline_mb = self._vram_mb()
        self.ram_baseline_mb = (_read_int(f"{CGROUP}/memory.current") or 0) / 2**20

    def _vram_mb(self) -> float:
        return vram_used_mb(self._h)

    def _energy_mj(self) -> int:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._h)

    def gpu_temp_c(self) -> int:
        return pynvml.nvmlDeviceGetTemperature(self._h, pynvml.NVML_TEMPERATURE_GPU)

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> "Telemetry":
        # Per-fd cgroup peak reset (kernel >= 6.12): reads from this fd report peak since the write.
        try:
            self._peak_fd = os.open(f"{CGROUP}/memory.peak", os.O_RDWR)
            os.write(self._peak_fd, b"reset")
        except OSError:
            self._peak_fd = None
        self._t_start = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.end_phase()
        self._stop.set()
        if self._thread:
            self._thread.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def elapsed(self) -> float:
        """Seconds since start(), on the same clock as sample["t"]."""
        return time.monotonic() - self._t_start

    def phase(self, name: str) -> None:
        self.end_phase()
        self.phases.append(Phase(name, time.monotonic(), self._energy_mj()))

    def end_phase(self) -> None:
        if self.phases and self.phases[-1].t1 is None:
            self.phases[-1].t1 = time.monotonic()
            self.phases[-1].e1_mj = self._energy_mj()

    def _run(self) -> None:
        while not self._stop.is_set():
            h = self._h
            s = {
                "t": time.monotonic() - self._t_start,
                "vram_mb": self._vram_mb(),
                "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000,
                "temp_c": float(self.gpu_temp_c()),
                "util": float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu),
                "reasons": float(_clock_reasons(h)),
                "ram_mb": (_read_int(f"{CGROUP}/memory.current") or 0) / 2**20,
            }
            if (clock := _gfx_clock_mhz(h)) is not None:
                s["clock_mhz"] = clock
            if self.watch_pid and (rss := _proc_rss_kb(self.watch_pid)) is not None:
                s["proc_rss_mb"] = rss / 1024
            self.samples.append(s)
            self._stop.wait(self.interval_s)

    # -- results ---------------------------------------------------------------
    def phase_samples(self, name: str) -> list[dict[str, float]]:
        ph = next((p for p in self.phases if p.name == name), None)
        if not ph:
            return []
        t0, t1 = ph.t0 - self._t_start, (ph.t1 or time.monotonic()) - self._t_start
        return [s for s in self.samples if t0 <= s["t"] <= t1]

    def summary(self) -> dict:
        reasons = [int(s["reasons"]) for s in self.samples]
        ram_peak = None
        if self._peak_fd is not None:
            os.lseek(self._peak_fd, 0, 0)
            ram_peak = int(os.read(self._peak_fd, 64).split()[0]) / 2**20
            os.close(self._peak_fd)
            self._peak_fd = None
        sampled_ram_peak = max((s["ram_mb"] for s in self.samples), default=0.0)
        return {
            "vram_baseline_mb": self.vram_baseline_mb,
            "vram_peak_mb": max((s["vram_mb"] for s in self.samples), default=0.0),
            "ram_baseline_mb": self.ram_baseline_mb,
            "ram_peak_mb": max(ram_peak or 0.0, sampled_ram_peak),
            "proc_rss_peak_mb": max((s.get("proc_rss_mb", 0.0) for s in self.samples), default=0.0),
            "temp_max_c": max((s["temp_c"] for s in self.samples), default=0.0),
            "power_max_w": max((s["power_w"] for s in self.samples), default=0.0),
            "clock_max_mhz": max((s["clock_mhz"] for s in self.samples if "clock_mhz" in s), default=None),
            "throttled": any(r & THROTTLE_BITS for r in reasons),
            "power_capped_frac": (sum(1 for r in reasons if r & POWER_CAP_BIT) / len(reasons)) if reasons else 0.0,
            "phases": {
                p.name: {
                    "duration_s": p.duration_s,
                    "energy_j": p.energy_j,
                    "power_avg_w": (p.energy_j / p.duration_s) if p.energy_j and p.duration_s else None,
                }
                for p in self.phases
            },
        }

    def downsampled(self, every_s: float = 1.0) -> list[dict[str, float]]:
        out, next_t = [], 0.0
        for s in self.samples:
            if s["t"] >= next_t:
                out.append({k: round(v, 2) for k, v in s.items()})
                next_t = s["t"] + every_s
        return out
