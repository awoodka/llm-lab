"""Hardware/software snapshot recorded with every run, so results stay comparable over time."""

import os
import platform
import re
from pathlib import Path

import pynvml

from lab.schema import HardwareSnapshot


def _os_name() -> str:
    try:
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)', Path("/etc/os-release").read_text(), re.M)
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def _cpu_model() -> str:
    m = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.M)
    return m.group(1).strip() if m else platform.processor()


def _ram_mb() -> int:
    m = re.search(r"^MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
    return int(m.group(1)) // 1024 if m else 0


def hardware_snapshot(gpu_index: int = 0) -> HardwareSnapshot:
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    cuda = pynvml.nvmlSystemGetCudaDriverVersion()
    return HardwareSnapshot(
        gpu_name=pynvml.nvmlDeviceGetName(h),
        gpu_vram_mb=pynvml.nvmlDeviceGetMemoryInfo(h).total // 2**20,
        driver=pynvml.nvmlSystemGetDriverVersion(),
        cuda=f"{cuda // 1000}.{cuda % 1000 // 10}",
        power_limit_w=pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
        pcie=f"Gen{pynvml.nvmlDeviceGetMaxPcieLinkGeneration(h)} x{pynvml.nvmlDeviceGetMaxPcieLinkWidth(h)}",
        cpu_model=_cpu_model(),
        cpu_threads_visible=len(os.sched_getaffinity(0)),
        ram_mb=_ram_mb(),
        kernel=platform.release(),
        os=_os_name(),
    )
