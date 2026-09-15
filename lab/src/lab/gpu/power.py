"""GPU power-limit guard. The PSU is only trusted with a capped 3090, so GPU jobs refuse to start above the cap.

The cap itself is set on the Proxmox host (the host's power-limit service); this only checks that it is in force.
`python -m lab.gpu.power` is the shell form (exit 1 when the limit is too high), used by the hosted unit.
"""

import os
import sys

import pynvml

DEFAULT_MAX_POWER_W = 250.0


class PowerLimitTooHigh(RuntimeError):
    pass


def max_power_w() -> float:
    return float(os.environ.get("LAB_MAX_POWER_W", DEFAULT_MAX_POWER_W))


def enforced_power_limit_w(gpu_index: int = 0) -> float:
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetEnforcedPowerLimit(pynvml.nvmlDeviceGetHandleByIndex(gpu_index)) / 1000


def check_power_limit(gpu_index: int = 0) -> float:
    """Return the enforced limit in W, or raise if it is above the configured maximum."""
    limit, cap = enforced_power_limit_w(gpu_index), max_power_w()
    if limit > cap + 0.5:
        raise PowerLimitTooHigh(
            f"GPU power limit is {limit:.0f} W, above the {cap:.0f} W maximum (LAB_MAX_POWER_W); "
            "check the host's power-limit service on the Proxmox host"
        )
    return limit


if __name__ == "__main__":
    try:
        print(f"ok: GPU power limit {check_power_limit():.0f} W")
    except (PowerLimitTooHigh, pynvml.NVMLError) as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        sys.exit(1)
