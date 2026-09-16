from lab.evals.drivers.base import Driver


def get_driver(key: str) -> Driver:
    if key == "aime_2025":
        from lab.evals.drivers.aime import Aime2025

        return Aime2025()
    if key == "bfcl":
        from lab.evals.drivers.bfcl import Bfcl

        return Bfcl()
    if key == "livecodebench":
        from lab.evals.drivers.livecodebench import LiveCodeBench

        return LiveCodeBench()
    raise NotImplementedError(f"benchmark {key!r} has no driver yet")
