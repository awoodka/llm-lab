from lab.evals.drivers.base import Driver


def get_driver(key: str) -> Driver:
    if key == "aime_2025":
        from lab.evals.drivers.aime import Aime2025

        return Aime2025()
    raise NotImplementedError(f"benchmark {key!r} has no driver yet")
