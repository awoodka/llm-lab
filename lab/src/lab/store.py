"""Local run store: runs/<timestamp>_<config>_<kind>/{run.json, raw/, logs/}.

run.json is a RunBundle — the exact payload `lab publish` sends.
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from lab import paths
from lab.schema import RunBundle


@dataclass
class RunDir:
    path: Path

    @property
    def bundle_path(self) -> Path:
        return self.path / "run.json"

    @property
    def raw(self) -> Path:
        return self.path / "raw"

    @property
    def logs(self) -> Path:
        return self.path / "logs"

    def save(self, bundle: RunBundle) -> None:
        tmp = self.bundle_path.with_suffix(".tmp")
        tmp.write_text(bundle.model_dump_json(indent=2))
        tmp.replace(self.bundle_path)

    def load(self) -> RunBundle:
        return RunBundle.model_validate_json(self.bundle_path.read_text())


def new_run_dir(config_slug: str, kind: str) -> tuple[str, RunDir]:
    run_id = str(uuid.uuid4())
    name = f"{time.strftime('%Y%m%dT%H%M%S')}_{config_slug}_{kind}"
    rd = RunDir(paths.RUNS / name)
    rd.raw.mkdir(parents=True)
    rd.logs.mkdir()
    return run_id, rd


def list_runs() -> list[RunDir]:
    if not paths.RUNS.is_dir():
        return []
    return [RunDir(p) for p in sorted(paths.RUNS.iterdir()) if (p / "run.json").is_file()]


def find_run(ref: str) -> RunDir:
    """Match by run UUID (or prefix) or by directory name (or prefix)."""
    matches = []
    for rd in list_runs():
        if rd.path.name.startswith(ref):
            matches.append(rd)
            continue
        if rd.load().run.id.startswith(ref):
            matches.append(rd)
    if not matches:
        raise KeyError(f"no run matches {ref!r}")
    if len(matches) > 1:
        raise KeyError(f"{ref!r} is ambiguous: {', '.join(m.path.name for m in matches)}")
    return matches[0]
