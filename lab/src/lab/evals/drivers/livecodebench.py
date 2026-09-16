"""LiveCodeBench: the 100 newest code-generation problems of the pinned release, pass@1.

The harness runs in the sandbox (sandbox/livecodebench), because grading runs the code the model wrote.
Newer problems are less likely to be in a model's training data; the date window they span is part of
the pin.
"""

from typing import Any

from lab.evals.drivers.base import BoxedDriver

NEWEST = 100


class LiveCodeBench(BoxedDriver):
    key = "livecodebench"
    sandbox = "livecodebench"
    subset_name = f"livecodebench_v6_newest{NEWEST}"

    def select(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(tasks) < NEWEST:
            raise ValueError(f"the release has {len(tasks)} problems, fewer than {NEWEST}")
        return sorted(tasks, key=lambda t: (t["meta"]["contest_date"], t["id"]))[-NEWEST:]

    def describe(self, chosen: list[dict[str, Any]]) -> dict[str, Any]:
        dates = [t["meta"]["contest_date"] for t in chosen]
        return {"newest": NEWEST, "date_window": [min(dates), max(dates)]}
