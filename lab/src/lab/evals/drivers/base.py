"""What a benchmark driver has to provide: a pinned set of tasks, and a verdict on one answer.

A driver never talks to the model. The runner sends every request through the allowance proxy, so one
place decides how long an answer may be and one log records what it cost.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from lab import paths


@dataclass(frozen=True)
class Task:
    """One item of work, already written as the messages that will be sent."""

    id: str
    messages: list[dict[str, Any]]
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)
    #: Extra request fields (tools, response_format). Sampling is never set here.
    request: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    passed: bool
    extracted: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Subset:
    """A pinned list of tasks, identified by name and the hash of its task ids."""

    name: str
    tasks: list[Task]
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        digest = hashlib.sha256("\n".join(t.id for t in self.tasks).encode()).hexdigest()
        return f"{self.name}:{digest[:12]}"


class Driver(Protocol):
    key: str

    def subset(self) -> Subset:
        """The pinned tasks, in a stable order."""
        ...

    def grade(self, task: Task, answer: str) -> Outcome:
        """Score one answer. `answer` is the assistant text; an unfinished answer arrives truncated."""
        ...


def subset_file(key: str) -> Path:
    """Committed pin: which task ids belong to the subset, and where they came from."""
    return paths.SUBSETS / f"{key}.json"


def load_pin(key: str) -> dict[str, Any] | None:
    p = subset_file(key)
    return json.loads(p.read_text()) if p.is_file() else None


def check_against_pin(key: str, subset: Subset) -> None:
    """Refuse to run a subset that has drifted from the committed one, so results stay comparable."""
    pin = load_pin(key)
    if pin is None:
        raise FileNotFoundError(f"no pinned subset for {key}; expected {subset_file(key)} (write it with `lab eval-pin {key}`)")
    if pin["task_ids"] != [t.id for t in subset.tasks]:
        raise ValueError(
            f"{key}: the dataset no longer yields the pinned task list "
            f"({len(pin['task_ids'])} pinned, {len(subset.tasks)} now); results would not be comparable"
        )


def write_pin(key: str, subset: Subset) -> Path:
    p = subset_file(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"name": subset.name, "source": subset.source, "task_ids": [t.id for t in subset.tasks]}, indent=2) + "\n")
    return p


class BoxedDriver:
    """A benchmark whose harness runs in a sandbox on `web` (see lab.evals.sandbox).

    The lab picks the tasks and carries the conversation. Building the prompts, running the model's code
    and grading all happen in the box, with the harness's own code.
    """

    sandboxed = True
    key: str
    #: Directory under lab/sandbox/ that builds the harness image.
    sandbox: str
    subset_name: str

    def box_args(self) -> list[str]:
        return []

    def select(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Which of the harness's tasks form the pinned subset, in order. All of them by default."""
        return tasks

    def describe(self, chosen: list[dict[str, Any]]) -> dict[str, Any]:
        """Anything about the chosen tasks that belongs in the pin, such as the date window they span."""
        return {}

    def image(self):
        from lab.evals.sandbox import ensure_image

        return ensure_image(self.sandbox)

    def subset(self) -> Subset:
        from lab.evals.sandbox import Box

        with Box(self.image(), self.box_args()) as box:
            listing = box.list_tasks()
        chosen = self.select(listing["tasks"])
        return Subset(
            name=self.subset_name,
            tasks=[Task(id=t["id"], messages=[], answer="", meta=t.get("meta") or {}) for t in chosen],
            source={**(listing.get("source") or {}), **self.describe(chosen)},
        )
