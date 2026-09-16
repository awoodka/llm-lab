"""AIME 2025: 30 competition problems, four attempts each, integer exact match.

Four attempts because one sample of a hard problem is mostly luck; the score is the mean over attempts
per problem, and the standard error is taken over the 30 problems. Answers are integers 0-999, so
grading needs no model and no judge: read the boxed number.
"""

import json
import re
from pathlib import Path

from lab.evals.drivers.base import Outcome, Subset, Task

REPO = "opencompass/AIME2025"
REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
FILES = ("aime2025-I.jsonl", "aime2025-II.jsonl")

INSTRUCTION = (
    "Solve the problem. The answer is an integer between 0 and 999. "
    "End your reply with the final answer on its own line as \\boxed{ANSWER}."
)

BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
INT_RE = re.compile(r"-?\d+")


def _download(file: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=REPO, filename=file, repo_type="dataset", revision=REVISION))


def extract_answer(text: str) -> str | None:
    """The last boxed value, or the last integer if the model never boxed one."""
    if boxed := BOXED_RE.findall(text):
        digits = INT_RE.findall(boxed[-1].replace(",", ""))
        if digits:
            return digits[-1]
    if numbers := INT_RE.findall(text.replace(",", "")):
        return numbers[-1]
    return None


class Aime2025:
    key = "aime_2025"

    def subset(self) -> Subset:
        tasks: list[Task] = []
        for file in FILES:
            part = file.removeprefix("aime2025-").removesuffix(".jsonl")  # "I" or "II"
            rows = [json.loads(line) for line in _download(file).read_text().splitlines() if line.strip()]
            for i, row in enumerate(rows, start=1):
                tasks.append(
                    Task(
                        id=f"aime_2025/{part}-{i}",
                        messages=[{"role": "user", "content": f"{row['question'].strip()}\n\n{INSTRUCTION}"}],
                        answer=str(row["answer"]).strip(),
                        meta={"part": part},
                    )
                )
        return Subset(name="aime_2025_full", tasks=tasks, source={"repo": REPO, "revision": REVISION, "files": list(FILES)})

    def grade(self, task: Task, answer: str) -> Outcome:
        extracted = extract_answer(answer)
        try:
            passed = extracted is not None and int(extracted) == int(task.answer)
        except ValueError:
            passed = False
        return Outcome(passed=passed, extracted=extracted, detail={"expected": task.answer})
