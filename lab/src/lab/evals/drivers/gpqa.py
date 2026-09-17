"""GPQA Diamond: 198 graduate-level multiple-choice questions, zero-shot chain of thought, accuracy.

The prompt is the simple-evals multiple-choice template. The four options are shuffled once per question,
seeded by its record id, so every run and every model sees the same order. The letter is read from the
answer (`content`), never from a model's thinking; the last "Answer: X" wins.

The dataset is gated on Hugging Face: accept its terms and give the lab a token (HF_TOKEN, or
`uv run hf auth login` in lab/). Its revision is resolved once, by `lab eval-pin gpqa_diamond`, and every later
run reads the questions at the revision in the pin. Questions never leave the machine: runs publish ids
and verdicts only, as the dataset's authors ask.
"""

import csv
import hashlib
import random
import re
from pathlib import Path

from lab.evals.drivers.base import Outcome, Subset, Task, load_pin

REPO = "Idavidrein/gpqa"
FILE = "gpqa_diamond.csv"
LETTERS = "ABCD"

TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)
ANSWER_RE = re.compile(r"(?i)\banswer\b\W{0,5}(?:is\W{0,3})?\(?([ABCD])\)?(?![A-Za-z])")
COLUMNS = ("Record ID", "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")


def _revision() -> str:
    """The pinned revision, or the dataset's current one when there is no pin yet."""
    pin = load_pin("gpqa_diamond")
    if pin:
        return pin["source"]["revision"]
    from huggingface_hub import HfApi

    return HfApi().dataset_info(REPO).sha


def _download(revision: str) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    try:
        return Path(hf_hub_download(repo_id=REPO, filename=FILE, repo_type="dataset", revision=revision))
    except GatedRepoError as e:
        raise PermissionError(
            f"{REPO} is gated: accept its terms on huggingface.co with your account, then set HF_TOKEN "
            "or run `uv run hf auth login` in lab/"
        ) from e


def shuffled(record_id: str, correct: str, wrong: list[str]) -> tuple[list[str], str]:
    """The options in this question's fixed order, and the letter of the correct one."""
    options = [correct, *wrong]
    random.Random(int(hashlib.sha256(record_id.encode()).hexdigest()[:16], 16)).shuffle(options)
    return options, LETTERS[options.index(correct)]


def extract_letter(text: str) -> str | None:
    found = ANSWER_RE.findall(text)
    return found[-1].upper() if found else None


class GpqaDiamond:
    key = "gpqa_diamond"

    def subset(self) -> Subset:
        revision = _revision()
        with open(_download(revision), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if rows and (missing := [c for c in COLUMNS if c not in rows[0]]):
            raise ValueError(f"{FILE} has no {missing} column; the dataset changed shape")
        tasks = []
        for row in sorted(rows, key=lambda r: r["Record ID"]):
            options, letter = shuffled(
                row["Record ID"], row["Correct Answer"].strip(), [row[f"Incorrect Answer {i}"].strip() for i in (1, 2, 3)]
            )
            prompt = TEMPLATE.format(question=row["Question"].strip(), **dict(zip(LETTERS, options, strict=True)))
            tasks.append(Task(id=f"gpqa_diamond/{row['Record ID']}", messages=[{"role": "user", "content": prompt}], answer=letter))
        return Subset(name="gpqa_diamond_full", tasks=tasks, source={"repo": REPO, "revision": revision, "files": [FILE]})

    def grade(self, task: Task, answer: str) -> Outcome:
        letter = extract_letter(answer)
        return Outcome(passed=letter == task.answer, extracted=letter, detail={"expected": task.answer})
