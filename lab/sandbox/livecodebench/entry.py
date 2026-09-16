"""LiveCodeBench code generation (commit 28fef95, release file test6.jsonl), graded by LCB's own code.

The prompt is LCB's generic chat prompt (LMStyle.OpenAIChat). The code is taken from the reply the way
LCB takes it for that style, and the grader is LCB's `codegen_metrics`, one generation at a time, with
its default 6-second limit per test. Only the answer (`content`) is read, never a model's thinking, as in
LCB's own OpenAI runner. A solution passes when every public and hidden test passes.

Problems are read one at a time: the release file is large, and each problem's hidden tests are decoded
only when it is graded.

Two imports that LCB's packages make are stubbed, because code generation never uses them: `datasets`
(the HF loader; the file is read directly here) and `anthropic` (prompt constants for other scenarios).
"""

import json
import sys
import types
from pathlib import Path

from boxproto import Channel


def _stub(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


def _not_here(*args, **kwargs):
    raise NotImplementedError("the sandbox reads the release file directly")


_stub("datasets", load_dataset=_not_here)
_stub("anthropic", HUMAN_PROMPT="\n\nHuman:", AI_PROMPT="\n\nAssistant:")

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # noqa: E402
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics  # noqa: E402
from lcb_runner.lm_styles import LMStyle  # noqa: E402
from lcb_runner.prompts.code_generation import format_prompt_generation  # noqa: E402
from lcb_runner.utils.extraction_utils import extract_code  # noqa: E402

COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE = Path("/opt/lcb/data/test6.jsonl")
STYLE = LMStyle.OpenAIChat
TIMEOUT_S = 6  # LCB's default, per test
PREFIX = "livecodebench/"

channel = Channel()

# Where each problem starts in the release file, and what the lab needs to choose a subset.
OFFSETS: dict[str, int] = {}
LISTING: list[dict] = []
with open(RELEASE, "rb") as fh:
    while True:
        offset = fh.tell()
        line = fh.readline()
        if not line:
            break
        row = json.loads(line)
        qid = row["question_id"]
        if qid in OFFSETS:
            raise ValueError(f"question {qid} appears twice in {RELEASE.name}")
        OFFSETS[qid] = offset
        LISTING.append({
            "id": PREFIX + qid,
            "meta": {k: row[k] for k in ("contest_date", "platform", "difficulty", "question_title")},
        })


def list_tasks() -> dict:
    return {
        "tasks": LISTING,
        "source": {
            "harness": "livecodebench", "commit": COMMIT, "scenario": "codegeneration", "lm_style": STYLE.value,
            "dataset": "livecodebench/code_generation_lite", "revision": DATASET_REVISION, "file": RELEASE.name,
            "timeout_per_test_s": TIMEOUT_S,
        },
    }


def load(qid: str) -> CodeGenerationProblem:
    with open(RELEASE, "rb") as fh:
        fh.seek(OFFSETS[qid])
        return CodeGenerationProblem(**json.loads(fh.readline()))


def run_task(task_id: str, attempt: int, options: dict) -> dict:
    qid = task_id.removeprefix(PREFIX)
    if qid not in OFFSETS:
        raise KeyError(f"no LiveCodeBench problem {qid!r} in {RELEASE.name}")
    problem = load(qid)
    messages = format_prompt_generation(problem, STYLE)
    reply = channel.chat({"messages": messages})
    choice = (reply.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    code = extract_code(message.get("content") or "", STYLE)

    sample = problem.get_evaluation_sample()
    n_tests = len(json.loads(sample["input_output"])["inputs"])
    metrics, results, metadata = codegen_metrics([sample], [[code]], k_list=[1], num_process_evaluate=1, timeout=TIMEOUT_S)
    outcome = json.loads(metadata[0][0])
    detail = {
        "platform": problem.platform.value,
        "difficulty": problem.difficulty.value,
        "tests": n_tests,
        "tests_run": len(results[0][0]),
    }
    for key in ("error_code", "error_message"):
        if key in outcome:
            detail[key] = outcome[key]
    if "error" in outcome:
        detail["error"] = str(outcome["error"])[:300]
    return {
        "passed": bool(metrics["pass@1"] > 0),
        "extracted": code[:4000],
        "detail": detail,
        "transcript": {"messages": messages + [message], "usage": reply.get("usage"), "finish_reason": choice.get("finish_reason")},
    }


channel.serve(list_tasks, run_task)
