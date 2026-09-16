"""BFCL (bfcl-eval 2026.3.23), run by its own handler and graded by its own checkers.

Five categories: simple_python, multiple, parallel, parallel_multiple and multi_turn_base. The lab picks
the subset, and the mode: "FC" (native tool calls) when the model's chat template supports tools,
otherwise "prompting", where BFCL puts the functions in the system prompt and parses Python-style calls
from the reply. Each task goes through BFCL's OpenAI chat-completions handler, whose client is the lab's
channel. Multi-turn calls execute in BFCL's simulated environments, which `eval` what the model asked
for, which is why this runs in the sandbox. The per-entry evaluation copies eval_runner.py's
`_evaluate_single_ast_entry` and `_evaluate_single_multi_turn_entry`.

Two parts of the package are replaced by stubs, because these categories never reach them and importing
them pulls in heavy dependencies:

- the model registry, which imports every vendor SDK. The stub holds one entry per mode, flagged like
  BFCL's real entries: FC maps its underscored tool names back to dotted ones, prompting keeps them.
- the Java and JavaScript parsers (tree-sitter).
"""

import copy
import json
import os
import sys
import types
from types import SimpleNamespace

from boxproto import Channel, ChannelOpenAI, ModelUnavailable

# BFCL decides FC vs prompting partly by whether "FC" is in the registry name.
REGISTRY = {"FC": "llama-server-FC", "prompting": "llama-server-prompt"}
MODEL = "lab"
CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "multi_turn_base")
VERSION = "2026.3.23"


def _stub(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


def _not_here(*args, **kwargs):
    raise NotImplementedError("Java and JavaScript categories aren't run in this sandbox")


_stub("bfcl_eval.constants.model_config",
      MODEL_CONFIG_MAPPING={
          REGISTRY["FC"]: SimpleNamespace(underscore_to_dot=True, is_fc_model=True),
          REGISTRY["prompting"]: SimpleNamespace(underscore_to_dot=False, is_fc_model=False),
      })
_stub("bfcl_eval.model_handler.parser.java_parser", parse_java_function_call=_not_here)
_stub("bfcl_eval.model_handler.parser.js_parser", parse_javascript_function_call=_not_here)
os.makedirs(os.environ["BFCL_PROJECT_ROOT"], exist_ok=True)

from bfcl_eval.constants.enums import Language, ReturnFormat  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # noqa: E402
from bfcl_eval.utils import is_function_calling_format_output, load_dataset_entry, load_ground_truth_entry  # noqa: E402

channel = Channel()

# What the model is asked (with BFCL's language hint, as generation loads it), what grading reads
# (without it, as evaluation loads it) and the ground truth, by entry id.
PROMPTS: dict[str, dict] = {}
EVAL_PROMPTS: dict[str, dict] = {}
TRUTH: dict[str, dict] = {}
for category in CATEGORIES:
    PROMPTS.update({e["id"]: e for e in load_dataset_entry(category)})
    EVAL_PROMPTS.update({e["id"]: e for e in load_dataset_entry(category, include_language_specific_hint=False)})
    TRUTH.update({e["id"]: e for e in load_ground_truth_entry(category)})


def category_of(entry_id: str) -> str:
    return entry_id.rsplit("_", 1)[0]


def list_tasks() -> dict:
    tasks = [{"id": entry_id, "meta": {"category": category_of(entry_id)}} for entry_id in PROMPTS]
    return {"tasks": tasks, "source": {"harness": "bfcl-eval", "harness_version": VERSION, "categories": list(CATEGORIES)}}


def forget_instances() -> None:
    """BFCL keeps each entry's simulated environments in module globals; a new attempt starts clean."""
    names = vars(multi_turn_utils)
    for name in [n for n in names if n.endswith("_instance")]:
        del names[name]


def new_handler(client: ChannelOpenAI, mode: str) -> OpenAICompletionsHandler:
    if mode not in REGISTRY:
        raise ValueError(f"unknown BFCL mode {mode!r}; FC or prompting")
    # BFCL's CLI default temperature. The lab strips sampling, so the config's own settings decide it.
    handler = OpenAICompletionsHandler(model_name=MODEL, temperature=0.001, registry_name=REGISTRY[mode], is_fc_model=mode == "FC")
    handler.client = client
    return handler


def grade_ast(handler, entry_id: str, result) -> dict:
    category = category_of(entry_id)
    prompt, truth = EVAL_PROMPTS[entry_id], TRUTH[entry_id]["ground_truth"]
    try:
        decoded = handler.decode_ast(result, ReturnFormat.PYTHON, False)
    except Exception as e:  # noqa: BLE001 - the official evaluation treats this as the model's failure
        return {"valid": False, "error_type": "ast_decoder:decoder_failed", "error": [f"Invalid syntax. Failed to decode AST. {e}"]}
    if not is_function_calling_format_output(decoded):
        return {"valid": False, "error_type": "ast_decoder:decoder_wrong_output_format", "decoded": str(decoded)}
    checked = ast_checker(prompt["function"], decoded, truth, Language.PYTHON, category, handler.registry_name)
    return {**checked, "decoded": decoded}


def grade_multi_turn(handler, entry_id: str, result) -> dict:
    truth = TRUTH[entry_id]["ground_truth"]
    prompt = copy.deepcopy(EVAL_PROMPTS[entry_id])
    prompt.pop("function", None)
    if not isinstance(result, list):
        return {"valid": False, "error_type": "multi_turn:inference_error"}
    if len(result) != len(truth):
        return {"valid": False, "error_type": "multi_turn:force_terminated", "turns": len(result), "expected_turns": len(truth)}
    decoded_turns = []
    for turn in result:
        decoded_steps = []
        for step in turn:
            try:
                decoded = handler.decode_execute(step, has_tool_call_tag=False)
            except Exception:  # noqa: BLE001 - as the official evaluation: a step that isn't a call is skipped
                continue
            if not is_empty_execute_response(decoded):
                decoded_steps.append(decoded)
        decoded_turns.append(decoded_steps)
    checked = multi_turn_checker(decoded_turns, truth, prompt, category_of(entry_id), handler.registry_name)
    return {**checked, "decoded": decoded_turns}


def run_task(entry_id: str, attempt: int, options: dict) -> dict:
    if entry_id not in PROMPTS:
        raise KeyError(f"no BFCL entry {entry_id!r} in {', '.join(CATEGORIES)}")
    forget_instances()
    client = ChannelOpenAI(channel)
    handler = new_handler(client, options.get("mode", "FC"))
    try:
        result, _metadata = handler.inference(copy.deepcopy(PROMPTS[entry_id]), False, False)
    except ModelUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 - BFCL's CLI records this as the model's answer, and so do we
        result = f"Error during inference: {e}"

    multi_turn = category_of(entry_id) == "multi_turn_base"
    verdict = grade_multi_turn(handler, entry_id, result) if multi_turn else grade_ast(handler, entry_id, result)
    forget_instances()

    decoded = verdict.pop("decoded", None)
    passed = bool(verdict.pop("valid", False))
    extracted = json.dumps(decoded, default=str)[:500] if decoded is not None else None
    detail = {"category": category_of(entry_id), "mode": options.get("mode", "FC")}
    if not passed:
        detail["error_type"] = verdict.get("error_type")
        detail["error"] = json.dumps(verdict.get("error") or verdict.get("details") or verdict, default=str)[:600]
    return {"passed": passed, "extracted": extracted, "detail": detail, "transcript": client.transcript()}


channel.serve(list_tasks, run_task)
