"""BFCL: 400 function-calling cases, 80 from each of five categories, graded by BFCL's own checkers.

The harness runs in the sandbox (sandbox/bfcl), because multi-turn cases execute the calls the model
makes. The subset is drawn here, once, with a fixed seed and committed as a pin. It is interleaved by
category, so a `--limit` smoke test touches every category.
"""

import random
from collections import defaultdict
from typing import Any

from lab.evals.drivers.base import BoxedDriver

CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "multi_turn_base")
PER_CATEGORY = 80
SEED = "lab-bfcl-quick-v1"


def _number(task: dict[str, Any]) -> int:
    return int(task["id"].rsplit("_", 1)[1])


class Bfcl(BoxedDriver):
    key = "bfcl"
    sandbox = "bfcl"
    subset_name = "bfcl_quick_400"

    def options(self, props: dict[str, Any]) -> dict[str, Any]:
        """Native tool calls when the chat template can do them; BFCL's prompting mode when it can't."""
        caps = props.get("chat_template_caps") or {}
        native = bool(caps.get("supports_tools") and caps.get("supports_tool_calls"))
        return {"mode": "FC" if native else "prompting"}

    def select(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in tasks:
            by_category[t["meta"]["category"]].append(t)
        rng = random.Random(SEED)
        picked = []
        for category in CATEGORIES:
            pool = sorted(by_category[category], key=_number)
            if len(pool) < PER_CATEGORY:
                raise ValueError(f"BFCL {category} has {len(pool)} cases, fewer than {PER_CATEGORY}")
            picked.append(sorted(rng.sample(pool, PER_CATEGORY), key=_number))
        return [group[i] for i in range(PER_CATEGORY) for group in picked]
