from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from lab.catalog import ConfigSpec, ModelSpec
from lab.schema import EngineBuild


#: Sampling every engine understands, dropped from a harness's request so the launch flags decide.
#: Constrained decoding lives here too: it is an engine capability, and letting one engine's harness
#: use it would put that row's numbers on a different footing from every other row's.
SHARED_SAMPLING_FIELDS = frozenset({
    "temperature", "top_p", "top_k", "min_p", "typical_p", "typ_p", "seed",
    "presence_penalty", "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
    "response_format", "structured_outputs",
    "guided_json", "guided_regex", "guided_choice", "guided_grammar", "guided_decoding_backend",
})


@dataclass(frozen=True)
class Warmup:
    """What a post-boot warmup saw, before anything is graded."""

    seconds: float
    replies: int
    #: Whether a tool request came back as a real tool call. None when tools weren't probed.
    tool_calls: bool | None = None


class ServerDialect(Protocol):
    """How one engine's OpenAI-compatible server is talked to during an eval.

    The runner and the allowance proxy hold one of these instead of branching on the engine's name. A
    dialect is built from a config alone — no checkout, no catalog, no GPU — so tests can use a real one
    against a stand-in server.
    """

    #: Request fields the proxy strips, because the config's launch flags decide how the model samples.
    sampling_fields: frozenset[str]

    def count_prompt(self, client: httpx.Client, upstream: str, body: dict[str, Any]) -> int:
        """Count the prompt the way this server will build it, including tools and template kwargs."""
        ...

    def cache_scope(self, scope: str) -> dict[str, Any]:
        """Request fields that keep one task's prompt cache to itself. {} when the server can't share one."""
        ...

    def warmup(self, client: httpx.Client, upstream: str, model_id: str, *, tools: bool) -> Warmup:
        """Spend the first request's one-off costs before the clock matters, and prove tool calls work."""
        ...

    def props(self, client: httpx.Client, upstream: str, warmup: Warmup | None = None) -> dict[str, Any]:
        """How the live server is set up, for the run's record. Never a local path: runs publish file names."""
        ...

    def eval_launch(self, cfg: ConfigSpec, parallel: int) -> tuple[ConfigSpec, dict[str, Any]]:
        """The config to serve `parallel` requests at once, and what that overrode."""
        ...


class Engine(Protocol):
    name: str
    #: How long a cold boot may take before the server counts as failed (vLLM compiles kernels on first boot).
    boot_timeout_s: float

    def server_argv(self, model: ModelSpec, cfg: ConfigSpec, *, host: str, port: int) -> list[str]:
        """Exact argv to serve this config (weights resolved to a local path)."""
        ...

    def display_command(self, model: ModelSpec, cfg: ConfigSpec) -> str:
        """Human-readable launch command for the showcase (weights shown by file name)."""
        ...

    def render_files(self, model: ModelSpec, cfg: ConfigSpec) -> dict[str, str]:
        """Extra config files the engine needs (e.g. TabbyAPI config.yml). {} if none."""
        ...

    def build_info(self) -> EngineBuild: ...

    def request_extras(self) -> dict:
        """Fields every benchmark request adds, so no request is served from another's prompt cache."""
        ...

    #: Where a runner writes this server's stdout and stderr.
    server_log_name: str

    def dialect(self, model: ModelSpec, cfg: ConfigSpec) -> ServerDialect:
        """How to talk to this config's server while evaluating it."""
        ...


def get_engine(name: str, cfg: ConfigSpec | None = None) -> Engine:
    """The engine a model runs on. For vLLM, pass the config: it launches from the checkout its pinned
    commit resolves to (lab.engines.vllm.checkout_for)."""
    if name == "llama.cpp":
        from lab.engines.llamacpp import LlamaCpp

        return LlamaCpp()
    if name == "vllm":
        from lab.engines.vllm import Vllm, checkout_for

        commit = getattr(cfg.params, "launcher_commit", None) if cfg is not None else None
        return Vllm(root=checkout_for(commit) if commit else None)
    raise NotImplementedError(f"engine {name!r} is not implemented yet")


def weights_display_name(path: Path) -> str:
    return path.name
