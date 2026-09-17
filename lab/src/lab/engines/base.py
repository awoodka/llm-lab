from pathlib import Path
from typing import Protocol

from lab.catalog import ConfigSpec, ModelSpec
from lab.schema import EngineBuild


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


def get_engine(name: str) -> Engine:
    if name == "llama.cpp":
        from lab.engines.llamacpp import LlamaCpp

        return LlamaCpp()
    if name == "vllm":
        from lab.engines.vllm import Vllm

        return Vllm()
    raise NotImplementedError(f"engine {name!r} is not implemented yet")


def weights_display_name(path: Path) -> str:
    return path.name
