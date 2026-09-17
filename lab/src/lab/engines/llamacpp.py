"""llama.cpp engine: llama-server / llama-bench argv rendering and build identity.

Every knob is passed explicitly. In particular `--fit off` (default on silently rewrites unset
args), `-np` (default auto) and `--cache-ram` (default 8192 MiB of host RAM) are always set.
"""

import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from lab import catalog, paths
from lab.catalog import ConfigSpec, ModelSpec, Params
from lab.engines.base import SHARED_SAMPLING_FIELDS, Warmup
from lab.schema import EngineBuild

BUILD_FLAG_RE = re.compile(r"^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|GGML_NATIVE|GGML_CUDA\w*):\w+=(.*)$", re.M)


def _runtime_flags(p: Params) -> list[str]:
    """Flags shared by llama-server and llama-bench (same spelling in both)."""
    return [
        "-ngl", str(p.gpu_layers),
        "-fa", p.flash_attn,
        "-ctk", p.cache_type_k,
        "-ctv", p.cache_type_v,
        "-b", str(p.batch),
        "-ub", str(p.ubatch),
        "-t", str(p.threads),
        "-ncmoe", str(p.n_cpu_moe),
        "-lm", p.load_mode,
    ]  # fmt: skip


def bench_depths(cfg: ConfigSpec) -> list[int]:
    b = cfg.bench
    fits = [d for d in b.depths if d + max(b.n_prompt, b.n_gen) <= cfg.params.ctx]
    return sorted(set(fits)) or [0]


#: llama.cpp's own sampler vocabulary, on top of the shared set.
LLAMACPP_SAMPLING_FIELDS = frozenset({
    "tfs_z", "top_n_sigma", "repeat_penalty", "repeat_last_n", "penalize_nl",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "dry_sequence_breakers",
    "xtc_probability", "xtc_threshold", "mirostat", "mirostat_tau", "mirostat_eta",
    "samplers", "min_keep", "n_probs",
})
#: Passed to the template renderer so the count matches what the server will build.
TEMPLATE_FIELDS = ("messages", "tools", "tool_choice", "chat_template_kwargs", "add_generation_prompt")
#: Sampling as the server reports it, so a result can be traced back to how it was produced.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "typical_p", "repeat_penalty", "repeat_last_n",
                 "presence_penalty", "frequency_penalty", "mirostat", "seed", "samplers", "dry_multiplier")


class LlamaCppDialect:
    """llama-server: renders a template on request, counts tokens separately, and serves one slot per request."""

    sampling_fields = SHARED_SAMPLING_FIELDS | LLAMACPP_SAMPLING_FIELDS

    def __init__(self, served_model: str):
        self.served_model = served_model

    def count_prompt(self, client: httpx.Client, upstream: str, body: dict[str, Any]) -> int:
        """Render the template, then tokenize it with special tokens added.

        `/apply-template` leaves out the BOS a model like Gemma 3 needs and the server adds it back when it
        tokenizes, so the count must too. Verified against llama-server b10883 on Qwen3.8 (no BOS) and
        Gemma 3 (BOS): computed count == reported `usage.prompt_tokens`.
        """
        rendered = client.post(f"{upstream}/apply-template", json={k: body[k] for k in TEMPLATE_FIELDS if k in body})
        rendered.raise_for_status()
        tokens = client.post(f"{upstream}/tokenize", json={"content": rendered.json()["prompt"], "add_special": True})
        tokens.raise_for_status()
        return len(tokens.json()["tokens"])

    def cache_scope(self, scope: str) -> dict[str, Any]:
        """Nothing to scope: a slot holds one conversation, and the allowance counts on it reusing the prefix."""
        return {}

    def warmup(self, client: httpx.Client, upstream: str, model_id: str, *, tools: bool) -> Warmup:
        started = time.monotonic()
        client.post(
            f"{upstream}/v1/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 32,
                  **LlamaCpp().request_extras()},
        ).raise_for_status()
        return Warmup(seconds=round(time.monotonic() - started, 2), replies=1)

    def props(self, client: httpx.Client, upstream: str, warmup: Warmup | None = None) -> dict[str, Any]:
        try:
            d = client.get(f"{upstream}/props", timeout=30).json()
        except httpx.HTTPError as e:
            return {"error": str(e)}
        params = d.get("default_generation_settings", {}).get("params", {})
        return {
            "sampling": {k: params[k] for k in SAMPLING_KEYS if k in params},
            "sampling_source": "llama-server /props",
            "n_ctx": d.get("default_generation_settings", {}).get("n_ctx") or d.get("n_ctx"),
            "total_slots": d.get("total_slots"),
            "build_info": d.get("build_info"),
            "chat_template_caps": d.get("chat_template_caps"),
        }

    def eval_launch(self, cfg: ConfigSpec, parallel: int) -> tuple[ConfigSpec, dict[str, Any]]:
        """`-np N` serves N requests at once out of one KV cache, so each slot needs the config's full context."""
        if parallel <= 1:
            return cfg, {}
        params = cfg.params.model_copy(deep=True)
        params.parallel = parallel
        params.ctx = cfg.params.ctx * parallel
        return cfg.model_copy(update={"params": params}), {"parallel": parallel, "ctx": params.ctx}


class LlamaCpp:
    name = "llama.cpp"
    boot_timeout_s = 300

    def __init__(self, root: Path | None = None):
        self.root = root or paths.LLAMA_CPP
        self.bin = self.root / "build" / "bin"

    # -- server ----------------------------------------------------------------
    def _server_argv(self, model: ModelSpec, cfg: ConfigSpec, *, host: str, port: int, binary: str, weights: str, draft: str | None) -> list[str]:
        p = cfg.params
        argv = [
            binary, "-m", weights,
            "--host", host, "--port", str(port),
            "-a", f"{model.slug}/{cfg.slug}",
            "--fit", "off",
            "-c", str(p.ctx),
            "-np", str(p.parallel),
            "--cache-ram", str(p.cache_ram_mib),
            *_runtime_flags(p),
            "--metrics",
        ]  # fmt: skip
        if p.spec:
            s = p.spec
            argv += ["--spec-type", s.type, "-md", draft or "", "-ngld", str(s.gpu_layers), "--spec-draft-n-max", str(s.n_max)]
            if s.cache_type_k:
                argv += ["-ctkd", s.cache_type_k]
            if s.cache_type_v:
                argv += ["-ctvd", s.cache_type_v]
        if p.reasoning_format:
            argv += ["--reasoning-format", p.reasoning_format]
        if p.reasoning_budget is not None:
            argv += ["--reasoning-budget", str(p.reasoning_budget)]
        return argv + p.extra_args

    def server_argv(self, model: ModelSpec, cfg: ConfigSpec, *, host: str, port: int) -> list[str]:
        weights = catalog.resolve_model_path(model)
        draft = None
        if cfg.params.spec:
            draft = str(catalog.resolve_model_path(catalog.load_model(cfg.params.spec.model)))
        return self._server_argv(model, cfg, host=host, port=port, binary=str(self.bin / "llama-server"), weights=str(weights), draft=draft)

    def display_command(self, model: ModelSpec, cfg: ConfigSpec) -> str:
        def shown(m: ModelSpec) -> str:
            return m.source.file or (Path(m.source.path).name if m.source.path else m.slug)

        draft = shown(catalog.load_model(cfg.params.spec.model)) if cfg.params.spec else None
        return shlex.join(
            self._server_argv(model, cfg, host="127.0.0.1", port=8080, binary="llama-server", weights=shown(model), draft=draft)
        )

    def render_files(self, model: ModelSpec, cfg: ConfigSpec) -> dict[str, str]:
        return {}

    def request_extras(self) -> dict:
        return {"cache_prompt": False}

    server_log_name = "llama-server.log"

    def dialect(self, model: ModelSpec, cfg: ConfigSpec) -> LlamaCppDialect:
        return LlamaCppDialect(served_model=f"{model.slug}/{cfg.slug}")

    # -- bench -----------------------------------------------------------------
    def bench_argv(self, weights: Path, cfg: ConfigSpec) -> list[str]:
        b = cfg.bench
        return [
            str(self.bin / "llama-bench"), "-m", str(weights),
            "-o", "jsonl",
            "-r", str(b.reps),
            "--delay", str(b.delay_s),
            *_runtime_flags(cfg.params),
            "-p", str(b.n_prompt),
            "-n", str(b.n_gen),
            "-d", ",".join(str(d) for d in bench_depths(cfg)),
        ]  # fmt: skip

    # -- identity --------------------------------------------------------------
    @staticmethod
    def parse_version(text: str) -> str | None:
        """Build number from `--version`: "version: 0.4.0-dev (build 10883, commit 91f6a6cf3)", or the older "version: 5930 (a1b2c3d)"."""
        m = re.search(r"\(build (\d+)", text) or re.search(r"version: (\d+) \(", text)
        return m.group(1) if m else None

    def build_info(self) -> EngineBuild:
        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True).stdout.strip()

        out = subprocess.run([str(self.bin / "llama-server"), "--version"], capture_output=True, text=True)
        version = self.parse_version(out.stdout + out.stderr)
        cache = self.root / "build" / "CMakeCache.txt"
        flags = ";".join(sorted(f"{k}={v}" for k, v in BUILD_FLAG_RE.findall(cache.read_text()))) if cache.is_file() else None
        extra: dict = {"dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
        try:
            nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True, text=True).stdout
            if m := re.search(r"release ([\d.]+)", nvcc):
                extra["cuda_toolkit"] = m.group(1)
        except FileNotFoundError:
            pass
        return EngineBuild(engine=self.name, version=version, commit_sha=git("rev-parse", "--short=9", "HEAD") or None,
                           build_flags=flags, extra=extra)
