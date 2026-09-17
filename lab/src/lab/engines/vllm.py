"""vLLM engine: the syv-ai single-user launcher (patched vLLM 0.28.0, W4A16 AutoRound, DFlash2 speculative decoding).

The lab never calls `vllm serve` itself. It runs the pinned checkout's launcher with every knob set as an
environment variable, because the launcher derives a dozen coupled flags and env vars from those knobs.
`--served-model-name` goes in EXTRA_ARGS, which the launcher places after its own, so the lab's id wins.

The launcher binds 0.0.0.0 unless HOST is set, and reads `api_key.txt` if it exists; the lab always sets HOST
and refuses a checkout with a key file, since neither the lab nor Open WebUI would send the key.
"""

import hashlib
import json
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from lab import catalog, paths
from lab.catalog import ConfigSpec, ModelSpec, VllmParams
from lab.engines.base import SHARED_SAMPLING_FIELDS, Warmup
from lab.schema import EngineBuild

#: Native in vLLM 0.28.0; the launcher repo keeps it in the series for older installs.
RETIRED_PATCHES = {"dflash2-backport.patch"}
SUPERSEDES_RE = re.compile(r"^Supersedes: (\S+)\s*$", re.M)
VERSIONS_PY = (
    "import importlib.metadata as m, json, sys; "
    "print(json.dumps({'python': sys.version.split()[0], "
    "**{p: m.version(p) for p in ('vllm', 'torch', 'flashinfer-python', 'transformers')}}))"
)


class CheckoutMismatch(RuntimeError):
    pass


def _flag(on: bool) -> str:
    return "1" if on else "0"


#: vLLM's own request vocabulary that steers generation, on top of the shared set.
VLLM_SAMPLING_FIELDS = frozenset({
    "repetition_penalty", "length_penalty", "use_beam_search", "best_of", "n",
    "prompt_logprobs", "logprob_token_ids", "allowed_token_ids", "bad_words", "stop_token_ids",
    "include_stop_str_in_output", "ignore_eos", "skip_special_tokens", "spaces_between_special_tokens",
    "truncate_prompt_tokens", "truncation_side", "thinking_token_budget", "reasoning_effort",
    "repetition_detection", "priority", "vllm_xargs", "echo", "cache_salt",
})
#: What a run records of how the server sampled. vLLM has no /props, so this comes from the model's own config.
GENERATION_KEYS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
#: Fields that change the prompt the chat template renders, so the count must carry them too.
TOKENIZE_FIELDS = ("tools", "chat_template_kwargs", "continue_final_message", "documents")


class VllmDialect:
    """vLLM: one tokenize call renders and counts, and the prompt cache is shared by every request."""

    sampling_fields = SHARED_SAMPLING_FIELDS | VLLM_SAMPLING_FIELDS

    def __init__(self, served_model: str, weights_dir: Path, params: VllmParams, build: EngineBuild | None = None):
        self.served_model = served_model
        self.weights_dir = weights_dir
        self.params = params
        self.build = build

    def count_prompt(self, client: httpx.Client, upstream: str, body: dict[str, Any]) -> int:
        """Ask the server to render and count in one call.

        `/tokenize` applies the same chat template the completion will, so tools and thinking-mode kwargs
        are counted. `add_special_tokens` is left at the chat path's own default (false: the template
        carries any BOS), so both paths agree. Verified against vLLM 0.28.0 on Qwen3.8 W4A16, with and
        without tools: computed count == reported `usage.prompt_tokens`.
        """
        payload: dict[str, Any] = {
            "model": self.served_model,
            "messages": body["messages"],
            "add_generation_prompt": body.get("add_generation_prompt", True),
        }
        payload.update({k: body[k] for k in TOKENIZE_FIELDS if body.get(k) is not None})
        r = client.post(f"{upstream}/tokenize", json=payload)
        r.raise_for_status()
        return int(r.json()["count"])

    def cache_scope(self, scope: str) -> dict[str, Any]:
        """One salt per task attempt: vLLM's prefix cache is global, so without this a task could be
        charged for a prefill another task already paid for, and a retry would inherit its own first try."""
        return {"cache_salt": hashlib.sha256(scope.encode()).hexdigest()[:32]}

    def _chat(self, client: httpx.Client, upstream: str, body: dict[str, Any]) -> dict[str, Any]:
        r = client.post(f"{upstream}/v1/chat/completions", json={"model": self.served_model, **body,
                                                                 "cache_salt": uuid.uuid4().hex})
        r.raise_for_status()
        return r.json()

    def warmup(self, client: httpx.Client, upstream: str, model_id: str, *, tools: bool) -> Warmup:
        """Pay the first request's one-off costs, and check that a tool call comes back as a tool call.

        The wrong --tool-call-parser does not fail: the call arrives as ordinary text, which a tool
        benchmark scores as a model that cannot use tools. Better to find out here than after a tier.
        """
        started = time.monotonic()
        replies = 0
        self._chat(client, upstream, {"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 32})
        replies += 1
        tool_calls = None
        if tools:
            data = self._chat(client, upstream, {
                "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather", "description": "Current weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                }}],
                "tool_choice": "auto", "max_tokens": 256,
            })
            choice = (data.get("choices") or [{}])[0]
            tool_calls = bool((choice.get("message") or {}).get("tool_calls")) or choice.get("finish_reason") == "tool_calls"
            replies += 1
        return Warmup(seconds=round(time.monotonic() - started, 2), replies=replies, tool_calls=tool_calls)

    def _generation_config(self) -> dict[str, Any]:
        """What vLLM will actually sample with: the model's generation config, then the launch's overrides."""
        try:
            gen = json.loads((self.weights_dir / "generation_config.json").read_text())
        except (OSError, json.JSONDecodeError):
            gen = {}
        args = self.params.extra_args
        if "--override-generation-config" in args:
            try:
                gen.update(json.loads(args[args.index("--override-generation-config") + 1]))
            except (IndexError, json.JSONDecodeError):
                pass
        return {k: gen[k] for k in GENERATION_KEYS if k in gen}

    def props(self, client: httpx.Client, upstream: str, warmup: Warmup | None = None) -> dict[str, Any]:
        """The /props llama-server would have given, assembled from what vLLM does expose.

        Only the served id and context length are taken from the model card: its `root` is a local path,
        and a run publishes file names, never paths.
        """
        try:
            card = client.get(f"{upstream}/v1/models", timeout=30).raise_for_status().json()["data"][0]
        except (httpx.HTTPError, IndexError, KeyError) as e:
            return {"error": str(e)}
        tools = bool(self.params.tools)
        return {
            "sampling": self._generation_config(),
            "sampling_source": "generation_config.json plus the launch's overrides (vLLM has no /props)",
            "n_ctx": card.get("max_model_len"),
            "served_model": card.get("id"),
            "total_slots": self.params.max_seqs,
            "build_info": f"vllm {self.build.version}" + (f" ({self.build.commit_sha})" if self.build and self.build.commit_sha else "")
            if self.build else None,
            "chat_template_caps": {
                "supports_tools": tools,
                "supports_tool_calls": bool(warmup.tool_calls) if warmup and warmup.tool_calls is not None else tools,
            },
        }

    def eval_launch(self, cfg: ConfigSpec, parallel: int) -> tuple[ConfigSpec, dict[str, Any]]:
        """Nothing to relaunch: the context is per request and the slots are already there.

        That is the point — a parallel eval measures exactly the config that serves chat.
        """
        seats = cfg.params.max_seqs
        if parallel > seats:
            raise ValueError(f"--parallel {parallel} wants more than this config's {seats} slots (max_seqs); "
                             "serving more would be a different config from the one that serves chat")
        return cfg, {"parallel": parallel} if parallel > 1 else {}


class Vllm:
    name = "vllm"
    boot_timeout_s = 1500

    def __init__(self, root: Path | None = None):
        self.root = root or paths.QWEN_SERVING

    # -- checkout ----------------------------------------------------------------
    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True).stdout.strip()

    def check_checkout(self, p: VllmParams) -> None:
        """Refuse to launch anything but the pinned commit, or a checkout that would demand an API key."""
        head = self._git("rev-parse", "HEAD")
        if head != p.launcher_commit:
            raise CheckoutMismatch(
                f"{self.root} is at {head[:12] or 'no commit'}, but the config pins {p.launcher_commit[:12]}; "
                "an upgraded stack needs a new config slug"
            )
        if (self.root / "api_key.txt").exists():
            raise CheckoutMismatch(f"{self.root / 'api_key.txt'} exists: the launcher would require that key, which the lab never sends")

    # -- server ------------------------------------------------------------------
    def launch_env(self, model: ModelSpec, cfg: ConfigSpec, *, host: str, port: int, weights: str, draft: str | None) -> dict[str, str]:
        p = cfg.params
        assert isinstance(p, VllmParams)
        extra = ["--served-model-name", f"{model.slug}/{cfg.slug}", "--load-format", p.load_format]
        if not p.prefix_cache:
            extra.append("--no-enable-prefix-caching")
        extra += p.extra_args
        if bad := [a for a in extra if not a or any(c.isspace() for c in a)]:
            raise ValueError(f"the launcher splits EXTRA_ARGS on whitespace, so {bad!r} can't be passed")
        env = {
            "CUDA_HOME": str(paths.CUDA_HOME),
            "HOST": host,
            "PORT": str(port),
            "MODEL": weights,
            "SPEC": p.spec,
            "CTX": p.ctx_profile,
            "MAX_LEN": str(p.ctx),
            "PREFIX_CACHE": _flag(p.prefix_cache),
            "KV_MEM": str(p.kv_mem),
            "MAX_SEQS": str(p.max_seqs),
            "GPU_UTIL": str(p.gpu_util),
            "DFLASH_TOKENS": str(p.dflash_tokens),
            "LOOKUP": _flag(p.lookup),
            "TOOLS": _flag(p.tools),
            "VISION": _flag(p.vision),
        }
        if draft:
            env["DRAFT"] = draft
        if p.cudagraph_mode:
            env["CUDAGRAPH_MODE"] = p.cudagraph_mode
        if clash := sorted((set(env) | {"EXTRA_ARGS", "VLLM_API_KEY"}) & set(p.env)):
            raise ValueError(f"params.env may not set {', '.join(clash)}; those come from the config's own fields")
        env.update(p.env)
        env["EXTRA_ARGS"] = " ".join(extra)
        return env

    def server_argv(self, model: ModelSpec, cfg: ConfigSpec, *, host: str, port: int) -> list[str]:
        p = cfg.params
        assert isinstance(p, VllmParams)
        self.check_checkout(p)
        weights = str(catalog.resolve_model_path(model))
        draft = str(self.root / p.draft) if p.draft else None
        env = self.launch_env(model, cfg, host=host, port=port, weights=weights, draft=draft)
        return ["env", "-u", "VLLM_API_KEY", *(f"{k}={v}" for k, v in env.items()), "/bin/bash", str(self.root / p.launcher)]

    def display_command(self, model: ModelSpec, cfg: ConfigSpec) -> str:
        p = cfg.params
        assert isinstance(p, VllmParams)
        env = self.launch_env(model, cfg, host="127.0.0.1", port=8080, weights=model.source.path or model.slug, draft=p.draft)
        return shlex.join([*(f"{k}={v}" for k, v in env.items()), p.launcher])

    def render_files(self, model: ModelSpec, cfg: ConfigSpec) -> dict[str, str]:
        return {}

    def request_extras(self) -> dict:
        return {"cache_salt": uuid.uuid4().hex}

    server_log_name = "vllm-server.log"

    def dialect(self, model: ModelSpec, cfg: ConfigSpec) -> VllmDialect:
        p = cfg.params
        assert isinstance(p, VllmParams)
        build = None
        try:
            build = self.build_info()
        except Exception:  # noqa: BLE001 - provenance only; a run must not fail because git is unhappy
            pass
        return VllmDialect(served_model=f"{model.slug}/{cfg.slug}",
                           weights_dir=catalog.resolve_model_path(model), params=p, build=build)

    @staticmethod
    def served_model(base_url: str) -> dict:
        """The one model the server lists: its id and max_model_len."""
        data = httpx.get(f"{base_url}/v1/models", timeout=10).raise_for_status().json()["data"]
        if len(data) != 1:
            raise RuntimeError(f"expected one served model, got {[m.get('id') for m in data]}")
        return data[0]

    # -- identity ----------------------------------------------------------------
    def _site_packages(self) -> Path:
        found = sorted(self.root.glob("venv/lib/python3.*/site-packages/vllm"))
        if not found:
            raise FileNotFoundError(f"no vllm package in {self.root}/venv")
        return found[-1]

    def patch_status(self) -> tuple[int, list[str]]:
        """(applied, missing) over patches/series, checked the way the launcher repo's verify.sh checks them."""
        site = self._site_packages()
        pdir = self.root / "patches"
        series = [ln.split("#")[0].strip() for ln in (pdir / "series").read_text().splitlines()]
        series = [n for n in series if n and n not in RETIRED_PATCHES]

        def applied(name: str) -> bool:
            with open(pdir / name, "rb") as fh:
                if subprocess.run(["patch", "-p1", "-R", "--dry-run", "-s", "-d", str(site)], stdin=fh, capture_output=True).returncode == 0:
                    return True
            # Overlapping hunks defeat the reverse dry-run; the repo's content check covers that case.
            python = self.root / "venv/bin/python"
            return subprocess.run([str(python), str(pdir / "_check_applied.py"), str(pdir / name), str(site)], capture_output=True).returncode == 0

        status = {n: applied(n) for n in series}
        for name in series:
            if status[name]:  # a later patch that rewrote an earlier one's region stands in for it
                for target in SUPERSEDES_RE.findall((pdir / name).read_text(errors="replace")):
                    if target in status:
                        status[target] = True
        missing = [n for n, ok in status.items() if not ok]
        return len(series) - len(missing), missing

    def build_info(self) -> EngineBuild:
        python = self.root / "venv/bin/python"
        out = subprocess.run([str(python), "-c", VERSIONS_PY], capture_output=True, text=True)
        versions = json.loads(out.stdout) if out.returncode == 0 else {}
        applied, missing = self.patch_status()
        extra: dict = {
            "launcher_repo": "syv-ai/qwen38-27b-rtx3090",
            "launcher_commit": self._git("rev-parse", "HEAD") or None,
            "dirty": bool(self._git("status", "--porcelain", "--untracked-files=no")),
            "patch_series_sha256": hashlib.sha256((self.root / "patches/series").read_bytes()).hexdigest(),
            "patches_applied": applied,
            "patches_missing": missing,
            "python": versions.get("python"),
            "torch": versions.get("torch"),
            "flashinfer": versions.get("flashinfer-python"),
            "transformers": versions.get("transformers"),
        }
        try:
            nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], capture_output=True, text=True).stdout
            if m := re.search(r"release ([\d.]+)", nvcc):
                extra["cuda_toolkit"] = m.group(1)
        except FileNotFoundError:
            pass
        return EngineBuild(engine=self.name, version=versions.get("vllm"), commit_sha=self._git("rev-parse", "--short=9", "HEAD") or None,
                           build_flags=None, extra=extra)
