# llm-lab

Benchmark local LLMs on one GPU, score what they can actually do, and publish the results to a read-only showcase
site. It runs **[Local Inference](https://localinference.alexwoodka.com)**: one RTX 3090, capped at 250 W, that also
hosts a private chat, with every setting behind every number on the page.

- `lab/`: a Python CLI (`lab`) for the GPU host. It keeps the model and config catalog, runs speed benchmarks and
  capability evals, owns the GPU lock and the hosted chat model, and publishes runs.
- `web/`: the site, Node/TypeScript (Hono + SQLite), for a separate web host, with its deploy scripts in
  `web/deploy/`. It has two listeners: public pages, and a publishing API that only the tailnet can reach.

A **model** is a base model plus a quant (Qwen 27B Q3_K_M and Q4_K_M are different models). A **config** is one set
of runtime settings for a model (context, KV-cache type, offload, speculative decoding and so on), referenced as
`<model>/<config>`. A **run** is one benchmark or eval sitting; it stays local until you publish it.

## Workflow

```sh
cd lab
uv run lab doctor

# register a GGUF (downloads to HF_HOME) and describe its base model in catalog/bases/<base>.yaml
uv run lab model add unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --base qwen3-30b-a3b

# create settings to try; keys without a prefix go to params
uv run lab config new qwen3-30b-a3b-q4_k_m-gguf 32k-q8kv --set ctx=32768 --set cache_type_k=q8_0 --set cache_type_v=q8_0
uv run lab config show qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv --cmd

# tinker interactively (pauses the hosted model; Ctrl-C resumes it)
uv run lab serve qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv --host 0.0.0.0

# benchmark and score → inspect → publish
uv run lab bench qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv
uv run lab eval qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv --tier quick
uv run lab runs ls --unpublished
uv run lab publish <run-dir-name-or-id>

# host the winner: it becomes the chat model
uv run lab promote qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv
```

Guarantees:
- Every knob is rendered explicitly, including `--fit off`, `-np` and `--cache-ram`, so engine defaults never leak into results.
- Configs are hash-locked: the site refuses a config slug that is republished with different settings.
- GPU jobs take `state/gpu.lock`, stop the hosted model, wait for idle VRAM and a GPU below 45 °C, then restart the hosted model afterwards, even after a crash: `lab-recover.timer` puts chat back within two minutes.
- GPU work refuses to start while the card's enforced power limit is above `LAB_MAX_POWER_W` (250 W by default).
- A tier that runs for hours survives the terminal that started it: `lab eval` treats a dropped connection like Ctrl-C, saving its checkpoint and handing chat back, and resumes with `--resume`.
- While a benchmark or `lab serve` has the GPU, the homepage shows the chat as paused and names the model. The report is best effort and never slows a benchmark: the site probes the model's health itself.
- Published data names files, never paths or tailnet addresses: `lab publish` refuses a bundle that contains one.
- Nothing public can write. The site's pages and its publishing API are separate listeners, and only the pages are reachable through Caddy.

## How it's wired

```
visitor → Cloudflare (Access on the chat hostname only) → cloudflared → Caddy on the web host's `edge` Docker network
            SITE_HOST → llmlab-site:3000        pages only
            CHAT_HOST → llmlab-open-webui:8080  Caddy also requires the Access JWT header
GPU host → https://web.<tailnet>.ts.net (tailscale serve) → 127.0.0.1:3000 → site API listener (:3100)   lab publish / promote / pause
web host → https://<gpu-host>.<tailnet>.ts.net:8443 (tailscale serve) → the hosted model's server       chat, status probe
```

The chat is [Open WebUI](https://github.com/open-webui/open-webui) with no login of its own: Cloudflare Access is the
gate, Caddy refuses requests without Access's header, and `check-public.sh` cuts Open WebUI off from Caddy if the chat
ever stops redirecting to Access. The tailnet is plumbing only; nothing public points into it.

## Engines

`llama.cpp` runs every GGUF. The second engine is vLLM, through a **pinned clone of
[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)**: patched vLLM 0.28.0 with a W4A16
AutoRound checkpoint and a DFlash2 draft model, which serves Qwen3.8 27B about five times faster than the Q4_K_M
GGUF on the same card (146 t/s against 30.6 t/s of chat generation). Its fork,
**[awoodka/qwen38-27b-rtx3090](https://github.com/awoodka/qwen38-27b-rtx3090)**, adds opt-in levers that steer how
Qwen3.8 spends its thinking, measured with this lab.

```
~/qwen-serving -> ~/qwen-serving-bae2023      the pin; an upgrade is a NEW clone, venv and config slug, then a symlink flip
  venv/                                       its own uv venv (Python 3.12); the GPU host itself stays free of torch
  single-user/start_qwen.sh                   the launcher; the lab sets every knob as an env var and never calls `vllm serve`
  models/ -> <models disk>/qwen-serving/models  the W4A16 weights and the DFlash2 draft
~/qwen-serving-<sha7>                         another pinned commit's own clone and venv, e.g. the fork
```

- A config pins `launcher_commit`. `lab doctor` checks the pin, the patch series, the weights, the draft and the CUDA
  toolkit; a checkout that has drifted, or that grew an `api_key.txt`, refuses to start.
- The pin also picks the checkout: `~/qwen-serving-<sha7>` when it exists and isn't what `~/qwen-serving` already
  points at, else `~/qwen-serving`, so the hosted config's command line never changes. `LAB_QWEN_SERVING` pins one
  checkout for everything. A run records its checkout's GitHub origin as `launcher_repo`, so a fork's runs say so,
  and `lab doctor` checks every pinned checkout.
- The launcher binds `0.0.0.0` by default, so the lab always passes `HOST`. It never sends an API key.
- `CUDA_HOME` is part of the launch environment: flashinfer JIT-compiles kernels on a config's first boot, and through
  the `/usr/bin/nvcc` symlink it would look for CUDA headers in `/usr` and fail. Override with `LAB_CUDA_HOME`.
- The first boot of a config compiles for a few minutes; later boots take about a minute. Never wipe
  `~/.cache/vllm/torch_compile_cache` or `~/.cache/flashinfer` — an env change already recompiles what it must.
- VRAM is preallocated (`gpu_util`, a fixed KV pool), so the card must be empty before boot. `lab gpu run` takes the
  lock, pauses chat, runs a command in its own process group and restores chat afterwards.

```sh
uv run lab bench qwen3.8-27b-w4a16-autoround-fast/64k-dflash2      # --http is the default for vLLM
uv run lab promote qwen3.8-27b-w4a16-autoround-fast/64k-dflash2    # rolls back on its own if it never gets healthy
```

Then set the default model in Open WebUI (Admin → Settings → Models) so new chats use it.

**Rollback:** `lab promote` the previous config (a llama.cpp config boots in about 20 s) and reset Open WebUI's
default model; for the serving stack itself, point `~/qwen-serving` back at the previous clone; `lab unpublish <run>`
removes published numbers.

## Speed: two methods, never mixed

- `lab bench <ref> --speed` is llama-bench: llama.cpp only, no chat template, short generations, and it sweeps context
  depth. It answers "how fast is this engine at depth N".
- `lab bench <ref> --http` is `chat-c1-v1`, any engine: eight real chat prompts through the server's own
  OpenAI-compatible API, one at a time, a discarded warmup and then a sampled and a greedy cohort, 1024 tokens each
  with thinking off. TTFT is the first content delta; TPOT spreads first-to-last delta over the server's token count,
  never over chunks, because speculative decoding packs several tokens into one. Every request must be cold, so vLLM
  gets a fresh `cache_salt` and llama.cpp gets `cache_prompt: false`.

The same model measures ~33 t/s under llama-bench and ~30.6 t/s over HTTP; both are right, and the site keeps them in
separate columns. Compare engines only through the HTTP method.

## Capability evals

```sh
uv run lab eval <ref> --tier quick                                  # LiveCodeBench, BFCL, GPQA Diamond, AIME 2025
uv run lab eval <ref> --tier quick --benchmarks aime_2025 --limit 2 # a smoke test; never publishable
uv run lab eval <ref> --resume <run> --stop-at 07:30                # tiers can span several sittings
```

- Each config is scored exactly as it serves chat, thinking included, against its own server.
- **A thinking allowance, not a stopwatch:** each task gets the tokens the config could generate in 5 minutes at its
  own published speed, after reading the prompt. Thinking counts against it, and an answer that doesn't finish is
  wrong, so speed buys reasoning. The allowance proxy (`lab/src/lab/evals/proxy.py`) is the only place request size
  is set; it counts every prompt with the server's own tokenizer and checks each answer against the server's usage.
- **Official harnesses where they exist:** LiveCodeBench and BFCL run their own published code for prompts and
  grading, in no-network containers on the web host, and ask the lab for completions over a pipe. AIME and GPQA are
  graded by the lab (the boxed integer; the last "Answer: X").
- **Pinned subsets:** `lab/subsets/` holds the task ids each benchmark is pinned to, so every run answers the same
  questions. GPQA's authors ask that its questions stay off the web: the repository holds its ids only, runs publish
  scores and task ids, and the lab's reports print aggregates (`lab runs markers` refuses GPQA outright).
- A tier is scored as one run; `lab publish` refuses partial tiers and `--limit` smoke tests.

The site's [methodology page](https://localinference.alexwoodka.com/methodology) is the full account: the allowance
formula, the coding-work score, the error bars, and what the numbers don't show.

### How a run spent its thinking

```sh
uv run lab runs thinking <run> [<run>...] [--baseline <run>] [--format table|markdown|csv]
uv run lab runs markers <run> --benchmarks livecodebench,bfcl
```

`thinking` gives, per benchmark: accuracy ± SE, reasoning tokens (the server's count where the run kept it, else the
completion tokens, labelled), length stops, reflection-marker rates per 1,000 reasoning words, tok/s, and for AIME the
share of reasoning after the answer first appears. With `--baseline` it adds task-paired deltas with bootstrap 95%
intervals and the net count of newly failed attempts. `markers` lists the words the reasoning opens its sentences with,
finished and cut-off attempts apart; it is where a marker penalty's word list comes from. Both print aggregates and task
ids only, and `markers` refuses GPQA outright.

## Deploying your own

You need a **GPU host** (Linux, an NVIDIA card, [uv](https://docs.astral.sh/uv/), a CUDA build of llama.cpp in
`~/llama.cpp`) and a **web host** with Docker, both on one [Tailscale](https://tailscale.com) tailnet, plus a
Cloudflare zone with a tunnel for the public side. The web host is expected to run a shared "edge": Caddy (with a
healthcheck) and cloudflared on a Docker network named `edge`, and a Caddyfile that defines a `secheaders` snippet.

On the **GPU host**, from a clone of this repository at `~/llm-lab` (the recovery unit expects it there):
```sh
cd ~/llm-lab/lab && uv sync
echo 'export HF_HOME=/path/to/a/big/disk/hf' >> ~/.bashrc     # models are large; LAB_MODELS moves the lab's own files
sudo loginctl enable-linger "$USER"                           # user units (lab-hosted, lab-recover) survive logout
mkdir -p ~/.config/systemd/user && cp systemd/lab-recover.* ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now lab-recover.timer               # chat comes back if a GPU job dies mid-run
sudo tailscale set --operator="$USER"                         # lets you run `tailscale serve`
tailscale serve --bg --https=8443 http://127.0.0.1:8080       # the hosted model, for the chat and the status probe
cp ../web/deploy/local.env.example ../web/deploy/local.env    # how push.sh reaches the web host; edit it
```
Cap the GPU's power at boot on the machine that owns the card, as root (`nvidia-smi -pm 1 && nvidia-smi -pl 250`,
for example from a oneshot unit), or set `LAB_MAX_POWER_W` to the limit you run: GPU work refuses to start above it.
`lab promote` writes the hosted model's user unit. Then put the site's publishing address and token in
`lab/settings.yaml` (gitignored, mode 600):
```yaml
web_url: https://web.<tailnet>.ts.net
ingest_token: "<INGEST_TOKEN from the app's .env on the web host>"
sandbox_host: you@web     # LiveCodeBench and BFCL run their containers here, over ssh (BatchMode): key or Tailscale SSH
```

On the **web host**, as the user who will deploy (uid 1000, which the site's container runs as), pick an app
directory, say `/opt/llmlab`, and copy `web/deploy/env.example` to `/opt/llmlab/.env` (mode 600). Fill in every
line: the two secrets (`openssl rand -hex 32` each), the public hostnames, the GPU host's tailnet name and address,
a data directory that user owns, and the edge Caddy's details; `env.example` explains each. Then:
```sh
sudo tailscale serve --bg --https=443 http://127.0.0.1:3000     # the publishing API, tailnet only
```
If the web host is a tagged tailnet node, member grants don't cover its outgoing traffic: grant it `tcp:8443` to the
GPU host.

In **Cloudflare Zero Trust**, in this order, so the chat is never reachable without Access:
1. Settings → Authentication: enable One-time PIN or an identity provider.
2. Access → Applications → Add → Self-hosted, for `CHAT_HOST` with an empty path, and a policy that allows only your
   own email. No Bypass, Service Auth or Everyone rules.
3. The tunnel's public hostnames: `SITE_HOST` → `http://caddy:80`, and `CHAT_HOST` → `http://caddy:80` with
   **Protect with Access** on.

Then deploy from the GPU host:
```sh
web/deploy/push.sh                                            # copy web/ to <app dir>/repo there and run deploy.sh
ssh -t <web host> <app dir>/repo/deploy/apply-caddy.sh        # first time, or after editing Caddyfile.llmlab
```
and add the watchdog to the web host's crontab:
```
*/10 * * * * <app dir>/repo/deploy/check-public.sh --chat-only >> <app dir>/logs/public-check.log 2>&1
```

- `push.sh` copies `web/` to `<app dir>/repo` on the web host (the previous tree stays in `repo.prev`) and runs `deploy.sh`.
- `deploy.sh` backs up the SQLite database, refuses to attach Open WebUI to `edge` while the chat is published without
  Access, rebuilds, and stops the stack if anything but 127.0.0.1:3000 is published or any other llmlab container joins
  `edge`. It finishes with `check-public.sh`.
- `check-public.sh` runs every 10 minutes with `--chat-only`. If the chat stops redirecting to Cloudflare Access, it
  disconnects Open WebUI from Caddy; if the public site answers on `/api` or leaks a tailnet address, it disconnects
  the site. `--dry-run` reports without changing anything.
- `apply-caddy.sh` renders `Caddyfile.llmlab` from `.env`, validates it in a throwaway Caddy, edits the shared
  Caddyfile in place (it's a single-file bind mount), restarts Caddy, checks the other sites and both hostnames, and
  restores the backup on any failure. `--check` only compares what Caddy would run with what it runs now.

Rollback: redeploy `repo.prev`; restore a Caddyfile backup (`$EDGE_CADDYFILE.bak-*`) with `cat backup > "$EDGE_CADDYFILE"`
and `docker restart "$EDGE_CADDY_CONTAINER"`; database backups are in `$LLMLAB_DATA_DIR/site/backups`. To switch the
chat off at once, delete its public hostname in Cloudflare or run `docker network disconnect edge llmlab-open-webui-1`.

## Development

```sh
cd lab && uv run pytest
cd web && npm test && npx tsc --noEmit
INGEST_TOKEN=devtoken npm run dev          # pages on http://127.0.0.1:3000, publishing API on http://127.0.0.1:3100
scripts/prepush-check.sh                   # before a push: your private literals and GPQA fragments (~/.config/llm-lab) in all history
```
To publish to a local dev site, set `web_url: http://127.0.0.1:3100` and `ingest_token: devtoken` in `lab/settings.yaml`.
`lab/tests/test_repo_hygiene.py` fails if a tracked file names a tailnet host or address, an inbox or a host's disk;
examples use `example-tailnet`, `100.64.0.1` and `example.com`.

## License

Apache License 2.0 ([LICENSE](LICENSE)). [NOTICE](NOTICE) lists the material this repository adapts (the chat
benchmark's prompts from syv-ai/qwen38-27b-rtx3090, BFCL's grading, the simple-evals GPQA prompt) and what it fetches
at run time under its own terms (LiveCodeBench, bfcl-eval, the datasets, Open WebUI).

The history was rewritten once, when the repository went public, to take one homelab's private details out of it;
[docs/history-rewrite.md](docs/history-rewrite.md) maps the old commit ids, which runs published before then record
as their lab version, to the new ones.
