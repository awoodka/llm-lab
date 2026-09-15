# llm-lab

Benchmark local LLMs on the `ai` container (RTX 3090) and publish the results to **Local Inference**, a public
showcase site on the `web` container, with a private chat for the model `ai` hosts.

- **https://localinference.alexwoodka.com**: the homepage (intro, headline numbers, live chat status), the benchmarks
  dashboard, model and run pages, and the methodology. Public, served by Caddy through the Cloudflare tunnel.
- **https://chat.alexwoodka.com**: Open WebUI for the hosted model, behind Cloudflare Access (you@example.com only).
- `lab/`: Python CLI (`lab`) that runs on `ai`. It manages the model/config catalog, benchmarks, the GPU lock, the hosted model and publishing.
- `web/`: Node/TypeScript site (Hono + SQLite) that runs on `web`, plus its deploy files in `web/deploy/`.

A **model** is a base model plus a quant (Qwen 27B Q3_K_M and Q4_K_M are different models).
A **config** is one set of runtime settings for a model (ctx, KV cache type, offload, and so on). You reference a config as `<model>/<config>`.

## Workflow

```sh
cd ~/llm-lab/lab
uv run lab doctor

# register a GGUF (downloads to HF_HOME) and describe its base model in catalog/bases/<base>.yaml
uv run lab model add unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --base qwen3-30b-a3b

# create settings to try; keys without a prefix go to params
uv run lab config new qwen3-30b-a3b-q4_k_m-gguf 32k-q8kv --set ctx=32768 --set cache_type_k=q8_0 --set cache_type_v=q8_0
uv run lab config show qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv --cmd

# tinker interactively (pauses the hosted model; Ctrl-C resumes it)
uv run lab serve qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv --host 0.0.0.0

# benchmark → inspect → publish
uv run lab bench qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv
uv run lab runs ls --unpublished
uv run lab publish <run-dir-name-or-id>

# host the winner (it becomes the model on chat.alexwoodka.com)
uv run lab promote qwen3-30b-a3b-q4_k_m-gguf/32k-q8kv
```

Guarantees:
- Every knob is rendered explicitly, including `--fit off`, `-np` and `--cache-ram`, so engine defaults never leak into results.
- Configs are hash-locked: the site refuses a config slug that is republished with different settings.
- GPU jobs take `state/gpu.lock`, stop the hosted model, wait for idle VRAM and a GPU below 45 °C, then restart the hosted model afterwards, even after a crash (on the next `lab` command).
- While a benchmark or `lab serve` has the GPU, the homepage shows the chat as paused and names the model. The report is best effort and never slows a benchmark: the site probes the model's health itself.
- Published data names files, never paths on `ai`: `lab publish` refuses a bundle that contains one.
- Nothing public can write. The site's pages and its publishing API are separate listeners, and only the pages are reachable through Caddy.

## How it's wired

```
visitor → Cloudflare (Access on chat.* only) → cloudflared → caddy on web's `edge` Docker network
            localinference.alexwoodka.com → llmlab-site:3000        pages only
            chat.alexwoodka.com           → llmlab-open-webui:8080  Caddy also requires the Access JWT header
ai  → https://web.example-tailnet.ts.net (tailscale serve) → 127.0.0.1:3000 → site API listener (:3100)   lab publish / promote / pause
web → https://ai.example-tailnet.ts.net:8443 → llama-server on ai                                            chat, status probe
```

## One-time setup (needs you: sudo, the Proxmox host or a dashboard)

On the **pve host** (this wipes the NVMe, so first confirm nothing on it is needed with `lsblk -f`):
```sh
wipefs -a /dev/<disk>
parted -s /dev/<disk> mklabel gpt mkpart models ext4 0% 100%
mkfs.ext4 -m 0 -L models /dev/<disk>p1
mkdir -p /mnt/nvme && echo 'LABEL=models /mnt/nvme ext4 defaults,noatime 0 2' >> /etc/fstab && mount /mnt/nvme
mkdir -p /mnt/nvme/models && chown <host-uid>:<host-gid> /mnt/nvme/models   # alex (uid 1000) in unprivileged CT <ctid>
pct set <ctid> -mp0 /mnt/nvme/models,mp=/mnt/models,backup=0
pct reboot <ctid>
```

On **ai** (as alex):
```sh
sudo loginctl enable-linger alex          # user systemd units (lab-hosted) survive logout/reboot
sudo tailscale set --operator=alex        # lets alex run `tailscale serve`
echo 'export HF_HOME=/mnt/models/hf' >> ~/.bashrc
# move the existing cache so nothing is re-downloaded:
mkdir -p /mnt/models/hf && mv ~/.cache/huggingface/hub /mnt/models/hf/
# expose the hosted model to the tailnet (for the chat and the site's status probe on web):
tailscale serve --bg --https=8443 http://127.0.0.1:8080
```
On ai itself the tailnet name resolves to 127.0.1.1 (`/etc/hosts`), so test the hosted model from another device or with
`curl --resolve ai.example-tailnet.ts.net:8443:100.64.0.1 …`.

In the **Tailscale policy file**:
- An `ssh` rule accepting members into `tag:web` as `alex`/`root`, then `sudo tailscale set --ssh` on web.
- `web` is tagged, so member grants don't cover its outgoing traffic. Add
  `{"src": ["tag:web"], "dst": ["100.64.0.1"], "ip": ["tcp:8443"]}` for the chat and status probe → hosted model.
  `ai → web:443` (publishing) is covered by the default member grant.

On **web**:
```sh
mkdir -p /opt/llmlab && (umask 077; printf 'INGEST_TOKEN=%s\nWEBUI_SECRET_KEY=%s\n' \
  "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > /opt/llmlab/.env)
sudo tailscale serve --bg --https=443 http://127.0.0.1:3000   # the publishing API, tailnet only
```
Then put the same ingest token in `lab/settings.yaml` on ai (gitignored, mode 600):
```yaml
web_url: https://web.example-tailnet.ts.net
ingest_token: "<INGEST_TOKEN from /opt/llmlab/.env on web>"
```

In **Cloudflare Zero Trust**, in this order, so the chat is never reachable without Access:
1. Settings → Authentication: enable One-time PIN or Google.
2. Access → Applications → Add → Self-hosted: hostname `chat.alexwoodka.com` (path empty), policy **Allow** →
   Include → Emails → `you@example.com`, session duration 1 week, no Bypass, Service Auth or Everyone rules.
3. Networks → Tunnels → the edge tunnel → Public hostnames: `localinference.alexwoodka.com` → `http://caddy:80`, and
   `chat.alexwoodka.com` → `http://caddy:80` with **Protect with Access** enabled.

## Deploying the site

```sh
web/deploy/push.sh                                                      # from ai
tailscale ssh alex@web /opt/llmlab/repo/deploy/apply-caddy.sh      # first time, or after editing Caddyfile.llmlab
```
- `push.sh` copies `web/` to `/opt/llmlab/repo` (the previous tree stays in `repo.prev`) and runs `deploy.sh`.
- `deploy.sh` backs up the SQLite database, refuses to attach Open WebUI to `edge` while the chat is published without
  Access, rebuilds, and stops the stack if anything but 127.0.0.1:3000 is published or any other llmlab container joins
  `edge`. It finishes with `check-public.sh`.
- `check-public.sh` also runs every 10 minutes from alex's crontab on web (`--chat-only`). If the chat stops redirecting
  to Cloudflare Access, it disconnects Open WebUI from Caddy; if the public site answers on `/api` or leaks a tailnet
  address, it disconnects the site.
- `apply-caddy.sh` validates the blocks in a throwaway Caddy, edits the shared `/opt/edge/Caddyfile` in place (it's a
  single-file bind mount), restarts Caddy, checks the other sites and both hostnames, and restores the backup on any failure.

Rollback: redeploy `repo.prev`; restore `/opt/edge/Caddyfile.bak-*` with `cat backup > /opt/edge/Caddyfile` and
`docker restart edge-caddy-1`; database backups are in `/opt/llmlab-data/site/backups`. To switch the chat off at once,
delete its public hostname in Cloudflare or run `docker network disconnect edge llmlab-open-webui-1` on web.

## Development

```sh
cd lab && uv run pytest
cd web && npm test && npx tsc --noEmit
INGEST_TOKEN=devtoken npm run dev          # pages on http://127.0.0.1:3000, publishing API on http://127.0.0.1:3100
```
To publish to a local dev site, set `web_url: http://127.0.0.1:3100` and `ingest_token: devtoken` in `lab/settings.yaml`.
