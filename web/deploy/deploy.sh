#!/usr/bin/env bash
# Rebuild and restart the LLM Lab stack on `web` (site + Open WebUI), then prove exposure is exactly as intended:
#   - the only port published on the host is 127.0.0.1:3000: the site's publishing API, for `tailscale serve`
#   - the only llmlab containers on the public `edge` network are the site and Open WebUI, which Caddy proxies
#   - chat.alexwoodka.com redirects unauthenticated requests to Cloudflare Access (check-public.sh)
# Safe to re-run. Normally started by push.sh on ai.
set -Eeuo pipefail

L=/opt/llmlab
DEPLOY=$(cd "$(dirname "$0")" && pwd)
cd "$L"

grep -qE '^INGEST_TOKEN=.{32,}$' .env || { echo "FAIL: INGEST_TOKEN is missing or short in $L/.env"; exit 1; }
grep -qE '^WEBUI_SECRET_KEY=.{32,}$' .env || { echo "FAIL: WEBUI_SECRET_KEY is missing or short in $L/.env"; exit 1; }
mkdir -p /opt/llmlab-data/site/backups /opt/llmlab-data/open-webui "$L/logs"

compose() { docker compose --env-file "$L/.env" -f "$DEPLOY/compose.yml" "$@"; }
edge_members() { docker network inspect edge --format '{{range .Containers}}{{.Name}}{{"\n"}}{{end}}'; }

# Back up the published data first. The image has node but no sqlite3 CLI.
if [ "$(docker inspect -f '{{.State.Running}}' llmlab-site-1 2>/dev/null)" = true ]; then
  ts=$(date +%Y%m%d-%H%M%S)
  docker exec -e NODE_NO_WARNINGS=1 llmlab-site-1 node -e \
    "new (require('node:sqlite').DatabaseSync)('/data/lab.db').exec(\"VACUUM INTO '/data/backups/lab-$ts.db'\")"
  ls -1t /opt/llmlab-data/site/backups/lab-*.db | tail -n +11 | xargs -r rm --
  echo "backup: /opt/llmlab-data/site/backups/lab-$ts.db"
fi

# Nothing gets attached to the public network unless this run proved the chat is behind Access. "Couldn't check"
# (exit 75, e.g. Cloudflare unreachable from web) is not proof, so it stops the deploy too; try again later.
if ! "$DEPLOY/check-public.sh" --chat-only; then
  echo "FAIL: could not prove chat.alexwoodka.com is behind Cloudflare Access; not deploying"
  exit 1
fi

compose build --pull
compose up -d --remove-orphans

# check-public.sh may have cut a container off earlier; reattach it now that the chat check passed.
for name in llmlab-site llmlab-open-webui; do
  if ! edge_members | grep -qx "$name-1"; then
    docker network connect --alias "$name" edge "$name-1"
    echo "reattached $name-1 to edge"
  fi
done
rm -f "$L/logs/open-webui-cut-off"

# The hard gate.
problems=()
ports=$(docker ps --filter label=com.docker.compose.project=llmlab --format '{{.Ports}}' \
        | tr ',' '\n' | sed 's/^ *//' | grep -- '->' | sort -u | paste -sd' ' || true)
[ "$ports" = "127.0.0.1:3000->3100/tcp" ] || problems+=("published ports are [$ports]; expected only 127.0.0.1:3000->3100/tcp")
on_edge=$(edge_members | grep '^llmlab-' | sort | paste -sd' ' || true)
[ "$on_edge" = "llmlab-open-webui-1 llmlab-site-1" ] || problems+=("llmlab containers on edge are [$on_edge]; expected llmlab-open-webui-1 llmlab-site-1")
risky=$(docker ps -q --filter label=com.docker.compose.project=llmlab \
        | xargs -r docker inspect -f '{{.Name}} {{.HostConfig.NetworkMode}} {{.HostConfig.Privileged}}' | awk '$2 == "host" || $3 == "true"')
[ -z "$risky" ] || problems+=("host networking or privileged mode: $risky")
if [ ${#problems[@]} -gt 0 ]; then
  printf 'FAIL: %s\n' "${problems[@]}"
  echo "Stopping the stack."
  compose down
  exit 1
fi
echo "exposure OK: only 127.0.0.1:3000 on the host; site and Open WebUI on edge"

for c in llmlab-site-1 llmlab-open-webui-1; do
  state=unknown
  for _ in $(seq 90); do
    state=$(docker inspect -f '{{.State.Health.Status}}' "$c")
    [ "$state" = healthy ] && break
    sleep 2
  done
  [ "$state" = healthy ] || { echo "FAIL: $c is $state"; compose ps; exit 1; }
  echo "healthy: $c"
done

# The host port is the API: it needs the token and never serves the public pages.
code=$(curl -sS -o /dev/null -w '%{http_code}' -X PUT http://127.0.0.1:3000/api/hosted || true)
[ "$code" = 401 ] || { echo "FAIL: 127.0.0.1:3000 answered $code to a tokenless API request; expected 401"; exit 1; }
root=$(curl -sS http://127.0.0.1:3000/ || true)
if [[ $root == *'<html'* ]]; then echo "FAIL: 127.0.0.1:3000 serves the public pages"; exit 1; fi
echo "API OK: 127.0.0.1:3000 requires the token and serves no pages"

# Final check. Exit 1 has already cut the exposed container off; 75 only means a check couldn't finish.
if "$DEPLOY/check-public.sh"; then
  :
elif [ $? = 75 ]; then
  echo "WARN: the final public check couldn't finish; the watchdog re-checks every 10 minutes"
else
  exit 1
fi
compose ps
