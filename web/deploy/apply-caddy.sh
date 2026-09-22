#!/usr/bin/env bash
# Install or update the LLM Lab blocks in the shared edge Caddyfile, which may also serve other sites.
#   1. render Caddyfile.llmlab with SITE_HOST and CHAT_HOST from the app's .env, and build a candidate with it
#      between the `# >>> llmlab` / `# <<< llmlab` markers
#   2. validate it in a throwaway container from the running Caddy image
#   3. write it in place: the Caddyfile is a single-file bind mount, so replacing its inode (editors, sed -i, mv)
#      would leave Caddy silently on the old file
#   4. restart Caddy (`admin off` rules out a hot reload), check every site, and restore the backup on any failure
# --check stops after 2 and writes nothing: it compares what Caddy would run (`caddy adapt`) for the candidate
# and for the live file, so a change to comments alone passes. Exit 0 same, 1 different, anything else: no verdict.
# .env also names the Caddyfile (EDGE_CADDYFILE), its container (EDGE_CADDY_CONTAINER) and, optionally, the other
# sites it serves (EDGE_OTHER_SITES, comma-separated URLs that must still answer after a restart).
# Run on the web host as the app's user: <app dir>/repo/deploy/apply-caddy.sh [--yes | --check]
set -Eeuo pipefail

DEPLOY=$(cd "$(dirname "$0")" && pwd)
. "$DEPLOY/env.sh"
load_env "$ENV_FILE" SITE_HOST CHAT_HOST EDGE_CADDYFILE EDGE_CADDY_CONTAINER || exit 2
EDGE_OTHER_SITES=${EDGE_OTHER_SITES:-$(env_value "$ENV_FILE" EDGE_OTHER_SITES)}
IFS=', ' read -ra other_sites <<< "$EDGE_OTHER_SITES"

F=$EDGE_CADDYFILE
C=$EDGE_CADDY_CONTAINER
BLOCK=$DEPLOY/Caddyfile.llmlab
mode=ask
case "${1:-}" in
  "") ;;
  --yes) mode=yes ;;
  --check) mode=check ;;
  *) echo "usage: $0 [--yes | --check]" >&2; exit 2 ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cand=$work/Caddyfile

SITE_HOST=$SITE_HOST CHAT_HOST=$CHAT_HOST python3 - "$F" "$BLOCK" "$cand" <<'EOF'
import os
import re
import sys

live, block, out = sys.argv[1:]
src = open(live).read()


def fail(message: str):
    print(message, file=sys.stderr)
    sys.exit(2)


def render(m: re.Match) -> str:
    name = m.group(1)
    if name not in ("SITE_HOST", "CHAT_HOST"):
        fail(f"Caddyfile.llmlab uses ${{{name}}}; only SITE_HOST and CHAT_HOST are rendered")
    value = os.environ[name]
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+", value):
        fail(f"{name}={value!r} in .env is not a lowercase hostname")
    return value


new = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", render, open(block).read()).rstrip("\n") + "\n"
if "${" in new:
    fail("Caddyfile.llmlab still has an unrendered ${...} placeholder")
start, end = "# >>> llmlab", "# <<< llmlab"
if start in src:
    i = src.index(start)
    j = src.index(end, i)
    j = src.find("\n", j) + 1 or len(src)
    result = src[:i] + new + src[j:]
else:
    anchor = next((a for a in ("# Anything cloudflared forwards", ":80 {") if a in src), None)
    if anchor is None:
        fail("no insertion point: found neither the llmlab markers nor the :80 fallback block")
    i = src.index(anchor)
    result = src[:i] + new + "\n" + src[i:]
open(out, "w").write(result)
EOF

if cmp -s "$F" "$cand"; then
  echo "Caddyfile already up to date"
  exit 0
fi

image=$(docker inspect -f '{{.Image}}' "$C")
caddy_run() {
  docker run --rm --network none -v "$1:/etc/caddy/Caddyfile:ro" "$image" "${@:2}" --config /etc/caddy/Caddyfile --adapter caddyfile
}
chmod 755 "$work"
chmod 644 "$cand"
caddy_run "$cand" caddy validate || { echo "FAIL: Caddy rejects the candidate; nothing changed"; exit 2; }
diff -u "$F" "$cand" || true

if [ "$mode" = check ]; then
  for which in live candidate; do
    src=$F
    [ "$which" = candidate ] && src=$cand
    caddy_run "$src" caddy adapt > "$work/$which.json" 2> "$work/adapt.log" \
      || { cat "$work/adapt.log"; echo "check: caddy adapt failed on the $which Caddyfile"; exit 2; }
  done
  if cmp -s "$work/live.json" "$work/candidate.json"; then
    echo "check: same config as the live Caddyfile; only comments or formatting differ"
    exit 0
  fi
  python3 -m json.tool "$work/live.json" > "$work/live.pretty.json"
  python3 -m json.tool "$work/candidate.json" > "$work/candidate.pretty.json"
  diff -u "$work/live.pretty.json" "$work/candidate.pretty.json" || true
  echo "check: applying this would change what Caddy runs (the adapted JSON differs above)"
  exit 1
fi

if [ "$mode" = ask ]; then
  read -rp "Apply this change and restart Caddy (every site it serves blips for a few seconds)? [y/N] " ok
  [ "$ok" = y ] || exit 1
fi

backup=$F.bak-$(date +%Y%m%d-%H%M%S)
cp -p "$F" "$backup"
inode=$(stat -c %i "$F")
echo "backup: $backup"

http_code() { local code; code=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' "$1" 2>/dev/null || true); echo "${code:-000}"; }
# HTTP status Caddy returns for Host $1, asked from inside its own container.
inside() {
  docker exec "$C" wget -q -S -O /dev/null --header "Host: $1" http://127.0.0.1/ 2>&1 \
    | grep -oE 'HTTP/1\.[01] [0-9]{3}' | tail -1 | cut -d' ' -f2 || true
}
wait_healthy() {
  for _ in $(seq 60); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' "$C")" = healthy ] && return 0
    sleep 2
  done
  return 1
}
restore() {
  echo "restoring $backup"
  cat "$backup" > "$F"
  docker restart "$C" >/dev/null
  wait_healthy || echo "WARN: Caddy is not healthy after restoring; check: docker logs $C"
  for url in "${other_sites[@]}"; do
    echo "$url after restore: $(http_code "$url")"
  done
  exit 1
}

cat "$cand" > "$F"
[ "$(stat -c %i "$F")" = "$inode" ] || { echo "FAIL: the Caddyfile's inode changed"; restore; }
docker exec "$C" cat /etc/caddy/Caddyfile | cmp -s - "$F" || { echo "FAIL: Caddy's container sees a different file"; restore; }

docker restart "$C" >/dev/null
wait_healthy || { echo "FAIL: Caddy did not become healthy"; restore; }

problems=()
for url in "${other_sites[@]}"; do
  code=$(http_code "$url")
  [[ $code =~ ^[23] ]] || problems+=("$url answered $code")
done
[ "$(inside "$SITE_HOST")" = 200 ] || problems+=("$SITE_HOST did not answer 200 inside Caddy")
[ "$(inside "$CHAT_HOST")" = 403 ] || problems+=("$CHAT_HOST without an Access header did not answer 403 inside Caddy")
if [ ${#problems[@]} -gt 0 ]; then
  printf 'FAIL: %s\n' "${problems[@]}"
  restore
fi
echo "Caddy updated: ${#other_sites[@]} other site(s) OK, $SITE_HOST serves the site, $CHAT_HOST refuses requests without an Access header"
