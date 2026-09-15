#!/usr/bin/env bash
# Install or update the LLM Lab blocks in the shared edge Caddyfile, which also serves the other sites.
#   1. build a candidate with Caddyfile.llmlab between the `# >>> llmlab` / `# <<< llmlab` markers
#   2. validate it in a throwaway container from the running Caddy image
#   3. write it in place: the Caddyfile is a single-file bind mount, so replacing its inode (editors, sed -i, mv)
#      would leave Caddy silently on the old file
#   4. restart Caddy (`admin off` rules out a hot reload), check every site, and restore the backup on any failure
# Run on web as alex: /opt/llmlab/repo/deploy/apply-caddy.sh [--yes]
set -Eeuo pipefail

F=/opt/edge/Caddyfile
C=edge-caddy-1
BLOCK=$(cd "$(dirname "$0")" && pwd)/Caddyfile.llmlab
yes=false
[ "${1:-}" = --yes ] && yes=true

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cand=$work/Caddyfile

python3 - "$F" "$BLOCK" "$cand" <<'EOF'
import sys

live, block, out = sys.argv[1:]
src = open(live).read()
new = open(block).read().rstrip("\n") + "\n"
start, end = "# >>> llmlab", "# <<< llmlab"
if start in src:
    i = src.index(start)
    j = src.index(end, i)
    j = src.find("\n", j) + 1 or len(src)
    result = src[:i] + new + src[j:]
else:
    anchor = next((a for a in ("# Anything cloudflared forwards", ":80 {") if a in src), None)
    if anchor is None:
        sys.exit("no insertion point: found neither the llmlab markers nor the :80 fallback block")
    i = src.index(anchor)
    result = src[:i] + new + "\n" + src[i:]
open(out, "w").write(result)
EOF

if cmp -s "$F" "$cand"; then
  echo "Caddyfile already up to date"
  exit 0
fi

chmod 755 "$work"
chmod 644 "$cand"
docker run --rm --network none -v "$cand:/etc/caddy/Caddyfile:ro" "$(docker inspect -f '{{.Image}}' "$C")" \
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
diff -u "$F" "$cand" || true
if ! $yes; then
  read -rp "Apply this change and restart Caddy (the other sites blip for a few seconds)? [y/N] " ok
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
  echo "other-app after restore: $(http_code https://other-app.example.com/)"
  exit 1
}

cat "$cand" > "$F"
[ "$(stat -c %i "$F")" = "$inode" ] || { echo "FAIL: the Caddyfile's inode changed"; restore; }
docker exec "$C" cat /etc/caddy/Caddyfile | cmp -s - "$F" || { echo "FAIL: Caddy's container sees a different file"; restore; }

docker restart "$C" >/dev/null
wait_healthy || { echo "FAIL: Caddy did not become healthy"; restore; }

problems=()
for url in https://other-app.example.com/ https://other-app.example.com/; do
  code=$(http_code "$url")
  [[ $code =~ ^[23] ]] || problems+=("$url answered $code")
done
[ "$(inside localinference.alexwoodka.com)" = 200 ] || problems+=("localinference.alexwoodka.com did not answer 200 inside Caddy")
[ "$(inside chat.alexwoodka.com)" = 403 ] || problems+=("chat.alexwoodka.com without an Access header did not answer 403 inside Caddy")
if [ ${#problems[@]} -gt 0 ]; then
  printf 'FAIL: %s\n' "${problems[@]}"
  restore
fi
echo "Caddy updated: the other sites OK, localinference serves the site, chat refuses requests without an Access header"
