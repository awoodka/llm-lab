#!/usr/bin/env bash
# Prove the public side of LLM Lab is safe. Run by deploy.sh, and every 10 minutes from alex's crontab on web:
#   */10 * * * * /opt/llmlab/repo/deploy/check-public.sh --chat-only >> /opt/llmlab/logs/public-check.log 2>&1
#
# chat.alexwoodka.com: unauthenticated requests, even with a forged Access header, must be redirected to Cloudflare
#   Access. Any other answer (including a 5xx), seen twice in a row, means Open WebUI, which has no login, may be
#   reachable, so it is disconnected from Caddy's network. A later run that sees Access working again reconnects it,
#   but only if this script was the one that disconnected it. If public DNS says the hostname doesn't exist, the chat
#   has been switched off, and that's ok.
# localinference.alexwoodka.com, once Caddy routes it: /api must answer 404 and the homepage must not mention a
#   tailnet address. Otherwise the site is disconnected from Caddy's network (deploy.sh reattaches it).
# Names are resolved with Cloudflare's DNS-over-HTTPS (1.1.1.1), so web's own resolver trouble can't skip a check.
# Exit: 0 ok, 1 something was exposed (and cut off), 75 a check couldn't finish (lookup or network failure, site 5xx).
# Usage: check-public.sh [--chat-only] [--dry-run]    (--dry-run reports without disconnecting or reconnecting)
set -Euo pipefail

CHAT=chat.alexwoodka.com
SITE=localinference.alexwoodka.com
DOH=${DOH:-https://1.1.1.1/dns-query}
ACCESS_REDIRECT='^30[1237] https://[a-z0-9-]+\.cloudflareaccess\.com/'
RETRY_AFTER_S=${RETRY_AFTER_S:-30}
CUT_MARK=${CUT_MARK:-/opt/llmlab/logs/open-webui-cut-off}

chat_only=false
dry_run=false
for arg in "$@"; do
  case $arg in
    --chat-only) chat_only=true ;;
    --dry-run) dry_run=true ;;
    *) echo "usage: $0 [--chat-only] [--dry-run]" >&2; exit 2 ;;
  esac
done

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S%z') $*"; }
answer() { curl -sS -m 15 --doh-url "$DOH" -o /dev/null -w '%{http_code} %{redirect_url}' "$@" 2>/dev/null || true; }
# Public DNS status of $1 over DoH: 0 exists, 3 doesn't exist, empty or anything else means we couldn't tell.
dns_status() {
  curl -sS -m 10 -H 'accept: application/dns-json' "$DOH?name=$1&type=A" 2>/dev/null \
    | grep -oE '"Status": ?[0-9]+' | grep -oE '[0-9]+$' || true
}
on_edge() { docker network inspect edge --format '{{range .Containers}}{{.Name}} {{end}}' | tr ' ' '\n' | grep -qx "$1"; }
cut_off() {
  if $dry_run; then
    log "dry run: would disconnect $1 from edge"
  elif docker network disconnect edge "$1" 2>/dev/null; then
    log "disconnected $1 from edge"
  else
    log "$1 was not on edge"
  fi
}

# One pass over the chat: fills bad[] with answers that aren't an Access redirect, and sets unreachable on network errors.
chat_round() {
  bad=()
  unreachable=false
  for path in / /api/config /api/models '/ws/socket.io/?EIO=4&transport=polling'; do
    got=$(answer "https://$CHAT$path")
    if [[ $got == 000* ]]; then
      unreachable=true
    elif [[ ! $got =~ $ACCESS_REDIRECT ]]; then
      bad+=("$path answered '$got'")
    fi
  done
  got=$(answer -H 'Cf-Access-Jwt-Assertion: forged' -b 'CF_Authorization=forged' "https://$CHAT/")
  if [[ $got == 000* ]]; then
    unreachable=true
  elif [[ ! $got =~ $ACCESS_REDIRECT && ! $got =~ ^403 ]]; then
    bad+=("/ with a forged Access header answered '$got'")
  fi
}

status=0

# -- chat ---------------------------------------------------------------------------------------------------
dns=$(dns_status "$CHAT")
if [ "$dns" = 3 ]; then
  log "ok: $CHAT does not exist in public DNS (chat switched off)"
elif [ "$dns" != 0 ]; then
  log "WARN could not look up $CHAT over DNS-over-HTTPS; nothing checked, nothing changed"
  status=75
else
  chat_round
  if [ ${#bad[@]} -gt 0 ]; then
    log "unexpected answers from $CHAT; checking again in ${RETRY_AFTER_S}s"
    sleep "$RETRY_AFTER_S"
    chat_round
  fi
  if [ ${#bad[@]} -gt 0 ]; then
    for b in "${bad[@]}"; do log "FAIL $CHAT$b; expected a Cloudflare Access redirect"; done
    cut_off llmlab-open-webui-1
    $dry_run || touch "$CUT_MARK"
    exit 1
  elif $unreachable; then
    log "WARN $CHAT was unreachable for some checks; nothing changed"
    status=75
  else
    log "ok: $CHAT requires Cloudflare Access"
    if [ -e "$CUT_MARK" ] && ! on_edge llmlab-open-webui-1; then
      if $dry_run; then
        log "dry run: would reconnect llmlab-open-webui-1 to edge"
      elif docker network connect --alias llmlab-open-webui edge llmlab-open-webui-1; then
        rm -f "$CUT_MARK"
        log "reconnected llmlab-open-webui-1 to edge: Access is enforced again"
      fi
    fi
  fi
fi

$chat_only && exit $status

# -- site ---------------------------------------------------------------------------------------------------
if ! grep -q '# >>> llmlab' /opt/edge/Caddyfile 2>/dev/null; then
  log "ok: $SITE is not routed by Caddy yet"
  exit $status
fi

bad=()
for method in GET POST PUT; do
  got=$(answer -X "$method" -H 'Authorization: Bearer bogus' "https://$SITE/api/ingest")
  code=${got%% *}
  case $code in
    404) ;;
    000 | 5??) log "WARN $method $SITE/api/ingest answered $code"; status=75 ;;
    *) bad+=("$method /api/ingest answered $code; expected 404") ;;
  esac
done
home=$(curl -sS -m 15 --doh-url "$DOH" "https://$SITE/" 2>/dev/null || true)
[[ $home == *ts.net* ]] && bad+=("the homepage mentions a tailnet address")
[[ $home == *'<h1>Local Inference</h1>'* ]] || { log "WARN $SITE homepage did not render"; status=75; }

if [ ${#bad[@]} -gt 0 ]; then
  for b in "${bad[@]}"; do log "FAIL $SITE: $b"; done
  cut_off llmlab-site-1
  exit 1
fi
[ "$status" = 0 ] && log "ok: $SITE serves pages and no API"
exit $status
