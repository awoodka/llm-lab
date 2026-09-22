# shellcheck shell=bash
# Sourced by the deploy scripts: where the app lives, and the settings in its .env files.
#
# On the web host the app directory holds .env (mode 600, see env.example), repo/ (this web/ tree, copied by
# push.sh), repo.prev/ and logs/. On the GPU host, push.sh reads deploy/local.env instead (see local.env.example).
# The files are read as data and never run as shell. A variable that is already set in the environment wins,
# as it does in docker compose's own interpolation, so the scripts and compose always agree.

APP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck disable=SC2034  # read by the scripts that source this file
ENV_FILE=$APP_DIR/.env

# env_value FILE NAME: the last NAME=value line of FILE, without one pair of surrounding quotes.
env_value() {
  local line
  line=$(grep -E "^$2=" "$1" 2>/dev/null | tail -n 1 || true)
  line=${line#*=}
  if [[ $line =~ ^\"(.*)\"$ || $line =~ ^\'(.*)\'$ ]]; then
    line=${BASH_REMATCH[1]}
  fi
  printf '%s' "$line"
}

# load_env FILE NAME...: set each NAME from the environment, else from FILE. Returns 1, naming what is
# missing, if any of them ends up empty.
load_env() {
  local file=$1 name missing=()
  shift
  for name in "$@"; do
    [ -n "${!name:-}" ] || printf -v "$name" '%s' "$(env_value "$file" "$name")"
    [ -n "${!name}" ] || missing+=("$name")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "missing from $file: ${missing[*]}" >&2
    return 1
  fi
}
