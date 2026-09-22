#!/usr/bin/env bash
# Run on the GPU host: replace <app dir>/repo on the web host with this web/ tree (the previous tree stays in
# repo.prev for rollback), then run deploy.sh there. deploy/local.env (gitignored; see local.env.example) says how
# to reach the web host (WEB_SSH) and where the app lives there (WEB_APP_DIR). Secrets stay in the app's .env there.
set -Eeuo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/deploy/env.sh"
load_env "$here/deploy/local.env" WEB_SSH WEB_APP_DIR \
  || { echo "copy deploy/local.env.example to deploy/local.env and fill it in" >&2; exit 1; }
read -ra ssh_cmd <<< "$WEB_SSH"

tar -C "$here" --exclude=node_modules --exclude=data --exclude=backups --exclude=local.env -cz . \
  | "${ssh_cmd[@]}" "set -e
      cd $(printf %q "$WEB_APP_DIR")
      rm -rf repo.new && mkdir repo.new && tar -C repo.new -xz
      rm -rf repo.prev
      if [ -d repo ]; then mv repo repo.prev; fi
      mv repo.new repo
      exec repo/deploy/deploy.sh"
