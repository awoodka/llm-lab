#!/usr/bin/env bash
# Run on ai: replace /opt/llmlab/repo on `web` with this web/ tree (the previous tree stays in repo.prev for
# rollback), then run deploy.sh there. Needs Tailscale SSH to `web`. Secrets stay in /opt/llmlab/.env on `web`.
set -Eeuo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)

tar -C "$here" --exclude=node_modules --exclude=data --exclude=backups -cz . \
  | tailscale ssh alex@web 'set -e
      cd /opt/llmlab
      rm -rf repo.new && mkdir repo.new && tar -C repo.new -xz
      rm -rf repo.prev
      if [ -d repo ]; then mv repo repo.prev; fi
      mv repo.new repo
      exec repo/deploy/deploy.sh'
