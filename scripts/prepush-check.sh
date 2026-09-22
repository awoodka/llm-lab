#!/usr/bin/env bash
# Run before every push: search the WHOLE history (every file of every commit reachable from any ref, and every
# commit message) for text that must never be public. lab/tests/test_repo_hygiene.py covers the current tree with
# patterns; this covers history with the exact literals, which are themselves private, so they live outside the
# repository, one fixed string per line:
#   ~/.config/llm-lab/redact-literals.txt   real hostnames, addresses, paths and names of the lab's infrastructure
#   ~/.config/llm-lab/gpqa-fragments.txt    a fragment of every GPQA question: its authors ask that the questions
#                                           stay off the web, so no transcript may ever be committed
# Both must be mode 600 (LLM_LAB_PRIVATE_DIR overrides the directory). Only commit and file names are printed,
# never the matching text. Exit 0 clean, 1 found something, 2 could not check.
set -Euo pipefail

dir=${LLM_LAB_PRIVATE_DIR:-$HOME/.config/llm-lab}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel) || exit 2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git -C "$repo" rev-list --all > "$work/revs" || exit 2
[ -s "$work/revs" ] || { echo "no commits to check"; exit 2; }
echo "checking $(wc -l < "$work/revs") commits reachable from: $(git -C "$repo" for-each-ref --format='%(refname)' | paste -sd' ')"

status=0
for name in redact-literals gpqa-fragments; do
  list=$dir/$name.txt
  if [ ! -s "$list" ]; then echo "missing or empty: $list"; exit 2; fi
  if [ "$(stat -c %a "$list")" != 600 ]; then echo "$list must be mode 600"; exit 2; fi
  grep -v '^[[:space:]]*$' "$list" > "$work/$name"   # a blank line would match everything

  # Each file of each commit, as commit:path. xargs splits the list, so the arguments never get too long; the
  # commits go last and there is no `--`, so git takes them as trees to search, not as paths.
  xargs -a "$work/revs" git -C "$repo" grep -I -l -F -f "$work/$name" > "$work/$name.files" 2> "$work/grep.err"
  if [ -s "$work/grep.err" ]; then cat "$work/grep.err"; echo "git grep failed"; exit 2; fi
  n_files=$(wc -l < "$work/$name.files")
  n_msgs=$(git -C "$repo" log --all --format=%B | grep -c -F -f "$work/$name" || true)
  n_diffs=$(git -C "$repo" log --all -p --format= | grep -c -F -f "$work/$name" || true)

  if [ "$n_files" -eq 0 ] && [ "$n_msgs" -eq 0 ] && [ "$n_diffs" -eq 0 ]; then
    echo "ok: no $name in any commit's files, messages or diffs"
  else
    status=1
    echo "FAIL: $name found in $n_files commit:file pairs, $n_msgs message lines and $n_diffs diff lines"
    cut -d: -f2- "$work/$name.files" | sort | uniq -c | sort -rn | head -20 | sed 's/^/  /'
  fi
done
exit $status
