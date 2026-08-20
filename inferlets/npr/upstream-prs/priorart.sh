#!/usr/bin/env bash
# For one file, group all upstream PR heads by that file's blob, and report each
# distinct version that differs from upstream/main. Loop form on purpose: the
# multi-rev `git grep` form silently returns nothing here and failed its own
# positive control.
set -u
path="$1"
main_blob=$(git rev-parse "upstream/main:$path" 2>/dev/null || echo NONE)
echo "### $path   (main blob ${main_blob:0:9})"
for r in $(git for-each-ref --format='%(refname:short)' refs/remotes/upstream-pr/); do
  b=$(git rev-parse "$r:$path" 2>/dev/null) || continue
  [ "$b" = "$main_blob" ] && continue
  echo "$b ${r#upstream-pr/}"
done | sort | awk '{ if ($1!=prev) { if (prev!="") print "  blob " substr(prev,1,9) " -> PRs " list; prev=$1; list=$2 } else list=list","$2 } END { if (prev!="") print "  blob " substr(prev,1,9) " -> PRs " list }'
