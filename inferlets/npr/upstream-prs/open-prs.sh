#!/usr/bin/env bash
# Opens the five upstream PRs. Run ONLY after the manager's go-ahead.
# `gh pr create` cannot be used: the fine-grained PAT on this host cannot run
# GraphQL mutations ("Resource not accessible"). The REST endpoint works.
set -euo pipefail
cd "$(dirname "$0")"

open_pr() {  # $1 = branch, $2 = title, $3 = body file
  gh api -X POST repos/pie-project/pie/pulls \
    -f title="$2" \
    -f head="YangLiuWillow:$1" \
    -f base=main \
    -F body=@"$3" \
    --jq '"\(.number)\t\(.html_url)"'
}

open_pr fix/sdk-context-destroy-trap \
  "fix(sdk): prevent trap in Context::destroy from double resource release" \
  1-sdk-destroy.md

open_pr fix/portable-kv-write-index \
  "fix(driver/portable): KV write index from slot order, not position id" \
  2-portable-kv-write-index.md

open_pr fix/runtime-explicit-mask-commit \
  "fix(runtime): allow non-monotonic committed positions for explicit-mask tokens" \
  3-runtime-commit-check.md

open_pr fix/cuda-prefill-logits-oob \
  "fix(driver/cuda): skip logits tail for prefill-only batches" \
  4-cuda-prefill-oob.md

open_pr fix/portable-graph-cache-slot-count \
  "fix(driver/portable): key the graph cache on the sampling-slot count" \
  5-portable-graph-cache-slots.md
