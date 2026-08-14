#!/usr/bin/env bash
# Fetch TTB's swe-bench-lite-first-20 dataset snapshot.
#
# Fetched rather than vendored: the snapshot lives in the private
# `shsym/test-time-bench` and ships its own LICENSE-DATA. Copying private
# benchmark data into this tree would relicense it by accident and duplicate a
# thing that has an owner.
#
# The snapshot is the SOURCE OF TRUTH for the prompt, not just the case list:
# each case carries a fully rendered `input`, so a prompt change is a dataset
# version change. That is the property `run_swebench.py`'s own PROMPT constant
# lacks — it can be edited without any version moving.
#
#   GH_TOKEN=... ./fetch_dataset.sh [outfile]
# or leave GH_TOKEN unset and it reads ~/Documents/Liszt_ai/.env
set -euo pipefail
OUT="${1:-$(cd "$(dirname "$0")" && pwd)/dataset.swe-bench-lite-first-20.json}"
REF="${TTB_REF:-main}"

if [ -z "${GH_TOKEN:-}" ] && [ -f "$HOME/Documents/Liszt_ai/.env" ]; then
    set -a; . "$HOME/Documents/Liszt_ai/.env"; set +a
fi
[ -n "${GH_TOKEN:-}" ] || { echo "FATAL: GH_TOKEN unset and no .env to read it from" >&2; exit 1; }

url="https://api.github.com/repos/shsym/test-time-bench/contents/benchmarks/swe-bench-lite-first-20/dataset.json?ref=$REF"
code=$(curl -s -o "$OUT.tmp" -w '%{http_code}' \
    -H "Authorization: Bearer $GH_TOKEN" -H "Accept: application/vnd.github.raw" "$url")
if [ "$code" != "200" ]; then
    echo "FATAL: HTTP $code fetching the snapshot (token lacks access to the private repo?)" >&2
    rm -f "$OUT.tmp"; exit 1
fi

# Verify it is the snapshot we think it is before overwriting anything.
python3 - "$OUT.tmp" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("schema") == "test-time-bench/dataset-snapshot.v2", f"unexpected schema: {d.get('schema')}"
cases = d["cases"]
assert cases and all("id" in c and "input" in c and "grading" in c for c in cases), "malformed cases"
print(f"snapshot ok: {len(cases)} cases, schema {d['schema']}")
PY
mv "$OUT.tmp" "$OUT"
echo "sha256  $(shasum -a 256 "$OUT" | cut -d' ' -f1)"
echo "wrote   $OUT"
echo "NOTE: record that sha256 in any run summary produced from it — the snapshot is private and can move."
