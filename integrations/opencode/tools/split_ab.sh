#!/usr/bin/env bash
# Split-K decode attention, on and off, through the real server.
#
# ## What this answers that the probe cannot
#
# `sdpa_paged_probe` checks the split kernel against a float64 reference and
# times it in isolation. Both are necessary and neither says anything about the
# WIRING, which is where this landing's risk actually is: the partials have to
# be bound at every layer's attention, the combine has to be emitted after every
# split, and `pso_for` and `launch_shape` have to agree about a grid that is now
# short in x and deep in z. Every one of those failures is silent -- a combine
# that did not run leaves the attention output holding whatever the activation
# pool last put there, which is not a crash and not a slowdown.
#
# So this runs the whole model both ways and compares the GENERATED TEXT at
# temperature 0, alongside the rate. Text is the end of the chain: if any of the
# wiring were wrong, the tokens would differ immediately and grossly.
#
# ## Why the two arms are the same binary
#
# `PIE_METAL_SDPA_SPLIT=0` removes the pipeline, both selection sites and the
# second dispatch, from one predicate. Same build, same config, same server --
# the only difference is the kernel. That is the same control `four_way.sh`
# uses for "original pie", and it is stronger than checking out a parent commit.
#
# ## Reading it
#
# Text identical  -> the wiring is right.
# Text divergent  -> read the two side by side before believing any rate here.
#                    bf16 rounding differs between a one-pass and a merged
#                    softmax, so a LATE divergence in a 200-token greedy sample
#                    is possible and not by itself a defect; an early or total
#                    one is.
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${SPLIT_AB_OUT:-/tmp/split-ab-$(date +%m%d-%H%M)}"
MODEL=qwen3-coder-30b
PIEPY="${PIEPY:-/Users/liuyang/.venvs/pie/bin/python}"
mkdir -p "$OUT"

# BUILD FIRST, and prove the binary is newer than the driver.
#
# The first complete run of this script produced 54.2/40.8/27.8 against
# 54.3/40.8/27.9 — a perfect 1.00x at every size, with byte-identical output
# text — and the reason was that `target/release/pie` predated the kernel by two
# hours. Both arms ran the same code, so of course they agreed. `cargo build -p
# pie` does not build it either: the binary crate is `pie-bin`, and the plain
# name silently fails to match a package.
#
# An A/B whose two arms are the same binary is the most convincing possible
# null result, which is exactly why this check is here and not in a comment.
( cd "$REPO" && cargo build --release -p pie-bin --features driver-metal ) || exit 1
NEWER=$(find "$REPO/driver/metal/src" -newer "$REPO/target/release/pie" -name '*.metal' \
        -o -newer "$REPO/target/release/pie" -name '*.cpp' \
        -o -newer "$REPO/target/release/pie" -name '*.hpp' 2>/dev/null | head -1)
if [ -n "$NEWER" ]; then
    echo "FATAL: driver source newer than the binary after a build: $NEWER" >&2
    exit 1
fi

cd "$REPO/integrations/opencode"

arm() {  # $1 label, $2 value of PIE_METAL_SDPA_SPLIT
    echo "── [$1] booting with PIE_METAL_SDPA_SPLIT=$2"
    # The strategy-B session shim imports the pie python client, which needs
    # msgpack; the system python3 does not have it and the boot dies with a
    # ModuleNotFoundError buried in a serve log rather than at the call site.
    tools/boot_pie.sh "$1" PIE_STRATEGY=b PIE_MODEL="$MODEL" \
        PIE_PYTHON="$PIEPY" \
        PIE_MAX_MODEL_LEN=65536 PIE_MAX_FORWARD_TOKENS=4096 \
        PIE_METAL_SDPA_SPLIT="$2" || return 1
    # More than the default three sizes. Split-K read 1.02x / 0.94x / 1.09x
    # across those, and a dip between two gains cannot be told from a curve
    # without points either side of it.
    python3 tools/rate_probe.py --base-url http://127.0.0.1:8080 --model "$MODEL" \
        --blocks "${SPLIT_AB_BLOCKS:-150,250,325,400,475,550,700}" \
        --label "$1" --json "$OUT/rate-$1.json"
    pkill -f "$REPO/target/release/pie .*serve" 2>/dev/null
    pkill -f session_shim.py 2>/dev/null
    sleep 6
}

# THE FIRST ARM IS REPEATED LAST, and it is not optional here.
#
# The first run of this script had split-on at TTFT 0.39s / 0.54s and split-off
# at 2.87s / 7.87s on the same prompts -- a 5-15x gap that split-K CANNOT cause,
# because it is a decode-only kernel and TTFT is prefill. Something carried
# across the boot and landed entirely in the first arm. Whatever it is, an
# ordering effect that large on one measurement is a reason to distrust the
# other, and the repeat is what tells the two apart: if split-on(first) and
# split-on(last) agree, the middle arm is comparable to both; if they do not,
# this run measured the order.
arm split-on 1 || exit 1
arm split-off 0 || exit 1
arm split-on-repeat 1 || exit 1

python3 - "$OUT" <<'PY'
import json, sys, difflib
out = sys.argv[1]
on  = json.load(open(f"{out}/rate-split-on.json"))
off = json.load(open(f"{out}/rate-split-off.json"))
rep = json.load(open(f"{out}/rate-split-on-repeat.json"))

# The drift control, read BEFORE anything else. Two runs of the same arm that
# disagree mean this session measured its own ordering, and the on-vs-off
# comparison below is then a number about the schedule rather than the kernel.
print(f"\n{'prompt':>8}  {'on(1st)':>9} {'on(last)':>9}  drift")
drift_ok = True
for a, b in zip(on, rep):
    if "error" in a or "error" in b:
        print(f"{a.get('blocks'):>8}  arm errored"); continue
    d = abs(b["decode_tok_s"] - a["decode_tok_s"]) / a["decode_tok_s"]
    if d > 0.03:
        drift_ok = False
    print(f"{a['prompt_tokens']:>8}  {a['decode_tok_s']:>9} {b['decode_tok_s']:>9}"
          f"  {d*100:>5.1f}%{'  <-- ORDER EFFECT' if d > 0.03 else ''}")
if not drift_ok:
    print("\n!! The same arm disagrees with itself by more than 3%. The table below\n"
          "   is NOT a measurement of the kernel. Interleave and re-run.")
print(f"\n{'prompt':>8}  {'off tok/s':>10} {'on tok/s':>10} {'speedup':>8}   text")
same = True
for a, b in zip(off, on):
    if "error" in a or "error" in b:
        print(f"  ERROR arm: {a.get('error') or b.get('error')}"); same = False; continue
    sp = b["decode_tok_s"] / a["decode_tok_s"] if a["decode_tok_s"] else 0
    ta, tb = a.get("text", ""), b.get("text", "")
    if ta == tb:
        verdict = "IDENTICAL"
    else:
        same = False
        # Where they part matters: bf16 rounding can push a greedy sample onto a
        # different token late in a 200-token sample, and that is not the same
        # finding as garbage from the first byte.
        i = next((k for k, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        verdict = f"DIVERGES at char {i} of {min(len(ta), len(tb))}"
    print(f"{a['prompt_tokens']:>8}  {a['decode_tok_s']:>10} {b['decode_tok_s']:>10} "
          f"{sp:>7.2f}x   {verdict}")
    if ta != tb:
        d = list(difflib.unified_diff(ta.splitlines(), tb.splitlines(),
                                      "split-off", "split-on", lineterm="", n=1))
        print("\n".join("      " + x for x in d[:24]))
print("\n" + ("Every arm produced the SAME text: the wiring is right."
              if same else
              "TEXT DIFFERS. Read the diff above before quoting any rate here."))
PY
echo "results in $OUT"
