#!/usr/bin/env bash
# One decode-attention switch, on and off, priced as a DECODE FIRE.
#
# `SPLIT_FIRE_VAR` picks which switch; it defaults to
# `PIE_METAL_SDPA_SPLIT` and `PIE_METAL_SDPA_UNROLL` is the other one.
#
# ## Why this and not the rate probe
#
# `tools/split_ab.sh` measures tokens/second through the HTTP server, and that
# instrument's own scatter is larger than the effect. On one arm, decode rate
# read 55.2 / 42.6 / 47.0 / 39.8 / 37.4 / 34.2 tok/s at 5.8k / 9.9k / 13.0k /
# 16.1k / 19.2k / 22.2k -- the 13k point is FASTER than the 9.9k one, which a
# decode cannot be. About 10% of scatter, against an effect of about 4%. A
# three-point A/B on that instrument produced 1.02x / 0.94x / 1.09x, which is
# noise wearing the shape of a finding.
#
# `decode-rows-probe` fires the real driver directly: fixed context, fixed row
# count, ten fires per configuration with the first two discarded, no HTTP, no
# session shim, no prefix cache, no tokenizer. It is the instrument this
# question deserves.
#
# ## What is compared
#
# `rows=1` at ctx 7424 -- a decode step, the shape an agentic turn is made of.
# The arms are the same binary under `PIE_METAL_SDPA_SPLIT`, and they are
# INTERLEAVED rather than run in sequence, because this machine's clocks ramp
# under load and a block of one arm followed by a block of the other puts the
# ramp entirely inside the second.
set -uo pipefail

REPO="${PIE_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="${SPLIT_FIRE_OUT:-/tmp/split-fire-$(date +%m%d-%H%M)}"
CONF="${ROWS_PROBE_CONF:-/tmp/rows-probe/config.toml}"
REPS="${SPLIT_FIRE_REPS:-3}"
mkdir -p "$OUT"
cd "$REPO"

# Same guard as `split_ab.sh`, for the same reason: an A/B whose two arms are
# the same binary is the most convincing possible null result.
cargo build --release -p pie-bin --features driver-metal || exit 1

WASM=runtime/engine/tests/inferlets/target/wasm32-wasip2/release/decode_rows_probe.wasm
MANIFEST=runtime/engine/tests/inferlets/decode-rows-probe/Pie.toml
if [ ! -f "$WASM" ]; then
    echo "FATAL: $WASM missing; build it with:" >&2
    echo "  (cd runtime/engine/tests/inferlets && cargo build --target wasm32-wasip2 --release -p decode-rows-probe)" >&2
    exit 1
fi

run_one() {  # $1 label, $2 split setting, $3 rep
    # Let the previous run's 16 GiB of weights actually go back.
    #
    # Back to back, the second invocation reports "only 12.02 GiB is
    # reclaimable" and refuses to load a 22.48 GiB model, while `vm_stat` says
    # 22.7 GiB free and 2.4 GiB wired. The pages are released but not yet
    # reclaimed, and the driver's own admission check is what notices. Without
    # this the FIRST arm always loads and the second always fails, which is a
    # per-arm bias, not just a flake.
    sleep "${SPLIT_FIRE_SETTLE:-25}"
    # `env` and not `VAR=val cmd`: the variable NAME is itself a variable here,
    # and bash does not re-parse an expansion as an assignment -- it would try to
    # execute a command called `PIE_METAL_SDPA_SPLIT=1`.
    env "${SPLIT_FIRE_VAR:-PIE_METAL_SDPA_SPLIT}=$2" ./target/release/pie -c "$CONF" run \
        --path "$WASM" --manifest "$MANIFEST" \
        > "$OUT/$1-$3.txt" 2>&1
    # The probe prints `[rows] ctx=N rows=K median_ms=X samples_ms=[...]`, one
    # line per configuration, interleaved with the engine's RPC logging. rows=1
    # is the decode step. Matched on the whole `rows=1 ` token so `rows=128`
    # cannot satisfy it.
    local line
    # BOTH contexts the probe fires, not just the first. `decode-rows-probe`
    # has SHORT and LONG for exactly this, and a switch gated on context inside
    # the kernel has to be priced on both sides of its gate -- one number cannot
    # show "no regression below" and "a win above" at once.
    line=$(grep -o "\[rows\] ctx=[0-9]* rows=1 median_ms=[0-9.]*" "$OUT/$1-$3.txt" \
           | sort -u | tr '\n' ' ')
    # A RUN THAT PRODUCED NOTHING IS FATAL, not blank.
    #
    # The first attempt at this A/B had both `off` runs die on model load --
    # 24.8 GiB was wired by servers an earlier measurement had left behind, so
    # the weights no longer fit -- while both `on` runs succeeded. The output
    # was two timings, both from the same arm, under headings that said
    # otherwise. Half an A/B looks exactly like an A/B.
    if [ -z "$line" ]; then
        echo "FATAL: $1 rep $3 produced no rows=1 timing. Tail of its log:" >&2
        tail -4 "$OUT/$1-$3.txt" >&2
        exit 1
    fi
    echo "$line"
}

# Nothing else may be holding the GPU. The model needs ~22.5 GiB resident and
# this machine has 48; one forgotten server is enough to make the second arm
# fail to load while the first succeeds.
if pgrep -f "$REPO/target/release/pie .*serve" >/dev/null; then
    echo "FATAL: a pie server is running; it will starve this probe of memory" >&2
    exit 1
fi

for r in $(seq 1 "$REPS"); do
    echo "── rep $r"
    echo -n "   split ON  : "; run_one on  1 "$r"
    echo -n "   split OFF : "; run_one off 0 "$r"
done

echo
echo "raw output in $OUT (read it if the lines above are empty -- the probe's"
echo "table format is the only thing the awk above depends on)"
