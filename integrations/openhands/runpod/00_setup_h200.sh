#!/usr/bin/env bash
# =============================================================================
# Bring-up on a FRESH RunPod H200 SXM box.
#
# Replaces 00_setup_a100.sh. Read AGENT_HANDOVER_H200.md first — it explains
# WHY this pod is an H200 and what the A100 run established.
#
# The one-line reason: Pie's fast attention paths are gated to compute
# capability >= 9. The A100 is sm_80 and failed both gates silently, so the
# A100 Pie arm measured Pie's fallback path. H200 is sm_90 and passes. Section
# [8] below VERIFIES that at runtime and is the most important step here.
#
# Bonus: vLLM SHIPS a tuned bf16 MoE config for this exact shape on H200
# (E=128,N=768,device_name=NVIDIA_H200.json), so there is NO autotune to run
# and no VLLM_TUNED_CONFIG_FOLDER to export. Section [6] verifies it exists.
#
# Idempotent: safe to re-run. Skips anything already done.
# =============================================================================
set -euo pipefail

say()  { printf '\n=== %s ===\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

# ---- paths ------------------------------------------------------------------
# THE STORAGE RULE, and it is not a preference:
#   /workspace is a MooseFS NETWORK volume. Small-file-heavy work there does not
#   run slow, it STALLS INDEFINITELY with no error. venvs, caches and the cargo
#   build tree MUST live on local disk (/root). Large sequential files (repo,
#   model weights, predictions, logs) are fine on /workspace.
export WORK=${WORK:-/workspace}
export FAST=${FAST:-/root}
export PIE_SRC=${PIE_SRC:-$WORK/pie}
export RUNPOD_DIR=$PIE_SRC/integrations/openhands/runpod
export HARNESS_DIR=$PIE_SRC/integrations/openhands
export PIE_VENV=${PIE_VENV:-$FAST/venvs/pie-vllm}
export HARNESS_VENV=${HARNESS_VENV:-$FAST/venvs/harness}
export HF_HOME=${HF_HOME:-$WORK/.cache/huggingface/}
export MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-$FAST/.cpm-cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$FAST/.uv-cache}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$WORK/.cache/pip/}
export BUILD_TREE=${BUILD_TREE:-$FAST/pie-target}
export ENV_FILE=${ENV_FILE:-$WORK/pie-bench-env.sh}

# Version pins. Each exists because the unpinned version broke — see
# AGENT_HANDOVER_H200.md §6. Do not "upgrade" these mid-experiment.
PY=${PY:-python3.12}
VLLM_VERSION=${VLLM_VERSION:-0.25.1}
HARNESS_EXCLUDE_NEWER=${HARNESS_EXCLUDE_NEWER:-2026-05-15}

mkdir -p "$FAST/venvs" "$CPM_SOURCE_CACHE" "$UV_CACHE_DIR" "$BUILD_TREE" "$PIP_CACHE_DIR"

# =============================================================================
say "[0] preflight — is this actually an H200?"
# =============================================================================
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU pod?"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
ARCH=${CC/./}
MAJOR=${CC%%.*}
[ -n "$ARCH" ] || die "could not read compute capability"

if [ "$MAJOR" -lt 9 ]; then
    die "compute cap $CC (major $MAJOR) is BELOW 9. Pie's fast attention paths
       (entry.cpp:989, attention_xqa.cu:274) will be OFF and the Pie arm will be
       invalid — this is exactly the A100 failure. Get an sm_90+ pod."
fi
echo "  compute cap $CC (sm_$ARCH, major $MAJOR) — Pie's >= 9 gates will PASS."
[ "$MAJOR" -ge 12 ] || echo "  note: major < 12, so the wide-prefill gates
  (cuda_memory_planner.cpp:219,:250) stay OFF. Expected on H200. State it."

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$GPU_MEM" -ge 130000 ] || warn "GPU has ${GPU_MEM} MiB — the H200 toml assumes ~141 GB."

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  device name: $GPU_NAME"

if command -v mountpoint >/dev/null && ! mountpoint -q "$WORK"; then
    warn "$WORK is not a separate mount — it may be ephemeral container disk."
fi
df -h "$WORK" "$FAST" | sed 's/^/  /'

# =============================================================================
say "[1] what survived the pod move"
# =============================================================================
# /root is LOCAL to the pod and is GONE. /workspace persists only if you
# re-attached the same network volume.
[ -d "$PIE_SRC/.git" ] || die "no repo at $PIE_SRC. Either the volume did not
       re-attach, or clone it:
         git clone -b openhands-integration-updated \\
             https://github.com/YangLiuWillow/pie.git $PIE_SRC"
echo "  repo:  $PIE_SRC ($(git -C "$PIE_SRC" rev-parse --abbrev-ref HEAD) @ $(git -C "$PIE_SRC" rev-parse --short HEAD))"

MODEL_DIR="$HF_HOME/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct"
if [ -d "$MODEL_DIR" ]; then
    echo "  model: present ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1)) — no re-download"
    SKIP_MODEL=${SKIP_MODEL:-1}
else
    warn "model NOT found at $MODEL_DIR — will download ~60 GB in section [7]."
    SKIP_MODEL=${SKIP_MODEL:-0}
fi

# =============================================================================
say "[2] toolchain"
# =============================================================================
command -v nvcc >/dev/null || warn "nvcc not on PATH — install a CUDA toolkit matching the driver."
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
rustup target add wasm32-wasip2 >/dev/null 2>&1 || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
command -v "$PY" >/dev/null || die "$PY not found. Python 3.12 is REQUIRED: on
       3.13 the vLLM set resolves and installs, then torch dies at import in the
       TorchScript overload parser. A successful resolve is not evidence the
       stack runs."
echo "  $($PY --version), cargo $(cargo --version | cut -d' ' -f2), uv $(uv --version 2>/dev/null | cut -d' ' -f2)"

# =============================================================================
say "[3] env file -> $ENV_FILE  (sm_$ARCH)"
# =============================================================================
cat > "$ENV_FILE" <<EOF
# generated by 00_setup_h200.sh — source before every run:
#   source $ENV_FILE
export WORK=$WORK
export FAST=$FAST
export PIE_SRC=$PIE_SRC
export RUNPOD_DIR=$RUNPOD_DIR
export HARNESS_DIR=$HARNESS_DIR
export PIE_VENV=$PIE_VENV
export HARNESS_VENV=$HARNESS_VENV
export HF_HOME=$HF_HOME
export MODEL=$MODEL
export PIE_BIN=$PIE_SRC/target/release/pie
export CPM_SOURCE_CACHE=$CPM_SOURCE_CACHE
export PIP_CACHE_DIR=$PIP_CACHE_DIR
export UV_CACHE_DIR=$UV_CACHE_DIR
export UV_HTTP_TIMEOUT=900
export PIP_DEFAULT_TIMEOUT=900
export UV_CONCURRENT_DOWNLOADS=4
export CMAKE_CUDA_ARCHITECTURES=$ARCH
export PIE_PORTABLE_CUDA_ARCH=$ARCH

# H200 ships its own tuned MoE config — do NOT export VLLM_TUNED_CONFIG_FOLDER.
# See AGENT_HANDOVER_H200.md §4.

export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
[ -f "\$HOME/.cargo/env" ] && . "\$HOME/.cargo/env"
EOF
echo "  wrote $ENV_FILE (CMAKE_CUDA_ARCHITECTURES=$ARCH)"

# =============================================================================
say "[4] venvs on LOCAL disk"
# =============================================================================
# uv streams downloads into its cache BEFORE linking into the venv, so a cache
# on MooseFS stalls the install no matter where the venv lives. UV_CACHE_DIR is
# already pointed at $FAST above. This cost the human several hours once.
if [ ! -x "$PIE_VENV/bin/python" ]; then
    echo "  creating vLLM venv ($PY, vllm==$VLLM_VERSION)"
    "$PY" -m venv "$PIE_VENV"
    VIRTUAL_ENV="$PIE_VENV" uv pip install -U pip
    VIRTUAL_ENV="$PIE_VENV" uv pip install "vllm==$VLLM_VERSION"
else
    echo "  vLLM venv exists"
fi
"$PIE_VENV/bin/python" -c "import vllm,torch;print(f'  vLLM {vllm.__version__}, torch {torch.__version__}')"

if [ ! -x "$HARNESS_VENV/bin/python" ]; then
    echo "  creating harness venv ($PY, deps pinned --exclude-newer $HARNESS_EXCLUDE_NEWER)"
    "$PY" -m venv "$HARNESS_VENV"
    VIRTUAL_ENV="$HARNESS_VENV" uv pip install -U pip
    ( cd "$HARNESS_DIR" && VIRTUAL_ENV="$HARNESS_VENV" \
        uv pip install --exclude-newer "$HARNESS_EXCLUDE_NEWER" -e . )
else
    echo "  harness venv exists"
fi
# openhands-sdk and openhands-tools ship in lockstep; swe_bench.py:370 imports
# openhands.tools.preset.default. Mismatched versions fail hours in, not at import.
"$HARNESS_VENV/bin/python" - <<'PY'
from importlib.metadata import version
sdk, tools = version("openhands-sdk"), version("openhands-tools")
print(f"  openhands-sdk {sdk} / openhands-tools {tools}", "OK" if sdk == tools else "*** MISMATCH ***")
print(f"  litellm {version('litellm')}")
assert sdk == tools, "openhands-sdk and openhands-tools must be identical"
PY
# .venv symlink so `cd integrations/openhands && python` picks up the harness
ln -sfn "$HARNESS_VENV" "$HARNESS_DIR/.venv"

# =============================================================================
say "[5] build pie for sm_$ARCH  (build tree on local disk)"
# =============================================================================
# `target/` is a symlink to local disk: a cargo build tree is tens of thousands
# of small files and will stall on MooseFS.
if [ ! -L "$PIE_SRC/target" ]; then
    [ -e "$PIE_SRC/target" ] && die "$PIE_SRC/target exists and is not a symlink — move it aside."
    ln -sfn "$BUILD_TREE" "$PIE_SRC/target"
fi
echo "  target -> $(readlink -f "$PIE_SRC/target")"

if [ -x "$PIE_SRC/target/release/pie" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
    echo "  pie binary present — set FORCE_REBUILD=1 to rebuild"
else
    export CMAKE_CUDA_ARCHITECTURES=$ARCH PIE_PORTABLE_CUDA_ARCH=$ARCH
    echo "  building for sm_$ARCH (this takes a while)"
    ( cd "$PIE_SRC" && cargo build -p pie-server --release --features driver-portable,driver-cuda )
fi
ls -la "$PIE_SRC/target/release/pie"

WASM=inferlets/openhands-coder-session/target/wasm32-wasip2/release/openhands_coder_session.wasm
if [ ! -f "$PIE_SRC/$WASM" ] || [ "${FORCE_REBUILD:-0}" = "1" ]; then
    echo "  building the coder-session wasm inferlet"
    ( cd "$PIE_SRC/inferlets/openhands-coder-session" && cargo build --target wasm32-wasip2 --release )
fi
ls -la "$PIE_SRC/$WASM"

# =============================================================================
say "[6] confirm vLLM ships the tuned MoE config for THIS device"
# =============================================================================
# This is why H200 needs no autotune. If it is missing, the fair tier is not
# available and assert_vllm_fair.sh will hard-fail it.
"$PIE_VENV/bin/python" - <<'PY'
import os, torch
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name
import vllm.model_executor.layers.fused_moe.fused_moe as fm
name = get_config_file_name(128, 768, None, None)
cfgdir = os.path.join(os.path.dirname(fm.__file__), "configs")
path = os.path.join(cfgdir, name)
print(f"  device      : {torch.cuda.get_device_properties(0).name}")
print(f"  vLLM expects: {name}")
if os.path.exists(path):
    import json
    d = json.load(open(path)); d.pop("triton_version", None)
    print(f"  SHIPPED     : {len(d)} batch-size keys — no autotune needed.")
else:
    print(f"  *** NOT SHIPPED at {path}")
    print("  *** The 'fair' tier needs a tuned config. See AGENT_HANDOVER_H200.md §4.")
    raise SystemExit(1)
PY
if [ -n "${VLLM_TUNED_CONFIG_FOLDER:-}" ]; then
    warn "VLLM_TUNED_CONFIG_FOLDER is set (=$VLLM_TUNED_CONFIG_FOLDER)."
    warn "On H200 it is not needed and can only cause confusion. Unset it."
fi

# =============================================================================
say "[7] model weights"
# =============================================================================
if [ "$SKIP_MODEL" = "1" ]; then
    echo "  present — skipping"
else
    HF_HOME="$HF_HOME" "$PIE_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
print("  model cached under $HF_HOME")
PY
fi

# =============================================================================
say "[8] THE CRITICAL CHECK — are Pie's fast paths actually ON?"
# =============================================================================
# This is the entire reason for moving off the A100. The driver prints its
# feature flags at model-load time. Anything other than
#     prefill_decode_plan=on xqa_decode=on
# means STOP — do not collect a Pie arm.
CFG="$RUNPOD_DIR/pie_cuda_native_config_30b_moe_h200.toml"
[ -f "$CFG" ] || die "missing $CFG"
PROBE_LOG=$WORK/pie/integrations/openhands/logs/pie_gatecheck_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$PROBE_LOG")"
echo "  starting pie briefly to read its banner -> $PROBE_LOG"
echo "  (loads ~58 GB of weights; several minutes of silence is normal)"
set +e
timeout 900 "$PIE_SRC/target/release/pie" serve --config "$CFG" > "$PROBE_LOG" 2>&1 &
PROBE_PID=$!
for _ in $(seq 1 180); do
    grep -q 'prefill_decode_plan=' "$PROBE_LOG" 2>/dev/null && break
    kill -0 "$PROBE_PID" 2>/dev/null || break
    sleep 5
done
BANNER=$(grep -oE 'prefill_decode_plan=[a-z]+ xqa_decode=[a-z]+[^ ]*' "$PROBE_LOG" 2>/dev/null | head -1)
kill "$PROBE_PID" 2>/dev/null; wait "$PROBE_PID" 2>/dev/null
set -e

echo "  banner: ${BANNER:-<not found>}"
case "$BANNER" in
    *prefill_decode_plan=on*xqa_decode=on*)
        echo "  PASS — both fast attention paths are ON. This pod can produce a valid Pie arm." ;;
    "")
        die "could not read the driver banner. Inspect $PROBE_LOG before proceeding." ;;
    *)
        die "FAST PATHS ARE OFF on this pod: $BANNER
       This is the A100 failure repeating. Do NOT collect a Pie arm.
       Check compute cap (needs major >= 9) and that pie was built for sm_$ARCH." ;;
esac

# =============================================================================
say "done"
# =============================================================================
cat <<EOF
Source the env in every shell:
    source $ENV_FILE
    cd $RUNPOD_DIR

Run order (one arm at a time, verify before the next):

  1. Pie arm — uses the H200 toml, KV_VERIFY=1, no vLLM involved
       CFG=$RUNPOD_DIR/pie_cuda_native_config_30b_moe_h200.toml \\
       ARM=pie bash 30_ab_run.sh

  2. vLLM fair — the shipped MoE config makes this the DEFAULT here.
       Do NOT export VLLM_TUNED_CONFIG_FOLDER.
       VLLM_TIER=fair ARM=litellm bash 30_ab_run.sh

  3. vLLM crippled — reproduces the original writeup's --enforce-eager baseline.
       Requires MOVING THE SHIPPED CONFIG ASIDE first; see
       AGENT_HANDOVER_H200.md §4. Read it, this differs from the A100 flow.

  4. summarize
       python summarize_ab.py pie=../predictions/*_pie_*.jsonl \\
           litellm-fair=../predictions/*_litellm_fair_*.jsonl

Read AGENT_HANDOVER_H200.md before step 1.
EOF
