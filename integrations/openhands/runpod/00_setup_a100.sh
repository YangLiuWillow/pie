#!/usr/bin/env bash
# =============================================================================
# One-time setup on a fresh runpod A100 SXM box for the fair-parity rerun.
#
# Builds BOTH arms from source ON THIS GPU so neither is crippled:
#   - Pie native CUDA driver, rebuilt for sm_80 (A100). Do NOT copy an sm_120
#     binary from the Blackwell cluster — it must be recompiled here.
#   - vLLM in its own venv (for the fair baseline).
#   - The OpenHands harness venv + the coder-session wasm inferlet.
#
# This is a SCAFFOLD/checklist as much as a script: stop and read the LOGISTICS
# notes below before running — a couple of steps (git ref, disk) need a decision.
# Edit the env vars at the top, then run section by section.
# =============================================================================
set -euo pipefail

# ---- paths (edit for the runpod box) ----------------------------------------
export WORK=${WORK:-/workspace}
export PIE_SRC=${PIE_SRC:-$WORK/pie}                      # pie checkout
export PIE_VENV=${PIE_VENV:-$HOME/.venvs/pie-vllm}        # vLLM venv
export HF_HOME=${HF_HOME:-$WORK/hf-cache}
export MODEL=${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}
export CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE:-$WORK/.cpm-cache}

# =============================================================================
# LOGISTICS — READ FIRST
# =============================================================================
# 1. GIT REF: the working harness + the §4 CUDA fix (cuda_memory_planner.cpp:
#    output_rows R0->N) + bug C (drain_queues starvation) live on the
#    `openhands-integration-updated` branch, and per the handover those fixes
#    are COMMITTED BUT NOT PUSHED. A fresh `git clone` from the remote will NOT
#    have them. Options, best first:
#      (a) push that branch, then clone it here; OR
#      (b) `git bundle create pie.bundle --all` on the cluster, scp it here,
#          `git clone pie.bundle`; OR
#      (c) rsync the whole `pie-updated-wt` tree to $PIE_SRC.
#    Without the §4 fix, any coder-session prefill >512 tokens faults the driver.
# 2. DISK: model (~60 GB bf16) + HF cache + two builds. runpod A100 volumes are
#    often 20-50 GB by default — provision a big /workspace volume first.
# 3. GPU: confirm it is actually an A100 SXM (sm_80), not a PCIe/40GB variant —
#    the 80 GB config assumes 80 GB. `nvidia-smi --query-gpu=name,memory.total`.
# =============================================================================

echo "=== [0] sanity ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true
python3 --version; git --version

echo "=== [1] system deps (Rust, CMake, CUDA toolkit assumed present in image) ==="
# runpod pytorch/cuda images ship nvcc; if not, install a matching CUDA toolkit.
command -v nvcc >/dev/null || echo "WARN: nvcc not on PATH — install CUDA toolkit matching the driver."
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
fi
rustup target add wasm32-wasip2 || true

echo "=== [2] fetch pie source into $PIE_SRC (see LOGISTICS note 1) ==="
if [ ! -d "$PIE_SRC/.git" ]; then
    echo "  $PIE_SRC not present. Bring it here (bundle/rsync/clone of the branch"
    echo "  that has the §4 + bug-C fixes) THEN re-run from section [3]."
    exit 0
fi

echo "=== [3] build Pie native CUDA driver for sm_80 (A100) ==="
# Arch is auto-detected from nvidia-smi; pin it explicitly to be safe.
export CMAKE_CUDA_ARCHITECTURES=80
export PIE_PORTABLE_CUDA_ARCH=80
( cd "$PIE_SRC" && cargo build -p pie-server --release --features driver-portable,driver-cuda )
echo "  built: $PIE_SRC/target/release/pie"

echo "=== [4] build the coder-session wasm inferlet ==="
( cd "$PIE_SRC/inferlets/openhands-coder-session" \
    && cargo build --target wasm32-wasip2 --release )

echo "=== [5] vLLM venv (fair baseline arm) ==="
if [ ! -x "$PIE_VENV/bin/python" ]; then
    python3 -m venv "$PIE_VENV"
    "$PIE_VENV/bin/pip" install -U pip
    "$PIE_VENV/bin/pip" install vllm            # pin a version to match the writeup
fi
"$PIE_VENV/bin/python" -c "import vllm; print('vLLM', vllm.__version__)"

echo "=== [6] OpenHands harness venv ==="
HARNESS_VENV="$PIE_SRC/integrations/openhands/.venv"
if [ ! -x "$HARNESS_VENV/bin/python" ]; then
    python3 -m venv "$HARNESS_VENV"
    "$HARNESS_VENV/bin/pip" install -U pip
    # install the harness package + its deps (openhands sdk, pie_client, etc.)
    ( cd "$PIE_SRC/integrations/openhands" && "$HARNESS_VENV/bin/pip" install -e . )
fi

echo "=== [7] download the model ==="
HF_HOME="$HF_HOME" "$PIE_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL")
print("model cached under \$HF_HOME")
PY

echo ""
echo "=== setup done ==="
echo "Next:"
echo "  1) bash 10_vllm_serve_fair.sh              # fair vLLM; must pass assert_vllm_fair.sh"
echo "     (if it reports default MoE config: bash 11_autotune_moe.sh, then relaunch)"
echo "  2) python 20_decode_microbench.py ...       # identical decode measurement, both engines"
echo "  3) bash 30_ab_run.sh                        # wall-clock A/B, normalized per-iteration"
