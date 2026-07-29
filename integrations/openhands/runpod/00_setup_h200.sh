#!/usr/bin/env bash
# =============================================================================
# Bring-up on a FRESH RunPod box with compute capability >= 9.
#
# Named for the H200 it was written on; it has since brought up an H100 and is
# board-agnostic where it matters. Read AGENT_HANDOVER_H200.md first — it
# explains what the A100 run established.
#
# The one-line reason for the >= 9 requirement: Pie's fast attention paths are
# gated to compute capability >= 9. The A100 is sm_80 and failed both gates
# silently, so the A100 Pie arm measured Pie's fallback path. Section [8] below
# VERIFIES the gates at runtime and is the most important step here.
#
# TWO THINGS THAT ARE PER-BOARD, NOT PER-ARCHITECTURE — sm_90 does not imply
# either, and assuming the H200 answer has cost time already:
#   - THE TUNED MoE CONFIG. vLLM 0.25.1 ships E=128,N=768 bf16 for H200 / B200 /
#     H20 / MI308X and NOT for A100 or H100. Section [6] resolves this against
#     the installed package and appends the right export to the env file.
#   - THE KV BUDGET. H200's 141 GB leaves ~72 GB of KV; H100's 80 GB leaves
#     ~15 GiB, ~4x less, which changes what a concurrent arm is measuring.
#     Section [0] computes it for this board and writes it into the env file.
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
# vLLM context cap. See the note in the generated env file (§3) for why this is
# 131072 here and was 32768 on the A100.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}

mkdir -p "$FAST/venvs" "$CPM_SOURCE_CACHE" "$UV_CACHE_DIR" "$BUILD_TREE" "$PIP_CACHE_DIR"

# =============================================================================
say "[0] preflight — is this board usable for BOTH arms?"
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
  (cuda_memory_planner.cpp:219,:250) stay OFF. Expected on any sm_90. State it."

GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# Not a failure — smaller boards work, they just have a much smaller KV pool,
# which the block below quantifies and the env file records.
[ "$GPU_MEM" -ge 130000 ] || warn "GPU has ${GPU_MEM} MiB, under the 141 GB an H200 has.
  The Pie tomls are memory_profile=auto so they plan correctly regardless, but the
  KV pool is proportionally smaller — see the budget computed next."

# Driver vs CUDA runtime. Compute capability is NOT the only pod-selection
# criterion — the DRIVER matters independently, and it bit us once:
#   vllm >= 0.20.0 pins torch==2.11.0, which is a CUDA 13 build (nvidia-nccl-cu13)
#   and whose kernels link libcudart.so.13. CUDA 13 requires an r580+ driver.
#   On an r570 pod (CUDA 12.8) the ENTIRE stack resolves, downloads and installs
#   without a single error, then dies at the first torch.cuda call with
#   "The NVIDIA driver on your system is too old (found version 12080)".
# A clean `uv pip install` is not evidence the stack can reach the GPU — same
# lesson as "it compiles is not it imports" (AGENT_HANDOVER_H200.md §9).
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
DRV_MAJOR=${DRV%%.*}
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
echo "  driver $DRV (supports CUDA up to ${DRV_CUDA:-unknown})"
if [ "${DRV_MAJOR:-0}" -lt 580 ] && [ "${ALLOW_OLD_DRIVER:-0}" != "1" ]; then
    die "driver $DRV supports CUDA ${DRV_CUDA:-<12.x>}, but vllm==$VLLM_VERSION needs
       CUDA 13 (torch 2.11.0, libcudart.so.13) and therefore an r580+ driver.
       Nothing will fail until the first CUDA init, hours in. Options:
         - get a pod whose nvidia-smi reports CUDA Version 13.x  (preferred)
         - VLLM_VERSION=0.19.0 (torch 2.10.0 / CUDA 12.8) — this is a PIN CHANGE
           and is on the escalate list; ask the human first.
       The Pie arm is unaffected: pie builds against the system CUDA toolkit.
       Set ALLOW_OLD_DRIVER=1 to run Pie-only bring-up on this pod anyway."
fi

# Does MAX_MODEL_LEN fit? KV is 96 KiB/token for this model (48 layers x 4 KV
# heads x 128 dim x 2 (K,V) x 2 bytes bf16). Weights are ~58 GB.
#
# Captured into shell variables as well as printed, because section [3] writes
# them into the env file. A KV figure quoted from the wrong board is how a
# writeup ends up claiming H200 headroom on an 80 GB card.
KV_FACTS=$(python3 - <<EOF
gpu_mib   = $GPU_MEM
mml       = $MAX_MODEL_LEN
kv_kib    = 96
weights   = 58_500          # MiB, measured
# ENGINE OVERHEAD, and leaving it out is why this check used to be optimistic.
# Neither engine gets util*total - weights for KV. Measured on the H100:
#   vLLM  weights 56.93 GiB + CUDA graphs 0.64 GiB -> KV 10.77 GiB
#   Pie   safety 810 MiB + arena 798 MiB + page_refs -> KV 12.14 GiB
# against a naive estimate of ~15 GiB. vLLM's activation peak and non-torch
# reservations account for most of the difference. 4200 MiB is calibrated from
# that measurement and is deliberately the LARGER (vLLM) overhead, so this check
# predicts the binding engine rather than the roomier one.
overhead  = 4_200
budget    = gpu_mib * 0.90 - weights - overhead
tokens    = int(budget * 1024 / kv_kib)
if tokens <= 0:
    raise SystemExit("no KV budget at all on this GPU")
# Clamp rather than fail: a board that cannot hold the requested context can
# still run the experiment at a smaller one, and the cap is not binding on this
# workload anyway (largest observed SWE-bench prompt: 46,870 tokens). Failing
# here would have stopped the H100 re-baseline for a limit nothing reaches.
if mml > tokens:
    mml = (tokens // 1024) * 1024
conc = tokens / mml if mml else 0
print(f"{budget/1024:.0f}|{tokens}|{conc:.1f}|{mml}")
EOF
) || die "could not compute a KV budget for this GPU."
KV_GIB=${KV_FACTS%%|*}
KV_TOKENS=$(printf '%s' "$KV_FACTS" | cut -d'|' -f2)
KV_CONC=$(printf '%s' "$KV_FACTS" | cut -d'|' -f3)
_MML_FIT=$(printf '%s' "$KV_FACTS" | cut -d'|' -f4)
printf "  KV budget ~%s GiB -> ~%s tokens (after ~4.2 GiB engine overhead)\n" "$KV_GIB" "$KV_TOKENS"
if [ "$_MML_FIT" != "$MAX_MODEL_LEN" ]; then
    warn "MAX_MODEL_LEN=$MAX_MODEL_LEN does not fit this board — clamped to $_MML_FIT.
  vLLM refuses to start when one max-length request exceeds its KV pool, and this
  is what that looks like BEFORE spending 5 minutes loading weights to find out.
  Not binding on this workload (largest observed SWE-bench prompt: 46,870), but
  it IS a protocol deviation from any board that ran the larger value — say so."
    MAX_MODEL_LEN=$_MML_FIT
fi
printf "  MAX_MODEL_LEN=%s -> ~%sx concurrency\n" "$MAX_MODEL_LEN" "$KV_CONC"
if awk -v c="$KV_CONC" 'BEGIN{exit !(c < 2)}'; then
    warn "under 2x concurrency at this MAX_MODEL_LEN. Fine for a serial A/B, but a
  c8/c16 arm will run under KV pressure — a REAL DIFFERENCE from the H200 arms
  (622,112 KV tokens) that must be stated next to any concurrent number from
  this box. It hits BOTH arms, but they do not degrade the same way."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
# Short tag used to pick per-board files: the Pie toml (30_ab_run.sh) and the
# borrowed-MoE folder ([6]). Derived from the device so it cannot disagree with
# the machine. "NVIDIA H100 80GB HBM3" -> h100, "NVIDIA A100-SXM4-80GB" -> a100.
GPU_TAG=$(printf '%s' "$GPU_NAME" | sed -e 's/NVIDIA //' -e 's/ .*//' -e 's/-.*//' | tr 'A-Z' 'a-z')
export GPU_TAG
echo "  device name: $GPU_NAME  (tag: $GPU_TAG)"

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

# Fork-only push guard. YangLiuWillow/pie is a FORK of pie-project/pie; this
# experiment's commits belong on the fork. `.git/hooks` is not versioned, so a
# fresh clone arrives without it — reinstall from the tracked copy every time.
if [ -f "$RUNPOD_DIR/git-hooks/pre-push" ]; then
    install -m 0755 "$RUNPOD_DIR/git-hooks/pre-push" "$PIE_SRC/.git/hooks/pre-push"
    git -C "$PIE_SRC" config remote.pushDefault origin
    echo "  push guard: pre-push hook installed; bare 'git push' pinned to origin (the fork)"
fi
PUSH_URL=$(git -C "$PIE_SRC" remote get-url --push origin 2>/dev/null || echo "?")
case "$PUSH_URL" in
    *YangLiuWillow/pie*) echo "  push target: $PUSH_URL (fork — correct)" ;;
    *) warn "origin pushes to $PUSH_URL, which is NOT the fork. Check before pushing." ;;
esac

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
# Python 3.12 is REQUIRED: on 3.13 the vLLM set resolves and installs, then
# torch dies at import in the TorchScript overload parser. A successful resolve
# is not evidence the stack runs.
#
# Do not assume the image ships it. The first H200 pod had python3.12 on PATH;
# the 2026-07-28 image is conda-based with 3.11 as `python3` and no 3.12 at all,
# which killed this script ~40 s in. uv can provision the exact version, so
# fetch it rather than dying on an image difference that is not a pin change.
if ! command -v "$PY" >/dev/null; then
    echo "  $PY not on PATH — provisioning it with uv (this is not a pin change)"
    uv python install 3.12 || die "uv could not install Python 3.12"
    PY=$(uv python find 3.12) \
        || die "uv installed Python 3.12 but 'uv python find 3.12' failed"
    [ -x "$PY" ] || die "resolved interpreter '$PY' is not executable"
fi
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

# REQUIRED for xqa_decode. Do not drop this and do not assume the toml covers it.
#
# sm_90 satisfies the ARCH gate in xqa_decode_bf16_supported(), but that function
# has SEVEN conditions and \`major >= 9\` is only the last one. One of the others is
# the KV page size: the gqa8 XQA kernel is compiled with TOKENS_PER_PAGE=32
# (attention_xqa_gqa8.cu:21), and the page_size==16 escape hatch applies only to
# GQA ratio 2 with PIE_CUDA_XQA_GQA2_P16 set. This model is ratio 8 (32 q / 4 kv),
# so it does not qualify.
#
# The memory planner's \`auto\` page-size selection picks 16 on this box and
# SILENTLY OVERRIDES kv_page_size=32 in the toml, which turns xqa_decode off while
# prefill_decode_plan stays on. Observed banner without this variable:
#     page_size=16 (auto) ... prefill_decode_plan=on xqa_decode=off
# With it:
#     page_size=32 (auto) ... prefill_decode_plan=on xqa_decode=on
export PIE_CUDA_KV_PAGE_SIZE=32

# Context cap for the vLLM arms. 131072 = the OpenHands SWE-bench norm (128k).
#
# MEASURED ON THIS BOX AT BRING-UP — do not carry another board's figure here:
#   device:     $GPU_NAME ($GPU_MEM MiB)
#   KV budget:  ~$KV_GIB GiB -> ~$KV_TOKENS tokens at 96 KiB/token
#               (48 layers x 4 KV heads x 128 dim x 2 (K,V) x 2 bytes bf16)
#   headroom:   ~${KV_CONC}x MAX_MODEL_LEN
#
# For scale, the reference points this experiment has actually run:
#   A100 80GB  12.59 GiB KV = 137,472 tokens -> 32768 was forced, not chosen
#   H200 141GB ~72 GB KV, and the sweep arms planned 622,112 KV tokens
#   H100 80GB  ~15 GiB KV = ~159k tokens -> ~1.2x at 131072
# If the headroom above is under ~2x, a concurrent (c8/c16) arm runs under KV
# pressure and that fact belongs beside the number, on both arms.
#
# This matters because the serving cap is the ONLY context limit in the stack:
# litellm has no entry for a self-hosted model (the benign "isn't mapped yet"
# line), so nothing truncates client-side, and the condenser bounds history by
# MESSAGE COUNT (240), not tokens. Pie has no fixed cap at all -- its ceiling is
# memory-planned -- so too low a value here fails vLLM on inputs Pie serves
# fine, which looks like an accuracy difference and is not.
# The model is native 262144 (no RoPE scaling below that); do not exceed it
# without YaRN.
export MAX_MODEL_LEN=$MAX_MODEL_LEN

# TUNED MoE CONFIG — decided in section [6], which APPENDS the export below this
# line if this board needs one. Whether it does is per-board and the H200 answer
# does not generalise: vLLM 0.25.1 ships E=128,N=768 bf16 for H200/B200/H20/
# MI308X and NOT for A100 or H100. Section [6] resolves it against the installed
# package rather than against this comment.

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
# `$PY -m venv` fails on a uv-provisioned interpreter: ensurepip exits non-zero
# (observed 2026-07-28, "returned non-zero exit status 1" creating pie-vllm).
# `uv venv --seed` builds the same layout, works with both system and
# uv-managed interpreters, and seeds pip so `<venv>/bin/pip` still exists for
# anything that shells out to it.
make_venv() {
    local dest="$1"
    rm -rf "$dest"
    uv venv --seed --python "$PY" "$dest" \
        || die "could not create venv at $dest with $PY"
}

if [ ! -x "$PIE_VENV/bin/python" ]; then
    echo "  creating vLLM venv ($PY, vllm==$VLLM_VERSION)"
    make_venv "$PIE_VENV"
    VIRTUAL_ENV="$PIE_VENV" uv pip install "vllm==$VLLM_VERSION"
else
    echo "  vLLM venv exists"
fi
"$PIE_VENV/bin/python" -c "import vllm,torch;print(f'  vLLM {vllm.__version__}, torch {torch.__version__}')"
# A successful install is not evidence torch can reach the GPU. Force a real CUDA
# init here rather than discovering it in section [6] or, worse, mid-arm.
"$PIE_VENV/bin/python" - <<'PY' || die "torch cannot initialize CUDA on this pod — see section [0]'s driver check."
import torch
torch.cuda.init()
print(f"  CUDA init OK: {torch.cuda.get_device_name(0)}, torch built for cuda {torch.version.cuda}")
PY

if [ ! -x "$HARNESS_VENV/bin/python" ]; then
    echo "  creating harness venv ($PY, deps pinned --exclude-newer $HARNESS_EXCLUDE_NEWER)"
    make_venv "$HARNESS_VENV"
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
say "[6] resolve the tuned MoE config for THIS device"
# =============================================================================
# The 'fair' tier is "CUDA graphs ON + a TUNED MoE kernel". Whether vLLM supplies
# one is PER-BOARD, and assuming the H200 answer is the trap this section exists
# to close:
#   H200 / B200 / H20 / MI308X : vLLM 0.25.1 ships E=128,N=768 bf16. Nothing to do.
#   A100 / H100                : it does NOT. The tier needs a borrowed config in
#                                VLLM_TUNED_CONFIG_FOLDER, or it is not fair and
#                                assert_vllm_fair.sh will hard-fail it.
#
# Resolution order matches vLLM's own (fused_moe.py:1075-1089): the env folder
# first, then the package dir. Whatever wins is APPENDED to the env file, so the
# arms inherit the same decision instead of depending on an exported shell var.
TUNED_DEFAULT=${TUNED_MOE_DIR:-$WORK/tuned_moe_${GPU_TAG}}
MOE_EXPORT=$("$PIE_VENV/bin/python" - "$TUNED_DEFAULT" <<'PY'
import json, os, sys, torch
import vllm.model_executor.layers.fused_moe.fused_moe as fm
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name

name = get_config_file_name(128, 768, None, None)
pkg = os.path.join(os.path.dirname(fm.__file__), "configs", name)
env_folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER") or ""
default_folder = sys.argv[1]

def describe(path):
    d = json.load(open(path)); d.pop("triton_version", None)
    return f"{len(d)} batch-size keys"

print(f"  device      : {torch.cuda.get_device_properties(0).name}", file=sys.stderr)
print(f"  vLLM expects: {name}", file=sys.stderr)

for folder, why in ((env_folder, "VLLM_TUNED_CONFIG_FOLDER"),
                    (default_folder, "conventional location")):
    if folder and os.path.exists(os.path.join(folder, name)):
        p = os.path.join(folder, name)
        print(f"  BORROWED    : {describe(p)} via {why}", file=sys.stderr)
        print(f"                {p}", file=sys.stderr)
        print(f"  NOTE        : borrowed, not autotuned here. {folder}/PROVENANCE.md",
              file=sys.stderr)
        print(f"                and VALIDATION.md must be cited with any fair-tier number.",
              file=sys.stderr)
        print(folder)   # stdout = the folder to export
        sys.exit(0)

if os.path.exists(pkg):
    print(f"  SHIPPED     : {describe(pkg)} — no autotune, no export needed.", file=sys.stderr)
    print("")           # stdout empty = export nothing
    sys.exit(0)

print(f"  *** NO TUNED CONFIG for this device.", file=sys.stderr)
print(f"  *** not packaged : {pkg}", file=sys.stderr)
print(f"  *** not borrowed : {default_folder}/{name}", file=sys.stderr)
print(f"  *** The 'fair' tier is NOT AVAILABLE until one exists. Options:", file=sys.stderr)
print(f"  ***   - drop a published config for this exact device name into", file=sys.stderr)
print(f"  ***     {default_folder}/ and document it (see runpod/tuned_moe/ for the", file=sys.stderr)
print(f"  ***     A100 precedent: PROVENANCE.md + a measured VALIDATION.md), or", file=sys.stderr)
print(f"  ***   - autotune, which was abandoned once at a 15-24 h projection with", file=sys.stderr)
print(f"  ***     no partial-progress artifact (RUN_STATE.md 3b).", file=sys.stderr)
sys.exit(1)
PY
) || die "no tuned MoE config for this device — the 'fair' tier cannot be run. See above."

if [ -n "$MOE_EXPORT" ]; then
    printf 'export VLLM_TUNED_CONFIG_FOLDER=%s\n' "$MOE_EXPORT" >> "$ENV_FILE"
    echo "  appended to $ENV_FILE: export VLLM_TUNED_CONFIG_FOLDER=$MOE_EXPORT"
    echo "  (correct for the 'fair' tier ONLY — 'crippled' and 'graphs-only' are"
    echo "   defined as DEFAULT MoE and assert_vllm_fair.sh hard-fails both if a"
    echo "   tuned config is live. Run those with 'env -u VLLM_TUNED_CONFIG_FOLDER'.)"
elif [ -n "${VLLM_TUNED_CONFIG_FOLDER:-}" ]; then
    warn "VLLM_TUNED_CONFIG_FOLDER is set (=$VLLM_TUNED_CONFIG_FOLDER) but this board
  ships its own config. It is not needed and can only cause confusion. Unset it."
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
# --- preflight: does this binary even have the driver? -----------------------
# `pie driver cuda-native doctor` is the driver's own self-check (driver/cuda/
# README.md). It answers "is cuda_native compiled into THIS binary and can it
# see the GPU" in under a second, against the ~3 minutes the banner probe below
# spends loading 58 GB of weights before it can fail. A binary built without
# --features driver-cuda, or a pod whose GPU has gone away, dies here instead.
echo "  driver self-check: pie driver cuda-native doctor"
if ! DOCTOR=$("$PIE_SRC/target/release/pie" driver cuda-native doctor 2>&1); then
    echo "$DOCTOR" | sed 's/^/    /'
    die "cuda-native doctor failed — the built binary cannot use this GPU."
fi
echo "$DOCTOR" | sed 's/^/    /'
case "$DOCTOR" in
    *"compiled in"*) ;;
    *) die "cuda-native is NOT compiled into $PIE_SRC/target/release/pie.
       Rebuild with --features driver-portable,driver-cuda." ;;
esac

# Probe the CONFIG A toml for THIS board, matching 30_ab_run.sh's selection.
# Hardcoding the h200 file made the gate check probe a config whose header
# documents a different KV budget — the values are identical (memory_profile =
# "auto"), so it still passed, which is exactly what makes it easy to miss.
CFG="$RUNPOD_DIR/pie_cuda_native_config_30b_moe_${GPU_TAG}.toml"
if [ ! -f "$CFG" ]; then
    CFG="$RUNPOD_DIR/pie_cuda_native_config_30b_moe_h200.toml"
    warn "no pie_cuda_native_config_30b_moe_${GPU_TAG}.toml — probing the h200 one.
  Its values plan correctly here (auto profile) but its header describes an H200.
  Add a ${GPU_TAG} variant before quoting KV numbers from it."
fi
[ -f "$CFG" ] || die "missing $CFG"
echo "  probing config: $CFG"
PROBE_LOG=$HARNESS_DIR/logs/pie_gatecheck_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$PROBE_LOG")"
echo "  starting pie briefly to read its banner -> $PROBE_LOG"
echo "  (loads ~58 GB of weights; several minutes of silence is normal)"
set +e
# See section [3]: without this the planner picks page_size=16 and xqa_decode is
# off even though the arch gate passes.
export PIE_CUDA_KV_PAGE_SIZE=${PIE_CUDA_KV_PAGE_SIZE:-32}
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

PLANNED_PAGE=$(grep -oaE 'page_size=[0-9]+' "$PROBE_LOG" 2>/dev/null | head -1)
echo "  planner: ${PLANNED_PAGE:-<not found>}  (must be page_size=32 for xqa)"
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

This board: $GPU_NAME (tag $GPU_TAG, sm_$ARCH)
  KV budget ~$KV_GIB GiB / ~$KV_TOKENS tokens, ~${KV_CONC}x at MAX_MODEL_LEN=$MAX_MODEL_LEN
  MoE config: ${MOE_EXPORT:-shipped by vLLM (nothing exported)}

Run order (one arm at a time, verify before the next):

  1. Pie arm — KV_VERIFY=1, no vLLM involved. 30_ab_run.sh picks the toml for
     this board automatically (PIE_CFG_VARIANT=auto_p32 by default).
       ARM=pie bash 30_ab_run.sh

  2. vLLM fair
       VLLM_TIER=fair ARM=litellm bash 30_ab_run.sh

  3. vLLM crippled / graphs-only — both are defined as DEFAULT MoE, so on a board
     where [6] exported a borrowed config they must be run with it suppressed:
       env -u VLLM_TUNED_CONFIG_FOLDER VLLM_TIER=crippled ARM=litellm bash 30_ab_run.sh
     assert_vllm_fair.sh hard-fails either tier if a tuned config is live, so
     this is verified rather than assumed.

  4. summarize — glob by GPU tag, or arms from different boards merge silently
       python summarize_ab.py pie=../predictions/ab_${GPU_TAG}_pie_*.jsonl \\
           litellm-fair=../predictions/ab_${GPU_TAG}_litellm_fair_*.jsonl

Read AGENT_HANDOVER_H200.md before step 1.
EOF
