#!/usr/bin/env bash
# Fused TMA-WS SwiGLU MoE: GEMM1 tactic sweep (handover next-action #1).
# 36 TMA configs exist for this shape (doctor probe); subsample every 4th
# index, GEMM2 pinned at 0. Index 0/0 reproduces the 14.5k default anchor.
set -u
source /workspace/pie-bench-env.sh
cd /workspace/pie/integrations/openhands/runpod
RES=/tmp/claude-0/-workspace/4d65e451-7d5b-452d-a69b-3796082b3ac0/scratchpad/tacsweep
mkdir -p "$RES"

for idx in 0 4 8 12 16 20 24 28 32; do
  echo "=== GEMM1_INDEX=$idx start $(date +%H:%M:%S)"
  PIE_QWEN35_MOE_CUTLASS_FUSED=1 \
  PIE_NEMOTRON_FLASHINFER_MOE_SELECT=raw \
  PIE_NEMOTRON_FLASHINFER_MOE_GEMM1_INDEX=$idx \
  PIE_NEMOTRON_FLASHINFER_MOE_GEMM2_INDEX=0 \
  PIE_NEMOTRON_FLASHINFER_MOE_LOG=1 \
  PIE_CUDA_PREFILL_TOKENS=2048 \
  PIE_QWEN35_MOE_ALIGNED_DECODE_BLOCK=64 \
  SWEEP_PREFILL_SUFFIXES=1024,4096 SWEEP_REPS=1 \
  SWEEP_ARMS=pie SWEEP_MODE=prefill \
    bash 42_context_sweep.sh > "$RES/g1_${idx}.log" 2>&1
  d=$(grep -oa "out=.*" "$RES/g1_${idx}.log" | head -1 | cut -d= -f2)
  echo "outdir $d"
  if [ -n "$d" ] && [ -f "$d/pie_server.log" ]; then
    grep -a "FlashInfer MoE" "$d/pie_server.log" | head -4
  fi
  grep -a "prefill fit\|self_prefill\|latency_ms" "$RES/g1_${idx}.log" | head -8
done
echo "SWEEP DONE $(date +%H:%M:%S)"
