#!/usr/bin/env python3
"""Apply the fp8-alpha-gate to the vendored moe_gemm_tma_ws_launcher.inl.
Idempotent. Usage: python3 this.py /path/to/flashinfer-src"""
import re, sys
p = sys.argv[1] + '/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl'
s = open(p).read()
if 'IsFinalizeFusion && IsFP8' in s:
    print('already applied'); sys.exit(0)
s = re.sub(r'construct_if_true<\(!IsSimpleAlphaBeta && !IsFinalizeFusion\)',
           'construct_if_true<(!IsSimpleAlphaBeta && !IsFinalizeFusion && IsFP8)', s)
s = re.sub(r'construct_if_true<\(IsSimpleAlphaBeta && !IsFinalizeFusion\)',
           'construct_if_true<(IsSimpleAlphaBeta && !IsFinalizeFusion && IsFP8)', s)
open(p, 'w').write(s)
print('applied')
