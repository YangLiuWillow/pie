#!/usr/bin/env python3
"""A2: what does MLX's quantized GEMM achieve at prefill shapes, on this GPU?

## The question this answers

A1 established that pie's prefill is **99.95% GPU execution** — encode, dispatch,
epilogue and dtype conversion together are 0.05%. So the deficit is in the
kernels. The remaining question is whether the arithmetic itself is simply
expensive on this hardware, or whether MLX extracts far more from the same
silicon.

vLLM-metal *is* MLX, so MLX's `quantized_matmul` is exactly the kernel the
baseline runs. This times it directly at the projection shapes of a real model,
at prefill width, and compares the sum against pie's measured per-forward GPU
time for the same token count.

- MLX sums to roughly pie's time  →  the kernels are comparable and pie's
  deficit is somewhere else in the forward (attention, norms, routing, layout).
- MLX sums to a fraction of pie's time  →  it is the GEMM, and the fix is to
  match MLX's tiling or call it.

## Why per-projection and not a whole forward

A whole-model comparison would fold in attention, norms and sampling, which is
what we already have end to end. Timing the projections alone isolates the one
op that dominates prefill FLOPs and is directly comparable across stacks —
`2 · M · K · N` is the same arithmetic wherever it runs.

Usage:
    ~/.venv-vllm-metal/bin/python mlx_gemm_roofline.py --tokens 2048 --layers 28
"""

import argparse
import time

import mlx.core as mx


def timed(fn, reps, warmup=3):
    """Min-of-reps wall time for a GPU op, with eval() forcing completion.

    Min rather than mean: on a shared laptop the fastest run is the one least
    perturbed by other work, and we want the kernel's capability, not the
    machine's current mood.
    """
    for _ in range(warmup):
        mx.eval(fn())
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        best = min(best, time.perf_counter() - t0)
    return best


def quantized(k, n, group, bits):
    """A [n, k] weight, quantized the way the checkpoint is."""
    w = mx.random.normal((n, k)).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=group, bits=bits)
    mx.eval(wq, scales, biases)
    return wq, scales, biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048, help="M — prefill width")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--intermediate", type=int, default=3072)
    ap.add_argument("--q-out", type=int, default=2048, help="n_heads * head_dim")
    ap.add_argument("--kv-out", type=int, default=1024, help="n_kv_heads * head_dim")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--pie-forward-ms", type=float, default=None,
                    help="pie's measured GPU time for one forward of --tokens, "
                         "from PIE_METAL_TIMING forward_wait_ns")
    args = ap.parse_args()

    M, H, I = args.tokens, args.hidden, args.intermediate
    # (name, K, N) — one transformer layer's dense projections.
    projections = [
        ("q_proj", H, args.q_out),
        ("k_proj", H, args.kv_out),
        ("v_proj", H, args.kv_out),
        ("o_proj", args.q_out, H),
        ("gate_proj", H, I),
        ("up_proj", H, I),
        ("down_proj", I, H),
    ]

    x_cache = {}
    total_s = 0.0
    total_flop = 0.0
    print(f"MLX quantized_matmul, M={M}, group={args.group}, bits={args.bits}, "
          f"device={mx.default_device()}")
    print(f"{'projection':>12} {'K':>6} {'N':>6} {'ms':>8} {'TFLOPS':>8}")
    for name, K, N in projections:
        if K not in x_cache:
            x_cache[K] = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x_cache[K])
        x = x_cache[K]
        wq, scales, biases = quantized(K, N, args.group, args.bits)
        fn = lambda x=x, wq=wq, s=scales, b=biases: mx.quantized_matmul(
            x, wq, s, b, transpose=True, group_size=args.group, bits=args.bits
        )
        secs = timed(fn, args.reps)
        flop = 2.0 * M * K * N
        total_s += secs
        total_flop += flop
        print(f"{name:>12} {K:>6} {N:>6} {secs*1e3:>8.3f} {flop/secs/1e12:>8.2f}")

    per_layer_ms = total_s * 1e3
    model_ms = per_layer_ms * args.layers
    model_flop = total_flop * args.layers
    print()
    print(f"one layer  : {per_layer_ms:8.3f} ms   ({total_flop/total_s/1e12:.2f} TFLOPS)")
    print(f"{args.layers} layers : {model_ms:8.1f} ms   "
          f"({model_flop/1e12:.2f} TFLOP at {model_flop/(model_ms/1e3)/1e12:.2f} TFLOPS)")

    if args.pie_forward_ms:
        print()
        print(f"pie forward (measured GPU time) : {args.pie_forward_ms:8.1f} ms")
        print(f"MLX projections alone           : {model_ms:8.1f} ms")
        print(f"ratio                           : {args.pie_forward_ms/model_ms:8.2f}x")
        print()
        print("If the ratio is large, pie's whole forward costs several times what")
        print("the same GEMMs cost MLX -- the kernel is the gap. If it is near 1,")
        print("the GEMMs are comparable and pie's deficit is elsewhere in the")
        print("forward (attention, norms, layout).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def bench_sdpa(M, heads, kv_heads, head_dim, reps=10):
    """Time MLX's fused attention at prefill width — the quadratic term.

    Separated from the GEMM bench because the two scale differently (O(n) vs
    O(n^2)) and, as it turns out, are behind by very different factors.
    """
    q = mx.random.normal((1, heads, M, head_dim)).astype(mx.float16)
    k = mx.random.normal((1, kv_heads, M, head_dim)).astype(mx.float16)
    v = mx.random.normal((1, kv_heads, M, head_dim)).astype(mx.float16)
    mx.eval(q, k, v)
    scale = head_dim ** -0.5
    fn = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    return timed(fn, reps)
