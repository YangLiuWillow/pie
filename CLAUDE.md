# Pie — working notes for Claude Code

Programmable LLM serving. Inferlets (WebAssembly guest programs) run custom
inference logic alongside the model; the engine is not modified to add an
algorithm.

> **`compiler/` builds a program, `runtime/` schedules it, `driver/` fires it.**

This file is a first pass. Refine it with `/init` and by reading the READMEs
named below — several are better than anything summarised here.

---

## Layout

| Path | What lives there |
|---|---|
| `compiler/` | PTIR toolchain: `dsl` (authoring) → `ir` (representation) → `plan` (analysis) → `codegen` (CUDA/Metal emission), plus `eval`, the tier-0 reference interpreter **every backend is diffed against**. See `compiler/README.md`. |
| `driver/metal/` | The Metal driver (C++/ObjC + CMake). Kernels in `src/kernels/*.metal`, per-device constants in `src/device_tuning.{hpp,cpp}`, MTL4 plumbing in `src/mtl4_context.mm`. |
| `driver/cuda/`, `driver/dummy/`, `driver/transport/` | Other drivers. |
| `loader/` | Weight-loading compiler: checkpoints → verified device-memory layouts. Quantization is spelled as a `Cast` into a quantized encoding in the contract. See `loader/README.md`. |
| `model/<generation>/` | One crate per model generation (`llama_3`, `qwen_3`, `qwen_3_5`, `gemma_4`, `deepseek_v4`, …). Adding a model generation means adding a crate. |
| `runtime/` | `engine`, `tokenizer`, `grammar`, `waker`. |
| `controller/`, `gateway/`, `worker/` | Role libraries (pure libs, never depend on each other). |
| `interface/` | Boundary-named ABI contracts + cbindgen tools. |
| `benches/` | Python benchmark + tuning harnesses (see below). |
| `tests/gpu/` | On-device end-to-end suite; mostly `#[ignore]`d, needs a real device. |

---

## Build and test

Toolchain is pinned in `rust-toolchain.toml` to **1.97.1**, edition 2024, with
the `wasm32-wasip2` target. Do not bump it casually — the `-D warnings` clippy
gate is calibrated against that exact release. Workspace MSRV floor is 1.91 and
is a *measurement*: re-measure rather than edit it to match what's installed.

```sh
cargo build --workspace --all-targets --exclude pie-server-py
cargo test  --workspace

# compiler crates are the only ones at zero warnings — CI gates them
cargo clippy --no-deps -p pie-ir -p pie-plan -p pie-eval -p pie-codegen \
  -p pie-dsl -p pie-compiler-tests --all-targets -- -D warnings
cargo fmt --check -p pie-ir -p pie-plan -p pie-eval -p pie-codegen -p pie-dsl

# loader, no GPU needed
cargo test -p pie-loader -p pie-loader-capi
UPDATE_GOLDEN=1 cargo test -p pie-loader --test golden_plans

# guest inferlets
cargo check --target wasm32-wasip2 --manifest-path tests/inferlets/Cargo.toml
```

The rest of the workspace still carries ~240 clippy warnings. Wire a crate into
`[workspace.lints]` once it is clean; do not attempt a bulk cleanup.

**Generated artifacts are checked in and drift-tested. Never hand-edit them.**
`compiler/codegen/include/`, `interface/driver/…/pie_driver_abi.h`,
`loader/capi/include/pie_loader.h`. Regenerate:

```sh
cargo run -p pie-driver-abi-cbindgen
cargo run -p pie-loader-cbindgen
```

---

## Metal driver — constraints that bite

**Kernels are compiled at run time through `newLibraryWithSource`, which does no
filesystem include resolution.** Every `.metal` file must be self-contained.
This is why `quantized_qmm_t.metal` vendors the MLX steel GEMM primitives inline
rather than including them.

**MLX is vendored deliberately, with attribution.** `quantized_qmv.metal` and
`quantized_qmm_t.metal` port MLX's `qmv_fast_impl` and `qmm_t_impl` so the math
matches by construction. MLX is MIT-licensed. Keep the provenance comments.

**The GEMM is not bit-identical to the GEMV** — accumulation order differs. Gate
it on token agreement with the reference, never on a hash.

**`driver/metal/src/device_tuning.hpp` has a rule, stated in the file:** a
default-constructed `DeviceTuning` reproduces the M1 Max measurements *exactly*.
Adding a device may never change what an unrecognised device does, and each
override must carry the measurement that justifies it. `benches/tune_device.py`
takes that measurement — and its header documents the methodology error to avoid
(a threshold only means something at batches that straddle it).

Undispatched kernel template instantiations in `quantized_qmm_t.metal` are
**intentional**: they keep closed occupancy sweeps re-runnable. Do not delete
them as dead code.

---

## This machine

Development target is an **Apple M5 Pro, 48 GB unified memory** — the same
configuration BaseRT evaluated on (arXiv:2607.19438 §4.1). Prefer Metal over
CUDA in suggestions, and assume a real device is available for `tests/gpu`.

**`device_tuning.cpp` has no M5 entry.** Apple families through M4 have
overrides; an M5 currently falls back to the M1 Max constants — which is exactly
the failure mode the file's own header warns about ("the GEMM crossover sits
three rows too high, so the batches where the GEMM already wins are still served
by the GEMV"). Measuring and adding that entry is the highest-value,
lowest-effort work available on this machine.

---

## Benchmarking

Harnesses already exist in `benches/`: `pie_bench.py`, `mlx_bench.py`,
`llamacpp_bench.py`, `three_way.py`, `tune_device.py`, `contention_sweep.py`.
Use them rather than writing new ones.

**Pinned baseline versions — do not upgrade:**

| Tool | Version |
|---|---|
| llama.cpp | build **b9960** |
| mlx-lm | **0.31.3** |
| MLX | **0.32.0** |

These match BaseRT §4.1. Upgrading any of them silently invalidates comparison
against their published tables. If a newer version is needed for something
unrelated, use a separate checkout.

**Protocol** (also BaseRT §4.1, so results stay comparable): prompt lengths
128/256/512/1024/2048; 128 generated tokens; 5 repetitions; report mean ± stddev.
Run on AC power, alternate arms rather than batching them, and use medians.

---

## Measurement discipline

This repo already holds the line; keep holding it.

- A measured number supersedes a published one. Published M5 bandwidth figures
  conflict (153 GB/s vs ~120 GB/s) and neither is credible for a Pro tier.
- Record measurements in `driver/metal/docs/` and reference them; do not restate
  numbers inline where they will drift.
- A tuning constant without a recorded measurement behind it is a guess. The
  `device_tuning.hpp` convention — every override carries its own table — is the
  standard for new constants too.
- Do not declare a task done without the number its success metric names.

**Where docs go.** `/docs/` and `*.csv` are gitignored repo-wide (`/docs/` is
"locally archived Markdown"). Tracked documentation lives next to the code it
describes — `compiler/README.md`, `loader/README.md`, `driver/metal/docs/`. Raw
benchmark CSV stays a local artifact; the committed record is a markdown table
carrying its method.

---

## Licensing boundary — BaseRT

BaseRT's **engine is proprietary and binary-only**. Only its CLI, `.base`
format, public headers, and bindings are Apache-2.0 (`basecompute/baseRT`).

- Do not vendor engine code. Do not disassemble the binary.
- Design ideas from the papers (arXiv:2607.00501, arXiv:2607.19438) are free to
  reimplement — cite them.
- Read the engine EULA before publishing comparative benchmark numbers.

Pie itself is Apache-2.0; vendored MLX kernels are MIT. Keep both attributions
intact.

---

## Current work

See `driver/metal/docs/m5-pro-bringup-plan.md` for the active plan and
`driver/metal/docs/` for the constants it produces.

---

## Verified against this clone (2026-08-19 /init pass)

**The tree this file describes does not live on `dev`.** `origin/dev` last
moved 2026-06-24 and contains none of: `compiler/`, `loader/`, `model/`, the
Metal driver sources, `tests/gpu/`, or `benches/tune_device.py` /
`three_way.py` / `mlx_bench.py`. Everything above that names those paths is
true on the kernel branches instead:

- **`liu/qwen38-27b`** — freshest tip (2026-08-19, merges `liu/a-fold-parking`);
  full Metal driver, `device_tuning.cpp` with cases for Apple families 9 and 8
  and **no M5 case** (confirming the plan's gap #1), all bench harnesses.
- `liu/qwen3-coder-kernels`, `liu/a-fold-parking` — same tree, older tips.

The `docs/m5-pro-bringup-plan` branch was cut from `dev` per setup
instructions; the bring-up patch also applies cleanly onto `liu/qwen38-27b`
(none of its four files exist there).

Corrections found while verifying paths the plan cites:

- Roofline/occupancy tooling is at **`driver/metal/tools/rawmetal/`**, not
  `tools/rawmetal/`.
- `tests/gpu` is crate `pie-gpu-tests`: integration tests only, no code of its
  own, and the driver is selected by cargo feature. Run with
  `--features driver-metal` — a bare `cargo test -p pie-gpu-tests -- --ignored`
  builds the **dummy-driver fallback**, not Metal.
- On the kernel branches `/docs/` is still gitignored, but `docs/HANDOVER.md`,
  `docs/NEXT-decode-dispatch-count.md`, and `docs/plan-attention-depage.md`
  are force-added and tracked; the prior session's handover is
  `docs/HANDOVER.md` there.
