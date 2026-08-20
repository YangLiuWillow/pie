# Kickoff prompt for a fresh Claude Code session

Paste one of these. They are scoped to a single phase on purpose — a session
given the whole plan will optimise for visible progress and skip the measurement
phases the later ones depend on.

---

## Session 1 — Phase 0 only

```
Read CLAUDE.md and driver/metal/docs/m5-pro-bringup-plan.md.

Execute Phase 0 (T0.1–T0.4) only. Before running anything, work through the
"Verify before executing" section at the bottom of the plan and correct the plan
file in place with what you find — the test target names, roofline_probe
invocation, and script arguments are inferred, not confirmed.

Fill the baseline tables in driver/metal/docs/m5-pro-measurements.md. Raw CSV
from the harnesses stays local — *.csv is gitignored repo-wide; the committed
record is the markdown table with its method.

Deliver each task (T0.1, T0.2, …) as its own branch and PR per the
"Workflow — one task, one branch, one PR" section in CLAUDE.md.

Do not start Phase 1. Do not modify driver/metal/src/device_tuning.cpp.
```

---

## Session 2 — Phase 1 only

```
Read CLAUDE.md, driver/metal/docs/m5-pro-bringup-plan.md, and
driver/metal/docs/m5-pro-measurements.md.

Execute Phase 1 (T1.1–T1.3). Read the header of benches/tune_device.py in full
before running it — its documented methodology error (sweeping a threshold at
batches that do not straddle it) is the failure to avoid.

The device_tuning.hpp invariant is non-negotiable: a default-constructed
DeviceTuning must still reproduce the M1 Max numbers exactly. Assert it with a
test, not by inspection.

Every field you override must carry its own measurement table in the house style
already used in that file.
```

---

## Session 3 — Phase 2 only

```
Read CLAUDE.md and driver/metal/docs/m5-pro-bringup-plan.md.

Execute Phase 2 (T2.1–T2.2): a standalone probe in driver/metal/tools/rawmetal/
comparing mpp::matmul2d against the existing simdgroup_matrix path, then the
same at fp8 and int8.

This is a measurement phase. Do not write any cooperative-tensor kernel for the
model path — Phase 3 is gated on the ratio this produces.

Record results in driver/metal/docs/m5-pro-measurements.md and state the Phase 3
go/no-go explicitly.
```

---

## Standing instructions worth repeating in any session

- **Success metrics are the contract.** A task is not done without the number
  its metric names. "Looks right" is not a measurement.
- **Do not upgrade the pinned baselines** (llama.cpp b9960, mlx-lm 0.31.3, MLX
  0.32.0). They match BaseRT §4.1; upgrading invalidates the comparison.
- **Phases are a dependency graph.** T2.1's ratio gates Phase 3; Phase 1's
  constants feed Phase 3's dispatch; Phase 0's baseline makes every later number
  interpretable.
- **Kernels must be self-contained** — `newLibraryWithSource` does no include
  resolution.
- **Undispatched kernel template instantiations are intentional.** They keep
  closed occupancy sweeps re-runnable. Do not remove them as dead code.
- **Do not vendor or disassemble BaseRT's engine binary.** It is proprietary;
  only its CLI, format, headers, and bindings are Apache-2.0.
- **Record measurements in `driver/metal/docs/`**, and reference them rather than
  restating numbers inline where they will drift.
