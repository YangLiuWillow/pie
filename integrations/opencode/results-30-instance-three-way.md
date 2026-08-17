# Three engines, 30 SWE-bench Verified instances — 2026-08-17

Accuracy, throughput and per-call latency for pie, mlx-lm and vLLM-metal on the
same instances, agent, machine and weights family. Every arm restarted its
server per instance; every call went through one proxy recording TTFT, decode
wall and the server's own token counts.

    Qwen3-Coder-30B-A3B-4bit · Apple M5 Pro 48 GB · stock opencode
    graded by swebench.harness.run_evaluation in Docker

## Read this first: 20 of the 30 instances were solved by nobody

| engine | overall | on the original 10 | on the 20 new |
|---|---:|---:|---:|
| pie | 5/30 | **5/10** | **0/20** |
| mlx-lm | 4/30 | 4/10 | **0/20** |
| vLLM-metal | 4/30 | 4/10 | **0/20** |

The first ten instances are django-only and were inherited from an earlier run
that had selected them for solvability. The twenty added here span eleven other
repositories — astropy, matplotlib, seaborn, flask, requests, xarray, pylint,
pytest, scikit-learn, sphinx, sympy — and **no engine resolved a single one**.

So this is not a 30-instance result. It is a 10-instance result on a set chosen
to be winnable, plus twenty zeros. **The honest reading of the headline numbers
is that they measure a biased subset**, and the unbiased subset says this
model-and-agent combination resolves approximately none of SWE-bench Verified.
Any future run should draw its instances at random and expect low single digits.

## Accuracy

| engine | resolved | which |
|---|---:|---|
| **pie** | **5/30** | 11099, 12276, 13089, 14373, 15569 |
| mlx-lm | 4/30 | 11099, 11133, 13089, 14373 |
| vLLM-metal | 4/30 | 12276, 13089, 14373, 15569 |

One instance of spread. That is **parity, not a ranking** — a two-instance swing
reorders it, and the loop rates below show a single instance flipping on repeat
runs of one unchanged configuration. Only `django-13089` falls to all three.

pie resolved `django-12276` here, which it lost to a loop the previous night.
That is the same coin-flip instance measured at a 20% loop rate, not an
improvement.

## Throughput

| engine | tok/s (weighted) | calls | calls/instance | wall |
|---|---:|---:|---:|---:|
| **pie** | **42.0** | 370 | **12.3** | 1h 25m |
| mlx-lm | 35.3 | 712 | 23.7 | 1h 33m |
| vLLM-metal | 17.0 | 688 | 22.9 | 3h 38m |

**pie is 1.19× mlx-lm and 2.47× vLLM-metal on sustained generation**, and needs
roughly half the agent round-trips to cover the same work.

That second property is the one no instance timing can show. Agent wall clock is
`turns × (prefill + decode)`; mlx-lm and vLLM both take ~23 calls per instance
where pie takes 12, so an engine can lose wall clock by provoking more turns
without being slower per call. mlx-lm is within 8 minutes of pie overall while
running 1.9× as many calls.

Throughput is token-weighted — `sum(completion_tokens)/sum(decode_s)` — because
a per-call rate on agentic traffic is dominated by two-token tool replies and
reaches four figures on a model whose bandwidth caps near 172 tok/s. Zero calls
across all three arms had degraded usage records, so every token count is the
server's own.

## Latency (seconds per call)

| engine | TTFT med | TTFT p90 | TTFT max | call med | call p90 | prompt med |
|---|---:|---:|---:|---:|---:|---:|
| pie | 0.97 | **13.12** | 57.88 | **2.65** | 22.31 | 11,433 |
| mlx-lm | 0.86 | **4.21** | 31.19 | 4.14 | **7.20** | 27,415 |
| vLLM-metal | **0.68** | 6.91 | 114.97 | 6.64 | 24.29 | 23,248 |

**pie wins the median call and loses the tail.** Its p90 TTFT is 3.1× mlx-lm's,
which matches the fixed-prompt finding that pie's decode falls to 0.78× of
mlx-lm at 28k context while leading at 16k and below.

The prompt-length column is a confound, not a result: pie's median prompt is
11.4k tokens against mlx-lm's 27.4k. The three engines took different
trajectories, so they were not asked the same questions, and the latency columns
are therefore not a controlled comparison. The fixed-prompt interleaved
measurement in `results-accuracy-three-way.md` is the controlled one.

## Degeneration: every engine loops on about a third of instances

| engine | looped | rate | worst |
|---|---:|---:|---|
| pie | 10/30 | **33%** | 140× (pylint-4551) |
| mlx-lm | 11/30 | **37%** | 263× (pylint-4604) |
| vLLM-metal | 13/30 | **43%** | 185× (django-11133) |

All three sit in the same band. That settles the attribution: degenerate
repetition under greedy decoding is **the checkpoint's behaviour, not any
engine's defect**.

It also corrects a number from the previous night. A 20-run repeat measurement
put pie's loop rate at 5% pooled and called a graded run's 40% an unexplained
discrepancy. At n=30 the rate is 33%, so **the 5% figure was the outlier** — it
came from four instances, three of which happen to be low-rate ones.

A loop makes a run *faster*, since it reaches the token cap sooner. Nothing in
the throughput or latency columns above can distinguish it from speed, which is
why `tools/transcript_health.py` runs beside them.

## Caveats

1. **Twenty of thirty instances were solved by nobody** — see the top. The
   headline numbers describe a set selected for solvability.
2. **n = 10 effective.** One instance moves any arm by 10 points on that subset.
3. **The weights are identical across engines.** Verified byte-level; see
   the note below the caveats.
4. **Latency is uncontrolled** — different trajectories meant different prompt
   lengths per engine.
5. **Nothing here separates agent from engine.** A resolved instance is both.

## Reproduce

```sh
integrations/opencode/tools/swe30_three_way.sh          # three arms, proxied
integrations/opencode/tools/swe_grade_overnight.sh      # SWE_OUT=/tmp/swe30
integrations/opencode/tools/swe30_report.py /tmp/swe30 --run-tag swe30
```

Grading needs `export PATH=$HOME/.local/lima/bin:$PATH` — colima bundles its own
lima there, and without it `colima status` reports `lima not found` while the VM
is already running.

Restarts gate on `tools/wait_for_memory.sh`, not a fixed sleep: two earlier
attempts died at instances 3 and 14 because macOS releases a 22.5 GiB model's
pages well before it reclaims them.

### On the weights being identical

**All three engines serve the same weights.** pie's artifact is
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` repacked into its container
format, not a separate conversion: 17.21 GB against the source's 17.20 GB,
`pie model info` names that repo as the source, and the `--quant` flag that
would requantize was not used. Verified at the byte level -- 512-byte slices
taken from the midpoints of a 4-bit packed weight, its bf16 scales, and the
8-bit router all appear verbatim in the artifact. The quantization parameters
match too, including MLX's per-tensor override of `mlp.gate` to 8 bits at group
64, which pie's driver reports at boot.
