# START HERE — H200 pod

This is the opening brief for whoever (human or agent) picks up the experiment on
the H200. It is deliberately short. The detail lives in
[`AGENT_HANDOVER_H200.md`](AGENT_HANDOVER_H200.md), and this file's only job is to
get you there and stop you making the one mistake that already cost a full run.

---

## Before the agent starts (human, once per pod)

0. **Check the pod before you pay for it.** Two independent requirements, and a
   pod that satisfies one and not the other looks fine until it wastes an hour:

   ```bash
   nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
   ```

   | need | makes valid | wrong value costs you |
   |---|---|---|
   | compute cap **9.0** | the **Pie** arm | the A100 failure — an invalid Pie arm |
   | driver **≥ 580** | the **vLLM** arm | vLLM installs cleanly, then cannot start |

   Driver r570 caps you at CUDA 12.8; `vllm >= 0.20` needs CUDA 13. Changing the
   container image does **not** fix this — see `AGENT_HANDOVER_H200.md` §1b. On
   RunPod, set the CUDA-version filter to 13.0+ *before* picking the GPU.

1. **Attach the same network volume** if `us-md-1` has H200 capacity — you keep
   the ~57 GB model and the repo. `/root` (venvs, cargo build tree, caches) is
   lost regardless; `00_setup_h200.sh` rebuilds it. If the volume cannot follow,
   the script clones and re-downloads, adding roughly an hour.
2. **Pull the branch.** Everything is pushed to
   `origin/openhands-integration-updated`. A stale checkout would not even
   contain the H200 files.

   ```bash
   git -C /workspace/pie pull
   ```

---

## The message to give the agent

> You're taking over a benchmark experiment on a fresh H200 pod. The repo is at
> `/workspace/pie` on branch `openhands-integration-updated` — run `git pull`
> first.
>
> **Read `integrations/openhands/runpod/AGENT_HANDOVER_H200.md` before doing
> anything else.** Then `TEST_PLAN.md` for the design and
> `SESSION_NOTES_20260727.txt` for what the previous run found.
>
> The short version of why you're on this GPU: the previous run was on an A100
> and produced an **invalid Pie arm**. Pie's fast attention paths require CUDA
> compute capability >= 9 (`driver/cuda/src/entry.cpp:989`,
> `driver/cuda/src/ops/attention_xqa.cu:274`) and the A100 is major 8, so both
> were silently off for the whole run. The driver said so in its own startup
> banner and it was read past. Do not repeat that.
>
> **But sm_90 alone is not enough, and the handover used to claim it was.**
> `xqa_decode_bf16_supported()` has **seven** conditions and the arch check is
> only the last. On a stock H200 the KV page size fails: the memory planner's
> `auto` mode picks `page_size=16` and silently overrides `kv_page_size = 32` in
> the toml, while the gqa8 XQA kernel needs `TOKENS_PER_PAGE=32`. The fix is
> `PIE_CUDA_KV_PAGE_SIZE=32`, which the setup script now exports. Read
> `AGENT_HANDOVER_H200.md` §1a — this is the trap that is *still* live.
>
> **Start with `bash integrations/openhands/runpod/00_setup_h200.sh`.** It is
> idempotent and refuses to continue unless the Pie driver banner reads
> `prefill_decode_plan=on xqa_decode=on`. Check the planner line it prints
> alongside — you need **`page_size=32`** too. If it does not print PASS, stop
> and tell me — do not work around it.
>
> Then run the arms in the order in §5, one at a time, verifying each before
> starting the next. **You have standing authorization for the whole sequence —
> do not ask between arms.** §8 is the escalate list; §5 flags one required edit
> to `30_ab_run.sh` that is pre-authorized.
>
> Report per §7: **s/iter and median per-call latency, never raw wall time**; the
> `assert_vllm_fair.sh` banner verbatim for each vLLM tier; the Pie driver
> feature banner verbatim; prefill-reuse % and kv-verify error count for the Pie
> arm; and `max_prompt_tokens` against the 131072 cap.
>
> Two things not to do: do not compare against the A100 Pie numbers (they are
> void — kept only for the record), and do not attempt accuracy scoring
> (deferred, and impossible on a GPU pod: no docker, no apptainer, no swebench).

---

## Push to the fork only

`YangLiuWillow/pie` is a **fork of `pie-project/pie`**. This experiment's commits
belong on the fork. Nothing here should ever land upstream without a deliberate,
separate decision by the human.

Two guards, both reinstalled by `00_setup_h200.sh` because neither survives a
fresh clone (`.git/hooks` is not versioned):

- `remote.pushDefault = origin` — a bare `git push` targets the fork regardless
  of branch tracking.
- `.git/hooks/pre-push` (tracked copy at `git-hooks/pre-push`) — **refuses any
  push whose target URL is not `YangLiuWillow/pie`**, in HTTPS or SSH form.
  Verified against `pie-project/pie`, `git@github.com:pie-project/pie.git` and a
  third-party URL; all blocked.

If you deliberately want to publish upstream, that is `--no-verify` plus a
conversation with the human first — it is on the escalate list.

Note that **opening a pull request is not a push**. The fork's default PR target
is upstream, so if you use `gh pr create`, pass `--repo YangLiuWillow/pie`
explicitly or you will propose the branch to `pie-project/pie`.

---

## Why this brief is short

`AGENT_HANDOVER_H200.md` is ~17 KB and carries the storage rule, the version
pins, the MoE-config inversion, the gotchas and the escalation policy. A long
opening prompt competes with it and invites acting on a summary instead of the
source. This brief's job is to get §1 read and §0 run; the document does the rest.

The one thing deliberately duplicated here is the **arch-gate story**, because an
agent that has not internalised *why* the last run failed is exactly the one who
will see `prefill_decode_plan=off`, decide it looks unimportant, and continue.

---

## The 30-second version, if you read nothing else

| | |
|---|---|
| **Check first** | driver **≥ 580** *and* compute cap **9.0** — different arms depend on each |
| **Do first** | `bash integrations/openhands/runpod/00_setup_h200.sh` |
| **Must see** | `prefill_decode_plan=on xqa_decode=on` **and** `page_size=32` — otherwise STOP |
| **Never** | collect a Pie arm when that banner reads `off` |
| **Don't set** | `VLLM_TUNED_CONFIG_FOLDER` — H200 ships its own MoE config |
| **Report** | s/iter and median per-call latency, **never** raw wall time |
| **Void** | the A100 Pie arm — do not compare against it |
