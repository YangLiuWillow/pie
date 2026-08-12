# OpenClaw ↔ Pie integration — progress log

Companion to `openclaw-integration.md` (two-strategy spec) and
`openclaw-integration-plan.md` (task-level plan). One entry per completed
task, newest first. Worktree: `Lin_startup/pie-openclaw`, branch
`liu/openclaw-integration` (from `liu/opencode-integration` @ `1053a7790`).
Builds: `CARGO_TARGET_DIR=$HOME/Desktop/Lin_startup/pie/target`.

## Task board

| id | task | status |
|---|---|---|
| oc-P0.1 | Source-derived OpenClaw wire audit → AUDIT.md | pending |
| oc-P0.2 | Real wire captures + fixture bank | pending |
| oc-P0.3 | Renderer parity on OpenClaw fixtures | pending |
| oc-PA.1 | Keyed sticky affinity in gateway (shared w/ opencode track) | pending |
| oc-PA.2 | `extensions/pie/` bundled provider in OpenClaw | pending |
| oc-PA.3 | Acceptance suite + e2e + A/B vs Ollama/llama.cpp | pending |
| oc-PB.1 | Shared session dialect (design + crate) | pending |
| oc-PB.2 | `openclaw-session` inferlet | pending |
| oc-PB.3 | `createStreamFn` WS transport | pending |
| oc-PB.4 | Strategy B measurement | pending |
| oc-PB.5 | Optional depth (grammar calls, speculation, embeddings) | pending |

## Log

### 2026-08-11 — project setup
- Codebase surveys completed (OpenClaw provider layer; pie dev serving
  surface); two-strategy spec written (`openclaw-integration.md`), reusing the
  opencode branch's infrastructure per its progress log.
- Worktree `pie-openclaw` created on new branch `liu/openclaw-integration`
  from `liu/opencode-integration` @ `1053a7790`.
- Detailed plan (`openclaw-integration-plan.md`) + this log committed.
