"""CLI wrapper around ``benchmarks.swe_bench``.

Examples:

  # Drive 1 problem with TestLLM (no model needed) — plumbing smoke
  python -m benchmarks.run_swe_bench \\
      --backend test \\
      --instance-id astropy__astropy-12907 \\
      --output /tmp/preds.jsonl

  # Drive the deterministic 50-problem subset through PieLLM
  # (pie serve must be running, inferlet installed)
  python -m benchmarks.run_swe_bench \\
      --backend pie \\
      --pie-uri ws://127.0.0.1:8080 \\
      --subset-size 50 \\
      --output predictions/pie_qwen3.jsonl \\
      --label pie+qwen3-coder-32b

  # Drive the same subset through a vanilla vLLM OpenAI-compatible endpoint (baseline)
  python -m benchmarks.run_swe_bench \\
      --backend litellm \\
      --model openai/qwen3-coder-32b \\
      --base-url http://localhost:8000/v1 \\
      --subset-size 50 \\
      --output predictions/vllm_qwen3.jsonl \\
      --label vllm+qwen3-coder-32b
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from benchmarks import swe_bench


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["pie", "litellm", "test"], default="test")

    # Subset selection
    g = p.add_mutually_exclusive_group()
    g.add_argument("--subset-size", type=int, default=swe_bench.DEFAULT_SUBSET_N,
                   help="Deterministic random subset size (default 50).")
    g.add_argument("--instance-id", action="append", default=[],
                   help="Specific instance_id(s); may be passed multiple times. "
                        "Overrides --subset-size.")

    # Backend tuning
    p.add_argument("--model", default=None,
                   help="Backend model name (for litellm / pie 'model' field).")
    p.add_argument("--base-url", default=None,
                   help="LiteLLM base_url (e.g. http://localhost:8000/v1 for a vLLM server).")
    p.add_argument("--api-key", default=None,
                   help="LiteLLM API key. If unset, uses LITELLM_API_KEY env var.")
    p.add_argument("--pie-uri", default="ws://127.0.0.1:8080",
                   help="Pie WebSocket URI.")
    p.add_argument("--pie-inferlet", default="openhands-completion@0.1.0")
    p.add_argument("--pie-session", action="store_true",
                   help="Keep each conversation's prompt KV alive across calls "
                        "via a named Pie context (requires a session-capable "
                        "inferlet, e.g. openhands-coder-session@0.1.0). Only "
                        "the token delta since the previous call is prefilled; "
                        "history rewrites fall back to a full rebuild.")
    p.add_argument("--kv-verify", action="store_true",
                   help="Fidelity mode (with --pie-session): the inferlet "
                        "asserts on every call that the session context's "
                        "accumulated tokens equal the from-scratch prompt "
                        "render, erroring out on mismatch.")
    p.add_argument("--no-grammar", action="store_true",
                   help="Disable the runtime's tool-call grammar constraint "
                        "(pie backend, session-capable inferlet only): the "
                        "model emits its native ChatML tool-call format and "
                        "the inferlet decoder parses it unconstrained — "
                        "parity with an unconstrained vLLM baseline.")
    p.add_argument("--pie-oneshot", action="store_true",
                   help="Launch a fresh inferlet process per LLM call instead "
                        "of keeping one alive for the conversation. This is "
                        "the pre-2026-07-28 transport, kept ONLY so the cost "
                        "of the old design can be measured — it charges a "
                        "websocket connect, an authenticate, a process launch "
                        "and a teardown on every call, none of which vLLM's "
                        "persistent server pays.")
    p.add_argument("--python-tool-parser", action="store_true",
                   help="Parse tool calls host-side with a verbatim port of "
                        "vLLM's qwen3_coder parser (the parser the litellm "
                        "baseline runs) instead of the inferlet's Rust decoder. "
                        "Implies --no-grammar and skips the JSON few-shot "
                        "examples so the model emits native Qwen3-Coder XML. "
                        "Maximizes tool-call parity with the baseline.")
    p.add_argument("--pie-request-timeout-s", type=float, default=1800.0,
                   help="Per-completion timeout in seconds (Pie backend only). "
                        "Default 1800s; one agent step on a CPU model can exceed 600s "
                        "due to the size of OpenHands' system prompt.")
    # Run shape
    p.add_argument("--max-iterations", type=int, default=100,
                   help="Max agent iterations per run (default 100, matching "
                        "official OpenHands evaluation).")
    p.add_argument("--max-stuck-retries", type=int, default=2,
                   help="On StuckDetector firing, send a corrective nudge message "
                        "and resume the run, up to this many times (0 disables).")
    p.add_argument("--max-fake-responses", type=int, default=10,
                   help="Max fake user responses when agent sends content-only "
                        "messages instead of using tools (0 disables).")
    p.add_argument("--no-condenser", action="store_true",
                   help="Disable the LLMSummarizingCondenser (enabled by default).")
    p.add_argument("--output", "-o", type=Path, required=True,
                   help="Output predictions JSONL file.")
    p.add_argument("--label", default=None,
                   help="model_name_or_path label in the predictions JSONL.")
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "swebench-clones",
                   help="Bare-repo cache for fast multi-problem runs.")
    p.add_argument("--resume", action="store_true",
                   help="Skip instances already present in the output file. "
                        "Useful for resuming after preemption.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Number of instances to solve concurrently against one "
                        "Pie server (default 1 = serial). Raising this is the "
                        "primary throughput lever: it fills the runtime's decode "
                        "batch. vLLM self-regulates against KV memory, so "
                        "over-subscribing queues rather than OOMs. Bound in "
                        "practice by host CPU/sandboxes (--cpus-per-task).")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-completions", type=Path, default=None,
                   help="If set, write raw LLM request/response JSON per completion "
                        "to this folder (openhands.sdk.llm.LLM's log_completions).")
    p.add_argument("--native-tool-calling", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Override openhands.sdk.llm.LLM's native_tool_calling "
                        "(default True upstream; PieLLM hardcodes False). Pass "
                        "--no-native-tool-calling on the litellm backend for an "
                        "apples-to-apples comparison against PieLLM's prompt-mocked "
                        "tool calling, without needing --enable-auto-tool-choice on "
                        "the vLLM server.")
    p.add_argument("--temperature", type=float, default=None,
                   help="LLM sampling temperature (0 = greedy). Passed to the "
                        "litellm/pie backend's LLM constructor.")


    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    backend_kwargs: dict = {}
    if args.temperature is not None:
        backend_kwargs["temperature"] = args.temperature
    if args.native_tool_calling is not None:
        backend_kwargs["native_tool_calling"] = args.native_tool_calling
    if args.log_completions:
        args.log_completions.mkdir(parents=True, exist_ok=True)
        backend_kwargs["log_completions"] = True
        backend_kwargs["log_completions_folder"] = str(args.log_completions)
    if args.backend == "litellm":
        if not args.model:
            p.error("--model is required when --backend litellm")
        backend_kwargs["model"] = args.model
        if args.base_url:
            backend_kwargs["base_url"] = args.base_url
        api_key = args.api_key or os.environ.get("LITELLM_API_KEY")
        if api_key:
            from pydantic import SecretStr
            backend_kwargs["api_key"] = SecretStr(api_key)
    elif args.backend == "pie":
        backend_kwargs["pie_uri"] = args.pie_uri
        backend_kwargs["pie_inferlet"] = args.pie_inferlet
        backend_kwargs["pie_request_timeout_s"] = args.pie_request_timeout_s
        if args.pie_session:
            backend_kwargs["pie_session"] = True
        if args.kv_verify:
            backend_kwargs["pie_kv_verify"] = True
        if args.no_grammar:
            backend_kwargs["pie_use_grammar"] = False
        if args.python_tool_parser:
            backend_kwargs["pie_python_tool_parser"] = True
        if args.pie_oneshot:
            backend_kwargs["pie_daemon"] = False
        if args.model:
            backend_kwargs["model"] = args.model

    options = swe_bench.RunOptions(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        subset_size=args.subset_size,
        instance_ids=args.instance_id or None,
        max_iterations=args.max_iterations,
        max_stuck_retries=args.max_stuck_retries,
        max_fake_responses=args.max_fake_responses,
        enable_condenser=not args.no_condenser,
        cache_dir=args.cache_dir,
        output_path=args.output,
        label=args.label or f"{args.backend}+{args.model or 'default'}",
        resume=args.resume,
        concurrency=args.concurrency,
    )

    try:
        swe_bench.run(options)
    except SystemExit as e:
        if e.code == 42:
            print(
                "\nPie server died — aborting. Use --resume to continue "
                "after restarting the server.",
                file=sys.stderr,
            )
        raise
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
