"""CLI wrapper around ``benchmarks.humanevalfix``.

A lighter smoke-test benchmark than SWE-Bench: single-file bug fixes, no repo
clone, no Docker (scoring runs locally, in-process, right after the agent
finishes). Good for a fast pass/fail signal on whether the OpenHands <-> Pie
tool-calling path is working before spending SWE-Bench's much longer
per-problem latency.

Examples:

  # Pure plumbing smoke — no model, no GPU needed
  python -m benchmarks.run_humanevalfix \\
      --backend test --subset-size 3 --output /tmp/heval_test.jsonl

  # Drive a 20-problem subset through PieLLM
  # (pie serve must be running, inferlet installed)
  python -m benchmarks.run_humanevalfix \\
      --backend pie \\
      --pie-uri ws://127.0.0.1:8080 \\
      --subset-size 20 \\
      --output predictions/humanevalfix_pie.jsonl \\
      --label pie+qwen2.5-coder-7b

  # Drive the same subset through a vanilla OpenAI-compatible endpoint (baseline)
  python -m benchmarks.run_humanevalfix \\
      --backend litellm \\
      --model openai/qwen2.5-coder-7b \\
      --base-url http://localhost:8000/v1 \\
      --subset-size 20 \\
      --output predictions/humanevalfix_litellm.jsonl \\
      --label litellm+qwen2.5-coder-7b
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from benchmarks import humanevalfix


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["pie", "litellm", "test"], default="test")

    # Subset selection
    g = p.add_mutually_exclusive_group()
    g.add_argument("--subset-size", type=int, default=humanevalfix.DEFAULT_SUBSET_N,
                   help="Deterministic random subset size (default 20; full set is 164).")
    g.add_argument("--task-id", action="append", default=[],
                   help="Specific task_id(s), e.g. Python/0; may be passed multiple "
                        "times. Overrides --subset-size.")

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
    p.add_argument("--pie-render-strategy", default="hf_chat_template",
                   choices=["hf_chat_template", "raw_concat"])
    p.add_argument("--pie-request-timeout-s", type=float, default=300.0,
                   help="Per-completion timeout in seconds (Pie backend only). "
                        "Default is much lower than SWE-Bench's since these are "
                        "single-file, single-function tasks.")

    # Run shape
    p.add_argument("--max-iterations", type=int, default=15,
                   help="Cap on agent steps (default 15 — a single-function fix "
                        "needs far fewer than SWE-Bench's default of 50).")
    p.add_argument("--max-stuck-retries", type=int, default=2)
    p.add_argument("--score-timeout-s", type=float, default=10.0,
                   help="Timeout for the held-out-test subprocess that scores "
                        "each fix (default 10s).")
    p.add_argument("--output", "-o", type=Path, required=True,
                   help="Output results JSONL file.")
    p.add_argument("--label", default=None,
                   help="model_name_or_path label in the results JSONL.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-completions", type=Path, default=None,
                   help="If set, write raw LLM request/response JSON per completion "
                        "to this folder (openhands.sdk.llm.LLM's log_completions).")
    p.add_argument("--native-tool-calling", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Override openhands.sdk.llm.LLM's native_tool_calling "
                        "(default True upstream; PieLLM hardcodes False). Pass "
                        "--no-native-tool-calling on the litellm backend for an "
                        "apples-to-apples comparison against PieLLM.")

    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    backend_kwargs: dict = {}
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
        backend_kwargs["pie_render_strategy"] = args.pie_render_strategy
        backend_kwargs["pie_request_timeout_s"] = args.pie_request_timeout_s
        if args.model:
            backend_kwargs["model"] = args.model

    options = humanevalfix.RunOptions(
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        subset_size=args.subset_size,
        task_ids=args.task_id or None,
        max_iterations=args.max_iterations,
        max_stuck_retries=args.max_stuck_retries,
        score_timeout_s=args.score_timeout_s,
        output_path=args.output,
        label=args.label or f"{args.backend}+{args.model or 'default'}",
    )

    humanevalfix.run(options)
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
