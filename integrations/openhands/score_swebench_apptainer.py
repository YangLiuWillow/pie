"""Score SWE-bench predictions using Apptainer (no Docker required).

Usage:
    PYTHONPATH=vendor/benchmarks .venv/bin/python score_swebench_apptainer.py \
        predictions/pie_agent_qwen3_coder_30b_moe_50.jsonl \
        --report-file predictions/pie_agent_qwen3_coder_30b_moe_50.report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor" / "benchmarks"))

from benchmarks.swebench.apptainer_eval import (
    DEFAULT_APPTAINER_SANDBOX_ROOT,
    run_swebench_evaluation_apptainer,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("predictions", type=Path,
                   help="Path to predictions JSONL (SWE-bench format)")
    p.add_argument("--report-file", type=Path, default=None,
                   help="Output report JSON (default: <predictions>.report.json)")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--split", default="test")
    p.add_argument("--timeout", type=int, default=3600,
                   help="Per-instance test timeout in seconds")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of workers (Apptainer currently runs sequentially)")
    p.add_argument("--score-dir", type=Path, default=None,
                   help="Directory for per-instance scoring artifacts "
                        "(default: <predictions_dir>/apptainer_eval)")
    p.add_argument("--sandbox-root", type=Path,
                   default=DEFAULT_APPTAINER_SANDBOX_ROOT,
                   help="Reusable Apptainer sandbox root")
    p.add_argument("--apptainer-cache", type=Path, default=None,
                   help="APPTAINER_CACHEDIR for image pulls")
    args = p.parse_args()

    if not args.predictions.exists():
        print(f"ERROR: {args.predictions} not found", file=sys.stderr)
        return 1

    report_file = args.report_file or args.predictions.with_suffix(".report.json")

    print(f"Predictions: {args.predictions}")
    print(f"Report:      {report_file}")
    print(f"Score dir:   {args.score_dir or args.predictions.parent / 'apptainer_eval'}")
    print(f"Sandbox:     {args.sandbox_root}")
    print(f"Timeout:     {args.timeout}s")
    print()

    run_swebench_evaluation_apptainer(
        predictions_file=args.predictions,
        report_file=report_file,
        dataset=args.dataset,
        split=args.split,
        timeout_seconds=args.timeout,
        workers=args.workers,
        score_dir=args.score_dir,
        sandbox_root=args.sandbox_root,
        apptainer_cache=args.apptainer_cache,
    )

    import json
    report = json.loads(report_file.read_text())
    print()
    print(f"=== Results ===")
    print(f"Total:      {report['total']}")
    print(f"Resolved:   {report['resolved']}")
    print(f"Unresolved: {report['unresolved']}")
    if report["resolved_ids"]:
        print(f"Resolved:   {', '.join(report['resolved_ids'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
