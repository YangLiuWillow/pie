"""CLI entry point: `python -m pie_openhistoria [options]`.

Every option also reads an env var so the adapter drops cleanly into a launcher
script or container.
"""

from __future__ import annotations

import argparse
import os

from .server import AdapterConfig, run


def _build_config(argv: list[str] | None = None) -> AdapterConfig:
    p = argparse.ArgumentParser(prog="pie-openhistoria")
    p.add_argument("--pie-uri", default=os.environ.get("PIE_URI", "ws://127.0.0.1:8080"))
    p.add_argument("--pie-username", default=os.environ.get("PIE_USERNAME", "local-dev"))
    p.add_argument("--inferlet", default=os.environ.get("PIE_INFERLET", "openhands-completion@0.1.0"))
    p.add_argument("--model-id", default=os.environ.get("PIE_MODEL_ID", "pie"))
    p.add_argument("--timeout", type=float, default=float(os.environ.get("PIE_TIMEOUT_S", "600")))
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = p.parse_args(argv)
    return AdapterConfig(
        pie_uri=args.pie_uri,
        pie_username=args.pie_username,
        inferlet=args.inferlet,
        model_id=args.model_id,
        request_timeout_s=args.timeout,
        host=args.host,
        port=args.port,
    )


def main(argv: list[str] | None = None) -> None:
    run(_build_config(argv))


if __name__ == "__main__":
    main()
