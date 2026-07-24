"""Pie backend: launch the completion inferlet and collect its Return payload.

This mirrors the proven round trip in `pie_openhands.llm.PieLLM._call_pie`
(from the openhands integration): every request opens its own connection,
authenticates, launches a fresh process, and drains events until the Return.
The inferlet is torn down between calls — KV pinning across turns is a later
phase (`openhands-coder-session`); see README §Roadmap.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pie_client import Event, PieClient


def parse_return_value(value: Any, stdout_chunks: list[str]) -> dict[str, Any]:
    """Normalize a Pie inferlet's Return payload into a dict.

    Inferlets may return a JSON-encoded string, an already-decoded dict, a plain
    string (→ `{"text": ...}`), or None (→ fall back to concatenated stdout).
    Ported from `pie_openhands.llm._parse_return_value`.
    """
    if value is None:
        return {"text": "".join(stdout_chunks)}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "replace")
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"text": value}
    return {"text": str(value)}


async def complete(
    *,
    pie_uri: str,
    pie_username: str,
    inferlet: str,
    input_payload: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    """Run one completion through Pie and return the inferlet's Return dict."""
    async with PieClient(pie_uri) as client:
        await client.authenticate(pie_username)
        proc = await client.launch_process(inferlet, input=input_payload)

        stdout_chunks: list[str] = []
        while True:
            event, value = await asyncio.wait_for(proc.recv(), timeout=timeout_s)
            if event == Event.Stdout:
                if isinstance(value, (bytes, bytearray)):
                    stdout_chunks.append(value.decode("utf-8", "replace"))
                else:
                    stdout_chunks.append(str(value))
            elif event == Event.Return:
                return parse_return_value(value, stdout_chunks)
            elif event == Event.Error:
                raise RuntimeError(f"Pie inferlet error: {value!r}")
            # Stderr / Message / File events are ignored in this scaffold.
