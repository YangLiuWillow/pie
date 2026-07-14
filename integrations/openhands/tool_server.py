"""Lightweight HTTP tool server for the openhands-agent inferlet.

The inferlet runs inside a WASM sandbox (no shell access). It sends
tool-execution requests here via HTTP POST; we run them on the host and
return the observation text.

Start with ``start_tool_server(working_dir)`` → ``(server, port)``.
Stop with ``server.shutdown()``.
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 10_000
BASH_TIMEOUT_S = 120


def start_tool_server(working_dir: str, *, host: str = "127.0.0.1", port: int = 0) -> tuple[HTTPServer, int]:
    """Start the tool server in a background thread.

    Returns ``(server, port)`` where *port* is the OS-assigned port when
    *port* is 0.
    """
    handler = _make_handler(working_dir)
    server = HTTPServer((host, port), handler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


def _make_handler(working_dir: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            from urllib.parse import urlparse, parse_qs, unquote
            parsed = urlparse(self.path)
            if not parsed.path.rstrip("/").endswith("/execute"):
                self._reply(404, {"error": "not found"})
                return
            qs = parse_qs(parsed.query)
            payload = qs.get("payload", [None])[0]
            if not payload:
                self._reply(400, {"error": "missing payload query param"})
                return
            try:
                req = json.loads(unquote(payload))
            except (json.JSONDecodeError, ValueError) as e:
                self._reply(400, {"error": f"bad json in payload: {e}"})
                return
            self._handle_request(req)

        def do_POST(self):
            from urllib.parse import urlparse
            parsed = urlparse(self.path)
            if not parsed.path.rstrip("/").endswith("/execute"):
                self._reply(404, {"error": "not found"})
                return
            body = self._read_body()
            try:
                req = json.loads(body)
            except json.JSONDecodeError as e:
                self._reply(400, {"error": f"bad json: {e}"})
                return
            self._handle_request(req)

        def _read_body(self) -> bytes:
            te = self.headers.get("Transfer-Encoding", "").lower()
            if "chunked" in te:
                return self._read_chunked()
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length)

        def _read_chunked(self) -> bytes:
            buf = bytearray()
            while True:
                line = self.rfile.readline().strip()
                chunk_len = int(line, 16)
                if chunk_len == 0:
                    self.rfile.readline()  # trailing CRLF
                    break
                buf.extend(self.rfile.read(chunk_len))
                self.rfile.readline()  # trailing CRLF
            return bytes(buf)

        def _handle_request(self, req):
            action = req.get("action", "")
            try:
                if action == "bash":
                    result = _exec_bash(req.get("command", ""), working_dir)
                elif action == "edit":
                    result = _exec_edit(
                        req.get("path", ""),
                        req.get("old_str", ""),
                        req.get("new_str", ""),
                        working_dir,
                    )
                elif action == "finish":
                    result = {"observation": "Task finished.", "exit_code": 0}
                else:
                    result = {"observation": f"Unknown action: {action!r}", "exit_code": 1}
            except Exception as e:
                result = {"observation": f"Error: {type(e).__name__}: {e}", "exit_code": 1}
            self._reply(200, result)

        def _reply(self, code: int, body: dict[str, Any]):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    return Handler


def _exec_bash(command: str, working_dir: str) -> dict[str, Any]:
    if not command.strip():
        return {"observation": "(empty command)", "exit_code": 1}
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"observation": f"Command timed out after {BASH_TIMEOUT_S}s", "exit_code": 124}
    output = proc.stdout + proc.stderr
    if len(output) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        output = output[:half] + f"\n\n... ({len(output) - MAX_OUTPUT_CHARS} chars truncated) ...\n\n" + output[-half:]
    return {"observation": output, "exit_code": proc.returncode}


def _exec_edit(path: str, old_str: str, new_str: str, working_dir: str) -> dict[str, Any]:
    if not path:
        return {"observation": "Error: path is required for edit", "exit_code": 1}
    file_path = Path(working_dir) / path.lstrip("/")
    if not old_str:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_str)
        return {"observation": f"File created: {path}", "exit_code": 0}
    if not file_path.exists():
        return {"observation": f"Error: file not found: {path}", "exit_code": 1}
    content = file_path.read_text()
    count = content.count(old_str)
    if count == 0:
        return {"observation": f"Error: old_str not found in {path}", "exit_code": 1}
    if count > 1:
        return {"observation": f"Error: old_str found {count} times in {path} (must be unique)", "exit_code": 1}
    new_content = content.replace(old_str, new_str, 1)
    file_path.write_text(new_content)

    obs_parts = [f"File edited: {path}"]

    # Syntax check for Python files
    if path.endswith(".py"):
        try:
            compile(new_content, path, "exec")
        except SyntaxError as e:
            obs_parts.append(f"WARNING: SyntaxError after edit — {e.msg} (line {e.lineno}). Please fix.")

    # Show context around the edit location
    edit_start = content.find(old_str)
    new_start = edit_start
    lines = new_content.splitlines()
    char_count = 0
    edit_line = 0
    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count > new_start:
            edit_line = i
            break
    ctx_start = max(0, edit_line - 2)
    ctx_end = min(len(lines), edit_line + len(new_str.splitlines()) + 2)
    context_lines = lines[ctx_start:ctx_end]
    snippet = "\n".join(f"  {ctx_start + j + 1:4d} | {l}" for j, l in enumerate(context_lines))
    obs_parts.append(f"Context:\n{snippet}")

    return {"observation": "\n".join(obs_parts), "exit_code": 0}
