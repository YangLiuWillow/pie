"""Lightweight HTTP tool server for the openhands-agent inferlet.

The inferlet runs inside a WASM sandbox (no shell access). It sends
tool-execution requests here via HTTP POST; we run them on the host and
return the observation text.

Uses OpenHands SDK's ``FileEditor`` for file operations (view, create,
str_replace, insert, undo_edit) when available, falling back to a simple
built-in implementation otherwise.

Start with ``start_tool_server(working_dir)`` → ``(server, port)``.
Stop with ``server.shutdown()``.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

MAX_OUTPUT_CHARS = 16_000
BASH_TIMEOUT_S = 120
MAX_EDIT_OLD_LINES = 50
MAX_EDIT_NEW_LINES = 100

import logging as _logging

_HAS_FILE_EDITOR = False
_saved_root_level = _logging.root.level
try:
    from openhands.tools.file_editor.editor import FileEditor
    from openhands.tools.file_editor.exceptions import ToolError as EditorToolError
    _HAS_FILE_EDITOR = True
except ImportError:
    pass
finally:
    _logging.root.setLevel(_saved_root_level)


# ---------------------------------------------------------------------------
# Persistent bash session
# ---------------------------------------------------------------------------

class PersistentBash:
    """A long-lived bash process that preserves cwd and env across commands."""

    def __init__(self, cwd: str):
        self._init_cwd = cwd
        self._proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env={**os.environ, "PS1": "", "TERM": "dumb", "LANG": "C.UTF-8"},
        )
        self._lock = threading.Lock()

    def run(self, command: str, timeout: int = BASH_TIMEOUT_S) -> tuple[str, int]:
        with self._lock:
            if self._proc.poll() is not None:
                self._restart()

            marker = f"___PIE_DONE_{os.urandom(8).hex()}___"
            wrapped = f"{command}\n_pie_ec=$?\nprintf '\\n{marker}%d\\n' \"$_pie_ec\"\n"

            try:
                self._proc.stdin.write(wrapped.encode())
                self._proc.stdin.flush()
            except BrokenPipeError:
                self._restart()
                return "(bash session died, restarted for next command)", 1

            output_lines: list[str] = []
            deadline = time.monotonic() + timeout
            buf = b""
            fd = self._proc.stdout.fileno()

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return (
                        "\n".join(output_lines)
                        + f"\n(command timed out after {timeout}s)",
                        124,
                    )

                ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
                if not ready:
                    continue

                chunk = os.read(fd, 65536)
                if not chunk:
                    # EOF: the command terminated the shell itself (e.g. a
                    # bare `exit 42`). Bash's own exit status *is* the
                    # command's status — harvest it before restarting so the
                    # agent sees the real code, matching one-shot semantics.
                    try:
                        exit_code = self._proc.wait(timeout=5)
                    except Exception:
                        exit_code = 1
                    tail = buf.decode("utf-8", "replace").strip()
                    if tail and marker not in tail:
                        output_lines.append(tail)
                    self._restart()
                    return "\n".join(output_lines), exit_code

                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", "replace")
                    if marker in line:
                        ec_str = line.split(marker)[1].strip()
                        exit_code = int(ec_str) if ec_str.isdigit() else 0
                        return "\n".join(output_lines), exit_code
                    output_lines.append(line)

    def _restart(self):
        # Preserve the agent's cwd across the restart when the old shell is
        # still inspectable; once it's dead /proc is gone, so fall back to
        # the workspace root (not /tmp — the agent must stay in its repo).
        cwd = self._init_cwd
        try:
            link = os.readlink(f"/proc/{self._proc.pid}/cwd")
            if os.path.isdir(link):
                cwd = link
        except Exception:
            pass
        try:
            self._proc.kill()
            self._proc.wait(timeout=2)
        except Exception:
            pass
        self._proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env={**os.environ, "PS1": "", "TERM": "dumb", "LANG": "C.UTF-8"},
        )

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()


def start_tool_server(working_dir: str, *, host: str = "127.0.0.1", port: int = 0) -> tuple[HTTPServer, int]:
    """Start the tool server in a background thread.

    Returns ``(server, port)`` where *port* is the OS-assigned port when
    *port* is 0.
    """
    bash = PersistentBash(working_dir)
    handler = _make_handler(working_dir, bash)
    server = HTTPServer((host, port), handler)
    actual_port = server.server_address[1]
    _orig_shutdown = server.shutdown
    def _shutdown():
        _orig_shutdown()
        bash.close()
    server.shutdown = _shutdown
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port


def _to_abs_path(path: str, working_dir: str) -> str:
    """Resolve *path* to an absolute path rooted at *working_dir*."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(working_dir) / path)


def _make_handler(working_dir: str, bash: PersistentBash):
    editor = FileEditor(workspace_root=working_dir) if _HAS_FILE_EDITOR else None

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
                    result = _exec_bash(bash, req.get("command", ""))
                elif action == "edit":
                    result = _exec_edit(
                        editor,
                        req.get("path", ""),
                        req.get("old_str", ""),
                        req.get("new_str", ""),
                        working_dir,
                    )
                elif action == "read_file":
                    result = _exec_read_file(
                        editor,
                        req.get("path", ""),
                        working_dir,
                        start_line=req.get("start_line"),
                        end_line=req.get("end_line"),
                    )
                elif action == "insert":
                    result = _exec_insert(
                        editor,
                        req.get("path", ""),
                        req.get("insert_line", 0),
                        req.get("new_str", ""),
                        working_dir,
                    )
                elif action == "undo_edit":
                    result = _exec_undo(editor, req.get("path", ""), working_dir)
                elif action == "finish":
                    result = {"observation": "Task finished.", "exit_code": 0}
                elif action == "has_diff":
                    result = _exec_has_diff(bash)
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


# ---------------------------------------------------------------------------
# Bash
# ---------------------------------------------------------------------------

def _exec_bash(bash: PersistentBash, command: str) -> dict[str, Any]:
    if not command.strip():
        return {"observation": "(empty command)", "exit_code": 1}
    output, exit_code = bash.run(command)
    if len(output) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        output = output[:half] + f"\n\n... ({len(output) - MAX_OUTPUT_CHARS} chars truncated) ...\n\n" + output[-half:]
    return {"observation": output, "exit_code": exit_code}


# ---------------------------------------------------------------------------
# Diff check
# ---------------------------------------------------------------------------

def _exec_has_diff(bash: PersistentBash) -> dict[str, Any]:
    """Check whether the working directory has any changes (tracked or untracked)."""
    output, _ = bash.run("git diff HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null")
    has_diff = len(output.strip()) > 0
    return {"observation": str(has_diff).lower(), "has_diff": has_diff, "exit_code": 0}


# ---------------------------------------------------------------------------
# File operations — delegated to OpenHands FileEditor when available
# ---------------------------------------------------------------------------

def _obs_text(obs) -> str:
    """Extract the text string from a ``FileEditorObservation``."""
    if hasattr(obs, "content") and obs.content:
        return obs.content[0].text
    return str(obs)


def _exec_edit(
    editor: "FileEditor | None",
    path: str,
    old_str: str,
    new_str: str,
    working_dir: str,
) -> dict[str, Any]:
    if not path:
        return {"observation": "Error: path is required for edit", "exit_code": 1}

    abs_path = _to_abs_path(path, working_dir)

    # --- Create ---
    if not old_str:
        if editor is not None:
            try:
                obs = editor(command="create", path=abs_path, file_text=new_str)
                return {"observation": _obs_text(obs), "exit_code": 0}
            except EditorToolError:
                pass
        # Fallback: simple write (handles overwrite, which FileEditor rejects)
        file_path = Path(abs_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_str)
        return {"observation": f"File created: {path}", "exit_code": 0}

    # --- str_replace ---
    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    if old_lines > MAX_EDIT_OLD_LINES:
        return {"observation": f"Error: old_str has {old_lines} lines (max {MAX_EDIT_OLD_LINES}). Break into smaller edits.", "exit_code": 1}
    if new_lines > MAX_EDIT_NEW_LINES:
        return {"observation": f"Error: new_str has {new_lines} lines (max {MAX_EDIT_NEW_LINES}). Break into smaller edits.", "exit_code": 1}

    if editor is not None:
        try:
            obs = editor(
                command="str_replace",
                path=abs_path,
                old_str=old_str,
                new_str=new_str,
            )
            text = _obs_text(obs)
            # Python syntax check on top of FileEditor's response
            if path.endswith(".py"):
                new_content = Path(abs_path).read_text()
                try:
                    compile(new_content, path, "exec")
                except SyntaxError as e:
                    text += f"\nWARNING: SyntaxError after edit — {e.msg} (line {e.lineno}). Please fix."
            return {"observation": text, "exit_code": 0}
        except EditorToolError as e:
            return {"observation": f"Error: {e.message}", "exit_code": 1}

    # Fallback: built-in str_replace
    return _exec_edit_builtin(path, old_str, new_str, working_dir)


def _exec_edit_builtin(path: str, old_str: str, new_str: str, working_dir: str) -> dict[str, Any]:
    """Built-in str_replace for when FileEditor is unavailable."""
    file_path = Path(working_dir) / path.lstrip("/")
    if not file_path.exists():
        return {"observation": f"Error: file not found: {path}", "exit_code": 1}
    content = file_path.read_text()
    count = content.count(old_str)
    if count == 0:
        return {"observation": f"Error: old_str not found in {path}. Use read_file to see the actual content.", "exit_code": 1}
    if count > 1:
        return {"observation": f"Error: old_str found {count} times in {path} (must be unique — add more surrounding context)", "exit_code": 1}
    new_content = content.replace(old_str, new_str, 1)
    file_path.write_text(new_content)

    obs_parts = [f"File edited: {path}"]
    if path.endswith(".py"):
        try:
            compile(new_content, path, "exec")
        except SyntaxError as e:
            obs_parts.append(f"WARNING: SyntaxError after edit — {e.msg} (line {e.lineno}). Please fix.")

    edit_start = content.find(old_str)
    lines = new_content.splitlines()
    char_count = 0
    edit_line = 0
    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count > edit_start:
            edit_line = i
            break
    ctx_start = max(0, edit_line - 4)
    ctx_end = min(len(lines), edit_line + len(new_str.splitlines()) + 4)
    context_lines = lines[ctx_start:ctx_end]
    snippet = "\n".join(f"{ctx_start + j + 1:6}\t{l}" for j, l in enumerate(context_lines))
    obs_parts.append(f"Context:\n{snippet}")
    return {"observation": "\n".join(obs_parts), "exit_code": 0}


def _exec_read_file(
    editor: "FileEditor | None",
    path: str,
    working_dir: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, Any]:
    if not path:
        return {"observation": "Error: path is required for read_file", "exit_code": 1}

    abs_path = _to_abs_path(path, working_dir)

    # Directory listing
    if Path(abs_path).is_dir():
        try:
            entries = sorted(Path(abs_path).iterdir())
            listing = "\n".join(
                f"  {e.name}/" if e.is_dir() else f"  {e.name}"
                for e in entries
            )
            return {"observation": f"Directory: {path}\n{listing}", "exit_code": 0}
        except OSError as e:
            return {"observation": f"Error listing directory: {e}", "exit_code": 1}

    if editor is not None:
        try:
            view_range = None
            if start_line is not None and end_line is not None:
                view_range = [int(start_line), int(end_line)]
            elif start_line is not None:
                view_range = [int(start_line), -1]
            obs = editor(command="view", path=abs_path, view_range=view_range)
            return {"observation": _obs_text(obs), "exit_code": 0}
        except EditorToolError as e:
            return {"observation": f"Error: {e.message}", "exit_code": 1}

    # Fallback: built-in
    return _exec_read_file_builtin(path, working_dir)


def _exec_read_file_builtin(path: str, working_dir: str) -> dict[str, Any]:
    file_path = Path(working_dir) / path.lstrip("/")
    if not file_path.exists():
        return {"observation": f"Error: file not found: {path}", "exit_code": 1}
    content = file_path.read_text()
    lines = content.splitlines()
    numbered = "\n".join(f"{i+1:6}\t{l}" for i, l in enumerate(lines))
    if len(numbered) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        numbered = numbered[:half] + f"\n\n... ({len(numbered) - MAX_OUTPUT_CHARS} chars truncated) ...\n\n" + numbered[-half:]
    return {"observation": numbered, "exit_code": 0}


def _exec_insert(
    editor: "FileEditor | None",
    path: str,
    insert_line: int,
    new_str: str,
    working_dir: str,
) -> dict[str, Any]:
    if not path:
        return {"observation": "Error: path is required for insert", "exit_code": 1}

    abs_path = _to_abs_path(path, working_dir)

    if editor is not None:
        try:
            obs = editor(
                command="insert",
                path=abs_path,
                insert_line=int(insert_line),
                new_str=new_str,
            )
            text = _obs_text(obs)
            if path.endswith(".py"):
                new_content = Path(abs_path).read_text()
                try:
                    compile(new_content, path, "exec")
                except SyntaxError as e:
                    text += f"\nWARNING: SyntaxError after edit — {e.msg} (line {e.lineno}). Please fix."
            return {"observation": text, "exit_code": 0}
        except EditorToolError as e:
            return {"observation": f"Error: {e.message}", "exit_code": 1}

    # Fallback: built-in line insertion
    file_path = Path(working_dir) / path.lstrip("/")
    if not file_path.exists():
        return {"observation": f"Error: file not found: {path}", "exit_code": 1}
    lines = file_path.read_text().splitlines(keepends=True)
    insert_line = int(insert_line)
    if insert_line < 0 or insert_line > len(lines):
        return {"observation": f"Error: insert_line {insert_line} out of range [0, {len(lines)}]", "exit_code": 1}
    new_lines = new_str.split("\n")
    for i, nl in enumerate(new_lines):
        lines.insert(insert_line + i, nl + "\n")
    file_path.write_text("".join(lines))
    return {"observation": f"Inserted {len(new_lines)} line(s) after line {insert_line} in {path}", "exit_code": 0}


def _exec_undo(
    editor: "FileEditor | None",
    path: str,
    working_dir: str,
) -> dict[str, Any]:
    if not path:
        return {"observation": "Error: path is required for undo_edit", "exit_code": 1}

    abs_path = _to_abs_path(path, working_dir)

    if editor is not None:
        try:
            obs = editor(command="undo_edit", path=abs_path)
            return {"observation": _obs_text(obs), "exit_code": 0}
        except EditorToolError as e:
            return {"observation": f"Error: {e.message}", "exit_code": 1}

    return {"observation": "Error: undo_edit requires OpenHands SDK (not installed)", "exit_code": 1}
