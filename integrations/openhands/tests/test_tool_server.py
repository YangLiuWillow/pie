"""Tests for the Pattern A tool server."""

from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import pytest

from tool_server import start_tool_server


@pytest.fixture()
def server_and_dir():
    with tempfile.TemporaryDirectory(prefix="toolsrv-test-") as tmpdir:
        server, port = start_tool_server(tmpdir)
        yield server, port, tmpdir
        server.shutdown()


def _post(port: int, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/execute",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


class TestBash:
    def test_echo(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "echo hello"})
        assert result["exit_code"] == 0
        assert "hello" in result["observation"]

    def test_cwd(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {"action": "bash", "command": "pwd"})
        assert result["exit_code"] == 0
        assert tmpdir in result["observation"]

    def test_nonzero_exit(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "exit 42"})
        assert result["exit_code"] == 42

    def test_empty_command(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": ""})
        assert result["exit_code"] == 1

    def test_stderr_captured(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "bash", "command": "echo err >&2"})
        assert "err" in result["observation"]


class TestEdit:
    def test_create_file(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "hello.txt",
            "old_str": "",
            "new_str": "hello world",
        })
        assert result["exit_code"] == 0
        assert (Path(tmpdir) / "hello.txt").read_text() == "hello world"

    def test_str_replace(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "test.py").write_text("x = 1\ny = 2\n")
        result = _post(port, {
            "action": "edit",
            "path": "test.py",
            "old_str": "x = 1",
            "new_str": "x = 42",
        })
        assert result["exit_code"] == 0
        assert "x = 42" in (Path(tmpdir) / "test.py").read_text()

    def test_ambiguous_match(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "dup.txt").write_text("a\na\n")
        result = _post(port, {
            "action": "edit",
            "path": "dup.txt",
            "old_str": "a",
            "new_str": "b",
        })
        assert result["exit_code"] == 1

    def test_no_match(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "empty.txt").write_text("")
        result = _post(port, {
            "action": "edit",
            "path": "empty.txt",
            "old_str": "not here",
            "new_str": "anything",
        })
        assert result["exit_code"] == 1
        # Homegrown editor says "not found"; the wrapped OpenHands FileEditor
        # says "No replacement was performed ... did not appear verbatim".
        obs = result["observation"].lower()
        assert "not found" in obs or "no replacement" in obs

    def test_missing_file(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "nonexistent.txt",
            "old_str": "x",
            "new_str": "y",
        })
        assert result["exit_code"] == 1

    def test_create_nested_dirs(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        result = _post(port, {
            "action": "edit",
            "path": "a/b/c.txt",
            "old_str": "",
            "new_str": "nested",
        })
        assert result["exit_code"] == 0
        assert (Path(tmpdir) / "a" / "b" / "c.txt").read_text() == "nested"


    def test_no_match_returns_error(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "test.py").write_text("def foo(x):\n    return x + 1\n")
        result = _post(port, {
            "action": "edit",
            "path": "test.py",
            "old_str": "def foo(y):",
            "new_str": "def foo(y, z):",
        })
        assert result["exit_code"] == 1
        obs = result["observation"].lower()
        assert "not found" in obs or "no replacement" in obs

    def test_old_str_too_large(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        big_old = "\n".join(f"line {i}" for i in range(60))
        (Path(tmpdir) / "big.txt").write_text(big_old + "\n")
        result = _post(port, {
            "action": "edit",
            "path": "big.txt",
            "old_str": big_old,
            "new_str": "replaced",
        })
        assert result["exit_code"] == 1
        assert "max" in result["observation"]

    def test_new_str_too_large(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "small.txt").write_text("x = 1\n")
        big_new = "\n".join(f"line {i}" for i in range(110))
        result = _post(port, {
            "action": "edit",
            "path": "small.txt",
            "old_str": "x = 1",
            "new_str": big_new,
        })
        assert result["exit_code"] == 1
        assert "max" in result["observation"]

    def test_ambiguous_match_returns_error(self, server_and_dir):
        """FileEditor and fallback both reject ambiguous matches."""
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "dup2.txt").write_text("foo\nbar\nfoo\n")
        result = _post(port, {
            "action": "edit",
            "path": "dup2.txt",
            "old_str": "foo",
            "new_str": "baz",
        })
        assert result["exit_code"] == 1


class TestReadFile:
    def test_read_existing(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "test.py").write_text("import os\nprint('hi')\n")
        result = _post(port, {"action": "read_file", "path": "test.py"})
        assert result["exit_code"] == 0
        assert "import os" in result["observation"]
        assert "1" in result["observation"]

    def test_read_missing(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "read_file", "path": "nope.txt"})
        assert result["exit_code"] == 1
        obs = result["observation"].lower()
        assert "not found" in obs or "error" in obs

    def test_read_no_path(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "read_file", "path": ""})
        assert result["exit_code"] == 1

    def test_read_with_line_range(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        lines = "\n".join(f"line {i}" for i in range(1, 21))
        (Path(tmpdir) / "big.py").write_text(lines + "\n")
        result = _post(port, {
            "action": "read_file",
            "path": "big.py",
            "start_line": 5,
            "end_line": 10,
        })
        assert result["exit_code"] == 0
        assert "line 5" in result["observation"]
        assert "line 10" in result["observation"]


class TestInsert:
    def test_insert_at_line(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        (Path(tmpdir) / "test.py").write_text("a\nb\nc\n")
        result = _post(port, {
            "action": "insert",
            "path": "test.py",
            "insert_line": 2,
            "new_str": "INSERTED",
        })
        assert result["exit_code"] == 0
        content = (Path(tmpdir) / "test.py").read_text()
        assert "INSERTED" in content

    def test_insert_no_path(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {
            "action": "insert",
            "path": "",
            "insert_line": 1,
            "new_str": "x",
        })
        assert result["exit_code"] == 1


class TestUndo:
    def test_undo_no_path(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "undo_edit", "path": ""})
        assert result["exit_code"] == 1


class TestMisc:
    def test_unknown_action(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "unknown"})
        assert result["exit_code"] == 1

    def test_finish_action(self, server_and_dir):
        _, port, _ = server_and_dir
        result = _post(port, {"action": "finish"})
        assert result["exit_code"] == 0

    def test_has_diff_no_changes(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        _post(port, {"action": "bash", "command": "git init && git add -A && git commit -m init --allow-empty"})
        result = _post(port, {"action": "has_diff"})
        assert result["exit_code"] == 0
        assert result["has_diff"] is False

    def test_has_diff_with_changes(self, server_and_dir):
        _, port, tmpdir = server_and_dir
        _post(port, {"action": "bash", "command": "git init && git add -A && git commit -m init --allow-empty"})
        _post(port, {"action": "bash", "command": "echo changed > somefile.txt"})
        result = _post(port, {"action": "has_diff"})
        assert result["exit_code"] == 0
        assert result["has_diff"] is True
