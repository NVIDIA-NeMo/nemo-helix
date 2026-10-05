# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the trace_reader module."""

import json
import sys
import tempfile
from pathlib import Path

# Add the shared directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from trace_reader import Session, ToolCall, ToolResult, get_session, parse_session


def test_parse_session_extracts_bash_commands() -> None:
    """Test that parse_session correctly extracts Bash tool calls."""
    # Create a sample session JSONL
    session_data = [
        {"type": "user", "sessionId": "test-session-123", "message": {"role": "user", "content": "test"}},
        {
            "type": "assistant",
            "sessionId": "test-session-123",
            "timestamp": "2026-02-07T12:00:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "nemo workspaces create test"},
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "sessionId": "test-session-123",
            "timestamp": "2026-02-07T12:01:00Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-2",
                        "name": "Bash",
                        "input": {"command": "nemo workspaces list"},
                    }
                ],
            },
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        session_file = Path(tmpdir) / "test-session.jsonl"
        with session_file.open("w") as f:
            for entry in session_data:
                f.write(json.dumps(entry) + "\n")

        session = parse_session(session_file)

        assert session.session_id == "test-session-123"
        assert len(session.tool_calls) == 2

        commands = session.get_bash_commands()
        assert len(commands) == 2
        assert "nemo workspaces create test" in commands
        assert "nemo workspaces list" in commands


def test_parse_session_handles_mcp_tools() -> None:
    """Test that parse_session correctly identifies MCP tool calls."""
    session_data = [
        {"type": "user", "sessionId": "test-mcp", "message": {"role": "user", "content": "test"}},
        {
            "type": "assistant",
            "sessionId": "test-mcp",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "mcp__nhx__create_workspace",
                        "input": {"name": "test-workspace"},
                    }
                ],
            },
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        session_file = Path(tmpdir) / "test-mcp.jsonl"
        with session_file.open("w") as f:
            for entry in session_data:
                f.write(json.dumps(entry) + "\n")

        session = parse_session(session_file)

        mcp_calls = session.get_mcp_calls()
        assert len(mcp_calls) == 1
        assert mcp_calls[0].name == "mcp__nhx__create_workspace"
        assert mcp_calls[0].input["name"] == "test-workspace"


def test_get_session_finds_most_recent() -> None:
    """Test that get_session returns the most recent session file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir)
        project_subdir = projects_dir / "-app"
        project_subdir.mkdir()

        # Create two session files with different timestamps
        old_session = project_subdir / "old-session.jsonl"
        new_session = project_subdir / "new-session.jsonl"

        old_session.write_text('{"type": "user", "sessionId": "old", "message": {}}\n')

        # Small delay to ensure different mtime
        import time

        time.sleep(0.01)

        new_session.write_text('{"type": "user", "sessionId": "new", "message": {}}\n')

        session = get_session(projects_dir)
        assert session.session_id == "new"


def test_tool_call_command_property() -> None:
    """Test the ToolCall.command property for Bash calls."""
    bash_call = ToolCall(name="Bash", tool_id="1", input={"command": "ls -la"})
    assert bash_call.command == "ls -la"

    other_call = ToolCall(name="Read", tool_id="2", input={"file_path": "/tmp/test"})
    assert other_call.command is None


def test_session_get_tool_calls_by_name() -> None:
    """Test filtering tool calls by name."""
    session = Session(
        session_id="test",
        tool_calls=[
            ToolCall(name="Bash", tool_id="1", input={"command": "cmd1"}),
            ToolCall(name="Read", tool_id="2", input={"file_path": "/tmp"}),
            ToolCall(name="Bash", tool_id="3", input={"command": "cmd2"}),
        ],
    )

    bash_calls = session.get_tool_calls_by_name("Bash")
    assert len(bash_calls) == 2

    read_calls = session.get_tool_calls_by_name("Read")
    assert len(read_calls) == 1


def test_parse_session_extracts_tool_results() -> None:
    """Test that parse_session correctly extracts tool results from user messages."""
    session_data = [
        {"type": "user", "sessionId": "test-results", "message": {"role": "user", "content": "test"}},
        {
            "type": "assistant",
            "sessionId": "test-results",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "echo hello"}},
                ],
            },
        },
        {
            "type": "user",
            "sessionId": "test-results",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "hello\n"},
                ],
            },
        },
        {
            "type": "assistant",
            "sessionId": "test-results",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tool-2", "name": "Bash", "input": {"command": "false"}},
                ],
            },
        },
        {
            "type": "user",
            "sessionId": "test-results",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-2", "content": "exit code 1", "is_error": True},
                ],
            },
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        session_file = Path(tmpdir) / "test-results.jsonl"
        with session_file.open("w") as f:
            for entry in session_data:
                f.write(json.dumps(entry) + "\n")

        session = parse_session(session_file)

        assert len(session.tool_results) == 2
        assert session.tool_results[0].tool_use_id == "tool-1"
        assert session.tool_results[0].content == "hello\n"
        assert session.tool_results[0].is_error is False
        assert session.tool_results[1].tool_use_id == "tool-2"
        assert session.tool_results[1].is_error is True


def test_get_tool_results_filtered_by_name() -> None:
    """Test filtering tool results by originating tool name."""
    session = Session(
        session_id="test",
        tool_calls=[
            ToolCall(name="Bash", tool_id="t1", input={"command": "echo hi"}),
            ToolCall(name="Read", tool_id="t2", input={"file_path": "/tmp/x"}),
            ToolCall(name="Bash", tool_id="t3", input={"command": "nemo chat ..."}),
        ],
        tool_results=[
            ToolResult(tool_use_id="t1", content="hi\n"),
            ToolResult(tool_use_id="t2", content="file contents"),
            ToolResult(tool_use_id="t3", content='{"choices": [{"message": {"content": "4"}}]}'),
        ],
    )

    all_results = session.get_tool_results()
    assert len(all_results) == 3

    bash_results = session.get_tool_results("Bash")
    assert len(bash_results) == 2
    assert bash_results[0].content == "hi\n"
    assert "choices" in bash_results[1].content

    read_results = session.get_tool_results("Read")
    assert len(read_results) == 1
    assert read_results[0].content == "file contents"
