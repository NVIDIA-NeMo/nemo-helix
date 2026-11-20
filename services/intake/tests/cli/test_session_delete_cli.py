# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deletion confirmations and real typed-client wire behavior."""

import httpx
import pytest


def test_delete_session_scopes_request(intake_cli):
    result, recorder = intake_cli.run(
        ["sessions", "delete", "session-a", "--workspace", "team-a", "--yes"], [httpx.Response(204)]
    )
    assert result.exit_code == 0, result.output
    assert len(recorder.requests) == 1
    assert recorder.last.method == "DELETE"
    assert recorder.last.url.path == "/apis/intake/v2/workspaces/team-a/sessions/session-a"
    assert recorder.last.content == b""
    assert "team-a" in result.output and "session-a" in result.output


def test_delete_session_uses_configured_workspace(intake_cli, monkeypatch):
    monkeypatch.setenv("NHX_WORKSPACE", "configured")
    result, recorder = intake_cli.run(["sessions", "delete", "session-a", "--yes"], [httpx.Response(204)])
    assert result.exit_code == 0, result.output
    assert "/workspaces/configured/" in recorder.last.url.path


def test_delete_session_requires_noninteractive_confirmation(intake_cli):
    result, recorder = intake_cli.run(["sessions", "delete", "session-a"])
    assert result.exit_code == 2, result.output
    assert "--yes" in result.output
    assert "ALL current traces and spans" in result.output
    assert not recorder.requests


@pytest.mark.parametrize("answer,exit_code,request_count", [("n\n", 1, 0), ("y\n", 0, 1)])
def test_delete_session_interactive_confirmation(intake_cli, monkeypatch, answer, exit_code, request_count):
    monkeypatch.setattr("nemo_helix_plugin.cli_confirm.is_stdin_available", lambda: False)
    result, recorder = intake_cli.run(["sessions", "delete", "session-a"], [httpx.Response(204)], input=answer)
    assert result.exit_code == exit_code, result.output
    assert len(recorder.requests) == request_count


@pytest.mark.parametrize(
    "status,message",
    [
        (403, "intake.sessions.delete"),
        (404, "already removed"),
        (409, "conflict"),
        (503, "even if sessions get returns 404"),
    ],
)
def test_delete_session_remote_errors(intake_cli, status, message):
    result, recorder = intake_cli.run(
        ["sessions", "delete", "session-a", "--yes"],
        [httpx.Response(status, json={"detail": "Backend explanation"})],
    )
    assert result.exit_code == 3, result.output
    assert message in result.output
    assert "Backend explanation" in result.output
    assert len(recorder.requests) == 1


def test_delete_session_code_sends_no_request(intake_cli):
    result, recorder = intake_cli.run(
        ["sessions", "delete", "session-a", "--workspace", "team-a", "--output-format", "code"]
    )
    assert result.exit_code == 0, result.output
    assert "delete_session(" in result.output
    assert "team-a" in result.output
    assert not recorder.requests
    compile(result.output, "<snippet>", "exec")
