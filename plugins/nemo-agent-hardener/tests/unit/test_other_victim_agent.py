# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `other-victim` example's tool executors.

`_python_executor` used to run model-supplied code with `exec()` in-process and no deadline, so a
permitted `while True: pass` could hang the server worker forever. It now shells out with a
timeout, matching `_bash_executor`'s existing pattern — these tests cover that timeout and the
output cap shared by both executors.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_AGENT_PATH = Path(__file__).resolve().parents[2] / "examples" / "other-victim" / "agent.py"


def _load_agent_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("other_victim_agent", _AGENT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agent = _load_agent_module()


def test_python_executor_captures_stdout():
    assert agent._python_executor("print('hi')") == "hi\n"


def test_python_executor_returns_a_timeout_error_instead_of_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    """A permitted `while True: pass` must not hang the worker; it should time out cleanly."""

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout") or 30)

    monkeypatch.setattr(agent.subprocess, "run", fake_run)

    assert agent._python_executor("while True: pass") == "error: execution timed out after 30s"


def test_bash_executor_reports_a_timeout_rather_than_looking_like_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An escaping TimeoutExpired becomes "tool call refused", which a war-game scores as a block."""

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout") or 30)

    monkeypatch.setattr(agent.subprocess, "run", fake_run)

    assert agent._bash_executor("sleep 600") == "error: execution timed out after 30s"


def test_output_under_the_cap_is_returned_verbatim():
    assert agent._cap_output("short") == "short"


def test_output_over_the_cap_is_truncated_with_a_marker():
    output = "x" * (agent._MAX_TOOL_OUTPUT_CHARS + 500)

    capped = agent._cap_output(output)

    assert capped.startswith("x" * agent._MAX_TOOL_OUTPUT_CHARS)
    assert f"truncated, {len(output)} chars total" in capped


def test_bash_executor_output_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    huge = "y" * (agent._MAX_TOOL_OUTPUT_CHARS + 100)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=huge, stderr="")

    monkeypatch.setattr(agent.subprocess, "run", fake_run)

    result = agent._bash_executor("echo huge")

    assert "truncated" in result
    assert len(result) < len(huge)
