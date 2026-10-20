# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The check script launches pytest with the resolved context and does not call the platform."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
from check import parse_args, run_check


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        context_name="local",
        cluster=SimpleNamespace(base_url="http://localhost:8080"),
    )


def test_check_runs_pytest_with_the_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_ACCESS_TOKEN", "kept-token")
    captured: dict[str, Any] = {}
    seen: dict[str, object] = {}

    def fake_get_context(overrides: object = None) -> SimpleNamespace:
        seen["overrides"] = overrides
        return _context()

    def fake_run(args: list[str], *, env: dict[str, str], check: bool) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["env"] = env
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("check.get_context", fake_get_context)
    monkeypatch.setattr("check.subprocess.run", fake_run)
    code = run_check(parse_args(["--context", "prod", "--models", "--model", "default/m", "--keep", "--timeout", "30"]))
    assert code == 0
    assert seen["overrides"] == {"current_context": "prod"}
    args_value = captured["args"]
    env_value = captured["env"]
    assert isinstance(args_value, list)
    assert isinstance(env_value, dict)
    args = [str(arg) for arg in args_value]
    env = {str(key): str(value) for key, value in env_value.items()}
    assert args[1:3] == ["-m", "pytest"]
    assert any(arg.endswith("instance_check.py") for arg in args)
    assert env["NHX_BASE_URL"] == "http://localhost:8080"
    assert env["NHX_CURRENT_CONTEXT"] == "local"
    assert env["NHX_ACCESS_TOKEN"] == "kept-token"
    assert env["NHX_CHECK_MODELS"] == "1"
    assert env["NHX_CHECK_MODEL"] == "default/m"
    assert env["NHX_CHECK_KEEP"] == "1"
    assert env["NHX_CHECK_TIMEOUT"] in {"30", "30.0"}


def test_check_propagates_the_pytest_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], *, env: dict[str, str], check: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 3)

    monkeypatch.setattr("check.get_context", lambda overrides=None: _context())
    monkeypatch.setattr("check.subprocess.run", fake_run)
    assert run_check(parse_args([])) == 3


def test_check_requires_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    called = SimpleNamespace(ran=False)

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        called.ran = True
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr("check.subprocess.run", fake_run)
    monkeypatch.setattr("check.importlib.util.find_spec", lambda name: None)
    assert run_check(parse_args([])) == 1
    assert called.ran is False
