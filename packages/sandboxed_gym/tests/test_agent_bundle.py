# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess

import pytest
from sandboxed_gym.runtime.agent_bundle import activate_agent_executable


def _runs(returncode: int = 0, calls: list[list[str]] | None = None):
    def run(argv: list[str], **_: object) -> subprocess.CompletedProcess:
        if calls is not None:
            calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode)

    return run


def test_a_working_agent_goes_first_on_path():
    environ = {"PATH": "/usr/bin:/bin"}
    calls: list[list[str]] = []

    activate_agent_executable("/installed-agent/bin/codex", environ, run=_runs(calls=calls))

    assert calls == [["/installed-agent/bin/codex", "--version"]]
    assert environ["PATH"] == "/installed-agent/bin:/usr/bin:/bin"


def test_no_agent_changes_nothing():
    environ = {"PATH": "/usr/bin:/bin"}

    activate_agent_executable(None, environ, run=_runs(calls=(calls := [])))

    assert calls == []
    assert environ == {"PATH": "/usr/bin:/bin"}


def test_a_failing_agent_fails_the_host_and_keeps_path():
    environ = {"PATH": "/usr/bin:/bin"}

    with pytest.raises(RuntimeError, match="--version exited 3"):
        activate_agent_executable("/installed-agent/bin/codex", environ, run=_runs(returncode=3))

    assert environ["PATH"] == "/usr/bin:/bin"


@pytest.mark.parametrize(
    "error", [FileNotFoundError("no such file"), subprocess.TimeoutExpired(["codex", "--version"], 120)]
)
def test_an_agent_that_cannot_run_fails_the_host(error: Exception):
    def run(argv: list[str], **_: object) -> subprocess.CompletedProcess:
        raise error

    with pytest.raises(RuntimeError, match="could not run"):
        activate_agent_executable("/installed-agent/bin/codex", {"PATH": "/bin"}, run=run)
