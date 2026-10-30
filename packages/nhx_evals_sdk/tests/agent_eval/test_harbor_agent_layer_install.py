# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The install step a Kubernetes agent layer's build runs: the agent's own setup, into the image being built."""

import json
import os
import pwd
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor", reason="the install drives a Harbor agent's setup")

from harbor.agents.factory import AgentFactory
from nhx_evals_sdk.agent_eval.runtimes.harbor import agent_layer_install
from nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer_install import LocalExecEnvironment, main

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="agent layers are built on Linux")


async def test_commands_run_here_with_the_given_environment(tmp_path: Path) -> None:
    environment = LocalExecEnvironment(default_user=None)

    result = await environment.exec('echo "$GREETING" > out.txt; echo done', cwd=str(tmp_path), env={"GREETING": "hi"})

    assert result.return_code == 0 and (result.stdout or "").strip() == "done"
    assert (tmp_path / "out.txt").read_text() == "hi\n"


async def test_a_failing_command_reports_its_code_and_output() -> None:
    result = await LocalExecEnvironment(default_user=None).exec("echo oops >&2; exit 3")

    assert result.return_code == 3 and (result.stderr or "").strip() == "oops"


async def test_the_default_user_runs_commands_with_its_home() -> None:
    """In a build the process is root and switches to the agent user; here it is already that user."""
    me = pwd.getpwuid(os.getuid())
    environment = LocalExecEnvironment(default_user=me.pw_name)

    result = await environment.exec('echo "$HOME $USER"')

    assert (result.stdout or "").split() == [me.pw_dir, me.pw_name]


async def test_uploads_copy_into_the_filesystem(tmp_path: Path) -> None:
    source = tmp_path / "bundle"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "agent.yaml").write_text("name: a\n")
    environment = LocalExecEnvironment(default_user=None)

    await environment.upload_dir(source, str(tmp_path / "target"))
    await environment.upload_file(source / "nested" / "agent.yaml", str(tmp_path / "file" / "agent.yaml"))

    assert (tmp_path / "target" / "nested" / "agent.yaml").read_text() == "name: a\n"
    assert (tmp_path / "file" / "agent.yaml").read_text() == "name: a\n"


def test_main_runs_the_named_agents_setup_as_its_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[Any] = []
    set_up: list[Any] = []

    class _Agent:
        async def setup(self, environment: Any) -> None:
            set_up.append(environment)

    def create(config: Any, *, logs_dir: Path) -> _Agent:
        built.append(config)
        return _Agent()

    monkeypatch.setattr(agent_layer_install.AgentFactory, "create_agent_from_config", create)
    spec = tmp_path / "agent.json"
    spec.write_text(
        json.dumps({"name": "codex", "import_path": None, "kwargs": {"version": "0.162.1"}, "user": "agent"})
    )

    assert main([str(spec)]) == 0

    (config,) = built
    assert config.name == "codex" and config.kwargs == {"version": "0.162.1"}
    (environment,) = set_up
    assert isinstance(environment, LocalExecEnvironment) and environment.default_user == "agent"


def test_main_needs_one_argument() -> None:
    assert main([]) == 2


def test_the_agent_is_built_by_harbors_factory() -> None:
    """The install builds the agent the way a trial does, so its setup is the one a trial would run."""
    assert agent_layer_install.AgentFactory is AgentFactory
