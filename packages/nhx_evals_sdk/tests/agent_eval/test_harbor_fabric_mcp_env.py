# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stdio MCP server gets the secret its ``${NAME}`` env names, filled in only inside the task container."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("harbor", reason="NemoFabricAgent subclasses Harbor's BaseAgent")

from nemo_fabric import FabricConfig, FabricConfigError
from nemo_fabric.integrations.harbor.fabric_agent import FabricAgent, fabric_runner_command
from nhx_evals_sdk.agent_eval.runtimes.harbor import mcp_env_runner
from nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_agent import NemoFabricAgent
from nhx_evals_sdk.agent_eval.runtimes.harbor.mcp_env_runner import fill_mcp_env, run_through

_SHIM = "/tmp/nemo-fabric-mcp-env-runner.py"
_RUNNER_PYTHON = "/opt/nemo-fabric/bin/python"


def _config(env: dict[str, str] | None = None) -> dict[str, Any]:
    servers: dict[str, Any] = {
        "web": {"transport": "streamable-http", "url": "http://search.internal/mcp"},
    }
    if env is not None:
        servers["calculator"] = {"transport": "stdio", "url": "/usr/bin/python3", "args": ["srv.py"], "env": env}
    return {
        "metadata": {"name": "codex-agent"},
        "harness": {"adapter_id": "nvidia.fabric.codex"},
        "mcp": {"servers": servers},
    }


_RUNNER_COMMAND = fabric_runner_command(
    fabric_python=_RUNNER_PYTHON, spec_path="/tmp/fabric-run-1.json", result_path="/tmp/fabric-result-1.json"
)


class _Upstream(FabricAgent):  # ty: ignore[unsupported-base]
    """Stands in for upstream's setup and run, so no shared class is patched: run executes the runner command."""

    async def setup(self, environment: Any) -> None:
        del environment

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        del instruction, context
        await environment.exec("mkdir -p /app")
        await environment.exec(_RUNNER_COMMAND, env={"CALC_TOKEN": "s3cret"})


class _Agent(NemoFabricAgent, _Upstream):
    """``NemoFabricAgent`` over the stand-in: its ``super()`` calls land on ``_Upstream``."""


def _agent(logs_dir: Path, env: dict[str, str] | None) -> NemoFabricAgent:
    logs_dir.mkdir(parents=True, exist_ok=True)  # Harbor creates the trial's agent logs before setup
    return _Agent(logs_dir=logs_dir, fabric_config=_config(env), fabric_workspace="/app")


class _Result:
    return_code = 0
    stdout = ""
    stderr = ""


class _Environment:
    """Records what the agent asks of the container."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploaded: dict[str, str] = {}

    async def exec(self, command: str, **kwargs: Any) -> _Result:
        self.commands.append(command)
        return _Result()

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        self.uploaded[target_path] = Path(source_path).read_text(encoding="utf-8")


def test_the_shim_fills_in_a_stdio_servers_templates_from_its_environment() -> None:
    config = FabricConfig.model_validate(_config({"CALC_TOKEN": "${CALC_TOKEN}", "MODE": "mock"}))

    fill_mcp_env(config, {"CALC_TOKEN": "s3cret"})

    assert config.mcp is not None
    assert config.mcp.servers["calculator"].env == {"CALC_TOKEN": "s3cret", "MODE": "mock"}


def test_the_shim_refuses_a_template_its_environment_does_not_set() -> None:
    config = FabricConfig.model_validate(_config({"CALC_TOKEN": "${CALC_TOKEN}"}))

    with pytest.raises(FabricConfigError, match="'calculator' reads CALC_TOKEN") as refused:
        fill_mcp_env(config, {})

    assert refused.value.code == "harbor_mcp_env_unset"


def test_the_shim_fills_the_payload_in_before_the_runner_runs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: Any = SimpleNamespace(config=FabricConfig.model_validate(_config({"CALC_TOKEN": "${CALC_TOKEN}"})))
    ran: list[dict[str, str]] = []

    async def runner_run(given: Any) -> str:
        ran.append(dict(given.config.mcp.servers["calculator"].env))
        return "result"

    runner: Any = SimpleNamespace(run=runner_run)
    runner.main = lambda: asyncio.run(runner.run(payload))  # as the runner's `main` looks `run` up
    monkeypatch.setenv("CALC_TOKEN", "s3cret")

    run_through(runner)

    assert ran == [{"CALC_TOKEN": "s3cret"}]


def test_started_as_the_runner_is_the_shim_runs_fabrics_runner(tmp_path: Path) -> None:
    """With the runner's own arguments it runs nemo-fabric's runner, which reports the unset variable."""
    agent = _agent(tmp_path, {"CALC_TOKEN": "${CALC_TOKEN}"})
    spec = tmp_path / "fabric-run.json"
    spec.write_text(agent._build_spec("solve it").model_dump_json(), encoding="utf-8")
    result = tmp_path / "fabric-result.json"

    finished = subprocess.run(
        [sys.executable, mcp_env_runner.__file__, "--spec", str(spec), "--result", str(result)],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert finished.returncode == 1, finished.stderr
    error = json.loads(result.read_text(encoding="utf-8"))["runner_error"]
    assert error["code"] == "harbor_mcp_env_unset" and "CALC_TOKEN" in error["message"]


async def test_the_agent_uploads_the_shim_only_when_a_stdio_server_names_a_variable(tmp_path: Path) -> None:
    templated, plain = _Environment(), _Environment()

    await _agent(tmp_path / "a", {"CALC_TOKEN": "${CALC_TOKEN}"}).prepare_mcp_env(templated)
    await _agent(tmp_path / "b", {"MODE": "mock"}).prepare_mcp_env(plain)

    assert templated.uploaded == {_SHIM: Path(mcp_env_runner.__file__).read_text(encoding="utf-8")}
    assert plain.uploaded == {}


async def test_setup_uploads_the_shim(tmp_path: Path) -> None:
    environment = _Environment()

    await _agent(tmp_path, {"CALC_TOKEN": "${CALC_TOKEN}"}).setup(environment)

    assert list(environment.uploaded) == [_SHIM]


@pytest.mark.parametrize(("env", "through_shim"), [({"CALC_TOKEN": "${CALC_TOKEN}"}, True), ({"MODE": "mock"}, False)])
async def test_only_the_runner_command_goes_through_the_shim(
    tmp_path: Path, env: dict[str, str], through_shim: bool
) -> None:
    environment = _Environment()

    await _agent(tmp_path, env)._run_harness("solve it", environment, None)

    runner = _RUNNER_COMMAND.replace("-m nemo_fabric.integrations.harbor.runner", _SHIM)
    assert environment.commands == ["mkdir -p /app", runner if through_shim else _RUNNER_COMMAND]
    assert runner.startswith(f"{_RUNNER_PYTHON} {_SHIM} --spec")


def test_the_host_never_fills_in_the_value(tmp_path: Path) -> None:
    """Harbor writes the payload to the trial's logs, so it carries the template, not the secret."""
    agent = _agent(tmp_path, {"CALC_TOKEN": "${CALC_TOKEN}"})
    agent._extra_env = {"CALC_TOKEN": "s3cret"}

    config = agent._build_config()

    assert config.mcp is not None and config.mcp.servers["calculator"].env == {"CALC_TOKEN": "${CALC_TOKEN}"}


def test_upstreams_runner_command_still_runs_the_runner_module_the_shim_replaces() -> None:
    """If nemo-fabric builds the command another way, the shim can't take its place; this fails on that upgrade."""
    command = fabric_runner_command(fabric_python="python", spec_path="spec", result_path="result")
    assert "-m nemo_fabric.integrations.harbor.runner --spec spec --result result" in command
