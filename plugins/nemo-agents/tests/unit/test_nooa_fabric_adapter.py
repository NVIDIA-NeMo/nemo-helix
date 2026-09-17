# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generic platform NOOA adapter's half of the calling convention.

Everything analyst-specific lives in the Insights tests; what is asserted here
is what the adapter promises *every* NOOA agent: how the entrypoint is resolved,
what a callable is handed, how a failure is reported, and that Relay is activated
and then cleaned up.
"""

from __future__ import annotations

import json
import os
import sys
import types
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from nemo_agents_plugin.nooa import fabric_adapter
from nemo_fabric_adapter_contract import models as contract
from nemo_platform_plugin.agents.nooa_contract import NooaInvocation


@pytest.fixture
def agent_module(monkeypatch: pytest.MonkeyPatch):
    """An importable stand-in for a user's own installed project."""
    module = types.ModuleType("fake_user_agent")
    monkeypatch.setitem(sys.modules, "fake_user_agent", module)
    return module


def _agent_config(settings: dict[str, Any]) -> contract.AgentConfig:
    return contract.AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "models": {
                "default": {"provider": "platform", "model": "default/gpt-5"},
                "fast": {"provider": "platform", "model": "default/gpt-5-mini"},
            },
        }
    )


def _runtime_context(telemetry: contract.RuntimeTelemetryContext | None = None) -> contract.RuntimeContext:
    return contract.RuntimeContext(
        runtime_id="runtime-1",
        invocation_id="invocation-1",
        request_id="request-1",
        environment=contract.EnvironmentHandle(
            environment_id="environment-1",
            provider="local",
            control_location=contract.ControlLocation.IN_ENV_CONTROL,
            ownership=contract.EnvironmentOwnership.CALLER_OWNED,
        ),
        artifacts=contract.ArtifactManifest(),
        telemetry=telemetry,
    )


def _request(context: dict[str, Any] | None = None) -> contract.AgentRunRequest:
    return contract.AgentRunRequest(input="Do the thing.", context=context or {})


def _succeeded(response: str = "done") -> contract.AgentRunResult:
    return contract.AgentRunResult(status=contract.AgentRunStatus.SUCCEEDED, output={"response": response})


async def test_entrypoint_receives_the_whole_invocation(agent_module) -> None:
    """request, context, settings and models all reach the callable."""
    seen: dict[str, Any] = {}

    async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
        seen["invocation"] = invocation
        return _succeeded("analysed")

    agent_module.run = run

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run", "flavour": "vanilla"})})
    result = await runtime.invoke(_request({"job_workspace": "w"}), _runtime_context())

    assert result.status is contract.AgentRunStatus.SUCCEEDED
    assert result.output == {"response": "analysed"}

    invocation = seen["invocation"]
    assert invocation.request.input == "Do the thing."
    assert invocation.request.context == {"job_workspace": "w"}
    assert invocation.context.invocation_id == "invocation-1"
    # The settings bag arrives verbatim, `entrypoint` included: the adapter does
    # not get to decide which of an agent's own settings it may see.
    assert invocation.settings == {"entrypoint": "fake_user_agent:run", "flavour": "vanilla"}
    assert invocation.models["default"].model == "default/gpt-5"
    assert invocation.models["fast"].model == "default/gpt-5-mini"


async def test_a_missing_entrypoint_module_names_the_packaging_fix(agent_module) -> None:
    """The overwhelmingly common cause is packaging without --pyproject."""
    runtime = fabric_adapter.PlatformNooaRuntime()

    with pytest.raises(fabric_adapter.PlatformNooaAdapterConfigError) as excinfo:
        await runtime.start({"config": _agent_config({"entrypoint": "no_such_module:run"})})

    assert "--pyproject" in str(excinfo.value)
    assert "no_such_module" in str(excinfo.value)


async def test_a_typo_in_the_entrypoint_fails_at_start_not_at_first_invoke(agent_module) -> None:
    """Fabric should never report an agent healthy that cannot be invoked."""
    runtime = fabric_adapter.PlatformNooaRuntime()

    with pytest.raises(fabric_adapter.PlatformNooaAdapterConfigError) as excinfo:
        await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})

    assert "has no attribute 'run'" in str(excinfo.value)


async def test_a_sync_entrypoint_fails_at_start_not_at_first_invoke(agent_module) -> None:
    """The contract is one async callable; a `def` must fail the runtime
    immediately rather than surfacing as a TypeError on the first request."""

    def run(invocation: NooaInvocation) -> contract.AgentRunResult:  # not async
        return _succeeded()

    agent_module.run = run

    runtime = fabric_adapter.PlatformNooaRuntime()

    with pytest.raises(fabric_adapter.PlatformNooaAdapterConfigError) as excinfo:
        await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})

    assert "not an async callable" in str(excinfo.value)


@pytest.mark.parametrize("spec", ["fake_user_agent", "fake_user_agent:", ":run", "   "])
async def test_a_malformed_entrypoint_is_rejected(agent_module, spec: str) -> None:
    runtime = fabric_adapter.PlatformNooaRuntime()

    with pytest.raises(fabric_adapter.PlatformNooaAdapterConfigError) as excinfo:
        await runtime.start({"config": _agent_config({"entrypoint": spec})})

    assert "module.path:callable" in str(excinfo.value)


async def test_a_missing_entrypoint_setting_is_rejected() -> None:
    runtime = fabric_adapter.PlatformNooaRuntime()

    with pytest.raises(fabric_adapter.PlatformNooaAdapterConfigError) as excinfo:
        await runtime.start({"config": _agent_config({})})

    assert "harness.settings.entrypoint is required" in str(excinfo.value)


async def test_a_callable_returning_the_wrong_type_fails_the_run(agent_module) -> None:
    """A callable that forgets to return an AgentRunResult should say so."""

    async def run(invocation: NooaInvocation):
        return {"response": "a dict is not an AgentRunResult"}

    agent_module.run = run

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})
    result = await runtime.invoke(_request(), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert "returned dict, expected an AgentRunResult" in result.error.message


async def test_a_raising_callable_becomes_a_failed_result_with_its_traceback(agent_module, caplog) -> None:
    """``str(error)`` alone loses the traceback, and the adapter's stderr is
    captured to a run artifact the job dumps on failure - so logging here is
    what puts the traceback somewhere anyone will actually read."""

    async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
        raise RuntimeError("gateway returned 502")

    agent_module.run = run

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})

    with caplog.at_level("ERROR", logger="nemo_agents_plugin.nooa.fabric_adapter"):
        result = await runtime.invoke(_request(), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "platform_nooa_agent_failed"
    assert result.error.retryable is False
    assert "gateway returned 502" in result.error.message
    assert "Platform NOOA agent run failed." in caplog.text
    assert "Traceback (most recent call last)" in caplog.text


async def test_the_whole_cause_chain_is_logged(agent_module, caplog) -> None:
    """The originating error is what distinguishes one failure from another:
    an LLM error raised after retries reads identically whatever provoked it."""
    originating = "Function 'abc-123': Not found for account"
    raised = "LLM API error after 3 retries"

    async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
        try:
            raise ValueError(originating)
        except ValueError as cause:
            raise RuntimeError(raised) from cause

    agent_module.run = run

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})

    with caplog.at_level("ERROR", logger="nemo_agents_plugin.nooa.fabric_adapter"):
        await runtime.invoke(_request(), _runtime_context())

    assert originating in caplog.text, "root cause was dropped"
    assert raised in caplog.text


async def test_relay_activates_fabrics_config_and_restores_the_environment(
    agent_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fabric resolves the whole export; the adapter only activates it.

    Nothing about the destination is decided here — the endpoint, credentials
    and agent name all arrive in the config file Fabric wrote.
    """
    relay_config = tmp_path / "relay-config.json"
    relay_config.write_text(
        json.dumps(
            {
                "relay": {
                    "config": {
                        "version": 1,
                        "components": [{"kind": "observability", "enabled": True, "config": {"version": 3}}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
        # Relay resolves header_env against the environment while exporting,
        # so the variables have to be set for the duration of the run.
        seen["env_during_run"] = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
        telemetry = invocation.context.telemetry
        assert telemetry is not None
        seen["relay_enabled_seen_by_callable"] = telemetry.relay_enabled
        return _succeeded()

    @asynccontextmanager
    async def fake_plugin(config: Any) -> AsyncIterator[None]:
        seen["plugin_config"] = config
        yield

    agent_module.run = run
    monkeypatch.setattr(fabric_adapter.relay_plugin, "plugin", fake_plugin)

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})
    telemetry = contract.RuntimeTelemetryContext(
        relay_enabled=True,
        config_path=str(relay_config),
        env={"FABRIC_RELAY_CONFIG_PATH": str(relay_config)},
    )

    result = await runtime.invoke(_request(), _runtime_context(telemetry))

    assert result.status is contract.AgentRunStatus.SUCCEEDED
    assert seen["plugin_config"]["components"][0]["kind"] == "observability"
    assert seen["env_during_run"] == str(relay_config)
    assert seen["relay_enabled_seen_by_callable"] is True
    # The runtime serves many invocations; a leftover config path would make
    # the next one export against a stale, possibly deleted, config.
    assert "FABRIC_RELAY_CONFIG_PATH" not in os.environ


async def test_without_relay_the_plugin_is_never_activated(agent_module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relay is opt-in per invocation; Fabric says when."""
    seen: dict[str, Any] = {}

    async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
        seen["telemetry"] = invocation.context.telemetry
        return _succeeded()

    @asynccontextmanager
    async def fail_if_called(config: Any) -> AsyncIterator[None]:
        raise AssertionError("Relay must not be activated when Fabric did not ask for it")
        yield

    agent_module.run = run
    monkeypatch.setattr(fabric_adapter.relay_plugin, "plugin", fail_if_called)

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})

    result = await runtime.invoke(_request(), _runtime_context(None))

    assert result.status is contract.AgentRunStatus.SUCCEEDED
    assert seen["telemetry"] is None


async def test_relay_enabled_without_a_config_path_is_an_error(agent_module) -> None:
    async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
        raise AssertionError("should not be reached")

    agent_module.run = run

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})
    telemetry = contract.RuntimeTelemetryContext(relay_enabled=True, config_path=None, env={})

    result = await runtime.invoke(_request(), _runtime_context(telemetry))

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert "Relay is enabled but Fabric supplied no config path" in result.error.message


async def test_stop_clears_the_runtime_for_reuse(agent_module) -> None:
    async def run(invocation: NooaInvocation) -> contract.AgentRunResult:
        return _succeeded()

    agent_module.run = run

    runtime = fabric_adapter.PlatformNooaRuntime()
    await runtime.start({"config": _agent_config({"entrypoint": "fake_user_agent:run"})})
    await runtime.stop()

    result = await runtime.invoke(_request(), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert "harness.settings.entrypoint is required" in result.error.message
