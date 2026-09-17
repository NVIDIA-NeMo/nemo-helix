# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Insights analyst as an entrypoint for the platform NOOA adapter.

The runtime lifecycle, Relay activation and failure mapping belong to the
generic adapter and are covered in
``plugins/nemo-agents/tests/unit/test_nooa_fabric_adapter.py``. What is left
here is the analyst's own half of the contract: reading its settings, resolving
its model refs, and naming its Relay scope.
"""

from __future__ import annotations

from typing import Any

import pytest
from nemo_fabric_adapter_contract import models as contract
from nemo_insights_plugin import fabric_adapter
from nemo_insights_plugin.analyst.result import AnalystResult
from nemo_platform_plugin.agents.nooa_contract import NooaInvocation


class _StubClient:
    """Stands in for the async SDK handle so a leak shows up as an unclosed client."""

    def __init__(self, service: str) -> None:
        self.service = service
        self.closed = False

    async def __aenter__(self) -> _StubClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True


def _stub_sdk_factory(clients: list[_StubClient]) -> Any:
    """Record every client handed out so a test can assert none were left open."""

    def factory(service: str) -> _StubClient:
        client = _StubClient(service)
        clients.append(client)
        return client

    return factory


def _models() -> dict[str, contract.AgentModelConfig]:
    config = contract.AgentConfig.from_mapping(
        {
            "harness": {"settings": {}},
            "models": {
                "default": {"provider": "platform", "model": "default/gpt-5"},
                "fast": {"provider": "platform", "model": "default/gpt-5-mini"},
            },
        }
    )
    return dict(config.models)


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


def _invocation(
    settings: dict[str, Any],
    *,
    request_context: dict[str, Any] | None = None,
    telemetry: contract.RuntimeTelemetryContext | None = None,
    models: dict[str, contract.AgentModelConfig] | None = None,
) -> NooaInvocation:
    return NooaInvocation(
        request=contract.AgentRunRequest(input="Analyze telemetry.", context=request_context or {}),
        context=_runtime_context(telemetry),
        settings=settings,
        models=_models() if models is None else models,
    )


async def test_the_analyst_returns_an_unpersisted_change_set(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    clients: list[_StubClient] = []

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="No high-impact failures found."), object()

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory(clients))

    result = await fabric_adapter.run(
        _invocation(
            {
                "entrypoint": "nemo_insights_plugin.fabric_adapter:run",
                "agent": "research-agent",
                "ethos": "# Ethos",
                "base_url": "http://platform",
                "since": "2026-08-21T12:00:00+00:00",
                "evaluation_id": "eval-123",
            },
            request_context={"job_workspace": "workspace"},
        )
    )

    assert result.status is contract.AgentRunStatus.SUCCEEDED
    assert result.output == {
        "response": "No high-impact failures found.",
        "analyst_result": {
            "summary": "No high-impact failures found.",
            "new_insights": [],
            "updated_insights": [],
        },
    }
    assert seen["agent"] == "research-agent"
    assert seen["ethos"] == "# Ethos"
    assert seen["workspace"] == "workspace"
    assert seen["base_url"] == "http://platform"
    assert seen["client"] is clients[0]
    assert clients[0].service == "insights"
    assert seen["since"].isoformat() == "2026-08-21T12:00:00+00:00"
    assert seen["evaluation_id"] == "eval-123"
    assert seen["model_refs"].default == "default/gpt-5"
    assert seen["model_refs"].fast == "default/gpt-5-mini"
    # The entrypoint owns the client, so a successful run must close it too.
    assert clients[0].closed


async def test_the_workspace_falls_back_to_the_request_context(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="done"), object()

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory([]))

    await fabric_adapter.run(
        _invocation({"agent": "research-agent"}, request_context={"job_workspace": "from-context"})
    )

    assert seen["workspace"] == "from-context"


async def test_no_client_is_leaked_when_a_model_ref_is_invalid(monkeypatch) -> None:
    """A settings error must resolve before the ``async with`` opens a client at all."""
    clients: list[_StubClient] = []

    async def fail_if_called(**kwargs: Any) -> tuple[AnalystResult, object]:
        raise AssertionError("run_analyst_change_set should not be reached")

    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory(clients))
    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fail_if_called)

    with pytest.raises(fabric_adapter.AnalystAdapterConfigError) as excinfo:
        await fabric_adapter.run(
            _invocation(
                {"agent": "research-agent", "default_model": "   "},
                request_context={"job_workspace": "workspace"},
            )
        )

    assert "harness.settings.default_model must be a non-empty string" in str(excinfo.value)
    assert all(client.closed for client in clients), "an SDK client was built and never closed"


async def test_a_missing_agent_setting_is_rejected() -> None:
    """The adapter turns this into a failed AgentRunResult; here it is a raise."""
    with pytest.raises(fabric_adapter.AnalystAdapterConfigError) as excinfo:
        await fabric_adapter.run(_invocation({}, request_context={"job_workspace": "workspace"}))

    assert "harness.settings.agent is required" in str(excinfo.value)


async def test_a_non_boolean_enable_observability_is_rejected() -> None:
    """A string `"false"` is truthy under bool(); validate the type instead."""
    with pytest.raises(fabric_adapter.AnalystAdapterConfigError) as excinfo:
        await fabric_adapter.run(
            _invocation(
                {"agent": "research-agent", "enable_observability": "false"},
                request_context={"job_workspace": "workspace"},
            )
        )

    assert "harness.settings.enable_observability must be a boolean" in str(excinfo.value)


async def test_the_fast_model_falls_back_to_the_default(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="done"), object()

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory([]))

    models = _models()
    del models["fast"]

    await fabric_adapter.run(_invocation({"agent": "research-agent"}, models=models))

    assert seen["model_refs"].default == "default/gpt-5"
    assert seen["model_refs"].fast == "default/gpt-5"


async def test_relay_scope_is_named_when_fabric_enabled_relay(monkeypatch) -> None:
    """The scope has to wrap the agent object, which only exists inside the run,
    so the adapter activates Relay and the entrypoint names the scope."""
    seen: dict[str, Any] = {}

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="done"), object()

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory([]))

    telemetry = contract.RuntimeTelemetryContext(relay_enabled=True, config_path="/tmp/relay.json", env={})
    await fabric_adapter.run(_invocation({"agent": "research-agent"}, telemetry=telemetry))

    # Unchanged across the move off a bespoke adapter: exported telemetry keeps
    # identifying the analyst by this name.
    assert seen["relay_scope_name"] == "insights-analyst"
    assert fabric_adapter.ANALYST_RELAY_SCOPE == "insights-analyst"


async def test_relay_scope_is_overridable_per_agent(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="done"), object()

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory([]))

    telemetry = contract.RuntimeTelemetryContext(relay_enabled=True, config_path="/tmp/relay.json", env={})
    await fabric_adapter.run(
        _invocation({"agent": "research-agent", "relay_scope": "custom-scope"}, telemetry=telemetry)
    )

    assert seen["relay_scope_name"] == "custom-scope"


async def test_without_relay_the_agent_runs_unscoped(monkeypatch) -> None:
    """``relay_scope_name=None`` is what tells the run not to open a scope."""
    seen: dict[str, Any] = {}

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="done"), object()

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory([]))

    await fabric_adapter.run(_invocation({"agent": "research-agent"}, telemetry=None))

    assert seen["relay_scope_name"] is None
