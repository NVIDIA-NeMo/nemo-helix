# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric adapter contract for the Insights analyst."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from nemo_fabric_adapter_contract import models as contract
from nemo_insights_plugin import fabric_adapter
from nemo_insights_plugin.analyst.result import AnalystResult


def _agent_config(settings: dict[str, Any]) -> contract.AgentConfig:
    return contract.AgentConfig.from_mapping(
        {
            "harness": {"settings": settings},
            "models": {
                "default": {
                    "provider": "platform",
                    "model": "default/gpt-5",
                },
                "fast": {
                    "provider": "platform",
                    "model": "default/gpt-5-mini",
                },
            },
        }
    )


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


def _runtime_context(telemetry: contract.RuntimeTelemetryContext | None = None) -> contract.RuntimeContext:
    """The context Fabric hands every invocation; telemetry is the part we read."""
    return contract.RuntimeContext(
        runtime_id="runtime-1",
        invocation_id="invocation-1",
        request_id="request-1",
        environment=contract.EnvironmentHandle(
            environment_id="environment-1",
            provider="local",
            control_location="in_env_control",
            ownership="caller_owned",
        ),
        artifacts=contract.ArtifactManifest(),
        telemetry=telemetry,
    )


def _request(context: dict[str, Any] | None = None) -> contract.AgentRunRequest:
    return contract.AgentRunRequest(input="Analyze telemetry.", context=context or {})


async def test_fabric_adapter_returns_unpersisted_analyst_result(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    clients: list[_StubClient] = []

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="No high-impact failures found."), object()

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory(clients))

    runtime = fabric_adapter.InsightsAnalystRuntime()
    await runtime.start(
        {
            "config": _agent_config(
                {
                    "agent": "research-agent",
                    "ethos": "# Ethos",
                    "base_url": "http://platform",
                    "since": "2026-08-21T12:00:00+00:00",
                    "evaluation_id": "eval-123",
                }
            )
        }
    )

    result = await runtime.invoke(_request({"job_workspace": "workspace"}), _runtime_context())

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
    # The adapter owns the client now, so a successful run must close it too.
    assert clients[0].closed


async def test_fabric_adapter_does_not_leak_a_client_when_a_model_ref_is_invalid(monkeypatch) -> None:
    """A settings error must resolve before the ``async with`` opens a client at all."""
    clients: list[_StubClient] = []

    async def fail_if_called(**kwargs: Any) -> tuple[AnalystResult, object]:
        raise AssertionError("run_analyst_change_set should not be reached")

    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory(clients))
    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fail_if_called)

    runtime = fabric_adapter.InsightsAnalystRuntime()
    await runtime.start({"config": _agent_config({"agent": "research-agent", "default_model": "   "})})

    result = await runtime.invoke(_request({"job_workspace": "workspace"}), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert "harness.settings.default_model must be a non-empty string" in result.error.message
    assert all(client.closed for client in clients), "an SDK client was built and never closed"


async def test_fabric_adapter_reports_configuration_failure() -> None:
    runtime = fabric_adapter.InsightsAnalystRuntime()
    await runtime.start({"config": _agent_config({})})

    result = await runtime.invoke(_request({"job_workspace": "workspace"}), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "insights_analyst_failed"
    assert "harness.settings.agent is required" in result.error.message


async def test_fabric_adapter_logs_the_failure_with_its_traceback(monkeypatch, caplog) -> None:
    """``str(error)`` alone loses the traceback, and the adapter's stderr is
    captured to a run artifact the job dumps on failure - so logging here is
    what puts the traceback somewhere anyone will actually read."""
    clients: list[_StubClient] = []

    async def fail(**kwargs: Any) -> tuple[AnalystResult, object]:
        raise RuntimeError("gateway returned 502")

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fail)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory(clients))

    runtime = fabric_adapter.InsightsAnalystRuntime()
    await runtime.start({"config": _agent_config({"agent": "research-agent"})})

    with caplog.at_level("ERROR", logger="nemo_insights_plugin.fabric_adapter"):
        result = await runtime.invoke(_request({"job_workspace": "workspace"}), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "insights_analyst_failed"
    assert "Insights analyst run failed." in caplog.text
    assert "gateway returned 502" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text


async def test_fabric_adapter_logs_the_whole_cause_chain(monkeypatch, caplog) -> None:
    """The originating error is what distinguishes one failure from another:
    an LLM error raised after retries reads identically whatever provoked it."""
    clients: list[_StubClient] = []

    originating = "Function 'abc-123': Not found for account"
    raised = "LLM API error after 3 retries"

    async def fail(**kwargs: Any) -> tuple[AnalystResult, object]:
        try:
            raise ValueError(originating)
        except ValueError as cause:
            raise RuntimeError(raised) from cause

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fail)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory(clients))

    runtime = fabric_adapter.InsightsAnalystRuntime()
    await runtime.start({"config": _agent_config({"agent": "research-agent"})})

    with caplog.at_level("ERROR", logger="nemo_insights_plugin.fabric_adapter"):
        await runtime.invoke(_request({"job_workspace": "workspace"}), _runtime_context())

    assert originating in caplog.text, "root cause was dropped"
    assert raised in caplog.text


async def test_relay_activates_fabrics_config_and_scopes_the_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fabric resolves the whole export; the adapter activates it and names the scope.

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
                        "components": [
                            {
                                "kind": "observability",
                                "enabled": True,
                                # The shape Fabric writes from the agents plugin's
                                # auto-wiring: destination and credentials already
                                # resolved, nothing for the adapter to decide.
                                "config": {
                                    "version": 3,
                                    "atif": {
                                        "enabled": True,
                                        "agent_name": "insights-analyst",
                                        "storage": [
                                            {
                                                "type": "http",
                                                "endpoint": "http://platform/apis/intake/v2/workspaces/w/ingest/atif",
                                                "header_env": {"X-NMP-Principal-Id": "NMP_HEADER"},
                                            }
                                        ],
                                    },
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="done"), object()

    @asynccontextmanager
    async def fake_plugin(config: Any) -> AsyncIterator[None]:
        seen["plugin_config"] = config
        # Relay resolves header_env against the environment while exporting,
        # so the variables have to be set for the duration of the run.
        seen["env_during_run"] = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
        yield

    monkeypatch.setattr(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set)
    monkeypatch.setattr(fabric_adapter.relay_plugin, "plugin", fake_plugin)
    monkeypatch.setattr(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory([]))

    runtime = fabric_adapter.InsightsAnalystRuntime()
    await runtime.start({"config": _agent_config({"agent": "research-agent"})})
    telemetry = contract.RuntimeTelemetryContext(
        relay_enabled=True,
        config_path=str(relay_config),
        env={"FABRIC_RELAY_CONFIG_PATH": str(relay_config)},
    )

    result = await runtime.invoke(_request({"job_workspace": "w"}), _runtime_context(telemetry))

    assert result.status is contract.AgentRunStatus.SUCCEEDED
    assert seen["relay_scope_name"] == fabric_adapter.ANALYST_RELAY_SCOPE
    assert seen["plugin_config"]["components"][0]["kind"] == "observability"
    assert seen["env_during_run"] == str(relay_config)
    # The runtime serves many invocations; a leftover config path would make
    # the next one export against a stale, possibly deleted, config.
    assert "FABRIC_RELAY_CONFIG_PATH" not in os.environ


async def test_without_relay_the_agent_runs_unscoped() -> None:
    """Relay is opt-in per invocation; Fabric says when."""
    seen: dict[str, Any] = {}

    async def fake_run_analyst_change_set(**kwargs: Any) -> tuple[AnalystResult, object]:
        seen.update(kwargs)
        return AnalystResult(summary="done"), object()

    runtime = fabric_adapter.InsightsAnalystRuntime()
    await runtime.start({"config": _agent_config({"agent": "research-agent"})})

    with (
        mock.patch.object(fabric_adapter, "run_analyst_change_set", fake_run_analyst_change_set),
        mock.patch.object(fabric_adapter, "get_async_task_sdk", _stub_sdk_factory([])),
    ):
        await runtime.invoke(_request({"job_workspace": "w"}), _runtime_context(None))

    assert seen["relay_scope_name"] is None
