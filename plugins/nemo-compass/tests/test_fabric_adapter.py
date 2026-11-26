# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric adapter contract for Compass."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from nemo_compass_plugin import fabric_adapter
from nemo_compass_plugin.analysis.result import CompassResult
from nemo_fabric_adapter_contract import models as contract


def _agent_config() -> contract.AgentConfig:
    """``compass.analyst``'s derived config: no settings and no models, since a run's inputs arrive in its context."""
    return contract.AgentConfig.from_mapping({"harness": {"settings": {}}})


def _run_context(*, workspace: str | None = "workspace", **fields: Any) -> dict[str, Any]:
    """A prepared Compass context, plus the platform's ``nemo_helix`` key as a job adds it."""
    context: dict[str, Any] = {
        "agent": "research-agent",
        "default_model": "default/gpt-5",
        "fast_model": "default/gpt-5-mini",
        **fields,
    }
    if workspace is not None:
        context["nemo_helix"] = {"job_workspace": workspace}
    return context


class _StubClient:
    """Stands in for the async client handle so a leak shows up as an unclosed client."""

    def __init__(self, service: str) -> None:
        self.service = service
        self.closed = False

    async def __aenter__(self) -> _StubClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True


def _stub_client_factory(clients: list[_StubClient]) -> Any:
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
            control_location=contract.ControlLocation.IN_ENV_CONTROL,
            ownership=contract.EnvironmentOwnership.CALLER_OWNED,
        ),
        artifacts=contract.ArtifactManifest(),
        telemetry=telemetry,
    )


def _request(context: dict[str, Any] | None = None) -> contract.AgentRunRequest:
    return contract.AgentRunRequest(input="Analyze telemetry.", context=context or {})


@pytest.fixture
def isolated_relay_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_home = tmp_path / "xdg"
    config_home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.chdir(tmp_path)
    return config_home


def _clean_activation() -> SimpleNamespace:
    return SimpleNamespace(report={"config": {"diagnostics": []}})


def _minimal_relay_telemetry(tmp_path: Path) -> contract.RuntimeTelemetryContext:
    config_path = tmp_path / "relay-minimal.json"
    config_path.write_text(
        json.dumps({"relay": {"config": {"version": 1, "components": [{"kind": "observability", "enabled": True}]}}}),
        encoding="utf-8",
    )
    return contract.RuntimeTelemetryContext(relay_enabled=True, config_path=str(config_path), env={})


async def test_fabric_adapter_returns_unpersisted_compass_result(monkeypatch) -> None:
    seen: dict[str, Any] = {}
    clients: list[_StubClient] = []

    async def fake_run_compass_change_set(**kwargs: Any) -> tuple[CompassResult, object]:
        seen.update(kwargs)
        return CompassResult(summary="No high-impact failures found."), object()

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fake_run_compass_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory(clients))
    monkeypatch.setenv("NHX_BASE_URL", "http://platform")

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    result = await runtime.invoke(
        _request(_run_context(ethos="# Ethos", since="2026-08-21T12:00:00+00:00", evaluation_id="eval-123")),
        _runtime_context(),
    )

    assert result.status is contract.AgentRunStatus.SUCCEEDED
    assert result.output == {
        "response": "No high-impact failures found.",
        "compass_result": {
            "summary": "No high-impact failures found.",
            "new_insights": [],
            "updated_insights": [],
            "notes": [],
        },
    }
    assert seen["agent"] == "research-agent"
    assert seen["ethos"] == "# Ethos"
    assert seen["workspace"] == "workspace"
    assert seen["base_url"] == "http://platform"
    assert seen["client"] is clients[0]
    assert clients[0].service == "compass"
    assert seen["since"].isoformat() == "2026-08-21T12:00:00+00:00"
    assert seen["evaluation_id"] == "eval-123"
    assert seen["model_refs"].default == "default/gpt-5"
    assert seen["model_refs"].fast == "default/gpt-5-mini"
    # The adapter owns the client now, so a successful run must close it too.
    assert clients[0].closed


async def test_fabric_adapter_never_enables_self_observability(monkeypatch) -> None:
    """Compass's own OTLP exporter cannot authenticate in a task pod, and the
    derived config has no setting to turn it off, so the adapter must."""
    seen: dict[str, Any] = {}

    async def fake_run_compass_change_set(**kwargs: Any) -> tuple[CompassResult, object]:
        seen.update(kwargs)
        return CompassResult(summary="ok"), object()

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fake_run_compass_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory([]))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})
    await runtime.invoke(_request(_run_context()), _runtime_context())

    assert seen["enable_observability"] is False


@pytest.mark.parametrize(
    "context",
    [
        pytest.param(_run_context(workspace=None), id="no-platform-key"),
        pytest.param({**_run_context(workspace=None), "job_workspace": "caller"}, id="top-level-only"),
        pytest.param(_run_context(workspace="  "), id="blank"),
    ],
)
async def test_fabric_adapter_fails_without_a_job_workspace(monkeypatch, context: dict[str, Any]) -> None:
    """There is no fallback workspace: outside an execute job, Compass has nowhere to read from."""
    clients: list[_StubClient] = []

    async def fail_if_called(**kwargs: Any) -> tuple[CompassResult, object]:
        raise AssertionError("run_compass_change_set should not be reached")

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fail_if_called)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory(clients))
    monkeypatch.setenv("NHX_WORKSPACE", "ambient")

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    result = await runtime.invoke(_request(context), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert "context.nemo_helix.job_workspace is required" in result.error.message
    assert clients == []


async def test_fabric_adapter_runs_unscoped_without_since_or_evaluation_id(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_run_compass_change_set(**kwargs: Any) -> tuple[CompassResult, object]:
        seen.update(kwargs)
        return CompassResult(summary="ok"), object()

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fake_run_compass_change_set)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory([]))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    await runtime.invoke(_request(_run_context(workspace="w")), _runtime_context())

    assert seen["since"] is None
    assert seen["evaluation_id"] is None


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        pytest.param({"since": "last tuesday"}, "since", id="bad-since"),
        pytest.param({"evaluation_id": 7}, "evaluation_id", id="numeric-eval"),
        pytest.param({"agent": None}, "agent", id="no-agent"),
        pytest.param({"default_model": None}, "default_model", id="no-default-model"),
        pytest.param({"workspace": "other"}, "workspace", id="unknown-key"),
    ],
)
async def test_fabric_adapter_rejects_a_context_that_is_not_a_compass_run(
    monkeypatch, fields: dict[str, Any], field: str
) -> None:
    """A malformed context must fail the run, not widen it to every trace, and
    before the ``async with`` opens a client at all."""
    clients: list[_StubClient] = []

    async def fail_if_called(**kwargs: Any) -> tuple[CompassResult, object]:
        raise AssertionError("run_compass_change_set should not be reached")

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fail_if_called)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory(clients))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})
    context = {key: value for key, value in {**_run_context(), **fields}.items() if value is not None}

    result = await runtime.invoke(_request(context), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert "the request context is not a Compass run context" in result.error.message
    assert field in result.error.message
    assert clients == []


async def test_fabric_adapter_logs_the_failure_with_its_traceback(monkeypatch, caplog) -> None:
    """``str(error)`` alone loses the traceback, and the adapter's stderr is
    captured to a run artifact the job dumps on failure - so logging here is
    what puts the traceback somewhere anyone will actually read."""
    clients: list[_StubClient] = []

    async def fail(**kwargs: Any) -> tuple[CompassResult, object]:
        raise RuntimeError("gateway returned 502")

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fail)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory(clients))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    with caplog.at_level("ERROR", logger="nemo_compass_plugin.fabric_adapter"):
        result = await runtime.invoke(_request(_run_context()), _runtime_context())

    assert result.status is contract.AgentRunStatus.FAILED
    assert result.error is not None
    assert result.error.code == "compass_failed"
    assert "Compass run failed." in caplog.text
    assert "gateway returned 502" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text


async def test_fabric_adapter_logs_the_whole_cause_chain(monkeypatch, caplog) -> None:
    """The originating error is what distinguishes one failure from another:
    an LLM error raised after retries reads identically whatever provoked it."""
    clients: list[_StubClient] = []

    originating = "Function 'abc-123': Not found for account"
    raised = "LLM API error after 3 retries"

    async def fail(**kwargs: Any) -> tuple[CompassResult, object]:
        try:
            raise ValueError(originating)
        except ValueError as cause:
            raise RuntimeError(raised) from cause

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fail)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory(clients))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    with caplog.at_level("ERROR", logger="nemo_compass_plugin.fabric_adapter"):
        await runtime.invoke(_request(_run_context()), _runtime_context())

    assert originating in caplog.text, "root cause was dropped"
    assert raised in caplog.text


async def test_relay_activates_fabrics_config_and_scopes_the_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_relay_config: Path
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
                                    "version": 4,
                                    "atif": {
                                        "enabled": True,
                                        "agent_name": "compass",
                                        "storage": [
                                            {
                                                "type": "http",
                                                "endpoint": "http://platform/apis/intake/v2/workspaces/w/ingest/atif",
                                                "header_env": {"X-NHX-Principal-Id": "NHX_HEADER"},
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

    async def fake_run_compass_change_set(**kwargs: Any) -> tuple[CompassResult, object]:
        seen.update(kwargs)
        return CompassResult(summary="done"), object()

    @asynccontextmanager
    async def fake_activate(config: Any) -> AsyncGenerator[SimpleNamespace, None]:
        seen["plugin_config"] = config
        # Relay resolves header_env against the environment while exporting,
        # so the variables have to be set for the duration of the run.
        seen["env_during_run"] = os.environ.get("FABRIC_RELAY_CONFIG_PATH")
        yield _clean_activation()

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fake_run_compass_change_set)
    monkeypatch.setattr(fabric_adapter, "activate_relay_plugin", fake_activate)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory([]))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})
    telemetry = contract.RuntimeTelemetryContext(
        relay_enabled=True,
        config_path=str(relay_config),
        env={"FABRIC_RELAY_CONFIG_PATH": str(relay_config)},
    )

    result = await runtime.invoke(_request(_run_context(workspace="w")), _runtime_context(telemetry))

    assert result.status is contract.AgentRunStatus.SUCCEEDED
    assert seen["relay_scope_name"] == fabric_adapter.COMPASS_RELAY_SCOPE
    assert seen["plugin_config"]["components"][0]["kind"] == "observability"
    assert seen["env_during_run"] == str(relay_config)
    # The runtime serves many invocations; a leftover config path would make
    # the next one export against a stale, possibly deleted, config.
    assert "FABRIC_RELAY_CONFIG_PATH" not in os.environ


async def test_relay_refuses_an_ambient_user_plugin_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_relay_config: Path
) -> None:
    (isolated_relay_config / "nemo-relay").mkdir()
    (isolated_relay_config / "nemo-relay" / "plugins.toml").write_text("version = 1\n", encoding="utf-8")
    activated: list[Any] = []

    @asynccontextmanager
    async def fake_activate(config: Any) -> AsyncGenerator[SimpleNamespace, None]:
        activated.append(config)
        yield _clean_activation()

    async def fail_if_called(**kwargs: Any) -> tuple[CompassResult, object]:
        raise AssertionError("compass ran")

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fail_if_called)
    monkeypatch.setattr(fabric_adapter, "activate_relay_plugin", fake_activate)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory([]))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    result = await runtime.invoke(
        _request(_run_context(workspace="w")), _runtime_context(_minimal_relay_telemetry(tmp_path))
    )

    assert result.status is contract.AgentRunStatus.FAILED
    assert "ambient user or project configuration" in result.output["response"]
    assert activated == []


async def test_relay_refuses_plugin_config_inherited_during_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_relay_config: Path
) -> None:
    inherited = SimpleNamespace(
        report={
            "config": {
                "diagnostics": [
                    {
                        "code": "plugin.configuration_inherited",
                        "level": "warning",
                        "message": "inherited plugin configuration from discovered file: /home/user/plugins.toml",
                    }
                ]
            }
        }
    )

    @asynccontextmanager
    async def fake_activate(config: Any) -> AsyncGenerator[SimpleNamespace, None]:
        yield inherited

    async def fail_if_called(**kwargs: Any) -> tuple[CompassResult, object]:
        raise AssertionError("compass ran")

    monkeypatch.setattr(fabric_adapter, "run_compass_change_set", fail_if_called)
    monkeypatch.setattr(fabric_adapter, "activate_relay_plugin", fake_activate)
    monkeypatch.setattr(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory([]))

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    result = await runtime.invoke(
        _request(_run_context(workspace="w")), _runtime_context(_minimal_relay_telemetry(tmp_path))
    )

    assert result.status is contract.AgentRunStatus.FAILED
    assert "/home/user/plugins.toml" in result.output["response"]


async def test_without_relay_the_agent_runs_unscoped() -> None:
    """Relay is opt-in per invocation; Fabric says when."""
    seen: dict[str, Any] = {}

    async def fake_run_compass_change_set(**kwargs: Any) -> tuple[CompassResult, object]:
        seen.update(kwargs)
        return CompassResult(summary="done"), object()

    runtime = fabric_adapter.CompassRuntime()
    await runtime.start({"config": _agent_config()})

    with (
        mock.patch.object(fabric_adapter, "run_compass_change_set", fake_run_compass_change_set),
        mock.patch.object(fabric_adapter, "get_async_task_nemo_client", _stub_client_factory([])),
    ):
        await runtime.invoke(_request(_run_context(workspace="w")), _runtime_context(None))

    assert seen["relay_scope_name"] is None
