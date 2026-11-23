# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deployment_mode ``opensandbox`` in the agents plugin."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from nemo_agents_plugin.api.v2.deployments import deployment_modes
from nemo_agents_plugin.config import AgentsConfig, DeploymentsRunnerConfig
from nemo_agents_plugin.deployment_routing import get_deployment_transport
from nemo_agents_plugin.entities import (
    AgentDeployment,
    Endpoint,
    is_container_deployment_mode,
    supports_image_entrypoint,
)
from nemo_agents_plugin.runner.deployments_backend import (
    DeploymentsRunnerBackend,
    UnreachableGatewayURLError,
    executor_for_mode,
    opensandbox_platform_egress_url,
    require_executor_matches_mode,
    resolve_agent_gateway_url,
)
from nemo_deployments_plugin.backends.opensandbox.endpoint import OpenSandboxEndpointTransport
from nemo_deployments_plugin.config import DeploymentsConfig, ExecutorConfigEntry

OSB = {"domain": "opensandbox-server.opensandbox-system.svc", "gateway_address": "opensandbox-gateway.svc"}
EGRESS = {"platform_egress": {"host": "nhx-api.nemo.svc.cluster.local", "port": 80}}
PLATFORM_URL = "http://nhx-api.nemo.svc.cluster.local:80"


def _configure(monkeypatch: pytest.MonkeyPatch, executor_config: dict[str, Any]) -> AgentsConfig:
    deployments = DeploymentsConfig(
        executors=[
            ExecutorConfigEntry(name="local-docker", backend="docker"),
            ExecutorConfigEntry(name="sandboxes", backend="opensandbox", config=executor_config),
        ],
        default_executor="local-docker",
    )
    agents = AgentsConfig(
        deployments=DeploymentsRunnerConfig(opensandbox_executor="sandboxes", default_image="fabric:latest")
    )
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: deployments))
    monkeypatch.setattr(AgentsConfig, "get", classmethod(lambda cls: agents))
    return agents


def test_opensandbox_is_a_container_mode_without_image_entrypoint_support() -> None:
    assert is_container_deployment_mode("opensandbox")
    assert not supports_image_entrypoint("opensandbox")


def test_executor_for_mode_prefers_the_opensandbox_executor() -> None:
    config = DeploymentsRunnerConfig(default_executor="default-exec", opensandbox_executor="sandboxes")
    assert executor_for_mode(config, "opensandbox") == "sandboxes"
    assert executor_for_mode(DeploymentsRunnerConfig(default_executor="default-exec"), "opensandbox") == "default-exec"


def test_gateway_url_is_the_executors_platform_egress() -> None:
    url = resolve_agent_gateway_url("http://localhost:8080", mode="opensandbox", platform_egress_url=PLATFORM_URL + "/")
    assert url == PLATFORM_URL
    with pytest.raises(UnreachableGatewayURLError, match="opensandbox executor has no platform_egress"):
        resolve_agent_gateway_url("http://localhost:8080", mode="opensandbox")
    assert (
        resolve_agent_gateway_url("http://localhost:8080", mode="opensandbox", override="https://helix.example")
        == "https://helix.example"
    )


def test_platform_egress_url_reads_the_executor_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, {**OSB, **EGRESS})
    assert opensandbox_platform_egress_url("sandboxes") == PLATFORM_URL
    assert opensandbox_platform_egress_url("local-docker") is None


def test_platform_egress_url_is_none_without_a_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, OSB)
    assert opensandbox_platform_egress_url("sandboxes") is None


def test_mode_must_run_on_an_opensandbox_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, OSB)
    require_executor_matches_mode("sandboxes", "opensandbox")
    with pytest.raises(ValueError, match="runs on 'docker'"):
        require_executor_matches_mode("local-docker", "opensandbox")


def test_mode_is_enabled_by_its_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    agents = _configure(monkeypatch, OSB)
    modes = {entry.mode: entry for entry in deployment_modes(agents).data}
    assert (modes["opensandbox"].enabled, modes["opensandbox"].requires_image) == (True, False)


def test_deployments_dial_through_the_opensandbox_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure(monkeypatch, OSB)
    deployment = AgentDeployment(
        name="dep",
        workspace="default",
        agent="calc",
        status="running",
        endpoint="",
        deployment_mode="opensandbox",
        endpoints=[Endpoint(name="http", url="http://abc-8000.osb.helix.internal")],
    )
    assert get_deployment_transport(deployment) == OpenSandboxEndpointTransport(
        connect_base="http://opensandbox-gateway.svc",
        server_base="http://opensandbox-server.opensandbox-system.svc",
    )


@asynccontextmanager
async def _entity_client_scope(entities: Any) -> AsyncGenerator[Any]:
    yield entities


async def test_create_deployment_compiles_a_config_for_the_opensandbox_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agents = _configure(monkeypatch, {**OSB, **EGRESS})
    backend = DeploymentsRunnerBackend(agents)
    entities = AsyncMock()
    backend._entity_client = MagicMock(side_effect=lambda: _entity_client_scope(entities))  # type: ignore[method-assign]
    config = {
        "llms": {
            "llm": {
                "_type": "openai",
                "base_url": "http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            }
        }
    }

    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        info = await backend.create_deployment(
            workspace="default", name="hello-dep", config=config, port=0, deployment_mode="opensandbox"
        )

    assert info.status == "starting"
    created_config = entities.create.await_args_list[0].args[0]
    created_deployment = entities.create.await_args_list[1].args[0]
    baked = yaml.safe_load(created_config.config_files[0].content)
    assert baked["llms"]["llm"]["base_url"] == (
        f"{PLATFORM_URL}/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    )
    container = created_config.containers[0]
    assert container.command == ["nat", "start", "fastapi"]
    assert {"name": "NHX_BASE_URL", "value": PLATFORM_URL} in [
        {"name": var.name, "value": var.value} for var in container.env
    ]
    assert created_config.labels["nemo.agents/mode"] == "opensandbox"
    assert created_deployment.executor == "sandboxes"


async def test_create_deployment_without_platform_egress_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    agents = _configure(monkeypatch, OSB)
    backend = DeploymentsRunnerBackend(agents)
    entities = AsyncMock()
    backend._entity_client = MagicMock(side_effect=lambda: _entity_client_scope(entities))  # type: ignore[method-assign]

    with patch("nemo_agents_plugin.runner.deployments_backend.get_base_url", return_value="http://localhost:8080"):
        info = await backend.create_deployment(
            workspace="default", name="hello-dep", config={"llms": {}}, port=0, deployment_mode="opensandbox"
        )

    assert info.status == "failed"
    assert "platform_egress" in (info.error or "")
    entities.create.assert_not_awaited()


async def test_create_deployment_rejects_the_image_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    agents = _configure(monkeypatch, {**OSB, **EGRESS})
    info = await DeploymentsRunnerBackend(agents).create_deployment(
        workspace="default",
        name="hello-dep",
        config={"llms": {}},
        port=0,
        deployment_mode="opensandbox",
        use_image_entrypoint=True,
    )
    assert info.status == "failed"
    assert "use_image_entrypoint is not supported for deployment_mode 'opensandbox'" in (info.error or "")
