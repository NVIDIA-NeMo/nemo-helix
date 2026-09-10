# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the agents plugin decides the address it dials for a deployment."""

from __future__ import annotations

import pytest
from nemo_agents_plugin.config import AgentsConfig, DeploymentsRunnerConfig
from nemo_agents_plugin.deployment_routing import get_deployment_transport
from nemo_agents_plugin.entities import AgentDeployment, DeploymentMode, Endpoint
from nemo_deployments_plugin.config import DeploymentsConfig, ExecutorConfigEntry
from nemo_deployments_plugin.endpoint_transport import DIRECT


def _deployment(mode: DeploymentMode) -> AgentDeployment:
    return AgentDeployment(
        name="dep",
        workspace="default",
        agent="calc",
        status="running",
        endpoint="http://localhost:9001" if mode == "subprocess" else "",
        deployment_mode=mode,
        endpoints=[] if mode == "subprocess" else [Endpoint(name="http", url="http://x:1")],
    )


@pytest.fixture
def executors(monkeypatch: pytest.MonkeyPatch) -> None:
    deployments = DeploymentsConfig(
        executors=[
            ExecutorConfigEntry(name="local-docker", backend="docker"),
            ExecutorConfigEntry(
                name="openshell",
                backend="openshell",
                config={"gateway_endpoint": "https://openshell.openshell.svc:8080"},
            ),
        ],
        default_executor="local-docker",
    )
    agents = AgentsConfig(deployments=DeploymentsRunnerConfig(docker_executor="local-docker", k8s_executor="openshell"))
    monkeypatch.setattr(DeploymentsConfig, "get", classmethod(lambda cls: deployments))
    monkeypatch.setattr(AgentsConfig, "get", classmethod(lambda cls: agents))


def test_subprocess_deployments_are_always_direct(executors: None) -> None:
    assert get_deployment_transport(_deployment("subprocess")) is DIRECT


def test_container_mode_follows_the_executor_the_deploy_path_chose(executors: None) -> None:
    assert get_deployment_transport(_deployment("docker")) is DIRECT
    assert get_deployment_transport(_deployment("k8s")).connect_base == "https://openshell.openshell.svc:8080"
