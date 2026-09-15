# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared endpoint resolution for agent deployments."""

from nemo_agents_plugin.entities import AgentDeployment, is_container_deployment_mode
from nemo_deployments_plugin.endpoint_transport import DIRECT, EndpointTransport, endpoint_transport


def get_deployment_endpoint(deployment: AgentDeployment) -> str | None:
    """Return the HTTP address for a subprocess or container deployment."""
    if is_container_deployment_mode(deployment.deployment_mode):
        for endpoint in deployment.endpoints:
            if endpoint.protocol in ("http", "https") and endpoint.url:
                return endpoint.url
        return None
    return deployment.endpoint or None


def is_deployment_routable(deployment: AgentDeployment) -> bool:
    """Return whether a running deployment currently has a resolvable endpoint."""
    return deployment.status == "running" and get_deployment_endpoint(deployment) is not None


def get_deployment_transport(deployment: AgentDeployment) -> EndpointTransport:
    """How the platform dials *deployment*'s endpoint."""
    if not is_container_deployment_mode(deployment.deployment_mode):
        return DIRECT
    from nemo_agents_plugin.config import AgentsConfig
    from nemo_agents_plugin.runner.deployments_backend import executor_for_mode

    return endpoint_transport(executor_for_mode(AgentsConfig.get().deployments, deployment.deployment_mode))
