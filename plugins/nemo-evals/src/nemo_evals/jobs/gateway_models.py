# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point Harbor agents at the Inference Gateway: check a gateway model at submit, wire it in the job."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from harbor.models.task.config import NetworkMode, TaskConfig
from nemo_agents_plugin.runner.deployments_backend import resolve_agent_gateway_url, rewrite_fabric_config_base_urls
from nemo_agents_plugin.utils import get_base_url, get_internal_base_url
from nemo_evals.jobs.agent_spec import HarborRunnerTarget, RegisteredAgentSource
from nemo_evals.jobs.metric_resolution import HelixMetricModelResolver
from nemo_evals.source_authorization import SourceReadAuthorizer
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.refs import model_entity_route_openai_url, parse_workspace_name_ref
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborGatewayAccess


async def check_gateway_model(
    target: object,
    *,
    async_sdk: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None,
) -> None:
    """Refuse a Harbor target whose gateway model is missing, unreadable by the caller, or uncallable.

    Raises:
        ValueError: No model or VirtualModel has the name, or platform auth is on and the agent has no
            platform key in ``env_secrets``.
    """
    if not isinstance(target, HarborRunnerTarget) or target.gateway_model is None:
        return
    env = target.gateway_agent_env
    assert env is not None
    if platform_auth_enabled() and env.api_key_env not in target.env_secrets:
        raise ValueError(
            f"platform auth is on, so the agent needs a key to call the Inference Gateway: put a Helix access key "
            f"scoped to `inference` in env_secrets under {env.api_key_env}"
        )
    if async_sdk is None:
        raise ValueError("resolving a gateway model requires a platform client")
    models_client = client_from_platform(async_sdk, AsyncModelsClient)
    await HelixMetricModelResolver(models_client, authorize=authorize).resolve_model(target.gateway_model)


def agent_gateway_origin(*, sandboxed: bool) -> str:
    """The platform origin an agent reaches from its task container.

    ``sandboxed`` is a Kubernetes task Job's OpenSandbox sandbox, which reaches the in-cluster API; otherwise
    Harbor's Docker task container, which reaches a loopback platform through ``host.docker.internal``.
    """
    base_url = get_base_url()
    if not sandboxed:
        return resolve_agent_gateway_url(base_url, mode="docker")
    internal_base_url = get_internal_base_url()
    if internal_base_url is None and urlsplit(base_url).hostname not in LOOPBACK_ADDRESSES:
        internal_base_url = base_url
    return resolve_agent_gateway_url(base_url, mode="k8s", internal_base_url=internal_base_url)


def harbor_gateway_access(target: HarborRunnerTarget, origin: str) -> HarborGatewayAccess | None:
    """The gateway route and variables for a Harbor agent that names a gateway model."""
    env = target.gateway_agent_env
    if target.gateway_model is None or env is None:
        return None
    workspace, name = parse_workspace_name_ref(target.gateway_model.root, label="Gateway model")
    route = model_entity_route_openai_url(base_url=origin, workspace=workspace, name=name).removesuffix("/v1")
    return HarborGatewayAccess(
        base_url=route + env.route_suffix, base_url_env=env.base_url_env, api_key_env=env.api_key_env
    )


def rebase_registered_agent_config(target: HarborRunnerTarget, agent_kwargs: dict[str, Any], origin: str) -> None:
    """Point a registered agent's gateway URLs, in ``agent_kwargs["fabric_config"]``, at ``origin``."""
    config = agent_kwargs.get("fabric_config")
    if isinstance(target.source, RegisteredAgentSource) and isinstance(config, dict):
        agent_kwargs["fabric_config"] = rewrite_fabric_config_base_urls(config, origin)


def check_tasks_reach_the_gateway(
    target: object, tasks: Iterable[tuple[str, Mapping[str, Any]]], agent_channel: Sequence[str]
) -> None:
    """Refuse no-network tasks whose sandboxes couldn't reach the gateway model the agent calls.

    ``tasks`` are (task id, task.toml config) pairs. A no-network sandbox reaches only the profile's
    ``agent_channel`` hosts.

    Raises:
        ValueError: Some no-network task's sandbox can't reach the gateway.
    """
    if not isinstance(target, HarborRunnerTarget) or target.gateway_model is None:
        return
    no_network = [
        task_id
        for task_id, config in tasks
        if TaskConfig.model_validate(config).environment.resolve_baseline().network_mode == NetworkMode.NO_NETWORK
    ]
    if not no_network:
        return
    host = urlsplit(agent_gateway_origin(sandboxed=True)).hostname or ""
    if not _in_agent_channel(host, agent_channel):
        raise ValueError(
            f"gateway_unreachable: no-network tasks {no_network} can't reach the Inference Gateway at {host}: "
            "add it to the sandbox profile's agent_channel"
        )


def _in_agent_channel(host: str, agent_channel: Sequence[str]) -> bool:
    host = host.strip().rstrip(".").lower()
    for entry in (entry.strip().rstrip(".").lower() for entry in agent_channel):
        if entry == host or (entry.startswith("*.") and host.endswith(entry[1:])):
            return True
        try:
            if ipaddress.ip_address(host) in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False
