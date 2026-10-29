# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point Harbor agents at the Inference Gateway: check a gateway model at submit, wire it in the job."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from harbor.models.task.config import NetworkMode, TaskConfig
from harbor.models.trial.config import AgentConfig as TrialAgentConfig
from harbor.trial.network_policy import resolve_agent_phase_policy
from nemo_agents_plugin.runner.deployments_backend import resolve_agent_gateway_url, rewrite_fabric_config_base_urls
from nemo_agents_plugin.utils import get_base_url, get_internal_base_url
from nemo_evals.gateway.middleware.closed_book import MIDDLEWARE_NAME as CLOSED_BOOK_MIDDLEWARE
from nemo_evals.jobs.agent_spec import HarborRunnerTarget, RegisteredAgentSource
from nemo_evals.jobs.gateway_agents import GATEWAY_PLACEHOLDER_KEY
from nemo_evals.jobs.metric_resolution import HelixMetricModelResolver
from nemo_evals.jobs.secret_env import SecretEnvError
from nemo_evals.source_authorization import SourceReadAuthorizer, authorize_source_read
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES
from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.refs import model_entity_route_openai_url, parse_workspace_name_ref
from nemo_helix_plugin.virtual_models.client import AsyncVirtualModelsClient
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborEnvironmentConfig, HarborRuntimeConfig
from nhx_evals_sdk.values.common import SecretRef


async def check_gateway_model(
    target: object,
    *,
    async_sdk: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None,
) -> None:
    """Refuse a Harbor target whose gateway model is missing, unreadable by the caller, or uncallable.

    Raises:
        ValueError: No model or VirtualModel has the name, or platform auth is on and the agent can't
            authenticate to the gateway.
    """
    if not isinstance(target, HarborRunnerTarget) or target.gateway_model is None:
        return
    env = target.gateway_agent_env
    assert env is not None
    if platform_auth_enabled() and env.api_style == "anthropic":
        raise ValueError(
            "platform auth is on, and an Anthropic-style agent can't authenticate to the Inference Gateway: it sends "
            "its key as `x-api-key`, and the gateway accepts only bearer tokens"
        )
    if platform_auth_enabled() and env.api_key_env not in target.env_secrets:
        raise ValueError(
            f"platform auth is on, so the agent needs a key to call the Inference Gateway: put a Helix access key "
            f"scoped to `inference` in env_secrets under {env.api_key_env}"
        )
    if async_sdk is None:
        raise ValueError("resolving a gateway model requires a platform client")
    models_client = client_from_platform(async_sdk, AsyncModelsClient)
    await HelixMetricModelResolver(models_client, authorize=authorize).resolve_model(target.gateway_model)


#: The OpenSandbox environment setting that lists the hosts a no-network sandbox may still reach.
AGENT_CHANNEL_HOSTS_KWARG = "agent_channel_hosts"


def sandbox_agent_channel(environment: HarborEnvironmentConfig | None) -> list[str]:
    """The hosts an OpenSandbox ``environment`` lets a no-network sandbox reach; none for another environment."""
    hosts = environment.kwargs.get(AGENT_CHANNEL_HOSTS_KWARG) if environment is not None else None
    return [host for host in hosts if isinstance(host, str)] if isinstance(hosts, list) else []


def agent_gateway_origin(*, sandboxed: bool, agent_channel: Sequence[str] = ()) -> str:
    """The platform origin an agent reaches from its task container.

    ``sandboxed`` is a Kubernetes task Job's OpenSandbox sandbox, which reaches the in-cluster API by the name its
    ``agent_channel`` allows; otherwise Harbor's Docker task container, which reaches a loopback platform through
    ``host.docker.internal``.
    """
    base_url = get_base_url()
    if not sandboxed:
        return resolve_agent_gateway_url(base_url, mode="docker")
    internal_base_url = get_internal_base_url()
    if internal_base_url is None and urlsplit(base_url).hostname not in LOOPBACK_ADDRESSES:
        internal_base_url = base_url
    origin = urlsplit(resolve_agent_gateway_url(base_url, mode="k8s", internal_base_url=internal_base_url))
    name = agent_channel_name(origin.hostname or "", agent_channel)
    if name is None or name == origin.hostname:
        return origin.geturl()
    netloc = name if origin.port is None else f"{name}:{origin.port}"
    return urlunsplit(origin._replace(netloc=netloc))


def point_agent_at_gateway_model(
    config: HarborRuntimeConfig, target: HarborRunnerTarget, origin: str
) -> HarborRuntimeConfig:
    """``config`` with the agent's base URL set to its gateway model's route, and its key read from ``env_secrets``.

    Without a key in the target's ``env_secrets``, the key is the placeholder :func:`add_gateway_key_placeholder` puts
    in the job's environment.
    """
    env = target.gateway_agent_env
    if target.gateway_model is None or env is None:
        return config
    workspace, name = parse_workspace_name_ref(target.gateway_model.root, label="Gateway model")
    route = model_entity_route_openai_url(base_url=origin, workspace=workspace, name=name).removesuffix("/v1")
    env_secrets = config.env_secrets
    if env.api_key_env not in env_secrets:
        env_secrets = {**env_secrets, env.api_key_env: SecretRef(GATEWAY_PLACEHOLDER_KEY)}
    return config.model_copy(
        update={"env_vars": {**config.env_vars, env.base_url_env: route + env.route_suffix}, "env_secrets": env_secrets}
    )


def add_gateway_key_placeholder(environment: list[EnvironmentVariable], target: object) -> list[EnvironmentVariable]:
    """``environment`` plus a gateway agent's placeholder key, when its target supplies no key of its own.

    Raises:
        SecretEnvError: The job's environment already sources the agent's key variable from a secret.
    """
    if not isinstance(target, HarborRunnerTarget) or target.gateway_model is None:
        return environment
    env = target.gateway_agent_env
    if env is None or env.api_key_env in target.env_secrets:
        return environment
    if any(variable.name == env.api_key_env for variable in environment):
        raise SecretEnvError(
            f"{env.api_key_env} is already sourced from a secret in this job, so the agent would be handed it to call "
            f"the Inference Gateway: put the agent's own key in the target's env_secrets under {env.api_key_env}, or "
            "use a secret under another name"
        )
    return [*environment, EnvironmentVariable(name=env.api_key_env, value=GATEWAY_PLACEHOLDER_KEY)]


def rebase_registered_agent_config(target: HarborRunnerTarget, agent_kwargs: dict[str, Any], origin: str) -> None:
    """Point a registered agent's gateway URLs, in ``agent_kwargs["fabric_config"]``, at ``origin``."""
    config = agent_kwargs.get("fabric_config")
    if isinstance(target.source, RegisteredAgentSource) and isinstance(config, dict):
        agent_kwargs["fabric_config"] = rewrite_fabric_config_base_urls(config, origin)


#: Harbor's built-in agents that call no model, so they have no model route to check.
MODEL_FREE_HARBOR_AGENTS = frozenset({"oracle", "nop"})


async def require_closed_book_route(
    target: object,
    tasks: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    async_sdk: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None,
) -> None:
    """Refuse no-network tasks unless the agent's model calls go through a closed-book VirtualModel.

    A no-network task's sandbox can still reach the model route, so the run is only closed-book if that route is.
    ``tasks`` are (task id, task.toml config) pairs. The VirtualModel must end its request middleware with the
    closed-book middleware, so the model it checks is the one the request is sent to.

    Raises:
        ValueError: Some task is no-network, and the agent's model route isn't a closed-book VirtualModel.
    """
    no_network = no_network_task_ids(tasks)
    if not no_network or (isinstance(target, HarborRunnerTarget) and target.agent_name in MODEL_FREE_HARBOR_AGENTS):
        return
    model = target.gateway_model if isinstance(target, HarborRunnerTarget) else None
    if model is None:
        raise ValueError(
            f"closed_book_required: no-network tasks {no_network} need the agent's model calls to go through a "
            f'closed-book VirtualModel: give a built-in agent one as `model: {{"ref": ...}}`'
        )
    if async_sdk is None:
        raise ValueError("checking a closed-book model route requires a platform client")
    workspace, name = parse_workspace_name_ref(model.root, label="Gateway model")
    await authorize_source_read(authorize, "virtual_model", workspace, name)
    try:
        virtual_model = (
            await client_from_platform(async_sdk, AsyncVirtualModelsClient).get_virtual_model(
                name=name, workspace=workspace
            )
        ).data()
    except NotFoundError:
        virtual_model = None
    if (
        virtual_model is None
        or not virtual_model.request_middleware
        or (virtual_model.request_middleware[-1].name != CLOSED_BOOK_MIDDLEWARE)
    ):
        raise ValueError(
            f"closed_book_required: no-network tasks {no_network} need a closed-book model route, and {model.root} "
            f"isn't one: make `{CLOSED_BOOK_MIDDLEWARE}` the last entry of its VirtualModel's request_middleware"
        )


def no_network_task_ids(tasks: Iterable[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """The ids of the tasks among (task id, task.toml config) pairs whose agent runs, in any step, with no network."""
    return [task_id for task_id, config in tasks if _agent_runs_without_network(TaskConfig.model_validate(config))]


def _agent_runs_without_network(task: TaskConfig) -> bool:
    baseline = task.environment.resolve_baseline()
    return any(
        resolve_agent_phase_policy(task, TrialAgentConfig(), baseline, step).network_mode == NetworkMode.NO_NETWORK
        for step in task.steps or [None]
    )


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
    no_network = no_network_task_ids(tasks)
    if not no_network:
        return
    host = urlsplit(agent_gateway_origin(sandboxed=True)).hostname or ""
    if agent_channel_name(host, agent_channel) is None:
        raise ValueError(
            f"gateway_unreachable: no-network tasks {no_network} can't reach the Inference Gateway at {host}: "
            f"add {host}, its fully qualified name, or an IP range containing it to the sandbox profile's "
            "agent_channel"
        )


def agent_channel_name(host: str, agent_channel: Sequence[str]) -> str | None:
    """The name an agent in a sandbox uses for ``host``: as its ``agent_channel`` allows it, else None.

    An entry allows the host when it is the host, an IP range containing it, or the host followed by more DNS
    labels, such as the fully qualified name of a Kubernetes service. The sandbox resolves only the names it
    allows, so the agent is given that one.
    """
    host = host.strip().rstrip(".").lower()
    entries = [entry.strip().rstrip(".").lower() for entry in agent_channel]
    for entry in entries:
        if entry == host:
            return host
        try:
            if ipaddress.ip_address(host) in ipaddress.ip_network(entry, strict=False):
                return host
        except ValueError:
            continue
    return next((entry for entry in entries if host and entry.startswith(f"{host}.")), None)
