# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point Harbor agents at the Inference Gateway: check a gateway model at submit, wire it in the job."""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

from harbor.models.task.config import NetworkMode, NetworkPolicy, TaskConfig
from harbor.models.trial.config import AgentConfig as TrialAgentConfig
from harbor.trial.network_policy import resolve_agent_phase_policy
from nemo_agents_plugin.config import AgentsConfig
from nemo_agents_plugin.runner.deployments_backend import resolve_agent_gateway_url, rewrite_fabric_config_base_urls
from nemo_agents_plugin.utils import get_base_url, get_internal_base_url
from nemo_evals.gateway.middleware.closed_book import MIDDLEWARE_NAME as CLOSED_BOOK_MIDDLEWARE
from nemo_evals.gateway.middleware.closed_book import ends_closed_book
from nemo_evals.gateway.middleware.trial_recording import MIDDLEWARE_NAME, TrialRecordingConfig
from nemo_evals.jobs.agent_spec import HarborRunnerTarget, RegisteredAgentSource, registered_agent_config
from nemo_evals.jobs.gateway_agents import GATEWAY_PLACEHOLDER_KEY
from nemo_evals.jobs.metric_resolution import HelixMetricModelResolver
from nemo_evals.jobs.secret_env import SecretEnvError
from nemo_evals.jobs.utils import run_with_isolated_async_client
from nemo_evals.source_authorization import SourceReadAuthorizer, authorize_source_read
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.client.adapter import AsyncHelixClient, client_from_platform
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError
from nemo_helix_plugin.client.tls import NHX_CLIENT_SSL_CERT_FILE_ENVVAR, client_certificate_authority_from_env
from nemo_helix_plugin.config import LOOPBACK_ADDRESSES
from nemo_helix_plugin.inference_middleware_models import VirtualModel
from nemo_helix_plugin.jobs.api_factory import EnvironmentVariable
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.refs import model_entity_route_openai_url, parse_workspace_name_ref
from nemo_helix_plugin.virtual_models.client import AsyncVirtualModelsClient, VirtualModelsClient
from nhx_evals_sdk.agent_eval.runtimes.harbor.runtime import HarborEnvironmentConfig, HarborRuntimeConfig
from nhx_evals_sdk.values import ModelRef
from nhx_evals_sdk.values.common import SecretRef

logger = logging.getLogger(__name__)


async def check_gateway_model(
    target: object,
    *,
    async_sdk: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None,
) -> None:
    """Refuse a Harbor target whose gateway model is missing, unreadable by the caller, or uncallable.

    Raises:
        ValueError: No model or VirtualModel has the name, or platform auth is on and the agent has no key for
            the gateway.
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


#: The OpenSandbox environment setting that lists the hosts a no-network sandbox may still reach.
AGENT_CHANNEL_HOSTS_KWARG = "agent_channel_hosts"
#: The OpenSandbox environment setting naming the CA bundle its sandbox trusts for the gateway.
GATEWAY_CA_FILE_KWARG = "gateway_ca_file"
#: Step variable carrying the agents plugin's ``deployments.gateway_url_override`` to a task Job.
GATEWAY_URL_OVERRIDE_ENVVAR = "NEMO_EVALS_GATEWAY_URL_OVERRIDE"


def sandbox_agent_channel(environment: HarborEnvironmentConfig | None) -> list[str]:
    """The hosts an OpenSandbox ``environment`` lets a no-network sandbox reach; none for another environment."""
    hosts = environment.kwargs.get(AGENT_CHANNEL_HOSTS_KWARG) if environment is not None else None
    return [host for host in hosts if isinstance(host, str)] if isinstance(hosts, list) else []


def agent_gateway_origin(*, sandboxed: bool, agent_channel: Sequence[str] = ()) -> str:
    """The platform origin an agent reaches from its task container.

    ``sandboxed`` is a Kubernetes task Job's OpenSandbox sandbox. It reaches the in-cluster API, or the agents
    plugin's ``deployments.gateway_url_override`` the Job was compiled with, by the name its ``agent_channel`` allows.
    Otherwise the task runs in Harbor's Docker task container, which reaches a loopback platform through
    ``host.docker.internal``.

    Raises:
        ValueError: The gateway is HTTPS and the sandbox would have to reach it by a name other than its own,
            which its certificate may not cover.
    """
    base_url = get_base_url()
    if not sandboxed:
        return resolve_agent_gateway_url(base_url, mode="docker")
    internal_base_url = get_internal_base_url()
    if internal_base_url is None and urlsplit(base_url).hostname not in LOOPBACK_ADDRESSES:
        internal_base_url = base_url
    origin = urlsplit(
        resolve_agent_gateway_url(
            base_url,
            mode="k8s",
            override=os.environ.get(GATEWAY_URL_OVERRIDE_ENVVAR) or None,
            internal_base_url=internal_base_url,
        )
    )
    name = agent_channel_name(origin.hostname or "", agent_channel)
    if name is None or name == origin.hostname:
        return origin.geturl()
    if origin.scheme == "https":
        raise ValueError(
            f"gateway_name_mismatch: sandboxes reach the gateway as {name}, but its certificate is for "
            f"{origin.hostname}: set deployments.gateway_url_override to https://{name}"
            f"{'' if origin.port is None else f':{origin.port}'}, with a certificate for that name"
        )
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


def trust_the_gateway_ca(config: HarborRuntimeConfig, origin: str) -> HarborRuntimeConfig:
    """``config`` whose sandbox trusts the CA this job trusts for the platform, when ``origin`` is HTTPS."""
    ca_file = client_certificate_authority_from_env(
        cert_file_envvars=(NHX_CLIENT_SSL_CERT_FILE_ENVVAR, "SSL_CERT_FILE")
    )
    if config.environment is None or urlsplit(origin).scheme != "https" or ca_file is None:
        return config
    kwargs = {**config.environment.kwargs, GATEWAY_CA_FILE_KWARG: ca_file}
    return config.model_copy(update={"environment": config.environment.model_copy(update={"kwargs": kwargs})})


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


def platform_gateway_url_override() -> str | None:
    """The agents plugin's ``deployments.gateway_url_override``, which a task Job can't read."""
    return AgentsConfig.get().deployments.gateway_url_override


def add_gateway_url_override(environment: list[EnvironmentVariable], override: str | None) -> list[EnvironmentVariable]:
    """``environment`` plus ``override``, for :func:`agent_gateway_origin` in the task Job."""
    if override is None:
        return environment
    return [*environment, EnvironmentVariable(name=GATEWAY_URL_OVERRIDE_ENVVAR, value=override)]


#: The gateway routes a Fabric model can name: a model's own route, or the OpenAI route with the model in the body.
_FABRIC_GATEWAY_ROUTE = re.compile(
    r"/apis/inference-gateway/v2/workspaces/(?P<workspace>[^/]+)/(?:model/(?P<name>[^/]+)/-|openai/-)(?:/|$)"
)
#: Where a registered agent reads its gateway key when its model names no ``api_key_env``.
FABRIC_GATEWAY_KEY_ENV = "NEMO_AGENTS_IGW_API_KEY"


def _fabric_base_url(model: Mapping[str, Any]) -> str:
    """The URL a Fabric model config calls, read as the runtime reads it."""
    settings = model.get("settings")
    url = model.get("base_url") or (settings.get("base_url") if isinstance(settings, Mapping) else None)
    return url if isinstance(url, str) else ""


def _fabric_gateway_model(model: object) -> ModelRef | None:
    """The gateway model one Fabric model config calls, or ``None`` if it doesn't call the Inference Gateway."""
    if not isinstance(model, Mapping):
        return None
    match = _FABRIC_GATEWAY_ROUTE.search(_fabric_base_url(model))
    name = (match["name"] or model.get("model")) if match is not None else None
    if not isinstance(name, str) or not name:
        return None
    assert match is not None
    return ModelRef(name if "/" in name else f"{match['workspace']}/{name}")


def fabric_gateway_route(config: Mapping[str, Any]) -> tuple[ModelRef, str] | None:
    """The gateway model a Fabric config's default model calls, and the variable it reads its key from.

    ``None`` when the default model doesn't call the Inference Gateway.
    """
    model = (config.get("models") or {}).get("default")
    ref = _fabric_gateway_model(model)
    if ref is None:
        return None
    assert isinstance(model, Mapping)
    return ref, str(model.get("api_key_env") or FABRIC_GATEWAY_KEY_ENV)


def fabric_gateway_key_envs(config: Mapping[str, Any]) -> list[str]:
    """The variables a Fabric config's gateway models read their key from, the default model's first, each once."""
    shared = config.get("models") or {}
    models = [shared.get("default"), *(model for name, model in shared.items() if name != "default")]
    models += [
        harness.get("model") for harness in (config.get("harnesses") or {}).values() if isinstance(harness, Mapping)
    ]
    envs = [
        str(model.get("api_key_env") or FABRIC_GATEWAY_KEY_ENV)
        for model in models
        if isinstance(model, Mapping) and _fabric_gateway_model(model) is not None
    ]
    return list(dict.fromkeys(envs))


def _fabric_models(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """A Fabric config's shared and harness model configs."""
    models = list((config.get("models") or {}).values())
    models += [
        harness.get("model") for harness in (config.get("harnesses") or {}).values() if isinstance(harness, Mapping)
    ]
    return [model for model in models if isinstance(model, Mapping)]


def fabric_gateway_models(config: Mapping[str, Any]) -> list[ModelRef]:
    """Every gateway model a Fabric config's shared and harness models call, each once.

    Raises:
        ValueError: A model's URL is a template, or calls the gateway on a route that names no virtual model.
    """
    refs: list[ModelRef] = []
    for model in _fabric_models(config):
        url = _fabric_base_url(model)
        if "$" in url:
            raise ValueError(f"a model's base_url {url!r} is a template, so where it calls can't be checked")
        ref = _fabric_gateway_model(model)
        if ref is None and urlsplit(url).path.startswith("/apis/inference-gateway/"):
            raise ValueError(f"a model calls the gateway at {urlsplit(url).path!r}, which names no virtual model")
        if ref is not None and ref not in refs:
            refs.append(ref)
    return refs


def fabric_calls_only_gateway(config: Mapping[str, Any]) -> bool:
    """Whether every shared and harness model of a Fabric config calls the Inference Gateway."""
    return all(_fabric_gateway_model(model) is not None for model in _fabric_models(config))


def rebase_registered_agent_config(target: HarborRunnerTarget, agent_kwargs: dict[str, Any], origin: str) -> None:
    """Point a registered agent's gateway URLs, in ``agent_kwargs["fabric_config"]``, at ``origin``."""
    config = agent_kwargs.get("fabric_config")
    if isinstance(target.source, RegisteredAgentSource) and isinstance(config, dict):
        agent_kwargs["fabric_config"] = rewrite_fabric_config_base_urls(config, origin)


#: Harbor's built-in agents that call no model, so they have no model route to check.
MODEL_FREE_HARBOR_AGENTS = frozenset({"oracle", "nop"})


def closed_book_run(target: object, tasks: Iterable[tuple[str, Mapping[str, Any]]]) -> bool:
    """Whether a run is closed-book: its target is labelled so, or some task's agent runs with no network."""
    if isinstance(target, HarborRunnerTarget) and target.agent_name in MODEL_FREE_HARBOR_AGENTS:
        return False
    return (isinstance(target, HarborRunnerTarget) and target.closed_book) or bool(no_network_task_ids(tasks))


async def require_closed_book_route(
    target: object,
    tasks: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    async_sdk: AsyncHelixClient | None,
    authorize: SourceReadAuthorizer | None,
) -> None:
    """Refuse a closed-book run unless every VirtualModel its agent can reach is closed-book capable.

    A run is closed-book when its target says so or a task's agent runs with no network: a no-network sandbox can
    still reach the model route. ``tasks`` are (task id, task.toml config) pairs. Each of the agent's gateway models,
    and each VirtualModel it routes to (its recording middleware's ``routes_to``), must end its request middleware with
    the closed-book middleware.

    Raises:
        ValueError: The run is closed-book, and some VirtualModel its agent can reach isn't closed-book capable, or its
            registered agent's default model doesn't call the gateway, or the target is labelled closed-book and its
            agent calls a model elsewhere.
    """
    if not closed_book_run(target, tasks):
        return
    why = _closed_book_reason(target, tasks)
    try:
        models = _gateway_models(target)
    except ValueError as exc:
        raise ValueError(f"closed_book_required: {why} need a closed-book model route: {exc}") from exc
    if (
        isinstance(target, HarborRunnerTarget)
        and isinstance(target.source, RegisteredAgentSource)
        and (config := registered_agent_config(target)) is not None
        and fabric_gateway_route(config) is None
    ):
        raise ValueError(
            f"closed_book_required: {why} need the agent's default model to call the Inference Gateway, which issues "
            "each trial its credential"
        )
    if closed_book_run(target, ()) and not calls_only_gateway_models(target):
        raise ValueError(
            f"closed_book_required: {why} need every model the agent calls to go through the Inference Gateway, and "
            "its config calls one elsewhere"
        )
    if not models:
        raise ValueError(
            f"closed_book_required: {why} need the agent's model calls to go through a closed-book VirtualModel: "
            f'give a built-in agent one as `model: {{"ref": ...}}`, or point a registered agent\'s models at one '
            "through the Inference Gateway"
        )
    if async_sdk is None:
        raise ValueError("checking a closed-book model route requires a platform client")
    checked: set[str] = set()
    pending = [model.root for model in models]
    while pending:
        ref = pending.pop(0)
        if ref in checked:
            continue
        checked.add(ref)
        pending.extend(await _require_closed_book_model(ModelRef(ref), why, async_sdk=async_sdk, authorize=authorize))


def _closed_book_reason(target: object, tasks: Iterable[tuple[str, Mapping[str, Any]]]) -> str:
    no_network = no_network_task_ids(tasks)
    return f"no-network tasks {no_network}" if no_network else "closed_book runs"


def calls_only_gateway_models(target: object) -> bool:
    """Whether every model a Harbor target's trials call goes through the Inference Gateway."""
    if not isinstance(target, HarborRunnerTarget) or not isinstance(target.source, RegisteredAgentSource):
        return True
    config = registered_agent_config(target)
    return config is None or fabric_calls_only_gateway(config)


def _gateway_models(target: object) -> list[ModelRef]:
    """The gateway models an agent's calls go to: a built-in agent's one, or every one a registered agent's config names."""
    if not isinstance(target, HarborRunnerTarget):
        return []
    if target.gateway_model is not None:
        return [target.gateway_model]
    if isinstance(target.source, RegisteredAgentSource) and (config := registered_agent_config(target)) is not None:
        return fabric_gateway_models(config)
    return []


async def _require_closed_book_model(
    model: ModelRef, why: str, *, async_sdk: AsyncHelixClient, authorize: SourceReadAuthorizer | None
) -> list[str]:
    """Refuse ``model`` unless it is closed-book capable; return the VirtualModels it routes to."""
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
    if virtual_model is None or not ends_closed_book(virtual_model.request_middleware):
        raise ValueError(
            f"closed_book_required: {why} need a closed-book model route, and {model.root} isn't one: make "
            f"`{CLOSED_BOOK_MIDDLEWARE}` the last entry of its VirtualModel's request_middleware"
        )
    return list(routes_to(virtual_model))


def gateway_model_id(model: ModelRef) -> str:
    """``workspace/name`` of the VirtualModel a gateway model reference names, as the gateway identifies it."""
    workspace, name = parse_workspace_name_ref(model.root, label="Gateway model")
    return f"{workspace}/{name}"


def no_network_task_ids(tasks: Iterable[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """The ids of the tasks among (task id, task.toml config) pairs whose agent runs, in any step, with no network."""
    return [
        task_id
        for task_id, config in tasks
        if any(policy.network_mode == NetworkMode.NO_NETWORK for policy in _agent_network_policies(config))
    ]


def _agent_network_policies(config: Mapping[str, Any]) -> list[NetworkPolicy]:
    """The network policies the agent runs under across a task's steps."""
    task = TaskConfig.model_validate(config)
    baseline = task.environment.resolve_baseline()
    return [resolve_agent_phase_policy(task, TrialAgentConfig(), baseline, step) for step in task.steps or [None]]


def check_tasks_reach_the_gateway(
    target: object, tasks: Iterable[tuple[str, Mapping[str, Any]]], agent_channel: Sequence[str]
) -> None:
    """Refuse no-network and allowlist tasks whose sandboxes couldn't reach the gateway model the agent calls.

    ``tasks`` are (task id, task.toml config) pairs. Such a sandbox reaches the profile's ``agent_channel``
    hosts and, in allowlist mode, the task's own ``allowed_hosts``.

    Raises:
        ValueError: Some no-network or allowlist task's sandbox can't reach the gateway.
    """
    if not isinstance(target, HarborRunnerTarget) or target.gateway_model is None:
        return
    restricted = {
        task_id: [policy for policy in _agent_network_policies(config) if policy.network_mode != NetworkMode.PUBLIC]
        for task_id, config in tasks
    }
    if not any(restricted.values()):
        return
    host = urlsplit(agent_gateway_origin(sandboxed=True)).hostname or ""
    if agent_channel_name(host, agent_channel) is not None:
        return
    # Without a channel entry the agent calls the gateway by ``host`` itself, so an allowlist must allow that name.
    unreachable = [
        task_id
        for task_id, policies in restricted.items()
        if any(
            policy.network_mode == NetworkMode.NO_NETWORK or agent_channel_name(host, policy.allowed_hosts) != host
            for policy in policies
        )
    ]
    if unreachable:
        raise ValueError(
            f"gateway_unreachable: no-network or allowlist tasks {unreachable} can't reach the Inference Gateway at "
            f"{host}: add {host}, its fully qualified name, or an IP range containing it to the sandbox profile's "
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


def trial_recording_gap(virtual_model: VirtualModel) -> str | None:
    """Why calls through ``virtual_model`` aren't fully recorded in their trials; ``None`` when they are."""
    if not any(call.name == MIDDLEWARE_NAME for call in virtual_model.response_middleware):
        return (
            f"it doesn't list {MIDDLEWARE_NAME} in response_middleware, so the run's gateway calls won't "
            "be recorded in their trials"
        )
    if not any(call.name == MIDDLEWARE_NAME for call in virtual_model.request_middleware):
        return (
            f"it lists {MIDDLEWARE_NAME} in response_middleware but not request_middleware, so each "
            "call's span starts when the gateway has the reply rather than when the call arrived"
        )
    return None


def routes_to(virtual_model: VirtualModel) -> tuple[str, ...]:
    """The VirtualModels ``virtual_model``'s router calls back, as its trial-recording middleware's config lists them."""
    refs: dict[str, None] = {}
    for call in (*virtual_model.request_middleware, *virtual_model.response_middleware):
        if call.name == MIDDLEWARE_NAME and call.config:
            refs.update(dict.fromkeys(TrialRecordingConfig.model_validate(call.config).routes_to))
    return tuple(refs)


class TrialRoute(NamedTuple):
    closed_book: bool
    records_calls: bool
    routes_to: tuple[str, ...] = ()


def check_trial_route(client: NemoClient | AsyncNemoClient, gateway_model: ModelRef) -> TrialRoute:
    """What the gateway model's route offers trials, warning when it won't record their calls.

    The route is closed-book capable when its VirtualModel ends its request middleware with the closed-book
    middleware, records calls when it lists the trial-recording middleware in its response middleware, and leads to
    the VirtualModels that middleware's ``routes_to`` lists. A VirtualModel that can't be read offers none of these.
    """
    workspace, name = parse_workspace_name_ref(gateway_model.root, label="Gateway model")

    async def get(async_client: AsyncNemoClient) -> VirtualModel:
        models = AsyncVirtualModelsClient.from_client(async_client)
        return (await models.get_virtual_model(workspace=workspace, name=name)).data()

    try:
        if isinstance(client, AsyncNemoClient):
            virtual_model = run_with_isolated_async_client(client, get)
        else:
            virtual_model = (
                VirtualModelsClient.from_client(client).get_virtual_model(workspace=workspace, name=name).data()
            )
    except NemoClientError as e:
        logger.warning(
            "Couldn't read virtual model %s to check that it records the run's gateway calls in their trials: %s",
            gateway_model.root,
            e,
        )
        return TrialRoute(closed_book=False, records_calls=False)
    if (gap := trial_recording_gap(virtual_model)) is not None:
        logger.warning("Virtual model %s: %s", gateway_model.root, gap)
    return TrialRoute(
        closed_book=ends_closed_book(virtual_model.request_middleware),
        records_calls=any(call.name == MIDDLEWARE_NAME for call in virtual_model.response_middleware),
        routes_to=routes_to(virtual_model),
    )
