# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeploymentsRunnerBackend — runs agents as containers via the nemo-deployments plugin.

Translates the :class:`~nemo_agents_plugin.runner.backend.RunnerBackend` interface into
nemo-deployments ``Deployment`` / ``DeploymentConfig`` entity operations. The deployments
controller reconciles those entities onto a configured executor (docker or k8s).

Long-running only (``restart_policy=Always``). Finite / run-to-completion belongs to
AgentRun (Razvan RFCs), not AgentDeployment.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml
from nemo_agents_plugin.config import AgentsConfig, DeploymentsRunnerConfig
from nemo_agents_plugin.entities import (
    AGENT_CONFIG_FILENAME,
    CONTAINER_DEPLOYMENT_MODES,
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    ComputeResources,
    DeploymentMode,
    DeploymentStatus,
    Endpoint,
)
from nemo_agents_plugin.fabric.gateway_credentials import platform_gateway_credential_env
from nemo_agents_plugin.runner.backend import DeploymentInfo, ExternalLog, LogLocation, RunnerBackend
from nemo_agents_plugin.runner.fabric_artifact_staging import (
    FabricArtifactStagingError,
    stage_fabric_ethos_config_files,
)
from nemo_agents_plugin.telemetry.intake_export import (
    configure_intake_atif_export,
    supports_intake_atif_export,
)
from nemo_agents_plugin.utils import get_base_url, get_internal_base_url
from nemo_deployments_plugin.auth_proxy import auth_proxy_port
from nemo_deployments_plugin.entities import (
    ConfigFile,
    Container,
    ContainerPort,
    Deployment,
    DeploymentConfig,
    EnvVar,
    HTTPGetAction,
    Probe,
    ResourceRequirements,
    SecretRef,
    VolumeMount,
    WorkloadIdentitySpec,
)
from nemo_platform_plugin.auth import AuthContext, platform_auth_enabled
from nemo_platform_plugin.auth.workload_identity import (
    get_workload_identity_token_audience,
    is_workload_identity_token_exchange_enabled,
)
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.config import LOOPBACK_ADDRESSES
from nemo_platform_plugin.entities.base import parse_qualified_name
from nemo_platform_plugin.entities.client import AsyncEntitiesClient
from nemo_platform_plugin.entity_client import NemoEntitiesClient, NemoEntityNotFoundError
from nemo_platform_plugin.files.client import AsyncFilesClient
from nemo_platform_plugin.sdk_provider import get_async_platform_sdk

logger = logging.getLogger(__name__)

_CONTAINER_NAME = "agent"
_HTTP_PORT_NAME = "http"
_PLUGIN_WHEELS_VOLUME = "plugin-wheels"
_PLUGIN_WHEELS_MOUNT = "/opt/nemo/plugin-wheels"
_NAT_CONFIG_ENV = "NAT_CONFIG_PATH"
_AGENT_CONFIG_PATH_ENV = "AGENT_CONFIG_PATH"
_FABRIC_SERVER_MODULE = "nemo_agents_plugin.fabric.server"
_AUTH_PROXY_IDENTITY = "agents"

# Env var names the backend generates on the agent container. A secret env var
# from the environment must not collide with these: docker would apply the
# secret value over the generated one, while k8s ignores the colliding secret
# (explicit ``env`` entries take precedence over the managed Secret's
# ``envFrom``), so the behavior would be inconsistent and surprising. Reject the
# collision at compile time instead. ``PYTHONPATH`` is only injected in the
# k8s plugin-wheels path, but is reserved unconditionally so the guard does not
# depend on the deployment mode.
_RESERVED_ENV_VAR_NAMES = frozenset(
    {
        "NMP_WORKSPACE",
        "NMP_AGENT_NAME",
        "NMP_BASE_URL",
        "PYTHONPATH",
        _AGENT_CONFIG_PATH_ENV,
        _NAT_CONFIG_ENV,
    }
)
_IMAGE_ENTRYPOINT_RESERVED_ENV_VAR_NAMES = _RESERVED_ENV_VAR_NAMES | {"PORT"}


# On delete, wait up to this long for the deployments controller to tear down the
# container and remove the Deployment entity before we drop the DeploymentConfig.
# Short per-reconcile wait: if the Deployment is still present, return False so
# the controller keeps AgentDeployment in ``deleting`` and retries next cycle
# instead of blocking the reconcile loop for tens of seconds. 5s gives docker
# SDK round-trips a bit more headroom than a 2s budget.
_DELETE_CONFIG_WAIT_S = 5.0
_DELETE_CONFIG_POLL_S = 0.5

# agents lifecycle status  <-  deployments-plugin status
_STATUS_MAP: dict[str, DeploymentStatus] = {
    "PENDING": "starting",
    "STARTING": "starting",
    "READY": "running",
    "SUCCEEDED": "failed",  # Always agents should not terminate successfully
    "FAILED": "failed",
    "LOST": "failed",
    "UNKNOWN": "starting",
    "DELETING": "deleting",
}


def map_status(backend_status: str) -> DeploymentStatus:
    """Return the agents lifecycle status for a deployments-plugin status."""
    return _STATUS_MAP.get(backend_status, "starting")


class UnreachableGatewayURLError(ValueError):
    """Raised when no inference base URL reachable from an agent container can be resolved."""


def resolve_agent_gateway_url(
    base_url: str,
    *,
    mode: DeploymentMode,
    override: str | None = None,
    internal_base_url: str | None = None,
) -> str:
    """Return the platform base URL an agent should call, reachable from its container.

    An explicit *override* wins for any mode. Otherwise ``k8s`` uses
    *internal_base_url* (the in-cluster API Service DNS) and ``docker`` rewrites a
    loopback *base_url* to ``host.docker.internal``, passing other hosts through.

    Only ``docker`` and ``k8s`` are supported; ``subprocess`` deployments are
    served by a different backend.

    Raises:
        UnreachableGatewayURLError: k8s mode with no *internal_base_url* or *override*.
        ValueError: *mode* is not a container deployment mode.
    """
    if mode not in CONTAINER_DEPLOYMENT_MODES:
        raise ValueError(
            f"resolve_agent_gateway_url only supports container deployment modes "
            f"{sorted(CONTAINER_DEPLOYMENT_MODES)}, got {mode!r}."
        )

    if override:
        return override.rstrip("/")

    if mode == "k8s":
        if internal_base_url:
            return internal_base_url.rstrip("/")
        raise UnreachableGatewayURLError(
            f"No container-reachable inference base URL for k8s deployment: platform base URL "
            f"{base_url!r} is not usable from an agent pod and no internal API Service URL is set. "
            "Set NEMO_INTERNAL_BASE_URL / NMP_INTERNAL_BASE_URL (or deployments.k8s_internal_base_url), "
            "or deployments.gateway_url_override."
        )

    parts = urlsplit(base_url.rstrip("/"))
    if (parts.hostname or "").lower() in LOOPBACK_ADDRESSES:
        netloc = "host.docker.internal"
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        return parts._replace(netloc=netloc).geturl()
    return parts.geturl()


def rewrite_config_base_urls(nat_config: dict[str, Any], gateway_url: str) -> dict[str, Any]:
    """Return a copy of *nat_config* with each Inference Gateway LLM base_url rebased onto *gateway_url*.

    Rewrites the scheme, host, and port of ``base_url`` on every ``openai``/``nim``
    LLM that points at the Inference Gateway, preserving the path. LLMs with an
    explicit third-party ``base_url`` are left unchanged.
    """
    reachable = urlsplit(gateway_url.rstrip("/"))
    reachable_origin = f"{reachable.scheme}://{reachable.netloc}"
    config = copy.deepcopy(nat_config)
    for llm_cfg in config.get("llms", {}).values():
        if not isinstance(llm_cfg, dict) or llm_cfg.get("_type") not in ("openai", "nim"):
            continue
        current = llm_cfg.get("base_url")
        if not isinstance(current, str) or "/apis/inference-gateway/" not in current:
            continue
        parts = urlsplit(current)
        llm_cfg["base_url"] = f"{reachable_origin}{parts.path}"
    return config


def rewrite_fabric_config_base_urls(agent_config: dict[str, Any], gateway_url: str) -> dict[str, Any]:
    """Return a copy of *agent_config* with Platform URLs rebased onto *gateway_url*.

    Rewrites ``models.*.base_url`` and harness ``model.base_url`` when the URL
    points at the Inference Gateway. Legacy ``settings.base_url`` values are
    also supported. Loopback ATIF HTTP storage endpoints are rewritten so they
    remain reachable from container deployments. Third-party URLs are left
    unchanged.
    """
    reachable = urlsplit(gateway_url.rstrip("/"))
    reachable_origin = f"{reachable.scheme}://{reachable.netloc}"
    config = copy.deepcopy(agent_config)
    model_configs: list[Any] = []

    models = config.get("models")
    if isinstance(models, dict):
        model_configs.extend(models.values())

    harnesses = config.get("harnesses")
    if isinstance(harnesses, dict):
        for harness in harnesses.values():
            if isinstance(harness, dict) and "model" in harness:
                model_configs.append(harness["model"])

    for model_config in model_configs:
        if not isinstance(model_config, dict):
            continue
        current = model_config.get("base_url")
        if isinstance(current, str) and "/apis/inference-gateway/" in current:
            parts = urlsplit(current)
            model_config["base_url"] = f"{reachable_origin}{parts.path}"
        settings = model_config.get("settings")
        if not isinstance(settings, dict):
            continue
        current = settings.get("base_url")
        if not isinstance(current, str) or "/apis/inference-gateway/" not in current:
            continue
        parts = urlsplit(current)
        settings["base_url"] = f"{reachable_origin}{parts.path}"

    telemetry = config.get("telemetry")
    atif = telemetry.get("atif") if isinstance(telemetry, dict) else None
    storage_configs = atif.get("storage", []) if isinstance(atif, dict) else []
    for storage_config in storage_configs:
        if not isinstance(storage_config, dict) or storage_config.get("type") != "http":
            continue
        endpoint = storage_config.get("endpoint")
        if not isinstance(endpoint, str):
            continue
        parts = urlsplit(endpoint)
        if parts.hostname not in LOOPBACK_ADDRESSES:
            continue
        scheme = "https" if "https" in (parts.scheme, reachable.scheme) else reachable.scheme
        storage_config["endpoint"] = parts._replace(scheme=scheme, netloc=reachable.netloc).geturl()
    return config


def _is_fabric_agent_config(agent_config: dict[str, Any]) -> bool:
    return agent_config.get("config_format") == NEMO_AGENTS_SPEC_CONFIG_FORMAT


def build_container_resources(resources: ComputeResources | None, *, mode: DeploymentMode) -> ResourceRequirements:
    """Compile a snapshotted compute spec into the container's k8s resources.

    k8s passes both requests and limits through. Docker has no notion of
    scheduling requests, so requests are consolidated into limits (limits win on
    key collision). Returns an empty ``ResourceRequirements`` when no compute
    spec was snapshotted (platform default).
    """
    if resources is None:
        return ResourceRequirements()
    if mode == "docker":
        consolidated = {**resources.requests, **resources.limits}
        return ResourceRequirements(limits=consolidated, requests={})
    return ResourceRequirements(limits=dict(resources.limits), requests=dict(resources.requests))


class ReservedSecretEnvVarError(ValueError):
    """A secret env var name collides with a platform-generated container env var."""


def _secret_env_vars(
    secrets: dict[str, str] | None,
    *,
    workspace: str,
    reserved_env_var_names: frozenset[str] = _RESERVED_ENV_VAR_NAMES,
) -> list[EnvVar]:
    """Compile resolved secret references into secret-backed container env vars.

    ``secrets`` maps ENV_VAR_NAME -> "workspace/secret-name" (an unqualified
    name resolves against the deployment ``workspace``). Each entry becomes an
    ``EnvVar`` carrying a ``secret_ref`` (never a plaintext value); the
    deployments-plugin substrate materializes the value at deploy time.

    Raises :class:`ReservedSecretEnvVarError` when a secret name collides with a
    platform-generated env var (see ``_RESERVED_ENV_VAR_NAMES``): such a
    collision behaves inconsistently across substrates, so it is rejected up
    front rather than silently shadowing platform wiring.
    """
    if not secrets:
        return []
    reserved = sorted(name for name in secrets if name in reserved_env_var_names)
    if reserved:
        raise ReservedSecretEnvVarError(
            "Environment secret variable name(s) collide with platform-reserved container env vars: "
            f"{', '.join(reserved)}. Rename the secret env var(s) to avoid "
            f"{', '.join(sorted(reserved_env_var_names))}."
        )
    env_vars: list[EnvVar] = []
    for env_name, ref in secrets.items():
        secret_workspace, secret_name = parse_qualified_name(ref, default_workspace=workspace)
        env_vars.append(EnvVar(name=env_name, secretRef=SecretRef(workspace=secret_workspace, name=secret_name)))
    return env_vars


def _fabric_config_mount_path(config_mount_path: str) -> str:
    parent = str(PurePosixPath(config_mount_path).parent)
    if parent in ("", "."):
        return f"/{AGENT_CONFIG_FILENAME}"
    return f"{parent}/{AGENT_CONFIG_FILENAME}"


def _fabric_server_cli_args(*, config_path: str, port: int) -> list[str]:
    # If this launch path forwards an idle-timeout override, the gateway must
    # use the same deployment-sourced value when computing ``expires_at``.
    return [
        "-m",
        _FABRIC_SERVER_MODULE,
        "--agent-config",
        config_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]


def executor_for_mode(config: DeploymentsRunnerConfig, mode: DeploymentMode) -> str | None:
    """Resolve the named deployments-plugin executor for *mode*."""
    if mode == "docker":
        return config.docker_executor or config.default_executor
    if mode == "k8s":
        return config.k8s_executor or config.default_executor
    return config.default_executor


def executor_backend(name: str | None) -> str | None:
    """Return the backend key the deployments plugin would run *name* on.

    An unset name is not "no executor": ``ExecutorRegistry.resolve`` falls back to
    the deployments plugin's own ``default_executor``, so resolving it here is what
    makes the mode check see the executor that will actually run.
    """
    from nemo_deployments_plugin.config import DeploymentsConfig

    config = DeploymentsConfig.get()
    resolved = name or config.default_executor
    if not resolved:
        return None
    for entry in config.executors:
        if entry.name == resolved:
            return entry.backend
    return None


def require_executor_matches_mode(executor: str | None, mode: DeploymentMode) -> None:
    """Refuse a container mode that would run somewhere other than it names.

    ``executor_for_mode`` falls back to ``default_executor``, so asking for k8s
    where none is configured silently lands on docker: the deployment reports
    running while nothing exists in the cluster, and a k8s-only failure cannot
    reproduce. The backend keys and the deployment modes share a vocabulary, so
    the mismatch is checkable here rather than discoverable by counting pods.
    """
    backend = executor_backend(executor)
    if backend is None or backend == mode:
        return
    alternative = f", or deploy with deployment_mode {backend!r}" if backend in CONTAINER_DEPLOYMENT_MODES else ""
    raise ValueError(
        f"deployment_mode {mode!r} resolved to executor {executor!r}, which runs on "
        f"{backend!r}. Set 'deployments.{mode}_executor' to an executor whose backend "
        f"is {mode!r}{alternative}."
    )


_HTTP_PROTOCOLS = frozenset({"http", "https"})


def _project_endpoints(deployment: Deployment) -> list[Endpoint]:
    """Map deployments-plugin endpoints onto the agents entity Endpoint model."""
    return [Endpoint.model_validate(ep.model_dump()) for ep in deployment.endpoints]


def _primary_http_url(endpoints: list[Endpoint]) -> str:
    return next((ep.url for ep in endpoints if ep.protocol in _HTTP_PROTOCOLS and ep.url), "")


def _info_from_deployment(deployment: Deployment) -> DeploymentInfo:
    endpoints = _project_endpoints(deployment)
    info = DeploymentInfo(
        name=deployment.name,
        status=map_status(deployment.status),
        endpoint=_primary_http_url(endpoints),
        endpoints=endpoints,
    )
    if info.status == "failed":
        info.error = deployment.status_message or "Deployment failed."
    return info


def build_deployment_config(
    *,
    name: str,
    workspace: str,
    image: str,
    port: int,
    agent_config: dict[str, Any],
    platform_base_url: str,
    config_mount_path: str,
    mode: DeploymentMode,
    plugin_wheels_init_image: str | None = None,
    labels: dict[str, str] | None = None,
    auth_proxy_identity: str | None = None,
    auth_proxy_on_behalf_of: str | None = None,
    config_files: list[ConfigFile] | None = None,
    resources: ComputeResources | None = None,
    secrets: dict[str, str] | None = None,
    use_image_entrypoint: bool = False,
    workload_identity_enabled: bool = False,
) -> DeploymentConfig:
    """Compile an agent into a long-running ``DeploymentConfig`` (Always).

    *agent_config* is either a NAT workflow config or a Platform-owned Fabric spec.
    NAT workflow configs start ``nat start fastapi`` with workflow YAML at
    *config_mount_path*. Fabric configs start
    ``python -m nemo_agents_plugin.fabric.server`` with ``agent.yaml`` beside the
    NAT config directory. When ``use_image_entrypoint`` is true, the generated
    DeploymentConfig leaves command/args empty so the image ENTRYPOINT/CMD runs
    instead. Docker mode materializes config from env because the docker backend
    ignores ``config_files``; k8s mounts ``config_files`` via ConfigMap subPath.
    The main container binds ``0.0.0.0`` and exposes a readiness probe on
    ``/health``.

    The caller is responsible for rebasing inference ``base_url`` values in
    *agent_config* to a container-reachable gateway before calling this helper.

    ``platform_base_url`` is the container-reachable platform origin used to
    build the Inference Gateway URL. It is also exported as ``NMP_BASE_URL`` so
    SDK calls from inside the agent use the same platform instead of falling
    back to a baked or host CLI context.
    """
    is_fabric = _is_fabric_agent_config(agent_config)
    config_yaml = yaml.safe_dump(agent_config, sort_keys=False)
    config_path = _fabric_config_mount_path(config_mount_path) if is_fabric else config_mount_path
    resolved_config_files = config_files or [ConfigFile(path=config_path, content=config_yaml)]
    env = [
        EnvVar(name="NMP_WORKSPACE", value=workspace),
        EnvVar(name="NMP_AGENT_NAME", value=name),
        EnvVar(name="NMP_BASE_URL", value=platform_base_url.rstrip("/")),
    ]
    if is_fabric:
        env.append(EnvVar(name=_AGENT_CONFIG_PATH_ENV, value=config_path))
        env.extend(
            EnvVar(name=env_name, value=env_value)
            for env_name, env_value in platform_gateway_credential_env(agent_config).items()
        )
    else:
        env.append(EnvVar(name=_NAT_CONFIG_ENV, value=config_mount_path))
    # Secret-backed env vars from the resolved environment: emitted as
    # secret_ref (never plaintext). The deployments-plugin substrate resolves
    # them (docker) or mounts a managed Secret via envFrom (k8s).
    reserved_env_var_names = (
        _IMAGE_ENTRYPOINT_RESERVED_ENV_VAR_NAMES if use_image_entrypoint else _RESERVED_ENV_VAR_NAMES
    )
    env.extend(_secret_env_vars(secrets, workspace=workspace, reserved_env_var_names=reserved_env_var_names))
    volume_mounts: list[VolumeMount] = []
    init_containers: list[Container] = []

    # K8s only: init container stages workspace plugin wheels into a shared volume.
    # Docker backend rejects init_containers in v1. Full wheel-source contract is AIRCORE-863.
    # Constructors use camelCase aliases (ty + pydantic alias validation).
    if mode == "k8s" and plugin_wheels_init_image:
        volume_mounts.append(VolumeMount(name=_PLUGIN_WHEELS_VOLUME, mountPath=_PLUGIN_WHEELS_MOUNT, readOnly=True))
        init_containers.append(
            Container(
                name="plugin-wheels",
                image=plugin_wheels_init_image,
                command=["sh", "-c"],
                args=[
                    f"echo 'plugin-wheels init stub; hardened in AIRCORE-863' "
                    f"&& mkdir -p {_PLUGIN_WHEELS_MOUNT} && touch {_PLUGIN_WHEELS_MOUNT}/.ready"
                ],
            ).model_copy(
                update={
                    "volume_mounts": [
                        VolumeMount(name=_PLUGIN_WHEELS_VOLUME, mountPath=_PLUGIN_WHEELS_MOUNT, readOnly=False)
                    ]
                }
            )
        )
        env.append(EnvVar(name="PYTHONPATH", value=_PLUGIN_WHEELS_MOUNT))

    if use_image_entrypoint:
        server_command = []
        server_args = []
        env.append(EnvVar(name="PORT", value=str(port)))
    elif is_fabric:
        server_command = ["python"]
        server_args = _fabric_server_cli_args(config_path=config_path, port=port)
    else:
        server_command = ["nat", "start", "fastapi"]
        server_args = [
            "--config_file",
            config_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]

    command = server_command
    args = server_args

    container = Container(
        name=_CONTAINER_NAME,
        image=image,
        command=command,
        args=args,
        ports=[ContainerPort(name=_HTTP_PORT_NAME, containerPort=port)],
        env=env,
    ).model_copy(
        update={
            "volume_mounts": volume_mounts,
            "resources": build_container_resources(resources, mode=mode),
            "readiness_probe": Probe(
                httpGet=HTTPGetAction(path="/health", port=port),
                initialDelaySeconds=2,
                periodSeconds=5,
                failureThreshold=12,
            ),
        }
    )

    workload_identity = (
        WorkloadIdentitySpec(
            enabled=True,
            workloadKind="agent_deployment",
            workloadId=name,
            tokenAudience=get_workload_identity_token_audience(),
        )
        if workload_identity_enabled
        else None
    )

    # Request the auth-proxy sidecar via the DeploymentConfig flags; the
    # deployments plugin compiles and injects it (and no-ops when auth is off).
    return DeploymentConfig(
        name=name,
        workspace=workspace,
        containers=[container],
        labels=labels or {},
    ).model_copy(
        update={
            "init_containers": init_containers,
            "config_files": resolved_config_files,
            "restart_policy": "Always",
            "auth_proxy_sidecar": auth_proxy_identity is not None,
            "auth_proxy_sidecar_identity": auth_proxy_identity,
            "auth_proxy_sidecar_on_behalf_of": auth_proxy_on_behalf_of,
            "workload_identity": workload_identity,
        }
    )


class DeploymentsRunnerBackend(RunnerBackend):
    """Runs agent deployments as containers via the nemo-deployments plugin."""

    def __init__(self, config: AgentsConfig) -> None:
        self._config: DeploymentsRunnerConfig = config.deployments
        self._entities: NemoEntitiesClient | None = None

    def _entity_client(self) -> NemoEntitiesClient:
        if self._entities is None:
            sdk = get_async_platform_sdk(as_service="agents", internal=True)
            self._entities = NemoEntitiesClient(client_from_platform(sdk, AsyncEntitiesClient))
        return self._entities

    async def create_deployment(
        self,
        workspace: str,
        name: str,
        config: dict[str, Any],
        port: int,
        *,
        agent: str = "",
        image: str | None = None,
        deployment_mode: DeploymentMode = "docker",
        created_by: str | None = None,
        auth_context: AuthContext | None = None,
        resources: ComputeResources | None = None,
        secrets: dict[str, str] | None = None,
        use_image_entrypoint: bool = False,
    ) -> DeploymentInfo:
        """Create DeploymentConfig + Deployment entities for the agent container."""
        del port  # Host port is allocated by the deployments executor, not agents.
        if deployment_mode not in CONTAINER_DEPLOYMENT_MODES:
            return DeploymentInfo(
                name=name,
                status="failed",
                error=f"DeploymentsRunnerBackend does not support deployment_mode={deployment_mode!r}.",
            )

        try:
            require_executor_matches_mode(executor_for_mode(self._config, deployment_mode), deployment_mode)
        except ValueError as exc:
            logger.error("Refusing to deploy agent %r: %s", name, exc)
            return DeploymentInfo(name=name, status="failed", error=str(exc))

        resolved_image = image or self._config.default_image
        if not resolved_image:
            return DeploymentInfo(
                name=name,
                status="failed",
                error="No container image provided and no deployments.default_image configured.",
            )

        entities = self._entity_client()
        # Deployment resolution injects the platform's own base URL, which is not
        # necessarily reachable from inside the agent container. Rebase it onto a
        # container-reachable address.
        try:
            internal_base_url = self._config.k8s_internal_base_url or get_internal_base_url()
            gateway = resolve_agent_gateway_url(
                get_base_url(),
                mode=deployment_mode,
                override=self._config.gateway_url_override,
                internal_base_url=internal_base_url,
            )
        except UnreachableGatewayURLError as exc:
            logger.error("Refusing to deploy agent %r: %s", name, exc)
            return DeploymentInfo(name=name, status="failed", error=str(exc))

        # When platform auth is enabled, the agent carries no platform credential,
        # so route its inference calls through a loopback auth-proxy sidecar (the
        # deployments plugin compiles the sidecar from the auth_proxy flags). The
        # agent targets the sidecar on localhost; the sidecar forwards to the
        # platform with a service-principal identity header.
        #
        # The sidecar also delegates to the deployment's creator via on-behalf-of
        # (when known) so the running agent's platform access is scoped to what the
        # creator can reach — the workspace(s) they have access to — rather than the
        # agents service principal's full (ServiceSystem) reach.
        auth_proxy_identity: str | None = None
        auth_proxy_on_behalf_of: str | None = None
        is_fabric = _is_fabric_agent_config(config)
        if platform_auth_enabled():
            auth_proxy_identity = _AUTH_PROXY_IDENTITY
            auth_proxy_on_behalf_of = created_by or None
            if not auth_proxy_on_behalf_of:
                logger.warning(
                    "Deployment %r has no creator principal; the agent will run as the "
                    "unscoped %s service principal without on-behalf-of delegation.",
                    name,
                    _AUTH_PROXY_IDENTITY,
                )
            rewrite_target = f"http://127.0.0.1:{auth_proxy_port()}"
        else:
            rewrite_target = gateway

        if is_fabric:
            # Wire the trajectory export before the rebase below, so the
            # endpoint it produces is rewritten for reachability alongside the
            # inference URLs. No header_env here: the auth-proxy sidecar stamps
            # identity on the way out, which is why the workload needs none.
            # config is the caller's deployment entity; rewrite_fabric_config_base_urls
            # deep-copies for the same reason, and wiring runs before it.
            if supports_intake_atif_export(config):
                config = copy.deepcopy(config)
                configure_intake_atif_export(config, workspace=workspace, base_url=rewrite_target)
            config = rewrite_fabric_config_base_urls(config, rewrite_target)
        else:
            config = rewrite_config_base_urls(config, rewrite_target)

        deployment_labels = {
            "nemo.agents/deployment": name,
            "nemo.agents/mode": deployment_mode,
        }
        if is_fabric:
            deployment_labels["nemo.agents/runtime"] = "fabric"

        staged_config_files: list[ConfigFile] | None = None
        if is_fabric and agent:
            agent_yaml_path = _fabric_config_mount_path(self._config.config_mount_path)
            try:
                sdk = get_async_platform_sdk(as_service="agents", internal=True)
                staged_config_files = await stage_fabric_ethos_config_files(
                    workspace=workspace,
                    agent_name=agent,
                    rewritten_agent_config=config,
                    agent_yaml_path=agent_yaml_path,
                    files_client=client_from_platform(sdk, AsyncFilesClient),
                )
            except FabricArtifactStagingError as exc:
                logger.error("Refusing to deploy Fabric agent %r: %s", name, exc)
                return DeploymentInfo(name=name, status="failed", error=str(exc))

        try:
            deployment_config = build_deployment_config(
                name=name,
                workspace=workspace,
                image=resolved_image,
                port=self._config.container_port,
                agent_config=config,
                platform_base_url=gateway,
                config_mount_path=self._config.config_mount_path,
                mode=deployment_mode,
                plugin_wheels_init_image=self._config.plugin_wheels_init_image,
                labels=deployment_labels,
                auth_proxy_identity=auth_proxy_identity,
                auth_proxy_on_behalf_of=auth_proxy_on_behalf_of,
                config_files=staged_config_files,
                resources=resources,
                secrets=secrets,
                use_image_entrypoint=use_image_entrypoint,
                workload_identity_enabled=auth_context is not None and is_workload_identity_token_exchange_enabled(),
            )
        except ReservedSecretEnvVarError as exc:
            logger.error("Refusing to deploy agent %r: %s", name, exc)
            return DeploymentInfo(name=name, status="failed", error=str(exc))
        await entities.create(deployment_config)
        try:
            deployment = Deployment(
                name=name,
                workspace=workspace,
                deployment_config=name,
                executor=executor_for_mode(self._config, deployment_mode),
                desired_state="READY",
                status="PENDING",
            ).with_auth_context(auth_context)
            await entities.create(deployment)
        except Exception:
            # Avoid orphaning the config if Deployment create fails.
            try:
                await entities.delete(
                    DeploymentConfig,
                    name=name,
                    workspace=workspace,
                    expected_db_version=deployment_config.db_version,
                )
            except Exception:
                logger.exception(
                    "Failed to clean up DeploymentConfig '%s/%s' after Deployment create failure",
                    workspace,
                    name,
                )
            raise

        logger.info(
            "Created deployment entities '%s/%s' (image=%s, mode=%s)",
            workspace,
            name,
            resolved_image,
            deployment_mode,
        )
        return DeploymentInfo(name=name, status="starting", endpoint="", endpoints=[])

    async def get_deployment_status(self, workspace: str, name: str) -> DeploymentInfo | None:
        entities = self._entity_client()
        try:
            deployment = await entities.get(Deployment, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        return _info_from_deployment(deployment)

    async def delete_deployment(self, workspace: str, name: str) -> bool:
        """Tear down plugin Deployment (+ Config).

        Returns ``True`` when the Deployment is gone and DeploymentConfig has
        been deleted (or was already absent) — the agents controller may then
        remove ``AgentDeployment``. Returns ``False`` when the Deployment is
        still present so a later reconcile can finish cleanup.
        """
        entities = self._entity_client()
        deployment_present = False
        try:
            deployment = await entities.get(Deployment, name=name, workspace=workspace)
            if deployment.status != "DELETING":
                deployment.status = "DELETING"
                deployment.desired_state = "STOPPED"
                await entities.update(deployment)
            deployment_present = True
        except NemoEntityNotFoundError:
            pass

        if deployment_present:
            gone = await self._wait_for_deployment_gone(workspace, name)
            if not gone:
                logger.warning(
                    "Deployment '%s/%s' still present after %.1fs; keeping AgentDeployment "
                    "and DeploymentConfig so teardown can finish on a later reconcile.",
                    workspace,
                    name,
                    _DELETE_CONFIG_WAIT_S,
                )
                return False

        try:
            deployment_config = await entities.get(DeploymentConfig, name=name, workspace=workspace)
            await entities.delete(
                DeploymentConfig,
                name=name,
                workspace=workspace,
                expected_db_version=deployment_config.db_version,
            )
        except NemoEntityNotFoundError:
            pass
        return True

    async def _wait_for_deployment_gone(self, workspace: str, dep_name: str) -> bool:
        """Return True if the Deployment entity disappears within the wait budget."""
        entities = self._entity_client()
        deadline = time.monotonic() + _DELETE_CONFIG_WAIT_S
        while time.monotonic() < deadline:
            try:
                await entities.get(Deployment, name=dep_name, workspace=workspace)
            except NemoEntityNotFoundError:
                return True
            await asyncio.sleep(_DELETE_CONFIG_POLL_S)
        return False

    async def list_deployments(self, workspace: str | None = None) -> list[DeploymentInfo]:
        entities = self._entity_client()
        result = await entities.list(Deployment, workspace=workspace or "-")
        return [_info_from_deployment(deployment) for deployment in result.data]

    async def health_check(self, endpoint: str) -> bool:
        # Container modes trust the deployments-plugin readiness projection; the
        # agents controller should not call this for docker/k8s. Kept for ABC parity.
        del endpoint
        return False

    def get_log_location(self, workspace: str, name: str) -> LogLocation:
        del workspace, name
        return ExternalLog(hint="Inspect container logs via the deployments plugin or substrate CLI.")

    async def shutdown(self) -> None:
        self._entities = None
