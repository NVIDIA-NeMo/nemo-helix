# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile DeploymentConfig + K8sDeploymentConfig into Kubernetes PodSpec objects.

Native sidecars require Kubernetes >= 1.29 (init container ``restartPolicy=Always``).
On older clusters, omit per-container restart policy on init containers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from kubernetes.client.models import V1EnvVar, V1Volume
from kubernetes.client.rest import ApiException
from nemo_deployments_plugin.auth_proxy import build_auth_proxy_container
from nemo_deployments_plugin.backends.k8s.client import k8s_client_module
from nemo_deployments_plugin.backends.k8s.status import resource_labels_match
from nemo_deployments_plugin.backends.k8s.workload_identity import service_account_name as workload_service_account_name
from nemo_deployments_plugin.backends.labels import (
    k8s_deployment_configmap_name,
    k8s_deployment_secret_name,
    k8s_volume_resource_name,
)
from nemo_deployments_plugin.constants import (
    DEFAULT_JOB_TTL_SECONDS_AFTER_FINISHED,
)
from nemo_deployments_plugin.entities import (
    Affinity,
    ConfigFile,
    Container,
    ContainerPort,
    DeploymentConfig,
    K8sDeploymentConfig,
    PodSecurityContext,
    Probe,
    Toleration,
    VolumeMount,
)
from nemo_deployments_plugin.types import RestartPolicy
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    WORKLOAD_IDENTITY_VOLUME_NAME,
    WORKLOAD_IDENTITY_VOLUME_PATH,
    get_workload_identity_token_audience,
    workload_identity_env,
)
from nemo_helix_plugin.config import ImagePullSecret, get_platform_config

CONFIG_FILES_VOLUME = "config-files"
NATIVE_SIDECAR_RESTART_POLICY: RestartPolicy = "Always"

logger = logging.getLogger(__name__)


class DeploymentConfigError(ValueError):
    """Invalid deployment config for k8s workload compilation."""


def merged_volume_mounts(config: DeploymentConfig, container: Container) -> list[VolumeMount]:
    mounts_by_name: dict[str, VolumeMount] = {}
    for mount in config.volume_mounts:
        mounts_by_name[mount.name] = mount
    for mount in container.volume_mounts:
        mounts_by_name[mount.name] = mount
    return list(mounts_by_name.values())


def build_env_vars(container: Container, *, extra_env: dict[str, str] | None = None) -> list[V1EnvVar]:
    """Build plaintext ``V1EnvVar`` entries for a container.

    Env vars carrying a ``secret_ref`` are intentionally skipped here — their
    values live in the per-deployment managed ``Secret`` and are injected via
    ``envFrom`` (see :func:`build_secret_env_from`), so the plaintext never
    appears in the pod manifest.
    """
    k8s = k8s_client_module()
    env: list[V1EnvVar] = [
        k8s.client.V1EnvVar(name=item.name, value=item.value) for item in container.env if item.value is not None
    ]
    for name, value in (extra_env or {}).items():
        env.append(k8s.client.V1EnvVar(name=name, value=value))
    return env


def build_secret_env_from(secret_name: str | None) -> list[Any]:
    """Build the ``envFrom`` projection for a deployment's managed Secret."""
    if secret_name is None:
        return []
    k8s = k8s_client_module()
    return [k8s.client.V1EnvFromSource(secret_ref=k8s.client.V1SecretEnvSource(name=secret_name))]


def build_resource_requirements(container: Container) -> Any | None:
    limits = container.resources.limits or None
    requests = container.resources.requests or None
    if not limits and not requests:
        return None
    k8s = k8s_client_module()
    return k8s.client.V1ResourceRequirements(limits=limits, requests=requests)


def build_pod_volumes(*, workspace: str, mounts: list[VolumeMount]) -> list[Any]:
    if not mounts:
        return []
    k8s = k8s_client_module()
    return [
        k8s.client.V1Volume(
            name=mount.name,
            persistent_volume_claim=k8s.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=k8s_volume_resource_name(workspace, mount.name),
            ),
        )
        for mount in mounts
    ]


def build_volume_mounts(mounts: list[VolumeMount]) -> list[Any]:
    if not mounts:
        return []
    k8s = k8s_client_module()
    return [
        k8s.client.V1VolumeMount(
            name=mount.name,
            mount_path=mount.mount_path,
            read_only=mount.read_only,
            sub_path=mount.sub_path,
        )
        for mount in mounts
    ]


def build_container_spec(
    container: Container,
    *,
    volume_mounts: list[VolumeMount] | None = None,
    secret_name: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> Any:
    k8s = k8s_client_module()
    kwargs: dict[str, Any] = {
        "name": container.name,
        "image": container.image,
        "env": build_env_vars(container, extra_env=extra_env) or None,
        "env_from": build_secret_env_from(secret_name) or None,
        "resources": build_resource_requirements(container),
    }
    if container.command:
        kwargs["command"] = list(container.command)
    if container.args:
        kwargs["args"] = list(container.args)
    if volume_mounts:
        kwargs["volume_mounts"] = build_volume_mounts(volume_mounts)
    return k8s.client.V1Container(**kwargs)


@dataclass(frozen=True)
class ExecutorK8sDefaults:
    """Executor-level k8s defaults applied to every workload the executor renders.

    These are the base layer shared by ALL deployments-plugin consumers (models,
    agents, ...). A per-entity ``K8sDeploymentConfig`` value overrides the default:
    annotations merge key-wise (per-entity key wins); node_selector / tolerations /
    affinity / topology_spread_constraints are applied only when the entity leaves
    them unset (per-entity wins wholesale).
    """

    pod_annotations: dict[str, str] = field(default_factory=dict)
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict[str, Any]] = field(default_factory=list)
    affinity: dict[str, Any] = field(default_factory=dict)
    topology_spread_constraints: list[dict[str, Any]] = field(default_factory=list)
    job_ttl_seconds_after_finished: int | None = DEFAULT_JOB_TTL_SECONDS_AFTER_FINISHED


@dataclass(frozen=True)
class CompiledWorkload:
    """Kubernetes objects derived from a DeploymentConfig."""

    pod_spec_kwargs: dict[str, Any]
    configmap_body: Any | None
    configmap_name: str | None
    service_containers: tuple[Container, ...]
    secret_body: Any | None = None
    secret_name: str | None = None
    pod_annotations: dict[str, str] = field(default_factory=dict)


def _reraise_api_unless(exc: ApiException, *allowed_statuses: int) -> None:
    if exc.status not in allowed_statuses:
        raise exc


def _validate_port_names(config: DeploymentConfig) -> None:
    seen_names: set[str] = set()
    seen_ports: set[tuple[int, str]] = set()
    for container in (*config.init_containers, *config.containers):
        for port in container.ports:
            port_name = port.name or f"port-{port.container_port}"
            if port_name in seen_names:
                raise DeploymentConfigError(f"duplicate container port name {port_name!r}")
            seen_names.add(port_name)
            port_key = (port.container_port, port.protocol)
            if port_key in seen_ports:
                raise DeploymentConfigError(
                    f"duplicate container port {port.container_port}/{port.protocol} across containers"
                )
            seen_ports.add(port_key)


def _validate_workload_identity_volume_mount_names(config: DeploymentConfig) -> None:
    if not workload_identity_enabled(config):
        return
    for mount in _collect_pvc_mounts(config):
        if mount.name == WORKLOAD_IDENTITY_VOLUME_NAME:
            raise DeploymentConfigError(
                f"volume mount name {WORKLOAD_IDENTITY_VOLUME_NAME!r} is reserved for workload identity"
            )
        if mount.mount_path == WORKLOAD_IDENTITY_VOLUME_PATH:
            raise DeploymentConfigError(
                f"volume mount path {WORKLOAD_IDENTITY_VOLUME_PATH!r} is reserved for workload identity"
            )


def validate_workload_config(config: DeploymentConfig) -> None:
    """Validate container lists shared by Job and Deployment backends."""
    if not config.containers:
        raise DeploymentConfigError("at least one container is required")
    _validate_port_names(config)
    _validate_workload_identity_volume_mount_names(config)
    for container in config.containers:
        if container.restart_policy is not None:
            raise DeploymentConfigError(
                f"container {container.name} sets restart_policy; only init_containers may use per-container restart_policy"
            )
    for init_container in config.init_containers:
        if init_container.restart_policy not in (None, NATIVE_SIDECAR_RESTART_POLICY):
            raise DeploymentConfigError(
                f"init container {init_container.name} has unsupported restart_policy "
                f"{init_container.restart_policy!r}; only Always (native sidecar) is supported"
            )


def validate_config_for_job(config: DeploymentConfig) -> None:
    validate_workload_config(config)
    if config.restart_policy == "Always":
        raise DeploymentConfigError("restart_policy Always requires a Deployment workload, not a Job")


def validate_config_for_deployment(config: DeploymentConfig) -> None:
    validate_workload_config(config)
    if config.restart_policy != "Always":
        raise DeploymentConfigError("restart_policy Always is required for Deployment + Service")


def configmap_data_key(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    key = normalized.lstrip("/").replace("/", "__")
    return key or "config"


def _deserialize_k8s(data: dict[str, Any], klass: str) -> Any:
    k8s = k8s_client_module()
    response = SimpleNamespace(data=json.dumps(data))
    return k8s.client.ApiClient().deserialize(response=response, response_type=klass)


def _build_probe(probe: Probe | None) -> Any | None:
    if probe is None:
        return None
    k8s = k8s_client_module()
    kwargs: dict[str, Any] = {
        "initial_delay_seconds": probe.initial_delay_seconds,
        "period_seconds": probe.period_seconds,
        "timeout_seconds": probe.timeout_seconds,
        "failure_threshold": probe.failure_threshold,
    }
    if probe.exec_action is not None:
        # V1Probe uses `_exec` (Python keyword `exec` cannot be a kwarg name).
        kwargs["_exec"] = k8s.client.V1ExecAction(command=list(probe.exec_action.command))
    elif probe.http_get is not None:
        kwargs["http_get"] = k8s.client.V1HTTPGetAction(
            path=probe.http_get.path,
            port=probe.http_get.port,
            scheme=probe.http_get.scheme,
        )
    elif probe.tcp_socket is not None:
        kwargs["tcp_socket"] = k8s.client.V1TCPSocketAction(port=probe.tcp_socket.port)
    else:
        return None
    return k8s.client.V1Probe(**kwargs)


def _build_container_ports(ports: list[ContainerPort]) -> list[Any]:
    if not ports:
        return []
    k8s = k8s_client_module()
    return [
        k8s.client.V1ContainerPort(
            name=port.name or f"port-{port.container_port}",
            container_port=port.container_port,
            protocol=port.protocol,
        )
        for port in ports
    ]


def _config_file_mounts(config_files: list[ConfigFile]) -> list[VolumeMount]:
    return [
        VolumeMount(
            name=CONFIG_FILES_VOLUME,
            mountPath=config_file.path,
            readOnly=True,
            subPath=config_file.path.lstrip("/"),
        )
        for config_file in config_files
    ]


def _collect_pvc_mounts(config: DeploymentConfig) -> list[VolumeMount]:
    mounts_by_name: dict[str, VolumeMount] = {}
    for mount in config.volume_mounts:
        mounts_by_name[mount.name] = mount
    for container in (*config.init_containers, *config.containers):
        for mount in container.volume_mounts:
            mounts_by_name[mount.name] = mount
    return list(mounts_by_name.values())


def workload_identity_enabled(config: DeploymentConfig) -> bool:
    spec = config.workload_identity
    return spec is not None and spec.enabled


def build_workload_identity_volume(config: DeploymentConfig) -> V1Volume:
    if config.workload_identity is None:
        raise DeploymentConfigError("workload_identity is required to build a workload identity volume")
    k8s = k8s_client_module()
    platform_audience = get_workload_identity_token_audience()
    configured_audience = config.workload_identity.token_audience
    if configured_audience is not None and configured_audience != platform_audience:
        raise DeploymentConfigError(
            f"workload_identity.tokenAudience must match the platform workload token audience {platform_audience!r}"
        )
    audience = configured_audience or platform_audience
    return k8s.client.V1Volume(
        name=WORKLOAD_IDENTITY_VOLUME_NAME,
        projected=k8s.client.V1ProjectedVolumeSource(
            sources=[
                k8s.client.V1VolumeProjection(
                    service_account_token=k8s.client.V1ServiceAccountTokenProjection(
                        path="token",
                        expiration_seconds=config.workload_identity.token_expiration_seconds,
                        audience=audience,
                    )
                )
            ]
        ),
    )


def pod_service_account_name(
    *,
    config: DeploymentConfig,
    k8s_config: K8sDeploymentConfig | None,
) -> str | None:
    if workload_identity_enabled(config):
        return workload_service_account_name(config=config, k8s_config=k8s_config)
    if k8s_config is not None and k8s_config.service_account:
        return k8s_config.service_account
    return None


def _workload_identity_mount() -> VolumeMount:
    return VolumeMount(
        name=WORKLOAD_IDENTITY_VOLUME_NAME,
        mountPath=WORKLOAD_IDENTITY_VOLUME_PATH,
        readOnly=True,
    )


def _workload_identity_env_for_container(
    config: DeploymentConfig, *, include_workload_identity: bool
) -> dict[str, str]:
    if not include_workload_identity or not workload_identity_enabled(config):
        return {}
    return workload_identity_env(token_file_path=WORKLOAD_IDENTITY_TOKEN_FILE_PATH)


def build_container(
    container: Container,
    *,
    config: DeploymentConfig,
    include_probes: bool,
    secret_name: str | None = None,
    include_workload_identity: bool = True,
) -> Any:
    """Build a V1Container from a plugin Container."""
    k8s = k8s_client_module()
    mounts = merged_volume_mounts(config, container)
    if config.config_files:
        mounts = [*mounts, *_config_file_mounts(config.config_files)]
    if include_workload_identity and workload_identity_enabled(config):
        mounts = [*mounts, _workload_identity_mount()]
    base = build_container_spec(
        container,
        volume_mounts=mounts or None,
        secret_name=secret_name,
        extra_env=_workload_identity_env_for_container(config, include_workload_identity=include_workload_identity),
    )
    kwargs: dict[str, Any] = {
        "name": base.name,
        "image": base.image,
        "command": base.command,
        "args": base.args,
        "env": base.env,
        "env_from": base.env_from,
        "resources": base.resources,
        "volume_mounts": build_volume_mounts(mounts) if mounts else base.volume_mounts,
        "ports": _build_container_ports(container.ports) or None,
    }
    if include_probes:
        kwargs["liveness_probe"] = _build_probe(container.liveness_probe)
        kwargs["readiness_probe"] = _build_probe(container.readiness_probe)
    if container.restart_policy == NATIVE_SIDECAR_RESTART_POLICY:
        kwargs["restart_policy"] = NATIVE_SIDECAR_RESTART_POLICY
    return k8s.client.V1Container(**{key: value for key, value in kwargs.items() if value is not None})


def _ordered_init_containers(config: DeploymentConfig) -> list[Container]:
    sequential = [c for c in config.init_containers if c.restart_policy != NATIVE_SIDECAR_RESTART_POLICY]
    sidecars = [c for c in config.init_containers if c.restart_policy == NATIVE_SIDECAR_RESTART_POLICY]
    return [*sequential, *sidecars]


def build_tolerations(tolerations: list[Toleration]) -> list[Any]:
    if not tolerations:
        return []
    k8s = k8s_client_module()
    return [k8s.client.V1Toleration(**item.model_dump(by_alias=False, exclude_none=True)) for item in tolerations]


def build_affinity(affinity: Affinity | None) -> Any | None:
    if affinity is None:
        return None
    payload = affinity.model_dump(by_alias=True, exclude_none=True)
    if not payload:
        return None
    return _deserialize_k8s(payload, "V1Affinity")


def build_topology_spread_constraints(constraints: list[dict[str, Any]]) -> list[Any]:
    if not constraints:
        return []
    return [_deserialize_k8s(item, "V1TopologySpreadConstraint") for item in constraints if item]


def build_pod_security_context(security_context: PodSecurityContext | None) -> Any | None:
    if security_context is None:
        return None
    k8s = k8s_client_module()
    payload = security_context.model_dump(by_alias=False, exclude_none=True)
    if not payload:
        return None
    return k8s.client.V1PodSecurityContext(**payload)


def build_configmap_body(
    *,
    workspace: str,
    deployment_name: str,
    labels: dict[str, str],
    config_files: list[ConfigFile],
) -> Any | None:
    if not config_files:
        return None
    k8s = k8s_client_module()
    data = {configmap_data_key(config_file.path): config_file.content for config_file in config_files}
    return k8s.client.V1ConfigMap(
        api_version="v1",
        kind="ConfigMap",
        metadata=k8s.client.V1ObjectMeta(
            name=k8s_deployment_configmap_name(workspace, deployment_name),
            labels=labels,
        ),
        data=data,
    )


def build_secret_body(
    *,
    workspace: str,
    deployment_name: str,
    labels: dict[str, str],
    secret_env: dict[str, str],
) -> Any | None:
    """Build the per-deployment managed ``V1Secret`` for resolved secret env vars.

    Returns ``None`` when the deployment references no secrets. The Secret holds
    every resolved secret env var (keyed by env var name) and is mounted into
    containers via ``envFrom`` so the plaintext never lands in the pod manifest.
    """
    if not secret_env:
        return None
    k8s = k8s_client_module()
    return k8s.client.V1Secret(
        api_version="v1",
        kind="Secret",
        type="Opaque",
        metadata=k8s.client.V1ObjectMeta(
            name=k8s_deployment_secret_name(workspace, deployment_name),
            labels=labels,
        ),
        string_data=dict(secret_env),
    )


def _build_config_file_volume(configmap_name: str, config_files: list[ConfigFile]) -> Any:
    k8s = k8s_client_module()
    items = [
        k8s.client.V1KeyToPath(key=configmap_data_key(config_file.path), path=config_file.path.lstrip("/"))
        for config_file in config_files
    ]
    return k8s.client.V1Volume(
        name=CONFIG_FILES_VOLUME,
        config_map=k8s.client.V1ConfigMapVolumeSource(name=configmap_name, items=items),
    )


def build_pod_image_pull_secrets(
    executor_image_pull_secrets: list[ImagePullSecret] | None = None,
) -> list[Any]:
    """Merge platform and executor image pull secrets for pod specs."""
    k8s = k8s_client_module()
    merged_names: dict[str, None] = {}
    for secret in get_platform_config().image_pull_secrets:
        merged_names[secret.name] = None
    for secret in executor_image_pull_secrets or []:
        merged_names[secret.name] = None
    return [k8s.client.V1LocalObjectReference(name=name) for name in merged_names]


def compile_workload(
    *,
    config: DeploymentConfig,
    workspace: str,
    deployment_name: str,
    labels: dict[str, str],
    k8s_config: K8sDeploymentConfig | None,
    pod_restart_policy: RestartPolicy,
    executor_image_pull_secrets: list[ImagePullSecret] | None = None,
    executor_defaults: ExecutorK8sDefaults | None = None,
    secret_env: dict[str, str] | None = None,
) -> CompiledWorkload:
    """Compile pod spec kwargs and optional ConfigMap/Secret for a Job or Deployment.

    ``secret_env`` maps env var names to resolved plaintext secret values. When
    non-empty, a single per-deployment ``Secret`` is compiled and mounted into
    every container via ``envFrom``.
    """
    validate_workload_config(config)
    pvc_mounts = _collect_pvc_mounts(config)
    volumes = build_pod_volumes(workspace=workspace, mounts=pvc_mounts)
    configmap_body = build_configmap_body(
        workspace=workspace,
        deployment_name=deployment_name,
        labels=labels,
        config_files=config.config_files,
    )
    configmap_name = configmap_body.metadata.name if configmap_body is not None else None
    if configmap_name is not None:
        volumes = [*volumes, _build_config_file_volume(configmap_name, config.config_files)]
    if workload_identity_enabled(config):
        volumes = [*volumes, build_workload_identity_volume(config)]

    secret_body = build_secret_body(
        workspace=workspace,
        deployment_name=deployment_name,
        labels=labels,
        secret_env=secret_env or {},
    )
    secret_name = secret_body.metadata.name if secret_body is not None else None

    ordered_init = list(_ordered_init_containers(config))
    init_containers = [
        build_container(
            container,
            config=config,
            include_probes=container.restart_policy == NATIVE_SIDECAR_RESTART_POLICY,
            secret_name=secret_name,
        )
        for container in ordered_init
    ]
    # Auth-proxy sidecar (native sidecar with restartPolicy=Always) is appended so
    # it starts before the main workload and keeps running. No-op when the config
    # does not request it or platform auth is disabled. It is a platform-managed
    # container and deliberately does NOT receive the workload's secret envFrom.
    auth_proxy = build_auth_proxy_container(config)
    if auth_proxy is not None:
        init_containers.append(
            build_container(auth_proxy, config=config, include_probes=True, include_workload_identity=False)
        )
    main_containers = [
        build_container(container, config=config, include_probes=True, secret_name=secret_name)
        for container in config.containers
    ]

    pod_spec_kwargs: dict[str, Any] = {
        "restart_policy": pod_restart_policy,
        "containers": main_containers,
    }
    if init_containers:
        pod_spec_kwargs["init_containers"] = init_containers
    if volumes:
        pod_spec_kwargs["volumes"] = volumes

    image_pull_secrets = build_pod_image_pull_secrets(executor_image_pull_secrets)
    if image_pull_secrets:
        pod_spec_kwargs["image_pull_secrets"] = image_pull_secrets

    if k8s_config is not None:
        tolerations = build_tolerations(k8s_config.tolerations)
        if tolerations:
            pod_spec_kwargs["tolerations"] = tolerations
        if k8s_config.node_selector:
            pod_spec_kwargs["node_selector"] = dict(k8s_config.node_selector)
        affinity = build_affinity(k8s_config.affinity)
        if affinity is not None:
            pod_spec_kwargs["affinity"] = affinity
        topology_spread_constraints = build_topology_spread_constraints(k8s_config.topology_spread_constraints)
        if topology_spread_constraints:
            pod_spec_kwargs["topology_spread_constraints"] = topology_spread_constraints
        security_context = build_pod_security_context(k8s_config.security_context)
        if security_context is not None:
            pod_spec_kwargs["security_context"] = security_context
    effective_service_account_name = pod_service_account_name(config=config, k8s_config=k8s_config)
    if effective_service_account_name:
        pod_spec_kwargs["service_account_name"] = effective_service_account_name

    # Apply executor-level defaults as the BASE layer, shared by every consumer
    # (models, agents, ...). A per-entity K8sDeploymentConfig value (set above)
    # wins: node_selector / tolerations / affinity / topology_spread are applied
    # only when the entity left them unset; annotations merge key-wise below.
    entity_pod_annotations = dict(k8s_config.pod_annotations) if k8s_config is not None else {}
    if executor_defaults is not None:
        if executor_defaults.node_selector and "node_selector" not in pod_spec_kwargs:
            pod_spec_kwargs["node_selector"] = dict(executor_defaults.node_selector)
        if executor_defaults.tolerations and "tolerations" not in pod_spec_kwargs:
            default_tolerations = build_tolerations(
                [Toleration.model_validate(item) for item in executor_defaults.tolerations if item]
            )
            if default_tolerations:
                pod_spec_kwargs["tolerations"] = default_tolerations
        if executor_defaults.affinity and "affinity" not in pod_spec_kwargs:
            default_affinity = _deserialize_k8s(executor_defaults.affinity, "V1Affinity")
            if default_affinity is not None:
                pod_spec_kwargs["affinity"] = default_affinity
        if executor_defaults.topology_spread_constraints and "topology_spread_constraints" not in pod_spec_kwargs:
            default_tsc = build_topology_spread_constraints(executor_defaults.topology_spread_constraints)
            if default_tsc:
                pod_spec_kwargs["topology_spread_constraints"] = default_tsc
        # Annotations merge key-wise: executor default first, per-entity wins.
        pod_annotations = {**executor_defaults.pod_annotations, **entity_pod_annotations}
    else:
        pod_annotations = entity_pod_annotations

    return CompiledWorkload(
        pod_spec_kwargs=pod_spec_kwargs,
        configmap_body=configmap_body,
        configmap_name=configmap_name,
        service_containers=tuple(config.containers),
        secret_body=secret_body,
        secret_name=secret_name,
        pod_annotations=pod_annotations,
    )


def create_configmap(
    core_v1: Any,
    *,
    namespace: str,
    body: Any,
    expected_labels: dict[str, str],
    timeout: float | None,
) -> None:
    try:
        core_v1.create_namespaced_config_map(namespace=namespace, body=body, _request_timeout=timeout)
    except ApiException as exc:
        _reraise_api_unless(exc, 409)
        existing = core_v1.read_namespaced_config_map(
            name=body.metadata.name,
            namespace=namespace,
            _request_timeout=timeout,
        )
        if not resource_labels_match(existing, expected_labels):
            raise


def delete_configmap_best_effort(
    core_v1: Any,
    *,
    namespace: str,
    name: str | None,
    expected_labels: dict[str, str],
    timeout: float | None,
) -> None:
    if name is None:
        return
    try:
        delete_configmap(
            core_v1,
            namespace=namespace,
            name=name,
            expected_labels=expected_labels,
            timeout=timeout,
        )
    except ApiException:
        logger.debug("Best-effort ConfigMap cleanup failed for %s", name, exc_info=True)


def delete_configmap(
    core_v1: Any,
    *,
    namespace: str,
    name: str,
    expected_labels: dict[str, str],
    timeout: float | None,
) -> None:
    try:
        configmap = core_v1.read_namespaced_config_map(name=name, namespace=namespace, _request_timeout=timeout)
    except ApiException as exc:
        _reraise_api_unless(exc, 404)
        return
    if not resource_labels_match(configmap, expected_labels):
        return
    try:
        core_v1.delete_namespaced_config_map(name=name, namespace=namespace, _request_timeout=timeout)
    except ApiException as exc:
        _reraise_api_unless(exc, 404)


def create_secret(
    core_v1: Any,
    *,
    namespace: str,
    body: Any,
    expected_labels: dict[str, str],
    timeout: float | None,
) -> None:
    try:
        core_v1.create_namespaced_secret(namespace=namespace, body=body, _request_timeout=timeout)
    except ApiException as exc:
        _reraise_api_unless(exc, 409)
        existing = core_v1.read_namespaced_secret(
            name=body.metadata.name,
            namespace=namespace,
            _request_timeout=timeout,
        )
        if not resource_labels_match(existing, expected_labels):
            raise


def delete_secret_best_effort(
    core_v1: Any,
    *,
    namespace: str,
    name: str | None,
    expected_labels: dict[str, str],
    timeout: float | None,
) -> None:
    if name is None:
        return
    try:
        delete_secret(
            core_v1,
            namespace=namespace,
            name=name,
            expected_labels=expected_labels,
            timeout=timeout,
        )
    except ApiException:
        logger.debug("Best-effort Secret cleanup failed for %s", name, exc_info=True)


def delete_secret(
    core_v1: Any,
    *,
    namespace: str,
    name: str,
    expected_labels: dict[str, str],
    timeout: float | None,
) -> None:
    try:
        secret = core_v1.read_namespaced_secret(name=name, namespace=namespace, _request_timeout=timeout)
    except ApiException as exc:
        _reraise_api_unless(exc, 404)
        return
    if not resource_labels_match(secret, expected_labels):
        return
    try:
        core_v1.delete_namespaced_secret(name=name, namespace=namespace, _request_timeout=timeout)
    except ApiException as exc:
        _reraise_api_unless(exc, 404)
