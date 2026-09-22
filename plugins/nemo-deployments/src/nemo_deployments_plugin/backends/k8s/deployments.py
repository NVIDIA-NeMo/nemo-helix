# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes Deployment + Service lifecycle for long-running workloads (Always).

The official kubernetes-client is synchronous; ``asyncio.to_thread`` matches the
Job backend (``jobs.py``). Async client support is not in scope for this phase.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from kubernetes.client.models import V1Deployment, V1Pod, V1ReplicaSet
from kubernetes.client.rest import ApiException
from nemo_deployments_plugin.backends.base import BackendStatusUpdate, LogResult
from nemo_deployments_plugin.backends.k8s.client import KubernetesClients, k8s_client_module
from nemo_deployments_plugin.backends.k8s.compiler import (
    CompiledWorkload,
    DeploymentConfigError,
    ExecutorK8sDefaults,
    compile_workload,
    create_configmap,
    create_secret,
    delete_configmap,
    delete_configmap_best_effort,
    delete_secret,
    delete_secret_best_effort,
    validate_config_for_deployment,
)
from nemo_deployments_plugin.backends.k8s.jobs import (
    newest_pod,
    resolve_deployment_namespace,
    resolve_k8s_deployment_config,
    trim_log_text,
)
from nemo_deployments_plugin.backends.k8s.status import (
    missing_deployment_status,
    resource_labels_match,
    status_from_deployment,
)
from nemo_deployments_plugin.backends.k8s.workload_identity import (
    POD_LIST_UNAVAILABLE,
    PodListResult,
    PodListUnavailable,
    ReplicaSetOwner,
    ReplicaSetOwnerListResult,
    deployment_pod_uid_delegation_pods,
    reconcile_pod_uid_delegations,
    replica_set_owner_from_resource,
    revoke_workload_delegations,
)
from nemo_deployments_plugin.backends.labels import (
    CONFIG_NAME_LABEL,
    DEPLOYMENT_NAME_LABEL,
    DEPLOYMENT_WORKSPACE_LABEL,
    RESTART_POLICY_LABEL,
    deployment_identity_labels,
    k8s_deployment_configmap_name,
    k8s_deployment_resource_name,
    k8s_deployment_secret_name,
    managed_by_label_selector,
)
from nemo_deployments_plugin.backends.workload_identity import (
    workload_identity_activation_error,
    workload_identity_reconcile_allowed,
    workload_identity_requested,
)
from nemo_deployments_plugin.entities import Container, DeploymentConfig, K8sDeploymentConfig
from nemo_deployments_plugin.types import Endpoint, RestartPolicy
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltDeployment:
    """An apps/v1.Deployment plus the compiled workload used to build its pod template."""

    deployment: V1Deployment
    compiled: CompiledWorkload


APP_LABEL = "app"
DEFAULT_SERVICE_PORT = 8080


def app_selector_labels(resource_name: str) -> dict[str, str]:
    return {APP_LABEL: resource_name}


def build_deployment_body(
    *,
    resource_name: str,
    labels: dict[str, str],
    config: DeploymentConfig,
    workspace: str,
    deployment_name: str,
    k8s_config: K8sDeploymentConfig | None,
    executor_image_pull_secrets: list | None = None,
    executor_defaults: ExecutorK8sDefaults | None = None,
    secret_env: dict[str, str] | None = None,
) -> BuiltDeployment:
    """Build an ``apps/v1.Deployment`` for create and its compiled workload."""
    k8s = k8s_client_module()
    selector_labels = app_selector_labels(resource_name)
    pod_labels = selector_labels | labels
    compiled = compile_workload(
        config=config,
        workspace=workspace,
        deployment_name=deployment_name,
        labels=labels,
        k8s_config=k8s_config,
        pod_restart_policy="Always",
        executor_image_pull_secrets=executor_image_pull_secrets,
        executor_defaults=executor_defaults,
        secret_env=secret_env,
    )
    deployment = k8s.client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=k8s.client.V1ObjectMeta(name=resource_name, labels=labels),
        spec=k8s.client.V1DeploymentSpec(
            replicas=1,
            selector=k8s.client.V1LabelSelector(match_labels=selector_labels),
            template=k8s.client.V1PodTemplateSpec(
                metadata=k8s.client.V1ObjectMeta(
                    labels=pod_labels,
                    annotations=compiled.pod_annotations or None,
                ),
                spec=k8s.client.V1PodSpec(**compiled.pod_spec_kwargs),
            ),
        ),
    )
    return BuiltDeployment(deployment=deployment, compiled=compiled)


def build_service_body(*, resource_name: str, labels: dict[str, str], containers: tuple[Container, ...]) -> Any:
    """Build a ClusterIP ``v1.Service`` exposing container ports in-cluster."""
    k8s = k8s_client_module()
    selector_labels = app_selector_labels(resource_name)
    service_ports: list[Any] = []
    seen_ports: set[tuple[int, str]] = set()
    for container in containers:
        for port in container.ports:
            port_key = (port.container_port, port.protocol)
            if port_key in seen_ports:
                continue
            seen_ports.add(port_key)
            port_name = port.name or f"port-{port.container_port}"
            service_ports.append(
                k8s.client.V1ServicePort(
                    name=port_name,
                    port=port.container_port,
                    target_port=port_name,
                    protocol=port.protocol,
                )
            )
    if not service_ports:
        service_ports.append(
            k8s.client.V1ServicePort(
                name="http",
                port=DEFAULT_SERVICE_PORT,
                target_port=DEFAULT_SERVICE_PORT,
                protocol="TCP",
            )
        )
    return k8s.client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=k8s.client.V1ObjectMeta(name=resource_name, labels=labels),
        spec=k8s.client.V1ServiceSpec(
            type="ClusterIP",
            selector=selector_labels,
            ports=service_ports,
        ),
    )


def build_in_cluster_endpoints(
    *,
    resource_name: str,
    namespace: str,
    containers: tuple[Container, ...],
) -> list[Endpoint]:
    """Build in-cluster endpoints for exposed container ports."""
    endpoints: list[Endpoint] = []
    host = f"{resource_name}.{namespace}.svc.cluster.local"
    for container in containers:
        for port in container.ports:
            endpoint_name = port.name or f"port-{port.container_port}"
            is_udp = port.protocol == "UDP"
            protocol = "tcp" if is_udp else "http"
            scheme = "tcp" if is_udp else "http"
            endpoints.append(
                Endpoint(
                    name=endpoint_name,
                    url=f"{scheme}://{host}:{port.container_port}",
                    protocol=protocol,
                )
            )
    if endpoints:
        return endpoints
    endpoints.append(Endpoint(name="http", url=f"http://{host}:{DEFAULT_SERVICE_PORT}", protocol="http"))
    return endpoints


def _label_selector(match_labels: dict[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in match_labels.items())


async def _read_pods(
    clients: KubernetesClients,
    *,
    namespace: str,
    match_labels: dict[str, str],
) -> PodListResult:
    timeout = clients.request_timeout
    core_v1 = clients.core_v1

    def _list() -> list[V1Pod]:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=_label_selector(match_labels),
            _request_timeout=timeout,
        )
        return list(pods.items or [])

    try:
        return await asyncio.to_thread(_list)
    except Exception:
        logger.debug("Could not list pods for selector %s", match_labels, exc_info=True)
        return POD_LIST_UNAVAILABLE


async def _read_replica_set_owners(
    clients: KubernetesClients,
    *,
    namespace: str,
    match_labels: dict[str, str],
) -> ReplicaSetOwnerListResult:
    timeout = clients.request_timeout
    apps_v1 = clients.apps_v1

    def _list() -> list[V1ReplicaSet]:
        replica_sets = apps_v1.list_namespaced_replica_set(
            namespace=namespace,
            label_selector=_label_selector(match_labels),
            _request_timeout=timeout,
        )
        return list(replica_sets.items or [])

    try:
        owners: list[ReplicaSetOwner] = []
        for replica_set in await asyncio.to_thread(_list):
            owner = replica_set_owner_from_resource(replica_set)
            if owner is not None:
                owners.append(owner)
        return owners
    except Exception:
        logger.debug("Could not list ReplicaSets for selector %s", match_labels, exc_info=True)
        return POD_LIST_UNAVAILABLE


def _metadata_uid(resource: V1Deployment) -> str | None:
    metadata = resource.metadata
    if metadata is None:
        return None
    return metadata.uid or None


def _newest_available_pod(pods: PodListResult) -> V1Pod | None:
    if isinstance(pods, PodListUnavailable):
        return None
    return newest_pod(pods)


async def _read_deployment_pod_uid_delegation_pods(
    clients: KubernetesClients,
    *,
    namespace: str,
    resource_name: str,
    deployment: V1Deployment,
    config: DeploymentConfig,
    k8s_config: K8sDeploymentConfig | None,
    pods: PodListResult,
) -> PodListResult:
    if not workload_identity_requested(config) or isinstance(pods, PodListUnavailable):
        return pods
    replica_set_owners = await _read_replica_set_owners(
        clients,
        namespace=namespace,
        match_labels=app_selector_labels(resource_name),
    )
    return deployment_pod_uid_delegation_pods(
        config=config,
        k8s_config=k8s_config,
        resource_name=resource_name,
        deployment_uid=_metadata_uid(deployment),
        replica_sets=replica_set_owners,
        pods=pods,
    )


async def _read_newest_pod(
    clients: KubernetesClients,
    *,
    namespace: str,
    match_labels: dict[str, str],
) -> V1Pod | None:
    return _newest_available_pod(await _read_pods(clients, namespace=namespace, match_labels=match_labels))


def _log_cleanup_ignored(resource_name: str, exc: ApiException) -> None:
    logger.debug(
        "Best-effort cleanup of %s failed (resource may already be deleted)",
        resource_name,
        exc_info=exc,
    )


async def create_deployment(
    clients: KubernetesClients,
    *,
    default_namespace: str,
    workspace: str,
    name: str,
    config_name: str,
    labels: dict[str, str],
    backend_config: dict[str, Any],
    config: DeploymentConfig,
    executor_image_pull_secrets: list | None = None,
    executor_defaults: ExecutorK8sDefaults | None = None,
    secret_env: dict[str, str] | None = None,
    auth_context: AuthContext | None = None,
    workload_delegation_store: WorkloadDelegationStore | None = None,
) -> BackendStatusUpdate:
    resource_name = k8s_deployment_resource_name(workspace, name)
    try:
        validate_config_for_deployment(config)
        k8s_config = resolve_k8s_deployment_config(backend_config)
        namespace = resolve_deployment_namespace(default_namespace=default_namespace, k8s_config=k8s_config)
        identity_error = workload_identity_activation_error(config=config, auth_context=auth_context)
        if identity_error is not None:
            return BackendStatusUpdate(status="FAILED", status_message=identity_error)
        identity_labels = deployment_identity_labels(
            workspace,
            name,
            config.restart_policy,
            config_name=config_name,
            backoff_limit=config.backoff_limit,
        )
        all_labels = {**labels, **config.labels, **identity_labels}
        built = build_deployment_body(
            resource_name=resource_name,
            labels=all_labels,
            config=config,
            workspace=workspace,
            deployment_name=name,
            k8s_config=k8s_config,
            executor_image_pull_secrets=executor_image_pull_secrets,
            executor_defaults=executor_defaults,
            secret_env=secret_env,
        )
        deployment_body = built.deployment
        compiled = built.compiled
        service_body = build_service_body(
            resource_name=resource_name,
            labels=all_labels,
            containers=compiled.service_containers,
        )
        timeout = clients.request_timeout
        apps_v1 = clients.apps_v1
        core_v1 = clients.core_v1

        def _rollback_partial_create(
            *, deployment_created: bool, should_delete_configmap: bool, should_delete_secret: bool
        ) -> None:
            if deployment_created:
                try:
                    apps_v1.delete_namespaced_deployment(
                        name=resource_name,
                        namespace=namespace,
                        propagation_policy="Background",
                        _request_timeout=timeout,
                    )
                except ApiException as cleanup_exc:
                    _log_cleanup_ignored(resource_name, cleanup_exc)
            if should_delete_configmap:
                delete_configmap_best_effort(
                    core_v1,
                    namespace=namespace,
                    name=compiled.configmap_name,
                    expected_labels=identity_labels,
                    timeout=timeout,
                )
            if should_delete_secret:
                delete_secret_best_effort(
                    core_v1,
                    namespace=namespace,
                    name=compiled.secret_name,
                    expected_labels=identity_labels,
                    timeout=timeout,
                )

        def _create() -> Any:
            deployment_created = False
            secret_written = compiled.secret_body is not None
            if secret_written:
                create_secret(
                    core_v1,
                    namespace=namespace,
                    body=compiled.secret_body,
                    expected_labels=identity_labels,
                    timeout=timeout,
                )
            configmap_written = compiled.configmap_body is not None
            if configmap_written:
                try:
                    create_configmap(
                        core_v1,
                        namespace=namespace,
                        body=compiled.configmap_body,
                        expected_labels=identity_labels,
                        timeout=timeout,
                    )
                except Exception:
                    if secret_written:
                        delete_secret_best_effort(
                            core_v1,
                            namespace=namespace,
                            name=compiled.secret_name,
                            expected_labels=identity_labels,
                            timeout=timeout,
                        )
                    raise
            try:
                apps_v1.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment_body,
                    _request_timeout=timeout,
                )
                deployment_created = True
            except ApiException as exc:
                if exc.status != 409:
                    if configmap_written:
                        delete_configmap_best_effort(
                            core_v1,
                            namespace=namespace,
                            name=compiled.configmap_name,
                            expected_labels=identity_labels,
                            timeout=timeout,
                        )
                    if secret_written:
                        delete_secret_best_effort(
                            core_v1,
                            namespace=namespace,
                            name=compiled.secret_name,
                            expected_labels=identity_labels,
                            timeout=timeout,
                        )
                    raise

            deployment = apps_v1.read_namespaced_deployment(
                name=resource_name,
                namespace=namespace,
                _request_timeout=timeout,
            )
            if not resource_labels_match(deployment, identity_labels):
                _rollback_partial_create(
                    deployment_created=deployment_created,
                    should_delete_configmap=configmap_written,
                    should_delete_secret=secret_written,
                )
                return deployment

            should_delete_configmap_on_service_failure = deployment_created and configmap_written
            should_delete_secret_on_service_failure = deployment_created and secret_written
            try:
                core_v1.create_namespaced_service(
                    namespace=namespace,
                    body=service_body,
                    _request_timeout=timeout,
                )
            except ApiException as exc:
                if exc.status == 409:
                    existing_service = core_v1.read_namespaced_service(
                        name=resource_name,
                        namespace=namespace,
                        _request_timeout=timeout,
                    )
                    if not resource_labels_match(existing_service, identity_labels):
                        _rollback_partial_create(
                            deployment_created=deployment_created,
                            should_delete_configmap=should_delete_configmap_on_service_failure,
                            should_delete_secret=should_delete_secret_on_service_failure,
                        )
                        raise
                    return deployment
                _rollback_partial_create(
                    deployment_created=deployment_created,
                    should_delete_configmap=should_delete_configmap_on_service_failure,
                    should_delete_secret=should_delete_secret_on_service_failure,
                )
                raise
            return deployment

        deployment = await asyncio.to_thread(_create)
        endpoints = build_in_cluster_endpoints(
            resource_name=resource_name,
            namespace=namespace,
            containers=compiled.service_containers,
        )
        if not resource_labels_match(deployment, identity_labels):
            return status_from_deployment(
                deployment=deployment,
                deployment_name=resource_name,
                expected_labels=identity_labels,
                endpoints=endpoints,
            )
        pods = await _read_pods(clients, namespace=namespace, match_labels=app_selector_labels(resource_name))
        if workload_identity_reconcile_allowed(config, auth_context):
            pod_uid_delegation_pods = await _read_deployment_pod_uid_delegation_pods(
                clients,
                namespace=namespace,
                resource_name=resource_name,
                deployment=deployment,
                config=config,
                k8s_config=k8s_config,
                pods=pods,
            )
            await reconcile_pod_uid_delegations(
                workload_delegation_store,
                config=config,
                auth_context=auth_context,
                workspace=workspace,
                deployment_name=name,
                namespace=namespace,
                k8s_config=k8s_config,
                pods=pod_uid_delegation_pods,
            )
        pod = _newest_available_pod(pods)
        return status_from_deployment(
            deployment=deployment,
            deployment_name=resource_name,
            expected_labels=identity_labels,
            endpoints=endpoints,
            pod=pod,
        )
    except DeploymentConfigError as exc:
        return BackendStatusUpdate(status="FAILED", status_message=str(exc))
    except Exception as exc:
        logger.exception("Failed to create Deployment %s", resource_name)
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to create Deployment: {exc}")


async def read_deployment_status(
    clients: KubernetesClients,
    *,
    default_namespace: str,
    workspace: str,
    name: str,
    backend_config: dict[str, Any],
    config_name: str,
    restart_policy: RestartPolicy,
    backoff_limit: int,
    containers: tuple[Container, ...],
    config: DeploymentConfig | None = None,
    auth_context: AuthContext | None = None,
    workload_delegation_store: WorkloadDelegationStore | None = None,
) -> BackendStatusUpdate:
    resource_name = k8s_deployment_resource_name(workspace, name)
    expected_labels = deployment_identity_labels(
        workspace,
        name,
        restart_policy,
        config_name=config_name,
        backoff_limit=backoff_limit,
    )
    try:
        k8s_config = resolve_k8s_deployment_config(backend_config)
        namespace = resolve_deployment_namespace(default_namespace=default_namespace, k8s_config=k8s_config)
        timeout = clients.request_timeout
        apps_v1 = clients.apps_v1

        def _read() -> Any:
            return apps_v1.read_namespaced_deployment(
                name=resource_name,
                namespace=namespace,
                _request_timeout=timeout,
            )

        deployment = await asyncio.to_thread(_read)
        deployment_labels = (deployment.metadata.labels or {}) if deployment.metadata else {}
        if not deployment_labels.get(CONFIG_NAME_LABEL):
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Deployment {resource_name} is missing deployment config identity labels",
            )
        service_containers = tuple(config.containers) if config is not None else containers
        endpoints = build_in_cluster_endpoints(
            resource_name=resource_name,
            namespace=namespace,
            containers=service_containers,
        )
        if not resource_labels_match(deployment, expected_labels):
            return status_from_deployment(
                deployment=deployment,
                deployment_name=resource_name,
                expected_labels=expected_labels,
                endpoints=endpoints,
            )
        pods = await _read_pods(clients, namespace=namespace, match_labels=app_selector_labels(resource_name))
        if config is not None and workload_identity_reconcile_allowed(config, auth_context):
            pod_uid_delegation_pods = await _read_deployment_pod_uid_delegation_pods(
                clients,
                namespace=namespace,
                resource_name=resource_name,
                deployment=deployment,
                config=config,
                k8s_config=k8s_config,
                pods=pods,
            )
            await reconcile_pod_uid_delegations(
                workload_delegation_store,
                config=config,
                auth_context=auth_context,
                workspace=workspace,
                deployment_name=name,
                namespace=namespace,
                k8s_config=k8s_config,
                pods=pod_uid_delegation_pods,
            )
        pod = _newest_available_pod(pods)
        return status_from_deployment(
            deployment=deployment,
            deployment_name=resource_name,
            expected_labels=expected_labels,
            endpoints=endpoints,
            pod=pod,
        )
    except ApiException as exc:
        if exc.status == 404:
            return missing_deployment_status(deployment_name=resource_name)
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to read Deployment: {exc}")
    except Exception as exc:
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to read Deployment: {exc}")


async def delete_deployment(
    clients: KubernetesClients,
    *,
    default_namespace: str,
    workspace: str,
    name: str,
    backend_config: dict[str, Any],
    expected_labels: dict[str, str],
    config: DeploymentConfig | None = None,
    workload_delegation_store: WorkloadDelegationStore | None = None,
) -> BackendStatusUpdate:
    resource_name = k8s_deployment_resource_name(workspace, name)
    configmap_name = k8s_deployment_configmap_name(workspace, name)
    secret_name = k8s_deployment_secret_name(workspace, name)
    try:
        k8s_config = resolve_k8s_deployment_config(backend_config)
        namespace = resolve_deployment_namespace(default_namespace=default_namespace, k8s_config=k8s_config)
        timeout = clients.request_timeout
        apps_v1 = clients.apps_v1
        core_v1 = clients.core_v1

        def _delete_config_resources() -> None:
            """Delete the ConfigMap and managed Secret for this deployment."""
            delete_configmap(
                core_v1,
                namespace=namespace,
                name=configmap_name,
                expected_labels=expected_labels,
                timeout=timeout,
            )
            delete_secret(
                core_v1,
                namespace=namespace,
                name=secret_name,
                expected_labels=expected_labels,
                timeout=timeout,
            )

        def _delete() -> str | None:
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=resource_name,
                    namespace=namespace,
                    _request_timeout=timeout,
                )
            except ApiException as exc:
                if exc.status == 404:
                    try:
                        service = core_v1.read_namespaced_service(
                            name=resource_name,
                            namespace=namespace,
                            _request_timeout=timeout,
                        )
                    except ApiException as service_read_exc:
                        if service_read_exc.status != 404:
                            raise
                        _delete_config_resources()
                        return None
                    if resource_labels_match(service, expected_labels):
                        try:
                            core_v1.delete_namespaced_service(
                                name=resource_name,
                                namespace=namespace,
                                _request_timeout=timeout,
                            )
                        except ApiException as service_exc:
                            if service_exc.status != 404:
                                raise
                    _delete_config_resources()
                    return None
                raise
            if not resource_labels_match(deployment, expected_labels):
                return "foreign"
            apps_v1.delete_namespaced_deployment(
                name=resource_name,
                namespace=namespace,
                propagation_policy="Background",
                _request_timeout=timeout,
            )
            try:
                core_v1.delete_namespaced_service(
                    name=resource_name,
                    namespace=namespace,
                    _request_timeout=timeout,
                )
            except ApiException as exc:
                if exc.status != 404:
                    raise
            _delete_config_resources()
            return "deleted"

        result = await asyncio.to_thread(_delete)
        if result == "foreign":
            delete_configmap_best_effort(
                core_v1,
                namespace=namespace,
                name=configmap_name,
                expected_labels=expected_labels,
                timeout=timeout,
            )
            delete_secret_best_effort(
                core_v1,
                namespace=namespace,
                name=secret_name,
                expected_labels=expected_labels,
                timeout=timeout,
            )
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Deployment {resource_name} exists but is not managed by this plugin",
            )
        await revoke_workload_delegations(
            workload_delegation_store,
            config=config,
            workspace=workspace,
            deployment_name=name,
        )
        return BackendStatusUpdate(status="SUCCEEDED", status_message=f"Deployment {resource_name} deleted")
    except Exception as exc:
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to delete Deployment: {exc}")


async def list_managed_deployment_names(clients: KubernetesClients, *, default_namespace: str) -> list[str]:
    """List plugin-managed Deployments with restart_policy Always in the default namespace."""
    timeout = clients.request_timeout
    apps_v1 = clients.apps_v1
    label_selector = f"{managed_by_label_selector()},{RESTART_POLICY_LABEL}=Always"

    def _list() -> Any:
        return apps_v1.list_namespaced_deployment(
            namespace=default_namespace,
            label_selector=label_selector,
            _request_timeout=timeout,
        )

    try:
        result = await asyncio.to_thread(_list)
    except Exception:
        logger.warning("Failed to list managed Deployments", exc_info=True)
        return []

    seen: set[str] = set()
    for deployment in result.items or []:
        labels = (deployment.metadata.labels or {}) if deployment.metadata else {}
        workspace = labels.get(DEPLOYMENT_WORKSPACE_LABEL)
        dep_name = labels.get(DEPLOYMENT_NAME_LABEL)
        if workspace and dep_name:
            seen.add(f"{workspace}/{dep_name}")
    return sorted(seen)


async def get_deployment_logs(
    clients: KubernetesClients,
    *,
    default_namespace: str,
    workspace: str,
    name: str,
    backend_config: dict[str, Any],
    tail: int,
) -> LogResult:
    resource_name = k8s_deployment_resource_name(workspace, name)
    try:
        k8s_config = resolve_k8s_deployment_config(backend_config)
        namespace = resolve_deployment_namespace(default_namespace=default_namespace, k8s_config=k8s_config)
        timeout = clients.request_timeout
        core_v1 = clients.core_v1
        selector = _label_selector(app_selector_labels(resource_name))

        def _logs() -> str:
            pods = core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=selector,
                _request_timeout=timeout,
            )
            if not pods.items:
                return ""
            pod = newest_pod(list(pods.items))
            if pod is None or pod.metadata is None:
                return ""
            return core_v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
                tail_lines=tail,
                timestamps=True,
                _request_timeout=timeout,
            )

        text = await asyncio.to_thread(_logs)
        lines, truncated = trim_log_text(text)
        return LogResult(lines=lines, truncated=truncated)
    except ApiException as exc:
        if exc.status == 404:
            return LogResult(lines=[])
        return LogResult(lines=[f"Failed to read Deployment logs: {exc}"])
    except Exception as exc:
        return LogResult(lines=[f"Failed to read Deployment logs: {exc}"])
