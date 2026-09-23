# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes Job lifecycle helpers for finite deployments (Never / OnFailure)."""

from __future__ import annotations

import asyncio
import datetime
import logging
from dataclasses import dataclass
from typing import Any

from kubernetes.client.models import V1Pod
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
    validate_config_for_job,
)
from nemo_deployments_plugin.backends.k8s.status import (
    LOG_MAX_CHARS,
    missing_job_status,
    resource_labels_match,
    status_from_job,
)
from nemo_deployments_plugin.backends.k8s.workload_identity import (
    POD_LIST_UNAVAILABLE,
    PodListResult,
    job_pod_uid_delegation_pods,
    reconcile_pod_uid_delegations,
    revoke_workload_delegations,
)
from nemo_deployments_plugin.backends.labels import (
    CONFIG_NAME_LABEL,
    DEFAULT_RESOURCE_SCOPE,
    DEPLOYMENT_NAME_LABEL,
    DEPLOYMENT_WORKSPACE_LABEL,
    MANAGED_BY_KEY,
    RESOURCE_SCOPE_LABEL,
    deployment_identity_labels,
    k8s_deployment_configmap_name,
    k8s_deployment_resource_name,
    k8s_deployment_secret_name,
    managed_by_label_selector,
)
from nemo_deployments_plugin.backends.workload_identity import (
    workload_identity_activation_error,
    workload_identity_reconcile_allowed,
)
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import DeploymentConfig, K8sDeploymentConfig
from nemo_deployments_plugin.types import RestartPolicy
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltJob:
    """A batch/v1.Job plus the compiled workload used to build its pod template."""

    job: Any
    compiled: CompiledWorkload


def resolve_k8s_deployment_config(backend_config: dict[str, Any]) -> K8sDeploymentConfig | None:
    """Parse and validate the k8s section of entity backend_config."""
    k8s_section = backend_config.get("k8s")
    if not k8s_section:
        return None
    return K8sDeploymentConfig.model_validate(k8s_section)


def resolve_deployment_namespace(*, default_namespace: str, k8s_config: K8sDeploymentConfig | None) -> str:
    """Resolve target namespace from parsed k8s deployment config."""
    if k8s_config is None or not k8s_config.namespace:
        return default_namespace
    return k8s_config.namespace


def deployment_scope_labels(workspace: str, name: str) -> dict[str, str]:
    """Minimum identity labels for orphan delete when the Deployment entity is gone."""
    return {
        MANAGED_BY_KEY: MANAGED_BY_LABEL,
        DEPLOYMENT_WORKSPACE_LABEL: workspace,
        DEPLOYMENT_NAME_LABEL: name,
        RESOURCE_SCOPE_LABEL: DEFAULT_RESOURCE_SCOPE,
    }


def newest_pod(pods: list[V1Pod]) -> V1Pod | None:
    """Return the pod with the latest metadata.creation_timestamp."""
    if not pods:
        return None

    def creation_timestamp(pod: V1Pod) -> tuple[bool, datetime.datetime]:
        metadata = pod.metadata
        if metadata is None:
            return (False, datetime.datetime.min)
        timestamp = metadata.creation_timestamp
        if not isinstance(timestamp, datetime.datetime):
            return (False, datetime.datetime.min)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone(datetime.UTC).replace(tzinfo=None)
        return (True, timestamp)

    return max(pods, key=creation_timestamp)


def trim_log_text(text: str) -> tuple[list[str], bool]:
    truncated = len(text) > LOG_MAX_CHARS
    if truncated:
        text = text[-LOG_MAX_CHARS:]
    lines = text.splitlines() if text else []
    return lines, truncated


def job_backoff_limit(config: DeploymentConfig) -> int:
    if config.restart_policy == "Never":
        return 0
    return config.backoff_limit


def _metadata_uid(resource: Any) -> str | None:
    metadata = getattr(resource, "metadata", None)
    uid = getattr(metadata, "uid", None)
    return uid if isinstance(uid, str) and uid else None


def resolve_job_ttl_seconds_after_finished(
    *,
    k8s_config: K8sDeploymentConfig | None,
    executor_defaults: ExecutorK8sDefaults | None,
) -> int | None:
    """Resolve ttlSecondsAfterFinished for a finite Job.

    A per-entity ``backend_config.k8s.jobTtlSecondsAfterFinished`` wins over the
    executor default; when neither is set the field is omitted (``None``) and the
    cluster default applies.
    """
    if k8s_config is not None and k8s_config.job_ttl_seconds_after_finished is not None:
        return k8s_config.job_ttl_seconds_after_finished
    if executor_defaults is not None:
        return executor_defaults.job_ttl_seconds_after_finished
    return None


def build_job_body(
    *,
    job_name: str,
    labels: dict[str, str],
    config: DeploymentConfig,
    workspace: str,
    deployment_name: str,
    k8s_config: K8sDeploymentConfig | None,
    executor_image_pull_secrets: list | None = None,
    executor_defaults: ExecutorK8sDefaults | None = None,
    secret_env: dict[str, str] | None = None,
) -> BuiltJob:
    """Build a ``batch/v1.Job`` for create."""
    k8s = k8s_client_module()
    compiled = compile_workload(
        config=config,
        workspace=workspace,
        deployment_name=deployment_name,
        labels=labels,
        k8s_config=k8s_config,
        pod_restart_policy=config.restart_policy,
        executor_image_pull_secrets=executor_image_pull_secrets,
        executor_defaults=executor_defaults,
        secret_env=secret_env,
    )
    ttl_seconds_after_finished = resolve_job_ttl_seconds_after_finished(
        k8s_config=k8s_config,
        executor_defaults=executor_defaults,
    )
    job = k8s.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=k8s.client.V1ObjectMeta(name=job_name, labels=labels),
        spec=k8s.client.V1JobSpec(
            backoff_limit=job_backoff_limit(config),
            ttl_seconds_after_finished=ttl_seconds_after_finished,
            template=k8s.client.V1PodTemplateSpec(
                metadata=k8s.client.V1ObjectMeta(
                    labels=labels,
                    annotations=compiled.pod_annotations or None,
                ),
                spec=k8s.client.V1PodSpec(**compiled.pod_spec_kwargs),
            ),
        ),
    )
    return BuiltJob(job=job, compiled=compiled)


async def read_pod_exit_code(
    clients: KubernetesClients,
    *,
    namespace: str,
    job_name: str,
) -> int | None:
    """Best-effort container exit code from the Job's most recent pod."""
    timeout = clients.request_timeout
    core_v1 = clients.core_v1

    def _read() -> int | None:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
            _request_timeout=timeout,
        )
        if not pods.items:
            return None
        pod = newest_pod(list(pods.items))
        if pod is None or pod.status is None or not pod.status.container_statuses:
            return None
        for container_status in pod.status.container_statuses:
            terminated = container_status.state.terminated if container_status.state else None
            if terminated is not None:
                return int(terminated.exit_code)
        return None

    try:
        return await asyncio.to_thread(_read)
    except Exception:
        logger.debug("Could not read exit code for Job %s", job_name, exc_info=True)
        return None


async def read_job_pods(
    clients: KubernetesClients,
    *,
    namespace: str,
    job_name: str,
) -> PodListResult:
    """Best-effort list of pods owned by a Job."""
    timeout = clients.request_timeout
    core_v1 = clients.core_v1

    def _read() -> list[V1Pod]:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
            _request_timeout=timeout,
        )
        return list(pods.items or [])

    try:
        return await asyncio.to_thread(_read)
    except Exception:
        logger.debug("Could not list pods for Job %s", job_name, exc_info=True)
        return POD_LIST_UNAVAILABLE


async def create_job(
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
    job_name = k8s_deployment_resource_name(workspace, name)
    try:
        validate_config_for_job(config)
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
        built = build_job_body(
            job_name=job_name,
            labels=all_labels,
            config=config,
            workspace=workspace,
            deployment_name=name,
            k8s_config=k8s_config,
            executor_image_pull_secrets=executor_image_pull_secrets,
            executor_defaults=executor_defaults,
            secret_env=secret_env,
        )
        body = built.job
        compiled = built.compiled
        timeout = clients.request_timeout
        batch_v1 = clients.batch_v1
        core_v1 = clients.core_v1

        def _cleanup_config_resources(*, configmap_written: bool, secret_written: bool) -> None:
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

        def _create() -> Any:
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
                    _cleanup_config_resources(configmap_written=False, secret_written=secret_written)
                    raise
            try:
                return batch_v1.create_namespaced_job(
                    namespace=namespace,
                    body=body,
                    _request_timeout=timeout,
                )
            except ApiException as exc:
                if exc.status == 409:
                    job = batch_v1.read_namespaced_job(
                        name=job_name,
                        namespace=namespace,
                        _request_timeout=timeout,
                    )
                    if not resource_labels_match(job, identity_labels):
                        _cleanup_config_resources(
                            configmap_written=configmap_written,
                            secret_written=secret_written,
                        )
                    return job
                _cleanup_config_resources(configmap_written=configmap_written, secret_written=secret_written)
                raise

        job = await asyncio.to_thread(_create)
        update = status_from_job(job=job, job_name=job_name, expected_labels=identity_labels)
        if not resource_labels_match(job, identity_labels):
            return update
        pods = await read_job_pods(clients, namespace=namespace, job_name=job_name)
        if workload_identity_reconcile_allowed(config, auth_context):
            await reconcile_pod_uid_delegations(
                workload_delegation_store,
                config=config,
                auth_context=auth_context,
                workspace=workspace,
                deployment_name=name,
                namespace=namespace,
                k8s_config=k8s_config,
                pods=job_pod_uid_delegation_pods(
                    config=config,
                    k8s_config=k8s_config,
                    job_name=job_name,
                    job_uid=_metadata_uid(job),
                    pods=pods,
                ),
            )
        if update.status in ("SUCCEEDED", "FAILED"):
            exit_code = await read_pod_exit_code(clients, namespace=namespace, job_name=job_name)
            if exit_code is not None:
                update = update.model_copy(update={"exit_code": exit_code})
        return update
    except DeploymentConfigError as exc:
        return BackendStatusUpdate(status="FAILED", status_message=str(exc))
    except Exception as exc:
        logger.exception("Failed to create Job %s", job_name)
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to create Job: {exc}")


async def read_job_status(
    clients: KubernetesClients,
    *,
    default_namespace: str,
    workspace: str,
    name: str,
    backend_config: dict[str, Any],
    config_name: str,
    restart_policy: RestartPolicy,
    backoff_limit: int,
    config: DeploymentConfig | None = None,
    auth_context: AuthContext | None = None,
    workload_delegation_store: WorkloadDelegationStore | None = None,
) -> BackendStatusUpdate:
    job_name = k8s_deployment_resource_name(workspace, name)
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
        batch_v1 = clients.batch_v1

        def _read() -> Any:
            return batch_v1.read_namespaced_job(
                name=job_name,
                namespace=namespace,
                _request_timeout=timeout,
            )

        job = await asyncio.to_thread(_read)
        job_labels = (job.metadata.labels or {}) if job.metadata else {}
        if not job_labels.get(CONFIG_NAME_LABEL):
            return BackendStatusUpdate(
                status="FAILED",
                status_message=f"Job {job_name} is missing deployment config identity labels",
            )
        update = status_from_job(job=job, job_name=job_name, expected_labels=expected_labels)
        if not resource_labels_match(job, expected_labels):
            return update
        if config is not None and workload_identity_reconcile_allowed(config, auth_context):
            pods = await read_job_pods(clients, namespace=namespace, job_name=job_name)
            await reconcile_pod_uid_delegations(
                workload_delegation_store,
                config=config,
                auth_context=auth_context,
                workspace=workspace,
                deployment_name=name,
                namespace=namespace,
                k8s_config=k8s_config,
                pods=job_pod_uid_delegation_pods(
                    config=config,
                    k8s_config=k8s_config,
                    job_name=job_name,
                    job_uid=_metadata_uid(job),
                    pods=pods,
                ),
            )
        if update.status in ("SUCCEEDED", "FAILED"):
            exit_code = await read_pod_exit_code(clients, namespace=namespace, job_name=job_name)
            if exit_code is not None:
                update = update.model_copy(update={"exit_code": exit_code})
        return update
    except ApiException as exc:
        if exc.status == 404:
            return missing_job_status(job_name=job_name)
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to read Job: {exc}")
    except Exception as exc:
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to read Job: {exc}")


async def delete_job(
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
    job_name = k8s_deployment_resource_name(workspace, name)
    try:
        k8s_config = resolve_k8s_deployment_config(backend_config)
        namespace = resolve_deployment_namespace(default_namespace=default_namespace, k8s_config=k8s_config)
        timeout = clients.request_timeout
        batch_v1 = clients.batch_v1
        core_v1 = clients.core_v1
        configmap_name = k8s_deployment_configmap_name(workspace, name)
        secret_name = k8s_deployment_secret_name(workspace, name)

        def _delete_config_resources() -> None:
            """Delete the ConfigMap and managed Secret for this job."""
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
                job = batch_v1.read_namespaced_job(
                    name=job_name,
                    namespace=namespace,
                    _request_timeout=timeout,
                )
            except ApiException as exc:
                if exc.status == 404:
                    _delete_config_resources()
                    return None
                raise
            if not resource_labels_match(job, expected_labels):
                return "foreign"
            batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                propagation_policy="Background",
                _request_timeout=timeout,
            )
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
                status_message=f"Job {job_name} exists but is not managed by this plugin",
            )
        await revoke_workload_delegations(
            workload_delegation_store,
            config=config,
            workspace=workspace,
            deployment_name=name,
        )
        return BackendStatusUpdate(status="SUCCEEDED", status_message=f"Job {job_name} deleted")
    except Exception as exc:
        return BackendStatusUpdate(status="FAILED", status_message=f"Failed to delete Job: {exc}")


async def list_managed_job_names(clients: KubernetesClients, *, default_namespace: str) -> list[str]:
    """List plugin-managed Jobs in the executor default namespace.

    Per-deployment namespaces from ``backend_config.k8s.namespace`` are not scanned
    here; custom namespaces are listed when Deployment resources are scanned in phase 4.
    """
    timeout = clients.request_timeout
    batch_v1 = clients.batch_v1
    label_selector = managed_by_label_selector()

    def _list() -> Any:
        return batch_v1.list_namespaced_job(
            namespace=default_namespace,
            label_selector=label_selector,
            _request_timeout=timeout,
        )

    try:
        result = await asyncio.to_thread(_list)
    except Exception:
        logger.warning("Failed to list managed Jobs", exc_info=True)
        return []

    seen: set[str] = set()
    for job in result.items or []:
        labels = (job.metadata.labels or {}) if job.metadata else {}
        workspace = labels.get(DEPLOYMENT_WORKSPACE_LABEL)
        dep_name = labels.get(DEPLOYMENT_NAME_LABEL)
        if workspace and dep_name:
            seen.add(f"{workspace}/{dep_name}")
    return sorted(seen)


async def get_job_logs(
    clients: KubernetesClients,
    *,
    default_namespace: str,
    workspace: str,
    name: str,
    backend_config: dict[str, Any],
    tail: int,
) -> LogResult:
    job_name = k8s_deployment_resource_name(workspace, name)
    try:
        k8s_config = resolve_k8s_deployment_config(backend_config)
        namespace = resolve_deployment_namespace(default_namespace=default_namespace, k8s_config=k8s_config)
        timeout = clients.request_timeout
        core_v1 = clients.core_v1

        def _logs() -> str:
            pods = core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}",
                _request_timeout=timeout,
            )
            if not pods.items:
                return ""
            pod = newest_pod(list(pods.items))
            if pod is None or pod.metadata is None:
                return ""
            pod_name = pod.metadata.name
            raw = core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                tail_lines=tail,
                timestamps=True,
                _request_timeout=timeout,
            )
            return raw

        text = await asyncio.to_thread(_logs)
        lines, truncated = trim_log_text(text)
        return LogResult(lines=lines, truncated=truncated)
    except ApiException as exc:
        if exc.status == 404:
            return LogResult(lines=[])
        return LogResult(lines=[f"Failed to read Job logs: {exc}"])
    except Exception as exc:
        return LogResult(lines=[f"Failed to read Job logs: {exc}"])
