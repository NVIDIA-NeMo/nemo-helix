# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes workload identity delegation reconciliation for deployments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from kubernetes.client.models import V1Pod, V1ReplicaSet
from nemo_deployments_plugin.backends.workload_identity import (
    workload_delegation_scope,
    workload_identity_activation_error,
    workload_identity_requested,
)
from nemo_deployments_plugin.entities import DeploymentConfig, K8sDeploymentConfig
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import (
    KUBERNETES_POD_UID_REFERENCE_NAME,
    WorkloadDelegationConflictError,
    WorkloadDelegationStore,
    as_aware_utc,
)
from nemo_helix_plugin.auth.workload_identity import (
    build_kubernetes_pod_uid_workload_delegation,
    get_workload_delegation_audience,
    workload_delegation_expires_at,
)

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_ACCOUNT = "default"
_TERMINAL_POD_PHASES = frozenset({"Succeeded", "Failed"})


class PodListUnavailable:
    """Sentinel for a Kubernetes list result that could not be read safely."""


POD_LIST_UNAVAILABLE = PodListUnavailable()
PodListResult = list[V1Pod] | PodListUnavailable


@dataclass(frozen=True)
class ReplicaSetOwner:
    """ReplicaSet ownership data needed to prove Deployment-owned Pods."""

    name: str
    uid: str
    deployment_name: str
    deployment_uid: str


ReplicaSetOwnerListResult = list[ReplicaSetOwner] | PodListUnavailable


def service_account_name(
    *,
    config: DeploymentConfig,
    k8s_config: K8sDeploymentConfig | None,
) -> str:
    spec = config.workload_identity
    if spec is not None and spec.service_account_name:
        return spec.service_account_name
    if k8s_config is not None and k8s_config.service_account:
        return k8s_config.service_account
    return DEFAULT_SERVICE_ACCOUNT


async def reconcile_pod_uid_delegations(
    store: WorkloadDelegationStore | None,
    *,
    config: DeploymentConfig,
    auth_context: AuthContext | None,
    workspace: str,
    deployment_name: str,
    namespace: str,
    k8s_config: K8sDeploymentConfig | None,
    pods: PodListResult,
) -> None:
    """Register live Pod UID-bound delegations and revoke stale ones for a workload."""
    if store is None or not workload_identity_requested(config):
        return
    if isinstance(pods, PodListUnavailable):
        logger.debug("Skipping Pod UID workload delegation reconciliation because Kubernetes pod list is unavailable")
        return
    error = workload_identity_activation_error(config=config, auth_context=auth_context)
    if error is not None:
        raise ValueError(error)
    if auth_context is None or config.workload_identity is None:
        return

    scope = workload_delegation_scope(workspace=workspace, deployment_name=deployment_name, config=config)
    existing = await store.list_by_workload(scope)
    existing_by_pod_uid = {
        entity.bound_reference_value: entity
        for entity in existing
        if entity.bound_reference_name == KUBERNETES_POD_UID_REFERENCE_NAME and entity.bound_reference_value
    }
    now = datetime.now(UTC)
    refresh_threshold_seconds = config.workload_identity.token_expiration_seconds / 2
    live_pod_uid_set: set[str] = set()
    for pod in pods:
        pod_uid = _pod_uid(pod)
        if pod_uid is not None and _pod_is_live(pod):
            live_pod_uid_set.add(pod_uid)
    live_pod_uids = sorted(live_pod_uid_set)

    for pod_uid in live_pod_uids:
        current = existing_by_pod_uid.get(pod_uid)
        if current is not None and current.is_active(now=now):
            remaining_seconds = (as_aware_utc(current.expires_at) - now).total_seconds()
            if remaining_seconds > refresh_threshold_seconds:
                continue
            refreshed = current.model_copy(
                update={
                    "expires_at": workload_delegation_expires_at(
                        ttl_seconds_active=config.workload_identity.token_expiration_seconds,
                        now=now,
                    )
                }
            )
            try:
                await store.update(refreshed, expected_db_version=current.db_version)
            except WorkloadDelegationConflictError:
                logger.debug(
                    "Skipping stale Pod UID workload delegation refresh",
                    extra={"delegation_name": current.name},
                )
            continue
        delegation = build_kubernetes_pod_uid_workload_delegation(
            scope=scope,
            workload_audience=get_workload_delegation_audience(),
            workload_generation=pod_uid,
            namespace=namespace,
            service_account_name=service_account_name(config=config, k8s_config=k8s_config),
            pod_uid=pod_uid,
            auth_context=auth_context,
            ttl_seconds_active=config.workload_identity.token_expiration_seconds,
        )
        try:
            await store.register(delegation)
        except WorkloadDelegationConflictError:
            logger.debug("Pod UID workload delegation already active", extra={"delegation_name": delegation.name})

    live_set = set(live_pod_uids)
    for pod_uid, entity in existing_by_pod_uid.items():
        if pod_uid not in live_set and entity.is_active():
            await store.revoke(entity.name)


def deployment_pod_uid_delegation_pods(
    *,
    config: DeploymentConfig,
    k8s_config: K8sDeploymentConfig | None,
    resource_name: str,
    deployment_uid: str | None,
    replica_sets: ReplicaSetOwnerListResult,
    pods: PodListResult,
) -> PodListResult:
    """Return Pods eligible for Deployment-owned Pod UID delegation."""
    if isinstance(pods, PodListUnavailable) or isinstance(replica_sets, PodListUnavailable):
        return POD_LIST_UNAVAILABLE
    if deployment_uid is None:
        logger.debug("Skipping Pod UID workload delegation reconciliation because Deployment UID is unavailable")
        return POD_LIST_UNAVAILABLE
    replica_set_uids_by_name = {
        replica_set.name: replica_set.uid
        for replica_set in replica_sets
        if replica_set.deployment_name == resource_name and replica_set.deployment_uid == deployment_uid
    }
    return _pod_uid_delegation_pods(
        config=config,
        k8s_config=k8s_config,
        controller_kind="ReplicaSet",
        controller_uids_by_name=replica_set_uids_by_name,
        pods=pods,
    )


def job_pod_uid_delegation_pods(
    *,
    config: DeploymentConfig,
    k8s_config: K8sDeploymentConfig | None,
    job_name: str,
    job_uid: str | None,
    pods: PodListResult,
) -> PodListResult:
    """Return Pods eligible for Job-owned Pod UID delegation."""
    if isinstance(pods, PodListUnavailable):
        return POD_LIST_UNAVAILABLE
    if job_uid is None:
        logger.debug("Skipping Pod UID workload delegation reconciliation because Job UID is unavailable")
        return POD_LIST_UNAVAILABLE
    return _pod_uid_delegation_pods(
        config=config,
        k8s_config=k8s_config,
        controller_kind="Job",
        controller_uids_by_name={job_name: job_uid},
        pods=pods,
    )


async def revoke_workload_delegations(
    store: WorkloadDelegationStore | None,
    *,
    config: DeploymentConfig | None,
    workspace: str,
    deployment_name: str,
) -> None:
    """Revoke all delegation rows owned by one physical deployment workload."""
    if store is None:
        return
    del config
    try:
        await store.revoke_by_workload(
            workload_delegation_scope(workspace=workspace, deployment_name=deployment_name, config=None)
        )
    except Exception:
        logger.warning(
            "Failed to revoke Kubernetes workload delegations for %s/%s", workspace, deployment_name, exc_info=True
        )


def _pod_uid_delegation_pods(
    *,
    config: DeploymentConfig,
    k8s_config: K8sDeploymentConfig | None,
    controller_kind: str,
    pods: list[V1Pod],
    controller_uids_by_name: dict[str, str],
) -> list[V1Pod]:
    expected_service_account = service_account_name(config=config, k8s_config=k8s_config)
    return [
        pod
        for pod in pods
        if _pod_has_controller_owner(
            pod,
            controller_kind=controller_kind,
            controller_uids_by_name=controller_uids_by_name,
        )
        and _pod_service_account_name(pod) == expected_service_account
    ]


def _pod_has_controller_owner(
    pod: V1Pod,
    *,
    controller_kind: str,
    controller_uids_by_name: dict[str, str],
) -> bool:
    metadata = pod.metadata
    owner_references = metadata.owner_references if metadata is not None else None
    for owner_reference in owner_references or []:
        if owner_reference.controller is not True:
            continue
        if owner_reference.kind != controller_kind:
            continue
        owner_name = owner_reference.name
        owner_uid = owner_reference.uid
        if owner_name is not None and owner_uid == controller_uids_by_name.get(owner_name):
            return True
    return False


def replica_set_owner_from_resource(replica_set: V1ReplicaSet) -> ReplicaSetOwner | None:
    """Extract the owning Deployment reference from a Kubernetes ReplicaSet."""
    metadata = replica_set.metadata
    replica_set_name = metadata.name if metadata is not None else None
    replica_set_uid = metadata.uid if metadata is not None else None
    if not replica_set_name or not replica_set_uid:
        return None
    owner_references = metadata.owner_references if metadata is not None else None
    for owner_reference in owner_references or []:
        if owner_reference.controller is not True:
            continue
        if owner_reference.kind != "Deployment":
            continue
        deployment_name = owner_reference.name
        deployment_uid = owner_reference.uid
        if deployment_name and deployment_uid:
            return ReplicaSetOwner(
                name=replica_set_name,
                uid=replica_set_uid,
                deployment_name=deployment_name,
                deployment_uid=deployment_uid,
            )
    return None


def _pod_service_account_name(pod: V1Pod) -> str:
    spec = pod.spec
    if spec is None:
        return DEFAULT_SERVICE_ACCOUNT
    return spec.service_account_name or spec.service_account or DEFAULT_SERVICE_ACCOUNT


def _pod_uid(pod: V1Pod) -> str | None:
    metadata = pod.metadata
    if metadata is None:
        return None
    return metadata.uid or None


def _pod_is_live(pod: V1Pod) -> bool:
    metadata = pod.metadata
    if metadata is not None and metadata.deletion_timestamp:
        return False
    status = pod.status
    phase = status.phase if status is not None else None
    return phase not in _TERMINAL_POD_PHASES
