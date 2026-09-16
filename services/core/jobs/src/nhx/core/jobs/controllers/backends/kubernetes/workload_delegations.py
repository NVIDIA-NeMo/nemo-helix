# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from kubernetes import client
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.jobs.types import PlatformJobStepWithContext
from nhx.common.auth import (
    WorkloadDelegationConflictError,
    WorkloadDelegationEntity,
    WorkloadDelegationScope,
    build_kubernetes_pod_uid_workload_delegation,
    is_workload_identity_token_exchange_enabled,
    kubernetes_pod_uid_delegation_name,
    kubernetes_service_account_subject,
)
from nhx.core.jobs.app.constants import (
    JOB_ATTEMPT_ID_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_LABEL,
    JOB_STEP_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_TYPE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
)
from nhx.core.jobs.controllers.backends.exceptions import JobStorageError
from nhx.core.jobs.controllers.backends.kubernetes.common import list_pods_by_labels

logger = logging.getLogger(__name__)

_REQUIRED_POD_SELECTOR_LABELS = (
    JOB_TYPE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
    JOB_ID_LABEL,
    JOB_ATTEMPT_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_STEP_ID_LABEL,
    JOB_MANAGED_BY_LABEL,
)


@dataclass(frozen=True)
class KubernetesPodBoundWorkloadDelegationTarget:
    """Normalized Kubernetes-family job object used for pod-bound delegations."""

    namespace: str
    name: str
    labels: Mapping[str, str]
    service_account_name: str


class KubernetesPodBoundWorkloadDelegationManager:
    """Create and revoke Pod UID-bound workload delegation rows for Kubernetes-family jobs."""

    def __init__(
        self,
        *,
        core_v1: client.CoreV1Api,
        namespace: str,
        ttl_seconds_active: Callable[[], int],
        workload_audience: Callable[[], str],
        register_workload_delegation: Callable[[WorkloadDelegationEntity], object | None],
        revoke_workload_delegation: Callable[[str], object | None],
    ) -> None:
        self._core_v1 = core_v1
        self._namespace = namespace
        self._ttl_seconds_active = ttl_seconds_active
        self._workload_audience = workload_audience
        self._register_workload_delegation = register_workload_delegation
        self._revoke_workload_delegation = revoke_workload_delegation
        self._delegations_by_target_key: dict[str, set[str]] = {}

    def should_manage_step(self, step: PlatformJobStepWithContext) -> bool:
        return is_workload_identity_token_exchange_enabled() and step.auth_context is not None

    def ensure_for_target(
        self,
        step: PlatformJobStepWithContext,
        target: KubernetesPodBoundWorkloadDelegationTarget,
    ) -> None:
        if not self.should_manage_step(step):
            return

        target_key = self._target_key(target.namespace, target.name)
        registered_delegations = self._delegations_by_target_key.setdefault(target_key, set())
        for pod in self._pods_for_target(target):
            delegation_name = self._delegation_name_for_pod(target, pod)
            if delegation_name is None or delegation_name in registered_delegations:
                continue
            delegation = self._build_workload_delegation(
                step=step,
                target=target,
                pod=pod,
                delegation_name=delegation_name,
            )
            try:
                self._register_workload_delegation(delegation)
            except WorkloadDelegationConflictError:
                logger.debug(
                    "Kubernetes workload delegation already exists",
                    extra={"delegation_name": delegation_name},
                )
            except Exception:
                logger.exception(
                    "Failed to register Kubernetes workload delegation; will retry on next sync",
                    extra={"delegation_name": delegation_name, "target_key": target_key},
                )
                continue
            registered_delegations.add(delegation_name)

    def revoke_for_target(self, target: KubernetesPodBoundWorkloadDelegationTarget) -> None:
        if not is_workload_identity_token_exchange_enabled():
            return

        target_key = self._target_key(target.namespace, target.name)
        recorded_delegations = self._delegations_by_target_key.get(target_key, set())
        delegation_names = set(recorded_delegations)
        if not delegation_names:
            for pod in self._pods_for_target(target):
                delegation_name = self._delegation_name_for_pod(target, pod)
                if delegation_name is not None:
                    delegation_names.add(delegation_name)

        self._revoke_names(target_key=target_key, delegation_names=delegation_names)

    def revoke_by_key(self, *, namespace: str, name: str) -> None:
        if not is_workload_identity_token_exchange_enabled():
            return

        target_key = self._target_key(namespace or self._namespace, name)
        self._revoke_names(
            target_key=target_key,
            delegation_names=set(self._delegations_by_target_key.get(target_key, set())),
        )

    @staticmethod
    def _target_key(namespace: str, name: str) -> str:
        return f"{namespace}/{name}"

    @staticmethod
    def _workload_subject_for_target(target: KubernetesPodBoundWorkloadDelegationTarget) -> str:
        return kubernetes_service_account_subject(
            namespace=target.namespace,
            service_account_name=target.service_account_name,
        )

    def _selector_labels(self, target: KubernetesPodBoundWorkloadDelegationTarget) -> dict[str, str] | None:
        selector_labels = {key: target.labels[key] for key in _REQUIRED_POD_SELECTOR_LABELS if key in target.labels}
        if len(selector_labels) != len(_REQUIRED_POD_SELECTOR_LABELS):
            logger.warning(
                "Skipping Kubernetes workload delegation reconciliation for target with missing labels",
                extra={"target_name": target.name, "labels": dict(target.labels)},
            )
            return None
        return selector_labels

    def _pods_for_target(self, target: KubernetesPodBoundWorkloadDelegationTarget) -> list[client.V1Pod]:
        selector_labels = self._selector_labels(target)
        if selector_labels is None:
            return []
        return list_pods_by_labels(self._core_v1, target.namespace, selector_labels)

    def _delegation_name_for_pod(
        self,
        target: KubernetesPodBoundWorkloadDelegationTarget,
        pod: client.V1Pod,
    ) -> str | None:
        pod_uid = getattr(pod.metadata, "uid", None)
        if not pod_uid:
            return None
        return kubernetes_pod_uid_delegation_name(
            workload_audience=self._workload_audience(),
            workload_subject=self._workload_subject_for_target(target),
            pod_uid=pod_uid,
        )

    def _build_workload_delegation(
        self,
        *,
        step: PlatformJobStepWithContext,
        target: KubernetesPodBoundWorkloadDelegationTarget,
        pod: client.V1Pod,
        delegation_name: str,
    ) -> WorkloadDelegationEntity:
        if step.auth_context is None:
            raise JobStorageError(
                "Kubernetes workload identity requires a job auth_context for on-behalf-of delegation"
            )

        pod_uid = getattr(pod.metadata, "uid", None)
        if not pod_uid:
            raise JobStorageError("Kubernetes workload identity requires a Pod UID for on-behalf-of delegation")

        auth_context = AuthContext.model_validate(step.auth_context.model_dump(mode="python", exclude_none=True))
        delegation = build_kubernetes_pod_uid_workload_delegation(
            scope=WorkloadDelegationScope(
                workload_workspace=step.workspace,
                workload_kind="job",
                workload_instance_id=step.job,
            ),
            workload_audience=self._workload_audience(),
            workload_generation=f"{step.attempt_id}/{step.id}/{pod_uid}",
            namespace=target.namespace,
            service_account_name=target.service_account_name,
            pod_uid=pod_uid,
            job_id=step.job,
            attempt_id=step.attempt_id,
            step_id=step.id,
            auth_context=auth_context,
            ttl_seconds_active=self._ttl_seconds_active(),
        )
        if delegation.name != delegation_name:
            raise JobStorageError("Kubernetes workload identity delegation name changed during build")
        return delegation

    def _revoke_names(self, *, target_key: str, delegation_names: set[str]) -> None:
        failed_delegations: set[str] = set()
        for delegation_name in sorted(delegation_names):
            try:
                self._revoke_workload_delegation(delegation_name)
            except Exception:
                logger.exception(
                    "Failed to revoke Kubernetes workload delegation; will retry on next sync",
                    extra={"delegation_name": delegation_name, "target_key": target_key},
                )
                failed_delegations.add(delegation_name)

        if failed_delegations:
            self._delegations_by_target_key[target_key] = failed_delegations
        else:
            self._delegations_by_target_key.pop(target_key, None)
