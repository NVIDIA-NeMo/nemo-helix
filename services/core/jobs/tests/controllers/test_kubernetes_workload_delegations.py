# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client
from nhx.common.auth import AuthContext
from nhx.common.auth.workload_delegations import (
    KUBERNETES_POD_UID_REFERENCE_NAME,
    WorkloadDelegationConflictError,
    reference_delegation_name,
)
from nhx.common.entities import SYSTEM_WORKSPACE
from nhx.core.jobs.api.v2.jobs.schemas import PlatformJobStepWithContext
from nhx.core.jobs.app.constants import (
    JOB_ATTEMPT_ID_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_JOBS_CONTROLLER,
    JOB_MANAGED_BY_LABEL,
    JOB_STEP_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_TYPE_JOB,
    JOB_TYPE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
)
from nhx.core.jobs.controllers.backends.kubernetes.workload_delegations import (
    KubernetesPodBoundWorkloadDelegationManager,
    KubernetesPodBoundWorkloadDelegationTarget,
)


@pytest.fixture
def test_step_pending_with_auth_context(test_step_pending: PlatformJobStepWithContext) -> PlatformJobStepWithContext:
    step = test_step_pending.model_copy(deep=True)
    step.auth_context = AuthContext(
        principal_id="creator@example.com",
        principal_email="creator@example.com",
        principal_groups=["engineering", "ml-team"],
    )
    return step


def _target() -> KubernetesPodBoundWorkloadDelegationTarget:
    return KubernetesPodBoundWorkloadDelegationTarget(
        namespace="test-namespace",
        name="job-test-job-test-step",
        labels={
            "app": "nemo-job",
            JOB_TYPE_LABEL: JOB_TYPE_JOB,
            JOB_WORKSPACE_ID_LABEL: "default",
            JOB_ID_LABEL: "test-job-id",
            JOB_ATTEMPT_ID_LABEL: "test-job-attempt-id",
            JOB_STEP_NAME_LABEL: "test-step",
            JOB_STEP_ID_LABEL: "test-step-id",
            JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        },
        service_account_name="default",
    )


def _pod(uid: str) -> client.V1Pod:
    return client.V1Pod(metadata=client.V1ObjectMeta(uid=uid, name=f"pod-{uid}"))


def _manager(
    *,
    core_v1: MagicMock | None = None,
    register: MagicMock | None = None,
    revoke: MagicMock | None = None,
) -> KubernetesPodBoundWorkloadDelegationManager:
    return KubernetesPodBoundWorkloadDelegationManager(
        core_v1=core_v1 or MagicMock(),
        namespace="test-namespace",
        ttl_seconds_active=lambda: 900,
        workload_audience=lambda: "nemo-helix",
        register_workload_delegation=register or MagicMock(),
        revoke_workload_delegation=revoke or MagicMock(),
    )


def _expected_name(bound_reference_value: str = "pod-uid-123") -> str:
    return reference_delegation_name(
        workload_audience="nemo-helix",
        workload_subject="system:serviceaccount:test-namespace:default",
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value=bound_reference_value,
    )


def test_manager_registers_pod_uid_bound_delegation(test_step_pending_with_auth_context):
    core_v1 = MagicMock()
    core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[_pod("pod-uid-123")])
    register = MagicMock()
    manager = _manager(core_v1=core_v1, register=register)

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.ensure_for_target(test_step_pending_with_auth_context, _target())

    delegation = register.call_args.args[0]
    assert delegation.name == _expected_name()
    assert delegation.workspace == SYSTEM_WORKSPACE
    assert delegation.workload_subject == "system:serviceaccount:test-namespace:default"
    assert delegation.workload_audience == "nemo-helix"
    assert delegation.bound_reference_name == KUBERNETES_POD_UID_REFERENCE_NAME
    assert delegation.bound_reference_value == "pod-uid-123"
    assert delegation.workload_workspace == "default"
    assert delegation.workload_kind == "job"
    assert delegation.workload_id == "test-job-id"
    assert delegation.workload_generation == "test-job-attempt-id/test-step-id/pod-uid-123"
    assert delegation.job_id == "test-job-id"
    assert delegation.attempt_id == "test-job-attempt-id"
    assert delegation.step_id == "test-step-id"
    assert delegation.auth_context == AuthContext.model_validate(
        test_step_pending_with_auth_context.auth_context.model_dump(mode="python", exclude_none=True)
    )
    assert delegation.expires_at > datetime.datetime.now(datetime.timezone.utc)


def test_manager_skips_work_when_step_has_no_auth_context(test_step_pending):
    core_v1 = MagicMock()
    register = MagicMock()
    manager = _manager(core_v1=core_v1, register=register)

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.ensure_for_target(test_step_pending, _target())

    core_v1.list_namespaced_pod.assert_not_called()
    register.assert_not_called()


def test_manager_treats_conflict_as_success_and_records_delegation(test_step_pending_with_auth_context):
    core_v1 = MagicMock()
    core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[_pod("pod-uid-123")])
    register = MagicMock(side_effect=WorkloadDelegationConflictError("already exists"))
    manager = _manager(core_v1=core_v1, register=register)
    target = _target()

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.ensure_for_target(test_step_pending_with_auth_context, target)
        manager.ensure_for_target(test_step_pending_with_auth_context, target)

    assert register.call_count == 1
    assert manager._delegations_by_target_key == {f"{target.namespace}/{target.name}": {_expected_name()}}


def test_manager_retries_failed_registration_on_next_ensure(test_step_pending_with_auth_context):
    core_v1 = MagicMock()
    core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[_pod("pod-uid-123")])
    register = MagicMock(side_effect=[RuntimeError("store offline"), None])
    manager = _manager(core_v1=core_v1, register=register)
    target = _target()

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.ensure_for_target(test_step_pending_with_auth_context, target)
        assert _expected_name() not in manager._delegations_by_target_key.get(
            f"{target.namespace}/{target.name}", set()
        )

        manager.ensure_for_target(test_step_pending_with_auth_context, target)

    assert register.call_count == 2
    assert manager._delegations_by_target_key == {f"{target.namespace}/{target.name}": {_expected_name()}}


def test_manager_revokes_recorded_names_without_relisting_pods():
    core_v1 = MagicMock()
    revoke = MagicMock()
    manager = _manager(core_v1=core_v1, revoke=revoke)
    target = _target()
    target_key = f"{target.namespace}/{target.name}"
    manager._delegations_by_target_key[target_key] = {"ref:recorded-delegation"}

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.revoke_for_target(target)

    revoke.assert_called_once_with("ref:recorded-delegation")
    core_v1.list_namespaced_pod.assert_not_called()
    assert target_key not in manager._delegations_by_target_key


def test_manager_derives_revoke_names_from_current_pods_when_no_names_are_recorded():
    core_v1 = MagicMock()
    core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[_pod("pod-uid-123")])
    revoke = MagicMock()
    manager = _manager(core_v1=core_v1, revoke=revoke)

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.revoke_for_target(_target())

    revoke.assert_called_once_with(_expected_name())


def test_manager_preserves_failed_revokes_for_retry():
    revoke = MagicMock(side_effect=[RuntimeError("store offline"), None, None])
    manager = _manager(revoke=revoke)
    target = _target()
    target_key = f"{target.namespace}/{target.name}"
    manager._delegations_by_target_key[target_key] = {"ref:a", "ref:b"}

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.revoke_for_target(target)
        assert manager._delegations_by_target_key == {target_key: {"ref:a"}}

        manager.revoke_by_key(namespace=target.namespace, name=target.name)

    assert revoke.call_count == 3
    assert target_key not in manager._delegations_by_target_key


def test_manager_can_revoke_by_key_after_native_job_object_is_gone():
    core_v1 = MagicMock()
    revoke = MagicMock()
    manager = _manager(core_v1=core_v1, revoke=revoke)
    target = _target()
    target_key = f"{target.namespace}/{target.name}"
    manager._delegations_by_target_key[target_key] = {"ref:recorded-delegation"}

    with patch(
        "nhx.core.jobs.controllers.backends.kubernetes.workload_delegations."
        "is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        manager.revoke_by_key(namespace=target.namespace, name=target.name)

    revoke.assert_called_once_with("ref:recorded-delegation")
    core_v1.list_namespaced_pod.assert_not_called()
    assert target_key not in manager._delegations_by_target_key
