# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from backends.k8s.k8s_helpers import (
    job_identity_labels,
    job_list_item,
    live_pod,
    mock_job,
    sample_config,
    sample_deployment,
    with_workload_identity,
    workload_auth_context,
)
from kubernetes.client.rest import ApiException
from nemo_deployments_plugin.backends.k8s import jobs as job_ops
from nemo_deployments_plugin.backends.k8s.client import KubernetesClients
from nemo_deployments_plugin.backends.k8s.compiler import ExecutorK8sDefaults
from nemo_deployments_plugin.backends.k8s.jobs import (
    job_backoff_limit,
    resolve_job_ttl_seconds_after_finished,
    trim_log_text,
    validate_config_for_job,
)
from nemo_deployments_plugin.backends.labels import (
    DEPLOYMENT_WORKSPACE_LABEL,
    MANAGED_BY_KEY,
    deployment_key,
    k8s_deployment_resource_name,
    k8s_deployment_secret_name,
)
from nemo_deployments_plugin.constants import MANAGED_BY_LABEL
from nemo_deployments_plugin.entities import DeploymentConfig, EnvVar, SecretRef
from nemo_deployments_plugin.types import RestartPolicy
from nemo_helix_plugin.auth.workload_delegations import WorkloadDelegationScope
from nemo_helix_plugin.auth.workload_identity import build_kubernetes_pod_uid_workload_delegation
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError


@pytest.fixture
def job_ops_clients(mock_k8s_clients: MagicMock) -> MagicMock:
    clients = MagicMock(spec=KubernetesClients)
    clients.batch_v1 = mock_k8s_clients.batch_v1
    clients.core_v1 = mock_k8s_clients.core_v1
    clients.request_timeout = mock_k8s_clients.request_timeout
    return clients


@pytest.mark.parametrize(
    ("restart_policy", "expected"),
    [
        ("Never", 0),
        ("OnFailure", 6),
    ],
)
def test_job_backoff_limit(restart_policy: RestartPolicy, expected: int) -> None:
    config = sample_config(restart_policy=restart_policy)
    assert job_backoff_limit(config) == expected


def test_validate_config_for_job_rejects_always() -> None:
    with pytest.raises(job_ops.DeploymentConfigError, match="Always"):
        validate_config_for_job(sample_config(restart_policy="Always"))


def test_resolve_job_ttl_prefers_per_entity_override() -> None:
    from nemo_deployments_plugin.entities import K8sDeploymentConfig

    assert (
        resolve_job_ttl_seconds_after_finished(
            k8s_config=K8sDeploymentConfig(jobTtlSecondsAfterFinished=15),
            executor_defaults=ExecutorK8sDefaults(job_ttl_seconds_after_finished=60),
        )
        == 15
    )


def test_resolve_job_ttl_falls_back_to_executor_default() -> None:
    from nemo_deployments_plugin.entities import K8sDeploymentConfig

    assert (
        resolve_job_ttl_seconds_after_finished(
            k8s_config=K8sDeploymentConfig(),
            executor_defaults=ExecutorK8sDefaults(job_ttl_seconds_after_finished=90),
        )
        == 90
    )


def test_resolve_job_ttl_none_when_unset_everywhere() -> None:
    assert (
        resolve_job_ttl_seconds_after_finished(
            k8s_config=None,
            executor_defaults=ExecutorK8sDefaults(job_ttl_seconds_after_finished=None),
        )
        is None
    )
    assert resolve_job_ttl_seconds_after_finished(k8s_config=None, executor_defaults=None) is None


def test_entity_job_ttl_rejects_below_floor() -> None:
    # A per-deployment TTL below the floor races the reconciler's completion read; rejected.
    from nemo_deployments_plugin.entities import K8sDeploymentConfig

    for bad in (0, 5, 9):
        with pytest.raises(ValueError):
            K8sDeploymentConfig(jobTtlSecondsAfterFinished=bad)


def test_entity_job_ttl_accepts_floor_and_none() -> None:
    from nemo_deployments_plugin.entities import K8sDeploymentConfig

    assert K8sDeploymentConfig(jobTtlSecondsAfterFinished=10).job_ttl_seconds_after_finished == 10
    assert K8sDeploymentConfig(jobTtlSecondsAfterFinished=None).job_ttl_seconds_after_finished is None


@pytest.mark.asyncio
async def test_create_job_sets_default_ttl_seconds_after_finished(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_k8s_clients.batch_v1.create_namespaced_job.return_value = mock_job(active=1)

    await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
    )

    body = mock_k8s_clients.batch_v1.create_namespaced_job.call_args.kwargs["body"]
    assert body.spec.ttl_seconds_after_finished == 60


@pytest.mark.asyncio
async def test_create_job_per_entity_ttl_override_wins(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    mock_k8s_clients.batch_v1.create_namespaced_job.return_value = mock_job(active=1)

    await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={"k8s": {"jobTtlSecondsAfterFinished": 15}},
    )

    body = mock_k8s_clients.batch_v1.create_namespaced_job.call_args.kwargs["body"]
    assert body.spec.ttl_seconds_after_finished == 15


@pytest.mark.asyncio
async def test_create_job_on_failure_uses_requested_restart_policy(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="OnFailure")
    mock_k8s_clients.batch_v1.create_namespaced_job.return_value = mock_job(restart_policy="OnFailure", active=1)

    await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
    )

    body = mock_k8s_clients.batch_v1.create_namespaced_job.call_args.kwargs["body"]
    assert body.spec.template.spec.restart_policy == "OnFailure"
    assert body.spec.backoff_limit == 6


@pytest.mark.asyncio
async def test_create_job_emits_separate_command_and_args(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    config = sample_config(restart_policy="Never")
    config.containers[0].command = ["echo"]
    config.containers[0].args = ["hello"]
    mock_entities.get.return_value = config
    mock_k8s_clients.batch_v1.create_namespaced_job.return_value = mock_job(active=1)

    await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
    )

    container = mock_k8s_clients.batch_v1.create_namespaced_job.call_args.kwargs["body"].spec.template.spec.containers[
        0
    ]
    assert container.command == ["echo"]
    assert container.args == ["hello"]


@pytest.mark.asyncio
async def test_create_job_emits_batch_job(k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    pending = mock_job(active=1)
    mock_k8s_clients.batch_v1.create_namespaced_job.return_value = pending

    update = await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={"managed-by": MANAGED_BY_LABEL},
        backend_config={},
    )

    assert update.status == "STARTING"
    call = mock_k8s_clients.batch_v1.create_namespaced_job
    call.assert_called_once()
    body = call.call_args.kwargs["body"]
    assert body.spec.backoff_limit == 0
    assert body.spec.template.spec.restart_policy == "Never"
    assert body.spec.template.spec.containers[0].image == "alpine:latest"


@pytest.mark.asyncio
async def test_create_job_registers_live_pod_uid_workload_delegation(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    auth_context = workload_auth_context()
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.batch_v1.create_namespaced_job.return_value = mock_job(active=1)
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[
            live_pod(
                "job-pod-uid",
                owner_kind="Job",
                owner_name=k8s_deployment_resource_name("default", "task"),
                service_account_name="dep-sa",
            )
        ]
    )

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.create_job(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            config_name="config1",
            labels={},
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config=config,
            auth_context=auth_context,
            workload_delegation_store=workload_store,
        )

    assert update.status == "STARTING"
    workload_store.list_by_workload.assert_awaited_once_with(
        WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="agent_deployment",
            workload_instance_id=deployment_key("default", "task"),
            workload_claim_id="logical-task",
        )
    )
    workload_store.register.assert_awaited_once()
    await_args = workload_store.register.await_args
    assert await_args is not None
    delegation = await_args.args[0]
    assert delegation.workload_id == deployment_key("default", "task")
    assert delegation.workload_claim_id == "logical-task"
    assert delegation.workload_generation == "job-pod-uid"
    assert delegation.workload_subject == "system:serviceaccount:dep-ns:dep-sa"
    assert delegation.bound_reference_value == "job-pod-uid"
    assert delegation.auth_context == auth_context


@pytest.mark.asyncio
async def test_create_job_fails_when_workload_identity_auth_context_missing(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.create_job(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            config_name="config1",
            labels={},
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config=config,
            auth_context=None,
            workload_delegation_store=workload_store,
        )

    assert update.status == "FAILED"
    assert "auth_context" in update.status_message
    mock_k8s_clients.batch_v1.create_namespaced_job.assert_not_called()
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_job_status_ignores_spoofed_pod_uid_workload_delegation(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(active=1, backoff_limit=0)
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[
            live_pod(
                "job-pod-uid",
                owner_kind="Job",
                owner_name="foreign-job",
                service_account_name="dep-sa",
            )
        ]
    )

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.read_job_status(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Never",
            backoff_limit=0,
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status == "STARTING"
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_job_status_ignores_pod_with_matching_job_name_wrong_uid(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    job_name = k8s_deployment_resource_name("default", "task")
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(active=1, backoff_limit=0)
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[
            live_pod(
                "job-pod-uid",
                owner_kind="Job",
                owner_name=job_name,
                owner_uid="spoofed-job-uid",
                service_account_name="dep-sa",
            )
        ]
    )

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.read_job_status(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Never",
            backoff_limit=0,
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status == "STARTING"
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_job_status_skips_workload_delegation_for_foreign_identity_labels(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    foreign = mock_job(active=1, backoff_limit=0)
    foreign.metadata.labels = {**job_identity_labels(backoff_limit=0), DEPLOYMENT_WORKSPACE_LABEL: "other-workspace"}
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = foreign

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.read_job_status(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Never",
            backoff_limit=0,
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status == "FAILED"
    assert "not managed" in update.status_message
    mock_k8s_clients.core_v1.list_namespaced_pod.assert_not_called()
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_job_status_revokes_stale_pod_uid_when_pod_list_is_confirmed_empty(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    auth_context = workload_auth_context()
    stale = build_kubernetes_pod_uid_workload_delegation(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="agent_deployment",
            workload_instance_id=deployment_key("default", "task"),
            workload_claim_id="logical-task",
        ),
        workload_audience="nemo-helix",
        workload_generation="old-pod",
        namespace="dep-ns",
        service_account_name="dep-sa",
        pod_uid="old-pod",
        auth_context=auth_context,
        ttl_seconds_active=900,
    )
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[stale])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(active=1, backoff_limit=0)
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.read_job_status(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Never",
            backoff_limit=0,
            config=config,
            auth_context=auth_context,
            workload_delegation_store=workload_store,
        )

    assert update.status == "STARTING"
    workload_store.register.assert_not_awaited()
    workload_store.revoke.assert_awaited_once_with(stale.name)


@pytest.mark.asyncio
async def test_read_job_status_keeps_delegation_when_pod_list_fails(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    auth_context = workload_auth_context()
    stale = build_kubernetes_pod_uid_workload_delegation(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="agent_deployment",
            workload_instance_id=deployment_key("default", "task"),
            workload_claim_id="logical-task",
        ),
        workload_audience="nemo-helix",
        workload_generation="old-pod",
        namespace="dep-ns",
        service_account_name="dep-sa",
        pod_uid="old-pod",
        auth_context=auth_context,
        ttl_seconds_active=900,
    )
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[stale])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(active=1, backoff_limit=0)
    mock_k8s_clients.core_v1.list_namespaced_pod.side_effect = RuntimeError("apiserver unavailable")

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.read_job_status(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Never",
            backoff_limit=0,
            config=config,
            auth_context=auth_context,
            workload_delegation_store=workload_store,
        )

    assert update.status == "STARTING"
    workload_store.list_by_workload.assert_not_awaited()
    workload_store.register.assert_not_awaited()
    workload_store.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_job_status_with_missing_auth_context_keeps_running_status(
    job_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_config(restart_policy="Never"))
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(active=1, backoff_limit=0)

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await job_ops.read_job_status(
            job_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Never",
            backoff_limit=0,
            config=config,
            auth_context=None,
            workload_delegation_store=workload_store,
        )

    assert update.status == "STARTING"
    mock_k8s_clients.core_v1.list_namespaced_pod.assert_not_called()
    workload_store.list_by_workload.assert_not_awaited()
    workload_store.register.assert_not_awaited()
    workload_store.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_job_conflict_reads_existing(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = sample_config(restart_policy="Never")
    existing = mock_job(complete=True)
    mock_k8s_clients.batch_v1.create_namespaced_job.side_effect = ApiException(status=409)
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = existing

    update = await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
    )

    assert update.status == "SUCCEEDED"
    mock_k8s_clients.batch_v1.read_namespaced_job.assert_called_once()


@pytest.mark.asyncio
async def test_create_job_conflict_rejects_foreign(job_ops_clients: MagicMock, mock_k8s_clients: MagicMock) -> None:
    foreign = mock_job(complete=True)
    foreign.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    mock_k8s_clients.batch_v1.create_namespaced_job.side_effect = ApiException(status=409)
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = foreign

    update = await job_ops.create_job(
        job_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=sample_config(restart_policy="Never"),
    )

    assert update.status == "FAILED"
    assert "not managed" in update.status_message


@pytest.mark.asyncio
async def test_read_job_status_rejects_stale_identity(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.side_effect = [sample_deployment(), sample_config(restart_policy="Never")]
    stale = mock_job(config_name="old-config", complete=True)
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = stale

    update = await k8s_backend.read_status(workspace="default", name="task")

    assert update.status == "FAILED"
    assert "not managed" in update.status_message


@pytest.mark.asyncio
async def test_read_job_status_complete(k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock) -> None:
    mock_entities.get.side_effect = [sample_deployment(), sample_config(restart_policy="Never")]
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(complete=True)

    update = await k8s_backend.read_status(workspace="default", name="task")

    assert update.status == "SUCCEEDED"


@pytest.mark.asyncio
async def test_delete_job_missing_is_success(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.side_effect = [sample_deployment(), sample_config(restart_policy="Never")]
    mock_k8s_clients.batch_v1.read_namespaced_job.side_effect = ApiException(status=404)

    update = await k8s_backend.delete_deployment("default", "task")

    assert update.status == "SUCCEEDED"
    mock_k8s_clients.batch_v1.delete_namespaced_job.assert_not_called()


@pytest.mark.asyncio
async def test_delete_job_rejects_foreign(job_ops_clients: MagicMock, mock_k8s_clients: MagicMock) -> None:
    foreign = mock_job(complete=True)
    foreign.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = foreign

    update = await job_ops.delete_job(
        job_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        backend_config={},
        expected_labels=job_identity_labels(),
    )

    assert update.status == "FAILED"
    mock_k8s_clients.batch_v1.delete_namespaced_job.assert_not_called()


def _job_config_with_secret_env() -> DeploymentConfig:
    config = sample_config(restart_policy="Never")
    config.containers[0].env = [EnvVar(name="APP_TOKEN", secretRef=SecretRef(workspace="default", name="app-token"))]
    return config


@pytest.mark.asyncio
async def test_create_job_creates_managed_secret(job_ops_clients: MagicMock, mock_k8s_clients: MagicMock) -> None:
    config = _job_config_with_secret_env()
    mock_k8s_clients.batch_v1.create_namespaced_job.return_value = mock_job(active=1)

    await job_ops.create_job(
        job_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=config,
        secret_env={"APP_TOKEN": "app-token-value"},
    )

    mock_k8s_clients.core_v1.create_namespaced_secret.assert_called_once()
    secret_body = mock_k8s_clients.core_v1.create_namespaced_secret.call_args.kwargs["body"]
    assert secret_body.metadata.name == k8s_deployment_secret_name("default", "task")
    assert secret_body.string_data == {"APP_TOKEN": "app-token-value"}
    container = mock_k8s_clients.batch_v1.create_namespaced_job.call_args.kwargs["body"].spec.template.spec.containers[
        0
    ]
    assert container.env_from[0].secret_ref.name == k8s_deployment_secret_name("default", "task")


@pytest.mark.asyncio
async def test_delete_job_removes_managed_secret(job_ops_clients: MagicMock, mock_k8s_clients: MagicMock) -> None:
    identity_labels = job_identity_labels()
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(complete=True)
    mock_k8s_clients.core_v1.read_namespaced_config_map.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=identity_labels),
    )
    mock_k8s_clients.core_v1.read_namespaced_secret.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=identity_labels),
    )

    update = await job_ops.delete_job(
        job_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        backend_config={},
        expected_labels=identity_labels,
    )

    assert update.status == "SUCCEEDED"
    mock_k8s_clients.core_v1.delete_namespaced_secret.assert_called_once_with(
        name=k8s_deployment_secret_name("default", "task"),
        namespace="default",
        _request_timeout=mock_k8s_clients.request_timeout,
    )


@pytest.mark.asyncio
async def test_list_managed_deployment_names(k8s_backend, mock_k8s_clients: MagicMock) -> None:
    listed = MagicMock()
    listed.items = [job_list_item(workspace="default", name="task"), job_list_item(workspace="ws", name="other")]
    mock_k8s_clients.batch_v1.list_namespaced_job.return_value = listed

    names = await k8s_backend.list_managed_deployment_names()

    assert names == ["default/task", "ws/other"]


def test_trim_log_text_caps_payload() -> None:
    text = "x" * 9000
    lines, truncated = trim_log_text(text)
    assert truncated is True
    assert sum(len(line) for line in lines) <= 8000


@pytest.mark.asyncio
async def test_get_job_logs(k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock) -> None:
    mock_entities.get.side_effect = [sample_deployment(), sample_config(restart_policy="Never")]
    pod = MagicMock()
    pod.metadata.name = "task-pod"
    pod.metadata.creation_timestamp = "2026-01-02T00:00:00Z"
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[pod])
    mock_k8s_clients.core_v1.read_namespaced_pod_log.return_value = "hello\nworld\n"

    result = await k8s_backend.get_logs(workspace="default", name="task", tail=10)

    assert result.lines == ["hello", "world"]


@pytest.mark.asyncio
async def test_read_job_status_includes_exit_code(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.side_effect = [sample_deployment(), sample_config(restart_policy="Never")]
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(complete=True)
    pod = MagicMock()
    pod.metadata.name = "task-pod"
    pod.metadata.creation_timestamp = "2026-01-02T00:00:00Z"
    pod.status.container_statuses = [MagicMock()]
    pod.status.container_statuses[0].state.terminated.exit_code = 0
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[pod])

    update = await k8s_backend.read_status(workspace="default", name="task")

    assert update.status == "SUCCEEDED"
    assert update.exit_code == 0


@pytest.mark.asyncio
async def test_create_job_malformed_backend_config_returns_failed(job_ops_clients: MagicMock) -> None:
    update = await job_ops.create_job(
        job_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={"k8s": {"namespace": 42}},
        config=sample_config(restart_policy="Never"),
    )

    assert update.status == "FAILED"
    job_ops_clients.batch_v1.create_namespaced_job.assert_not_called()


@pytest.mark.asyncio
async def test_delete_deployment_still_deletes_job_when_entity_missing(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.side_effect = NemoEntityNotFoundError("missing")
    mock_k8s_clients.apps_v1.read_namespaced_deployment.side_effect = ApiException(status=404)
    mock_k8s_clients.batch_v1.read_namespaced_job.return_value = mock_job(complete=True)

    with patch.object(
        k8s_backend._workload_identity_ops,
        "revoke_workload_delegations",
        AsyncMock(),
    ) as revoke_mock:
        update = await k8s_backend.delete_deployment("default", "task")

    assert update.status == "SUCCEEDED"
    mock_k8s_clients.batch_v1.delete_namespaced_job.assert_called_once()
    revoke_mock.assert_awaited_once_with(
        k8s_backend._workload_delegations,
        config=None,
        workspace="default",
        deployment_name="task",
    )
