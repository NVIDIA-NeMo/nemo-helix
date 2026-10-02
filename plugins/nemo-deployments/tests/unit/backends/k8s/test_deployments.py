# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from backends.k8s.k8s_helpers import (
    always_identity_labels,
    deployment_list_item,
    live_pod,
    mock_deployment,
    mock_pod,
    mock_replicaset,
    sample_always_config,
    sample_deployment,
    with_workload_identity,
    workload_auth_context,
)
from kubernetes.client.models import V1ReplicaSet
from kubernetes.client.rest import ApiException
from nemo_deployments_plugin.backends.k8s import deployments as deployment_ops
from nemo_deployments_plugin.backends.k8s.client import KubernetesClients
from nemo_deployments_plugin.backends.k8s.compiler import validate_config_for_deployment
from nemo_deployments_plugin.backends.k8s.deployments import (
    build_in_cluster_endpoints,
)
from nemo_deployments_plugin.backends.k8s.workload_identity import reconcile_pod_uid_delegations
from nemo_deployments_plugin.backends.labels import (
    DEPLOYMENT_WORKSPACE_LABEL,
    MANAGED_BY_KEY,
    deployment_key,
    k8s_deployment_configmap_name,
    k8s_deployment_resource_name,
    k8s_deployment_secret_name,
)
from nemo_deployments_plugin.entities import (
    ConfigFile,
    Container,
    ContainerPort,
    DeploymentConfig,
    EnvVar,
    SecretRef,
)
from nemo_helix_plugin.auth.workload_delegations import KUBERNETES_POD_UID_REFERENCE_NAME, WorkloadDelegationScope
from nemo_helix_plugin.auth.workload_identity import build_kubernetes_pod_uid_workload_delegation
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError


@pytest.fixture
def always_deployment_context(mock_entities: AsyncMock) -> None:
    mock_entities.get.side_effect = [sample_deployment(), sample_always_config()]


@pytest.fixture
def deployment_ops_clients(mock_k8s_clients: MagicMock) -> MagicMock:
    clients = MagicMock(spec=KubernetesClients)
    clients.apps_v1 = mock_k8s_clients.apps_v1
    clients.core_v1 = mock_k8s_clients.core_v1
    clients.request_timeout = mock_k8s_clients.request_timeout
    return clients


def test_validate_config_for_deployment_rejects_never() -> None:
    from nemo_deployments_plugin.backends.k8s.compiler import DeploymentConfigError

    with pytest.raises(DeploymentConfigError, match="Always"):
        validate_config_for_deployment(sample_always_config().model_copy(update={"restart_policy": "Never"}))


def test_build_in_cluster_endpoints_uses_cluster_dns() -> None:
    resource_name = k8s_deployment_resource_name("default", "task")
    endpoints = build_in_cluster_endpoints(
        resource_name=resource_name,
        namespace="nemo-deployments",
        containers=tuple(sample_always_config().containers),
    )
    assert endpoints[0].url == f"http://{resource_name}.nemo-deployments.svc.cluster.local:8080"


def test_build_in_cluster_endpoints_uses_tcp_scheme_for_udp() -> None:
    resource_name = k8s_deployment_resource_name("default", "task")
    container = (
        sample_always_config()
        .containers[0]
        .model_copy(update={"ports": [ContainerPort(name="metrics", containerPort=9090, protocol="UDP")]})
    )
    endpoints = build_in_cluster_endpoints(
        resource_name=resource_name,
        namespace="nemo-deployments",
        containers=(container,),
    )
    assert endpoints[0].protocol == "tcp"
    assert endpoints[0].url == f"tcp://{resource_name}.nemo-deployments.svc.cluster.local:9090"


def _deployment_replicaset_name() -> str:
    return f"{k8s_deployment_resource_name('default', 'task')}-abc123"


def _deployment_replicaset() -> V1ReplicaSet:
    return mock_replicaset(
        name=_deployment_replicaset_name(),
        deployment_name=k8s_deployment_resource_name("default", "task"),
    )


@pytest.mark.asyncio
async def test_create_deployment_emits_deployment_and_service(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = sample_always_config()
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
    )

    deployment_body = mock_k8s_clients.apps_v1.create_namespaced_deployment.call_args.kwargs["body"]
    service_body = mock_k8s_clients.core_v1.create_namespaced_service.call_args.kwargs["body"]
    assert deployment_body.kind == "Deployment"
    assert deployment_body.spec.replicas == 1
    assert deployment_body.spec.selector.match_labels["app"] == deployment_body.metadata.name
    assert service_body.spec.type == "ClusterIP"
    assert service_body.spec.selector["app"] == service_body.metadata.name


@pytest.mark.asyncio
async def test_create_deployment_ready_reports_endpoints(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.return_value = sample_always_config()
    ready = mock_deployment(ready_replicas=1)
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = ready
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = ready
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    update = await k8s_backend.create_deployment(
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
    )

    assert update.status == "READY"
    assert update.endpoints
    assert "svc.cluster.local" in update.endpoints[0].url


@pytest.mark.asyncio
async def test_create_deployment_registers_live_pod_uid_workload_delegation(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    auth_context = workload_auth_context()
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    pod = live_pod(
        "pod-uid-1",
        owner_kind="ReplicaSet",
        owner_name=_deployment_replicaset_name(),
        service_account_name="dep-sa",
    )
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(items=[_deployment_replicaset()])
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[pod])

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.create_deployment(
            deployment_ops_clients,
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

    assert update.status != "FAILED"
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
    assert delegation.workload_workspace == "default"
    assert delegation.workload_kind == "agent_deployment"
    assert delegation.workload_id == deployment_key("default", "task")
    assert delegation.workload_claim_id == "logical-task"
    assert delegation.workload_generation == "pod-uid-1"
    assert delegation.workload_subject == "system:serviceaccount:dep-ns:dep-sa"
    assert delegation.bound_reference_name == KUBERNETES_POD_UID_REFERENCE_NAME
    assert delegation.bound_reference_value == "pod-uid-1"
    assert delegation.auth_context == auth_context


@pytest.mark.asyncio
async def test_create_deployment_fails_when_workload_identity_auth_context_missing(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.create_deployment(
            deployment_ops_clients,
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
    mock_k8s_clients.apps_v1.create_namespaced_deployment.assert_not_called()
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_pod_uid_delegations_scopes_shared_workload_id_by_deployment() -> None:
    config = with_workload_identity(sample_always_config(), workload_id="shared-logical")
    auth_context = workload_auth_context()
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        await reconcile_pod_uid_delegations(
            workload_store,
            config=config,
            auth_context=auth_context,
            workspace="default",
            deployment_name="task-a",
            namespace="dep-ns",
            k8s_config=None,
            pods=[live_pod("pod-a", service_account_name="dep-sa")],
        )
        await reconcile_pod_uid_delegations(
            workload_store,
            config=config,
            auth_context=auth_context,
            workspace="default",
            deployment_name="task-b",
            namespace="dep-ns",
            k8s_config=None,
            pods=[live_pod("pod-b", service_account_name="dep-sa")],
        )

    assert [call.args[0].workload_instance_id for call in workload_store.list_by_workload.await_args_list] == [
        deployment_key("default", "task-a"),
        deployment_key("default", "task-b"),
    ]
    assert [call.args[0].workload_claim_id for call in workload_store.list_by_workload.await_args_list] == [
        "shared-logical",
        "shared-logical",
    ]
    delegations = [call.args[0] for call in workload_store.register.await_args_list]
    assert [delegation.workload_id for delegation in delegations] == [
        deployment_key("default", "task-a"),
        deployment_key("default", "task-b"),
    ]
    assert [delegation.workload_claim_id for delegation in delegations] == [
        "shared-logical",
        "shared-logical",
    ]
    assert delegations[0].name != delegations[1].name
    workload_store.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_deployment_ignores_spoofed_pod_uid_workload_delegation(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    pod = live_pod(
        "pod-uid-1",
        owner_kind="ReplicaSet",
        owner_name="foreign-replicaset",
        service_account_name="dep-sa",
    )
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(items=[])
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[pod])

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.create_deployment(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            config_name="config1",
            labels={},
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_deployment_ignores_pod_with_matching_replicaset_name_wrong_uid(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    pod = live_pod(
        "pod-uid-1",
        owner_kind="ReplicaSet",
        owner_name=_deployment_replicaset_name(),
        owner_uid="spoofed-replicaset-uid",
        service_account_name="dep-sa",
    )
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(items=[_deployment_replicaset()])
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[pod])

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.create_deployment(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            config_name="config1",
            labels={},
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_deployment_ignores_matching_prefix_replicaset_not_owned_by_deployment(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    pod = live_pod(
        "pod-uid-1",
        owner_kind="ReplicaSet",
        owner_name=_deployment_replicaset_name(),
        service_account_name="dep-sa",
    )
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(
        items=[
            mock_replicaset(
                name=_deployment_replicaset_name(),
                deployment_name=k8s_deployment_resource_name("default", "task"),
                deployment_uid="other-deployment-uid",
            )
        ]
    )
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[pod])

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.create_deployment(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            config_name="config1",
            labels={},
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_deployment_status_skips_workload_delegation_for_foreign_identity_labels(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    foreign = mock_deployment()
    foreign.metadata.labels = {**always_identity_labels(), DEPLOYMENT_WORKSPACE_LABEL: "other-workspace"}
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = foreign

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.read_deployment_status(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Always",
            backoff_limit=6,
            containers=tuple(config.containers),
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status == "FAILED"
    assert "not managed" in update.status_message
    mock_k8s_clients.core_v1.list_namespaced_pod.assert_not_called()
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_deployment_status_revokes_stale_pod_uid_workload_delegation(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
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
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(items=[_deployment_replicaset()])
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[
            live_pod(
                "new-pod",
                owner_kind="ReplicaSet",
                owner_name=_deployment_replicaset_name(),
                service_account_name="dep-sa",
            )
        ]
    )

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.read_deployment_status(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Always",
            backoff_limit=6,
            containers=tuple(config.containers),
            config=config,
            auth_context=auth_context,
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.register.assert_awaited_once()
    workload_store.revoke.assert_awaited_once_with(stale.name)


@pytest.mark.asyncio
async def test_read_deployment_status_refreshes_active_pod_uid_before_expiry(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    auth_context = workload_auth_context()
    active = build_kubernetes_pod_uid_workload_delegation(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="agent_deployment",
            workload_instance_id=deployment_key("default", "task"),
            workload_claim_id="logical-task",
        ),
        workload_audience="nemo-helix",
        workload_generation="pod-uid-1",
        namespace="dep-ns",
        service_account_name="dep-sa",
        pod_uid="pod-uid-1",
        auth_context=auth_context,
        ttl_seconds_active=900,
    ).model_copy(update={"expires_at": datetime.now(timezone.utc) + timedelta(seconds=60)})
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[active])
    workload_store.register = AsyncMock()
    workload_store.update = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(items=[_deployment_replicaset()])
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[
            live_pod(
                "pod-uid-1",
                owner_kind="ReplicaSet",
                owner_name=_deployment_replicaset_name(),
                service_account_name="dep-sa",
            )
        ]
    )

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.read_deployment_status(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Always",
            backoff_limit=6,
            containers=tuple(config.containers),
            config=config,
            auth_context=auth_context,
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.register.assert_not_awaited()
    workload_store.revoke.assert_not_awaited()
    workload_store.update.assert_awaited_once()
    await_args = workload_store.update.await_args
    assert await_args is not None
    refreshed = await_args.args[0]
    assert refreshed.name == active.name
    assert refreshed.expires_at > active.expires_at
    assert await_args.kwargs == {"expected_db_version": active.db_version}


@pytest.mark.asyncio
async def test_read_deployment_status_revokes_stale_pod_uid_when_pod_list_is_confirmed_empty(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
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
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(items=[])
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.read_deployment_status(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Always",
            backoff_limit=6,
            containers=tuple(config.containers),
            config=config,
            auth_context=auth_context,
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.register.assert_not_awaited()
    workload_store.revoke.assert_awaited_once_with(stale.name)


@pytest.mark.asyncio
async def test_read_deployment_status_keeps_delegation_when_pod_list_fails(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
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
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.list_namespaced_pod.side_effect = RuntimeError("apiserver unavailable")

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.read_deployment_status(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Always",
            backoff_limit=6,
            containers=tuple(config.containers),
            config=config,
            auth_context=auth_context,
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.list_by_workload.assert_not_awaited()
    workload_store.register.assert_not_awaited()
    workload_store.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_deployment_status_with_missing_auth_context_keeps_ready_status(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment(ready_replicas=1)
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.read_deployment_status(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Always",
            backoff_limit=6,
            containers=tuple(config.containers),
            config=config,
            auth_context=None,
            workload_delegation_store=workload_store,
        )

    assert update.status == "READY"
    workload_store.list_by_workload.assert_not_awaited()
    workload_store.register.assert_not_awaited()
    workload_store.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_deployment_status_ignores_wrong_service_account_pod_uid_workload_delegation(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = with_workload_identity(sample_always_config())
    workload_store = MagicMock()
    workload_store.list_by_workload = AsyncMock(return_value=[])
    workload_store.register = AsyncMock()
    workload_store.revoke = AsyncMock()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.list_namespaced_replica_set.return_value = MagicMock(items=[_deployment_replicaset()])
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[
            live_pod(
                "pod-uid-1",
                owner_kind="ReplicaSet",
                owner_name=_deployment_replicaset_name(),
                service_account_name="other-sa",
            )
        ]
    )

    with patch(
        "nemo_deployments_plugin.backends.workload_identity.is_workload_identity_token_exchange_enabled",
        return_value=True,
    ):
        update = await deployment_ops.read_deployment_status(
            deployment_ops_clients,
            default_namespace="default",
            workspace="default",
            name="task",
            backend_config={"k8s": {"namespace": "dep-ns"}},
            config_name="config1",
            restart_policy="Always",
            backoff_limit=6,
            containers=tuple(config.containers),
            config=config,
            auth_context=workload_auth_context(),
            workload_delegation_store=workload_store,
        )

    assert update.status != "FAILED"
    workload_store.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_deployment_conflict_rejects_foreign(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    foreign = mock_deployment()
    foreign.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    mock_k8s_clients.apps_v1.create_namespaced_deployment.side_effect = ApiException(status=409)
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = foreign

    update = await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=sample_always_config(),
    )

    assert update.status == "FAILED"
    assert "not managed" in update.status_message
    mock_k8s_clients.core_v1.create_namespaced_service.assert_not_called()


@pytest.mark.asyncio
async def test_create_deployment_service_conflict_rejects_foreign(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    foreign_service = MagicMock()
    foreign_service.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    mock_k8s_clients.core_v1.create_namespaced_service.side_effect = ApiException(status=409)
    mock_k8s_clients.core_v1.read_namespaced_service.return_value = foreign_service

    update = await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=sample_always_config(),
    )

    assert update.status == "FAILED"
    mock_k8s_clients.apps_v1.delete_namespaced_deployment.assert_called_once()


@pytest.mark.asyncio
async def test_create_deployment_rolls_back_when_service_create_fails(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.create_namespaced_service.side_effect = ApiException(status=500)

    update = await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=sample_always_config(),
    )

    assert update.status == "FAILED"
    mock_k8s_clients.apps_v1.delete_namespaced_deployment.assert_called_once()


@pytest.mark.asyncio
async def test_create_deployment_rolls_back_configmap_when_service_create_fails(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = sample_always_config().model_copy(
        update={"config_files": [ConfigFile(path="/etc/app/config.yaml", content="key: value")]}
    )
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.create_namespaced_service.side_effect = ApiException(status=500)
    identity_labels = always_identity_labels(backoff_limit=config.backoff_limit)
    mock_k8s_clients.core_v1.read_namespaced_config_map.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=identity_labels),
    )

    update = await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=config,
    )

    assert update.status == "FAILED"
    mock_k8s_clients.core_v1.create_namespaced_config_map.assert_called_once()
    mock_k8s_clients.core_v1.read_namespaced_config_map.assert_called_once_with(
        name=k8s_deployment_configmap_name("default", "task"),
        namespace="default",
        _request_timeout=mock_k8s_clients.request_timeout,
    )
    mock_k8s_clients.core_v1.delete_namespaced_config_map.assert_called_once()


@pytest.mark.asyncio
async def test_create_deployment_adopted_service_failure_keeps_configmap(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = sample_always_config().model_copy(
        update={"config_files": [ConfigFile(path="/etc/app/config.yaml", content="key: value")]}
    )
    mock_k8s_clients.apps_v1.create_namespaced_deployment.side_effect = ApiException(status=409)
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.create_namespaced_service.side_effect = ApiException(status=500)

    update = await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=config,
    )

    assert update.status == "FAILED"
    mock_k8s_clients.apps_v1.delete_namespaced_deployment.assert_not_called()
    mock_k8s_clients.core_v1.delete_namespaced_config_map.assert_not_called()


def _config_with_secret_env() -> DeploymentConfig:
    base = sample_always_config()
    container = base.containers[0].model_copy(
        update={"env": [EnvVar(name="APP_TOKEN", secretRef=SecretRef(workspace="default", name="app-token"))]}
    )
    return base.model_copy(update={"containers": [container]})


@pytest.mark.asyncio
async def test_create_deployment_creates_managed_secret(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = _config_with_secret_env()
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    update = await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=config,
        secret_env={"APP_TOKEN": "app-token-value"},
    )

    assert update.status != "FAILED"
    mock_k8s_clients.core_v1.create_namespaced_secret.assert_called_once()
    secret_body = mock_k8s_clients.core_v1.create_namespaced_secret.call_args.kwargs["body"]
    assert secret_body.metadata.name == k8s_deployment_secret_name("default", "task")
    assert secret_body.string_data == {"APP_TOKEN": "app-token-value"}


@pytest.mark.asyncio
async def test_create_deployment_no_secret_when_secret_env_empty(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = sample_always_config()
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=config,
        secret_env={},
    )

    mock_k8s_clients.core_v1.create_namespaced_secret.assert_not_called()


@pytest.mark.asyncio
async def test_create_deployment_rolls_back_secret_when_service_create_fails(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    config = _config_with_secret_env()
    mock_k8s_clients.apps_v1.create_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.create_namespaced_service.side_effect = ApiException(status=500)
    identity_labels = always_identity_labels(backoff_limit=config.backoff_limit)
    mock_k8s_clients.core_v1.read_namespaced_secret.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=identity_labels),
    )

    update = await deployment_ops.create_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        config_name="config1",
        labels={},
        backend_config={},
        config=config,
        secret_env={"APP_TOKEN": "app-token-value"},
    )

    assert update.status == "FAILED"
    mock_k8s_clients.core_v1.create_namespaced_secret.assert_called_once()
    mock_k8s_clients.core_v1.delete_namespaced_secret.assert_called_once()


@pytest.mark.asyncio
async def test_delete_deployment_removes_managed_secret(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    identity_labels = always_identity_labels()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.read_namespaced_config_map.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=identity_labels),
    )
    mock_k8s_clients.core_v1.read_namespaced_secret.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=identity_labels),
    )

    update = await deployment_ops.delete_deployment(
        deployment_ops_clients,
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
async def test_delete_deployment_skips_foreign_secret(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    identity_labels = always_identity_labels()
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.read_namespaced_config_map.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=identity_labels),
    )
    # A secret with mismatched labels must not be deleted.
    mock_k8s_clients.core_v1.read_namespaced_secret.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels={MANAGED_BY_KEY: "other-plugin"}),
    )

    await deployment_ops.delete_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        backend_config={},
        expected_labels=identity_labels,
    )

    mock_k8s_clients.core_v1.delete_namespaced_secret.assert_not_called()


@pytest.mark.asyncio
async def test_read_status_accepts_init_containers(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    config = sample_always_config().model_copy(update={"init_containers": [Container(name="init", image="busybox")]})
    mock_entities.get.side_effect = [sample_deployment(), config]
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[])

    update = await k8s_backend.read_status(workspace="default", name="task")

    assert update.status == "STARTING"


@pytest.mark.asyncio
async def test_read_deployment_status_lost_on_404(
    k8s_backend, mock_k8s_clients: MagicMock, always_deployment_context: None
) -> None:
    mock_k8s_clients.apps_v1.read_namespaced_deployment.side_effect = ApiException(status=404)

    update = await k8s_backend.read_status(workspace="default", name="task")

    assert update.status == "LOST"
    assert "not found" in update.status_message


@pytest.mark.asyncio
async def test_read_deployment_status_image_pull_backoff(
    k8s_backend, mock_k8s_clients: MagicMock, always_deployment_context: None
) -> None:
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment(ready_replicas=0)
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[mock_pod(waiting_reason="ImagePullBackOff", waiting_message="pull access denied")]
    )

    update = await k8s_backend.read_status(workspace="default", name="task")

    assert update.status == "STARTING"
    assert "ImagePullBackOff" in update.status_message


@pytest.mark.asyncio
async def test_read_deployment_status_crash_loop_failed(
    k8s_backend, mock_k8s_clients: MagicMock, always_deployment_context: None
) -> None:
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment(ready_replicas=0)
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(
        items=[mock_pod(waiting_reason="CrashLoopBackOff", waiting_message="back-off", restart_count=3)]
    )

    update = await k8s_backend.read_status(workspace="default", name="task")

    assert update.status == "FAILED"
    assert "CrashLoopBackOff" in update.status_message


@pytest.mark.asyncio
async def test_delete_deployment_removes_service(
    k8s_backend, mock_k8s_clients: MagicMock, always_deployment_context: None
) -> None:
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()

    update = await k8s_backend.delete_deployment("default", "task")

    assert update.status == "SUCCEEDED"
    mock_k8s_clients.apps_v1.delete_namespaced_deployment.assert_called_once()
    mock_k8s_clients.core_v1.delete_namespaced_service.assert_called_once()


@pytest.mark.asyncio
async def test_delete_deployment_rejects_foreign(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    foreign = mock_deployment()
    foreign.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = foreign
    mock_k8s_clients.core_v1.read_namespaced_config_map.return_value = SimpleNamespace(
        metadata=SimpleNamespace(labels=always_identity_labels()),
    )

    update = await deployment_ops.delete_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        backend_config={},
        expected_labels=always_identity_labels(),
    )

    assert update.status == "FAILED"
    mock_k8s_clients.apps_v1.delete_namespaced_deployment.assert_not_called()
    mock_k8s_clients.core_v1.read_namespaced_config_map.assert_called_once_with(
        name=k8s_deployment_configmap_name("default", "task"),
        namespace="default",
        _request_timeout=mock_k8s_clients.request_timeout,
    )


@pytest.mark.asyncio
async def test_list_managed_deployment_names_includes_always(k8s_backend, mock_k8s_clients: MagicMock) -> None:
    listed = MagicMock()
    listed.items = [deployment_list_item(workspace="default", name="server")]
    mock_k8s_clients.apps_v1.list_namespaced_deployment.return_value = listed
    mock_k8s_clients.batch_v1.list_namespaced_job.return_value = MagicMock(items=[])

    names = await k8s_backend.list_managed_deployment_names()

    assert names == ["default/server"]


@pytest.mark.asyncio
async def test_get_deployment_logs(k8s_backend, mock_k8s_clients: MagicMock, always_deployment_context: None) -> None:
    pod = mock_pod()
    mock_k8s_clients.core_v1.list_namespaced_pod.return_value = MagicMock(items=[pod])
    mock_k8s_clients.core_v1.read_namespaced_pod_log.return_value = "listening\n"

    result = await k8s_backend.get_logs(workspace="default", name="task", tail=10)

    assert result.lines == ["listening"]


@pytest.mark.asyncio
async def test_delete_deployment_orphan_returns_deployment_failure(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.side_effect = NemoEntityNotFoundError("missing")
    foreign = mock_deployment()
    foreign.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = foreign
    mock_k8s_clients.batch_v1.read_namespaced_job.side_effect = ApiException(status=404)

    with patch.object(
        k8s_backend._workload_identity_ops,
        "revoke_workload_delegations",
        AsyncMock(),
    ) as revoke_mock:
        update = await k8s_backend.delete_deployment("default", "task")

    assert update.status == "FAILED"
    assert "not managed" in update.status_message
    mock_k8s_clients.apps_v1.delete_namespaced_deployment.assert_not_called()
    revoke_mock.assert_awaited_once_with(
        k8s_backend._workload_delegations,
        config=None,
        workspace="default",
        deployment_name="task",
    )


@pytest.mark.asyncio
async def test_delete_deployment_orphan_skips_foreign_service(
    deployment_ops_clients: MagicMock, mock_k8s_clients: MagicMock
) -> None:
    foreign_service = MagicMock()
    foreign_service.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    mock_k8s_clients.apps_v1.read_namespaced_deployment.side_effect = ApiException(status=404)
    mock_k8s_clients.core_v1.read_namespaced_service.return_value = foreign_service

    update = await deployment_ops.delete_deployment(
        deployment_ops_clients,
        default_namespace="default",
        workspace="default",
        name="task",
        backend_config={},
        expected_labels=always_identity_labels(),
    )

    assert update.status == "SUCCEEDED"
    mock_k8s_clients.core_v1.delete_namespaced_service.assert_not_called()


@pytest.mark.asyncio
async def test_delete_deployment_orphan_when_entity_missing(
    k8s_backend, mock_k8s_clients: MagicMock, mock_entities: AsyncMock
) -> None:
    mock_entities.get.side_effect = NemoEntityNotFoundError("missing")
    mock_k8s_clients.apps_v1.read_namespaced_deployment.return_value = mock_deployment()
    mock_k8s_clients.batch_v1.read_namespaced_job.side_effect = ApiException(status=404)

    with patch.object(
        k8s_backend._workload_identity_ops,
        "revoke_workload_delegations",
        AsyncMock(),
    ) as revoke_mock:
        update = await k8s_backend.delete_deployment("default", "task")

    assert update.status == "SUCCEEDED"
    mock_k8s_clients.apps_v1.delete_namespaced_deployment.assert_called_once()
    revoke_mock.assert_awaited_once_with(
        k8s_backend._workload_delegations,
        config=None,
        workspace="default",
        deployment_name="task",
    )
