# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from kubernetes.client.rest import ApiException
from nemo_deployments_plugin.backends.k8s import volumes as volume_ops
from nemo_deployments_plugin.backends.k8s.client import KubernetesClients
from nemo_deployments_plugin.backends.k8s.volumes import map_pvc_phase_to_status, status_from_pvc
from nemo_deployments_plugin.backends.labels import (
    MANAGED_BY_KEY,
    VOLUME_NAME_LABEL,
    VOLUME_WORKSPACE_LABEL,
    volume_identity_labels,
)


@pytest.mark.parametrize(
    ("phase", "expected_status"),
    [
        ("Pending", "PENDING"),
        ("Bound", "BOUND"),
        ("Lost", "FAILED"),
        (None, "PENDING"),
    ],
)
def test_map_pvc_phase_to_status(phase: str | None, expected_status: str) -> None:
    update = map_pvc_phase_to_status(pvc_name="dep-vol-default-data-abc12345", phase=phase)
    assert update.status == expected_status


@pytest.mark.asyncio
async def test_create_volume_emits_pvc_with_storage_class_and_size(k8s_backend, mock_k8s_clients: MagicMock) -> None:
    mock_pvc = MagicMock()
    mock_pvc.status.phase = "Pending"
    mock_pvc.metadata.labels = volume_identity_labels("default", "weights")
    mock_pvc.metadata.deletion_timestamp = None
    mock_k8s_clients.core_v1.create_namespaced_persistent_volume_claim.return_value = mock_pvc
    mock_k8s_clients.request_timeout = 30

    update = await k8s_backend.create_volume(
        workspace="default",
        name="weights",
        size="10Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={"k8s": {"storageClass": "fast-ssd", "namespace": "models"}},
    )

    assert update.status == "PENDING"
    call = mock_k8s_clients.core_v1.create_namespaced_persistent_volume_claim
    call.assert_called_once()
    assert call.call_args.kwargs["namespace"] == "models"
    assert call.call_args.kwargs["_request_timeout"] == 30
    body = call.call_args.kwargs["body"]
    assert body.spec.resources.requests["storage"] == "10Gi"
    assert body.spec.storage_class_name == "fast-ssd"
    assert body.spec.access_modes == ["ReadWriteOnce"]
    assert body.metadata.labels[VOLUME_WORKSPACE_LABEL] == "default"
    assert body.metadata.labels[VOLUME_NAME_LABEL] == "weights"


@pytest.mark.asyncio
async def test_create_volume_conflict_reads_existing_pvc(mock_k8s_clients: MagicMock) -> None:
    existing = MagicMock()
    existing.status.phase = "Bound"
    existing.metadata.labels = volume_identity_labels("default", "data")
    existing.metadata.deletion_timestamp = None
    mock_k8s_clients.core_v1.create_namespaced_persistent_volume_claim.side_effect = ApiException(status=409)
    mock_k8s_clients.core_v1.read_namespaced_persistent_volume_claim.return_value = existing
    clients = MagicMock(spec=KubernetesClients)
    clients.core_v1 = mock_k8s_clients.core_v1
    clients.request_timeout = 60

    update = await volume_ops.create_volume(
        clients,
        default_namespace="default",
        workspace="default",
        name="data",
        size="1Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={},
    )

    assert update.status == "BOUND"
    mock_k8s_clients.core_v1.read_namespaced_persistent_volume_claim.assert_called_once()


@pytest.mark.asyncio
async def test_create_volume_conflict_rejects_foreign_pvc(mock_k8s_clients: MagicMock) -> None:
    foreign = MagicMock()
    foreign.metadata.labels = {MANAGED_BY_KEY: "other-plugin"}
    foreign.status.phase = "Bound"
    mock_k8s_clients.core_v1.create_namespaced_persistent_volume_claim.side_effect = ApiException(status=409)
    mock_k8s_clients.core_v1.read_namespaced_persistent_volume_claim.return_value = foreign
    clients = MagicMock(spec=KubernetesClients)
    clients.core_v1 = mock_k8s_clients.core_v1
    clients.request_timeout = 60

    update = await volume_ops.create_volume(
        clients,
        default_namespace="default",
        workspace="default",
        name="data",
        size="1Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={},
    )

    assert update.status == "FAILED"
    assert "not managed" in update.status_message


def test_status_from_pvc_deleting_reports_deleting() -> None:
    labels = volume_identity_labels("default", "data")
    pvc = MagicMock()
    pvc.metadata.labels = labels
    pvc.metadata.deletion_timestamp = "2026-01-01T00:00:00Z"
    pvc.status.phase = "Bound"

    update = status_from_pvc(pvc=pvc, pvc_name="dep-vol-default-data-abc12345", expected_labels=labels)

    assert update.status == "DELETING"


@pytest.mark.asyncio
async def test_read_volume_status_uses_entity_namespace(k8s_backend, mock_k8s_clients: MagicMock) -> None:
    mock_pvc = MagicMock()
    mock_pvc.status.phase = "Bound"
    mock_pvc.metadata.labels = volume_identity_labels("default", "weights")
    mock_pvc.metadata.deletion_timestamp = None
    mock_k8s_clients.core_v1.read_namespaced_persistent_volume_claim.return_value = mock_pvc

    update = await k8s_backend.read_volume_status(
        workspace="default",
        name="weights",
        backend_config={"k8s": {"namespace": "models"}},
    )

    assert update.status == "BOUND"
    call = mock_k8s_clients.core_v1.read_namespaced_persistent_volume_claim
    assert call.call_args.kwargs["namespace"] == "models"


@pytest.mark.asyncio
async def test_create_volume_malformed_backend_config_returns_failed(mock_k8s_clients: MagicMock) -> None:
    clients = MagicMock(spec=KubernetesClients)
    clients.core_v1 = mock_k8s_clients.core_v1
    clients.request_timeout = 60

    update = await volume_ops.create_volume(
        clients,
        default_namespace="default",
        workspace="default",
        name="data",
        size="1Gi",
        access_modes=["ReadWriteOnce"],
        backend_config={"k8s": {"storageClass": 123}},
    )

    assert update.status == "FAILED"
    mock_k8s_clients.core_v1.create_namespaced_persistent_volume_claim.assert_not_called()


@pytest.mark.asyncio
async def test_read_volume_status_malformed_backend_config_returns_failed(k8s_backend) -> None:
    update = await k8s_backend.read_volume_status(
        workspace="default",
        name="weights",
        backend_config={"k8s": {"namespace": 42}},
    )

    assert update.status == "FAILED"


@pytest.mark.asyncio
async def test_read_volume_status_not_found(k8s_backend, mock_k8s_clients: MagicMock) -> None:
    mock_k8s_clients.core_v1.read_namespaced_persistent_volume_claim.side_effect = ApiException(status=404)

    update = await k8s_backend.read_volume_status(workspace="default", name="missing")

    assert update.status == "FAILED"
    assert "not found" in update.status_message


@pytest.mark.asyncio
async def test_delete_volume_missing_is_released(k8s_backend, mock_k8s_clients: MagicMock) -> None:
    mock_k8s_clients.request_timeout = 30
    mock_k8s_clients.core_v1.delete_namespaced_persistent_volume_claim.side_effect = ApiException(status=404)

    update = await k8s_backend.delete_volume("default", "gone")

    assert update.status == "RELEASED"


@pytest.mark.asyncio
async def test_delete_volume_api_failure_is_reported(k8s_backend, mock_k8s_clients: MagicMock) -> None:
    mock_k8s_clients.core_v1.delete_namespaced_persistent_volume_claim.side_effect = ApiException(status=403)

    update = await k8s_backend.delete_volume("default", "weights")

    assert update.status == "FAILED"
    assert "Failed to delete PVC" in update.status_message
