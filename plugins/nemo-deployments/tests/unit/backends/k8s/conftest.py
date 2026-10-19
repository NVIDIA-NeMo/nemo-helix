# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for k8s backend unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_deployments_plugin.backends.k8s.backend import K8sDeploymentBackend


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_entities() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_k8s_clients() -> MagicMock:
    clients = MagicMock()
    clients.core_v1 = MagicMock()
    clients.apps_v1 = MagicMock()
    clients.batch_v1 = MagicMock()
    clients.request_timeout = 30
    return clients


@pytest.fixture
def k8s_backend(
    mock_client: MagicMock,
    mock_k8s_clients: MagicMock,
    mock_entities: AsyncMock,
) -> Iterator[K8sDeploymentBackend]:
    with (
        patch("nemo_deployments_plugin.backends.k8s.backend.KubernetesClients", return_value=mock_k8s_clients),
        patch("nemo_deployments_plugin.backends.k8s.backend.AsyncEntitiesClient"),
        patch("nemo_deployments_plugin.backends.k8s.backend.NemoEntitiesClient", return_value=mock_entities),
    ):
        backend = K8sDeploymentBackend(
            mock_client,
            {"default_namespace": "nemo-deployments", "request_timeout": 30},
        )
        backend._entities = mock_entities
        yield backend
