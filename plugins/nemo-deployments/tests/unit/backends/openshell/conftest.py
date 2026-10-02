# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for OpenShell backend unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_deployments_plugin.backends.openshell.backend import OpenShellDeploymentBackend


@pytest.fixture
def mock_sdk() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_entities() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_stub() -> MagicMock:
    stub = MagicMock()
    # read_status re-derives endpoints via ListServices; default to none.
    stub.ListServices.return_value = MagicMock(services=[], next_page_token="")
    return stub


@pytest.fixture
def openshell_backend(
    mock_sdk: MagicMock, mock_entities: AsyncMock, mock_stub: MagicMock
) -> Iterator[OpenShellDeploymentBackend]:
    with (
        patch("nemo_deployments_plugin.backends.openshell.backend.client_from_platform"),
        patch("nemo_deployments_plugin.backends.openshell.backend.NemoEntitiesClient", return_value=mock_entities),
        patch("grpc.insecure_channel", return_value=MagicMock()),
    ):
        backend = OpenShellDeploymentBackend(mock_sdk, {"gateway_endpoint": "http://127.0.0.1:17670"})
        backend._stub = mock_stub
        backend._entities = mock_entities
        yield backend
