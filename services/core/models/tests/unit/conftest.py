# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common test fixtures and configuration for Models service tests."""

from collections.abc import Callable
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from nhx.common.entities.client import EntityClient
from nhx.core.models.service import ModelsService
from nhx.testing import create_test_client
from nhx.testing.blockbuster import blockbuster_fixture

blockbuster = blockbuster_fixture(autouse=True)

# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_collection_modifyitems(config, items):
    """
    Modify test items during collection.

    Auto-marks tests based on their location:
    - Tests in e2e/ directories get the 'e2e' marker
    - Tests in integration/ directories get the 'integration' marker
    - Tests without category markers get the 'unit' marker
    """
    category_markers = {
        "unit",
        "e2e",
        "integration",
        "regression",
        "canary",
        "slow",
        "skip_in_ci",
    }

    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}

        if "/e2e/" in str(item.fspath):
            if "e2e" not in marker_names:
                item.add_marker(pytest.mark.e2e)
                marker_names.add("e2e")
        elif "/integration/" in str(item.fspath):
            if "integration" not in marker_names:
                item.add_marker(pytest.mark.integration)
                marker_names.add("integration")

        if not marker_names.intersection(category_markers):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# Mock EntityClient Fixtures (for unit tests)
# ============================================================================


@pytest.fixture
def mock_entity_client() -> AsyncMock:
    """Create a mock EntityClient for unit tests.

    Use this when testing service logic in isolation.
    """
    mock = AsyncMock(spec=EntityClient)

    # Configure default return values
    mock.create.return_value = None
    mock.get.return_value = None
    mock.get_by_name.return_value = None
    mock.update.return_value = None
    mock.delete.return_value = None

    return mock


# ============================================================================
# Integration Test Fixtures (with real in-memory storage)
# ============================================================================


@pytest.fixture
def entity_client() -> EntityClient:
    """Create an EntityClient backed by in-memory storage.

    Use this for integration tests that need real storage behavior.
    """
    # Include workspaces needed by tests (default + workspace filter tests)
    workspaces = ["default", "workspace1", "workspace2", "production"]
    with create_test_client(client_type=EntityClient, workspaces=workspaces) as client:
        yield client


@pytest.fixture
def models_service() -> ModelsService:
    """Get the Models service instance."""
    return ModelsService()


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the Models API with mocked dependencies.

    This provides a full FastAPI test client with MockEntityClient for
    integration tests.
    """
    with create_test_client(ModelsService, client_type=TestClient) as tc:
        yield tc


@pytest.fixture
def models_app() -> ModelsService:
    """Get the Models FastAPI app instance for testing."""
    return ModelsService().app


# ============================================================================
# SDK Error Factory Fixture
# ============================================================================


@pytest.fixture
def make_sdk_error() -> Callable:
    """Factory fixture for creating SDK API errors with mock request/response."""

    def _make(cls: type, status_code: int = 400):
        request = httpx.Request("GET", "http://test")
        response = httpx.Response(status_code=status_code, request=request)
        return cls("error", response=response, body=None)

    return _make
