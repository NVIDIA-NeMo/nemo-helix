# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration test fixtures for auth service with authorization ENABLED.

This test package runs with authorization enabled to test auth middleware,
role bindings, and the embedded policy engine.

Uses the create_test_client pattern with auth_enabled=True for simplified
auth setup.
"""

from typing import Generator, Iterator

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.client import NemoClient
from nhx.testing.client import create_test_client

# Test principal for authenticated requests (service-level access)
SERVICE_PRINCIPAL = "service:integration-test"


@pytest.fixture(scope="module")
def test_client() -> Generator[TestClient, None, None]:
    """TestClient with auth service running (auth enabled).

    Module-scoped for efficiency - the auth service is expensive to start.

    Uses auth_enabled=True which:
    - Enables AuthorizationMiddleware with embedded PDP
    - Creates default workspace/project with TEST_USER_EMAIL as principal
    - Sets up SharedAuthConfig and AuthServiceConfig overrides
    """
    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
    ) as client:
        yield client


@pytest.fixture(scope="module")
def client(test_client: TestClient) -> Iterator[NemoClient]:
    """Typed platform client backed by the test client.

    Module-scoped because it shares the TestClient.
    """
    yield NemoClient(base_url="http://testserver", http_client=test_client)


@pytest.fixture
def http_client(test_client: TestClient) -> Iterator[TestClient]:
    """Raw HTTP client without auth headers.

    Returns the TestClient directly - tests should not add auth headers.
    """
    yield test_client
