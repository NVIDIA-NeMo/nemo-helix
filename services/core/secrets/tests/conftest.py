# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for auditor tests."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.secrets.client import SecretsClient
from nhx.common.secrets.encryption import get_base64_encoded_random_bytes
from nhx.core.secrets.config import SecretsServiceConfig
from nhx.core.secrets.service import SecretsService
from nhx.testing import ClientContext, SDKTestClientAdapter, create_test_client
from nhx.testing.blockbuster import blockbuster_fixture

# Enable BlockBuster to detect blocking calls in async code
blockbuster = blockbuster_fixture(autouse=True)


@pytest.fixture
def current_encryption_key():
    """Generate a current encryption key for tests."""
    return get_base64_encoded_random_bytes(32)


@pytest.fixture
def old_encryption_key():
    """Generate an old encryption key for tests."""
    return get_base64_encoded_random_bytes(32)


@pytest.fixture
def service_config(current_encryption_key: str, old_encryption_key: str) -> SecretsServiceConfig:
    """Create service config for tests."""
    return SecretsServiceConfig(
        encryption={
            "current_provider": "current",
            "providers": {
                "secret_key": {
                    "current": {
                        "value": current_encryption_key,
                    },
                    "old": {
                        "value": old_encryption_key,
                    },
                }
            },
        },
    )


@pytest.fixture
def test_client(service_config) -> Generator[TestClient, None, None]:
    """Create test client"""
    with create_test_client(
        SecretsService, client_type=TestClient, service_configs={SecretsService: service_config}
    ) as tc:
        yield tc


@pytest.fixture
def sdk(test_client: TestClient) -> SecretsClient:
    """Typed Secrets client backed by the test client."""
    return SecretsClient(
        base_url="http://testserver",
        workspace="default",
        http_client=SDKTestClientAdapter(test_client),
    )


@pytest.fixture
def client_context(service_config) -> Generator[ClientContext, None, None]:
    """Context client for tests."""
    with create_test_client(
        SecretsService, client_type=ClientContext, service_configs={SecretsService: service_config}
    ) as cc:
        yield cc
