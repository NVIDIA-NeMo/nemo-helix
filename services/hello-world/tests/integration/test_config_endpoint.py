# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for config-info endpoint demonstrating config dependency injection."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from nhx.common.config import HelixConfig
from nhx.common.service.dependencies import get_platform_config
from nhx.hello_world.config import HelloWorldConfig
from nhx.hello_world.service import HelloWorldService
from nhx.testing import create_test_client


class TestConfigInfoEndpoint:
    """Tests for the /config-info endpoint."""

    @pytest.fixture
    def mock_platform_config(self):
        """Create mock platform config."""
        return HelixConfig(base_url="http://test-platform.example.com")

    @pytest.fixture
    def mock_service_config(self):
        """Create mock service config."""
        return HelloWorldConfig(greeting_prefix="Howdy", max_message_length=200)

    @pytest.fixture
    def http_client(self, mock_platform_config, mock_service_config) -> Generator[TestClient, None, None]:
        """Create a test client with mocked configs."""
        with create_test_client(
            HelloWorldService,
            client_type=TestClient,
            dependency_overrides={
                get_platform_config: lambda: mock_platform_config,
            },
            service_configs={HelloWorldService: mock_service_config},
        ) as client:
            yield client

    def test_config_info_returns_both_configs(self, http_client: TestClient):
        """Test GET /config-info returns both platform and service config values."""
        response = http_client.get("/apis/hello-world/v2/workspaces/default/config-info")

        assert response.status_code == 200
        data = response.json()
        assert data["platform_base_url"] == "http://test-platform.example.com"
        assert data["greeting_prefix"] == "Howdy"
        assert data["max_message_length"] == 200

    def test_config_info_with_default_values(self):
        """Test /config-info works with default config values."""
        default_platform = HelixConfig()
        default_service = HelloWorldConfig()

        with create_test_client(
            HelloWorldService,
            client_type=TestClient,
            dependency_overrides={
                get_platform_config: lambda: default_platform,
            },
            service_configs={HelloWorldService: default_service},
        ) as http_client:
            response = http_client.get("/apis/hello-world/v2/workspaces/default/config-info")

            assert response.status_code == 200
            data = response.json()
            assert data["platform_base_url"] == "http://localhost:8080"
            assert data["greeting_prefix"] == "Hello"
            assert data["max_message_length"] == 100
