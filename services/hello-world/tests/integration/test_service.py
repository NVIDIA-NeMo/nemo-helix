# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for HelloWorld API endpoints."""

from typing import Generator

import pytest
from nemo_helix import NeMoHelix
from nhx.hello_world.service import HelloWorldService
from nhx.testing import create_test_client


class TestHelloWorldEndpoints:
    """Tests for hello world API endpoints."""

    @pytest.fixture
    def sdk(self) -> Generator[NeMoHelix, None, None]:
        """Create SDK client for testing."""
        with create_test_client(HelloWorldService) as client:
            yield client

    def test_hello_endpoint(self, sdk: NeMoHelix):
        """Test GET /apis/hello-world/v2/workspaces/{workspace_id}/hello returns hello message."""
        response = sdk._client.get("/apis/hello-world/v2/workspaces/default/hello")

        assert response.status_code == 200
        assert response.json() == {"message": "Hello World from workspace 'default'"}

    def test_health_endpoint(self, sdk: NeMoHelix):
        """Test GET /health/ready returns ready status."""
        response = sdk._client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
        # Full service breakdown is on /status
        status_response = sdk._client.get("/status")
        assert "hello-world" in status_response.json()["services"]["ready"]

    def test_health_live_endpoint(self, sdk: NeMoHelix):
        """Test GET /health/live returns live."""
        response = sdk._client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "live"}
