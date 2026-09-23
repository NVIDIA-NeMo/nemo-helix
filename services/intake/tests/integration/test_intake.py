# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Intake service route surface."""

from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from nemo_helix import NeMoHelix
from nhx.intake.config import ClickHouseConfig, IntakeConfig
from nhx.intake.service import IntakeService
from nhx.testing.client import SDKTestClientAdapter, create_test_client


@pytest.fixture(scope="module")
def http_client() -> Generator[TestClient, None, None]:
    """Route-surface client that does not require a live ClickHouse."""
    intake_config = IntakeConfig(
        clickhouse_config=ClickHouseConfig(url="http://127.0.0.1:1"),
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(IntakeService, "is_ready", AsyncMock(return_value=True))
        with create_test_client(
            IntakeService,
            client_type=TestClient,
            service_configs={IntakeService: intake_config},
        ) as client:
            yield client


@pytest.fixture(scope="module")
def sdk(http_client: TestClient) -> NeMoHelix:
    """SDK client backed by the test client."""
    return NeMoHelix(base_url="http://testserver", http_client=SDKTestClientAdapter(http_client))


def test_intake_openapi_keeps_span_era_routes(sdk: NeMoHelix) -> None:
    response = sdk._client.get("/openapi.json")
    assert response.status_code == 200

    paths = response.json().get("paths", {})

    assert "/apis/intake/v2/workspaces/{workspace}/spans" in paths
    assert "get" in paths["/apis/intake/v2/workspaces/{workspace}/spans"]
    span_groups_operation = paths["/apis/intake/v2/workspaces/{workspace}/spans/groups"]["get"]
    assert "400" in span_groups_operation["responses"]
    assert "/apis/intake/v2/workspaces/{workspace}/spans/{span_id}" in paths
    assert "get" in paths["/apis/intake/v2/workspaces/{workspace}/spans/{span_id}"]
    assert "/apis/intake/v2/workspaces/{workspace}/traces" in paths
    assert "get" in paths["/apis/intake/v2/workspaces/{workspace}/traces"]
    assert "/apis/intake/v2/workspaces/{workspace}/traces/{id}" in paths
    assert "get" in paths["/apis/intake/v2/workspaces/{workspace}/traces/{id}"]
    assert "/apis/intake/v2/workspaces/{workspace}/sessions/{id}" in paths
    assert "get" in paths["/apis/intake/v2/workspaces/{workspace}/sessions/{id}"]
    assert "/apis/intake/v2/workspaces/{workspace}/annotations" in paths
    assert "/apis/intake/v2/workspaces/{workspace}/evaluator-results" in paths
    assert "/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces" in paths
    assert "/apis/intake/v2/workspaces/{workspace}/ingest/spans" in paths
