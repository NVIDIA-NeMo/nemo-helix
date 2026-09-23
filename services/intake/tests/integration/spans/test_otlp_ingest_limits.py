# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP ingest body size limit tests."""

import pytest
from fastapi.testclient import TestClient
from nhx.intake.config import ClickHouseConfig, IntakeConfig
from nhx.intake.service import IntakeService
from nhx.intake.spans.clickhouse_client import ClickHouseSettings
from nhx.testing import create_test_client


@pytest.fixture
def small_limit_client(clickhouse_settings: ClickHouseSettings):
    intake_config = IntakeConfig(
        clickhouse_config=ClickHouseConfig(
            url=clickhouse_settings.url,
            user=clickhouse_settings.user,
            password=clickhouse_settings.password,
            database=clickhouse_settings.database,
        ),
        otlp_max_body_bytes=1024,
    )
    with create_test_client(
        IntakeService,
        client_type=TestClient,
        service_configs={IntakeService: intake_config},
    ) as test_client:
        yield test_client


def test_otlp_ingest_rejects_oversized_body_via_content_length(small_limit_client: TestClient):
    body = b"\x00" * 2048
    response = small_limit_client.post(
        "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
        content=body,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert response.status_code == 413, response.text
    assert "exceeds limit" in response.json()["detail"]


def test_otlp_ingest_accepts_body_under_limit(small_limit_client: TestClient, make_otlp_request):
    body = make_otlp_request(
        [
            {
                "name": "tiny",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "gen_ai.conversation.id": "conv-tiny",
                },
            }
        ]
    )
    assert len(body) <= 1024
    response = small_limit_client.post(
        "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces",
        content=body,
        headers={"Content-Type": "application/x-protobuf"},
    )
    assert response.status_code == 200, response.text
