# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP ingest media type tests."""

import pytest
from fastapi.testclient import TestClient
from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nmp.intake.config import IntakeConfig
from nmp.intake.service import IntakeService
from nmp.testing.client import SDKTestClientAdapter, create_test_client

OTLP_TRACES_PATH = "/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces"
OTLP_TRACES_ROUTE = "/apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces"


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({"Content-Type": "application/json"}, id="json"),
        pytest.param({"Content-Type": "application/octet-stream"}, id="octet-stream"),
    ],
)
def test_otlp_ingest_rejects_non_protobuf_content_type(client: TestClient, headers: dict[str, str]):
    response = client.post(OTLP_TRACES_PATH, content=b"", headers=headers)

    assert response.status_code == 415, response.text
    assert response.json()["detail"] == "OTLP trace ingest only accepts application/x-protobuf"


def test_otlp_ingest_rejects_blank_content_type(client: TestClient):
    response = client.post(OTLP_TRACES_PATH, content=b"", headers={"Content-Type": ""})

    assert response.status_code == 415, response.text


def test_otlp_ingest_declares_a_protobuf_request_body(client: TestClient):
    operation = client.app.openapi()["paths"][OTLP_TRACES_ROUTE]["post"]
    request_body = operation["requestBody"]

    assert request_body["required"] is True
    assert list(request_body["content"]) == ["application/x-protobuf"]
    assert request_body["content"]["application/x-protobuf"]["schema"]["format"] == "binary"
    # A content-type parameter alongside the body makes generated clients set the header
    # twice, by two mechanisms that can disagree.
    assert "content-type" not in [parameter["name"] for parameter in operation["parameters"]]


def test_sdk_create_sends_the_protobuf_body(client: TestClient, make_otlp_request):
    # The generated SDK tests for this method are all skipped ("Mock server tests are
    # disabled"), so this is the only executed coverage that create() reaches the endpoint.
    sdk = NeMoPlatform(base_url="http://testserver", http_client=SDKTestClientAdapter(client))
    body = make_otlp_request(
        [
            {
                "name": "sdk-span",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "gen_ai.conversation.id": "conv-sdk",
                },
            }
        ]
    )

    response = sdk.intake.ingest.otlp.v1.traces.create(body=body, workspace="default")

    assert response.errors == []
    spans = sdk.intake.spans.list(workspace="default", filter={"session_id": "conv-sdk"})
    assert [span.name for span in spans.data] == ["sdk-span"]


@pytest.fixture
def async_sdk(intake_config: IntakeConfig):
    with create_test_client(
        IntakeService,
        client_type=AsyncNeMoPlatform,
        service_configs={IntakeService: intake_config},
    ) as sdk:
        yield sdk


@pytest.mark.asyncio
async def test_async_sdk_create_sends_the_protobuf_body(async_sdk: AsyncNeMoPlatform, make_otlp_request):
    # The async resource builds its request body through a separate code path from the
    # sync one, and publish_to_intake is async — so it needs its own coverage.
    body = make_otlp_request(
        [
            {
                "name": "async-sdk-span",
                "attributes": {
                    "openinference.span.kind": "LLM",
                    "gen_ai.conversation.id": "conv-async-sdk",
                },
            }
        ]
    )

    response = await async_sdk.intake.ingest.otlp.v1.traces.create(body=body, workspace="default")

    assert response.errors == []
    spans = await async_sdk.intake.spans.list(workspace="default", filter={"session_id": "conv-async-sdk"})
    assert [span.name for span in spans.data] == ["async-sdk-span"]


def test_otlp_ingest_accepts_protobuf_content_type_with_parameters(client: TestClient, make_otlp_request):
    body = make_otlp_request([{"name": "span", "attributes": {"openinference.span.kind": "LLM"}}])

    response = client.post(
        OTLP_TRACES_PATH,
        content=body,
        headers={"Content-Type": "application/x-protobuf; charset=utf-8"},
    )

    assert response.status_code == 200, response.text
