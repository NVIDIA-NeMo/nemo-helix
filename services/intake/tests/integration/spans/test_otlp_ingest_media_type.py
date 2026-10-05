# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTLP ingest media type tests."""

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.intake.client import AsyncIntakeClient, IntakeClient
from nhx.intake.config import IntakeConfig
from nhx.intake.service import IntakeService
from nhx.testing.client import SDKTestClientAdapter, create_test_client

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


def test_client_create_sends_the_protobuf_body(client: TestClient, make_otlp_request):
    # Executed coverage that the typed client's create_otlp_traces reaches the endpoint.
    intake = IntakeClient.from_client(
        NemoClient(base_url="http://testserver", http_client=SDKTestClientAdapter(client))
    )
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

    response = intake.create_otlp_traces(content=body, workspace="default").data()

    assert response.errors == []
    spans = intake.list_spans(workspace="default", query_params={"filter": {"session_id": "conv-sdk"}})
    assert [span.name for span in spans.items()] == ["sdk-span"]


@pytest.fixture
def async_intake(intake_config: IntakeConfig):
    with create_test_client(
        IntakeService,
        client_type=AsyncNemoClient,
        service_configs={IntakeService: intake_config},
    ) as async_client:
        yield AsyncIntakeClient.from_client(async_client)


@pytest.mark.asyncio
async def test_async_client_create_sends_the_protobuf_body(async_intake: AsyncIntakeClient, make_otlp_request):
    # The async client sends its request body through a separate code path from the
    # sync one, and publish_to_intake is async, so it needs its own coverage.
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

    response = (await async_intake.create_otlp_traces(content=body, workspace="default")).data()

    assert response.errors == []
    spans = await async_intake.list_spans(
        workspace="default", query_params={"filter": {"session_id": "conv-async-sdk"}}
    )
    assert [span.name async for span in spans.items()] == ["async-sdk-span"]


def test_otlp_ingest_accepts_protobuf_content_type_with_parameters(client: TestClient, make_otlp_request):
    body = make_otlp_request([{"name": "span", "attributes": {"openinference.span.kind": "LLM"}}])

    response = client.post(
        OTLP_TRACES_PATH,
        content=body,
        headers={"Content-Type": "application/x-protobuf; charset=utf-8"},
    )

    assert response.status_code == 200, response.text
