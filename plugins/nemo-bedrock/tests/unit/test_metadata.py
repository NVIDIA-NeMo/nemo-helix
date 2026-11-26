# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

import httpx
import pytest
from nemo_bedrock_plugin.metadata import (
    BedrockMetadataSource,
    control_plane_url,
    foundation_model_candidates,
    parse_foundation_model,
)
from nemo_helix_plugin.models.metadata import MetadataSourceContext, MetadataTarget, SourceUnavailableError
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelModality, ModelProvider

CATALOG = {
    "modelSummaries": [
        {
            "modelId": "anthropic.claude-sonnet-4-6",
            "modelName": "Claude Sonnet 4.6",
            "providerName": "Anthropic",
            "inputModalities": ["TEXT", "IMAGE"],
            "outputModalities": ["TEXT"],
            "modelLifecycle": {"status": "ACTIVE"},
        },
        {
            "modelId": "amazon.titan-embed-text-v2:0",
            "modelName": "Titan Text Embeddings V2",
            "inputModalities": ["TEXT"],
            "outputModalities": ["EMBEDDING"],
        },
    ]
}


def _provider(
    host_url: str = "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1", secret: str | None = "bedrock-key"
) -> ModelProvider:
    now = datetime(2026, 11, 13, tzinfo=timezone.utc)
    return ModelProvider(
        name="bedrock",
        workspace="default",
        host_url=host_url,
        api_key_secret_name=secret,
        created_at=now,
        updated_at=now,
    )


def _key_resolver(value: str = "bedrock-api-key"):
    async def resolve(provider: ModelProvider) -> str | None:
        assert provider.api_key_secret_name == "bedrock-key"
        return value

    return resolve


@pytest.mark.parametrize(
    ("host_url", "expected"),
    [
        ("https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1", "https://bedrock.us-west-2.amazonaws.com"),
        ("https://bedrock-runtime-fips.us-east-1.amazonaws.com", "https://bedrock-fips.us-east-1.amazonaws.com"),
        ("https://bedrock-mantle.eu-central-1.api.aws/v1", "https://bedrock.eu-central-1.amazonaws.com"),
        ("https://integrate.api.nvidia.com/v1", None),
        ("https://bedrock.example.com/v1", None),
    ],
)
def test_control_plane_url(host_url: str, expected: str | None):
    assert control_plane_url(host_url) == expected


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("anthropic.claude-sonnet-4-6", ["anthropic.claude-sonnet-4-6"]),
        ("us.anthropic.claude-sonnet-4-6", ["us.anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-4-6"]),
        ("global.anthropic.claude-sonnet-4-6", ["global.anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-4-6"]),
    ],
)
def test_foundation_model_candidates(model_id: str, expected: list[str]):
    assert foundation_model_candidates(model_id) == expected


def test_parse_foundation_model():
    assert parse_foundation_model(CATALOG["modelSummaries"][0]) == ModelMetadata(
        display_name="Claude Sonnet 4.6",
        input_modalities=[ModelModality.TEXT, ModelModality.IMAGE],
        output_modalities=[ModelModality.TEXT],
    )


def test_parse_foundation_model_marks_embedding_models():
    metadata = parse_foundation_model(CATALOG["modelSummaries"][1])

    assert metadata is not None
    assert metadata.capabilities == [ModelCapability.EMBEDDING]


async def test_bedrock_source_reads_catalog_once_per_cycle_with_provider_key():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CATALOG)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = BedrockMetadataSource()
        ctx = MetadataSourceContext(http=http, api_key_resolver=_key_resolver())
        profile = await source.fetch(
            MetadataTarget(upstream_model_id="us.anthropic.claude-sonnet-4-6", provider=_provider()), ctx
        )
        missing = await source.fetch(MetadataTarget(upstream_model_id="meta.llama-unknown", provider=_provider()), ctx)

    assert [str(request.url) for request in requests] == ["https://bedrock.us-west-2.amazonaws.com/foundation-models"]
    assert requests[0].headers["Authorization"] == "Bearer bedrock-api-key"
    assert profile is not None
    assert profile.display_name == "Claude Sonnet 4.6"
    assert missing is None


@pytest.mark.parametrize(
    ("host_url", "expected"),
    [
        ("https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1", True),
        ("https://bedrock-mantle.eu-central-1.api.aws/v1", True),
        ("https://integrate.api.nvidia.com/v1", False),
    ],
)
def test_bedrock_source_handles_bedrock_hosts(host_url: str, expected: bool):
    assert BedrockMetadataSource().handles(_provider(host_url=host_url)) is expected


async def test_bedrock_source_returns_none_without_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        metadata = await BedrockMetadataSource().fetch(
            MetadataTarget(upstream_model_id="anthropic.claude-sonnet-4-6", provider=_provider(secret=None)),
            MetadataSourceContext(http=http, api_key_resolver=_key_resolver()),
        )

    assert metadata is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403),
        httpx.Response(503),
        httpx.Response(200, json={"message": "nope"}),
        httpx.ConnectError("down"),
    ],
    ids=["forbidden", "server-error", "no-summaries", "unreachable"],
)
async def test_bedrock_source_raises_when_catalog_unavailable(response: httpx.Response | httpx.ConnectError):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(response, Exception):
            raise response
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(SourceUnavailableError):
            await BedrockMetadataSource().fetch(
                MetadataTarget(upstream_model_id="anthropic.claude-sonnet-4-6", provider=_provider()),
                MetadataSourceContext(http=http, api_key_resolver=_key_resolver()),
            )
