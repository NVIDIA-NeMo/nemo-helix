# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timezone

import httpx
import pytest
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.metadata import MetadataSourceContext, MetadataTarget, SourceUnavailableError
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelModality, ModelProvider
from nhx.core.models.metadata.provider_source import ProviderModelsSource, parse_provider_model

PROVIDER_MODELS_PATH = "/apis/inference-gateway/v2/workspaces/default/provider/spark/-/v1/models"


def test_parse_provider_model_reads_openrouter_fields():
    metadata = parse_provider_model(
        {
            "id": "meta-llama/llama-3.1-8b-instruct",
            "name": "Meta: Llama 3.1 8B Instruct",
            "description": "Meta's latest class of model",
            "context_length": 131072,
            "architecture": {"input_modalities": ["text", "file"], "output_modalities": ["text"]},
            "top_provider": {"max_completion_tokens": 16384},
            "supported_parameters": ["tools", "tool_choice", "response_format", "temperature"],
        }
    )

    assert metadata == ModelMetadata(
        display_name="Meta: Llama 3.1 8B Instruct",
        description="Meta's latest class of model",
        capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT],
        input_modalities=[ModelModality.TEXT, ModelModality.PDF],
        output_modalities=[ModelModality.TEXT],
        context_length=131072,
        max_output_tokens=16384,
    )


def test_parse_provider_model_reads_vllm_max_model_len():
    metadata = parse_provider_model({"id": "Qwen/Qwen2.5-7B-Instruct", "owned_by": "vllm", "max_model_len": 32768})

    assert metadata == ModelMetadata(context_length=32768)


def test_parse_provider_model_returns_empty_metadata_for_bare_openai_entry():
    assert parse_provider_model({"id": "gpt-4o", "object": "model", "owned_by": "openai"}) == ModelMetadata()


def test_parse_provider_model_truncates_long_text():
    metadata = parse_provider_model({"id": "m", "name": "n" * 300, "description": "d" * 5000})

    assert metadata is not None
    assert metadata.display_name == "n" * 255
    assert metadata.description == "d" * 4000


def _provider() -> ModelProvider:
    now = datetime(2026, 11, 13, tzinfo=timezone.utc)
    return ModelProvider(
        name="spark", workspace="default", host_url="http://spark:8089/v1", created_at=now, updated_at=now
    )


async def _no_api_key(provider: ModelProvider) -> str | None:
    return None


def _source(handler) -> tuple[ProviderModelsSource, MetadataSourceContext]:
    client = AsyncNemoClient(
        base_url="http://platform",
        default_headers={"X-NMP-Principal-Id": "service:model-metadata"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    source = ProviderModelsSource(client, request_timeout_seconds=15.0)
    return source, MetadataSourceContext(http=httpx.AsyncClient(), api_key_resolver=_no_api_key)


async def test_provider_source_lists_models_once_per_cycle_with_platform_headers():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"data": [{"id": "gemma-4-12b", "name": "Gemma 4 12B", "context_length": 262144}]}
        )

    source, ctx = _source(handler)
    first = await source.fetch(MetadataTarget(upstream_model_id="gemma-4-12b", provider=_provider()), ctx)
    missing = await source.fetch(MetadataTarget(upstream_model_id="not-served", provider=_provider()), ctx)

    assert [request.url.path for request in requests] == [PROVIDER_MODELS_PATH]
    assert requests[0].headers["X-NMP-Principal-Id"] == "service:model-metadata"
    assert first == ModelMetadata(display_name="Gemma 4 12B", context_length=262144)
    assert missing is None


async def test_provider_source_raises_when_listing_fails_and_retries_next_cycle():
    responses = [httpx.Response(424, json={"detail": "provider not ready"}), httpx.Response(200, json={"data": []})]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    source, ctx = _source(handler)
    target = MetadataTarget(upstream_model_id="gemma-4-12b", provider=_provider())

    with pytest.raises(SourceUnavailableError):
        await source.fetch(target, ctx)
    with pytest.raises(SourceUnavailableError):
        await source.fetch(target, ctx)
    source.begin_cycle()

    assert await source.fetch(target, ctx) is None


@pytest.mark.parametrize("body", [{"object": "list"}, ["gemma-4-12b"], "<html>maintenance</html>"])
async def test_provider_source_raises_on_malformed_listing(body: object):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    source, ctx = _source(handler)

    with pytest.raises(SourceUnavailableError):
        await source.fetch(MetadataTarget(upstream_model_id="gemma-4-12b", provider=_provider()), ctx)


async def test_provider_source_treats_upstream_without_model_list_as_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/ready"):
            return httpx.Response(200, json={"workspace": "default", "name": "spark"})
        return httpx.Response(404, json={"message": "Not Found"})

    source, ctx = _source(handler)

    assert await source.fetch(MetadataTarget(upstream_model_id="gemma-4-12b", provider=_provider()), ctx) is None


async def test_provider_source_raises_when_gateway_does_not_know_provider_yet():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Model provider not found for default/spark"})

    source, ctx = _source(handler)

    with pytest.raises(SourceUnavailableError):
        await source.fetch(MetadataTarget(upstream_model_id="gemma-4-12b", provider=_provider()), ctx)


async def test_provider_source_does_not_retry_failed_listing():
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, json={"detail": "upstream down"})

    source, ctx = _source(handler)

    with pytest.raises(SourceUnavailableError):
        await source.fetch(MetadataTarget(upstream_model_id="gemma-4-12b", provider=_provider()), ctx)

    assert len(attempts) == 1
