# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelModality, ModelProvider
from nemo_model_metadata.sources.base import MetadataTarget
from nemo_model_metadata.sources.models_dev import ModelsDevSource, find_catalog_entry, parse_models_dev_model
from nemo_model_metadata.sources.provider import parse_provider_model

MODELS_DEV_CATALOG: dict[str, dict[str, Any]] = {
    "openai": {
        "id": "openai",
        "models": {
            "gpt-4o": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "description": "Omni-era GPT for multimodal chat",
                "family": "gpt",
                "reasoning": False,
                "tool_call": True,
                "structured_output": True,
                "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                "limit": {"context": 128000, "output": 16384},
            }
        },
    },
    "nvidia": {
        "id": "nvidia",
        "api": "https://integrate.api.nvidia.com/v1",
        "models": {
            "openai/gpt-oss-20b": {
                "id": "openai/gpt-oss-20b",
                "name": "GPT OSS 20B (NVIDIA)",
                "reasoning": True,
                "tool_call": True,
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 131072, "output": 32768},
            }
        },
    },
    "openrouter": {
        "id": "openrouter",
        "api": "https://openrouter.ai/api/v1",
        "models": {
            "openai/gpt-oss-20b": {
                "id": "openai/gpt-oss-20b",
                "name": "GPT OSS 20B (OpenRouter)",
                "modalities": {"input": ["text"], "output": ["text"]},
                "limit": {"context": 64000, "output": 8000},
            },
            "nv-embed": {"id": "nv-embed", "name": "NV Embed", "type": "embedding", "limit": {"context": 4096}},
        },
    },
    "oci": {
        "id": "oci",
        "models": {"us.anthropic.claude-sonnet-4-6": {"name": "Claude Sonnet 4.6 (OCI)", "limit": {"context": 200000}}},
    },
    "amazon-bedrock": {
        "id": "amazon-bedrock",
        "models": {"us.anthropic.claude-sonnet-4-6": {"name": "Claude Sonnet 4.6 (US)", "limit": {"context": 1000000}}},
    },
}


def _provider(host_url: str) -> ModelProvider:
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    return ModelProvider(name="p", workspace="default", host_url=host_url, created_at=now, updated_at=now)


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


def test_parse_provider_model_returns_none_for_bare_openai_entry():
    assert parse_provider_model({"id": "gpt-4o", "object": "model", "owned_by": "openai"}) is None


def test_parse_models_dev_model():
    metadata = parse_models_dev_model(MODELS_DEV_CATALOG["openai"]["models"]["gpt-4o"])

    assert metadata == ModelMetadata(
        display_name="GPT-4o",
        description="Omni-era GPT for multimodal chat",
        family="gpt",
        capabilities=[ModelCapability.TOOL_CALLING, ModelCapability.STRUCTURED_OUTPUT],
        input_modalities=[ModelModality.TEXT, ModelModality.IMAGE, ModelModality.PDF],
        output_modalities=[ModelModality.TEXT],
        context_length=128000,
        max_output_tokens=16384,
    )


def test_parse_models_dev_model_maps_type_to_capability():
    metadata = parse_models_dev_model(MODELS_DEV_CATALOG["openrouter"]["models"]["nv-embed"])

    assert metadata.capabilities == [ModelCapability.EMBEDDING]


@pytest.mark.parametrize(
    ("host_url", "expected_name"),
    [
        ("https://integrate.api.nvidia.com/v1", "GPT OSS 20B (NVIDIA)"),
        ("https://openrouter.ai/api/v1", "GPT OSS 20B (OpenRouter)"),
    ],
)
def test_find_catalog_entry_prefers_provider_with_matching_api_host(host_url: str, expected_name: str):
    entry = find_catalog_entry(MODELS_DEV_CATALOG, _provider(host_url), "openai/gpt-oss-20b")

    assert entry is not None
    assert entry["name"] == expected_name


def test_find_catalog_entry_matches_first_party_host_without_api_url():
    entry = find_catalog_entry(MODELS_DEV_CATALOG, _provider("https://api.openai.com/v1"), "gpt-4o")

    assert entry is not None
    assert entry["name"] == "GPT-4o"


@pytest.mark.parametrize(
    "host_url",
    ["https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1", "https://bedrock-mantle.us-east-1.api.aws/v1"],
)
def test_find_catalog_entry_matches_bedrock_regional_hosts(host_url: str):
    entry = find_catalog_entry(MODELS_DEV_CATALOG, _provider(host_url), "us.anthropic.claude-sonnet-4-6")

    assert entry is not None
    assert entry["name"] == "Claude Sonnet 4.6 (US)"


def test_find_catalog_entry_ignores_other_providers_for_unknown_host():
    assert find_catalog_entry(MODELS_DEV_CATALOG, _provider("http://my-vllm:8000/v1"), "gpt-4o") is None


def test_find_catalog_entry_ignores_other_providers_when_host_lacks_model():
    assert find_catalog_entry(MODELS_DEV_CATALOG, _provider("https://api.openai.com/v1"), "nv-embed") is None


def test_find_catalog_entry_returns_none_without_provider():
    assert find_catalog_entry(MODELS_DEV_CATALOG, None, "gpt-4o") is None


async def test_models_dev_source_fetches_catalog_once():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=MODELS_DEV_CATALOG)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = ModelsDevSource(http, "https://models.dev/api.json", refresh_interval_seconds=3600)
        target = MetadataTarget(upstream_model_id="gpt-4o", provider=_provider("https://api.openai.com/v1"))
        first = await source.fetch(target)
        second = await source.fetch(target)

    assert len(requests) == 1
    assert first is not None
    assert first.display_name == "GPT-4o"
    assert second == first


async def test_models_dev_source_returns_none_when_catalog_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = ModelsDevSource(http, "https://models.dev/api.json", refresh_interval_seconds=3600)
        result = await source.fetch(MetadataTarget(upstream_model_id="gpt-4o", provider=None))

    assert result is None
