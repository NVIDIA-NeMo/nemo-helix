# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import httpx
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelProvider
from nemo_model_metadata.config import SourceName
from nemo_model_metadata.sources.base import MetadataSource, MetadataTarget
from nemo_model_metadata.sources.parsing import as_positive_int, modalities

logger = logging.getLogger(__name__)

_CAPABILITY_PARAMETERS: dict[str, ModelCapability] = {
    "tools": ModelCapability.TOOL_CALLING,
    "tool_choice": ModelCapability.TOOL_CALLING,
    "reasoning": ModelCapability.REASONING,
    "include_reasoning": ModelCapability.REASONING,
    "structured_outputs": ModelCapability.STRUCTURED_OUTPUT,
    "response_format": ModelCapability.STRUCTURED_OUTPUT,
}


def parse_provider_model(entry: dict[str, Any]) -> ModelMetadata | None:
    """Parse the extra fields some OpenAI-compatible providers (OpenRouter, vLLM) add to ``GET /v1/models``."""
    architecture = entry.get("architecture")
    if not isinstance(architecture, dict):
        architecture = {}
    top_provider = entry.get("top_provider")
    if not isinstance(top_provider, dict):
        top_provider = {}
    supported = entry.get("supported_parameters")
    capabilities: list[ModelCapability] = []
    if isinstance(supported, list):
        for parameter in supported:
            capability = _CAPABILITY_PARAMETERS.get(parameter) if isinstance(parameter, str) else None
            if capability is not None and capability not in capabilities:
                capabilities.append(capability)

    name = entry.get("name")
    description = entry.get("description")
    metadata = ModelMetadata(
        display_name=name if isinstance(name, str) and name else None,
        description=description if isinstance(description, str) and description else None,
        capabilities=capabilities,
        input_modalities=modalities(architecture.get("input_modalities")),
        output_modalities=modalities(architecture.get("output_modalities")),
        context_length=as_positive_int(entry.get("context_length")) or as_positive_int(entry.get("max_model_len")),
        max_output_tokens=as_positive_int(top_provider.get("max_completion_tokens")),
    )
    if metadata == ModelMetadata():
        return None
    return metadata


class ProviderModelsSource(MetadataSource):
    """Reads the provider's own model list through the inference gateway, which supplies the provider's credentials."""

    name = SourceName.PROVIDER

    def __init__(self, models_client: AsyncModelsClient) -> None:
        self._models_client = models_client
        self._listings: dict[str, dict[str, dict[str, Any]]] = {}

    def begin_cycle(self) -> None:
        self._listings.clear()

    async def fetch(self, target: MetadataTarget) -> ModelMetadata | None:
        if target.provider is None:
            return None
        listing = await self._listing(target.provider)
        entry = listing.get(target.upstream_model_id)
        if entry is None:
            return None
        return parse_provider_model(entry)

    async def _listing(self, provider: ModelProvider) -> dict[str, dict[str, Any]]:
        key = f"{provider.workspace}/{provider.name}"
        cached = self._listings.get(key)
        if cached is not None:
            return cached
        listing: dict[str, dict[str, Any]] = {}
        url = f"{self._models_client.get_provider_route_openai_url(provider)}/models"
        try:
            response = await self._models_client.http_client.get(url)
            response.raise_for_status()
            body: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.debug("Could not list models for provider %s: %s", key, exc)
            body = None
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    listing[entry["id"]] = entry
        self._listings[key] = listing
        return listing
