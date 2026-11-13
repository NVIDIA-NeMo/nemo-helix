# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nemo_helix_plugin.client.errors import NemoClientError
from nemo_helix_plugin.inference_gateway.client import AsyncInferenceGatewayClient
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelProvider
from nemo_model_metadata.config import SourceName
from nemo_model_metadata.sources.base import MetadataSource, MetadataTarget, SourceUnavailableError
from nemo_model_metadata.sources.parsing import (
    MAX_DESCRIPTION_LENGTH,
    MAX_DISPLAY_NAME_LENGTH,
    as_positive_int,
    bounded_text,
    modalities,
)

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

    metadata = ModelMetadata(
        display_name=bounded_text(entry.get("name"), MAX_DISPLAY_NAME_LENGTH),
        description=bounded_text(entry.get("description"), MAX_DESCRIPTION_LENGTH),
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

    def __init__(self, gateway_client: AsyncInferenceGatewayClient) -> None:
        self._gateway_client = gateway_client
        self._listings: dict[str, dict[str, dict[str, Any]] | SourceUnavailableError] = {}

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
        if cached is None:
            cached = await self._fetch_listing(provider)
            self._listings[key] = cached
        if isinstance(cached, SourceUnavailableError):
            raise cached
        return cached

    async def _fetch_listing(self, provider: ModelProvider) -> dict[str, dict[str, Any]] | SourceUnavailableError:
        try:
            body = await self._gateway_client.get_provider_models(workspace=provider.workspace, name=provider.name)
        except NemoClientError as exc:
            return SourceUnavailableError(
                f"Could not list models for provider {provider.workspace}/{provider.name}: {exc}"
            )
        data = body.get("data") if isinstance(body, dict) else None
        listing: dict[str, dict[str, Any]] = {}
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    listing[entry["id"]] = entry
        return listing
