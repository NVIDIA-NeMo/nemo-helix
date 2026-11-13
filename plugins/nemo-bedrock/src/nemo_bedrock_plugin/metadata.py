# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any
from urllib.parse import urlparse

import httpx
from nemo_helix_plugin.models.metadata import (
    MAX_DISPLAY_NAME_LENGTH,
    MetadataSourceContext,
    MetadataTarget,
    NemoModelMetadataSource,
    SourceUnavailableError,
    bounded_text,
    modalities,
)
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelProvider

_INFERENCE_HOST = re.compile(r"^bedrock-(runtime|runtime-fips|mantle)\.([a-z0-9-]+)\.(?:amazonaws\.com|api\.aws)$")
# Cross-region inference profiles prefix the base model id with a geography, e.g. "us.anthropic.claude-...".
_INFERENCE_PROFILE_PREFIX = re.compile(r"^(?:us|us-gov|eu|apac|au|ca|jp|global)\.")


def control_plane_url(host_url: str) -> str | None:
    match = _INFERENCE_HOST.match(urlparse(host_url).hostname or "")
    if match is None:
        return None
    endpoint, region = match.group(1), match.group(2)
    service = "bedrock-fips" if endpoint == "runtime-fips" else "bedrock"
    return f"https://{service}.{region}.amazonaws.com"


def foundation_model_candidates(model_id: str) -> list[str]:
    return list(dict.fromkeys([model_id, _INFERENCE_PROFILE_PREFIX.sub("", model_id, count=1)]))


def parse_foundation_model(summary: dict[str, Any]) -> ModelMetadata:
    output_modalities = summary.get("outputModalities")
    capabilities = []
    if isinstance(output_modalities, list) and "EMBEDDING" in output_modalities:
        capabilities.append(ModelCapability.EMBEDDING)
    return ModelMetadata(
        display_name=bounded_text(summary.get("modelName"), MAX_DISPLAY_NAME_LENGTH),
        capabilities=capabilities,
        input_modalities=modalities(summary.get("inputModalities")),
        output_modalities=modalities(output_modalities),
    )


class BedrockMetadataSource(NemoModelMetadataSource):
    """Reads Amazon Bedrock's foundation model catalog with the provider's own Bedrock API key."""

    name = "bedrock"

    def handles(self, provider: ModelProvider) -> bool:
        return control_plane_url(provider.host_url) is not None

    async def fetch(self, target: MetadataTarget, ctx: MetadataSourceContext) -> ModelMetadata | None:
        provider = target.provider
        base_url = control_plane_url(provider.host_url)
        if base_url is None or not provider.api_key_secret_name:
            return None
        catalog = await self.cached_for_cycle(
            f"{base_url}|{provider.workspace}/{provider.api_key_secret_name}",
            lambda: _fetch_catalog(provider, base_url, ctx),
        )
        for model_id in foundation_model_candidates(target.upstream_model_id):
            summary = catalog.get(model_id)
            if summary is not None:
                return parse_foundation_model(summary)
        return None


async def _fetch_catalog(
    provider: ModelProvider, base_url: str, ctx: MetadataSourceContext
) -> dict[str, dict[str, Any]]:
    api_key = await ctx.provider_api_key(provider)
    if not api_key:
        return {}
    url = f"{base_url}/foundation-models"
    try:
        response = await ctx.http.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as exc:
        raise SourceUnavailableError(f"Bedrock request to {url} failed: {exc}") from exc
    if response.status_code != 200:
        raise SourceUnavailableError(f"Bedrock request to {url} returned HTTP {response.status_code}")
    try:
        body: object = response.json()
    except ValueError as exc:
        raise SourceUnavailableError(f"Bedrock returned invalid JSON for {url}") from exc
    summaries = body.get("modelSummaries") if isinstance(body, dict) else None
    if not isinstance(summaries, list):
        raise SourceUnavailableError(f"Bedrock returned no modelSummaries for {url}")
    return {
        summary["modelId"]: summary
        for summary in summaries
        if isinstance(summary, dict) and isinstance(summary.get("modelId"), str)
    }
