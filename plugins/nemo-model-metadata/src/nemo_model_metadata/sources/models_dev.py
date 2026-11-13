# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from nemo_helix_plugin.models.types import ModelCapability, ModelMetadata, ModelProvider
from nemo_model_metadata.config import SourceName
from nemo_model_metadata.sources.base import MetadataSource, MetadataTarget
from nemo_model_metadata.sources.parsing import as_positive_int, modalities

logger = logging.getLogger(__name__)

# First-party providers whose models.dev entry has no ``api`` URL because their SDKs hardcode it.
_FIRST_PARTY_HOSTS: dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "google",
    "api.groq.com": "groq",
    "api.mistral.ai": "mistral",
    "api.x.ai": "xai",
    "api.cohere.com": "cohere",
    "api.together.xyz": "togetherai",
}

_BEDROCK_PROVIDER = "amazon-bedrock"

_TYPE_CAPABILITIES: dict[str, ModelCapability] = {
    "embedding": ModelCapability.EMBEDDING,
    "reranking": ModelCapability.RERANKING,
}

Catalog = dict[str, dict[str, Any]]


def _host(url: str | None) -> str | None:
    if not url:
        return None
    return urlparse(url).hostname


def parse_models_dev_model(entry: dict[str, Any]) -> ModelMetadata:
    capabilities: list[ModelCapability] = []
    if entry.get("tool_call") is True:
        capabilities.append(ModelCapability.TOOL_CALLING)
    if entry.get("reasoning") is True:
        capabilities.append(ModelCapability.REASONING)
    if entry.get("structured_output") is True:
        capabilities.append(ModelCapability.STRUCTURED_OUTPUT)
    model_type = entry.get("type")
    type_capability = _TYPE_CAPABILITIES.get(model_type) if isinstance(model_type, str) else None
    if type_capability is not None:
        capabilities.append(type_capability)

    model_modalities = entry.get("modalities")
    if not isinstance(model_modalities, dict):
        model_modalities = {}
    limit = entry.get("limit")
    if not isinstance(limit, dict):
        limit = {}
    name = entry.get("name")
    description = entry.get("description")
    family = entry.get("family")
    return ModelMetadata(
        display_name=name if isinstance(name, str) and name else None,
        description=description if isinstance(description, str) and description else None,
        family=family if isinstance(family, str) and family else None,
        capabilities=capabilities,
        input_modalities=modalities(model_modalities.get("input")),
        output_modalities=modalities(model_modalities.get("output")),
        context_length=as_positive_int(limit.get("context")),
        max_output_tokens=as_positive_int(limit.get("output")),
    )


def _catalog_provider_for_host(host: str) -> str | None:
    if host.startswith("bedrock") and host.endswith((".amazonaws.com", ".api.aws")):
        return _BEDROCK_PROVIDER
    return _FIRST_PARTY_HOSTS.get(host)


def find_catalog_entry(catalog: Catalog, provider: ModelProvider | None, model_id: str) -> dict[str, Any] | None:
    """Find the model under the catalog provider that serves the same host as the ModelProvider."""
    host = _host(provider.host_url) if provider is not None else None
    if host is None:
        return None
    known_provider = _catalog_provider_for_host(host)
    for provider_key, catalog_provider in catalog.items():
        if provider_key != known_provider and _host(catalog_provider.get("api")) != host:
            continue
        models = catalog_provider.get("models")
        if isinstance(models, dict) and isinstance(models.get(model_id), dict):
            return models[model_id]
    return None


class ModelsDevSource(MetadataSource):
    name = SourceName.MODELS_DEV

    def __init__(self, http: httpx.AsyncClient, url: str, refresh_interval_seconds: float) -> None:
        self._http = http
        self._url = url
        self._refresh_interval_seconds = refresh_interval_seconds
        self._catalog: Catalog | None = None
        self._fetched_at: float | None = None

    async def fetch(self, target: MetadataTarget) -> ModelMetadata | None:
        catalog = await self._load_catalog()
        entry = find_catalog_entry(catalog, target.provider, target.upstream_model_id)
        if entry is None:
            return None
        return parse_models_dev_model(entry)

    async def _load_catalog(self) -> Catalog:
        now = time.monotonic()
        if self._fetched_at is not None and now - self._fetched_at < self._refresh_interval_seconds:
            return self._catalog or {}
        self._fetched_at = now
        try:
            response = await self._http.get(self._url)
            response.raise_for_status()
            body: object = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Could not load the models.dev catalog from %s: %s", self._url, exc)
            return self._catalog or {}
        if isinstance(body, dict):
            self._catalog = {key: value for key, value in body.items() if isinstance(value, dict)}
        return self._catalog or {}
