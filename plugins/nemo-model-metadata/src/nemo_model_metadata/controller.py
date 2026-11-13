# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import ClassVar

import httpx
from nemo_helix_plugin.client.errors import ConflictError, NemoClientError, NotFoundError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.types import ModelEntity, ModelProvider, UpdateModelEntityRequest
from nemo_model_metadata.config import ModelMetadataConfig, SourceName
from nemo_model_metadata.merge import merge_metadata
from nemo_model_metadata.sources.base import MetadataSource, MetadataTarget
from nemo_model_metadata.sources.models_dev import ModelsDevSource
from nemo_model_metadata.sources.ngc import NgcSource
from nemo_model_metadata.sources.provider import ProviderModelsSource

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100
_ACCEPT_ENCODING = "gzip, deflate"


def build_http_client(config: ModelMetadataConfig) -> httpx.AsyncClient:
    # httpx 0.28 fails to decode NGC's zstd responses ("cannot use a decompressobj multiple times").
    return httpx.AsyncClient(timeout=config.request_timeout_seconds, headers={"Accept-Encoding": _ACCEPT_ENCODING})


def build_sources(
    config: ModelMetadataConfig, models_client: AsyncModelsClient, http: httpx.AsyncClient
) -> list[MetadataSource]:
    sources: list[MetadataSource] = []
    for source_name in config.sources:
        if source_name == SourceName.PROVIDER:
            sources.append(ProviderModelsSource(models_client))
        elif source_name == SourceName.NGC:
            sources.append(NgcSource(http, config.ngc_url, config.ngc_org))
        elif source_name == SourceName.MODELS_DEV:
            sources.append(ModelsDevSource(http, config.models_dev_url, config.refresh_interval_seconds))
    return sources


def upstream_model_id(model: ModelEntity, provider: ModelProvider | None) -> str:
    entity_id = f"{model.workspace}/{model.name}"
    if provider is not None:
        for mapping in provider.served_models or ():
            if mapping.model_entity_id == entity_id:
                return mapping.served_model_name
    if model.api_endpoint is not None and model.api_endpoint.model_id:
        return model.api_endpoint.model_id
    return model.name


class ModelMetadataController(NemoController):
    """Fills in ``ModelEntity.metadata`` for models served by a provider, and refreshes it periodically."""

    name: ClassVar[str] = "model-metadata"
    dependencies: ClassVar[list[str]] = ["entities", "models", "inference-gateway"]

    def __init__(self) -> None:
        self._config: ModelMetadataConfig | None = None
        self._models_client: AsyncModelsClient | None = None
        self._http: httpx.AsyncClient | None = None
        self._sources: list[MetadataSource] = []
        self._last_miss: dict[str, float] = {}

    @property
    def interval_seconds(self) -> float:
        if self._config is None:
            return 300.0
        return self._config.interval_seconds

    @property
    def config(self) -> ModelMetadataConfig:
        if self._config is None:
            raise RuntimeError("ModelMetadataController.config accessed before on_startup()")
        return self._config

    @property
    def models_client(self) -> AsyncModelsClient:
        if self._models_client is None:
            raise RuntimeError("ModelMetadataController.models_client accessed before on_startup()")
        return self._models_client

    async def on_startup(self) -> None:
        config = ModelMetadataConfig.get()
        models_client = AsyncModelsClient.from_client(get_async_nemo_client(as_service=self.name, internal=True))
        http = build_http_client(config)
        self._config = config
        self._models_client = models_client
        self._http = http
        self._sources = build_sources(config, models_client, http)
        logger.info("Model metadata controller started with sources: %s", [source.name for source in self._sources])

    async def on_shutdown(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def reconcile(self) -> None:
        if not self._sources:
            return
        try:
            providers = await self._list_providers()
            models = await self._list_models()
        except NemoClientError:
            logger.exception("Failed to list models or providers")
            return
        for source in self._sources:
            source.begin_cycle()
        for model in models:
            if self.stop_requested():
                return
            provider = providers.get(model.model_providers[0]) if model.model_providers else None
            await self.reconcile_model(model, provider)

    async def list_objects(self) -> list:
        raise NotImplementedError

    async def reconcile_one(self, obj: object) -> None:
        raise NotImplementedError

    def needs_refresh(self, model: ModelEntity, now: datetime) -> bool:
        if not model.model_providers:
            return False
        refresh_interval = timedelta(seconds=self.config.refresh_interval_seconds)
        metadata = model.metadata
        if metadata is not None:
            if not metadata.sources:
                return False
            if metadata.refreshed_at is not None and now - metadata.refreshed_at < refresh_interval:
                return False
        last_miss = self._last_miss.get(f"{model.workspace}/{model.name}")
        if last_miss is not None and time.monotonic() - last_miss < self.config.refresh_interval_seconds:
            return False
        return True

    async def reconcile_model(self, model: ModelEntity, provider: ModelProvider | None) -> None:
        now = datetime.now(timezone.utc)
        if not self.needs_refresh(model, now):
            return
        model_key = f"{model.workspace}/{model.name}"
        target = MetadataTarget(upstream_model_id=upstream_model_id(model, provider), provider=provider)
        results = []
        for source in self._sources:
            try:
                metadata = await source.fetch(target)
            except Exception:
                logger.exception("Metadata source %s failed for model %s", source.name, model_key)
                continue
            if metadata is not None:
                results.append((str(source.name), metadata))
        merged = merge_metadata(results, refreshed_at=now)
        if merged is None:
            self._last_miss[model_key] = time.monotonic()
            logger.debug("No metadata found for model %s (%s)", model_key, target.upstream_model_id)
            return
        self._last_miss.pop(model_key, None)
        try:
            await self.models_client.update_model(
                workspace=model.workspace, name=model.name, body=UpdateModelEntityRequest(metadata=merged)
            )
        except (ConflictError, NotFoundError) as exc:
            logger.debug("Skipped metadata update for model %s: %s", model_key, exc)
            return
        logger.info("Updated metadata for model %s from %s", model_key, merged.sources)

    async def _list_providers(self) -> dict[str, ModelProvider]:
        page = await self.models_client.list_providers(workspace="-", query_params={"page_size": _PAGE_SIZE})
        return {f"{provider.workspace}/{provider.name}": provider async for provider in page.items()}

    async def _list_models(self) -> list[ModelEntity]:
        page = await self.models_client.list_models(
            workspace="-", query_params={"page_size": _PAGE_SIZE, "verbose": False}
        )
        return [model async for model in page.items()]
