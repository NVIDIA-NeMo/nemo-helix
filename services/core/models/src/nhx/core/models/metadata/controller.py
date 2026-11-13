# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import monotonic

import httpx
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NemoClientError, NotFoundError
from nemo_helix_plugin.discovery import discover_entry_points
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.metadata import (
    METADATA_SOURCES_GROUP,
    MetadataSourceContext,
    MetadataTarget,
    NemoModelMetadataSource,
)
from nemo_helix_plugin.models.refs import served_model_name_for_entity
from nemo_helix_plugin.models.types import ModelEntity, ModelMetadata, ModelProvider, UpdateModelEntityRequest
from nhx.core.models.config import MetadataControllerConfig
from nhx.core.models.metadata.provider_source import ProviderModelsSource

logger = logging.getLogger(__name__)

_PAGE_SIZE = 100
_ACCEPT_ENCODING = "gzip, deflate"


def build_http_client(config: MetadataControllerConfig) -> httpx.AsyncClient:
    # httpx 0.28 fails to decode NGC's zstd responses ("cannot use a decompressobj multiple times").
    return httpx.AsyncClient(timeout=config.request_timeout_seconds, headers={"Accept-Encoding": _ACCEPT_ENCODING})


def load_plugin_sources() -> list[NemoModelMetadataSource]:
    sources: list[NemoModelMetadataSource] = []
    for name, entry_point in sorted(discover_entry_points(METADATA_SOURCES_GROUP).items()):
        try:
            source_cls = entry_point.load()
            sources.append(source_cls())
        except Exception:
            logger.exception("Failed to load model metadata source %s", name)
    return sources


def upstream_model_id(model: ModelEntity, provider: ModelProvider) -> str:
    served_model_name = served_model_name_for_entity(provider, model)
    if served_model_name:
        return served_model_name
    if model.api_endpoint is not None and model.api_endpoint.model_id:
        return model.api_endpoint.model_id
    return model.name


def is_hand_set(metadata: ModelMetadata | None) -> bool:
    return metadata is not None and not metadata.sources


@dataclass(frozen=True)
class LookupResult:
    metadata: ModelMetadata | None
    failure: str | None = None


class ModelMetadataController:
    """Keeps ``ModelEntity.metadata`` filled in from the source that handles each model's provider."""

    def __init__(
        self,
        config: MetadataControllerConfig,
        client: AsyncNemoClient,
        http: httpx.AsyncClient,
        plugin_sources: list[NemoModelMetadataSource],
    ) -> None:
        self._config = config
        self._models_client = AsyncModelsClient.from_client(client)
        self._ctx = MetadataSourceContext(http=http, client=client)
        self._plugin_sources = plugin_sources
        self._provider_source = ProviderModelsSource(request_timeout_seconds=config.request_timeout_seconds)
        self._providers: dict[str, ModelProvider] = {}
        self._retry_after: dict[str, float] = {}
        self._consecutive_failures: dict[str, int] = {}

    @property
    def source_names(self) -> list[str]:
        return [*(source.name for source in self._plugin_sources), self._provider_source.name]

    def source_for(self, provider: ModelProvider) -> NemoModelMetadataSource:
        for source in self._plugin_sources:
            try:
                if source.handles(provider):
                    return source
            except Exception:
                logger.exception("Metadata source %s failed to check provider %s", source.name, provider.name)
        return self._provider_source

    async def reconcile(self, on_progress: Callable[[], None] | None = None) -> None:
        for model in await self.list_models_to_refresh():
            try:
                await self.reconcile_model(model)
            except Exception:
                logger.exception("Failed to refresh metadata for model %s/%s", model.workspace, model.name)
            if on_progress is not None:
                on_progress()

    async def list_models_to_refresh(self) -> list[ModelEntity]:
        try:
            self._providers = await self._list_providers()
            models = await self._list_models()
        except NemoClientError:
            logger.exception("Failed to list models or providers")
            return []
        for source in [*self._plugin_sources, self._provider_source]:
            source.begin_cycle()
        now = datetime.now(timezone.utc)
        return [model for model in models if self.needs_refresh(model, now)]

    def needs_refresh(self, model: ModelEntity, now: datetime) -> bool:
        if not model.model_providers:
            return False
        metadata = model.metadata
        if is_hand_set(metadata):
            return False
        refresh_interval = timedelta(seconds=self._config.refresh_interval_seconds)
        if (
            metadata is not None
            and metadata.refreshed_at is not None
            and now - metadata.refreshed_at < refresh_interval
        ):
            return False
        retry_after = self._retry_after.get(f"{model.workspace}/{model.name}")
        if retry_after is not None and monotonic() < retry_after:
            return False
        return True

    async def reconcile_model(self, model: ModelEntity) -> None:
        model_key = f"{model.workspace}/{model.name}"
        result = await self.look_up(model, refreshed_at=datetime.now(timezone.utc))
        if result.failure is not None:
            delay = self._record_failure(model_key)
            logger.warning(
                "Skipped metadata refresh for model %s (%s); retrying in %.0fs", model_key, result.failure, delay
            )
            return
        self._consecutive_failures.pop(model_key, None)
        if result.metadata is None:
            self._retry_after[model_key] = monotonic() + self._config.refresh_interval_seconds
            logger.debug("No metadata found for model %s", model_key)
            return
        self._retry_after.pop(model_key, None)
        try:
            current = (await self._models_client.get_model(workspace=model.workspace, name=model.name)).data()
            if is_hand_set(current.metadata):
                return
            await self._models_client.update_model(
                workspace=model.workspace, name=model.name, body=UpdateModelEntityRequest(metadata=result.metadata)
            )
        except (ConflictError, NotFoundError) as exc:
            logger.debug("Skipped metadata update for model %s: %s", model_key, exc)
            return
        logger.info("Updated metadata for model %s from %s", model_key, result.metadata.sources)

    def _record_failure(self, model_key: str) -> float:
        failures = self._consecutive_failures.get(model_key, 0) + 1
        self._consecutive_failures[model_key] = failures
        delay = min(self._config.interval_seconds * (2 ** (failures - 1) - 1), self._config.refresh_interval_seconds)
        self._retry_after[model_key] = monotonic() + delay
        return delay

    async def look_up(self, model: ModelEntity, refreshed_at: datetime) -> LookupResult:
        failure: str | None = None
        for provider_ref in model.model_providers:
            provider = self._providers.get(provider_ref)
            if provider is None:
                continue
            source = self.source_for(provider)
            target = MetadataTarget(upstream_model_id=upstream_model_id(model, provider), provider=provider)
            try:
                metadata = await source.fetch(target, self._ctx)
            except Exception as exc:
                failure = f"{source.name} failed for {target.upstream_model_id}: {exc}"
                continue
            if metadata is not None:
                stamped = metadata.model_copy(update={"sources": [source.name], "refreshed_at": refreshed_at})
                return LookupResult(metadata=stamped)
        return LookupResult(metadata=None, failure=failure)

    async def _list_providers(self) -> dict[str, ModelProvider]:
        page = await self._models_client.list_providers(workspace="-", query_params={"page_size": _PAGE_SIZE})
        return {f"{provider.workspace}/{provider.name}": provider async for provider in page.items()}

    async def _list_models(self) -> list[ModelEntity]:
        page = await self._models_client.list_models(
            workspace="-", query_params={"page_size": _PAGE_SIZE, "verbose": False}
        )
        return [model async for model in page.items()]
