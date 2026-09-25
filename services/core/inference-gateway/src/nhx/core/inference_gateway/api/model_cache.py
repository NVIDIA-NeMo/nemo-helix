# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable

from nemo_helix import APIConnectionError, APIStatusError, AsyncNeMoHelix
from nemo_helix.types.inference import ModelProvider, ServedModelMapping
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoHTTPError as PluginHTTPError
from nemo_helix_plugin.client.errors import NemoTransportError as PluginTransportError
from nemo_helix_plugin.inference_middleware import BackendFormat
from nemo_helix_plugin.models.client import AsyncModelsClient
from nemo_helix_plugin.models.types import ModelEntity
from nhx.common.entities.utils import parse_model_entity_ref
from nhx.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nhx.core.inference_gateway.api.proxy import retrieve_secret_value
from nhx.core.inference_gateway.api.virtual_model_cache import (
    VirtualModelCache,
    VirtualModelCacheRefreshError,
    refresh_virtual_model_cache,
)
from nhx.core.inference_gateway.config import SECRETS_TTL_SEC, DebugModelProvider

if TYPE_CHECKING:
    from nhx.core.inference_gateway.api.middleware_registry import MiddlewareRegistry

logger = logging.getLogger(__name__)


class ModelProviderRefreshError(Exception):
    """Exception raised when refreshing model provider information fails."""


@dataclass
class ModelProviderInfo:
    """Container for model information including cached secrets."""

    model_provider: ModelProvider
    """The model provider configuration"""

    secret_value: str | None = None
    """Cached authentication token for the provider"""

    secret_value_updated_at: datetime = datetime.min
    """Timestamp of last secret refresh"""


@dataclass
class ModelEntityInfo:
    workspace: str
    """Workspace this model entity belongs to."""

    name: str
    """Model entity name within the workspace."""

    model_providers: list[tuple[str, ModelProviderInfo]] = field(default_factory=list)
    """Routing data: (served_model_name, ModelProviderInfo) pairs for all providers
    currently serving this entity."""

    spec: object | None = None
    """Model specification (capabilities, context size, family).  Populated from the
    models service when available; ``None`` otherwise.  Structurally satisfies the
    ``ModelSpec`` Protocol from ``nemo_helix_plugin.inference_middleware``."""

    finetuning_type: str | None = None
    """Fine-tuning type (e.g. ``"lora_merged"``), or ``None`` for base models."""

    backend_format: BackendFormat | None = None
    """Inference API wire format expected by the backend, or ``None`` if unset."""

    @property
    def providers(self) -> list:
        """All provider objects serving this entity (satisfies the ``ModelEntity``
        Protocol's ``providers`` field from ``nemo_helix_plugin.inference_middleware``)."""
        return [info.model_provider for _, info in self.model_providers]


@dataclass
class ModelCache:
    """Cache for model provider information with TTL-based refresh logic."""

    workspace_name_provider_map: dict[tuple[str, str], ModelProviderInfo] = field(default_factory=dict)
    """Mapping of provider (workspace, name) tuples to ModelProviderInfo"""

    model_entity_info_map: dict[tuple[str, str], ModelEntityInfo] = field(default_factory=dict)
    """Mapping of model entity (workspace, name) tuples to ModelEntityInfo"""

    secret_value_ttl: int = SECRETS_TTL_SEC
    """Time-to-live in seconds for cached secrets (0 = always refresh)"""

    _entity_map_signature: frozenset[tuple[str, str, tuple[tuple[str, str], ...]]] | None = field(default=None)
    """Signature of the provider layer as of the last ``rebuild_model_entity_map`` run.

    Used by :func:`refresh_model_cache` to skip a redundant rebuild when the provider set
    and its served-models are unchanged since the previous cycle. ``None`` means "no rebuild
    has run yet" (forces the first rebuild); the signature is computed inline in
    :func:`refresh_model_cache`.
    """

    def get_from_provider(self, workspace: str, provider_name: str) -> ModelProviderInfo | None:
        model_info = self.workspace_name_provider_map.get((workspace, provider_name))
        return model_info

    def get_from_model_entity(self, workspace: str, model_entity_name: str) -> ModelEntityInfo | None:
        return self.model_entity_info_map.get((workspace, model_entity_name))

    def update_model_info(self, model_info: ModelProviderInfo):
        self.workspace_name_provider_map[(model_info.model_provider.workspace, model_info.model_provider.name)] = (
            model_info
        )

    def rebuild_model_entity_map(self) -> None:
        """
        This will crawl over all the cached model_providers and rebuild model_entity_info_map.
        This map will be used when doing routing for a given model_entity.
        Splits model_entity_id on the first "/" only so LoRA ids
        (workspace/base&adapters/adapter-workspace/adapter-name) work.
        """
        existing_metadata = self.model_entity_info_map
        rebuilt_map: dict[tuple[str, str], ModelEntityInfo] = {}
        for model_provider_info in self.workspace_name_provider_map.values():
            served_models = model_provider_info.model_provider.served_models or []
            for served_model in served_models:
                # First-"/"-only split (preserving a LoRA composite as the entity name) via the
                # shared parser; a malformed id (no "/" or empty segment) raises ValueError, which
                # we log and skip. The parser also strips surrounding whitespace.
                try:
                    ref = parse_model_entity_ref(served_model.model_entity_id)
                except ValueError:
                    logger.warning("Skipping malformed entity_id %r", served_model.model_entity_id)
                    continue
                workspace, model_entity_name = ref.workspace, ref.name
                key = (workspace, model_entity_name)
                model_entity_info = rebuilt_map.get(key)
                if model_entity_info is None:
                    previous = existing_metadata.get(key)
                    model_entity_info = ModelEntityInfo(
                        workspace=workspace,
                        name=model_entity_name,
                        spec=previous.spec if previous is not None else None,
                        finetuning_type=previous.finetuning_type if previous is not None else None,
                        backend_format=_to_plugin_backend_format(previous.backend_format)
                        if previous is not None
                        else None,
                    )
                model_entity_info.model_providers.append((served_model.served_model_name, model_provider_info))
                rebuilt_map[key] = model_entity_info
        self.model_entity_info_map = rebuilt_map

    def update_model_entity_metadata(self, model_entities: list[ModelEntity]) -> None:
        """Populate cached ModelEntity metadata used by inference middleware."""
        incoming_keys: set[tuple[str, str]] = set()
        for model_entity in model_entities:
            key = (model_entity.workspace, model_entity.name)
            incoming_keys.add(key)
            model_entity_info = self.model_entity_info_map.get(key)
            if model_entity_info is None:
                continue
            if _field_is_present(model_entity, "spec"):
                model_entity_info.spec = model_entity.spec
            if _field_is_present(model_entity, "finetuning_type"):
                model_entity_info.finetuning_type = model_entity.finetuning_type
            if _field_is_present(model_entity, "backend_format"):
                model_entity_info.backend_format = _to_plugin_backend_format(model_entity.backend_format)

        for key in set(self.model_entity_info_map) - incoming_keys:
            model_entity_info = self.model_entity_info_map[key]
            model_entity_info.spec = None
            model_entity_info.finetuning_type = None
            model_entity_info.backend_format = None


def _field_is_present(model_entity: object, field: str) -> bool:
    fields_set = getattr(model_entity, "model_fields_set", None)
    if fields_set is not None:
        return field in fields_set
    return hasattr(model_entity, field)


def _to_plugin_backend_format(value: object | None) -> BackendFormat | None:
    if isinstance(value, BackendFormat):
        return value

    enum_value = getattr(value, "value", value)
    if isinstance(enum_value, str):
        try:
            return BackendFormat(enum_value)
        except ValueError:
            return None

    return None


def model_provider_getter_from_sdk(models_sdk: AsyncNeMoHelix) -> Callable[[], Awaitable[list[ModelProvider]]]:
    async def _model_provider_getter() -> list[ModelProvider]:
        try:
            # SDK returns AsyncPaginator - iterate through all pages to get all providers
            resp = models_sdk.inference.providers.list(
                workspace="-",  # Cross-workspace query
                page_size=200,
                extra_headers=MARK_INTERNAL_REQUEST_HEADERS,
            )
            providers = [provider async for provider in resp]
            return providers
        except APIConnectionError as exc:
            raise ModelProviderRefreshError(f"Error connecting to models service: {exc}") from exc
        except APIStatusError as exc:
            raise ModelProviderRefreshError(
                f"Error refreshing from models service: {exc.status_code}, {exc.body}"
            ) from exc

    return _model_provider_getter


def model_entity_getter_from_sdk(models_sdk: AsyncNeMoHelix) -> Callable[[], Awaitable[list[ModelEntity]]]:
    async def _model_entity_getter() -> list[ModelEntity]:
        try:
            # SDK returns AsyncPaginator - iterate through all pages to get all model entities.
            resp = await client_from_platform(models_sdk, AsyncModelsClient).list_models(
                workspace="-",  # Cross-workspace query
                query_params={"page_size": 200, "verbose": False},
            )
            models = [model async for model in resp.items()]
            return models
        except PluginTransportError as exc:
            raise ModelProviderRefreshError(f"Error connecting to models service: {exc}") from exc
        except PluginHTTPError as exc:
            raise ModelProviderRefreshError(f"Error refreshing model entities from models service: {exc}") from exc

    return _model_entity_getter


def debug_model_provider_getter(
    debug_model_providers: list[DebugModelProvider],
) -> Callable[[], Awaitable[list[ModelProvider]]]:
    async def _model_provider_getter() -> list[ModelProvider]:
        return [
            ModelProvider(
                workspace=d.workspace,
                name=d.name,
                host_url=d.host_url,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                served_models=[
                    ServedModelMapping(served_model_name=sm.served_model_name, model_entity_id=sm.model_entity_id)
                    for sm in d.served_models
                ],
            )
            for d in debug_model_providers
        ]

    return _model_provider_getter


async def refresh_model_cache(
    model_cache: ModelCache,
    model_provider_getter: Callable[[], Awaitable[list[ModelProvider]]],
    secrets_sdk: AsyncNeMoHelix,
    model_entity_getter: Callable[[], Awaitable[list[ModelEntity]]] | None = None,
    virtual_model_cache: VirtualModelCache | None = None,
    middleware_registry: MiddlewareRegistry | None = None,
) -> None:
    try:
        model_providers = await model_provider_getter()
        logger.debug(f"Successfully fetched {len(model_providers)} model providers from models service")
    except Exception as exc:
        msg = "Error trying to refresh model provider cache"
        logger.exception(msg)
        raise ModelProviderRefreshError(msg) from exc

    # Build a set of current provider keys from the fetched list
    current_provider_keys = {(mp.workspace, mp.name) for mp in model_providers}

    # Remove providers that are no longer in the fetched list
    cached_provider_keys = set(model_cache.workspace_name_provider_map.keys())
    stale_provider_keys = cached_provider_keys - current_provider_keys
    for stale_key in stale_provider_keys:
        del model_cache.workspace_name_provider_map[stale_key]
        logger.debug(f"Removed stale provider from cache: {stale_key[0]}/{stale_key[1]}")

    # Update or add providers from the fetched list
    for model_provider in model_providers:
        model_info = model_cache.get_from_provider(model_provider.workspace, model_provider.name)
        if model_info is None:
            model_info = ModelProviderInfo(model_provider=model_provider)
        else:
            # Update existing provider with fresh data from API (e.g., served_models may have changed)
            model_info.model_provider = model_provider
        await refresh_model_provider_info(
            model_cache=model_cache,
            model_info=model_info,
            secrets_sdk=secrets_sdk,
        )

    # Skip the entity-map rebuild when the provider layer is unchanged since the last
    # cycle. Compute-only optimization: it saves no network calls, but avoids the
    # synchronous O(providers x served_models) rebuild burst on the event loop every cycle.
    #
    # The signature is order-sensitive over each provider's served_models because the
    # rebuild appends to model_providers[] in that order and consumers pick [0] — a reorder
    # is a real routing change and must invalidate. Metadata (spec/finetuning_type/
    # backend_format) and provider config (host_url/secrets) are excluded: both are applied
    # in place, not via rebuild. Cold start is covered by the initial None signature, so an
    # empty map with an unchanged signature (no served_models yet, or only malformed ids) is
    # a valid steady state and is correctly skipped.
    entity_map_signature = frozenset(
        (
            mp.workspace,
            mp.name,
            tuple((sm.model_entity_id, sm.served_model_name) for sm in (mp.served_models or [])),
        )
        for mp in model_providers
    )
    if entity_map_signature != model_cache._entity_map_signature:
        model_cache.rebuild_model_entity_map()
        model_cache._entity_map_signature = entity_map_signature
    else:
        logger.debug("Provider layer unchanged since last cycle; skipping model entity map rebuild")
    if model_entity_getter is not None:
        try:
            model_entities = await model_entity_getter()
            model_cache.update_model_entity_metadata(model_entities)
        except Exception:
            logger.exception(
                "Failed to refresh ModelEntity metadata; preserving previous metadata where available and leaving "
                "backend_format unset for unknown models"
            )
    logger.debug(f"Updated model provider cache with {len(model_providers)} providers")

    if virtual_model_cache is not None:
        try:
            await refresh_virtual_model_cache(virtual_model_cache, secrets_sdk, registry=middleware_registry)
        except VirtualModelCacheRefreshError:
            logger.exception("Failed to refresh VirtualModel cache; stale entries will be used until next cycle")


async def refresh_model_provider_info(
    model_cache: ModelCache,
    model_info: ModelProviderInfo,
    secrets_sdk: AsyncNeMoHelix,
):
    now = datetime.now()
    model_provider = model_info.model_provider

    api_key_secret_name = model_provider.api_key_secret_name
    secret_value_diff = now - model_info.secret_value_updated_at
    if api_key_secret_name and (secret_value_diff.total_seconds() > model_cache.secret_value_ttl):
        await _refresh_secret_value(model_info, secrets_sdk, api_key_secret_name)
        model_info.secret_value_updated_at = now

    model_cache.update_model_info(model_info)


async def _refresh_secret_value(model_info: ModelProviderInfo, secrets_sdk: AsyncNeMoHelix, api_key_secret_name: str):
    """
    For a given model provider, update the cached secret value. This function
    mutates the passed-in model_info.
    """
    model_provider = model_info.model_provider
    try:
        model_info.secret_value = await retrieve_secret_value(
            workspace=model_provider.workspace,
            secret_name=api_key_secret_name,
            secrets_sdk=secrets_sdk,
        )
        logger.debug(f"Updated secret cache for model provider: {model_provider.workspace}/{model_provider.name}")
    except Exception:
        logger.exception(
            f"Failed to update secret cache for model provider: {model_provider.workspace}/{model_provider.name}."
        )


async def _async_pause(delay: float) -> None:
    await asyncio.sleep(delay)


async def refresh_model_cache_task(
    model_cache: ModelCache,
    model_provider_getter: Callable[[], Awaitable[list[ModelProvider]]],
    secrets_sdk: AsyncNeMoHelix,
    sleep_duration_s: int,
    max_consecutive_failures: int = 10,
    model_entity_getter: Callable[[], Awaitable[list[ModelEntity]]] | None = None,
    virtual_model_cache: VirtualModelCache | None = None,
    middleware_registry: MiddlewareRegistry | None = None,
) -> None:
    """Background task to periodically refresh the model provider cache.

    Args:
        model_cache: The cache to refresh
        model_provider_getter: Function to fetch model providers
        secrets_sdk: SDK client for accessing secrets service
        sleep_duration_s: Interval between refresh attempts in seconds
        max_consecutive_failures: Maximum number of consecutive failures before raising an exception

    Raises:
        ModelProviderRefreshError: If max_consecutive_failures is exceeded
    """
    logger.info(f"Starting model provider cache refresh task (interval: {sleep_duration_s}s)")
    consecutive_failures = 0

    while True:
        await _async_pause(sleep_duration_s)
        logger.debug("Starting model provider cache refresh cycle")
        try:
            await refresh_model_cache(
                model_cache,
                model_provider_getter,
                secrets_sdk,
                model_entity_getter=model_entity_getter,
                virtual_model_cache=virtual_model_cache,
                middleware_registry=middleware_registry,
            )
            logger.debug("Model provider cache refresh completed successfully")
            # Reset failure counter on success
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.exception(
                f"Error trying to refresh model provider cache "
                f"(consecutive failures: {consecutive_failures}/{max_consecutive_failures})"
            )

            if consecutive_failures >= max_consecutive_failures:
                error_msg = (
                    f"Model provider cache refresh failed {consecutive_failures} consecutive times. "
                    f"Models API appears to be persistently unavailable."
                )
                logger.error(error_msg)
                raise ModelProviderRefreshError(error_msg) from e
