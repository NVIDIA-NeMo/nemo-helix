# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config registry for managing guardrail configurations."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Union

from nhx.common.entities import EntityNotFoundError
from nhx.common.entities.client import EntityClient
from nhx.common.entities.utils import parse_entity_ref
from nhx.guardrails.app.exceptions.application_exceptions import GuardrailConfigurationNotFoundError
from nhx.guardrails.app.services.configs.sources import get_config
from nhx.guardrails.app.utils.key_generator import generate_key
from nhx.guardrails.config import settings
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import RailsConfig

logger = logging.getLogger(__name__)


@dataclass
class ConfigCacheEntry:
    key: str
    config: RailsConfig
    config_ids: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_older_than(self, threshold: timedelta) -> bool:
        return datetime.now(timezone.utc) - self.created_at > threshold


class ConfigRegistry:
    """ConfigRegistry configuration objects.
    It provides a simple API to register and retrieve
        configuration objects. It also provides a method to load all configurations
        from a specified directory.

    Raises:
        ValueError: Value error if configurations are not found or invalid.
    """

    def __init__(
        self,
        entities_client: Optional[EntityClient] = None,
        ttl: int = settings.config_cache_ttl,
        staleness_threshold: int = settings.config_cache_staleness_threshold,
    ):
        """
        Args:
            entities_client (Optional[EntityClient]): The entity client to use.
            ttl (int): How long a config entry remains in the cache since it was created.
        """
        self._configs: Dict[str, ConfigCacheEntry] = {}
        self.entities_client = entities_client
        self.ttl = timedelta(seconds=ttl)
        self.staleness_threshold = timedelta(seconds=staleness_threshold)

        self._cache_lock = asyncio.Lock()
        # Tracks configs that are actively being added to the cache.
        # Key: Config ID(s)
        # Value: Event that indicates the cache is being written to. Subsequent requests to get this config will wait
        # for this event to finish, then access the cache, rather than directly querying the database for the config.
        self._cache_loading_events: Dict[str, asyncio.Event] = {}

        logger.info(f"ConfigRegistry initialized with TTL {ttl} seconds")

    async def register(self, config: RailsConfig, config_ids: Union[str, List[str]]):
        if isinstance(config_ids, str):
            config_ids = [config_ids]

        key = self._generate_cache_key(config_ids=config_ids)
        async with self._cache_lock:
            self._configs[key] = ConfigCacheEntry(key=key, config=config, config_ids=config_ids)

    def _build_combined_config(self, config_ids: List[str], db_configs: Dict[str, GuardrailConfig]) -> RailsConfig:
        """Build a combined RailsConfig from multiple database configs."""
        full_llm_rails_config = None

        for config_id in config_ids:
            if config_id not in db_configs:
                raise ValueError(f"Config {config_id} not found in database")

            config_obj = db_configs[config_id]
            rails_config = self._make_rails_config(config_obj)

            if full_llm_rails_config is None:
                full_llm_rails_config = rails_config
            elif rails_config is not None:
                full_llm_rails_config += rails_config

        if full_llm_rails_config is None:
            raise ValueError("Failed to build combined config - no valid configs found")

        return full_llm_rails_config

    async def refresh_all(self):
        """Refresh all configs in the cache."""
        if self._cache_lock.locked():
            logger.debug("Refresh already in progress, skipping")
            return

        if not self.entities_client:
            logger.warning("No entities client available, skipping refresh")
            return

        async with self._cache_lock:
            if not self._configs:
                return

            logger.debug(f"Starting refresh of {len(self._configs)} cached configs")

            # Collect all unique config_ids from cache entries (already fully-qualified workspace/name)
            all_config_ids = {config_id for entry in self._configs.values() for config_id in entry.config_ids}

            # Fetch all currently cached configs from the database
            db_configs: Dict[str, GuardrailConfig] = {}

            for config_id in all_config_ids:
                ref = parse_entity_ref(config_id)
                try:
                    config = await self.entities_client.get(GuardrailConfig, name=ref.name, workspace=ref.workspace)
                    db_configs[config_id] = config
                except Exception:
                    # Config may have been deleted
                    pass

            # Now process each cache entry
            keys_to_delete: List[str] = []
            for entry in self._configs.values():
                # Check if any config in this entry is missing from database
                missing_configs = [cid for cid in entry.config_ids if cid not in db_configs]
                if missing_configs:
                    logger.debug(f"Config {entry.key} has missing configs {missing_configs}, marking for removal")
                    keys_to_delete.append(entry.key)
                    continue

                configs_changed = any(
                    db_configs[cid].updated_at and db_configs[cid].updated_at > entry.created_at
                    for cid in entry.config_ids
                    if cid in db_configs
                )
                if configs_changed:
                    logger.debug(f"Config {entry.key} changed, rebuilding config")
                    # Rebuild the combined config directly from database configs
                    try:
                        new_combined_config = self._build_combined_config(entry.config_ids, db_configs)
                        self._configs[entry.key] = ConfigCacheEntry(
                            key=entry.key, config=new_combined_config, config_ids=entry.config_ids
                        )
                        logger.debug(
                            f"Config {entry.key} {'updated' if configs_changed else 'up to date'}, keeping for {(self.ttl - (datetime.now(timezone.utc) - entry.created_at)).total_seconds()} seconds"
                        )
                        continue
                    except Exception as e:
                        logger.error(f"Failed to rebuild config entry {entry.key}: {e}")
                        # If rebuild fails, mark for deletion so it gets rebuilt on next request
                        keys_to_delete.append(entry.key)
                        continue

                if entry.is_older_than(self.ttl):
                    logger.debug(f"Config {entry.key} has expired, marking for removal")
                    keys_to_delete.append(entry.key)
                    continue

            # Clean up expired/invalid entries
            original_count = len(self._configs)
            for key in keys_to_delete:
                if key in self._configs:
                    del self._configs[key]
                    logger.debug(f"Removed config {key} from cache")

            if original_count != len(self._configs):
                logger.debug(f"Cache refresh completed: removed {original_count - len(self._configs)} entries")

    def _generate_cache_key(self, config_ids: Union[str, List[str]]) -> str:
        """Generates a cache key for the given config ids."""
        config_ids = config_ids if isinstance(config_ids, list) else [config_ids]
        return generate_key(value=config_ids)

    def get_cache_entry(self, config_ids: Union[str, List[str]]) -> Optional[ConfigCacheEntry]:
        """Get a cache entry for the given config ids.

        If there is no entry for the given config ids, None is returned.
        If the entry is older than the TTL, it is deleted from the cache and None is returned.

        Args:
            config_ids (Union[str, List[str]]): The config ids to get the cache entry for.

        """
        entry = self._configs.get(self._generate_cache_key(config_ids=config_ids))
        if entry is None or entry.is_older_than(self.ttl):
            # We should assume that an expired config will soon be removed, treat it as not found
            return None

        # If the entry is older than the TTL, delete it
        if entry.is_older_than(self.staleness_threshold):
            asyncio.create_task(self.refresh_all())

        return entry

    async def get(self, config_ids: Union[str, List[str]]) -> RailsConfig:
        """
        Retrieve configuration(s) based on the type of config_id_or_ids.

        Args:
            config_ids (str or List[str]): Single config ID or list of config IDs.

        Returns:
            RailsConfig: The retrieved RailsConfig object.

        Raises:
            TypeError: If config_ids is neither a string nor a list.
            ValueError: If configurations are not found or invalid.
            AttributeError: If expected attributes are missing.
        """
        # Normalize to list for consistent processing
        if isinstance(config_ids, str):
            config_list = [config_ids]
            logger.debug(f"Fetching single config: {config_ids}")
        elif isinstance(config_ids, list):
            config_list = config_ids
            logger.debug(f"Fetching multiple configs: {config_ids}")
        else:
            logger.error(f"Unsupported type for config_ids: {type(config_ids)}")
            raise TypeError("config_ids must be a string or a list of strings.")

        # Generate cache key for this request
        cache_key = self._generate_cache_key(config_ids=config_list)

        # Check if we have this combination in cache
        if entry := self.get_cache_entry(config_ids=config_list):
            logger.debug(f"Config {cache_key} found in cache")
            rails_config = entry.config
        else:
            # Check if this config is currently being loaded
            if cache_key in self._cache_loading_events:
                logger.debug(f"Waiting for config {cache_key} to be added to cache. Skipping loading from database.")
                await self._cache_loading_events[cache_key].wait()
                # After waiting, check cache again
                if entry := self.get_cache_entry(config_ids=config_list):
                    rails_config = entry.config
                else:
                    # This shouldn't happen if the loading completed successfully
                    raise ValueError(f"Config {cache_key} was being loaded but not found in cache after completion")
            else:
                logger.debug(
                    f"Config {cache_key} is neither in cache nor being loaded into cache. Loading from database."
                )
                # Start loading the config
                rails_config = await self._load_config_with_event(cache_key, config_list)

        # Validate config
        if rails_config.colang_version != "1.0":
            raise ValueError(
                "Guardrail configurations using Colang 2.0 is not supported (only Colang 1.0)",
            )

        return rails_config

    async def _load_config_with_event(self, cache_key: str, config_ids: List[str]) -> RailsConfig:
        """
        Returns the RailsConfig for the given config IDs.
        Before loading, creates an event to indicate the given config IDs are being loaded, so that any subsequent
        requests to get this config will wait for the event to finish before accessing the cache.
        """
        # Create and set the loading event
        loading_event = asyncio.Event()
        self._cache_loading_events[cache_key] = loading_event

        try:
            logger.debug(f"Loading config {cache_key} from database")
            rails_config = await self._get_multiple_config(config_ids)
            return rails_config
        finally:
            # Signal completion and clean up the event
            loading_event.set()
            # Remove the loading event
            await self._cleanup_loading_event(cache_key)

    async def _cleanup_loading_event(self, cache_key: str):
        """Clean up cache loading event for config with the given cache key."""
        if cache_key in self._cache_loading_events:
            del self._cache_loading_events[cache_key]

    def _make_rails_config(self, config_obj: GuardrailConfig) -> RailsConfig:
        """Make a RailsConfig object from a GuardrailConfig object."""
        raw_files_url = getattr(config_obj, "files_url", None)
        files_url = str(raw_files_url) if raw_files_url else None

        config_data = {}
        if config_obj.data is not None:
            # Convert RailsConfig to dict. Use exclude_unset to avoid including default values that may have extra fields
            if hasattr(config_obj.data, "model_dump"):
                config_data = config_obj.data.model_dump(exclude_unset=True, exclude_none=True)
            else:
                config_data = config_obj.data

        return get_config(files_url=files_url, config_data=config_data if config_data else None)

    async def _get_single_config(self, config_id: str) -> RailsConfig:
        """Get a single config by ID

        Args:
            config_id (str): The ID of the config to fetch

        Returns:
            RailsConfig: The RailsConfig object for the given ID

        Raises:
            ValueError: If the config is not found
        """
        if entry := self.get_cache_entry(config_ids=[config_id]):
            return entry.config

        if not self.entities_client:
            raise ValueError("No entities client available")

        # Get config from database
        ref = parse_entity_ref(config_id)
        try:
            config_obj = await self.entities_client.get(GuardrailConfig, name=ref.name, workspace=ref.workspace)
        except EntityNotFoundError as e:
            raise GuardrailConfigurationNotFoundError(
                config_id,
                f"Guardrail config not found for ID: {config_id}",
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to retrieve guardrail config: {config_id}: {e}") from e

        config = self._make_rails_config(config_obj)

        if config is not None:
            await self.register(config=config, config_ids=[config_id])
            return config

        raise ValueError(f"Config not found for ID: {config_id}. It exists in the database but could not be loaded.")

    async def _get_multiple_config(self, config_ids: list) -> RailsConfig:
        """Get multiple configs by ID"""
        full_llm_rails_config = None

        for config_id in config_ids:
            rails_config = await self._get_single_config(config_id)

            if full_llm_rails_config is None:
                full_llm_rails_config = rails_config
            elif rails_config is not None:
                full_llm_rails_config += rails_config

        # Store the combined config in cache
        if full_llm_rails_config is not None:
            await self.register(config=full_llm_rails_config, config_ids=config_ids)

        return full_llm_rails_config  # type: ignore

    def get_cache_stats(self) -> dict:
        """Get stats about the config cache."""
        return {
            "num_configs": len(self._configs),
            "num_cache_loading_events": len(self._cache_loading_events),
            "details": {
                entry.key: {
                    "config_ids": entry.config_ids,
                    "created_at": entry.created_at.isoformat(),
                }
                for entry in self._configs.values()
            },
        }

    def __repr__(self):
        return "ConfigRegistry()"


def get_config_registry_instance():
    return ConfigRegistry()
