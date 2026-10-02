# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from nemoguardrails import LLMRails
from nhx.guardrails.app.utils.key_generator import generate_key

logger = logging.getLogger(__name__)


@dataclass
class RailsCacheEntry:
    llm_rails: LLMRails
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RailsRegistry:
    """A registry for LLMRails instances.

    This class is a singleton that stores LLMRails instances for a given config id.

    Attributes:
        _llm_rails_instances (dict): A dictionary that stores LLMRails instances for a given config id.
        _llm_rails_events_history_cache (dict): A dictionary that stores the events history cache for a given config id.

    Raises:
        KeyError: If no rails instance is found for the given config id.
    """

    _llm_rails_instances: Dict[str, Dict[str, RailsCacheEntry]] = defaultdict(dict)
    _llm_rails_events_history_cache: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(dict)

    def get(
        self,
        config_ids: List[str],
        engine: str,
        model_name: str,
        token_hash=None,
    ) -> LLMRails | None:
        cache_key = self._generate_cache_key(config_ids)

        if cache_key not in self._llm_rails_instances:
            raise KeyError(f"No rails instance found for config_ids {config_ids}")

        model_key = generate_model_key(engine, model_name, token_hash)

        if model_key not in self._llm_rails_instances[cache_key]:
            return None

        llm_rails = self._llm_rails_instances[cache_key][model_key].llm_rails
        llm_rails.events_history_cache = self._llm_rails_events_history_cache[cache_key][model_key] or {}

        return llm_rails

    def add(
        self,
        config_ids: List[str],
        engine: str,
        model_name: str,
        token_hash,
        llm_rails: LLMRails,
    ):
        model_key = generate_model_key(engine, model_name, token_hash)

        cache_key = self._generate_cache_key(config_ids)

        logger.debug(f"Adding rails instance for config_ids {config_ids} and model key {model_key}")

        self._llm_rails_instances[cache_key][model_key] = RailsCacheEntry(llm_rails=llm_rails)

        self._llm_rails_events_history_cache[cache_key][model_key] = llm_rails.events_history_cache

    def _generate_cache_key(self, config_ids: List[str]) -> str:
        """Generates a cache key for the given config ids."""
        return generate_key(value=config_ids)

    def update_cache_history(self, config_ids: List[str], value: Dict[str, dict]):
        """Updates the cache with the given config ids."""

        key = self._generate_cache_key(config_ids)
        for model_key in self._llm_rails_events_history_cache[key]:
            self._llm_rails_events_history_cache[key][model_key] = value[model_key]

    def get_cache_history(self, config_ids: List[str]) -> Dict[str, dict]:
        """Returns the cache history for the given config ids."""
        key = self._generate_cache_key(config_ids)
        return self._llm_rails_events_history_cache[key]

    def invalidate_config_cache(self, config_id: List[str]):
        """Invalidates the config cache and updates the cache history."""

        if self.contains(config_id):
            rails_instances = self.get_cache(config_id)
            self.delete(config_id)
            if rails_instances:
                val = self.get_cache_history(config_id)
                # We save the events history cache, to restore it on the new instance
                self.update_cache_history(config_id, val)

    def get_cache(self, config_ids: List[str]):
        """Returns the cache."""
        key = self._generate_cache_key(config_ids)
        if key not in self._llm_rails_instances:
            raise KeyError(f"No rails instance found for config_ids {config_ids}")
        return self._llm_rails_instances[key]

    def get_cache_entry(
        self,
        config_ids: List[str],
        engine: str,
        model_name: str,
        token_hash: Optional[str],
    ) -> Optional[RailsCacheEntry]:
        """
        Returns the cache entry for the given config ids and model key.
        The model key is generated based on the following fields:
          - engine
          - req.body.model
          - request custom headers (that start with 'x')
          - token (if present)
        """
        key = self._generate_cache_key(config_ids)
        model_key = generate_model_key(engine, model_name, token_hash)
        if entry := self._llm_rails_instances[key].get(model_key):
            return entry

        logger.debug(f"No rails instance found for config_ids {config_ids} and model key {model_key}")

    def set_cache(self, config_ids: List[str], value: Dict[str, LLMRails]):
        """Sets the cache."""
        key = self._generate_cache_key(config_ids)
        self._llm_rails_instances[key] = {
            model_key: RailsCacheEntry(llm_rails=llm_rails) for model_key, llm_rails in value.items()
        }

    def contains(self, config_ids: List[str]) -> bool:
        cache_key = self._generate_cache_key(config_ids)
        return cache_key in self._llm_rails_instances

    def delete(self, config_ids: List[str]):
        key = self._generate_cache_key(config_ids)
        if key not in self._llm_rails_instances:
            raise KeyError(f"No rails instance found for config_ids {config_ids}")
        del self._llm_rails_instances[key]


def generate_model_key(
    engine: str,
    model_name: str,
    token_hash: str | None = None,
):
    return engine + "_" + model_name + ("_" + token_hash if token_hash else "")


def get_rails_registry_instance() -> RailsRegistry:
    return RailsRegistry()
