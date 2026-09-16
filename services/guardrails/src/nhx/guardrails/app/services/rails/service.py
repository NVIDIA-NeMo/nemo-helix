# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import List, Union

from nemoguardrails import LLMRails
from nhx.guardrails.app.handlers.utils import get_main_model_from_rails_config, to_internal_rails_config
from nhx.guardrails.app.services.configs.registry import ConfigRegistry
from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.utils.config_utils import configure_rails_config
from nhx.guardrails.app.utils.context_utils import set_main_model_into_context
from nhx.guardrails.entities.values._private import Model, RailsConfig

logger = logging.getLogger(__name__)


class RailsService:
    def __init__(self, rails_registry: RailsRegistry, config_registry: ConfigRegistry):
        self._rails_registry = rails_registry
        self._config_registry = config_registry

    async def get_config(self, config_ids: Union[str, List[str]]) -> RailsConfig:
        """Returns the rails instance for the given config id."""
        return await self._config_registry.get(config_ids=config_ids)

    async def get_rails(
        self,
        config_ids: List[str],
        model: Model,
        req_headers_cache_key: str | None = None,
    ) -> LLMRails:
        """Returns the rails instance for the given config id."""

        engine = model.engine or "nim"
        model_name = model.model or "main"

        if cached_rails_entry := self._rails_registry.get_cache_entry(
            config_ids, engine, model_name, req_headers_cache_key
        ):
            cached_config_entry = self._config_registry.get_cache_entry(config_ids)
            if cached_config_entry and cached_rails_entry.created_at >= cached_config_entry.created_at:
                logger.debug(
                    f"Found cached rails entry for config_ids {config_ids}, engine {engine}, model_name {model_name}, req_headers_cache_key {req_headers_cache_key}"
                )
                return cached_rails_entry.llm_rails

        base_config = await self.get_config(config_ids)
        rails_config = configure_rails_config(base_config, model)

        # Set the main model into context, so we can extract relevant information (ex. base URL) at inference time
        main_model = get_main_model_from_rails_config(rails_config)
        set_main_model_into_context(main_model)

        try:
            # Run LLMRails init in a thread to avoid blocking the event loop.
            # LLMRails.__init__ has blocking file I/O and a blocking thread.join() for KB init.
            llm_rails = await asyncio.to_thread(LLMRails, config=to_internal_rails_config(rails_config))
        except Exception as e:
            logger.error(f"Failed to instantiate LLMRails instance for config {config_ids} and model {model_name}")
            raise e

        self._rails_registry.add(config_ids, engine, model_name, req_headers_cache_key, llm_rails)

        return llm_rails
