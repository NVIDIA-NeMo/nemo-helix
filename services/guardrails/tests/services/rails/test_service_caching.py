# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemoguardrails import LLMRails
from nhx.common.entities import EntityNotFoundError
from nhx.guardrails.app.exceptions.application_exceptions import GuardrailConfigurationNotFoundError
from nhx.guardrails.app.services.configs.registry import ConfigRegistry
from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.services.rails.service import RailsService
from nhx.guardrails.app.utils.hash_utils import compute_token_headers_hash
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import Model, RailsConfig


class TestRailsServiceCaching(unittest.IsolatedAsyncioTestCase):
    """Engine reuse across requests, wired the way the API wires it.

    The API builds a new ConfigRegistry (with the caller's entity client) for every request and
    shares one RailsRegistry, so each ``get_rails`` call here goes through a fresh RailsService
    and ConfigRegistry. Mocking a config-cache hit instead is what hid the engine being rebuilt
    on every request.
    """

    def setUp(self):
        # RailsRegistry keeps its cache at class level, so give each test its own config.
        self.config_id = f"default/test-config-{uuid.uuid4().hex[:8]}"
        self.config_ids = [self.config_id]
        self.rails_registry = RailsRegistry()

        self.stored = self._stored_config(entity_id="config-1", db_version=1)
        self.entities_client = MagicMock()
        self.entities_client.get = AsyncMock(side_effect=lambda *args, **kwargs: self.stored)

        self.create_engine_patcher = patch("nhx.guardrails.app.services.rails.service.create_rails_engine")
        self.mock_create_engine = self.create_engine_patcher.start()
        self.mock_create_engine.side_effect = self.create_mock_llm_rails

        self.model = Model(engine="nim", model="main", type="main")

    def tearDown(self):
        self.create_engine_patcher.stop()

    def _stored_config(self, entity_id: str, db_version: int) -> GuardrailConfig:
        name = self.config_id.split("/")[1]
        config = GuardrailConfig(
            name=name, workspace="default", data=RailsConfig(models=[Model(type="main", model="main", engine="nim")])
        )
        config._id = entity_id
        config._db_version = db_version
        return config

    def create_mock_llm_rails(self, config):
        mock_llm = MagicMock(spec=LLMRails)
        mock_llm.events_history_cache = {}
        return mock_llm

    async def get_rails(self, token: str = "token"):
        """One request: a new ConfigRegistry per call, the shared RailsRegistry."""
        service = RailsService(
            config_registry=ConfigRegistry(entities_client=self.entities_client),
            rails_registry=self.rails_registry,
        )
        return await service.get_rails(
            config_ids=self.config_ids,
            model=self.model,
            req_headers_cache_key=compute_token_headers_hash(token),
        )

    async def test_reuses_engine_across_requests(self):
        first = await self.get_rails()
        second = await self.get_rails()

        self.mock_create_engine.assert_called_once()
        assert first is second

    async def test_reads_config_as_caller_on_every_request(self):
        await self.get_rails()
        await self.get_rails()

        assert self.entities_client.get.await_count == 2

    async def test_caching_with_different_tokens(self):
        """Engines are cached per caller credential, so different tokens never share one."""
        valid = await self.get_rails(token="valid_token_12345")
        invalid = await self.get_rails(token="invalid_token_67890")

        assert self.mock_create_engine.call_count == 2
        assert valid is not invalid

    async def test_rebuilds_when_config_is_updated(self):
        first = await self.get_rails()
        self.stored = self._stored_config(entity_id="config-1", db_version=2)
        second = await self.get_rails()

        assert self.mock_create_engine.call_count == 2
        assert first is not second
        # The newer engine is the one kept.
        assert await self.get_rails() is second
        assert self.mock_create_engine.call_count == 2

    async def test_rebuilds_when_config_is_recreated_under_the_same_name(self):
        """A recreated config restarts at db_version 1; its new entity id must still miss."""
        first = await self.get_rails()
        self.stored = self._stored_config(entity_id="config-2", db_version=1)
        second = await self.get_rails()

        assert self.mock_create_engine.call_count == 2
        assert first is not second

    async def test_caller_who_cannot_read_config_never_gets_cached_engine(self):
        await self.get_rails()
        self.entities_client.get = AsyncMock(side_effect=EntityNotFoundError("not found"))

        with pytest.raises(GuardrailConfigurationNotFoundError):
            await self.get_rails()
        self.mock_create_engine.assert_called_once()


if __name__ == "__main__":
    unittest.main()
