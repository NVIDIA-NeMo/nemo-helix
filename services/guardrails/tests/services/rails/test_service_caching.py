# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.services.rails.service import LLMRails, RailsService
from nhx.guardrails.app.utils.hash_utils import compute_token_headers_hash
from nhx.guardrails.entities.values._private import Model, RailsConfig


class TestRailsServiceCaching(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config_ids = ["test-config"]
        self.rails_registry = RailsRegistry()
        #
        # Initialize RailsService with registry
        self.rails_service = RailsService(rails_registry=MagicMock(), config_registry=MagicMock())
        self.rails_service._rails_registry = self.rails_registry

        # Mock LLMRails with a mock that includes 'events_history_cache'
        self.llm_rails_patcher = patch("nhx.guardrails.app.services.rails.service.LLMRails")
        self.mock_llm_rails_class = self.llm_rails_patcher.start()
        self.mock_llm_rails_class.side_effect = self.create_mock_llm_rails

        # Create a real RailsConfig for proper model_dump() behavior
        self.mock_original_config = RailsConfig(models=[Model(type="main", model="main", engine="nim")])

        # Mock the config loading in ConfigRegistry with AsyncMock
        self.config_registry_patcher = patch.object(
            self.rails_service._config_registry,
            "get",
            new_callable=AsyncMock,
            return_value=self.mock_original_config,
        )
        self.mock_config_registry_get = self.config_registry_patcher.start()

        # Mock the config cache entry to return a mock with old datetime so rails cache is always newer
        mock_config_cache_entry = MagicMock()
        mock_config_cache_entry.created_at = datetime(2000, 1, 1, tzinfo=timezone.utc)  # Very old date
        self.config_cache_entry_patcher = patch.object(
            self.rails_service._config_registry,
            "get_cache_entry",
            return_value=mock_config_cache_entry,
        )
        self.mock_config_cache_entry = self.config_cache_entry_patcher.start()

    def tearDown(self):
        self.llm_rails_patcher.stop()
        self.config_registry_patcher.stop()
        self.config_cache_entry_patcher.stop()

    def create_mock_llm_rails(self, config):
        mock_llm = MagicMock(spec=LLMRails)
        # Manually add 'events_history_cache' attribute
        mock_llm.events_history_cache = {}
        return mock_llm

    async def test_caching_with_different_tokens(self):
        """Test that caching differentiates between different tokens."""

        # 1. Set a valid token and compute its hash
        valid_token = "valid_token_12345"
        token_hash_valid = compute_token_headers_hash(valid_token)

        engine = "nim"
        model_name = "main"
        model = Model(engine=engine, model=model_name, type="main")

        # Call `get_rails` with the valid token
        rails_instance_valid = await self.rails_service.get_rails(
            config_ids=self.config_ids,
            model=model,
            req_headers_cache_key=token_hash_valid,
        )

        # ensure that LLMRails was instantiated once
        self.mock_llm_rails_class.assert_called_once()

        # Ensure the rails instance was added to the registry
        cached_instance_valid = self.rails_registry.get(self.config_ids, engine, model_name, token_hash_valid)
        assert cached_instance_valid is not None
        assert cached_instance_valid == rails_instance_valid

        # Call `get_rails` with the invalid token - should create a new instance
        invalid_token = "invalid_token_67890"
        token_hash_invalid = compute_token_headers_hash(invalid_token)

        rails_instance_invalid = await self.rails_service.get_rails(
            config_ids=self.config_ids,
            model=model,
            req_headers_cache_key=token_hash_invalid,
        )

        # ensure that LLMRails was instantiated twice - once for valid, once for invalid token
        assert self.mock_llm_rails_class.call_count == 2

        # ensure that the two instances are different (not cached)
        assert rails_instance_valid != rails_instance_invalid

    async def test_caching_with_same_token(self):
        """Test that the same rails instance is returned for the same token hash"""

        valid_token = "same_token_11111"
        token_hash_valid = compute_token_headers_hash(valid_token)

        engine = "nim"
        model_name = "main"
        model = Model(engine=engine, model=model_name, type="main")

        # Call `get_rails` the first time
        rails_instance_first = await self.rails_service.get_rails(
            config_ids=self.config_ids,
            model=model,
            req_headers_cache_key=token_hash_valid,
        )

        # Call `get_rails` the second time with the same token
        rails_instance_second = await self.rails_service.get_rails(
            config_ids=self.config_ids,
            model=model,
            req_headers_cache_key=token_hash_valid,
        )

        # LLMRails should only be instantiated once
        self.mock_llm_rails_class.assert_called_once()

        # Rails instance should be retrieved from the cache in the second call
        assert rails_instance_first == rails_instance_second
        assert rails_instance_first is not None
        assert rails_instance_second is not None


if __name__ == "__main__":
    unittest.main()
