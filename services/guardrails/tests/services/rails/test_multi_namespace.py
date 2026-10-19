# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nhx.guardrails.app.services.configs.registry import ConfigRegistry
from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.services.rails.service import RailsService
from nhx.guardrails.app.services.utils import normalize_config_ids
from nhx.guardrails.app.utils.hash_utils import compute_token_headers_hash
from nhx.guardrails.entities.values._private import Model, RailsConfig


class TestNamespaceFunctions:
    """Tests for basic namespace functions"""

    def test_normalize_config_ids(self):
        """Test normalizing config IDs with different namespaces"""

        config_ids = ["config1", "nvidia/config2", "default/config3"]
        normalized = normalize_config_ids(config_ids, default_workspace="default")
        assert "default/config1" in normalized
        assert "nvidia/config2" in normalized
        assert "default/config3" in normalized


class TestRailsServiceNamespaceSupport(unittest.IsolatedAsyncioTestCase):
    """Tests for RailsService with namespace support"""

    def setUp(self):
        self.mock_config_registry = AsyncMock(spec=ConfigRegistry)
        self.mock_rails_registry = MagicMock(spec=RailsRegistry)
        self.rails_service = RailsService(
            config_registry=self.mock_config_registry, rails_registry=self.mock_rails_registry
        )

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
        self.config_cache_entry_patcher.stop()

    @patch("nhx.guardrails.app.services.rails.service.LLMRails")
    async def test_get_rails_with_namespaced_config(self, mock_llm_rails):
        """Test getting rails with namespaced configs"""

        config_ids = ["nvidia/test-config"]
        model_name = "test-model"
        engine = "test-engine"
        token = "test-token"
        model = Model(model=model_name, type="main", engine=engine)
        token_hash = compute_token_headers_hash(token)

        mock_config = RailsConfig(models=[Model(type="main", model=model_name, engine=engine)])
        mock_rails = MagicMock()

        self.mock_config_registry.get.return_value = mock_config
        mock_llm_rails.return_value = mock_rails
        self.mock_rails_registry.get_cache_entry.return_value = None

        result = await self.rails_service.get_rails(
            config_ids=config_ids, model=model, req_headers_cache_key=token_hash
        )

        self.mock_config_registry.get.assert_called_once()
        mock_llm_rails.assert_called_once()
        self.mock_rails_registry.add.assert_called_once()

        assert result == mock_rails

    async def test_get_rails_with_multiple_namespaced_configs(self):
        """Test getting rails with multiple namespaced configs"""

        config_ids = ["nvidia/test-config", "default/test-config2"]
        model_name = "test-model"
        engine = "test-engine"
        model = Model(model=model_name, type="main", engine=engine)

        mock_rails = MagicMock()
        mock_cache_entry = MagicMock()
        mock_cache_entry.llm_rails = mock_rails
        mock_cache_entry.created_at = datetime.now(timezone.utc)

        self.mock_rails_registry.get_cache_entry.return_value = mock_cache_entry

        result = await self.rails_service.get_rails(config_ids=config_ids, model=model)

        self.mock_rails_registry.get_cache_entry.assert_called_once()
        self.mock_config_registry.get.assert_not_called()

        assert result == mock_rails
