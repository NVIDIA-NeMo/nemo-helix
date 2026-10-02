# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for namespace support in config registry."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from nhx.common.entities import DEFAULT_WORKSPACE
from nhx.guardrails.app.services.configs.registry import ConfigRegistry


class TestConfigNamespaceSupport:
    """Tests for namespace support in config registry"""

    pass


class TestConfigNamespaceAsyncSupport(unittest.IsolatedAsyncioTestCase):
    """Tests for async namespace support in config registry"""

    async def test_get_multiple_configs_with_different_namespaces(self):
        """Test getting multiple configs from different namespaces"""
        # Create a mock entities client
        mock_entities_client = AsyncMock()

        registry = ConfigRegistry(entities_client=mock_entities_client)

        nvidia_config = MagicMock()
        nvidia_config.namespace = "nvidia"
        nvidia_config.name = "config1"

        default_config = MagicMock()
        default_config.namespace = DEFAULT_WORKSPACE
        default_config.name = "config2"

        # _get_single_config to return our configs
        with patch.object(
            registry, "_get_single_config", side_effect=[nvidia_config, default_config]
        ) as mock_get_single:
            _ = await registry._get_multiple_config(["nvidia/config1", "default/config2"])

            assert mock_get_single.call_count == 2
            mock_get_single.assert_any_call("nvidia/config1")
            mock_get_single.assert_any_call("default/config2")
