# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

import pytest
from nemoguardrails import LLMRails
from nhx.guardrails.app.services.rails.registry import RailsRegistry
from nhx.guardrails.app.utils.hash_utils import compute_token_headers_hash


class TestRailsRegistry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Initialize the singleton RailsRegistry
        self.registry = RailsRegistry()

        # Mock LLMRails
        self.llm_rails_patcher = patch("nhx.guardrails.app.services.rails.registry.LLMRails")
        self.mock_llm_rails_class = self.llm_rails_patcher.start()
        self.mock_llm_rails_class.side_effect = self.create_mock_llm_rails

    def tearDown(self):
        self.llm_rails_patcher.stop()

    def create_mock_llm_rails(self, *args, **kwargs):
        """
        Helper method to create a mock LLMRails instance with 'events_history_cache' attribute.
        """
        mock_llm = MagicMock(spec=LLMRails)
        mock_llm.events_history_cache = {}
        return mock_llm

    async def test_add_and_get(self):
        """
        Test adding and retrieving a LLMRails instance from the registry.
        """
        mock_rails = self.create_mock_llm_rails()
        token_hash = compute_token_headers_hash("test_token")

        # Add the mock_rails to the registry
        self.registry.add(
            config_ids=["default/config1"],
            engine="engine1",
            model_name="model1",
            token_hash=token_hash,
            llm_rails=mock_rails,
        )

        # Retrieve the rails instance
        retrieved_rails = self.registry.get(
            config_ids=["default/config1"],
            engine="engine1",
            model_name="model1",
            token_hash=token_hash,
        )

        assert retrieved_rails is not None
        assert retrieved_rails is mock_rails

    async def test_get_nonexistent(self):
        """
        Test that retrieving a non-existent LLMRails instance raises a KeyError.
        """
        with pytest.raises(KeyError):
            self.registry.get(
                config_ids=["default/nonexistent"],
                engine="engine1",
                model_name="model1",
                token_hash=None,
            )

    async def test_contains(self):
        """
        Test the 'contains' method of RailsRegistry.

        IDs reaching the registry are always fully-qualified (workspace/name) — normalization
        happens upstream in the request handler, not here.
        """
        mock_rails = self.create_mock_llm_rails()
        token_hash = compute_token_headers_hash("test_token")

        # Add to the registry using fully-qualified IDs
        self.registry.add(
            config_ids=["default/config1"],
            engine="engine1",
            model_name="model1",
            token_hash=token_hash,
            llm_rails=mock_rails,
        )

        assert self.registry.contains(config_ids=["default/config1"])
        assert not self.registry.contains(config_ids=["nvidia/config1"])

        self.registry.add(
            config_ids=["nvidia/config3"],
            engine="engine3",
            model_name="model3",
            token_hash=token_hash,
            llm_rails=mock_rails,
        )

        self.registry.add(
            config_ids=["default/config2"],
            engine="engine2",
            model_name="model2",
            token_hash=token_hash,
            llm_rails=mock_rails,
        )

        self.registry.add(
            config_ids=["nvidia/config2"],
            engine="nv_engine2",
            model_name="nv_model2",
            token_hash=token_hash,
            llm_rails=mock_rails,
        )

        assert self.registry.contains(config_ids=["default/config2"])
        assert self.registry.contains(config_ids=["nvidia/config2"])

        # Configs from different workspaces are distinct cache entries
        assert self.registry.get_cache(["nvidia/config3"]) != self.registry.get_cache(["default/config2"])
        assert self.registry.get_cache(["default/config2"]) != self.registry.get_cache(["nvidia/config2"])

        assert self.registry.contains(config_ids=["nvidia/config3"])
        assert not self.registry.contains(config_ids=["default/config3"])

        # Check for non-existent entry
        assert not self.registry.contains(config_ids=["default/nonexistent"])

    async def test_setitem(self):
        """
        Test the '__setitem__' method of RailsRegistry.
        """
        mock_rails = self.create_mock_llm_rails()

        # Set using __setitem__
        self.registry.set_cache(
            config_ids=["default/configx"],
            value={"enginex_modelx": mock_rails},
        )

        # Retrieve to verify
        assert self.registry.get_cache(["default/configx"])["enginex_modelx"].llm_rails == mock_rails

    async def test_delitem(self):
        """
        Test the '__delitem__' method of RailsRegistry.
        """
        mock_rails = self.create_mock_llm_rails()
        token_hash = compute_token_headers_hash("test_token")

        # Add to the registry
        self.registry.add(
            config_ids=["default/config1"],
            engine="engine1",
            model_name="model1",
            token_hash=token_hash,
            llm_rails=mock_rails,
        )

        # Delete the entry
        self.registry.delete(["default/config1"])

        # Attempt to retrieve should raise KeyError

        with pytest.raises(KeyError):
            self.registry.get_cache(["default/config1"])


if __name__ == "__main__":
    unittest.main()
