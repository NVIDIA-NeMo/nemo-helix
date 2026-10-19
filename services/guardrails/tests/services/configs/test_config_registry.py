# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio  # noqa: F401
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from nhx.guardrails.app.services.configs.registry import ConfigCacheEntry, ConfigRegistry
from nhx.guardrails.app.utils.key_generator import generate_key
from nhx.guardrails.entities import GuardrailConfig
from nhx.guardrails.entities.values._private import RailsConfig


class TestConfigRegistry(unittest.TestCase):
    def setUp(self):
        self.config_registry = ConfigRegistry()
        self.mock_config = MagicMock(spec=RailsConfig)
        self.mock_config_id = "test-config-id"

    def test_generate_cache_key(self):
        key = self.config_registry._generate_cache_key(config_ids=[self.mock_config_id])
        expected_key = generate_key(value=[self.mock_config_id])
        assert key == expected_key

    def test_get_cache_entry_not_found(self):
        """Test that get_cache_entry returns None when entry doesn't exist."""
        entry = self.config_registry.get_cache_entry(config_ids=["non-existent"])
        assert entry is None

    def test_get_cache_entry_expired(self):
        """Test that get_cache_entry returns None for expired entries."""
        # Create an expired entry
        expired_entry = ConfigCacheEntry(
            key="test-key",
            config=self.mock_config,
            config_ids=[self.mock_config_id],
            created_at=datetime.now(timezone.utc) - timedelta(seconds=3700),  # Older than default TTL
        )

        # Manually add to cache
        key = self.config_registry._generate_cache_key(config_ids=[self.mock_config_id])
        self.config_registry._configs[key] = expired_entry

        # Should return None for expired entry
        entry = self.config_registry.get_cache_entry(config_ids=[self.mock_config_id])
        assert entry is None

    def test_get_cache_entry_stale_triggers_refresh(self):
        """Test that stale entries trigger background refresh but still return the entry."""
        # Create a stale entry (older than staleness_threshold but younger than TTL)
        # Default TTL is 60s, so staleness_threshold is 45s
        # Create entry that's 50 seconds old (stale but not expired)
        stale_entry = ConfigCacheEntry(
            key="test-key",
            config=self.mock_config,
            config_ids=[self.mock_config_id],
            created_at=datetime.now(timezone.utc)
            - timedelta(seconds=50),  # Older than staleness threshold (45s), younger than TTL (60s)
        )

        # Manually add to cache
        key = self.config_registry._generate_cache_key(config_ids=[self.mock_config_id])
        self.config_registry._configs[key] = stale_entry

        # Mock create_task to capture and properly close the coroutine to prevent warnings
        captured_coro = None

        def capture_and_close_coro(coro):
            nonlocal captured_coro
            captured_coro = coro
            coro.close()  # Properly close the coroutine to prevent warning
            return MagicMock()

        with patch("asyncio.create_task", side_effect=capture_and_close_coro) as mock_create_task:
            # Should return the stale entry (not None)
            entry = self.config_registry.get_cache_entry(config_ids=[self.mock_config_id])

            assert entry is not None, "Stale entry should be returned (not None)"
            assert entry.config == self.mock_config, "Returned entry should have correct config"

            # Should have triggered background refresh
            mock_create_task.assert_called_once()
            # Verify the coroutine was for refresh_all
            assert captured_coro is not None


class TestConfigRegistryAsync(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_entities_client = MagicMock()
        self.config_registry = ConfigRegistry(entities_client=self.mock_entities_client)
        self.mock_config = MagicMock(spec=RailsConfig)
        self.mock_config.colang_version = "1.0"
        self.mock_config.__add__ = MagicMock(return_value=self.mock_config)
        self.mock_config_id = "default/test-config-id"

    async def test_register_and_get_single_config(self):
        await self.config_registry.register(self.mock_config, [self.mock_config_id])
        key = self.config_registry._generate_cache_key(config_ids=[self.mock_config_id])
        assert key in self.config_registry._configs
        assert self.config_registry._configs[key].config == self.mock_config

    async def test_get_cache_entry_valid(self):
        """Test that get_cache_entry returns valid entries."""
        await self.config_registry.register(self.mock_config, [self.mock_config_id])
        entry = self.config_registry.get_cache_entry(config_ids=[self.mock_config_id])
        assert entry is not None
        assert entry.config == self.mock_config

    async def test_get_single_config(self):
        """Test getting a single config asynchronously."""
        config_obj = GuardrailConfig(name="test-config-id", workspace="default", data=None)
        self.mock_entities_client.get = AsyncMock(return_value=config_obj)

        # Mock the _make_rails_config method to return our mock config
        with patch.object(self.config_registry, "_make_rails_config", return_value=self.mock_config):
            # Test the get method with a fully-qualified config ID
            config = await self.config_registry.get(self.mock_config_id)
            assert config == self.mock_config

            # Verify entities_client was called
            self.mock_entities_client.get.assert_called_once()

    async def test_get_multiple_configs(self):
        """Test getting multiple configs asynchronously."""
        config_obj1 = GuardrailConfig(name="config-id-1", workspace="default", data=None)
        config_obj2 = GuardrailConfig(name="config-id-2", workspace="default", data=None)

        self.mock_entities_client.get = AsyncMock(side_effect=[config_obj1, config_obj2])

        # Create separate mock configs for multiple configs test
        mock_config1 = MagicMock(spec=RailsConfig)
        mock_config1.colang_version = "1.0"
        mock_config2 = MagicMock(spec=RailsConfig)
        mock_config2.colang_version = "1.0"

        # Mock the combined config result
        combined_config = MagicMock(spec=RailsConfig)
        combined_config.colang_version = "1.0"
        mock_config1.__add__ = MagicMock(return_value=combined_config)

        # Mock the _make_rails_config method to return our mock configs
        with patch.object(self.config_registry, "_make_rails_config", side_effect=[mock_config1, mock_config2]):
            # Test the get method with multiple fully-qualified configs
            config_ids = ["default/config-id-1", "default/config-id-2"]
            config = await self.config_registry.get(config_ids)
            assert config == combined_config

            # Verify entities_client was called twice
            assert self.mock_entities_client.get.call_count == 2

    async def test_get_config_invalid_colang_version(self):
        """Test that invalid colang version raises ValueError."""
        # Mock config with invalid colang version
        invalid_config = MagicMock(spec=RailsConfig)
        invalid_config.colang_version = "2.0"

        with patch.object(self.config_registry, "_get_single_config", return_value=invalid_config):
            with self.assertRaises(ValueError) as context:
                await self.config_registry.get("default/test-config")

            assert "Colang 2.0 is not supported" in str(context.exception)

    async def test_get_config_not_found_raises_guardrail_configuration_not_found_error(self):
        """Test that get raises GuardrailConfigurationNotFoundError when the entity store returns EntityNotFoundError."""
        from nhx.common.entities.client import EntityNotFoundError
        from nhx.guardrails.app.exceptions.application_exceptions import GuardrailConfigurationNotFoundError

        self.mock_entities_client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))

        with self.assertRaises(GuardrailConfigurationNotFoundError) as context:
            await self.config_registry.get("default/non-existent-config")

        assert "non-existent-config" in str(context.exception)

    async def test_get_config_db_entry_unloadable_raises_value_error(self):
        """Test that get raises ValueError when the config exists in the DB, but fails to load."""
        config_obj = GuardrailConfig(name="bad-config", workspace="default", data=None)
        self.mock_entities_client.get = AsyncMock(return_value=config_obj)

        # _make_rails_config returns None to simulate a config that exists but cannot be loaded
        with patch.object(self.config_registry, "_make_rails_config", return_value=None):
            with self.assertRaises(ValueError) as context:
                await self.config_registry.get("default/bad-config")

        assert "bad-config" in str(context.exception)

    async def test_get_cache_stats(self):
        """Test that get_cache_stats returns correct information."""
        await self.config_registry.register(self.mock_config, [self.mock_config_id])
        stats = self.config_registry.get_cache_stats()

        assert "num_configs" in stats
        assert "num_cache_loading_events" in stats
        assert "details" in stats
        assert stats["num_configs"] == 1
        assert stats["num_cache_loading_events"] == 0
        assert len(stats["details"]) == 1


class TestConfigRegistryConcurrentAccess(unittest.IsolatedAsyncioTestCase):
    """Test concurrent access to ConfigRegistry to ensure no duplicate entity client calls."""

    def setUp(self):
        self.mock_entities_client = MagicMock()
        self.config_registry = ConfigRegistry(entities_client=self.mock_entities_client)
        self.mock_config = MagicMock(spec=RailsConfig)
        self.mock_config.colang_version = "1.0"
        self.mock_config_id = "default/test-config-id"

    async def test_concurrent_cold_config_requests(self):
        """Test that multiple concurrent requests for a cold config result in only one entity client call."""
        # Create actual config object from entity store
        config_obj = GuardrailConfig(name="test-config-id", workspace="default", data=None)
        self.mock_entities_client.get = AsyncMock(return_value=config_obj)

        # Mock the _make_rails_config method to return our mock config
        with patch.object(self.config_registry, "_make_rails_config", return_value=self.mock_config):
            # Create multiple concurrent requests for the same cold config
            async def get_config():
                return await self.config_registry.get(self.mock_config_id)

            # Start multiple concurrent requests
            tasks = [get_config() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All requests should return the same config
            for result in results:
                assert result == self.mock_config

            # Verify entity client was called only once (not 5 times)
            self.mock_entities_client.get.assert_called_once()

            # Verify the config was cached
            cache_key = self.config_registry._generate_cache_key(config_ids=[self.mock_config_id])
            assert cache_key in self.config_registry._configs

    async def test_concurrent_multiple_config_requests(self):
        """Test that multiple concurrent requests for multiple configs result in correct entity client calls."""
        config_obj1 = GuardrailConfig(name="config-id-1", workspace="default", data=None)
        config_obj2 = GuardrailConfig(name="config-id-2", workspace="default", data=None)

        self.mock_entities_client.get = AsyncMock(side_effect=[config_obj1, config_obj2])

        # Create separate mock configs
        mock_config1 = MagicMock(spec=RailsConfig)
        mock_config1.colang_version = "1.0"
        mock_config2 = MagicMock(spec=RailsConfig)
        mock_config2.colang_version = "1.0"

        # Mock the combined config result
        combined_config = MagicMock(spec=RailsConfig)
        combined_config.colang_version = "1.0"
        mock_config1.__add__ = MagicMock(return_value=combined_config)

        # Mock the _make_rails_config method to return our mock configs
        with patch.object(self.config_registry, "_make_rails_config", side_effect=[mock_config1, mock_config2]):
            # Create multiple concurrent requests for the same multiple fully-qualified configs
            async def get_configs():
                return await self.config_registry.get(["default/config-id-1", "default/config-id-2"])

            # Start multiple concurrent requests
            tasks = [get_configs() for _ in range(3)]
            results = await asyncio.gather(*tasks)

            # All requests should return the same combined config
            for result in results:
                assert result == combined_config

            # Verify entity client was called only twice (not 6 times)
            assert self.mock_entities_client.get.call_count == 2

            # Verify the combined config was cached
            cache_key = self.config_registry._generate_cache_key(
                config_ids=["default/config-id-1", "default/config-id-2"]
            )
            assert cache_key in self.config_registry._configs

    async def test_loading_event_cleanup(self):
        """Test that loading events are properly cleaned up after config loading."""
        # Create actual config object from entity store
        config_obj = GuardrailConfig(name="test-config-id", workspace="default", data=None)
        self.mock_entities_client.get = AsyncMock(return_value=config_obj)

        # Mock the _make_rails_config method to return our mock config
        with patch.object(self.config_registry, "_make_rails_config", return_value=self.mock_config):
            # Make a request to trigger loading
            await self.config_registry.get(self.mock_config_id)

            # Config isn't in cache yet, so verify entity client was called
            assert self.mock_entities_client.get.call_count == 1

            # Check that loading event was cleaned up
            cache_key = self.config_registry._generate_cache_key(config_ids=[self.mock_config_id])
            assert cache_key not in self.config_registry._cache_loading_events


class TestConfigRegistryRefreshAll(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_entities_client = MagicMock()
        # Use a longer TTL (1 hour) to prevent entries from expiring during tests
        self.config_registry = ConfigRegistry(entities_client=self.mock_entities_client, ttl=3600)

    async def test_refresh_all_empty_cache(self):
        """Test _refresh_all handles empty cache gracefully."""
        await self.config_registry.refresh_all()

        # Should not make any entity client calls
        assert not self.mock_entities_client.list.called

    async def test_refresh_all_populated_cache(self):
        """Test _refresh_all makes correct calls to get_by_name for configs, given a populated cache."""
        # Add cache entries
        entry_1 = ConfigCacheEntry(key="config-1", config=MagicMock(spec=RailsConfig), config_ids=["default/config-1"])
        entry_2 = ConfigCacheEntry(
            key="config-1/config-2",
            config=MagicMock(spec=RailsConfig),
            config_ids=["default/config-1", "default/config-2"],
        )
        self.config_registry._configs["config-1"] = entry_1
        self.config_registry._configs["config-1/config-2"] = entry_2

        # Mock entity client response
        db_config_1 = MagicMock(spec=GuardrailConfig)
        db_config_1.workspace = "default"
        db_config_1.name = "config-1"
        db_config_1.updated_at = datetime.now(timezone.utc)

        db_config_2 = MagicMock(spec=GuardrailConfig)
        db_config_2.workspace = "default"
        db_config_2.name = "config-2"
        db_config_2.updated_at = datetime.now(timezone.utc)

        # Mock get_by_name to return configs
        self.mock_entities_client.get = AsyncMock(side_effect=[db_config_1, db_config_2])

        await self.config_registry.refresh_all()

        # Verify that get_by_name was called twice (once for each unique config)
        assert self.mock_entities_client.get.call_count == 2

    async def test_refresh_all_ttl_eviction(self):
        """Test that unused entries are removed after disuse_ttl."""
        # Add a cache entry with old last_used time
        old_entry = ConfigCacheEntry(
            key="old-key", config=MagicMock(spec=RailsConfig), config_ids=["default/old-config"]
        )
        old_entry.created_at = datetime.now(timezone.utc) - timedelta(seconds=3700)  # Older than TTL

        self.config_registry._configs["old-key"] = old_entry

        # Mock entity client to raise exception (simulating config doesn't exist)
        from nhx.common.entities.client import EntityNotFoundError

        self.mock_entities_client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))

        await self.config_registry.refresh_all()

        # Old entry should be removed
        assert "old-key" not in self.config_registry._configs

    async def test_refresh_all_missing_config_handling(self):
        """Test that entries are removed when configs no longer exist in entity store."""
        # Add a cache entry
        entry = ConfigCacheEntry(
            key="test-key", config=MagicMock(spec=RailsConfig), config_ids=["default/missing-config"]
        )
        self.config_registry._configs["test-key"] = entry

        # Mock entity client to raise exception (config doesn't exist)
        from nhx.common.entities.client import EntityNotFoundError

        self.mock_entities_client.get = AsyncMock(side_effect=EntityNotFoundError("Not found"))

        await self.config_registry.refresh_all()

        # Entry should be removed due to missing config
        assert "test-key" not in self.config_registry._configs

    async def test_refresh_all_change_detection(self):
        """Test that configs are rebuilt when entity store updated_at > cache added_at."""
        # Add a cache entry with old added_at time
        old_config = MagicMock(spec=RailsConfig)
        old_created_at = datetime.now(timezone.utc) - timedelta(seconds=120)
        entry = ConfigCacheEntry(key="test-key", config=old_config, config_ids=["default/test-config"])
        entry.created_at = old_created_at

        self.config_registry._configs["test-key"] = entry

        # Create actual entity store config with newer updated_at
        db_config = GuardrailConfig(
            name="test-config",
            workspace="default",
            data=None,
        )
        db_config._updated_at = datetime.now(timezone.utc)

        self.mock_entities_client.get = AsyncMock(return_value=db_config)

        new_config = MagicMock(spec=RailsConfig)
        with patch.object(self.config_registry, "_build_combined_config", return_value=new_config) as mock_build:
            await self.config_registry.refresh_all()
            # Verify that _build_combined_config was called
            mock_build.assert_called_once()

        # Updated entries don't rebuild their key, so we can use the test key to verify the rebuild occurred
        assert "test-key" in self.config_registry._configs
        # The config should be the new one returned by _build_combined_config
        assert self.config_registry._configs["test-key"].config == new_config
        # The created_at should be newer than the old one
        assert self.config_registry._configs["test-key"].created_at > old_created_at

    async def test_refresh_all_unchanged_config_preservation(self):
        """Test that stable configs remain cached."""
        # Add a cache entry with recent added_at time
        original_config = MagicMock(spec=RailsConfig)
        entry = ConfigCacheEntry(key="test-key", config=original_config, config_ids=["default/test-config"])
        entry.created_at = datetime.now(timezone.utc)  # Recent

        self.config_registry._configs["test-key"] = entry

        # Create actual entity store config with older updated_at
        db_config = GuardrailConfig(
            name="test-config",
            workspace="default",
            data=None,
            updated_at=datetime.now(timezone.utc) - timedelta(seconds=60),
        )

        # Mock get_by_name to return the config
        self.mock_entities_client.get = AsyncMock(return_value=db_config)

        await self.config_registry.refresh_all()

        # Verify that configs are preserved (not rebuilt when unchanged)
        assert "test-key" in self.config_registry._configs
        assert self.config_registry._configs["test-key"].config == original_config

    async def test_refresh_all_rebuild_failure_handling(self):
        """Test that failed rebuilds mark entries for deletion."""
        # Add a cache entry with old created_at
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        entry = ConfigCacheEntry(key="test-key", config=MagicMock(spec=RailsConfig), config_ids=["default/test-config"])
        entry.created_at = old_time

        self.config_registry._configs["test-key"] = entry

        # Create actual entity store config with newer updated_at (definitely newer than cache)
        db_config = GuardrailConfig(
            name="test-config",
            workspace="default",
            data=None,
        )
        db_config._updated_at = datetime.now(timezone.utc)

        self.mock_entities_client.get = AsyncMock(return_value=db_config)

        # Mock _build_combined_config to fail
        with patch.object(
            self.config_registry, "_build_combined_config", side_effect=Exception("Build failed")
        ) as mock_build:
            await self.config_registry.refresh_all()
            # Verify that _build_combined_config was called and failed
            mock_build.assert_called_once()

        # Entry should be removed due to build failure
        assert "test-key" not in self.config_registry._configs


class TestConfigCacheEntry(unittest.TestCase):
    def test_is_older_than(self):
        """Test ConfigCacheEntry.is_older_than method."""
        # Create entry with old timestamp
        old_entry = ConfigCacheEntry(
            key="test",
            config=MagicMock(spec=RailsConfig),
            config_ids=["test"],
            created_at=datetime.now(timezone.utc) - timedelta(seconds=100),
        )

        # Test that it's older than 50 seconds
        assert old_entry.is_older_than(timedelta(seconds=50))

        # Test that it's not older than 200 seconds
        assert not old_entry.is_older_than(timedelta(seconds=200))
