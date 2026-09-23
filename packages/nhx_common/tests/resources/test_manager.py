# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SharedResourceManager."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from nhx.common.resources.manager import SharedResourceManager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the SharedResourceManager singleton before and after each test."""
    SharedResourceManager.reset_instance()
    yield
    SharedResourceManager.reset_instance()


def create_mock_config(reserved_gpu_ids: str = "all") -> MagicMock:
    """Create a mock platform config with the given reserved_gpu_device_ids value."""
    mock_config = MagicMock()
    mock_config.docker.reserved_gpu_device_ids = reserved_gpu_ids

    # Implement get_reserved_gpu_ids() behavior matching DockerConfig
    def get_reserved_gpu_ids():
        if reserved_gpu_ids.lower() == "all":
            return None
        if reserved_gpu_ids.lower() == "none" or not reserved_gpu_ids:
            return []
        return [int(p.strip()) for p in reserved_gpu_ids.split(",") if p.strip()]

    mock_config.docker.get_reserved_gpu_ids = get_reserved_gpu_ids
    return mock_config


class TestSharedResourceManagerSingleton:
    """Tests for singleton behavior."""

    def test_get_instance_returns_same_instance(self):
        """Test that get_instance always returns the same instance."""
        instance1 = SharedResourceManager.get_instance()
        instance2 = SharedResourceManager.get_instance()
        assert instance1 is instance2

    def test_reset_instance_allows_new_instance(self):
        """Test that reset_instance allows creating a new instance."""
        instance1 = SharedResourceManager.get_instance()
        SharedResourceManager.reset_instance()
        instance2 = SharedResourceManager.get_instance()
        assert instance1 is not instance2

    def test_concurrent_get_instance_returns_same_instance(self):
        """Test that concurrent calls to get_instance return the same instance."""
        instances = []
        lock = threading.Lock()

        def get_and_store():
            instance = SharedResourceManager.get_instance()
            with lock:
                instances.append(instance)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_and_store) for _ in range(100)]
            for future in futures:
                future.result()

        # All instances should be the same
        assert len(instances) == 100
        assert all(instance is instances[0] for instance in instances)


class TestSharedResourceManagerGPUPool:
    """Tests for GPU pool management with config-based initialization."""

    def test_get_gpu_pool_with_all_auto_detects_gpus(self):
        """Test that 'all' config value triggers auto-detection."""
        with (
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
        ):
            mock_detect.return_value = [0, 1, 2, 3]
            mock_get_config.return_value = create_mock_config("all")

            manager = SharedResourceManager.get_instance()
            pool = manager.get_gpu_pool()

            assert pool is not None
            assert pool.num_reserved_gpus == 4
            assert set(pool.gpu_to_workload_id.keys()) == {0, 1, 2, 3}
            mock_detect.assert_called_once()

    def test_get_gpu_pool_returns_none_when_no_gpus_detected(self):
        """Test that get_gpu_pool returns None when GPU detection finds no GPUs."""
        with (
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
        ):
            mock_detect.return_value = None
            mock_get_config.return_value = create_mock_config("all")

            manager = SharedResourceManager.get_instance()
            pool = manager.get_gpu_pool()
            assert pool is None

    @pytest.mark.parametrize(
        "reserved_gpu_ids,detected_gpus,expected_pool_size,expected_gpu_ids",
        [
            pytest.param("0", [0, 1, 2, 3], 1, {0}, id="single_gpu"),
            pytest.param("0,1,2,3", [0, 1, 2, 3], 4, {0, 1, 2, 3}, id="four_gpus"),
            pytest.param("0, 1", [0, 1, 2, 3], 2, {0, 1}, id="two_gpus_with_spaces"),
            pytest.param("1,3", [0, 1, 2, 3], 2, {1, 3}, id="non_contiguous_gpus"),
        ],
    )
    def test_get_gpu_pool_with_explicit_ids(
        self, reserved_gpu_ids, detected_gpus, expected_pool_size, expected_gpu_ids
    ):
        """Test that explicit GPU IDs create the correct pool when all IDs are valid."""
        with (
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
        ):
            mock_detect.return_value = detected_gpus
            mock_get_config.return_value = create_mock_config(reserved_gpu_ids)

            manager = SharedResourceManager.get_instance()
            pool = manager.get_gpu_pool()

            assert pool is not None
            assert pool.num_reserved_gpus == expected_pool_size
            assert set(pool.gpu_to_workload_id.keys()) == expected_gpu_ids

    def test_get_gpu_pool_filters_invalid_ids_and_logs_warning(self, caplog):
        """Test that invalid GPU IDs are filtered out with a warning."""
        import logging

        with (
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
        ):
            # User configured GPUs 0,1,2,3 but only 0,1 exist
            mock_detect.return_value = [0, 1]
            mock_get_config.return_value = create_mock_config("0,1,2,3")

            with caplog.at_level(logging.WARNING):
                manager = SharedResourceManager.get_instance()
                pool = manager.get_gpu_pool()

            # Pool should only contain valid GPUs
            assert pool is not None
            assert pool.num_reserved_gpus == 2
            assert set(pool.gpu_to_workload_id.keys()) == {0, 1}

            # Warning should mention the invalid IDs
            assert "not found on system" in caplog.text
            assert "[2, 3]" in caplog.text

    def test_get_gpu_pool_returns_none_when_all_ids_invalid(self, caplog):
        """Test that get_gpu_pool returns None when all configured IDs are invalid."""
        import logging

        with (
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
        ):
            # User configured GPUs 4,5,6 but only 0,1 exist
            mock_detect.return_value = [0, 1]
            mock_get_config.return_value = create_mock_config("4,5,6")

            with caplog.at_level(logging.WARNING):
                manager = SharedResourceManager.get_instance()
                pool = manager.get_gpu_pool()

            assert pool is None
            assert "None of the configured GPU IDs exist" in caplog.text

    def test_get_gpu_pool_trusts_config_when_detection_unavailable(self, caplog):
        """Test that user config is trusted when GPU detection fails."""
        import logging

        with (
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
        ):
            # GPU detection returns None (not available or failed)
            mock_detect.return_value = None
            mock_get_config.return_value = create_mock_config("0,1,2,3")

            with caplog.at_level(logging.WARNING):
                manager = SharedResourceManager.get_instance()
                pool = manager.get_gpu_pool()

            # Pool should be created with user-specified IDs
            assert pool is not None
            assert pool.num_reserved_gpus == 4
            assert set(pool.gpu_to_workload_id.keys()) == {0, 1, 2, 3}

            # Warning should mention we're trusting the config
            assert "Could not detect GPUs to validate" in caplog.text
            assert "Trusting configured GPU IDs" in caplog.text

    @pytest.mark.parametrize(
        "config_value",
        [
            pytest.param("", id="empty_string"),
            pytest.param("none", id="none_string"),
        ],
    )
    def test_get_gpu_pool_returns_none_and_logs_warning_when_gpus_disabled(self, config_value, caplog):
        """Test that get_gpu_pool returns None and logs warning when GPUs are explicitly disabled."""
        import logging

        with patch("nhx.common.resources.manager.get_platform_config") as mock_get_config:
            mock_get_config.return_value = create_mock_config(config_value)

            with caplog.at_level(logging.WARNING):
                manager = SharedResourceManager.get_instance()
                pool = manager.get_gpu_pool()

            assert pool is None
            assert "GPU support explicitly disabled" in caplog.text

    def test_get_gpu_pool_returns_same_pool_on_subsequent_calls(self):
        """Test that get_gpu_pool returns the same pool on subsequent calls."""
        with patch("nhx.common.resources.manager.get_platform_config") as mock_get_config:
            mock_get_config.return_value = create_mock_config("0,1")

            manager = SharedResourceManager.get_instance()
            pool1 = manager.get_gpu_pool()
            pool2 = manager.get_gpu_pool()

            assert pool1 is pool2

    def test_get_gpu_pool_with_all_only_detects_once(self):
        """Test that get_gpu_pool only runs detection on first call when using 'all'."""
        with (
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
        ):
            mock_detect.return_value = [0]
            mock_get_config.return_value = create_mock_config("all")

            manager = SharedResourceManager.get_instance()

            # First call runs detection
            manager.get_gpu_pool()
            assert mock_detect.call_count == 1

            # Subsequent calls don't re-detect
            manager.get_gpu_pool()
            manager.get_gpu_pool()
            assert mock_detect.call_count == 1

    def test_concurrent_get_gpu_pool_returns_same_pool(self):
        """Test that concurrent calls to get_gpu_pool return the same pool."""
        with patch("nhx.common.resources.manager.get_platform_config") as mock_get_config:
            mock_get_config.return_value = create_mock_config("0,1,2,3")

            manager = SharedResourceManager.get_instance()

            pools = []
            lock = threading.Lock()

            def get_and_store():
                pool = manager.get_gpu_pool()
                with lock:
                    pools.append(pool)

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(get_and_store) for _ in range(100)]
                for future in futures:
                    future.result()

            # All pools should be the same
            assert len(pools) == 100
            assert all(pool is pools[0] for pool in pools)


class TestSharedResourceManagerClear:
    """Tests for clear functionality."""

    def test_clear_removes_gpu_pool(self):
        """Test that clear removes the GPU pool."""
        with patch("nhx.common.resources.manager.get_platform_config") as mock_get_config:
            mock_get_config.return_value = create_mock_config("0")

            manager = SharedResourceManager.get_instance()

            pool1 = manager.get_gpu_pool()
            assert pool1 is not None

            manager.clear()

            # After clear, get_gpu_pool should create a new pool
            pool2 = manager.get_gpu_pool()
            assert pool2 is not None
            assert pool1 is not pool2

    def test_clear_is_safe_when_no_pool(self):
        """Test that clear is safe to call when no pool exists."""
        manager = SharedResourceManager.get_instance()
        # Should not raise
        manager.clear()


class TestSharedResourceManagerIntegration:
    """Integration tests simulating jobs and models services sharing the pool."""

    def test_two_services_share_same_pool(self):
        """Test that two 'services' get the same GPU pool."""
        with patch("nhx.common.resources.manager.get_platform_config") as mock_get_config:
            mock_get_config.return_value = create_mock_config("0,1,2,3")

            # Simulate jobs service getting the pool
            jobs_manager = SharedResourceManager.get_instance()
            jobs_pool = jobs_manager.get_gpu_pool()

            # Simulate models service getting the pool
            models_manager = SharedResourceManager.get_instance()
            models_pool = models_manager.get_gpu_pool()

            # Should be the same pool
            assert jobs_pool is models_pool

    def test_gpu_allocations_are_coordinated(self):
        """Test that GPU allocations from different 'services' don't overlap."""
        with (
            patch("nhx.common.resources.manager.get_platform_config") as mock_get_config,
            patch("nhx.common.resources.manager.detect_gpu_device_ids") as mock_detect,
        ):
            mock_detect.return_value = [0, 1, 2, 3]
            mock_get_config.return_value = create_mock_config("0,1,2,3")

            manager = SharedResourceManager.get_instance()
            pool = manager.get_gpu_pool()
            status = pool.get_pool_status()
            assert status.available == 4

            # Jobs service allocates 2 GPUs
            jobs_gpus = pool.allocate_gpu("job-123", num_requested=2)

            # Models service allocates 2 GPUs
            models_gpus = pool.allocate_gpu("deployment-456", num_requested=2)

            # Allocations should not overlap
            assert len(jobs_gpus) == 2
            assert len(models_gpus) == 2
            assert set(jobs_gpus).intersection(set(models_gpus)) == set()

            # All 4 GPUs should now be allocated
            assert pool.get_available_count() == 0
