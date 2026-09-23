# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU detection utilities."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from nhx.common.docker.gpu_detection import get_gpu_count


@pytest.fixture
def mock_pynvml():
    """Create a mock GPU detection (NVML) module for testing."""
    mock_module = MagicMock()
    mock_module.NVMLError = Exception  # Use Exception as the error type for mocking
    return mock_module


class TestDetectGPUDeviceIds:
    """Tests for detect_gpu_device_ids function."""

    @pytest.mark.parametrize(
        "gpu_count,expected_ids",
        [
            pytest.param(1, [0], id="single_gpu"),
            pytest.param(2, [0, 1], id="two_gpus"),
            pytest.param(4, [0, 1, 2, 3], id="four_gpus"),
            pytest.param(8, [0, 1, 2, 3, 4, 5, 6, 7], id="eight_gpus"),
        ],
    )
    def test_detect_gpus_success(self, mock_pynvml, gpu_count, expected_ids):
        """Test successful GPU detection with various GPU counts."""
        mock_pynvml.nvmlDeviceGetCount.return_value = gpu_count

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            # Need to reimport to pick up the mocked module
            from nhx.common.docker import gpu_detection

            result = gpu_detection.detect_gpu_device_ids()

            assert result == expected_ids
            mock_pynvml.nvmlInit.assert_called_once()
            mock_pynvml.nvmlDeviceGetCount.assert_called_once()
            mock_pynvml.nvmlShutdown.assert_called_once()

    def test_detect_gpus_returns_none_when_no_gpus(self, mock_pynvml):
        """Test that detection returns None when NVML reports zero GPUs."""
        mock_pynvml.nvmlDeviceGetCount.return_value = 0

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            from nhx.common.docker import gpu_detection

            result = gpu_detection.detect_gpu_device_ids()

            assert result is None

    def test_detect_gpus_returns_none_when_gpu_detection_unavailable(self):
        """Test that detection returns None when the GPU detection library is not installed."""
        with patch.dict(sys.modules, {"pynvml": None}):
            from nhx.common.docker import gpu_detection

            result = gpu_detection.detect_gpu_device_ids()

            assert result is None

    def test_detect_gpus_returns_none_on_init_failure(self, mock_pynvml):
        """Test that detection returns None when NVML initialization fails."""
        mock_pynvml.nvmlInit.side_effect = mock_pynvml.NVMLError("Driver not loaded")

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            from nhx.common.docker import gpu_detection

            result = gpu_detection.detect_gpu_device_ids()

            assert result is None

    def test_detect_gpus_returns_none_on_count_failure(self, mock_pynvml):
        """Test that detection returns None when getting device count fails."""
        mock_pynvml.nvmlDeviceGetCount.side_effect = mock_pynvml.NVMLError("Query failed")

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            from nhx.common.docker import gpu_detection

            result = gpu_detection.detect_gpu_device_ids()

            assert result is None
            # Shutdown should still be called even on error
            mock_pynvml.nvmlShutdown.assert_called_once()

    def test_detect_gpus_handles_shutdown_error_gracefully(self, mock_pynvml):
        """Test that shutdown errors are ignored."""
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlShutdown.side_effect = mock_pynvml.NVMLError("Shutdown failed")

        with patch.dict(sys.modules, {"pynvml": mock_pynvml}):
            from nhx.common.docker import gpu_detection

            # Should still return results despite shutdown error
            result = gpu_detection.detect_gpu_device_ids()

            assert result == [0, 1]


class TestGetGPUCount:
    """Tests for get_gpu_count function."""

    @pytest.mark.parametrize(
        "detected_ids,expected_count",
        [
            pytest.param([0], 1, id="single_gpu"),
            pytest.param([0, 1, 2, 3], 4, id="four_gpus"),
            pytest.param(None, 0, id="no_gpus"),
        ],
    )
    def test_get_gpu_count(self, detected_ids, expected_count):
        """Test GPU count for various detection results."""
        with patch("nhx.common.docker.gpu_detection.detect_gpu_device_ids") as mock_detect:
            mock_detect.return_value = detected_ids

            result = get_gpu_count()

            assert result == expected_count
