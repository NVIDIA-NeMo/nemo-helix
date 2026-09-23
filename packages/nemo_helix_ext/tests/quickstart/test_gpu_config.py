# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quickstart GPU config (CUDA_VISIBLE_DEVICES filtering and parsing)."""

import logging

import pytest
from nemo_helix_ext.quickstart.gpu_config import (
    apply_cuda_visible_devices_filter,
    format_gpu_ids_for_storage,
    parse_cuda_visible_devices_integers,
)


class TestParseCudaVisibleDevicesIntegers:
    """Tests for parse_cuda_visible_devices_integers (integer-only parsing)."""

    def test_parses_comma_separated_integers(self):
        """Valid comma-separated integers return list in order."""
        assert parse_cuda_visible_devices_integers("0,1,2") == [0, 1, 2]
        assert parse_cuda_visible_devices_integers("2,1,0") == [2, 1, 0]

    def test_parses_with_spaces(self):
        """Spaces around commas are allowed."""
        assert parse_cuda_visible_devices_integers("0, 1, 2") == [0, 1, 2]

    @pytest.mark.parametrize(
        "value",
        ["", "   ", "GPU-abc-123", "0,GPU-xyz,1", "-1", "0,-1,1", "0,foo,2", "all"],
        ids=["empty", "whitespace", "uuid_like", "mixed_uuid", "negative", "mixed_negative", "non_numeric", "all"],
    )
    def test_returns_none_for_invalid_or_unsupported(self, value: str):
        """Invalid or unsupported values (empty, UUID-like, negative, non-numeric) return None."""
        assert parse_cuda_visible_devices_integers(value) is None


class TestApplyCudaVisibleDevicesFilter:
    """Tests for apply_cuda_visible_devices_filter (intersection, order, exclusions)."""

    @pytest.mark.parametrize(
        "env_value",
        [None, ""],
        ids=["unset", "empty"],
    )
    def test_missing_or_empty_env_returns_full_detected_list(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str | None
    ):
        """When CUDA_VISIBLE_DEVICES is unset or empty, return full detected list."""
        if env_value is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env_value)
        detected = [0, 1, 2]
        assert apply_cuda_visible_devices_filter(detected) == [0, 1, 2]

    def test_filter_subset_preserves_order_of_env(self, monkeypatch: pytest.MonkeyPatch):
        """Order of result follows CUDA_VISIBLE_DEVICES, not detected order."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0,1")
        detected = [0, 1, 2, 3]
        assert apply_cuda_visible_devices_filter(detected) == [2, 0, 1]

    def test_intersection_excludes_ids_not_in_detected(self, monkeypatch: pytest.MonkeyPatch, caplog):
        """IDs in CUDA_VISIBLE_DEVICES that are not in detected set are excluded and logged."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,5,1")
        detected = [0, 1, 2, 3]
        # Capture from module logger so logs are visible under xdist (worker root logger may differ).
        with caplog.at_level(logging.WARNING, logger="nemo_helix_ext.quickstart.gpu_config"):
            result = apply_cuda_visible_devices_filter(detected, log_exclusions=True)
        assert result == [0, 1]
        assert "5" in caplog.text and "exclud" in caplog.text.lower()

    def test_non_integer_env_returns_full_list(self, monkeypatch: pytest.MonkeyPatch):
        """When CUDA_VISIBLE_DEVICES contains non-integers, return full detected list (no filter)."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,0,1")
        detected = [0, 1, 2]
        assert apply_cuda_visible_devices_filter(detected) == [0, 1, 2]

    def test_log_exclusions_false_does_not_log(self, monkeypatch: pytest.MonkeyPatch, caplog):
        """When log_exclusions=False, exclusions are not logged."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,99,1")
        detected = [0, 1, 2]
        with caplog.at_level(logging.WARNING, logger="nemo_helix_ext.quickstart.gpu_config"):
            result = apply_cuda_visible_devices_filter(detected, log_exclusions=False)
        assert result == [0, 1]
        assert "99" not in caplog.text
        assert "exclud" not in caplog.text.lower()


class TestFormatGpuIdsForStorage:
    """Tests for format_gpu_ids_for_storage."""

    def test_formats_as_comma_separated_string(self):
        """List of ints is formatted as comma-separated string."""
        assert format_gpu_ids_for_storage([0, 1, 2]) == "0,1,2"

    def test_empty_list(self):
        """Empty list produces empty string."""
        assert format_gpu_ids_for_storage([]) == ""

    def test_single_id(self):
        """Single ID produces single token."""
        assert format_gpu_ids_for_storage([3]) == "3"
