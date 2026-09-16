# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for automatic loopback address detection."""

from unittest.mock import mock_open, patch

from nemo_helix_plugin.config import (
    PlatformConfig,
    _is_running_in_docker,
    determine_loopback_override,
)


class TestIsRunningInDocker:
    """Tests for _is_running_in_docker detection."""

    def test_detects_dockerenv_file(self):
        """Test detection via /.dockerenv file."""
        with patch("nemo_helix_plugin.config.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            assert _is_running_in_docker() is True

    def test_detects_docker_in_cgroup(self):
        """Test detection via /proc/self/cgroup containing 'docker'."""
        with (
            patch("nemo_helix_plugin.config.Path") as mock_path,
            patch("builtins.open", mock_open(read_data="1:name=systemd:/docker/abc123\n")),
        ):
            mock_path.return_value.exists.return_value = False
            assert _is_running_in_docker() is True

    def test_detects_containerd_in_cgroup(self):
        """Test detection via /proc/self/cgroup containing 'containerd'."""
        with (
            patch("nemo_helix_plugin.config.Path") as mock_path,
            patch("builtins.open", mock_open(read_data="1:name=systemd:/containerd/abc123\n")),
        ):
            mock_path.return_value.exists.return_value = False
            assert _is_running_in_docker() is True

    def test_returns_false_when_not_in_docker(self):
        """Test returns False when not running in Docker."""
        with (
            patch("nemo_helix_plugin.config.Path") as mock_path,
            patch("builtins.open", mock_open(read_data="1:name=systemd:/user.slice\n")),
        ):
            mock_path.return_value.exists.return_value = False
            assert _is_running_in_docker() is False

    def test_handles_missing_cgroup_file(self):
        """Test handles missing /proc/self/cgroup file gracefully."""
        with (
            patch("nemo_helix_plugin.config.Path") as mock_path,
            patch("builtins.open", side_effect=FileNotFoundError),
        ):
            mock_path.return_value.exists.return_value = False
            assert _is_running_in_docker() is False


class TestDetermineLoopbackOverride:
    """Tests for determine_loopback_override function."""

    def test_macos_returns_host_docker_internal(self):
        """Test macOS returns 'host.docker.internal'."""
        with (
            patch("nemo_helix_plugin.config.platform.system", return_value="Darwin"),
            patch("nemo_helix_plugin.config._is_running_in_docker", return_value=False),
        ):
            result = determine_loopback_override()
            assert result == "host.docker.internal"

    def test_docker_container_returns_hostname(self, monkeypatch):
        """Test running in Docker container returns container hostname."""
        # Mock socket.gethostname using monkeypatch
        import socket as socket_module

        monkeypatch.setattr(socket_module, "gethostname", lambda: "container-abc123")

        with (
            patch("nemo_helix_plugin.config.platform.system", return_value="Linux"),
            patch("nemo_helix_plugin.config._is_running_in_docker", return_value=True),
        ):
            result = determine_loopback_override()
            assert result == "container-abc123"

    def test_linux_host_network_returns_none(self):
        """Test Linux host network returns None (no override needed)."""
        with (
            patch("nemo_helix_plugin.config.platform.system", return_value="Linux"),
            patch("nemo_helix_plugin.config._is_running_in_docker", return_value=False),
        ):
            result = determine_loopback_override()
            assert result is None


class TestPlatformConfigSharedEnvvars:
    """Tests for PlatformConfig.to_shared_envvars with automatic loopback detection."""

    def test_uses_automatic_detection_when_no_override(self):
        """Test that automatic detection is used when loopback_address is not configured."""
        config = PlatformConfig(base_url="http://localhost:8000")

        with (
            patch("nemo_helix_plugin.config.platform.system", return_value="Darwin"),
            patch("nemo_helix_plugin.config._is_running_in_docker", return_value=False),
        ):
            envvars = config.to_shared_envvars()

            # On macOS, localhost should be replaced with host.docker.internal
            assert "host.docker.internal" in envvars["NHX_BASE_URL"]
            assert "host.docker.internal" in envvars["NHX_JOBS_URL"]

    def test_configured_loopback_address_takes_precedence(self):
        """Test that configured loopback_address field takes precedence over automatic detection."""
        config = PlatformConfig(
            base_url="http://localhost:8000",
            loopback_address="custom.configured",
        )

        # Even on macOS, the configured value should be used
        with (
            patch("nemo_helix_plugin.config.platform.system", return_value="Darwin"),
            patch("nemo_helix_plugin.config._is_running_in_docker", return_value=False),
        ):
            envvars = config.to_shared_envvars()

            assert "custom.configured" in envvars["NHX_BASE_URL"]
            assert "custom.configured" in envvars["NHX_JOBS_URL"]

    def test_callsite_loopback_address_takes_precedence(self):
        """Test that callers can request URLs for a specific network context."""
        config = PlatformConfig(
            base_url="http://localhost:8000",
            loopback_address="custom.configured",
        )

        envvars = config.to_shared_envvars(loopback_address="subprocess.local")

        assert "subprocess.local" in envvars["NHX_BASE_URL"]
        assert "subprocess.local" in envvars["NHX_JOBS_URL"]

    def test_loopback_address_via_env_var(self, monkeypatch):
        """Test that loopback_address can be set via environment variable."""
        monkeypatch.setenv("NHX_LOOPBACK_ADDRESS", "env.override")

        # Need to create config after setting env var for pydantic to pick it up
        config = PlatformConfig(base_url="http://localhost:8000")

        # Environment variable should set the field
        assert config.loopback_address == "env.override"

        envvars = config.to_shared_envvars()

        # The env var value should be used
        assert "env.override" in envvars["NHX_BASE_URL"]
        assert "env.override" in envvars["NHX_JOBS_URL"]

    def test_no_replacement_when_no_loopback_addresses(self):
        """Test that URLs without loopback addresses are not modified."""
        config = PlatformConfig(base_url="http://my-service:8000")

        with (
            patch("nemo_helix_plugin.config.platform.system", return_value="Darwin"),
            patch("nemo_helix_plugin.config._is_running_in_docker", return_value=False),
        ):
            envvars = config.to_shared_envvars()

            # No loopback addresses, so URLs should remain unchanged (all use base_url)
            assert envvars["NHX_BASE_URL"] == "http://my-service:8000"
            assert envvars["NHX_JOBS_URL"] == "http://my-service:8000"

    def test_linux_host_network_no_replacement(self):
        """Test that on Linux host network, no replacement happens."""
        config = PlatformConfig(base_url="http://localhost:8000")

        with (
            patch("nemo_helix_plugin.config.platform.system", return_value="Linux"),
            patch("nemo_helix_plugin.config._is_running_in_docker", return_value=False),
        ):
            envvars = config.to_shared_envvars()

            # On Linux host network, no override is needed; URLs unchanged
            assert envvars["NHX_BASE_URL"] == "http://localhost:8000"
            assert envvars["NHX_JOBS_URL"] == "http://localhost:8000"
