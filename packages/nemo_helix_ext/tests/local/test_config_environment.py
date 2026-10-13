# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for configuration and environment resolution.

These tests exercise the real ``apply_run_environment`` code path with
actual YAML config files, verifying that environment variables are set
correctly for different host, port, and base_url scenarios.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nhx.platform_runner.config import (
    ResolvedRunConfiguration,
    apply_run_environment,
    default_config_path,
)


def _resolved(
    *,
    services: set[str] | None = None,
    controllers: set[str] | None = None,
    sidecars: set[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    config_path: str | None = None,
    socket_path: str | None = None,
) -> ResolvedRunConfiguration:
    return ResolvedRunConfiguration(
        services=services or set(),
        controllers=controllers or set(),
        sidecars=sidecars or set(),
        host=host,
        port=port,
        config_path=config_path or default_config_path(),
        socket_path=socket_path,
        available_services={},
        available_controllers={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_apply_run_environment_preserves_external_base_url() -> None:
    """Pre-set NHX_BASE_URL (e.g. from k8s/Helm) must not be overwritten."""
    env: dict[str, str] = {"NHX_BASE_URL": "https://platform.k8s.internal:443"}
    config = _resolved(host="0.0.0.0", port=9090)

    apply_run_environment(config, env=env)

    assert env["NHX_BASE_URL"] == "https://platform.k8s.internal:443"


@pytest.mark.integration
def test_apply_run_environment_wildcard_host_becomes_loopback(tmp_path: Path) -> None:
    """A wildcard bind host (0.0.0.0) in the config file should resolve to
    127.0.0.1 for the base URL, using the actual bind port."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("platform:\n  base_url: http://0.0.0.0:8080\n")

    env: dict[str, str] = {}
    config = _resolved(host="0.0.0.0", port=9090, config_path=str(config_file))

    apply_run_environment(config, env=env)

    assert env["NHX_BASE_URL"] == "http://127.0.0.1:9090"
    assert env["NHX_SERVICE_HOST"] == "127.0.0.1"
    assert env["NHX_SERVICE_PORT"] == "9090"


@pytest.mark.integration
def test_apply_run_environment_ipv6_literal_bracketed(tmp_path: Path) -> None:
    """An IPv6 config base_url should produce a bracketed host in the resolved URL."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("platform:\n  base_url: http://[::1]:8080\n")

    env: dict[str, str] = {}
    config = _resolved(host="::1", port=9090, config_path=str(config_file))

    apply_run_environment(config, env=env)

    assert env["NHX_BASE_URL"] == "http://[::1]:9090"


@pytest.mark.integration
def test_config_file_base_url_malformed_yaml_falls_back(tmp_path: Path) -> None:
    """A corrupt config file should fall back to the bind-derived URL."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{{{{not valid yaml at all")

    env: dict[str, str] = {}
    config = _resolved(host="127.0.0.1", port=7777, config_path=str(config_file))

    apply_run_environment(config, env=env)

    # Falls back to bind-derived: http://<host>:<port>
    assert env["NHX_BASE_URL"] == "http://127.0.0.1:7777"


@pytest.mark.integration
def test_config_file_missing_falls_back(tmp_path: Path) -> None:
    """A missing config file should fall back to the bind-derived URL."""
    env: dict[str, str] = {}
    config = _resolved(host="127.0.0.1", port=5555, config_path=str(tmp_path / "nonexistent.yaml"))

    apply_run_environment(config, env=env)

    assert env["NHX_BASE_URL"] == "http://127.0.0.1:5555"


@pytest.mark.integration
def test_apply_run_environment_clears_empty_service_lists() -> None:
    """When services/controllers/sidecars are empty sets, their env vars
    should be removed (popped) rather than set to empty strings."""
    env: dict[str, str] = {
        "NHX_SERVICES": "old-service",
        "NHX_CONTROLLERS": "old-controller",
        "NHX_SIDECARS": "old-sidecar",
    }
    config = _resolved(services=set(), controllers=set(), sidecars=set())

    apply_run_environment(config, env=env)

    assert "NHX_SERVICES" not in env
    assert "NHX_CONTROLLERS" not in env
    assert "NHX_SIDECARS" not in env


@pytest.mark.integration
def test_apply_run_environment_sets_service_lists() -> None:
    """Non-empty service/controller/sidecar sets should be written as
    comma-separated, sorted env var values."""
    env: dict[str, str] = {}
    config = _resolved(
        services={"models", "auth", "secrets"},
        controllers={"beta-controller"},
        sidecars={"adapters"},
    )

    apply_run_environment(config, env=env)

    assert env["NHX_SERVICES"] == "auth,models,secrets"
    assert env["NHX_CONTROLLERS"] == "beta-controller"
    assert env["NHX_SIDECARS"] == "adapters"


@pytest.mark.integration
def test_apply_run_environment_uds_transport_uses_unix_base_url() -> None:
    """When a socket_path is set (UDS transport), the base URL should use
    the ``unix://`` scheme."""
    env: dict[str, str] = {}
    config = _resolved(socket_path="/tmp/nemo.sock")

    apply_run_environment(config, env=env)

    assert env["NHX_BASE_URL"] == "unix:///tmp/nemo.sock"


@pytest.mark.integration
def test_apply_run_environment_preserves_external_host_and_port() -> None:
    """Pre-set NHX_SERVICE_HOST and NHX_SERVICE_PORT should not be overwritten."""
    env: dict[str, str] = {
        "NHX_SERVICE_HOST": "10.0.0.1",
        "NHX_SERVICE_PORT": "443",
    }
    config = _resolved(host="0.0.0.0", port=9090)

    apply_run_environment(config, env=env)

    assert env["NHX_SERVICE_HOST"] == "10.0.0.1"
    assert env["NHX_SERVICE_PORT"] == "443"


@pytest.mark.integration
def test_apply_run_environment_ipv6_wildcard_becomes_loopback(tmp_path: Path) -> None:
    """The IPv6 wildcard ``::`` should resolve to ``::1`` for internal clients."""
    # Use a config file without platform.base_url so the bind host drives the URL.
    config_file = tmp_path / "config.yaml"
    config_file.write_text("platform:\n  seed_on_startup: false\n")

    env: dict[str, str] = {}
    config = _resolved(host="::", port=8080, config_path=str(config_file))

    apply_run_environment(config, env=env)

    assert env["NHX_SERVICE_HOST"] == "::1"
    # Base URL should have bracketed IPv6.
    assert env["NHX_BASE_URL"] == "http://[::1]:8080"
