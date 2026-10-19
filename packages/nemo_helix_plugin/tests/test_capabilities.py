# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_helix_plugin.capabilities — shared Docker capability probe."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import DockerException
from nemo_helix_plugin.capabilities import (
    CapabilityUnavailableError,
    probe_docker,
    require_docker,
    reset_capability_cache,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout


@pytest.fixture(autouse=True)
def _clear_capability_cache() -> Iterator[None]:
    reset_capability_cache()
    yield
    reset_capability_cache()


def test_probe_docker_available() -> None:
    client = MagicMock()
    client.ping.return_value = True
    with patch("docker.from_env", return_value=client) as from_env:
        result = probe_docker()
    assert result.available is True
    assert result.detail is None
    from_env.assert_called_once_with(timeout=5)
    client.ping.assert_called_once()
    client.close.assert_called_once()


@pytest.mark.parametrize(
    "exc",
    [
        DockerException("boom"),
        RequestsConnectionError("refused"),
        RequestsTimeout("timed out"),
        OSError("no such file"),
    ],
)
def test_probe_docker_unavailable_on_connection_failures(exc: Exception) -> None:
    with patch("docker.from_env", side_effect=exc):
        result = probe_docker()
    assert result.available is False
    assert result.detail is not None
    assert "unreachable" in result.detail.lower() or "Docker" in result.detail


def test_probe_docker_unavailable_on_ping_failure() -> None:
    client = MagicMock()
    client.ping.side_effect = DockerException("ping failed")
    with patch("docker.from_env", return_value=client):
        result = probe_docker()
    assert result.available is False
    client.close.assert_called_once()


def test_probe_docker_caches_result() -> None:
    client = MagicMock()
    with patch("docker.from_env", return_value=client) as from_env:
        first = probe_docker()
        second = probe_docker()
    assert first.available is True
    assert second.available is True
    assert from_env.call_count == 1


def test_reset_capability_cache_allows_reprobe() -> None:
    client = MagicMock()
    with patch("docker.from_env", return_value=client) as from_env:
        probe_docker()
        reset_capability_cache()
        probe_docker()
    assert from_env.call_count == 2


def test_probe_docker_use_cache_false_bypasses_memo() -> None:
    client = MagicMock()
    with patch("docker.from_env", return_value=client) as from_env:
        probe_docker()
        probe_docker(use_cache=False)
    assert from_env.call_count == 2


def test_probe_docker_host_keys_are_independent() -> None:
    default_client = MagicMock()
    remote_client = MagicMock()
    remote_client.ping.side_effect = DockerException("remote down")

    def _from_env(**kwargs):
        env = kwargs.get("environment") or {}
        if env.get("DOCKER_HOST") == "tcp://remote:2375":
            return remote_client
        return default_client

    with patch("docker.from_env", side_effect=_from_env) as from_env:
        default = probe_docker()
        remote = probe_docker(docker_host="tcp://remote:2375")

    assert default.available is True
    assert remote.available is False
    assert from_env.call_count == 2
    from_env.assert_any_call(timeout=5)
    remote_call = next(c for c in from_env.call_args_list if c.kwargs.get("environment"))
    assert remote_call.kwargs["timeout"] == 5
    assert remote_call.kwargs["environment"]["DOCKER_HOST"] == "tcp://remote:2375"


def test_probe_docker_explicit_host_returns_unavailable_without_typeerror() -> None:
    """Regression: docker-py 7.x rejects base_url= on from_env; must not raise TypeError."""
    result = probe_docker(docker_host="tcp://127.0.0.1:1", use_cache=False)
    assert result.available is False
    assert result.detail is not None


def test_require_docker_raises_when_unavailable() -> None:
    with patch("docker.from_env", side_effect=DockerException("down")):
        with pytest.raises(CapabilityUnavailableError, match="unreachable"):
            require_docker()


def test_validate_docker_available_delegates_to_probe() -> None:
    from nemo_helix_plugin.config import validate_docker_available

    with patch("nemo_helix_plugin.config.probe_docker") as probe:
        probe.return_value = MagicMock(available=False)
        assert validate_docker_available() is False
        probe.assert_called_once()
