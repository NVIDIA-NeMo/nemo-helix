# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative recovery for stale CI testcontainers bridge networks."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from docker.errors import APIError
from nhx.testing.e2e import docker as backend

EXHAUSTED = "all predefined address pools have been fully subnetted"


def test_normal_creation_never_prunes(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    docker = MagicMock()
    monkeypatch.setattr(backend.docker, "from_env", docker)
    network = MagicMock()
    backend._create_test_network(network)
    network.create.assert_called_once()
    docker.assert_not_called()


def test_ci_exhaustion_prunes_only_old_unused_testcontainers_networks(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    # DockerClient has close(), but no context-manager protocol. Do not use an
    # unrestricted MagicMock client that would invent __enter__/__exit__.
    client = SimpleNamespace(networks=MagicMock(), close=MagicMock())
    docker = MagicMock(return_value=client)
    client.networks.prune.return_value = {"NetworksDeleted": ["stale"]}
    monkeypatch.setattr(backend.docker, "from_env", docker)
    network = MagicMock()
    network.create.side_effect = [APIError(EXHAUSTED), None]
    backend._create_test_network(network)
    assert network.create.call_count == 2
    client.networks.prune.assert_called_once_with(
        filters={"label": ["org.testcontainers=true", "org.testcontainers.lang=python"], "until": "24h"}
    )
    client.close.assert_called_once()


@pytest.mark.parametrize("ci,error", [(False, EXHAUSTED), (True, "permission denied")])
def test_local_or_unrelated_errors_are_not_recovered(monkeypatch, ci, error):
    monkeypatch.setenv("GITHUB_ACTIONS", "true" if ci else "false")
    docker = MagicMock()
    monkeypatch.setattr(backend.docker, "from_env", docker)
    network = MagicMock()
    network.create.side_effect = APIError(error)
    with pytest.raises(APIError, match=error):
        backend._create_test_network(network)
    network.create.assert_called_once()
    docker.assert_not_called()


def test_continued_exhaustion_is_not_ignored(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    client = SimpleNamespace(networks=MagicMock(), close=MagicMock())
    docker = MagicMock(return_value=client)
    monkeypatch.setattr(backend.docker, "from_env", docker)
    network = MagicMock()
    network.create.side_effect = APIError(EXHAUSTED)
    with pytest.raises(APIError, match=EXHAUSTED):
        backend._create_test_network(network)
    assert network.create.call_count == 2
    client.networks.prune.assert_called_once()
    client.close.assert_called_once()


def test_prune_failure_closes_client_and_does_not_retry(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    client = SimpleNamespace(networks=MagicMock(), close=MagicMock())
    client.networks.prune.side_effect = APIError("prune denied")
    monkeypatch.setattr(backend.docker, "from_env", lambda: client)
    network = MagicMock()
    network.create.side_effect = APIError(EXHAUSTED)
    with pytest.raises(APIError, match="prune denied"):
        backend._create_test_network(network)
    network.create.assert_called_once()
    client.close.assert_called_once()
