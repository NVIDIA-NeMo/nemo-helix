# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import socket
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from nemo_helix_plugin.config import Runtime
from nhx.platform_runner import local_network

import docker


@pytest.fixture
def local_docker(monkeypatch):
    monkeypatch.setattr(local_network.platform, "system", lambda: "Linux")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr("nemo_helix_plugin.config.get_platform_config", lambda: SimpleNamespace(runtime=Runtime.DOCKER))
    client = MagicMock(spec=docker.DockerClient)
    client.networks = MagicMock()
    client.networks.get.return_value.attrs = {"IPAM": {"Config": [{"Gateway": "172.17.0.1"}]}}
    create = MagicMock(return_value=client)
    monkeypatch.setattr("docker.from_env", create)
    return client, create


def test_native_linux_discovers_and_validates_local_bridge(local_docker, monkeypatch):
    client, create = local_docker
    sockets = MagicMock()
    monkeypatch.setattr(local_network.socket, "socket", sockets)
    assert local_network.docker_listener_host("127.0.0.1") == "172.17.0.1"
    sockets.return_value.__enter__.return_value.bind.assert_called_once_with(("172.17.0.1", 0))
    client.close.assert_called_once()
    create.assert_called_once_with(timeout=3)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.0.2.2"])
def test_explicit_non_loopback_binding_is_not_expanded(local_docker, host):
    _, create = local_docker
    assert local_network.docker_listener_host(host) is None
    create.assert_not_called()


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_desktop_routing_is_preserved(local_docker, monkeypatch, system):
    _, create = local_docker
    monkeypatch.setattr(local_network.platform, "system", lambda: system)
    assert local_network.docker_listener_host("127.0.0.1") is None
    create.assert_not_called()


def test_remote_daemon_does_not_create_local_listener(local_docker, monkeypatch):
    _, create = local_docker
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.example:2376")
    assert local_network.docker_listener_host("127.0.0.1") is None
    create.assert_not_called()


def test_no_docker_runtime_does_not_create_listener(local_docker, monkeypatch):
    _, create = local_docker
    monkeypatch.setattr("nemo_helix_plugin.config.get_platform_config", lambda: SimpleNamespace(runtime=Runtime.NONE))
    monkeypatch.setitem(sys.modules, "docker", None)
    monkeypatch.setitem(sys.modules, "docker.errors", None)
    assert local_network.docker_listener_host("127.0.0.1") is None
    create.assert_not_called()


def test_bridge_in_another_namespace_is_not_advertised(local_docker, monkeypatch):
    sockets = MagicMock()
    sockets.return_value.__enter__.return_value.bind.side_effect = OSError("address not available")
    monkeypatch.setattr(local_network.socket, "socket", sockets)
    assert local_network.docker_listener_host("127.0.0.1") is None


def test_tcp_listener_cleanup_on_partial_bind_failure():
    try:
        with socket.socket(socket.AF_INET6) as probe:
            probe.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback is unavailable on this host")
    # Reserve the second endpoint; the first socket must be released on failure.
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        with pytest.raises(OSError):
            with local_network.tcp_listener_sockets("::1", port, docker_host="127.0.0.1"):
                pytest.fail("The occupied second listener must fail startup")
        with socket.socket(socket.AF_INET6) as replacement:
            replacement.bind(("::1", port))


def test_tcp_listener_cleanup_after_shutdown():
    with local_network.tcp_listener_sockets("127.0.0.1", 0) as listeners:
        assert len(listeners) == 1
        address = listeners[0].getsockname()
        assert listeners[0].fileno() >= 0
    assert listeners[0].fileno() == -1
    with socket.socket() as replacement:
        replacement.bind(address)
