# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform-owned HTTP listeners for local Docker workloads."""

import ipaddress
import logging
import os
import platform
import socket
from collections.abc import Generator
from contextlib import closing, contextmanager

logger = logging.getLogger(__name__)


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def docker_listener_host(host: str) -> str | None:
    """Find a local Docker bridge interface without changing the primary bind.

    macOS Docker Desktop reaches the native host through its own host routing;
    it does not need or expose a Linux bridge interface to bind on the Mac.
    A remote daemon's bridge belongs to another machine and must not be
    advertised or bound here. Probe the address
    locally too: a platform in a separate container/rootless namespace cannot
    bind the daemon host's interface, while a host-network process can.
    """
    if platform.system() != "Linux" or not _loopback(host):
        return None
    docker_host = os.environ.get("DOCKER_HOST", "")
    if docker_host and not docker_host.startswith("unix://"):
        return None

    from nemo_helix_plugin.config import Runtime, get_platform_config

    if get_platform_config().runtime != Runtime.DOCKER:
        return None

    from docker.errors import DockerException

    import docker

    try:
        with closing(docker.from_env(timeout=3)) as client:
            configs = client.networks.get("bridge").attrs.get("IPAM", {}).get("Config", [])
        for config in configs:
            gateway = config.get("Gateway")
            if not gateway:
                continue
            address = ipaddress.ip_address(gateway)
            if address.version != 4 or address.is_loopback or address.is_unspecified:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((str(address), 0))
            return str(address)
    except (DockerException, OSError, ValueError):
        logger.warning(
            "Local Docker bridge listener unavailable; container access needs an explicit reachable endpoint",
            exc_info=True,
        )
    return None


@contextmanager
def tcp_listener_sockets(
    host: str, port: int, *, docker_host: str | None = None
) -> Generator[list[socket.socket], None, None]:
    """Bind the primary and Docker-facing listeners to one port and own cleanup."""
    sockets: list[socket.socket] = []
    try:
        hosts = [host]
        if docker_host is not None and _loopback(host) and docker_host not in hosts:
            hosts.append(docker_host)
        for address in hosts:
            family = socket.AF_INET6 if ":" in address else socket.AF_INET
            listener = socket.socket(family, socket.SOCK_STREAM)
            sockets.append(listener)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((address, port))
            listener.set_inheritable(True)
            if port == 0:
                port = listener.getsockname()[1]
        yield sockets
    finally:
        for listener in sockets:
            listener.close()
