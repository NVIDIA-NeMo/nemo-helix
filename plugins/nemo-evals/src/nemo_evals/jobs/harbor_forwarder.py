# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation-scoped forwarding from Linux Docker's bridge to a loopback platform."""

import json
import select
import socket
import socketserver
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from ipaddress import ip_address
from typing import cast
from urllib.parse import urlsplit


class _Forwarder(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, bind_host: str, upstream: tuple[str, int]) -> None:
        self.upstream = upstream
        self.stopping = threading.Event()
        self.connections: set[socket.socket] = set()
        self.connections_lock = threading.Lock()
        super().__init__((bind_host, 0), _ForwardConnection)


class _ForwardConnection(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = cast(_Forwarder, self.server)
        with socket.create_connection(server.upstream, timeout=10) as upstream:
            streams = (self.request, upstream)
            with server.connections_lock:
                server.connections.update(streams)
            try:
                self._relay(streams, server.stopping)
            except (ConnectionError, OSError):
                return
            finally:
                with server.connections_lock:
                    server.connections.difference_update(streams)

    @staticmethod
    def _relay(streams: tuple[socket.socket, socket.socket], stopping: threading.Event) -> None:
        buffers = {stream: bytearray() for stream in streams}
        peers = {streams[0]: streams[1], streams[1]: streams[0]}
        closed: set[socket.socket] = set()
        write_closed: set[socket.socket] = set()
        for stream in streams:
            stream.setblocking(False)
        while not stopping.is_set():
            readers = [stream for stream in streams if stream not in closed and len(buffers[peers[stream]]) < 65536]
            writers = [stream for stream in streams if buffers[stream]]
            if not readers and not writers:
                return
            readable, writable, _ = select.select(readers, writers, [], 0.1)
            for source in readable:
                destination = peers[source]
                try:
                    data = source.recv(65536)
                except BlockingIOError:
                    continue
                if data:
                    buffers[destination].extend(data)
                else:
                    closed.add(source)
            for destination in writable:
                try:
                    count = destination.send(buffers[destination])
                except BlockingIOError:
                    continue
                del buffers[destination][:count]
            for source in closed:
                destination = peers[source]
                if not buffers[destination] and destination not in write_closed:
                    destination.shutdown(socket.SHUT_WR)
                    write_closed.add(destination)


def _docker_bridge_host() -> str:
    """Get the host interface used by Docker's default bridge, without changing it."""
    result = subprocess.run(
        ["docker", "network", "inspect", "bridge", "--format", "{{json .IPAM.Config}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    for config in json.loads(result.stdout):
        gateway = config.get("Gateway")
        if gateway and ":" not in gateway:
            return gateway
    raise RuntimeError("Docker's default bridge has no IPv4 host gateway")


@contextmanager
def loopback_forwarder(base_url: str, bind_host: str) -> Iterator[str]:
    """Expose a loopback service on one Docker interface for the lifetime of a run.

    Bytes are forwarded unchanged, including streaming bodies and the caller's
    authentication headers. No service identity or credential is added.
    """
    if ip_address(bind_host).is_unspecified:
        raise ValueError("The Docker forwarder must bind a specific interface")
    parts = urlsplit(base_url)
    if parts.hostname != "localhost" and not (
        parts.hostname and (ip_address(parts.hostname).is_loopback or parts.hostname == "0.0.0.0")
    ):
        raise ValueError("The Docker forwarder only accepts loopback upstreams")
    host = "127.0.0.1" if parts.hostname == "0.0.0.0" else parts.hostname
    with _Forwarder(bind_host, (host, parts.port or (443 if parts.scheme == "https" else 80))) as server:
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        thread.start()
        try:
            yield f"{parts.scheme}://{bind_host}:{server.server_address[1]}"
        finally:
            server.stopping.set()
            with server.connections_lock:
                for connection in server.connections:
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            server.shutdown()
            thread.join(timeout=5)
