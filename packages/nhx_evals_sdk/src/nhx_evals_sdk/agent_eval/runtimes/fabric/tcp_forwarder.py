# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A loopback TCP forwarder, run beside a harness so Relay can reach a plain-HTTP model gateway.

Relay sends model credentials only over HTTPS or to a loopback host, and a task container reaches the platform's
gateway over plain HTTP. Listening on loopback and copying bytes both ways leaves every request untouched:
headers, streamed responses and keep-alive all pass through as they were sent.

Standard library only, like the OTLP receiver: it runs as a script in a container that has Fabric's interpreter
and nothing of this package.
"""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

LISTEN_HOST = "127.0.0.1"
CONNECT_TIMEOUT_S = 30.0
_CHUNK_BYTES = 64 * 1024


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    try:
        while chunk := source.recv(_CHUNK_BYTES):
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        # Half-close, so the other direction can still finish a response already in flight.
        for close in (lambda: sink.shutdown(socket.SHUT_WR), lambda: source.shutdown(socket.SHUT_RD)):
            try:
                close()
            except OSError:
                pass


def _serve(client: socket.socket, upstream_host: str, upstream_port: int) -> None:
    with client:
        try:
            upstream = socket.create_connection((upstream_host, upstream_port), timeout=CONNECT_TIMEOUT_S)
        except OSError:
            return
        with upstream:
            upstream.settimeout(None)
            outbound = threading.Thread(target=_pipe, args=(client, upstream), daemon=True)
            outbound.start()
            _pipe(upstream, client)
            outbound.join()


def forward(listener: socket.socket, upstream_host: str, upstream_port: int) -> None:
    """Forward every connection ``listener`` accepts to ``upstream_host:upstream_port``, until it is closed."""
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        threading.Thread(target=_serve, args=(client, upstream_host, upstream_port), daemon=True).start()


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(f"usage: {argv[0] if argv else 'forwarder'} <listen port> <upstream host> <upstream port> <ready file>")
        return 2
    listen_port, upstream_host, upstream_port, ready = int(argv[1]), argv[2], int(argv[3]), Path(argv[4])
    listener = socket.create_server((LISTEN_HOST, listen_port))
    ready.write_text("ready\n", encoding="utf-8")
    try:
        forward(listener, upstream_host, upstream_port)
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
