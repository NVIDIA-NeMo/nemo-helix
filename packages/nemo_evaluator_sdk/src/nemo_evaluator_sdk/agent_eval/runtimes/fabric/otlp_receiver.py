# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OTLP endpoint Relay exports to, for both Fabric runtimes.

Relay exports OTLP only over the network, so capturing a trace means listening for one. The local
runtime imports this and runs it in a thread; the container runtime seeds this same file into the
sandbox and runs it as a script beside the harness, where it writes into the directory that is
downloaded when the task ends.

Standard library only, and it stays that way: the sandbox image ships Fabric and its harnesses, not
this package, so an import of anything else would have to be installed there first. That is also
why each export is stored verbatim rather than as OTLP/JSON: decoding needs protobuf, which the
image does not guarantee, so it happens after the exports are back on the host.
"""

from __future__ import annotations

import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import TracebackType
from urllib.parse import urlparse

TRACES_PATH = "/v1/traces"
PROTOBUF_MEDIA_TYPE = "application/x-protobuf"
MAX_EXPORT_BYTES = 256 * 1024 * 1024
REQUEST_TIMEOUT_S = 30.0
#: Written once the socket is bound, so the caller can start Relay without racing the listener.
READY_FILENAME = "receiver.ready"
#: Read back in export order, which `sorted()` gives while the counter is zero-padded.
EXPORT_SUFFIX = ".otlp.pb"


class _Handler(BaseHTTPRequestHandler):
    server: OTLPReceiver  # type: ignore[assignment]
    timeout = REQUEST_TIMEOUT_S

    def do_POST(self) -> None:
        if urlparse(self.path).path != TRACES_PATH:
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != PROTOBUF_MEDIA_TYPE:
            self.send_response(415)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return
        # Bounded below as well as above: ``rfile.read(-1)`` reads to EOF, so a negative length would
        # hold this single-threaded receiver open for as long as the client kept the connection, and
        # the agent sharing this loopback is untrusted code.
        if length < 0:
            self.send_response(400)
            self.end_headers()
            return
        if length > MAX_EXPORT_BYTES:
            self.send_response(413)
            self.end_headers()
            return
        body = self.rfile.read(length)
        try:
            self.server.exports += 1
            target = self.server.directory / f"{self.server.exports:06d}{EXPORT_SUFFIX}"
            target.write_bytes(body)
        except OSError:
            # The agent is still running and its result stands on its own.
            self.send_response(500)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", PROTOBUF_MEDIA_TYPE)
        self.end_headers()
        self.wfile.write(b"")

    def log_message(self, format: str, *args: object) -> None:
        return


class OTLPReceiver(HTTPServer):
    """Loopback OTLP endpoint that stores each export verbatim under ``directory``.

    Used two ways: as a context manager, which serves on its own thread, and as a script via
    :func:`main`, which serves on the main one.
    """

    def __init__(self, directory: Path, port: int = 0) -> None:
        # Port 0 lets the OS pick, which is what the local runtime wants so parallel trials cannot
        # collide. The container passes a fixed one because it has a sandbox to itself.
        super().__init__(("127.0.0.1", port), _Handler)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.exports = 0
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        return f"http://{self.server_address[0]}:{self.server_address[1]}{TRACES_PATH}"

    def signal_ready(self) -> None:
        """Announce that the socket is bound, for a caller that has to wait out of process."""
        (self.directory / READY_FILENAME).write_text("ready\n", encoding="utf-8")

    def __enter__(self) -> OTLPReceiver:
        self._thread = threading.Thread(target=self.serve_forever, name="otlp-receiver", daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self.server_close()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0] if argv else 'receiver'} <directory> <port>", file=sys.stderr)
        return 2
    directory, port = Path(argv[1]), int(argv[2])
    receiver = OTLPReceiver(directory, port)
    receiver.signal_ready()
    try:
        receiver.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        receiver.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
