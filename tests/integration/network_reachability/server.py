# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linux host surrogate; run only inside the suite's isolated Docker network."""

import json
import os
import runpy
import socket
import threading
import time
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Echo(BaseHTTPRequestHandler):
    def do_POST(self):
        assert isinstance(self.server, ThreadingHTTPServer)
        body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.dumps(
            {
                "path": self.path,
                "body": body.decode(),
                "authorization": self.headers.get("Authorization"),
                "listener_port": self.server.server_port,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


host = socket.gethostbyname(socket.gethostname())
# Default stays loopback-only. Widen only for the explicit diagnostic experiment.
platform_bind = "0.0.0.0" if os.environ.get("PROBE_PLATFORM_BIND") == "all" else "127.0.0.1"


def start_server(address, port, listener=None):
    server = ThreadingHTTPServer((address, port), Echo, bind_and_activate=listener is None)
    if listener is not None:
        server.socket.close()
        server.socket = listener
        server.server_address = listener.getsockname()
        server.server_port = port
        server.server_name = address
        server.server_activate()
    threading.Thread(target=server.serve_forever, daemon=True).start()


# Keep a separate bare-loopback listener as a negative topology control even
# after the production platform intentionally serves both loopback and Docker.
for address, port in [("127.0.0.1", 8081), ("127.0.0.1", 4318), (host, 9090)]:
    start_server(address, port)

# These are container-only destinations mounted by conftest.py.
# The host checkout path is resolved independently by the fixture.
with ExitStack() as stack:
    if Path("/platform_local_network.py").is_file():
        bind = runpy.run_path("/platform_local_network.py")["tcp_listener_sockets"]
        listeners = stack.enter_context(bind(platform_bind, 8080, docker_host=host))
        for listener in listeners:
            start_server(listener.getsockname()[0], 8080, listener)
    else:
        start_server(platform_bind, 8080)
    gateway = None
    if Path("/harbor_forwarder.py").is_file():
        forward = runpy.run_path("/harbor_forwarder.py")["loopback_forwarder"]
        gateway = stack.enter_context(forward("http://127.0.0.1:8080", host))
    print(json.dumps({"host": host, "gateway": gateway}), flush=True)
    while True:
        time.sleep(1)
