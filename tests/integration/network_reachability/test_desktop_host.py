# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker Desktop containers reaching a listener in the real macOS host process."""

import json
import platform
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from nemo_agents_plugin.runner.deployments_backend import resolve_agent_gateway_url
from nhx.platform_runner.local_network import docker_listener_host

from tests.integration.network_reachability.probe import IMAGE, docker

pytestmark = [pytest.mark.integration, pytest.mark.skipif(platform.system() != "Darwin", reason="macOS host routing")]


@pytest.mark.parametrize("host_gateway", [False, True], ids=["desktop-dns", "harbor-host-gateway"])
def test_desktop_container_reaches_host_loopback(request, host_gateway):
    """Use actual Desktop routing, including Harbor's host-gateway override."""
    if not request.config.getoption("--run-network-reachability"):
        pytest.skip("Enable explicitly with --run-network-reachability")
    assert shutil.which("docker"), "Docker CLI is required after opt-in"
    info = docker("info", "--format", "{{.OperatingSystem}}")
    assert "Docker Desktop" in info.stdout, info.stdout
    docker("image", "inspect", IMAGE)
    assert docker_listener_host("127.0.0.1") is None

    class Echo(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.path.encode())

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Echo) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            endpoint = resolve_agent_gateway_url(base, mode="docker") + "/host-probe?source=desktop"
            script = (
                "import json, os, urllib.request; "
                "opener=urllib.request.build_opener(urllib.request.ProxyHandler({})); "
                "response=opener.open(os.environ['PROBE_URL'], timeout=5); "
                "print(json.dumps({'status':response.status, 'body':response.read().decode()}))"
            )
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    *(["--add-host", "host.docker.internal:host-gateway"] if host_gateway else []),
                    "--env",
                    f"PROBE_URL={endpoint}",
                    IMAGE,
                    "python",
                    "-c",
                    script,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert json.loads(result.stdout) == {"status": 200, "body": "/host-probe?source=desktop"}
        finally:
            server.shutdown()
            thread.join(timeout=5)
