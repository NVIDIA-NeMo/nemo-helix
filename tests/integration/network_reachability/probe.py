# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport probe shared by the endpoint contract tests."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

IMAGE = "python:3.12-slim-bookworm"


class ReachabilityFailure(AssertionError):
    """The selected endpoint failed a real request across the namespace boundary."""


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, timeout=30)


@dataclass(frozen=True)
class DockerProbe:
    name: str
    image: str
    directory: Path
    host: str
    gateway: str | None

    @property
    def external(self) -> str:
        return f"http://{self.host}:9090"

    def post(self, url: str, *, shared_host_namespace: bool = False) -> dict:
        network = f"container:{self.name}" if shared_host_namespace else self.name
        result = docker(
            "run",
            "--rm",
            "--network",
            network,
            *([] if shared_host_namespace else ["--add-host", f"host.docker.internal:{self.host}"]),
            "--mount",
            f"type=bind,src={self.directory},dst=/probe,readonly",
            "--env",
            f"PROBE_URL={url}",
            self.image,
            "python",
            "/probe/client.py",
        )
        return json.loads(result.stdout)

    def require_delivery(self, url: str, *, shared_host_namespace: bool = False, port: int = 8080) -> None:
        result = self.post(url, shared_host_namespace=shared_host_namespace)
        if "network_error" in result:
            raise ReachabilityFailure(f"{url}: {result['network_error']}: {result['detail']}")
        assert result["status"] == 200, result
        received = result["received"]
        assert received["path"] == urlsplit(url).path + ("?" + urlsplit(url).query if urlsplit(url).query else ""), (
            result
        )
        assert received["body"] == '{"probe":"network-reachability"}', result
        assert received["authorization"] == "Bearer probe-only-not-a-credential", result
        assert received["listener_port"] == port, result
