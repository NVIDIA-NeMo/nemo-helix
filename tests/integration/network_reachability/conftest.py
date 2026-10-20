# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in suite: missing prerequisites fail after explicit opt-in, never skip."""

import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.integration.network_reachability.probe import IMAGE, DockerProbe, docker


def pytest_addoption(parser):
    parser.addoption(
        "--include-known-network-gaps",
        action="store_true",
        help="Run explicitly skipped custom host-collector and bridge Jobs contracts as ordinary tests",
    )
    parser.addoption(
        "--reachability-platform-bind",
        choices=("loopback", "all"),
        default="loopback",
        help="Diagnostic platform listener binding inside isolated Linux namespace",
    )
    parser.addoption(
        "--run-network-reachability", action="store_true", help="Require and run Linux Docker network probes"
    )


@pytest.fixture(scope="module")
def network_probe(request) -> Iterator[DockerProbe]:
    if not request.config.getoption("--run-network-reachability"):
        pytest.skip("Enable explicitly with --run-network-reachability")
    if not shutil.which("docker"):
        pytest.fail("Reachability suite requires Docker CLI")
    for args in [("info",), ("image", "inspect", IMAGE)]:
        try:
            docker(*args)
        except (OSError, subprocess.SubprocessError) as error:
            pytest.fail(f"Reachability prerequisite failed: {args}: {error}. Pre-pull {IMAGE}.")
    name = f"helix-network-probe-{uuid.uuid4().hex[:10]}"
    directory = Path(__file__).parent.resolve()
    source_root = Path(__file__).resolve().parents[3]
    listener = source_root / "packages/nhx_platform_runner/src/nhx/platform_runner/local_network.py"
    listener_mount = (
        ["--mount", f"type=bind,src={listener},dst=/platform_local_network.py,readonly"] if listener.is_file() else []
    )
    forwarder = source_root / "plugins/nemo-evals/src/nemo_evals/jobs/harbor_forwarder.py"
    forwarder_mount = (
        ["--mount", f"type=bind,src={forwarder},dst=/harbor_forwarder.py,readonly"] if forwarder.is_file() else []
    )
    docker("network", "create", "--internal", name)
    try:
        docker(
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--network",
            name,
            "--mount",
            f"type=bind,src={directory},dst=/probe,readonly",
            *forwarder_mount,
            *listener_mount,
            "--env",
            f"PROBE_PLATFORM_BIND={request.config.getoption('--reachability-platform-bind')}",
            IMAGE,
            "python",
            "/probe/server.py",
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            logs = docker("logs", name).stdout.strip()
            if logs:
                try:
                    ready = json.loads(logs.splitlines()[0])
                except json.JSONDecodeError:
                    pytest.fail(f"Linux host surrogate returned invalid readiness output:\n{logs}")
                yield DockerProbe(name, IMAGE, directory, ready["host"], ready["gateway"])
                return
            if docker("inspect", "--format", "{{.State.Running}}", name).stdout.strip() != "true":
                break
            time.sleep(0.1)
        logs = docker("logs", name, check=False)
        pytest.fail(f"Linux host surrogate failed readiness:\n{logs.stdout}\n{logs.stderr}")
    finally:
        docker("rm", "--force", name, check=False)
        docker("network", "rm", name, check=False)


@pytest.fixture
def native_linux_host(monkeypatch):
    for key in os.environ:
        if key.startswith("NHX_") and (key.endswith("_URL") or key == "NHX_LOOPBACK_ADDRESS"):
            monkeypatch.delenv(key)
    # Only OS/context facts are supplied. Never stub endpoint resolution or HTTP.
    monkeypatch.setattr("nemo_helix_plugin.config.platform.system", lambda: "Linux")
    monkeypatch.setattr("nemo_helix_plugin.config._is_running_in_docker", lambda: False)
