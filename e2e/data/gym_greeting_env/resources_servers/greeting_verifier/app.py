# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The greeting environment's resources server, which runs inside the sandboxed Gym host.

A row without a ``probe`` is scored on the agent's reply: 1.0 when it holds the greeting. A row with one
is scored on that probe alone, one fact about the sandbox the server runs in: 1.0 when it holds,
whatever the model replied. So each probe is a task of its own, and one below 1.0 names the fact that
failed. ``hang`` never answers, for a run that must fail. ``docker_compose`` needs the Docker-enabled Gym
host image and egress to Docker Hub.

The sandboxed Gym tests in e2e/test_evaluator_plugin.py upload this directory as an environment FileSet,
with ``greeting_extra``, this server's one extra dependency, as a wheel they build. NeMo Gym and
``greeting_extra`` are importable only inside the Gym host.
"""

import asyncio
import errno
import json
import os
import socket
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from greeting_extra import normalize  # ty: ignore[unresolved-import]
from nemo_gym.base_resources_server import (  # ty: ignore[unresolved-import]
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

ENVIRONMENT = Path(os.environ.get("NHX_ENVIRONMENT_PATH") or "/job/environment")
WORK = Path(os.environ.get("NHX_WORK_PATH") or "/job/work")


class GreetingConfig(BaseResourcesServerConfig):
    pass


class GreetingVerifyRequest(BaseVerifyRequest):
    expected_greeting: str
    probe: str = ""
    probe_target: str = ""


class GreetingVerifyResponse(BaseVerifyResponse):
    expected_greeting: str
    probe: str = ""
    probe_target: str = ""
    observed_text: str


def _assistant_text(response):
    parts = []
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "output_text":
                parts.append(content.text)
    return "".join(parts)


def _environment_staged(target):
    # The manifest this package was uploaded with, read from where the host mounted it.
    return "name: e2e-greeting" in (ENVIRONMENT / "nemo-environment.yaml").read_text()


def _environment_read_only(target):
    try:
        (ENVIRONMENT / "written-by-the-environment").write_text("refused")
    except OSError as error:
        # A permission or missing-path error would say nothing about how the path is mounted.
        return error.errno == errno.EROFS
    return False


def _workspace_writable(target):
    marker = WORK / "written-by-the-environment"
    marker.write_text("kept")
    return marker.read_text() == "kept"


def _connect(target):
    host, _, port = target.rpartition(":")
    socket.create_connection((host, int(port)), timeout=5).close()


def _connects(target):
    _connect(target)
    return True


# How a connection fails when egress is denied: the name is not resolved, the packets are dropped
# (a timeout), or the connection is refused, reset, unroutable or stopped by a local firewall rule.
BLOCKED_ERRNOS = {
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTUNREACH,
    errno.ENETUNREACH,
    errno.EPERM,
    errno.EACCES,
}


def _does_not_connect(target):
    try:
        _connect(target)
    except (socket.gaierror, TimeoutError):
        return True
    except OSError as error:
        # Any other error is a fault in the probe, which must not pass for a blocked destination.
        return error.errno in BLOCKED_ERRNOS
    return False


#: Where execd, OpenSandbox's command server, listens in every sandbox.
EXECD_COMMAND_URL = "http://127.0.0.1:44772/command"


def _execd_refuses(target):
    # Environment code must not hold execd's token, nor get a command run by sending none.
    if "EXECD_ACCESS_TOKEN" in os.environ:
        return False
    request = urllib.request.Request(
        EXECD_COMMAND_URL,
        data=json.dumps({"command": "id"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # An execd it cannot reach raises URLError, an OSError: the probe could not run, so it does not pass.
    try:
        with urllib.request.urlopen(request, timeout=10):
            return False
    except urllib.error.HTTPError as error:
        return error.code == 401


# Two nginx services and a client that must reach both by service name, get an HTTP answer from
# Docker Hub's registry, and get none from a public name or address the egress allowlist leaves out.
# An answer, not a connection, so that a DNS or TLS failure cannot pass for the egress policy.
DOCKER_COMPOSE_PROJECT = """\
services:
  web:
    image: docker.io/library/nginx:1.27-alpine
  web2:
    image: docker.io/library/nginx:1.27-alpine
  client:
    image: docker.io/library/alpine:3.20
    depends_on: [web, web2]
    command: >-
      sh -c "sleep 3;
      wget -qO- http://web | grep -q nginx && echo web-ok;
      wget -qO- http://web2 | grep -q nginx && echo web2-ok;
      wget -S -T 8 -O /dev/null https://registry-1.docker.io/v2/ 2>&1 | grep -q HTTP/ && echo allowed-answered;
      wget -S -T 8 -O /dev/null https://example.com 2>&1 | grep -q HTTP/ && echo denied-ANSWERED || echo denied-silent;
      wget -S -T 8 -O /dev/null https://1.1.1.1/ 2>&1 | grep -q HTTP/ && echo denied-ip-ANSWERED || echo denied-ip-silent"
"""
DOCKER_COMPOSE_EXPECTED = {"web-ok", "web2-ok", "allowed-answered", "denied-silent", "denied-ip-silent"}
DOCKER_SHELL_PROJECT = "services:\n  hello:\n    image: docker.io/library/alpine:3.20\n    command: echo shell-ok\n"


def _unprivileged():
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    privileges = {name: status[name].strip() for name in ("CapEff", "CapBnd", "NoNewPrivs")}
    return os.getuid() != 0 and privileges == {"CapEff": "0" * 16, "CapBnd": "0" * 16, "NoNewPrivs": "1"}


def _execd_token_readable():
    # Any process environment this server can read, the host's own included.
    for entry in os.listdir("/proc"):
        try:
            if b"EXECD_ACCESS_TOKEN=" in Path(f"/proc/{entry}/environ").read_bytes():
                return True
        except OSError:
            continue
    return False


def _compose_project_works():
    project = f"e2e-{os.getpid()}"
    compose = ["docker", "compose", "-p", project]
    with tempfile.TemporaryDirectory() as work:
        Path(work, "compose.yaml").write_text(DOCKER_COMPOSE_PROJECT)
        try:
            up = subprocess.run(
                [*compose, "up", "--abort-on-container-exit", "--exit-code-from", "client"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=600,
            )
        finally:
            subprocess.run([*compose, "down"], cwd=work, capture_output=True, timeout=120)
    said = {
        line.split("|", 1)[1].strip() for line in up.stdout.splitlines() if line.startswith("client") and "|" in line
    }
    return up.returncode == 0 and DOCKER_COMPOSE_EXPECTED <= said


def _compose_works_from_an_empty_environment():
    # An agent's shell tool, such as a Fabric Deep Agents local shell, runs commands with only the
    # agent's own variables: no PATH, HOME or DOCKER_HOST from the image. Both spellings must work.
    with tempfile.TemporaryDirectory() as work:
        Path(work, "compose.yaml").write_text(DOCKER_SHELL_PROJECT)
        for cli in ("docker compose", "docker-compose"):
            project = f"e2e-shell-{os.getpid()}-{cli.replace(' ', '-')}"
            script = (
                f"{cli} -p {project} up --abort-on-container-exit; status=$?; {cli} -p {project} down; exit $status"
            )
            shell = subprocess.run(
                ["/bin/sh", "-c", script], env={}, cwd=work, capture_output=True, text=True, timeout=300
            )
            if shell.returncode != 0 or "shell-ok" not in shell.stdout:
                return False
    return True


def _docker_compose(target):
    # Unprivileged, unable to read execd's token anywhere, and still able to run Compose projects.
    return (
        _unprivileged()
        and not _execd_token_readable()
        and _compose_project_works()
        and _compose_works_from_an_empty_environment()
    )


# Each takes the probe's target, whether or not it needs one, so all can be called the same way.
PROBES = {
    "environment_staged": _environment_staged,
    "environment_read_only": _environment_read_only,
    "workspace_writable": _workspace_writable,
    "egress_allowed": _connects,
    "egress_denied": _does_not_connect,
    "execd_refuses": _execd_refuses,
    "docker_compose": _docker_compose,
}


class GreetingVerifier(SimpleResourcesServer):
    config: GreetingConfig

    async def verify(self, body: GreetingVerifyRequest) -> GreetingVerifyResponse:
        if body.probe == "hang":
            await asyncio.sleep(3600)
        observed = _assistant_text(body.response)
        if not body.probe:
            held = normalize(body.expected_greeting) in normalize(observed)
        else:
            try:
                # Off the event loop: a denied connection takes its whole timeout to fail.
                held = await asyncio.to_thread(PROBES[body.probe], body.probe_target)
            except (KeyError, OSError, ValueError):
                # An unknown probe, or one that could not run, has not shown its fact.
                held = False
        return GreetingVerifyResponse(**body.model_dump(), reward=float(held), observed_text=observed)


if __name__ == "__main__":
    GreetingVerifier.run_webserver()
