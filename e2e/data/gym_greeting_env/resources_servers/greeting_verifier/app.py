# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The greeting environment's resources server, which runs inside the sandboxed Gym host.

A row without a ``probe`` is scored on the agent's reply: 1.0 when it holds the greeting. A row with one
is scored on that probe alone, one fact about the sandbox the server runs in: 1.0 when it holds,
whatever the model replied. So each probe is a task of its own, and one below 1.0 names the fact that
failed. ``hang`` never answers, for a run that must fail.

The sandboxed Gym tests in e2e/test_evaluator_plugin.py upload this directory as an environment FileSet,
with ``greeting_extra``, this server's one extra dependency, as a wheel they build. NeMo Gym and
``greeting_extra`` are importable only inside the Gym host.
"""

import asyncio
import errno
import os
import socket
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


# Each takes the probe's target, whether or not it needs one, so all can be called the same way.
PROBES = {
    "environment_staged": _environment_staged,
    "environment_read_only": _environment_read_only,
    "workspace_writable": _workspace_writable,
    "egress_allowed": _connects,
    "egress_denied": _does_not_connect,
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
