# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An in-memory Secrets service for unit tests.

Answers secret access requests from a dict and records which secrets were asked for, behind a
real :class:`AsyncSecretsClient`::

    secrets = InMemorySecretsService({("default", "judge-key"): "sk-test"})
    value = await resolve(secrets.client())
    assert secrets.lookups == [("default", "judge-key")]
"""

import re
from urllib.parse import unquote

import httpx
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretAccessResponse
from nhx.testing.client import mock_async_nemo_client

_ACCESS = re.compile(r"^/apis/secrets/v2/workspaces/(?P<workspace>[^/]+)/secrets/(?P<name>[^/]+)/access$")


class InMemorySecretsService:
    """Serves ``access_secret`` from ``(workspace, name) -> value`` and records each lookup."""

    def __init__(self, secrets: dict[tuple[str, str], str] | None = None) -> None:
        self.secrets: dict[tuple[str, str], str] = dict(secrets or {})
        #: Every ``(workspace, name)`` whose value was requested, in order.
        self.lookups: list[tuple[str, str]] = []

    def client(self, *, workspace: str = "default") -> AsyncSecretsClient:
        """Return a real async Secrets client backed by this service."""
        return mock_async_nemo_client(self, AsyncSecretsClient, workspace=workspace)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        match = _ACCESS.match(unquote(request.url.path))
        if match is None or request.method != "GET":
            return httpx.Response(
                404, json={"detail": f"no in-memory Secrets route for {request.method} {request.url.path}"}
            )
        key = (match["workspace"], match["name"])
        self.lookups.append(key)
        if key not in self.secrets:
            return httpx.Response(404, json={"detail": f"secret {key[0]}/{key[1]} not found"})
        access = HelixSecretAccessResponse(name=key[1], workspace=key[0], value=self.secrets[key])
        return httpx.Response(200, json=access.model_dump(mode="json"))
