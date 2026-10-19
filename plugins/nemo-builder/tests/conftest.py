# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test doubles shared by the builder's tests, as fixtures: ``from conftest import`` is ambiguous across roots."""

from __future__ import annotations

import hashlib
import json
import re

import httpx
import pytest
from nemo_builder_plugin.registry import OCI_MANIFEST

#: What the fake registry requires of every request.
AUTHORIZATION = "Bearer registry-token"


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class FakeOCIRegistry:
    """A registry in memory, over httpx: manifests by tag and digest, blobs, and uploads in one request."""

    def __init__(self) -> None:
        self.manifests: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.blobs: dict[tuple[str, str], bytes] = {}
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def push(self, repository: str, tag: str, manifest: dict[str, object], media_type: str = OCI_MANIFEST) -> str:
        """Store ``manifest`` under ``tag`` and under its digest, as a push would. Returns the digest."""
        body = json.dumps(manifest).encode()
        self.manifests[(repository, tag)] = self.manifests[(repository, _digest(body))] = (body, media_type)
        return _digest(body)

    def manifest(self, repository: str, reference: str) -> dict[str, object]:
        return json.loads(self.manifests[(repository, reference)][0])

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("Authorization") != AUTHORIZATION:
            return httpx.Response(401)
        if found := re.fullmatch(r"/v2/(.+)/manifests/([^/]+)", request.url.path):
            repository, reference = found.groups()
            if request.method == "GET":
                if (repository, reference) not in self.manifests:
                    return httpx.Response(404)
                body, media_type = self.manifests[(repository, reference)]
                return httpx.Response(200, content=body, headers={"Content-Type": media_type})
            manifest = json.loads(request.content)
            if any(
                (repository, blob["digest"]) not in self.blobs for blob in [manifest["config"], *manifest["layers"]]
            ):
                return httpx.Response(400)
            stored = (request.content, request.headers["Content-Type"])
            self.manifests[(repository, reference)] = self.manifests[(repository, _digest(request.content))] = stored
            return httpx.Response(201)
        if (found := re.fullmatch(r"/v2/(.+)/blobs/uploads/", request.url.path)) and request.method == "POST":
            return httpx.Response(
                202, headers={"Location": f"/v2/{found.group(1)}/blobs/uploads/session?_state=opaque"}
            )
        if (found := re.fullmatch(r"/v2/(.+)/blobs/uploads/session", request.url.path)) and request.method == "PUT":
            digest = request.url.params.get("digest")
            if request.url.params.get("_state") != "opaque" or digest != _digest(request.content):
                return httpx.Response(400)
            self.blobs[(found.group(1), digest)] = request.content
            return httpx.Response(201)
        return httpx.Response(404)


@pytest.fixture
def oci_registry() -> FakeOCIRegistry:
    return FakeOCIRegistry()
