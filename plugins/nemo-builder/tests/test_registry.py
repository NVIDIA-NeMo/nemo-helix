# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The registry client, against a fake registry that speaks the OCI Distribution auth flow.

The minikube run could not exercise any of this: its registry is anonymous and plain HTTP, so
the challenge / token-endpoint path never executes there. A real registry -- GAR, measured --
issues short-lived bearer tokens, which is exactly where the adversarial review found the bug.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from nemo_builder_plugin.registry import ReferenceNotFound, RegistryClient, RegistryError

REGISTRY = "reg.example.com"
REPO = "team/app"
DIGEST = "sha256:" + "e" * 64


class FakeRegistry:
    """Issues bearer tokens, accepts only current ones, and can expire them all at once."""

    def __init__(self, *, accept_tokens: bool = True) -> None:
        self.valid: set[str] = set()
        self.issued = 0
        self.manifest_requests = 0
        self.accept_tokens = accept_tokens

    def expire_all_tokens(self) -> None:
        self.valid.clear()

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            self.issued += 1
            token = f"t{self.issued}"
            if self.accept_tokens:
                self.valid.add(token)
            return httpx.Response(200, json={"token": token})

        self.manifest_requests += 1
        auth = request.headers.get("Authorization", "")
        if not (auth.startswith("Bearer ") and auth.removeprefix("Bearer ") in self.valid):
            return httpx.Response(
                401,
                headers={
                    "WWW-Authenticate": (
                        f'Bearer realm="https://{REGISTRY}/token",service="{REGISTRY}",scope="repository:{REPO}:pull"'
                    )
                },
            )
        if request.url.path.endswith("/manifests/missing"):
            return httpx.Response(404)
        return httpx.Response(
            200,
            headers={
                "Docker-Content-Digest": DIGEST,
                "Content-Type": "application/vnd.oci.image.manifest.v1+json",
            },
            content=json.dumps({"schemaVersion": 2}).encode(),
        )


def _client(fake: FakeRegistry) -> RegistryClient:
    return RegistryClient(username="u", password="p", transport=httpx.MockTransport(fake.handler))


class TestTokenLifetime:
    def test_a_stale_cached_token_is_replaced_not_reused(self) -> None:
        """The regression for the adversarial review's finding.

        The token cache was never invalidated: once a cached bearer token expired, every lookup
        presented it, the single retry 401'd again, and the reconciler stopped resolving anything
        for the rest of its life -- within the hour against GAR.
        """
        fake = FakeRegistry()
        client = _client(fake)
        assert client.resolve(REGISTRY, REPO, "v1").digest == DIGEST

        fake.expire_all_tokens()

        assert client.resolve(REGISTRY, REPO, "v1").digest == DIGEST
        assert fake.issued == 2, "a fresh token should have been exchanged after the 401"

    def test_a_valid_cached_token_is_sent_on_the_first_request(self) -> None:
        """No unauthenticated round trip just to be told to authenticate."""
        fake = FakeRegistry()
        client = _client(fake)
        client.resolve(REGISTRY, REPO, "v1")
        before = fake.manifest_requests

        client.resolve(REGISTRY, REPO, "v1")

        assert fake.manifest_requests - before == 1
        assert fake.issued == 1

    def test_a_registry_that_keeps_refusing_fails_rather_than_looping(self) -> None:
        """Exactly one fresh exchange per call. A second 401 is a real authorization failure."""
        fake = FakeRegistry(accept_tokens=False)
        client = _client(fake)
        with pytest.raises(RegistryError, match="401"):
            client.resolve(REGISTRY, REPO, "v1")
        assert fake.issued == 1
        assert fake.manifest_requests == 2


class TestProtocol:
    def test_the_digest_comes_from_the_registrys_header(self) -> None:
        resolved = _client(FakeRegistry()).resolve(REGISTRY, REPO, "v1")
        assert resolved.digest == resolved.manifest_digest == DIGEST

    def test_a_missing_reference_is_not_found_rather_than_an_error(self) -> None:
        """The reconciler treats these differently: not-found may simply be early."""
        with pytest.raises(ReferenceNotFound):
            _client(FakeRegistry()).resolve(REGISTRY, REPO, "missing")

    def test_the_basic_credential_goes_to_the_token_endpoint(self) -> None:
        seen: list[str] = []
        fake = FakeRegistry()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/token":
                seen.append(request.headers.get("Authorization", ""))
            return fake.handler(request)

        client = RegistryClient(username="u", password="p", transport=httpx.MockTransport(handler))
        client.resolve(REGISTRY, REPO, "v1")
        assert seen == ["Basic " + base64.b64encode(b"u:p").decode()]
