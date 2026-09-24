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


class BasicRegistry:
    """`distribution` with htpasswd: challenges with Basic, and its realm is a label, not a URL."""

    def __init__(self, *, username: str = "robot", password: str = "s3cret") -> None:
        self.expected = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        #: The `Authorization` header each request carried, in order.
        self.requests: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization", "")
        self.requests.append(auth)
        if auth != self.expected:
            return httpx.Response(401, headers={"WWW-Authenticate": 'Basic realm="Registry Realm"'})
        return httpx.Response(
            200,
            headers={"Docker-Content-Digest": DIGEST, "Content-Type": "application/vnd.oci.image.manifest.v1+json"},
            content=json.dumps({"schemaVersion": 2}).encode(),
        )


def _basic_client(
    fake: BasicRegistry, *, username: str | None = "robot", password: str | None = "s3cret"
) -> RegistryClient:
    return RegistryClient(username=username, password=password, transport=httpx.MockTransport(fake.handler))


class TestBasicChallenge:
    def test_it_is_answered_with_the_credential(self) -> None:
        """The regression: the realm was read as a token URL, and the httpx error that produced
        escaped the reconcile loop, so every row on such a registry sat `pending` forever."""
        fake = BasicRegistry()
        assert _basic_client(fake).resolve(REGISTRY, REPO, "v1").digest == DIGEST
        assert fake.requests == ["", fake.expected], "the credential goes only to a registry that asked"

    def test_the_answer_is_sent_first_next_time(self) -> None:
        fake = BasicRegistry()
        client = _basic_client(fake)
        client.resolve(REGISTRY, REPO, "v1")
        client.resolve(REGISTRY, REPO, "v1")
        assert fake.requests == ["", fake.expected, fake.expected]

    def test_with_no_credential_it_is_a_registry_error(self) -> None:
        """A `RegistryError` is what the reconciler counts against the attempt budget, so the row
        fails with a reason instead of an exception escaping the loop on every cycle."""
        with pytest.raises(RegistryError, match="401"):
            _basic_client(BasicRegistry(), username=None, password=None).resolve(REGISTRY, REPO, "v1")

    def test_a_wrong_credential_is_refused_once_not_retried(self) -> None:
        fake = BasicRegistry(password="right")
        with pytest.raises(RegistryError, match="401"):
            _basic_client(fake, password="wrong").resolve(REGISTRY, REPO, "v1")
        assert len(fake.requests) == 2

    def test_the_scheme_is_matched_case_insensitively(self) -> None:
        fake = BasicRegistry()

        def handler(request: httpx.Request) -> httpx.Response:
            response = fake.handler(request)
            if response.status_code == 401:
                response.headers["WWW-Authenticate"] = 'BASIC realm="Registry Realm"'
            return response

        client = RegistryClient(username="robot", password="s3cret", transport=httpx.MockTransport(handler))
        assert client.resolve(REGISTRY, REPO, "v1").digest == DIGEST
