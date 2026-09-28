# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The registry client, against a fake registry that speaks the OCI Distribution auth flow.

The minikube run could not exercise any of this: its registry is anonymous and plain HTTP, so
the challenge / token-endpoint path never executes there. A real registry -- GAR, measured --
issues short-lived bearer tokens, which is exactly where the adversarial review found the bug.
"""

from __future__ import annotations

import base64
import hashlib
import json

import httpx
import pytest
from nemo_builder_plugin.registry import (
    NoMatchingImage,
    ReferenceNotFound,
    RegistryClient,
    RegistryError,
    parse_challenges,
)

REGISTRY = "reg.example.com"
REPO = "team/app"
MANIFEST = json.dumps({"schemaVersion": 2}).encode()
#: A content address addresses content: the digest of the manifest these fakes serve.
DIGEST = "sha256:" + hashlib.sha256(MANIFEST).hexdigest()


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
            content=MANIFEST,
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


def _serving(body: bytes, headers: dict[str, str]) -> RegistryClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, content=body)

    return RegistryClient(transport=httpx.MockTransport(handler))


class TestProtocol:
    def test_the_digest_is_the_hash_of_the_bytes_served(self) -> None:
        resolved = _client(FakeRegistry()).resolve(REGISTRY, REPO, "v1")
        assert resolved.digest == resolved.manifest_digest == DIGEST

    def test_a_header_that_disagrees_with_the_bytes_is_refused(self) -> None:
        """Taken at its word, the header would let the body -- and the index child chosen from
        it -- be anything."""
        client = _serving(MANIFEST, {"Docker-Content-Digest": "sha256:" + "e" * 64})
        with pytest.raises(RegistryError, match="hashes to"):
            client.resolve(REGISTRY, REPO, "v1")

    def test_a_registry_that_sends_no_header_still_resolves(self) -> None:
        """The spec makes the header optional; the bytes are what count."""
        client = _serving(MANIFEST, {"Content-Type": "application/vnd.oci.image.manifest.v1+json"})
        assert client.resolve(REGISTRY, REPO, "v1").digest == DIGEST

    def test_a_digest_reference_served_as_something_else_is_refused(self) -> None:
        client = _serving(MANIFEST, {})
        with pytest.raises(RegistryError, match="was served as"):
            client.resolve(REGISTRY, REPO, "sha256:" + "0" * 64)

    def test_an_index_entry_without_a_digest_is_a_registry_error(self) -> None:
        body = json.dumps({"manifests": [{"digest": "sha256:x\nRUN evil"}]}).encode()
        client = _serving(body, {"Content-Type": "application/vnd.oci.image.index.v1+json"})
        with pytest.raises(RegistryError, match="without a sha256 digest"):
            client.resolve(REGISTRY, REPO, "v1")

    def test_docker_hub_is_asked_at_its_api_host(self) -> None:
        """`docker.io` answers `/v2/` with a redirect to its website."""
        hosts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            hosts.append(request.url.host)
            return httpx.Response(200, content=MANIFEST)

        RegistryClient(transport=httpx.MockTransport(handler)).resolve("docker.io", "library/alpine", "3.20")
        assert hosts == ["registry-1.docker.io"]


class TestEveryFailureIsARegistryError:
    """Anything else escapes the reconciler's handling and is retried forever, with a traceback."""

    def test_a_transport_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(RegistryError, match="ConnectError"):
            RegistryClient(transport=httpx.MockTransport(handler)).resolve(REGISTRY, REPO, "v1")

    def test_a_token_endpoint_that_does_not_answer_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/token":
                return httpx.Response(200, content=b"<html>")
            return httpx.Response(401, headers={"WWW-Authenticate": f'Bearer realm="https://{REGISTRY}/token"'})

        with pytest.raises(RegistryError, match="not JSON"):
            RegistryClient(transport=httpx.MockTransport(handler)).resolve(REGISTRY, REPO, "v1")


class TestRedirects:
    def test_a_redirect_off_https_is_refused(self) -> None:
        """The registry names the redirect; a metadata server answers plain HTTP."""
        reached: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "169.254.169.254":
                reached.append(str(request.url))
                return httpx.Response(200, content=MANIFEST)
            return httpx.Response(307, headers={"Location": "http://169.254.169.254/latest/meta-data/"})

        with pytest.raises(RegistryError, match="plain-HTTP"):
            RegistryClient(transport=httpx.MockTransport(handler)).resolve(REGISTRY, REPO, "v1")
        assert reached == []

    def test_an_https_redirect_is_followed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "cdn.example.com":
                return httpx.Response(200, content=MANIFEST)
            return httpx.Response(307, headers={"Location": "https://cdn.example.com/m"})

        assert RegistryClient(transport=httpx.MockTransport(handler)).resolve(REGISTRY, REPO, "v1").digest == DIGEST


class TestChallenges:
    def test_several_challenges_are_read_separately(self) -> None:
        header = 'Bearer realm="https://auth/token",service="r", Basic realm="Registry Realm"'
        assert parse_challenges(header) == [
            ("bearer", {"realm": "https://auth/token", "service": "r"}),
            ("basic", {"realm": "Registry Realm"}),
        ]

    def test_unquoted_values_and_any_case(self) -> None:
        assert parse_challenges("BEARER Realm=https://x/token,service=r") == [
            ("bearer", {"realm": "https://x/token", "service": "r"})
        ]

    def test_bearer_is_answered_when_both_are_offered(self) -> None:
        """A later Basic realm -- a label -- used to overwrite the Bearer realm, and was then
        fetched as a URL."""
        fake = FakeRegistry()

        def handler(request: httpx.Request) -> httpx.Response:
            response = fake.handler(request)
            if response.status_code == 401:
                response.headers["WWW-Authenticate"] += ', Basic realm="Registry Realm"'
            return response

        client = RegistryClient(username="u", password="p", transport=httpx.MockTransport(handler))
        assert client.resolve(REGISTRY, REPO, "v1").digest == DIGEST

    def test_a_token_endpoint_that_is_not_https_gets_no_credential(self) -> None:
        """The registry names the realm; the credential goes there."""
        sent: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "169.254.169.254":
                sent.append(request.headers.get("Authorization", ""))
                return httpx.Response(200, json={"token": "t"})
            return httpx.Response(401, headers={"WWW-Authenticate": 'Bearer realm="http://169.254.169.254/token"'})

        client = RegistryClient(username="u", password="p", transport=httpx.MockTransport(handler))
        with pytest.raises(RegistryError, match="not HTTPS"):
            client.resolve(REGISTRY, REPO, "v1")
        assert sent == []

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
            content=MANIFEST,
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
        """A `RegistryError` is what the reconciler handles -- the row waits, with the reason
        logged -- instead of an exception escaping the loop on every cycle."""
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


INDEX_TYPE = "application/vnd.oci.image.index.v1+json"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"
AMD64 = "sha256:" + "a" * 64
ARM64 = "sha256:" + "b" * 64


def _sha(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


INDEX_BODY = json.dumps(
    {
        "manifests": [
            {"digest": "sha256:" + "0" * 64, "platform": {"os": "unknown", "architecture": "unknown"}},
            {"digest": ARM64, "platform": {"os": "linux", "architecture": "arm64", "variant": "v8"}},
            {"digest": AMD64, "platform": {"os": "linux", "architecture": "amd64"}},
        ]
    }
).encode()
INDEX = _sha(INDEX_BODY)


def _index_registry(requests: list[httpx.Request], body: bytes = INDEX_BODY) -> httpx.MockTransport:
    """Serves one multi-platform index, anonymously, and records every request it gets."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"Docker-Content-Digest": _sha(body), "Content-Type": INDEX_TYPE}, content=body
        )

    return httpx.MockTransport(handler)


def _single_platform_registry(os_name: str, architecture: str) -> tuple[httpx.MockTransport, str]:
    """Serves one plain manifest, and the config blob that says what platform it is for."""
    config = json.dumps({"os": os_name, "architecture": architecture}).encode()
    manifest = json.dumps({"schemaVersion": 2, "config": {"digest": _sha(config)}, "layers": []}).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if "/blobs/" in request.url.path:
            return httpx.Response(200, content=config)
        return httpx.Response(200, headers={"Content-Type": MANIFEST_TYPE}, content=manifest)

    return httpx.MockTransport(handler), _sha(manifest)


class TestPlatforms:
    def test_an_index_resolves_to_the_requested_platforms_child(self) -> None:
        client = RegistryClient(transport=_index_registry([]))
        resolved = client.resolve(REGISTRY, REPO, INDEX, platform="linux/amd64")
        assert (resolved.digest, resolved.manifest_digest) == (INDEX, AMD64)

    def test_a_variant_narrows_the_match(self) -> None:
        client = RegistryClient(transport=_index_registry([]))
        assert client.resolve(REGISTRY, REPO, INDEX, platform="linux/arm64/v8").manifest_digest == ARM64

    def test_no_matching_child_is_an_error_not_the_first_entry(self) -> None:
        """The first entry here is an attestation manifest; recording it would be recording nonsense."""
        client = RegistryClient(transport=_index_registry([]))
        with pytest.raises(RegistryError, match="no linux/s390x manifest"):
            client.resolve(REGISTRY, REPO, INDEX, platform="linux/s390x")

    def test_a_digest_served_as_another_digest_is_refused(self) -> None:
        client = RegistryClient(transport=_index_registry([]))
        with pytest.raises(RegistryError, match="was served as"):
            client.resolve(REGISTRY, REPO, AMD64, platform="linux/amd64")

    def test_arm64_without_a_variant_is_arm64_v8(self) -> None:
        """buildx lists arm64 with no variant; a caller writing `linux/arm64/v8` means that image."""
        body = json.dumps({"manifests": [{"digest": ARM64, "platform": {"os": "linux", "architecture": "arm64"}}]})
        client = RegistryClient(transport=_index_registry([], body.encode()))
        digest = _sha(body.encode())
        assert client.resolve(REGISTRY, REPO, digest, platform="linux/arm64/v8").manifest_digest == ARM64

    def test_a_missing_platform_is_not_found_rather_than_an_error(self) -> None:
        """The caller's reference, not the registry, is what is wrong: a 400, not a 502."""
        client = RegistryClient(transport=_index_registry([]))
        with pytest.raises(NoMatchingImage):
            client.resolve(REGISTRY, REPO, INDEX, platform="linux/s390x")
        assert issubclass(NoMatchingImage, ReferenceNotFound)

    def test_docker_hub_is_asked_at_its_api_host(self) -> None:
        """`https://docker.io/v2/` redirects to the marketing site, measured."""
        requests: list[httpx.Request] = []
        RegistryClient(transport=_index_registry(requests)).resolve(
            "docker.io", "harborframework/terminal-bench", INDEX, platform="linux/amd64"
        )
        assert requests[0].url.host == "registry-1.docker.io"
        assert requests[0].url.path == f"/v2/harborframework/terminal-bench/manifests/{INDEX}"


class TestImports:
    """`resolve_import`: a caller's pinned reference to a registry this deployment does not run."""

    def test_a_single_platform_image_for_the_platform_is_accepted(self) -> None:
        transport, digest = _single_platform_registry("linux", "amd64")
        resolved = RegistryClient(transport=transport).resolve_import(REGISTRY, REPO, digest, platform="linux/amd64")
        assert resolved.manifest_digest == digest

    def test_a_single_platform_image_for_another_platform_is_refused(self) -> None:
        """Only an index lists platforms. An arm64-only image imported as amd64 would otherwise be
        published, signed and recorded as amd64 -- and fail where it runs."""
        transport, digest = _single_platform_registry("linux", "arm64")
        with pytest.raises(NoMatchingImage, match="single linux/arm64 image"):
            RegistryClient(transport=transport).resolve_import(REGISTRY, REPO, digest, platform="linux/amd64")

    def test_refused_anonymous_access_is_not_found(self) -> None:
        """Docker Hub answers a missing or private repository with 401, not 404."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/token":
                return httpx.Response(200, json={"token": "anonymous"})
            return httpx.Response(401, headers={"WWW-Authenticate": f'Bearer realm="https://{REGISTRY}/token"'})

        with pytest.raises(ReferenceNotFound, match="not public"):
            RegistryClient(transport=httpx.MockTransport(handler)).resolve_import(
                REGISTRY, REPO, INDEX, platform="linux/amd64"
            )
