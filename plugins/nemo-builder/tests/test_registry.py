# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The registry calls the push step makes itself: reading a manifest, and writing blobs and manifests."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest
from nemo_builder_plugin.registry import (
    ATTEMPTS,
    OCI_MANIFEST,
    RETRY_STATUSES,
    ManifestNotFound,
    Registry,
    RegistryError,
    registry_client,
)

HOST = "reg.example.com"
AUTHORIZATION = "Bearer registry-token"
REGISTRY_TIMEOUT = 5


def _answering(
    *responses: httpx.Response | Exception, sleeps: list[float] | None = None
) -> tuple[Registry, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response = responses[len(seen) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    http = httpx.Client(transport=httpx.MockTransport(handle))
    return Registry(HOST, AUTHORIZATION, http=http, sleep=(sleeps if sleeps is not None else []).append), seen


class TestManifest:
    def test_its_digest_is_computed_from_the_bytes_returned(self, oci_registry) -> None:
        digest = oci_registry.push("ws/app", "v1", {"schemaVersion": 2, "layers": []})
        found, manifest = Registry(HOST, AUTHORIZATION, http=oci_registry.client()).manifest("ws/app", "v1")
        assert (found, manifest) == (digest, {"schemaVersion": 2, "layers": []})
        sent = oci_registry.requests[0]
        assert sent.headers["Authorization"] == AUTHORIZATION and OCI_MANIFEST in sent.headers["Accept"]

    def test_nothing_at_the_tag_is_its_own_error(self, oci_registry) -> None:
        with pytest.raises(ManifestNotFound):
            Registry(HOST, AUTHORIZATION, http=oci_registry.client()).manifest("ws/app", "v1")

    def test_a_plain_http_registry_is_reached_over_http(self, oci_registry) -> None:
        oci_registry.push("ws/app", "v1", {})
        Registry(HOST, AUTHORIZATION, http=oci_registry.client(), plain_http=True).manifest("ws/app", "v1")
        assert oci_registry.requests[0].url.scheme == "http"


class TestWrites:
    def test_a_blob_is_uploaded_where_the_registry_says_with_its_digest(self, oci_registry) -> None:
        digest = Registry(HOST, AUTHORIZATION, http=oci_registry.client()).put_blob("ws/app", b"payload")
        assert digest == "sha256:" + hashlib.sha256(b"payload").hexdigest()
        assert oci_registry.blobs[("ws/app", digest)] == b"payload"

    def test_a_manifest_is_written_under_its_tag(self, oci_registry) -> None:
        registry = Registry(HOST, AUTHORIZATION, http=oci_registry.client())
        config = registry.put_blob("ws/app", b"{}")
        body = json.dumps({"schemaVersion": 2, "config": {"digest": config}, "layers": []}).encode()
        registry.put_manifest("ws/app", "v1", body, OCI_MANIFEST)
        assert oci_registry.manifests[("ws/app", "v1")] == (body, OCI_MANIFEST)


class TestRefusals:
    def test_an_upload_location_on_another_host_is_refused_before_the_credential_goes_there(self) -> None:
        registry, seen = _answering(httpx.Response(202, headers={"Location": "https://elsewhere.example.com/upload"}))
        with pytest.raises(RegistryError, match="another host"):
            registry.put_blob("ws/app", b"payload")
        assert len(seen) == 1

    def test_a_refusal_names_the_status_but_not_the_body(self) -> None:
        registry, seen = _answering(httpx.Response(403, text="internal details"))
        with pytest.raises(RegistryError, match="answered 403") as raised:
            registry.manifest("ws/app", "v1")
        assert "internal" not in str(raised.value)
        assert len(seen) == 1

    def test_a_manifest_that_is_not_json_is_refused(self) -> None:
        registry, _ = _answering(httpx.Response(200, content=b"<html>"))
        with pytest.raises(RegistryError, match="not a JSON manifest"):
            registry.manifest("ws/app", "v1")

    def test_an_unreachable_registry_is_a_registry_error(self) -> None:
        registry, seen = _answering(*[httpx.ConnectError("refused")] * ATTEMPTS)
        with pytest.raises(RegistryError, match="could not be reached"):
            registry.manifest("ws/app", "v1")
        assert len(seen) == ATTEMPTS


class TestRetries:
    """As crane retries the push: three tries, a second and then three apart."""

    @pytest.mark.parametrize("status", sorted(RETRY_STATUSES))
    def test_a_transient_status_is_tried_again(self, status: int) -> None:
        sleeps: list[float] = []
        registry, seen = _answering(httpx.Response(status), httpx.Response(201), sleeps=sleeps)
        registry.put_manifest("ws/app", "v1", b"{}", OCI_MANIFEST)
        assert len(seen) == 2 and sleeps == [1.0]
        assert seen[1].content == b"{}"

    @pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
    def test_a_dropped_connection_or_a_timeout_is_tried_again(self, error: Exception) -> None:
        sleeps: list[float] = []
        registry, _ = _answering(error, error, httpx.Response(201), sleeps=sleeps)
        registry.put_manifest("ws/app", "v1", b"{}", OCI_MANIFEST)
        assert sleeps == [1.0, 3.0]

    def test_the_last_answer_stands_after_three_tries(self) -> None:
        sleeps: list[float] = []
        registry, seen = _answering(*[httpx.Response(503)] * ATTEMPTS, sleeps=sleeps)
        with pytest.raises(RegistryError, match="answered 503"):
            registry.manifest("ws/app", "v1")
        assert len(seen) == ATTEMPTS and sleeps == [1.0, 3.0]

    def test_an_error_that_will_not_pass_is_not_tried_again(self) -> None:
        registry, seen = _answering(httpx.UnsupportedProtocol("ftp"))
        with pytest.raises(RegistryError, match="could not be reached"):
            registry.manifest("ws/app", "v1")
        assert len(seen) == 1


class TestTheClient:
    def test_a_cookie_the_registry_sets_is_never_sent_back(self) -> None:
        # Harbor sets a session cookie, then refuses every write that carries it back, for want of a CSRF token.
        client = registry_client(REGISTRY_TIMEOUT)
        request = httpx.Request("GET", f"https://{HOST}/v2/ws/app/manifests/v1")
        client.cookies.extract_cookies(httpx.Response(200, headers={"Set-Cookie": "sid=0123; Path=/"}, request=request))
        assert not client.cookies

    def test_it_follows_no_redirect(self) -> None:
        assert not registry_client(REGISTRY_TIMEOUT).follow_redirects
