# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The local stand-in for a registry's token service -- and the broker's client speaking to it.

The stand-in is local-only, but what it stands in for is not: the last class here runs the
broker's Distribution token client against it, which is the contract every real token service is
held to -- a token scoped to the intersection of what was asked and what the caller holds.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from nemo_builder_plugin.registry_auth import DistributionTokenClient, FileCredential
from nemo_builder_plugin.run.token_server import (
    Grant,
    Scope,
    TokenServer,
    TokenServerConfig,
    grant,
    jwks,
    key_id,
    load_key,
    parse_scope,
)

KEY = ec.generate_private_key(ec.SECP256R1())
REGISTRY = "registry.nhx-registry.svc.cluster.local:5000"
BROKER = "system:serviceaccount:nhx-build-broker:nhx-build-broker"
CONFIG = TokenServerConfig(
    service=REGISTRY,
    key_file="/unused",
    grants=[Grant(subject=BROKER, actions=["pull", "push"])],
)


def _claims(token: str) -> dict[str, object]:
    header, claims, signature = token.split(".")
    raw = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
    der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    KEY.public_key().verify(der, f"{header}.{claims}".encode(), ec.ECDSA(hashes.SHA256()))
    assert json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))["kid"] == key_id(KEY.public_key())
    return json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))


def _server(reviews: dict[str, str] | None = None) -> TokenServer:
    known = reviews if reviews is not None else {"broker-sa-token": BROKER}

    def review(token: str, audience: str) -> str | None:
        assert audience == "nhx-local-token-server"
        return known.get(token)

    return TokenServer(CONFIG, KEY, review=review, clock=lambda: 1_000_000)


def _basic(token: str) -> str:
    return "Basic " + base64.b64encode(f"nhx-build-broker:{token}".encode()).decode()


class TestGrant:
    def test_the_intersection_of_what_was_asked_and_what_is_allowed(self) -> None:
        entry = [Grant(subject="a", repositories="proj", actions=["pull"])]
        requested = [Scope("proj/ws/app", ("pull", "push")), Scope("other/app", ("pull",))]
        assert grant("a", requested, entry) == [Scope("proj/ws/app", ("pull",))]

    def test_an_unknown_subject_is_granted_nothing(self) -> None:
        assert grant("b", [Scope("x", ("pull",))], [Grant(subject="a")]) == []

    def test_delete_is_never_granted(self) -> None:
        entry = [Grant(subject="a", actions=["pull", "push", "delete", "*"])]
        assert grant("a", [Scope("x", ("delete", "*", "push"))], entry) == [Scope("x", ("push",))]

    @pytest.mark.parametrize(
        ("value", "scope"),
        [
            ("repository:ws/app:pull,push", Scope("ws/app", ("pull", "push"))),
            ("repository:host:5000/app:pull", Scope("host:5000/app", ("pull",))),
            ("registry:catalog:*", None),
            ("repository:ws/app:", None),
            ("nonsense", None),
        ],
    )
    def test_scopes_are_parsed_or_ignored(self, value: str, scope: Scope | None) -> None:
        assert parse_scope(value) == scope


class TestIssue:
    def test_the_token_carries_exactly_the_grant_and_verifies_under_the_jwks_key(self) -> None:
        reply = _server().issue(
            authorization=_basic("broker-sa-token"),
            service=REGISTRY,
            scopes=["repository:ws/app:pull,push", "registry:catalog:*"],
        )
        assert reply.status == 200 and reply.body["expires_in"] == 900
        claims = _claims(str(reply.body["token"]))
        assert claims["access"] == [{"type": "repository", "name": "ws/app", "actions": ["pull", "push"]}]
        assert (claims["aud"], claims["sub"], claims["exp"]) == (REGISTRY, BROKER, 1_000_000 + 900)

    def test_a_caller_without_a_grant_gets_a_token_for_nothing(self) -> None:
        """A push step presenting its own token here, bypassing the broker."""
        reply = _server({"push-sa-token": "system:serviceaccount:nhx-builds:nhx-build-push"}).issue(
            authorization=_basic("push-sa-token"), service=REGISTRY, scopes=["repository:ws/app:pull,push"]
        )
        assert reply.status == 200 and _claims(str(reply.body["token"]))["access"] == []

    def test_an_unknown_token_is_a_401(self) -> None:
        assert _server().issue(authorization=_basic("forged"), service=REGISTRY, scopes=[]).status == 401
        assert _server().issue(authorization=None, service=REGISTRY, scopes=[]).status == 401

    def test_another_service_is_refused(self) -> None:
        assert _server().issue(authorization=_basic("broker-sa-token"), service="elsewhere", scopes=[]).status == 400

    def test_a_review_that_fails_is_a_503(self) -> None:
        def broken(token: str, audience: str) -> str | None:
            raise RuntimeError("api server down")

        server = TokenServer(CONFIG, KEY, review=broken)
        assert server.issue(authorization=_basic("t"), service=REGISTRY, scopes=[]).status == 503


class TestJwks:
    def test_the_jwks_names_the_key_by_its_own_thumbprint(self) -> None:
        document = jwks(KEY.public_key())
        assert document["keys"][0]["kid"] == key_id(KEY.public_key())
        assert document["keys"][0]["alg"] == "ES256"

    def test_only_a_p256_key_is_loaded(self) -> None:
        pem = KEY.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        assert isinstance(load_key(pem), ec.EllipticCurvePrivateKey)
        other = ec.generate_private_key(ec.SECP384R1()).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        with pytest.raises(ValueError, match="P-256"):
            load_key(other)


class TestTheBrokersClientAgainstIt:
    """The broker's token-spec client and a spec-following token service, wired together."""

    def test_the_broker_gets_a_token_narrowed_to_its_jobs_destinations(self, tmp_path: Path) -> None:
        server = _server()

        def registry_and_realm(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/":
                challenge = f'Bearer realm="http://token-server/token",service="{REGISTRY}"'
                return httpx.Response(401, headers={"WWW-Authenticate": challenge})
            query = parse_qs(urlsplit(str(request.url)).query)
            reply = server.issue(
                authorization=request.headers.get("Authorization"),
                service=query["service"][0],
                scopes=query.get("scope", []),
            )
            return httpx.Response(reply.status, json=reply.body)

        (tmp_path / "token").write_text("broker-sa-token")
        client = DistributionTokenClient(
            REGISTRY,
            FileCredential("nhx-build-broker", tmp_path / "token"),
            plain_http=True,
            require_narrowing=True,
            http=httpx.Client(transport=httpx.MockTransport(registry_and_realm)),
        )
        issued = client.issue(["ws-a/app", "ws-a/tools"], ["pull", "push"])
        assert issued.narrowed is True
        assert [entry["name"] for entry in _claims(issued.token)["access"]] == ["ws-a/app", "ws-a/tools"]  # ty: ignore[not-iterable]
