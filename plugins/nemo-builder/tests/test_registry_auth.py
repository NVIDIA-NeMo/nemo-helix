# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The broker as a client of a registry's token service (the Distribution token specification).

A fake registry and token service answer through ``httpx.MockTransport``, so each test states what
the registry said and asserts what the broker asked for -- above all, that it asked for exactly the
repositories it was told to, and that it notices when it got more.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx
import pytest
from nemo_builder_plugin.registry_auth import (
    DistributionTokenClient,
    FileCredential,
    RegistryAuthError,
    ScopeWidened,
    narrowed,
    parse_challenges,
)

REGISTRY = "reg.example.com"
REALM = "https://auth.example.com/token"


def _jwt(access: list[dict[str, object]], **claims: object) -> str:
    def part(document: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(document).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'ES256'})}.{part({'access': access, **claims})}.c2lnbmF0dXJl"


class Credential:
    """A credential source that counts its reads: the broker must read it on every use."""

    def __init__(self) -> None:
        self.reads = 0

    def read(self) -> tuple[str, str]:
        self.reads += 1
        return "nhx-build-broker", f"secret-{self.reads}"


class Registry:
    """A registry that challenges, and a token service that grants what ``grant`` says."""

    def __init__(self, *, challenge: str | None = None, token: str | None = None, status: int = 200) -> None:
        self.challenge = challenge if challenge is not None else f'Bearer realm="{REALM}",service="{REGISTRY}"'
        self.token = token
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v2/":
            return httpx.Response(401, headers={"WWW-Authenticate": self.challenge})
        scopes = request.url.params.get_list("scope")
        token = self.token or _jwt(
            [{"type": "repository", "name": s.split(":")[1], "actions": s.split(":")[2].split(",")} for s in scopes]
        )
        return httpx.Response(self.status, json={"token": token, "expires_in": 300})


def _client(registry: Registry, credential: Credential | None = None, **kwargs: bool) -> DistributionTokenClient:
    return DistributionTokenClient(
        REGISTRY,
        credential or Credential(),
        http=httpx.Client(transport=httpx.MockTransport(registry)),
        **kwargs,
    )


class TestAskingForExactlyTheJobsDestinations:
    def test_one_scope_per_repository_at_the_realm_the_registry_named(self) -> None:
        registry = Registry()
        issued = _client(registry).issue(["ws-a/app", "ws-a/tools"], ["pull", "push"])
        token_request = registry.requests[-1]
        assert str(token_request.url).startswith(REALM)
        assert token_request.url.params["service"] == REGISTRY
        assert token_request.url.params.get_list("scope") == [
            "repository:ws-a/app:pull,push",
            "repository:ws-a/tools:pull,push",
        ]
        assert issued.expires_in == 300 and issued.narrowed is True

    def test_the_brokers_credential_is_presented_as_basic_and_read_fresh_each_time(self) -> None:
        """A projected token rotates, so it is never cached."""
        registry, credential = Registry(), Credential()
        client = _client(registry, credential)
        client.issue(["ws-a/app"], ["pull"])
        client.issue(["ws-a/app"], ["pull"])
        assert credential.reads == 2
        expected = base64.b64encode(b"nhx-build-broker:secret-2").decode()
        assert registry.requests[-1].headers["Authorization"] == f"Basic {expected}"

    def test_the_challenge_is_read_once(self) -> None:
        registry = Registry()
        client = _client(registry)
        client.issue(["a"], ["pull"])
        client.issue(["b"], ["pull"])
        assert [r.url.path for r in registry.requests].count("/v2/") == 1

    def test_nothing_is_issued_for_nothing(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _client(Registry()).issue([], ["pull"])


class TestWhatTheRegistryGranted:
    def test_a_token_wider_than_asked_is_reported(self) -> None:
        wide = _jwt([{"type": "repository", "name": "ws-b/app", "actions": ["pull", "push"]}])
        assert _client(Registry(token=wide)).issue(["ws-a/app"], ["pull", "push"]).narrowed is False

    def test_and_refused_where_the_registry_is_said_to_narrow(self) -> None:
        wide = _jwt([{"type": "repository", "name": "ws-a/app", "actions": ["pull", "push", "delete"]}])
        with pytest.raises(ScopeWidened):
            _client(Registry(token=wide), require_narrowing=True).issue(["ws-a/app"], ["pull", "push"])

    def test_an_opaque_token_says_nothing_either_way(self) -> None:
        """GAR's tokens are opaque, so nothing can be read from them."""
        issued = _client(Registry(token="ya29.opaque"), require_narrowing=True).issue(["ws-a/app"], ["pull"])
        assert issued.narrowed is None

    def test_a_catalog_grant_is_wider(self) -> None:
        assert narrowed(_jwt([{"type": "registry", "name": "catalog", "actions": ["*"]}]), ["a"], ["pull"]) is False

    def test_access_token_is_accepted_for_token_and_a_missing_lifetime_is_the_specs_default(self) -> None:
        def service(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/":
                return httpx.Response(401, headers={"WWW-Authenticate": f'Bearer realm="{REALM}"'})
            return httpx.Response(200, json={"access_token": "opaque"})

        client = DistributionTokenClient(
            REGISTRY, Credential(), http=httpx.Client(transport=httpx.MockTransport(service))
        )
        issued = client.issue(["a"], ["pull"])
        assert (issued.token, issued.expires_in) == ("opaque", 60)

    @pytest.mark.parametrize("stated", [None, 60])
    def test_a_readable_token_reports_the_lifetime_it_actually_has(self, stated: int | None) -> None:
        """What the registry enforces is the token's `exp`, whatever the reply around it said."""
        token = _jwt([{"type": "repository", "name": "a", "actions": ["pull"]}], exp=int(time.time()) + 3600)

        def service(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/":
                return httpx.Response(401, headers={"WWW-Authenticate": f'Bearer realm="{REALM}"'})
            return httpx.Response(200, json={"token": token} | ({"expires_in": stated} if stated else {}))

        client = DistributionTokenClient(
            REGISTRY, Credential(), http=httpx.Client(transport=httpx.MockTransport(service))
        )
        assert 3590 <= client.issue(["a"], ["pull"]).expires_in <= 3600


class TestRefusals:
    def test_a_registry_that_does_not_challenge_with_bearer_is_refused(self) -> None:
        """A Basic registry cannot narrow, and handing out the broker's own password is not an option."""
        with pytest.raises(RegistryAuthError, match="not Bearer"):
            _client(Registry(challenge='Basic realm="Registry Realm"')).issue(["a"], ["pull"])

    def test_a_realm_that_is_not_https_is_refused_before_the_credential_goes_to_it(self) -> None:
        registry = Registry(challenge='Bearer realm="http://169.254.169.254/token",service="x"')
        with pytest.raises(RegistryAuthError, match="not HTTPS"):
            _client(registry).issue(["a"], ["pull"])
        assert [r.url.path for r in registry.requests] == ["/v2/"]

    def test_plain_http_is_accepted_only_for_a_plain_http_registry(self) -> None:
        registry = Registry(challenge='Bearer realm="http://auth.local/token",service="x"')
        assert _client(registry, plain_http=True).issue(["a"], ["pull"]).token

    def test_a_token_service_refusal_is_an_error_that_does_not_echo_the_body(self) -> None:
        with pytest.raises(RegistryAuthError, match="refused the broker: 401") as caught:
            _client(Registry(status=401)).issue(["a"], ["pull"])
        assert "token" not in str(caught.value).split(":")[-1]

    def test_an_unreachable_registry_is_an_error(self) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = DistributionTokenClient(REGISTRY, Credential(), http=httpx.Client(transport=httpx.MockTransport(down)))
        with pytest.raises(RegistryAuthError, match="could not be reached"):
            client.issue(["a"], ["pull"])


class TestFileCredential:
    def test_the_file_is_read_on_every_use(self, tmp_path: Path) -> None:
        secret = tmp_path / "token"
        secret.write_text("one\n")
        credential = FileCredential("broker", secret)
        assert credential.read() == ("broker", "one")
        secret.write_text("two")
        assert credential.read() == ("broker", "two")

    def test_a_missing_or_empty_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(RegistryAuthError, match="cannot read"):
            FileCredential("broker", tmp_path / "absent").read()
        (tmp_path / "empty").write_text("")
        with pytest.raises(RegistryAuthError, match="empty"):
            FileCredential("broker", tmp_path / "empty").read()


class TestChallenges:
    def test_bearer_parameters_are_read_per_challenge(self) -> None:
        header = 'Bearer realm="https://a/token",service="reg", Basic realm="Registry Realm"'
        assert dict(parse_challenges(header)) == {
            "bearer": {"realm": "https://a/token", "service": "reg"},
            "basic": {"realm": "Registry Realm"},
        }

    def test_quoted_values_unescape(self) -> None:
        assert parse_challenges('Bearer realm="a\\"b"') == [("bearer", {"realm": 'a"b'})]
