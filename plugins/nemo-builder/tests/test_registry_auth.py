# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging in to a registry with the workspace's credential, as its challenge asks: Basic, or a Bearer token."""

from __future__ import annotations

import base64

import httpx
import pytest
from nemo_builder_plugin.registry_auth import RegistryAuthError, login, parse_challenges

REGISTRY = "reg.example.com"
REALM = "https://auth.example.com/token"
BASIC = "Basic " + base64.b64encode(b"builder:secret").decode()


class Registry:
    """A registry's `/v2/` challenge, and its token service."""

    def __init__(self, *, status: int = 401, challenge: str | None = None, token_status: int = 200) -> None:
        self.status = status
        self.challenge = challenge if challenge is not None else f'Bearer realm="{REALM}",service="{REGISTRY}"'
        self.token_status = token_status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/v2/":
            return httpx.Response(self.status, headers={"WWW-Authenticate": self.challenge} if self.challenge else {})
        return httpx.Response(self.token_status, json={"token": "registry-token", "expires_in": 300})


def _login(registry: Registry, *, plain_http: bool = False) -> str:
    http = httpx.Client(transport=httpx.MockTransport(registry))
    return login(REGISTRY, "builder", "secret", "ws/app", http=http, plain_http=plain_http)


class TestTheChallengeDecides:
    def test_a_bearer_challenge_gets_a_token_for_the_repository_from_the_realm_it_names(self) -> None:
        registry = Registry()
        assert _login(registry) == "Bearer registry-token"
        asked = registry.requests[1]
        assert str(asked.url.copy_with(query=None)) == REALM
        assert asked.url.params["service"] == REGISTRY
        assert asked.url.params["scope"] == "repository:ws/app:pull,push"
        assert asked.headers["Authorization"] == BASIC

    def test_a_basic_challenge_gets_the_credential_itself(self) -> None:
        registry = Registry(challenge='Basic realm="Registry Realm"')
        assert _login(registry) == BASIC
        assert len(registry.requests) == 1

    def test_a_registry_that_asks_for_nothing_still_gets_the_credential(self) -> None:
        assert _login(Registry(status=200, challenge="")) == BASIC

    def test_access_token_is_read_where_token_is_not(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v2/":
                return httpx.Response(401, headers={"WWW-Authenticate": f'Bearer realm="{REALM}"'})
            return httpx.Response(200, json={"access_token": "from-oauth"})

        http = httpx.Client(transport=httpx.MockTransport(handle))
        assert login(REGISTRY, "builder", "secret", "ws/app", http=http) == "Bearer from-oauth"


class TestRefusals:
    def test_a_realm_that_is_not_https_is_refused_before_the_credential_goes_to_it(self) -> None:
        registry = Registry(challenge='Bearer realm="http://169.254.169.254/token",service="x"')
        with pytest.raises(RegistryAuthError, match="not HTTPS"):
            _login(registry)
        assert [r.url.path for r in registry.requests] == ["/v2/"]

    def test_plain_http_is_accepted_only_for_a_plain_http_registry(self) -> None:
        registry = Registry(challenge='Bearer realm="http://auth.local/token",service="x"')
        assert _login(registry, plain_http=True) == "Bearer registry-token"

    def test_a_token_service_refusal_is_an_error_that_does_not_echo_the_body(self) -> None:
        with pytest.raises(RegistryAuthError, match="refused the credential: 401") as caught:
            _login(Registry(token_status=401))
        assert "registry-token" not in str(caught.value)

    def test_an_unavailable_registry_says_so(self) -> None:
        with pytest.raises(RegistryAuthError, match="answered /v2/ with 503"):
            _login(Registry(status=503, challenge=""))

    def test_an_unreachable_registry_is_an_error(self) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        http = httpx.Client(transport=httpx.MockTransport(down))
        with pytest.raises(RegistryAuthError, match="could not be reached"):
            login(REGISTRY, "builder", "secret", "ws/app", http=http)


class TestChallenges:
    def test_bearer_parameters_are_read_per_challenge(self) -> None:
        header = 'Bearer realm="https://a/token",service="reg", Basic realm="Registry Realm"'
        assert dict(parse_challenges(header)) == {
            "bearer": {"realm": "https://a/token", "service": "reg"},
            "basic": {"realm": "Registry Realm"},
        }

    def test_quoted_values_unescape(self) -> None:
        assert parse_challenges('Bearer realm="a\\"b"') == [("bearer", {"realm": 'a"b'})]
