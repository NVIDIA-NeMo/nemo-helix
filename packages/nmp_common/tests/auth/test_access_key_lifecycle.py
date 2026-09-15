# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for auth-service Scoped Access Key lifecycle validation."""

from unittest.mock import patch

import httpx
import pytest
from nmp.common.auth.access_key_lifecycle import (
    AccessKeyLifecycleAuthenticator,
    AccessKeyLifecycleUnavailableError,
)
from nmp.common.auth.token_resolver import ResolvedBearerToken
from nmp.common.config import AuthConfig, Configuration, PlatformConfig


def _config() -> AuthConfig:
    return AuthConfig(policy_decision_point_request_timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_authenticator_uses_sdk_routing_and_returns_trusted_claims() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "principal": "alice@example.com",
                "email": "alice@example.com",
                "groups": ["team-ml"],
                "scopes": ["models:read"],
                "jti": "ak_example",
                "token_kind": "access_key",
            },
        )

    platform_config = PlatformConfig(
        base_url="https://platform.example.com",
        service_discovery={"auth": "https://auth.internal:8080"},
        services="",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with patch.object(Configuration, "get_platform_config", return_value=platform_config):
            result = await authenticator.authenticate("scoped-access-key")

    assert isinstance(result, ResolvedBearerToken)
    assert result.claims.subject == "alice@example.com"
    assert result.claims.groups == ["team-ml"]
    assert result.claims.scopes == ["models:read"]
    assert requests[0].url == httpx.URL("https://auth.internal:8080/apis/auth/authenticate")


@pytest.mark.asyncio
async def test_authenticator_allows_loopback_http_auth_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "principal": "alice@example.com",
                "groups": [],
                "scopes": [],
                "jti": "ak_example",
                "token_kind": "access_key",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with patch.object(
            Configuration,
            "get_platform_config",
            return_value=PlatformConfig(base_url="http://127.0.0.1:8080", services=""),
        ):
            result = await authenticator.authenticate("candidate-token")

    assert isinstance(result, ResolvedBearerToken)
    assert requests[0].url == httpx.URL("http://127.0.0.1:8080/apis/auth/authenticate")


@pytest.mark.asyncio
async def test_authenticator_rejects_remote_http_auth_endpoint_before_sending_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with (
            patch.object(
                Configuration,
                "get_platform_config",
                return_value=PlatformConfig(
                    base_url="https://platform.example.com",
                    service_discovery={"auth": "http://auth.internal:8080"},
                    services="",
                ),
            ),
            pytest.raises(AccessKeyLifecycleUnavailableError) as exc_info,
        ):
            await authenticator.authenticate("candidate-token")

    assert exc_info.value.status_code == 503
    assert requests == []


@pytest.mark.asyncio
async def test_authenticator_rejects_redirect_following_injected_http_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with (
            patch.object(
                Configuration,
                "get_platform_config",
                return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
            ),
            pytest.raises(AccessKeyLifecycleUnavailableError) as exc_info,
        ):
            await authenticator.authenticate("candidate-token")

    assert exc_info.value.status_code == 503
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("token_kind", ["oidc_access_token", "workload_access_token", "workload_subject_token"])
async def test_authenticator_rejects_successful_non_access_key_response(token_kind: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "principal": "alice@example.com",
                "groups": [],
                "scopes": [],
                "token_kind": token_kind,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with patch.object(
            Configuration,
            "get_platform_config",
            return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
        ):
            assert await authenticator.authenticate("candidate-token") is None


@pytest.mark.asyncio
async def test_authenticator_rejects_malformed_sdk_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "principal": 123,
                "groups": [],
                "scopes": [],
                "jti": "ak_example",
                "token_kind": "access_key",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with (
            patch.object(
                Configuration,
                "get_platform_config",
                return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
            ),
            pytest.raises(AccessKeyLifecycleUnavailableError) as exc_info,
        ):
            await authenticator.authenticate("candidate-token")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_authenticator_returns_none_when_auth_service_rejects_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request, json={"detail": "Invalid bearer token"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with patch.object(
            Configuration,
            "get_platform_config",
            return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
        ):
            assert await authenticator.authenticate("rejected-token") is None


@pytest.mark.asyncio
async def test_authenticator_rejects_access_key_response_without_jti() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "principal": "alice@example.com",
                "groups": [],
                "scopes": [],
                "token_kind": "access_key",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with (
            patch.object(
                Configuration,
                "get_platform_config",
                return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
            ),
            pytest.raises(AccessKeyLifecycleUnavailableError) as exc_info,
        ):
            await authenticator.authenticate("candidate-token")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_authenticator_maps_transport_timeout_to_gateway_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with (
            patch.object(
                Configuration,
                "get_platform_config",
                return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
            ),
            pytest.raises(AccessKeyLifecycleUnavailableError) as exc_info,
        ):
            await authenticator.authenticate("candidate-token")

    assert exc_info.value.status_code == 504


@pytest.mark.asyncio
async def test_authenticator_opens_circuit_and_recovers_after_window() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count <= 3:
            return httpx.Response(500, request=request, json={"detail": "auth unavailable"})
        return httpx.Response(
            200,
            request=request,
            json={
                "principal": "alice@example.com",
                "groups": [],
                "scopes": [],
                "jti": "ak_example",
                "token_kind": "access_key",
            },
        )

    monotonic_now = 0.0

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        authenticator = AccessKeyLifecycleAuthenticator(_config(), http_client=http_client)
        with (
            patch.object(
                Configuration,
                "get_platform_config",
                return_value=PlatformConfig(base_url="https://platform.example.com", services=""),
            ),
            patch("nmp.common.auth.access_key_lifecycle.time.monotonic", side_effect=lambda: monotonic_now),
        ):
            for expected_retry_after in (None, None, 5):
                with pytest.raises(AccessKeyLifecycleUnavailableError) as exc_info:
                    await authenticator.authenticate("candidate-token")
                assert exc_info.value.retry_after == expected_retry_after

            with pytest.raises(AccessKeyLifecycleUnavailableError) as exc_info:
                await authenticator.authenticate("candidate-token")
            assert exc_info.value.status_code == 503
            assert exc_info.value.retry_after == 5
            assert request_count == 3

            monotonic_now = 6.0
            result = await authenticator.authenticate("candidate-token")

    assert isinstance(result, ResolvedBearerToken)
    assert request_count == 4
