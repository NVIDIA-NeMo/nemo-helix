# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest
from nhx.common.auth import auth_client_context
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig
from nhx.common.platform_client_context import (
    build_platform_client_context,
    request_service_principal_auth_headers,
    request_service_principal_auth_headers_async,
    service_principal_auth_headers,
    service_principal_auth_headers_async,
)


@pytest.mark.parametrize("token_exchange", [False, True])
def test_explicit_direct_and_obo_selection_ignores_ambient_user(
    token_exchange: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://localhost:8080")
    config = AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            workload=OIDCWorkloadConfig(
                client_id="workload",
                token_private_key_file="/tmp/unused-key",
            )
        )
        if token_exchange
        else OIDCConfig(),
    )
    alice = Principal(id="alice", email="alice@example.com", groups=["readers"])
    bob = Principal(id="bob", groups=["other"])
    incoming = AuthClient(
        principal=Principal(
            id="service:jobs",
            on_behalf_of=alice.id,
            on_behalf_of_email=alice.email,
            on_behalf_of_groups=alice.groups,
        ),
        config=config,
    )
    token = auth_client_context.set(incoming)
    try:
        with patch("nhx.common.platform_client_context.ServiceWorkloadAccessTokenProvider") as provider:
            provider.return_value.get_access_token.return_value = "token"
            direct = service_principal_auth_headers("jobs")
            if token_exchange:
                provider.assert_called_once_with(config, "jobs", None)
                assert direct == {"Authorization": "Bearer token"}
            else:
                assert direct["X-NHX-Principal-Id"] == "service:jobs"
                assert "X-NHX-Principal-On-Behalf-Of" not in direct
            provider.reset_mock()
            explicit = service_principal_auth_headers("jobs", on_behalf_of=bob)
            if token_exchange:
                provider.assert_called_once_with(config, "jobs", bob)
            else:
                assert explicit["X-NHX-Principal-On-Behalf-Of"] == "bob"
            provider.reset_mock()
            inherited = request_service_principal_auth_headers("jobs")
            if token_exchange:
                provider.assert_called_once_with(config, "jobs", alice.effective_principal)
            else:
                assert inherited["X-NHX-Principal-On-Behalf-Of"] == "alice"
            assert auth_client_context.get() is incoming
            assert incoming.principal.on_behalf_of == "alice"
    finally:
        auth_client_context.reset(token)


@pytest.mark.asyncio
async def test_async_helpers_have_same_explicit_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_BASE_URL", "http://localhost:8080")
    config = AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            workload=OIDCWorkloadConfig(
                client_id="workload",
                token_private_key_file="/tmp/unused-key",
            )
        ),
    )
    alice = Principal(id="alice")
    incoming = AuthClient(principal=alice, config=config)
    token = auth_client_context.set(incoming)
    try:
        with patch("nhx.common.platform_client_context.ServiceWorkloadAccessTokenProvider") as provider:
            provider.return_value.get_access_token_async = AsyncMock(return_value="token")
            await service_principal_auth_headers_async("jobs")
            provider.assert_called_once_with(config, "jobs", None)
            provider.reset_mock()
            await request_service_principal_auth_headers_async("jobs")
            provider.assert_called_once_with(config, "jobs", alice.effective_principal)
    finally:
        auth_client_context.reset(token)


def test_client_context_matches_explicit_header_helper_selection() -> None:
    alice = Principal(id="alice")
    config = AuthConfig()
    incoming = AuthClient(principal=alice, config=config)
    token = auth_client_context.set(incoming)
    try:
        direct = build_platform_client_context(as_service="jobs", on_behalf_of=None)
        delegated = build_platform_client_context(as_service="jobs", on_behalf_of=alice)
        assert direct.auth_plan.headers["X-NHX-Principal-Id"] == "service:jobs"
        assert "X-NHX-Principal-On-Behalf-Of" not in direct.auth_plan.headers
        assert delegated.auth_plan.headers["X-NHX-Principal-On-Behalf-Of"] == "alice"
        assert auth_client_context.get() is incoming
    finally:
        auth_client_context.reset(token)
