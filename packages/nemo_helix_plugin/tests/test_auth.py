# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import nhx.common.auth as nhx_common_auth
import nhx.common.auth.dependencies as nhx_common_auth_dependencies
import pytest
from nemo_helix_plugin.auth import (
    AuthContext,
    current_workload_delegation_id,
    is_service_principal_id,
    workload_delegation_verifiable,
)
from nemo_helix_plugin.auth.workload_identity import (
    DEFAULT_WORKLOAD_AUDIENCE,
    WorkloadIdentityConfigError,
    get_workload_delegation_audience,
    is_workload_identity_token_exchange_enabled,
)
from nhx.common.auth import AuthClient, AuthConfig, Principal, auth_as_service, auth_client_context
from nhx.common.auth.token_claims import ActorClaims, TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig

_DELEGATION_ID = "workload_delegation-abc123"


def _config(tmp_path: Path, *, enabled: bool, token_exchange: bool) -> AuthConfig:
    if not token_exchange:
        return AuthConfig(enabled=enabled)
    return AuthConfig(
        enabled=enabled,
        # Token exchange needs a signing key to be named. These tests never sign or verify, so none is written.
        token_signing=TokenSigningConfig(key_id="test-workload", private_key_file=str(tmp_path / "workload.pem")),
        oidc=OIDCConfig(workload=OIDCWorkloadConfig(client_id="nemo-helix-workload")),
    )


def _client_with_delegated_token(config: AuthConfig) -> AuthClient:
    """The auth client the middleware builds for a job step's unexpired token that names ``_DELEGATION_ID``."""
    resolved = ResolvedBearerToken(
        claims=TokenClaims(
            subject="creator@example.com",
            email=None,
            groups=[],
            scopes=[],
            raw_claims={
                "exp": time.time() + 300,
                "nhx_service_attributes": {"auth": {"delegation_id": _DELEGATION_ID}},
            },
            actor=ActorClaims(subject="system:serviceaccount:nemo-runs:job-runner", groups=[]),
        ),
        token_kind="workload_access_token",
    )
    return AuthClient(principal=resolved.principal, config=config, resolved_bearer_token=resolved)


def _client_without_token(config: AuthConfig) -> AuthClient:
    return AuthClient(principal=Principal(id="user@example.com"), config=config)


@pytest.mark.parametrize(
    ("build_client", "enabled", "token_exchange", "delegation_id", "verifiable"),
    [
        pytest.param(_client_with_delegated_token, True, True, _DELEGATION_ID, True, id="delegated token"),
        pytest.param(_client_without_token, True, True, None, True, id="no token, token exchange mode"),
        pytest.param(_client_without_token, True, False, None, False, id="no token, trusted-header mode"),
        pytest.param(_client_with_delegated_token, True, False, None, False, id="delegated token, no token exchange"),
        pytest.param(_client_with_delegated_token, False, True, None, False, id="delegated token, auth off"),
    ],
)
def test_workload_delegation_accessors_read_the_request_auth_client(
    tmp_path: Path,
    build_client: Callable[[AuthConfig], AuthClient],
    enabled: bool,
    token_exchange: bool,
    delegation_id: str | None,
    verifiable: bool,
) -> None:
    token = auth_client_context.set(build_client(_config(tmp_path, enabled=enabled, token_exchange=token_exchange)))
    try:
        assert current_workload_delegation_id() == delegation_id
        assert workload_delegation_verifiable() is verifiable
    finally:
        auth_client_context.reset(token)


def test_workload_delegation_accessors_without_request_auth_context() -> None:
    token = auth_client_context.set(None)
    try:
        assert current_workload_delegation_id() is None
        assert workload_delegation_verifiable() is False
    finally:
        auth_client_context.reset(token)


def test_service_credentials_in_a_request_hide_its_workload_delegation_id(tmp_path: Path) -> None:
    token = auth_client_context.set(_client_with_delegated_token(_config(tmp_path, enabled=True, token_exchange=True)))
    try:
        with auth_as_service("builder"):
            # The mode is still the request's. The ID belongs to the request's token, which this context lacks.
            assert current_workload_delegation_id() is None
            assert workload_delegation_verifiable() is True

        assert current_workload_delegation_id() == _DELEGATION_ID
    finally:
        auth_client_context.reset(token)


@pytest.mark.parametrize("token_exchange", [True, False])
def test_service_credentials_without_a_request_report_the_platform_mode(tmp_path: Path, token_exchange: bool) -> None:
    """A background task that switches to service credentials has an auth context, although nothing was verified."""
    config = _config(tmp_path, enabled=True, token_exchange=token_exchange)
    token = auth_client_context.set(None)
    try:
        with patch("nhx.common.config.get_auth_config", return_value=config), auth_as_service("builder"):
            assert current_workload_delegation_id() is None
            assert workload_delegation_verifiable() is token_exchange

        assert workload_delegation_verifiable() is False
    finally:
        auth_client_context.reset(token)


def test_workload_delegation_accessors_without_nhx_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token = auth_client_context.set(_client_with_delegated_token(_config(tmp_path, enabled=True, token_exchange=True)))
    try:
        # Without nhx_common the request auth context can't be imported.
        monkeypatch.delattr(nhx_common_auth_dependencies, "auth_client_context")

        assert current_workload_delegation_id() is None
        assert workload_delegation_verifiable() is False
    finally:
        monkeypatch.undo()
        auth_client_context.reset(token)


def test_auth_context_from_runtime_or_headers_uses_platform_runtime_context() -> None:
    token = auth_client_context.set(
        AuthClient(
            principal=Principal(
                id="bearer@example.com",
                account_id="account-bearer",
                email="bearer@example.com",
                groups=["runtime"],
                authz_aliases=["bearer@example.com"],
            ),
            config=AuthConfig(),
        )
    )
    try:
        auth_context = AuthContext.from_runtime_or_headers({"X-NHX-Principal-Id": "user:header"})
    finally:
        auth_client_context.reset(token)

    assert auth_context is not None
    assert auth_context.principal_id == "bearer@example.com"
    assert auth_context.principal_account_id == "account-bearer"
    assert auth_context.principal_email == "bearer@example.com"
    assert auth_context.principal_groups == ["runtime"]
    assert auth_context.principal_authz_aliases == ["bearer@example.com"]


def test_auth_context_from_headers_preserves_account_alias_context() -> None:
    auth_context = AuthContext.from_headers(
        {
            "X-NHX-Principal-Id": "service:worker",
            "X-NHX-Actor-Account-Id": "account-worker",
            "X-NHX-Actor-Aliases": "service:worker,legacy-worker",
            "X-NHX-Principal-On-Behalf-Of": "user@example.com",
            "X-NHX-Subject-Account-Id": "account-user",
            "X-NHX-Subject-Aliases": "legacy-user,user@example.com",
        }
    )

    assert auth_context is not None
    assert auth_context.principal_account_id == "account-worker"
    assert auth_context.principal_authz_aliases == ["service:worker", "legacy-worker"]
    assert auth_context.principal_on_behalf_of_account_id == "account-user"
    assert auth_context.principal_on_behalf_of_authz_aliases == ["legacy-user", "user@example.com"]

    principal = auth_context.to_principal()
    assert principal.account_id == "account-worker"
    assert principal.authz_aliases == ["service:worker", "legacy-worker"]
    assert principal.on_behalf_of_account_id == "account-user"
    assert principal.on_behalf_of_authz_aliases == ["legacy-user", "user@example.com"]


def test_auth_context_from_runtime_or_headers_ignores_raw_headers() -> None:
    token = auth_client_context.set(None)
    try:
        auth_context = AuthContext.from_runtime_or_headers(
            {
                "X-NHX-Principal-Id": "user:alice",
                "X-NHX-Principal-Email": "alice@example.com",
                "X-NHX-Principal-Groups": "research,platform",
            }
        )
    finally:
        auth_client_context.reset(token)

    assert auth_context is None


def test_get_workload_delegation_audience_warns_and_defaults_on_config_error(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch("nhx.common.config.get_auth_config", side_effect=RuntimeError("invalid config")),
        caplog.at_level(logging.WARNING, logger="nemo_helix_plugin.auth.workload_identity"),
    ):
        audience = get_workload_delegation_audience()

    assert audience == DEFAULT_WORKLOAD_AUDIENCE
    assert "Could not resolve auth config for workload delegation audience" in caplog.text


def test_workload_identity_token_exchange_enabled_raises_distinct_config_error() -> None:
    with patch("nhx.common.config.get_auth_config", side_effect=RuntimeError("invalid config")):
        with pytest.raises(WorkloadIdentityConfigError, match="Could not resolve auth config"):
            is_workload_identity_token_exchange_enabled()


@pytest.mark.parametrize(
    ("principal_id", "expected"),
    [
        ("service:deployments", True),
        (" service:jobs-controller ", True),
        ("user@example.com", False),
        ("service:", False),
        ("service:has spaces", False),
        ("service:/path", False),
        ("service:*", False),
        ("service:-starts-with-punctuation", False),
        ("service-account:deployments", False),
    ],
)
def test_service_principal_id_fallback_without_nhx_common(
    monkeypatch: pytest.MonkeyPatch, principal_id: str, expected: bool
) -> None:
    # Without nhx_common the shared service-principal parser can't be imported.
    monkeypatch.delattr(nhx_common_auth, "is_service_principal")

    assert is_service_principal_id(principal_id) is expected
