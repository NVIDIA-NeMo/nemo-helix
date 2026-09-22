# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import logging
from collections.abc import Mapping, Sequence
from types import ModuleType
from unittest.mock import patch

import pytest
from nemo_platform_plugin.auth import AuthContext, is_service_principal_id
from nemo_platform_plugin.auth.workload_identity import (
    DEFAULT_WORKLOAD_AUDIENCE,
    WorkloadIdentityConfigError,
    get_workload_delegation_audience,
    is_workload_identity_token_exchange_enabled,
)
from nmp.common.auth import AuthClient, AuthConfig, Principal, auth_client_context


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
        auth_context = AuthContext.from_runtime_or_headers({"X-NMP-Principal-Id": "user:header"})
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
            "X-NMP-Principal-Id": "service:worker",
            "X-NMP-Actor-Account-Id": "account-worker",
            "X-NMP-Actor-Aliases": "service:worker,legacy-worker",
            "X-NMP-Principal-On-Behalf-Of": "user@example.com",
            "X-NMP-Subject-Account-Id": "account-user",
            "X-NMP-Subject-Aliases": "legacy-user,user@example.com",
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
                "X-NMP-Principal-Id": "user:alice",
                "X-NMP-Principal-Email": "alice@example.com",
                "X-NMP-Principal-Groups": "research,platform",
            }
        )
    finally:
        auth_client_context.reset(token)

    assert auth_context is None


def test_get_workload_delegation_audience_warns_and_defaults_on_config_error(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch("nmp.common.config.get_auth_config", side_effect=RuntimeError("invalid config")),
        caplog.at_level(logging.WARNING, logger="nemo_platform_plugin.auth.workload_identity"),
    ):
        audience = get_workload_delegation_audience()

    assert audience == DEFAULT_WORKLOAD_AUDIENCE
    assert "Could not resolve auth config for workload delegation audience" in caplog.text


def test_workload_identity_token_exchange_enabled_raises_distinct_config_error() -> None:
    with patch("nmp.common.config.get_auth_config", side_effect=RuntimeError("invalid config")):
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
def test_service_principal_id_fallback_without_nmp_common(principal_id: str, expected: bool) -> None:
    real_import = builtins.__import__

    def import_without_nmp_common(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "nmp.common.auth":
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=import_without_nmp_common):
        assert is_service_principal_id(principal_id) is expected
