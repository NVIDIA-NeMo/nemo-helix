# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_identity import (
    DEFAULT_WORKLOAD_AUDIENCE,
    WorkloadIdentityConfigError,
    get_workload_delegation_audience,
    is_workload_identity_token_exchange_enabled,
)
from nhx.common.auth import AuthClient, AuthConfig, Principal, auth_client_context


def test_auth_context_from_runtime_or_headers_uses_platform_runtime_context() -> None:
    token = auth_client_context.set(
        AuthClient(
            principal=Principal(id="bearer@example.com", email="bearer@example.com", groups=["runtime"]),
            config=AuthConfig(),
        )
    )
    try:
        auth_context = AuthContext.from_runtime_or_headers({"X-NHX-Principal-Id": "user:header"})
    finally:
        auth_client_context.reset(token)

    assert auth_context is not None
    assert auth_context.principal_id == "bearer@example.com"
    assert auth_context.principal_email == "bearer@example.com"
    assert auth_context.principal_groups == ["runtime"]


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
