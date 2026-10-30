# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.config.base import (
    AuthConfig,
    OIDCConfidentialClientConfig,
    OIDCConfig,
    OIDCPublicClientConfig,
)
from pydantic import ValidationError


def test_oidc_bearer_token_source_defaults_to_access_token() -> None:
    assert OIDCPublicClientConfig(client_id="public").bearer_token_source == "access_token"


@pytest.mark.parametrize("client_config_type", [OIDCPublicClientConfig, OIDCConfidentialClientConfig])
def test_oidc_bearer_token_source_rejects_id_token(
    client_config_type: type[OIDCPublicClientConfig | OIDCConfidentialClientConfig],
) -> None:
    with pytest.raises(ValidationError, match="bearer_token_source"):
        client_config_type.model_validate(
            {
                "client_id": "client",
                "client_secret_env_var": "OIDC_CLIENT_SECRET",
                "login_redirect_uri": "https://helix.example/apis/auth/v2/login/callback",
                "bearer_token_source": "id_token",
            }
        )


def test_oidc_bearer_token_source_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError, match="bearer_token_source"):
        OIDCPublicClientConfig.model_validate({"client_id": "public", "bearer_token_source": "refresh_token"})


def test_oidc_enabled_requires_audience() -> None:
    with pytest.raises(ValidationError, match="auth.oidc.audience is required"):
        OIDCConfig(enabled=True, issuer="https://sso.example.com")


@pytest.mark.parametrize("audience", ["", "   "])
def test_oidc_enabled_rejects_blank_audience(audience: str) -> None:
    with pytest.raises(ValidationError, match="auth.oidc.audience is required"):
        OIDCConfig(enabled=True, issuer="https://sso.example.com", audience=audience)


def test_oidc_enabled_accepts_audience() -> None:
    assert OIDCConfig(enabled=True, issuer="https://sso.example.com", audience="nemo-helix").audience == "nemo-helix"


def test_oidc_additional_audiences_default_empty() -> None:
    config = OIDCConfig(enabled=True, issuer="https://sso.example.com", audience="nemo-helix")
    assert config.additional_audiences == []


def test_oidc_disabled_does_not_require_audience() -> None:
    assert OIDCConfig(enabled=False).audience is None


def test_auth_config_rejects_enabled_oidc_without_audience() -> None:
    with pytest.raises(ValidationError, match="auth.oidc.audience is required"):
        AuthConfig(enabled=True, oidc={"enabled": True, "issuer": "https://sso.example.com"})


def test_oidc_rejects_userinfo_token_resolution() -> None:
    with pytest.raises(ValidationError, match="resolve_opaque_tokens_via_userinfo is no longer supported"):
        OIDCConfig(
            enabled=True,
            issuer="https://sso.example.com",
            audience="nemo-helix",
            resolve_opaque_tokens_via_userinfo=True,
        )


def test_oidc_rejects_userinfo_token_resolution_when_disabled() -> None:
    with pytest.raises(ValidationError, match="resolve_opaque_tokens_via_userinfo is no longer supported"):
        OIDCConfig(enabled=False, resolve_opaque_tokens_via_userinfo=True)


def test_oidc_userinfo_token_resolution_defaults_to_false() -> None:
    config = OIDCConfig(enabled=True, issuer="https://sso.example.com", audience="nemo-helix")
    assert config.resolve_opaque_tokens_via_userinfo is False
