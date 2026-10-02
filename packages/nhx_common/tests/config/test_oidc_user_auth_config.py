# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.config.base import OIDCConfig
from pydantic import ValidationError


def test_oidc_user_auth_compatibility_defaults_are_standard() -> None:
    config = OIDCConfig()

    assert config.cli_client_id is None
    assert config.bearer_token_source == "access_token"
    assert config.device_authorization_requires_device_id is False
    assert config.device_authorization_display_name is None
    assert config.device_token_request_includes_scope is True


def test_oidc_user_auth_compatibility_accepts_provider_overrides() -> None:
    config = OIDCConfig(
        cli_client_id="nhx-cli",
        bearer_token_source="id_token",
        device_authorization_requires_device_id=True,
        device_authorization_display_name="NeMo Helix CLI",
        device_token_request_includes_scope=False,
    )

    assert config.cli_client_id == "nhx-cli"
    assert config.bearer_token_source == "id_token"
    assert config.device_authorization_requires_device_id is True
    assert config.device_authorization_display_name == "NeMo Helix CLI"
    assert config.device_token_request_includes_scope is False


def test_oidc_platform_client_defaults_to_public() -> None:
    config = OIDCConfig()

    assert config.token_endpoint_auth_method == "none"
    assert config.client_secret_env_var is None
    assert config.public_client_id is None


def test_oidc_confidential_platform_client_requires_env_var_names() -> None:
    config = OIDCConfig(
        client_id="nemo-helix-user",
        token_endpoint_auth_method="client_secret_basic",
        client_secret_env_var="NHX_OIDC_CLIENT_SECRET",
        session_encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY",
        public_client_id="nemo-helix-cli",
    )

    assert config.client_id == "nemo-helix-user"
    assert config.public_client_id == "nemo-helix-cli"


def test_oidc_confidential_platform_client_rejects_missing_secret_env_var() -> None:
    with pytest.raises(ValidationError, match="client_secret_env_var"):
        OIDCConfig(
            token_endpoint_auth_method="client_secret_basic",
            session_encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY",
        )


def test_oidc_rejects_inline_client_secret() -> None:
    with pytest.raises(ValidationError, match="client_secret_env_var"):
        OIDCConfig(client_secret="super-secret")  # type: ignore[call-arg]


def test_oidc_rejects_inline_session_encryption_key() -> None:
    with pytest.raises(ValidationError, match="session_encryption_key_env_var"):
        OIDCConfig(session_encryption_key="key")  # type: ignore[call-arg]


def test_oidc_public_client_must_differ_from_platform_client() -> None:
    with pytest.raises(ValidationError, match="public_client_id"):
        OIDCConfig(client_id="same", public_client_id="same")


def test_oidc_bearer_token_source_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError, match="bearer_token_source"):
        OIDCConfig(bearer_token_source="refresh_token")  # type: ignore[arg-type]
