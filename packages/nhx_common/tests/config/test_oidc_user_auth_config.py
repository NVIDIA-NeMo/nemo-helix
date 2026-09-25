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


def test_oidc_bearer_token_source_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError, match="bearer_token_source"):
        OIDCConfig(bearer_token_source="refresh_token")  # type: ignore[arg-type]
