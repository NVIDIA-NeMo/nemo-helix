# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.auth.discovery import (
    AuthDiscoveryBearerTokenSourceError,
    AuthDiscoveryResponse,
    OIDCDiscoveryResponse,
    parse_bearer_token_source,
)
from pydantic import ValidationError


def test_auth_discovery_response_accepts_endpoint_shape() -> None:
    response = AuthDiscoveryResponse(
        auth_enabled=True,
        oidc=OIDCDiscoveryResponse(
            issuer="https://sso.example.com",
            client_id="nhx",
            bearer_token_source="id_token",
        ),
    )

    assert response.auth_enabled is True
    assert response.oidc is not None
    assert response.oidc.bearer_token_source == "id_token"


def test_auth_discovery_response_rejects_malformed_scalar_types() -> None:
    with pytest.raises(ValidationError, match="auth_enabled"):
        AuthDiscoveryResponse.model_validate({"auth_enabled": "true"})


def test_oidc_discovery_response_rejects_unknown_bearer_token_source() -> None:
    with pytest.raises(AuthDiscoveryBearerTokenSourceError, match="bearer_token_source"):
        OIDCDiscoveryResponse.model_validate(
            {
                "issuer": "https://sso.example.com",
                "client_id": "nhx",
                "bearer_token_source": "refresh_token",
            }
        )


def test_parse_bearer_token_source_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="bearer_token_source"):
        parse_bearer_token_source("refresh_token")
