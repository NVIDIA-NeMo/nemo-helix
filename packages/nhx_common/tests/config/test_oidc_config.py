# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nhx.common.config.base import OIDCConfidentialClientConfig, OIDCPublicClientConfig
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
