# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from nemo_helix_plugin.config import NemoHelixConfig, Runtime
from nhx.common.config import AuthConfig
from nhx.common.config.base import (
    OIDCConfidentialClientConfig,
    OIDCConfig,
    OIDCPublicClientConfig,
    OIDCServerSessionsConfig,
    OIDCWorkloadConfig,
)
from pydantic import ValidationError


def test_oidc_profiles_are_independently_optional() -> None:
    config = OIDCConfig()

    assert config.public_client is None
    assert config.confidential_client is None
    assert config.server_sessions is None
    assert config.workload is None


def test_oidc_public_client_accepts_provider_overrides() -> None:
    config = OIDCConfig(
        public_client=OIDCPublicClientConfig(
            client_id="nhx-public",
            bearer_token_source="id_token",
            device_authorization_requires_device_id=True,
            device_authorization_display_name="NeMo Helix CLI",
            device_token_request_includes_scope=False,
        )
    )

    assert config.public_client is not None
    assert config.public_client.client_id == "nhx-public"
    assert config.public_client.bearer_token_source == "id_token"
    assert config.public_client.device_authorization_requires_device_id is True
    assert config.public_client.device_authorization_display_name == "NeMo Helix CLI"
    assert config.public_client.device_token_request_includes_scope is False


def test_oidc_confidential_client_requires_secret_references_and_redirect() -> None:
    with pytest.raises(ValidationError, match="client_secret_env_var"):
        OIDCConfidentialClientConfig.model_validate(
            {
                "client_id": "nemo-helix-user",
                "login_redirect_uri": "https://gateway.example/apis/auth/v2/login/callback",
            }
        )


def test_oidc_interactive_clients_must_have_distinct_ids() -> None:
    with pytest.raises(ValidationError, match="different client_id"):
        OIDCConfig(
            public_client=OIDCPublicClientConfig(client_id="same"),
            confidential_client=OIDCConfidentialClientConfig(
                client_id="same",
                client_secret_env_var="NHX_OIDC_CLIENT_SECRET",
                login_redirect_uri="https://gateway.example/apis/auth/v2/login/callback",
            ),
            server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
        )


def test_oidc_interactive_clients_share_validation_scope_prefix() -> None:
    with pytest.raises(ValidationError, match="same scope_prefix"):
        OIDCConfig(
            public_client=OIDCPublicClientConfig(client_id="public", scope_prefix="public/"),
            confidential_client=OIDCConfidentialClientConfig(
                client_id="confidential",
                client_secret_env_var="NHX_OIDC_CLIENT_SECRET",
                login_redirect_uri="https://gateway.example/apis/auth/v2/login/callback",
                scope_prefix="confidential/",
            ),
            server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
        )


def test_oidc_public_client_defaults_to_direct_sessions() -> None:
    config = OIDCConfig(public_client=OIDCPublicClientConfig(client_id="public"))

    assert config.public_client is not None
    assert config.public_client.server_side_sessions is False
    assert config.server_sessions is None


def test_oidc_brokered_public_client_requires_server_session_key() -> None:
    with pytest.raises(ValidationError, match="server_sessions.encryption_key_env_var"):
        OIDCConfig(public_client=OIDCPublicClientConfig(client_id="public", server_side_sessions=True))


def test_oidc_confidential_client_requires_server_session_key() -> None:
    confidential = OIDCConfidentialClientConfig(
        client_id="confidential",
        client_secret_env_var="NHX_OIDC_CLIENT_SECRET",
        login_redirect_uri="https://gateway.example/apis/auth/v2/login/callback",
    )

    with pytest.raises(ValidationError, match="server_sessions.encryption_key_env_var"):
        OIDCConfig(confidential_client=confidential)


def test_oidc_confidential_client_rejects_legacy_session_key_location() -> None:
    with pytest.raises(ValidationError, match="server_sessions.encryption_key_env_var"):
        OIDCConfidentialClientConfig.model_validate(
            {
                "client_id": "confidential",
                "client_secret_env_var": "NHX_OIDC_CLIENT_SECRET",
                "session_encryption_key_env_var": "NHX_AUTH_SESSION_ENCRYPTION_KEY",
                "login_redirect_uri": "https://gateway.example/apis/auth/v2/login/callback",
            }
        )


def test_oidc_server_sessions_do_not_enable_public_broker_by_presence() -> None:
    config = OIDCConfig(
        public_client=OIDCPublicClientConfig(client_id="public"),
        server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
    )

    assert config.public_client is not None
    assert config.public_client.server_side_sessions is False


def test_oidc_workload_profile_presence_enables_workload_configuration() -> None:
    config = OIDCConfig(workload=OIDCWorkloadConfig(client_id="nemo-workload", audience="nemo-helix"))

    assert config.workload is not None
    assert config.workload.client_id == "nemo-workload"
    assert config.workload.audience == "nemo-helix"


@pytest.mark.parametrize(
    ("removed_field", "replacement"),
    [
        ("client_id", "public_client.client_id or confidential_client.client_id"),
        ("client_authentication", "public_client or confidential_client"),
        ("session_encryption_key_env_var", "server_sessions.encryption_key_env_var"),
        ("workload_client_id", "workload.client_id"),
    ],
)
def test_oidc_removed_flat_fields_have_exact_migration_error(removed_field: str, replacement: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        OIDCConfig.model_validate({removed_field: "removed"})

    assert f"{removed_field} -> {replacement}" in str(exc_info.value)
    assert "Removed auth.oidc fields must use explicit client profiles" in str(exc_info.value)


def test_oidc_bearer_token_source_rejects_unknown_values() -> None:
    with pytest.raises(ValidationError, match="bearer_token_source"):
        OIDCPublicClientConfig.model_validate({"client_id": "public", "bearer_token_source": "refresh_token"})


def test_advertised_base_url_defaults_to_connection_base_url() -> None:
    config = NemoHelixConfig(runtime=Runtime.NONE, base_url="http://nemo-api:8080")

    assert config.effective_advertised_base_url == "http://nemo-api:8080"


def test_advertised_base_url_can_override_connection_base_url() -> None:
    config = NemoHelixConfig(
        runtime=Runtime.NONE,
        base_url="http://nemo-api:8080",
        advertised_base_url="https://127.0.0.1:18082",
    )

    assert config.effective_advertised_base_url == "https://127.0.0.1:18082"


def test_unix_connection_base_url_requires_an_advertised_http_url_for_client_discovery() -> None:
    config = NemoHelixConfig(runtime=Runtime.NONE, base_url="unix:///tmp/nemo-helix.sock")

    with pytest.raises(ValueError, match="advertised_base_url must be configured"):
        _ = config.effective_advertised_base_url

    advertised_config = NemoHelixConfig(
        runtime=Runtime.NONE,
        base_url="unix:///tmp/nemo-helix.sock",
        advertised_base_url="https://gateway.example",
    )
    assert advertised_config.effective_advertised_base_url == "https://gateway.example"


def test_platform_base_urls_are_normalized_and_preserve_path_prefixes() -> None:
    config = NemoHelixConfig(
        runtime=Runtime.NONE,
        base_url="http://nemo-api:8080/platform/",
        advertised_base_url="https://gateway.example/helix/",
    )

    assert config.base_url == "http://nemo-api:8080/platform"
    assert config.effective_advertised_base_url == "https://gateway.example/helix"


@pytest.mark.parametrize(
    "url",
    [
        "gateway.example",
        "ftp://gateway.example",
        "https://gateway.example?from=test",
        "https://gateway.example#fragment",
    ],
)
def test_platform_base_urls_reject_invalid_origins(url: str) -> None:
    with pytest.raises(ValidationError, match="absolute HTTP|query string"):
        NemoHelixConfig(runtime=Runtime.NONE, advertised_base_url=url)


def test_oidc_profiles_load_from_nested_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NHX_AUTH_OIDC__PUBLIC_CLIENT__CLIENT_ID", "public-from-env")
    monkeypatch.setenv("NHX_AUTH_OIDC__SERVER_SESSIONS__ENCRYPTION_KEY_ENV_VAR", "SESSION_KEY")
    monkeypatch.setenv("NHX_AUTH_OIDC__WORKLOAD__CLIENT_ID", "workload-from-env")
    monkeypatch.setenv("NHX_AUTH_OIDC__WORKLOAD__TOKEN_PRIVATE_KEY_FILE", "/tmp/workload.pem")

    config = AuthConfig()

    assert config.oidc.public_client is not None
    assert config.oidc.public_client.client_id == "public-from-env"
    assert config.oidc.server_sessions is not None
    assert config.oidc.server_sessions.encryption_key_env_var == "SESSION_KEY"
    assert config.oidc.workload is not None
    assert config.oidc.workload.client_id == "workload-from-env"
