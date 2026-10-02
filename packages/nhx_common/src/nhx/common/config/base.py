# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform configuration — re-exports from nemo_helix_plugin.config with platform-specific extensions.

Core config classes and HelixConfig live in nemo_helix_plugin.config. This module
re-exports them and adds platform-specific config classes that require heavy
dependencies (sqlalchemy, etc.) or internal platform logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from nemo_helix_plugin.config import LOOPBACK_ADDRESSES as LOOPBACK_ADDRESSES
from nemo_helix_plugin.config import NHX_CONFIG_FILE_PATH_DEFAULT as NHX_CONFIG_FILE_PATH_DEFAULT
from nemo_helix_plugin.config import NHX_CONFIG_FILE_PATH_ENV_VAR as NHX_CONFIG_FILE_PATH_ENV_VAR
from nemo_helix_plugin.config import NHX_CONFIG_WARNINGS_DISABLED_ENV_VAR as NHX_CONFIG_WARNINGS_DISABLED_ENV_VAR
from nemo_helix_plugin.config import NHX_PREFIX_BASE as NHX_PREFIX_BASE
from nemo_helix_plugin.config import CommonServiceConfig as CommonServiceConfig

# Re-export everything from nemo-helix-plugin config (canonical source)
from nemo_helix_plugin.config import Configuration as Configuration
from nemo_helix_plugin.config import DockerConfig as DockerConfig
from nemo_helix_plugin.config import EnvironmentFirstSettings as EnvironmentFirstSettings
from nemo_helix_plugin.config import ImagePullSecret as ImagePullSecret
from nemo_helix_plugin.config import NemoHelixConfig as _PluginHelixConfig
from nemo_helix_plugin.config import Runtime as Runtime
from nemo_helix_plugin.config import ServiceConfig as ServiceConfig
from nemo_helix_plugin.config import create_service_config_class as create_service_config_class
from nemo_helix_plugin.config import determine_loopback_override as determine_loopback_override
from nemo_helix_plugin.config import get_platform_config_class as get_platform_config_class
from nemo_helix_plugin.config import get_service_config as get_service_config
from nemo_helix_plugin.config import get_service_config_prefix as get_service_config_prefix
from nemo_helix_plugin.config import internal_field as internal_field
from nemo_helix_plugin.config import register_platform_config_class as register_platform_config_class
from nhx.common.config.paths import nhx_user_data_dir
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import URL

# Kept here for backward compat (used by services and tests)
NHX_SERVICES_ENV_VAR = "NHX_SERVICES"
NHX_CONTROLLERS_ENV_VAR = "NHX_CONTROLLERS"
NHX_SIDECARS_ENV_VAR = "NHX_SIDECARS"

T = _PluginHelixConfig  # Backward compat for TypeVar usage


class HelixConfig(_PluginHelixConfig):
    """Platform-wide configuration settings.

    Extends NemoHelixConfig with local-service routing: when a service is
    running in the same process, get_service_url() returns the local host/port
    from CommonServiceConfig instead of the base URL.
    """

    def get_service_url(self, api_name: str) -> str:
        if self._is_service_local(api_name):
            common = get_common_service_config()
            return common.get_host_url()
        return super().get_service_url(api_name)


# Register HelixConfig so get_platform_config() returns the extended version
# with local-service routing.
Configuration.register_platform_config_class(HelixConfig)


class OIDCPublicClientConfig(BaseSettings):
    """Public OIDC client used by browser and device-based applications."""

    client_id: str
    server_side_sessions: bool = False
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    device_authorization_endpoint: str | None = None
    bearer_token_source: Literal["access_token", "id_token"] = "access_token"
    default_scopes: str = "openid profile email offline_access"
    scope_prefix: str | None = None
    device_authorization_requires_device_id: bool = False
    device_authorization_display_name: str | None = None
    device_token_request_includes_scope: bool = True


class OIDCConfidentialClientConfig(BaseSettings):
    """Confidential OIDC client whose secret is held by the auth service."""

    client_id: str
    client_secret_env_var: str
    login_redirect_uri: str
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    bearer_token_source: Literal["access_token", "id_token"] = "access_token"
    default_scopes: str = "openid profile email offline_access"
    scope_prefix: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_removed_session_key_field(cls, values: object) -> object:
        if isinstance(values, dict) and "session_encryption_key_env_var" in values:
            raise ValueError(
                "Removed auth.oidc.confidential_client.session_encryption_key_env_var; "
                "use auth.oidc.server_sessions.encryption_key_env_var"
            )
        return values


class OIDCServerSessionsConfig(BaseSettings):
    """Shared encryption configuration for server-managed OIDC sessions."""

    encryption_key_env_var: str


class OIDCWorkloadConfig(BaseSettings):
    """OIDC workload identity token-exchange client."""

    client_id: str
    token_endpoint: str | None = None
    audience: str | None = None
    scope: str | None = None
    token_issuer: str | None = None
    token_ttl_seconds: int = Field(default=300, ge=1)
    token_key_id: str | None = None
    token_private_key_file: str | None = None
    allowed_audiences: list[str] = Field(default_factory=list)
    subject_jwks_uri: str | None = None
    subject_issuers: list[str] = Field(default_factory=list)
    subject_jwks_cache_ttl_seconds: int = Field(default=3600, ge=0)
    kubernetes_token_review_enabled: bool = False


class OIDCConfig(BaseSettings):
    """OIDC identity-provider and client-profile configuration."""

    _REMOVED_CLIENT_FIELDS: ClassVar[dict[str, str]] = {
        "client_id": "public_client.client_id or confidential_client.client_id",
        "client_authentication": "public_client or confidential_client",
        "client_secret_env_var": "confidential_client.client_secret_env_var",
        "session_encryption_key_env_var": "server_sessions.encryption_key_env_var",
        "login_redirect_uri": "confidential_client.login_redirect_uri",
        "public_client_id": "public_client.client_id",
        "user_login_client_id": "public_client.client_id",
        "bearer_token_source": "public_client.bearer_token_source or confidential_client.bearer_token_source",
        "authorization_endpoint": "public_client.authorization_endpoint or confidential_client.authorization_endpoint",
        "token_endpoint": "public_client.token_endpoint or confidential_client.token_endpoint",
        "device_authorization_endpoint": "public_client.device_authorization_endpoint",
        "device_authorization_requires_device_id": "public_client.device_authorization_requires_device_id",
        "device_authorization_display_name": "public_client.device_authorization_display_name",
        "device_token_request_includes_scope": "public_client.device_token_request_includes_scope",
        "default_scopes": "public_client.default_scopes or confidential_client.default_scopes",
        "scope_prefix": "public_client.scope_prefix or confidential_client.scope_prefix",
        "workload_token_exchange_enabled": "workload",
        "workload_client_id": "workload.client_id",
        "workload_token_endpoint": "workload.token_endpoint",
        "workload_audience": "workload.audience",
        "workload_scope": "workload.scope",
        "workload_token_issuer": "workload.token_issuer",
        "workload_token_ttl_seconds": "workload.token_ttl_seconds",
        "workload_token_key_id": "workload.token_key_id",
        "workload_token_private_key_file": "workload.token_private_key_file",
        "workload_allowed_audiences": "workload.allowed_audiences",
        "workload_subject_jwks_uri": "workload.subject_jwks_uri",
        "workload_subject_issuers": "workload.subject_issuers",
        "workload_subject_jwks_cache_ttl_seconds": "workload.subject_jwks_cache_ttl_seconds",
        "workload_kubernetes_token_review_enabled": "workload.kubernetes_token_review_enabled",
    }

    enabled: bool = Field(
        default=False,
        description="Enable native OIDC token validation.",
    )

    issuer: str = Field(
        default="",
        description="OIDC issuer URL (e.g., https://sso.nvidia.com). "
        "Used for token validation and .well-known discovery.",
    )

    additional_issuers: list[str] = Field(
        default_factory=list,
        description="Additional valid issuers for token validation. "
        "Useful for Azure AD where access tokens use v1.0 issuer format "
        "(https://sts.windows.net/{tenant}/) while endpoints use v2.0.",
    )

    public_client: OIDCPublicClientConfig | None = None
    confidential_client: OIDCConfidentialClientConfig | None = None
    server_sessions: OIDCServerSessionsConfig | None = None
    workload: OIDCWorkloadConfig | None = None

    jwks_uri: str | None = Field(
        default=None,
        description="Override JWKS URI for token validation (defaults to discovery).",
    )

    introspect_opaque_tokens: bool = Field(
        default=False,
        description="Fall back to RFC 7662 token introspection when a bearer token is not a JWT "
        "(some IdPs issue opaque access tokens). Requires introspection_endpoint to be set or "
        "discoverable, and typically introspection_client_secret_env_var pointing at a secret-backed "
        "environment variable.",
    )

    introspection_endpoint: str | None = Field(
        default=None,
        description="Override RFC 7662 token introspection endpoint (defaults to discovery). "
        "Only used when introspect_opaque_tokens is enabled.",
    )

    introspection_client_id: str | None = Field(
        default=None,
        description=(
            "Client ID used to authenticate RFC 7662 introspection requests. Defaults to the confidential client ID, "
            "then the public client ID, when unset."
        ),
    )

    introspection_client_secret_env_var: str | None = Field(
        default=None,
        description="Environment variable containing the client secret used to authenticate "
        "RFC 7662 introspection requests. Required by most IdPs when introspect_opaque_tokens "
        "is enabled. Store only the environment variable name here; the secret value must come "
        "from the process environment.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_removed_client_fields(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        removed = sorted(cls._REMOVED_CLIENT_FIELDS.keys() & values.keys())
        if not removed:
            return values
        migrations = ", ".join(f"{name} -> {cls._REMOVED_CLIENT_FIELDS[name]}" for name in removed)
        raise ValueError(f"Removed auth.oidc fields must use explicit client profiles: {migrations}")

    @model_validator(mode="after")
    def validate_interactive_clients(self) -> Self:
        sessions_required = self.confidential_client is not None or (
            self.public_client is not None and self.public_client.server_side_sessions
        )
        if sessions_required and self.server_sessions is None:
            raise ValueError(
                "auth.oidc.server_sessions.encryption_key_env_var is required for server-side OIDC sessions"
            )
        if (
            self.public_client is not None
            and self.confidential_client is not None
            and self.public_client.client_id == self.confidential_client.client_id
        ):
            raise ValueError("auth.oidc public_client and confidential_client must use different client_id values")
        prefixes = {
            client.scope_prefix
            for client in (self.public_client, self.confidential_client)
            if client is not None and client.scope_prefix is not None
        }
        if len(prefixes) > 1:
            raise ValueError("auth.oidc public_client and confidential_client must use the same scope_prefix")
        return self

    @property
    def effective_scope_prefix(self) -> str | None:
        """Return the shared scope prefix used to normalize validated tokens."""
        if self.confidential_client is not None and self.confidential_client.scope_prefix is not None:
            return self.confidential_client.scope_prefix
        if self.public_client is not None:
            return self.public_client.scope_prefix
        return None

    resolve_opaque_tokens_via_userinfo: bool = Field(
        default=False,
        description="Fall back to the OIDC UserInfo endpoint when a bearer token is not a JWT "
        "(some IdPs issue opaque access tokens and don't support introspecting them via RFC 7662). "
        "Requires userinfo_endpoint to be set or discoverable. Takes priority over "
        "introspect_opaque_tokens when both are enabled, except when oidc.audience is also "
        "configured: UserInfo responses can't be checked against an audience, so validation "
        "falls back to introspect_opaque_tokens (if enabled) instead. The token's original OAuth "
        "scope grant also can't be recovered from UserInfo responses, so it goes unenforced for "
        "tokens resolved this way (RBAC/permissions still apply). Enable only for IdPs that don't "
        "issue narrower per-token scopes in practice.",
    )

    userinfo_endpoint: str | None = Field(
        default=None,
        description="Override OIDC UserInfo endpoint (defaults to discovery). "
        "Only used when resolve_opaque_tokens_via_userinfo is enabled.",
    )

    # Token validation settings
    audience: str | None = Field(
        default=None,
        description="Expected token audience. When set, tokens must include this value in their 'aud' claim. "
        "When not set, audience validation is skipped entirely.",
    )

    email_claim: str = Field(
        default="email",
        description="JWT claim containing the principal email (maps to X-NHX-Principal-Email). "
        "Set explicitly for your IdP.",
    )

    groups_claim: str = Field(
        default="groups",
        description="JWT claim containing user groups. "
        "Supports 'groups' (standard) and 'cognito:groups' (AWS Cognito).",
    )

    subject_claim: str = Field(
        default="sub",
        description="JWT claim to use as principal ID (maps to X-NHX-Principal-Id). Set explicitly for your IdP.",
    )

    discovery_cache_ttl: int = Field(
        default=300,
        description="TTL in seconds for caching IdP discovery document responses. "
        "Used by the discovery endpoint to avoid per-request IdP calls. "
        "Set to 0 to disable caching.",
    )


class TokenSigningConfig(BaseSettings):
    """Shared NeMo auth-service token signing configuration."""

    issuer: str | None = Field(
        default=None,
        description="Shared issuer for NeMo Helix-minted JWTs. Defaults to <platform.base_url>/apis/auth.",
    )
    key_id: str = Field(
        default="nemo-helix-signing",
        description="Shared JWT key id advertised by NeMo Helix JWKS endpoints.",
    )
    private_key_file: str | None = Field(
        default=None,
        description=(
            "Path to the PEM-encoded RSA private key used by the auth service to sign NeMo Helix-minted JWTs. "
            "Intended for mounted shared secrets."
        ),
    )


def _default_access_key_accepted_formats() -> list[Literal["jwt"]]:
    return ["jwt"]


AccessKeyAcceptedFormat = Literal["jwt"]


class AccessKeyConfig(BaseSettings):
    """NeMo Helix Scoped Access Key configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable NeMo Helix Scoped Access Key creation and validation.",
    )
    issue_format: Literal["jwt"] = Field(
        default="jwt",
        description="Token format to issue for newly created Scoped Access Keys.",
    )
    accepted_formats: Annotated[list[AccessKeyAcceptedFormat], NoDecode] = Field(
        default_factory=_default_access_key_accepted_formats,
        description="Scoped Access Key token formats accepted by validators.",
    )
    audience: str = Field(
        default="nemo-helix-access-key",
        description="Expected audience for NeMo Helix Scoped Access Key JWTs.",
    )
    default_expires_in_seconds: int | None = Field(
        default=30 * 24 * 60 * 60,
        ge=1,
        description=(
            "Default finite lifetime in seconds for newly created Scoped Access Keys when "
            "expires_in_seconds is omitted. Set to null to require callers to provide an expiry "
            "when max_expires_in_seconds is finite, or to make omitted expiry non-time-delimited "
            "when max_expires_in_seconds is also null."
        ),
    )
    max_expires_in_seconds: int | None = Field(
        default=30 * 24 * 60 * 60,
        ge=1,
        description=(
            "Maximum finite lifetime accepted when creating Scoped Access Keys. "
            "Set to null to allow explicit no-expiration requests."
        ),
    )
    rotation_grace_period_seconds: int = Field(
        default=48 * 60 * 60,
        ge=1,
        description=(
            "Default grace period in seconds for a rotated-out Scoped Access Key, used when "
            "grace_period_seconds is omitted from POST /v2/access-keys/{jti}/rotate. The key "
            "remains usable for this long after rotation before it is treated as revoked. "
            "Gives callers a dual-active window to cut over to the newly issued key."
        ),
    )
    max_rotation_grace_period_seconds: int | None = Field(
        default=30 * 24 * 60 * 60,
        ge=1,
        description=(
            "Maximum grace period accepted when a caller explicitly requests grace_period_seconds "
            "on POST /v2/access-keys/{jti}/rotate. Set to null to allow any explicitly requested "
            "grace period."
        ),
    )

    @staticmethod
    def _parse_nullable_expiry(value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    @field_validator("accepted_formats", mode="before")
    @classmethod
    def parse_accepted_formats(cls, value: object) -> object:
        if not isinstance(value, str):
            return value

        return [part.strip() for part in value.split(",") if part.strip()]

    @field_validator("default_expires_in_seconds", "max_expires_in_seconds", mode="before")
    @classmethod
    def parse_nullable_expiry(cls, value: object) -> object:
        return cls._parse_nullable_expiry(value)

    @model_validator(mode="after")
    def validate_expiry_policy(self) -> Self:
        if self.max_expires_in_seconds is None:
            return self
        if (
            self.default_expires_in_seconds is not None
            and self.default_expires_in_seconds > self.max_expires_in_seconds
        ):
            raise ValueError(
                "auth.access_keys.default_expires_in_seconds must be less than or equal to "
                "auth.access_keys.max_expires_in_seconds"
            )
        return self


class AuthConfig(create_service_config_class("auth")):  # ty: ignore[unsupported-base]
    """
    Shared authorization configuration read from the 'auth' key in config.yaml.

    This config is read by all services to know if auth is enabled and where the PDP is.
    The auth service extends this with additional fields (admin_email, etc.).
    """

    model_config = SettingsConfigDict(
        env_prefix=get_service_config_prefix("auth"),
        env_nested_delimiter="__",
        extra="allow",
        populate_by_name=True,
    )

    enabled: bool = Field(
        default=False,
        description="Master switch for authorization. If False, all requests are allowed.",
    )

    policy_decision_point_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for the Policy Decision Point (auth service or external OPA).",
    )

    policy_decision_point_provider: Literal["embedded", "opa"] = Field(
        default="embedded",
        description=(
            "Policy Decision Point provider: "
            "'embedded' for auth service's built-in WASM engine, "
            "'opa' for external OPA sidecar."
        ),
    )

    policy_decision_point_request_timeout_seconds: int = Field(
        default=5,
        ge=1,
        description=("HTTP timeout in seconds for outbound Policy Decision Point (PDP) requests"),
    )

    propagation_poll_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Default polling interval (seconds) used by AuthClient.wait_role and "
            "wait_permissions. Lower values reduce role-propagation wait time at the "
            "cost of more PDP requests; tests typically override this to a small value."
        ),
    )

    allow_unsigned_jwt: bool = Field(
        default=False,
        description=(
            "Allow unsigned JWTs (`alg=none`) for local development/testing. "
            "Disabled by default and should not be enabled in production."
        ),
    )

    embedded_pdp_auto_build_wasm: bool = Field(
        default=True,
        description=(
            "When auth is enabled with the embedded PDP and policy.wasm is missing, "
            "build it automatically from a local NeMo Helix source checkout. Packaged deployments "
            "should include policy.wasm at build time and can disable this for fail-fast startup."
        ),
    )

    oidc: OIDCConfig = Field(
        default_factory=OIDCConfig,
        description="OIDC configuration for native token validation.",
    )

    token_signing: TokenSigningConfig = Field(
        default_factory=TokenSigningConfig,
        description="Shared token signing configuration for NeMo Helix-minted JWTs.",
    )

    access_keys: AccessKeyConfig = Field(
        default_factory=AccessKeyConfig,
        description="Scoped Access Key configuration.",
    )

    @model_validator(mode="after")
    def validate_workload_token_signing_config(self) -> Self:
        workload = self.oidc.workload
        if workload is None:
            return self

        key_id = workload.token_key_id or self.token_signing.key_id
        if not key_id or not key_id.strip():
            raise ValueError(
                "auth.oidc.workload.token_key_id or auth.token_signing.key_id must be configured "
                "when auth.oidc.workload is configured"
            )
        workload_private_key_file = self._normalized_private_key_file(workload.token_private_key_file)
        access_key_private_key_file = self._normalized_private_key_file(self.token_signing.private_key_file)
        if not workload_private_key_file and not access_key_private_key_file:
            raise ValueError(
                "auth.oidc.workload.token_private_key_file or auth.token_signing.private_key_file must be "
                "configured when auth.oidc.workload is configured"
            )
        if (
            self.access_keys.enabled
            and workload_private_key_file
            and access_key_private_key_file
            and workload_private_key_file != access_key_private_key_file
            and key_id.strip() == self.token_signing.key_id.strip()
        ):
            raise ValueError(
                "auth.oidc.workload.token_key_id must be distinct from auth.token_signing.key_id "
                "when auth.oidc.workload.token_private_key_file differs from "
                "auth.token_signing.private_key_file and Scoped Access Keys are enabled"
            )
        return self

    @staticmethod
    def _normalized_private_key_file(value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return str(Path(value).expanduser().resolve(strict=False))

    def get_pdp_url(self, entrypoint: str) -> str:
        # Import lazily to avoid a module cycle: platform_endpoint imports
        # HelixConfig from nhx.common.config, which is defined in this file.
        from nhx.common.platform_endpoint import parse_platform_endpoint

        endpoint = parse_platform_endpoint(self.policy_decision_point_base_url)
        if self.policy_decision_point_provider == "opa":
            return f"{endpoint.connect_base_url}/v1/data/authz/{entrypoint}"
        return f"{endpoint.connect_base_url}/apis/auth/v2/authz/{entrypoint}"

    @property
    def auth_url(self) -> str:
        return self.get_pdp_url("allow")


# --------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------


def get_common_service_config() -> CommonServiceConfig:
    return Configuration.get_service_config(CommonServiceConfig)


def get_platform_config() -> HelixConfig:
    return Configuration.get_service_config(HelixConfig)


def get_auth_config() -> AuthConfig:
    return Configuration.get_service_config(AuthConfig)


# --------------------------------------------------------------------------
# DatabaseConfig (needs sqlalchemy)
# --------------------------------------------------------------------------


class DatabaseConfig(EnvironmentFirstSettings):
    """
    Common configuration for database connections used by services.

    Reads database configuration from DATABASE_* environment variables.
    For services that need multiple database connections, create multiple
    DatabaseConfig instances with different env_prefix values.

    Default behavior:
    - If no connection parameters (host, path, name, port, user, password) are set,
      defaults to a SQLite database under the NeMo Helix user data directory
      (see ``nhx_user_data_dir``) — typically
      ``~/.local/share/nemo/nhx-platform.db`` — for local development
      convenience. The parent directory is created on first use.
    - If any connection parameters are set but dialect is not specified,
      defaults to postgresql.
    """

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    url: str | None = Field(default=None, description="Full database URL (overrides other settings)")
    dialect: str = Field(default="postgresql", description="Database dialect - either sqlite or postgresql")
    host: str = Field(default="", description="Database hostname")
    path: str = Field(default="", description="Database path")
    name: str = Field(default="", description="Database name")
    port: int | None = Field(default=None, description="Optional database port")
    user: str | None = Field(default=None, description="Optional database username")
    password: str | None = Field(default=None, description="Optional database password")
    connections_limit: int = Field(
        default=10, description="Maximum number of connections in the database connection pool"
    )
    connect_timeout_seconds: int = Field(
        default=30,
        description="Connection timeout in seconds. For PostgreSQL (asyncpg) and SQLite, how long to wait when connecting or acquiring a lock.",
    )
    echo: bool = Field(default=False, description="Enable SQLAlchemy echo for the database connection")

    def sqlalchemy_database_url(self) -> str:
        if self.url:
            return self.url
        has_connection_params = any([self.host, self.path, self.name, self.port, self.user, self.password])
        if not has_connection_params:
            db_path = nhx_user_data_dir() / "nhx-platform.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{db_path}"

        username = None
        password = None
        host = None
        port = None

        if self.dialect == "sqlite":
            if not self.path:
                raise ValueError("SQLite database requires 'path' to be set")
            database = self.path
        else:
            if self.path:
                raise ValueError(f"{self.dialect} database should not use 'path' field")
            if not self.name:
                raise ValueError(f"{self.dialect} database requires 'name' to be set")
            database = self.name
            username = self.user
            password = self.password
            host = self.host
            port = self.port

        url = URL.create(
            drivername=self.dialect, username=username, password=password, host=host, port=port, database=database
        )
        return url.render_as_string(hide_password=False)
