# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform configuration — re-exports from nemo_platform_plugin.config with platform-specific extensions.

Core config classes and PlatformConfig live in nemo_platform_plugin.config. This module
re-exports them and adds platform-specific config classes that require heavy
dependencies (sqlalchemy, etc.) or internal platform logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from nemo_platform_plugin.config import LOOPBACK_ADDRESSES as LOOPBACK_ADDRESSES
from nemo_platform_plugin.config import NMP_CONFIG_FILE_PATH_DEFAULT as NMP_CONFIG_FILE_PATH_DEFAULT
from nemo_platform_plugin.config import NMP_CONFIG_FILE_PATH_ENV_VAR as NMP_CONFIG_FILE_PATH_ENV_VAR
from nemo_platform_plugin.config import NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR as NMP_CONFIG_WARNINGS_DISABLED_ENV_VAR
from nemo_platform_plugin.config import NMP_PREFIX_BASE as NMP_PREFIX_BASE
from nemo_platform_plugin.config import CommonServiceConfig as CommonServiceConfig

# Re-export everything from nemo-platform-plugin config (canonical source)
from nemo_platform_plugin.config import Configuration as Configuration
from nemo_platform_plugin.config import DockerConfig as DockerConfig
from nemo_platform_plugin.config import EnvironmentFirstSettings as EnvironmentFirstSettings
from nemo_platform_plugin.config import ImagePullSecret as ImagePullSecret
from nemo_platform_plugin.config import NemoPlatformConfig as _PluginPlatformConfig
from nemo_platform_plugin.config import Runtime as Runtime
from nemo_platform_plugin.config import ServiceConfig as ServiceConfig
from nemo_platform_plugin.config import create_service_config_class as create_service_config_class
from nemo_platform_plugin.config import determine_loopback_override as determine_loopback_override
from nemo_platform_plugin.config import get_platform_config_class as get_platform_config_class
from nemo_platform_plugin.config import get_service_config as get_service_config
from nemo_platform_plugin.config import get_service_config_prefix as get_service_config_prefix
from nemo_platform_plugin.config import internal_field as internal_field
from nemo_platform_plugin.config import register_platform_config_class as register_platform_config_class
from nmp.common.config.paths import nmp_user_data_dir
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import URL

# Kept here for backward compat (used by services and tests)
NMP_SERVICES_ENV_VAR = "NMP_SERVICES"
NMP_CONTROLLERS_ENV_VAR = "NMP_CONTROLLERS"
NMP_SIDECARS_ENV_VAR = "NMP_SIDECARS"

T = _PluginPlatformConfig  # Backward compat for TypeVar usage


class PlatformConfig(_PluginPlatformConfig):
    """Platform-wide configuration settings.

    Extends NemoPlatformConfig with local-service routing: when a service is
    running in the same process, get_service_url() returns the local host/port
    from CommonServiceConfig instead of the base URL.
    """

    def get_service_url(self, api_name: str) -> str:
        if self._is_service_local(api_name):
            common = get_common_service_config()
            return common.get_host_url()
        return super().get_service_url(api_name)


# Register PlatformConfig so get_platform_config() returns the extended version
# with local-service routing.
Configuration.register_platform_config_class(PlatformConfig)


class OIDCConfig(BaseSettings):
    """OIDC Identity Provider configuration for native token validation."""

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

    client_id: str = Field(
        default="",
        description="OAuth client ID for this NeMo Platform deployment. Used for device flow and token audience validation.",
    )

    # Optional: Override endpoints if not using standard discovery
    authorization_endpoint: str | None = Field(
        default=None,
        description="Override authorization endpoint (defaults to discovery).",
    )

    token_endpoint: str | None = Field(
        default=None,
        description="Override token endpoint (defaults to discovery).",
    )

    device_authorization_endpoint: str | None = Field(
        default=None,
        description="Override device authorization endpoint (defaults to discovery).",
    )

    jwks_uri: str | None = Field(
        default=None,
        description="Override JWKS URI for token validation (defaults to discovery).",
    )

    # Token validation settings
    audience: str | None = Field(
        default=None,
        description="Expected token audience. When set, tokens must include this value in their 'aud' claim. "
        "When not set, audience validation is skipped entirely.",
    )

    email_claim: str = Field(
        default="email",
        description="JWT claim containing the principal email (maps to X-NMP-Principal-Email). "
        "Set explicitly for your IdP.",
    )

    groups_claim: str = Field(
        default="groups",
        description="JWT claim containing user groups. "
        "Supports 'groups' (standard) and 'cognito:groups' (AWS Cognito).",
    )

    subject_claim: str = Field(
        default="sub",
        description="JWT claim to use as principal ID (maps to X-NMP-Principal-Id). Set explicitly for your IdP.",
    )

    # Scope configuration
    default_scopes: str = Field(
        default="openid profile email offline_access",
        description="Space-separated OAuth scopes to request during authentication. "
        "For Azure AD with custom API, use: 'api://{app-id}/.default openid profile email'",
    )

    workload_token_exchange_enabled: bool = Field(
        default=False,
        description="Enable SDK workload identity token exchange using NMP_WORKLOAD_IDENTITY_TOKEN_FILE subject tokens.",
    )

    workload_client_id: str | None = Field(
        default=None,
        description="OAuth client ID to use for workload identity token exchange. Defaults to client_id when unset.",
    )

    workload_token_endpoint: str | None = Field(
        default=None,
        description="OAuth token endpoint to use for workload identity token exchange. Defaults to token_endpoint.",
    )

    workload_audience: str | None = Field(
        default=None,
        description="RFC 8693 audience requested for workload identity token exchange.",
    )

    workload_scope: str | None = Field(
        default=None,
        description="Space-separated OAuth scopes requested for workload identity token exchange.",
    )

    workload_token_issuer: str | None = Field(
        default=None,
        description=(
            "Issuer to stamp on workload identity access tokens minted by the NeMo auth service. "
            "Defaults to the platform auth endpoint origin serving the token exchange request."
        ),
    )

    workload_token_ttl_seconds: int = Field(
        default=300,
        ge=1,
        description="Lifetime in seconds for workload identity access tokens minted by the NeMo auth service.",
    )

    workload_token_key_id: str | None = Field(
        default=None,
        description=(
            "Workload-specific JWT key id advertised by the NeMo auth service workload identity JWKS endpoint. "
            "When unset, workload token exchange uses auth.token_signing.key_id."
        ),
    )

    workload_token_private_key_file: str | None = Field(
        default=None,
        description=(
            "Path to a PEM-encoded RSA private key used by the NeMo auth service to sign workload identity "
            "access tokens. Intended for mounted shared secrets."
        ),
    )

    workload_allowed_audiences: list[str] = Field(
        default_factory=list,
        description=(
            "Additional RFC 8693 audience values accepted by the NeMo auth service workload token exchange endpoint. "
            "The configured workload_audience is always accepted."
        ),
    )

    workload_subject_jwks_uri: str | None = Field(
        default=None,
        description=(
            "JWKS URI used by the NeMo auth service to validate JWT subject tokens for workload token exchange. "
            "Leave unset when only Kubernetes TokenReview subject validation is enabled."
        ),
    )

    workload_subject_issuers: list[str] = Field(
        default_factory=list,
        description=(
            "Allowed JWT subject token issuers for workload token exchange. "
            "Required when workload_subject_jwks_uri is set."
        ),
    )

    workload_subject_jwks_cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description=("TTL in seconds for caching workload subject JWKS responses. Set to 0 to disable caching."),
    )

    workload_kubernetes_token_review_enabled: bool = Field(
        default=False,
        description=(
            "Allow the NeMo auth service workload token exchange endpoint to validate Kubernetes projected "
            "service account subject tokens using the TokenReview API."
        ),
    )

    scope_prefix: str | None = Field(
        default=None,
        description="Prefix to strip from token scopes before authorization. "
        "For example, if IdP returns 'api://my-app/models:read', set prefix to "
        "'api://my-app/' to normalize to 'models:read'. "
        "If not set, scopes are used as-is.",
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
        description="Shared issuer for NeMo Platform-minted JWTs. Defaults to <platform.base_url>/apis/auth.",
    )
    key_id: str = Field(
        default="nemo-platform-signing",
        description="Shared JWT key id advertised by NeMo Platform JWKS endpoints.",
    )
    private_key_file: str | None = Field(
        default=None,
        description=(
            "Path to the PEM-encoded RSA private key used by the auth service to sign NeMo Platform-minted JWTs. "
            "Intended for mounted shared secrets."
        ),
    )


def _default_access_key_accepted_formats() -> list[Literal["jwt"]]:
    return ["jwt"]


AccessKeyAcceptedFormat = Literal["jwt"]


class AccessKeyConfig(BaseSettings):
    """NeMo Platform Scoped Access Key configuration."""

    enabled: bool = Field(
        default=False,
        description="Enable NeMo Platform Scoped Access Key creation and validation.",
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
        default="nemo-platform-access-key",
        description="Expected audience for NeMo Platform Scoped Access Key JWTs.",
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
    def _parse_nullable_expiry(value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    @field_validator("accepted_formats", mode="before")
    @classmethod
    def parse_accepted_formats(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        return [part.strip() for part in value.split(",") if part.strip()]

    @field_validator("default_expires_in_seconds", "max_expires_in_seconds", mode="before")
    @classmethod
    def parse_nullable_expiry(cls, value: Any) -> Any:
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
            "build it automatically from a local NeMo Platform source checkout. Packaged deployments "
            "should include policy.wasm at build time and can disable this for fail-fast startup."
        ),
    )

    oidc: OIDCConfig = Field(
        default_factory=OIDCConfig,
        description="OIDC configuration for native token validation.",
    )

    token_signing: TokenSigningConfig = Field(
        default_factory=TokenSigningConfig,
        description="Shared token signing configuration for NeMo Platform-minted JWTs.",
    )

    access_keys: AccessKeyConfig = Field(
        default_factory=AccessKeyConfig,
        description="Scoped Access Key configuration.",
    )

    @model_validator(mode="after")
    def validate_workload_token_signing_key_id(self) -> Self:
        if not self.oidc.workload_token_exchange_enabled:
            return self

        key_id = self.oidc.workload_token_key_id or self.token_signing.key_id
        if not key_id or not key_id.strip():
            raise ValueError(
                "auth.oidc.workload_token_key_id or auth.token_signing.key_id must be configured "
                "when auth.oidc.workload_token_exchange_enabled is true"
            )
        workload_private_key_file = self._normalized_private_key_file(self.oidc.workload_token_private_key_file)
        access_key_private_key_file = self._normalized_private_key_file(self.token_signing.private_key_file)
        if (
            self.access_keys.enabled
            and workload_private_key_file
            and access_key_private_key_file
            and workload_private_key_file != access_key_private_key_file
            and key_id.strip() == self.token_signing.key_id.strip()
        ):
            raise ValueError(
                "auth.oidc.workload_token_key_id must be distinct from auth.token_signing.key_id "
                "when auth.oidc.workload_token_private_key_file differs from "
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
        # PlatformConfig from nmp.common.config, which is defined in this file.
        from nmp.common.platform_endpoint import parse_platform_endpoint

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


def get_platform_config() -> PlatformConfig:
    return Configuration.get_service_config(PlatformConfig)


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
      defaults to a SQLite database under the NeMo Platform user data directory
      (see ``nmp_user_data_dir``) — typically
      ``~/.local/share/nemo/nmp-platform.db`` — for local development
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
            db_path = nmp_user_data_dir() / "nmp-platform.db"
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
