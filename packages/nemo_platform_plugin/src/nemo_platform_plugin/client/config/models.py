# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-facing configuration models for NemoClient.

These models represent the ``~/.config/nmp/config.yaml`` file used by the CLI
and SDK to store clusters, users (with auth tokens), and contexts.

This is distinct from the server-side config in
:mod:`nemo_platform_plugin.config` which reads ``/etc/nmp/config.yaml``.
"""

from __future__ import annotations

from abc import ABC
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    HttpUrl,
    SecretStr,
    SerializationInfo,
    Tag,
    field_serializer,
    field_validator,
    model_validator,
)

DEFAULT_WORKSPACE = "default"
DEFAULT_CONTEXT = "default"
DEFAULT_BASE_URL = "http://localhost:8080"

# CLI-only type aliases kept as plain strings for config round-tripping.
OutputFormat = Literal["table", "json", "yaml", "markdown", "csv", "raw"]
TimestampFormat = Literal["relative", "iso8601"]


class BaseUser(BaseModel, ABC):
    def get_client_config(self) -> dict[str, object]:
        return {}


class OAuthUser(BaseUser):
    """User with OAuth token authentication (from device flow, browser flow, etc.)."""

    type: Literal["oauth"] = "oauth"

    name: str = Field(..., min_length=1, description="Unique user name")
    token: SecretStr = Field(..., min_length=1, description="Access token (JWT)")
    refresh_token: SecretStr | None = Field(default=None, description="Refresh token for automatic renewal")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError("User name cannot be empty")
        return v

    @field_serializer("token", "refresh_token", when_used="json")
    def serialize_tokens(self, value: SecretStr | None, info: SerializationInfo) -> str | None:
        if value is None:
            return None
        if info.context and info.context.get("include_secrets"):
            return value.get_secret_value()
        return "***REDACTED***"

    def get_client_config(self) -> dict[str, object]:
        return {
            "default_headers": {
                "Authorization": f"Bearer {self.token.get_secret_value()}",
            }
        }


class NoAuthUser(BaseUser):
    """User with no authentication."""

    type: Literal["no-auth"] = "no-auth"

    name: str = Field(..., min_length=1, description="Unique user name")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError("User name cannot be empty")
        return v


def _get_user_type(data: Any) -> str:
    default = "no-auth"
    if isinstance(data, dict):
        return data.get("type", default)
    return getattr(data, "type", default)


User = Annotated[
    Annotated[OAuthUser, Tag("oauth")] | Annotated[NoAuthUser, Tag("no-auth")],
    Discriminator(_get_user_type),
]


class Cluster(BaseModel):
    """Cluster configuration — connection info only."""

    name: str = Field(..., min_length=1, description="Unique cluster name")
    base_url: HttpUrl = Field(..., description="Base URL for the cluster")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional cluster metadata")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError("Cluster name cannot be empty")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Cluster name must be alphanumeric (hyphens and underscores allowed)")
        return v


class Preferences(BaseModel):
    """User preferences for CLI/SDK behavior — can be overridden at runtime."""

    output_format: OutputFormat = Field(default="table", description="Default output format")
    timestamp_format: TimestampFormat = Field(default="iso8601", description="Timestamp display format")
    truncate: bool = Field(default=True, description="Truncate long strings in output")
    color_output: bool = Field(default=True, description="Enable colored output")


class ContextDefinition(BaseModel):
    """Context definition from the config file (serialization model).

    Contains string references to cluster and user names that must be
    resolved at runtime via :meth:`Config.resolve`.
    """

    name: str = Field(..., min_length=1, description="Unique context name")
    cluster: str = Field(..., min_length=1, description="Reference to cluster name")
    user: str = Field(..., min_length=1, description="Reference to user name")
    workspace: str | None = Field(default=None, description="Default workspace")
    default_model: str | None = Field(default=None, description="Default model entity ID for inference")
    fast_model: str | None = Field(default=None, description="Low-latency model entity ID for agent workloads")
    preferences: Preferences = Field(default_factory=Preferences, description="Context-specific preferences")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")

    @field_validator("name", "cluster", "user")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError("Identifier cannot be empty")
        return v


class ConfigParams(TypedDict, total=False):
    """Type-safe configuration parameters for overrides."""

    current_context: str
    base_url: str
    access_token: str | None
    refresh_token: str | None
    workspace: str
    default_model: str
    fast_model: str
    output_format: OutputFormat
    timestamp_format: TimestampFormat
    truncate: bool
    color_output: bool


class LocalServicesConfig(BaseModel):
    """User-selected paths for ``nemo services run`` local state."""

    data_dir: str | None = Field(default=None, description="Directory for local service state")


class ConfigFile(BaseModel):
    """Root configuration from the YAML config file."""

    current_context: str | None = Field(default=None, description="Name of the currently active context")
    clusters: list[Cluster] = Field(default_factory=list, description="List of configured clusters")
    users: list[User] = Field(default_factory=list, description="List of configured users")
    contexts: list[ContextDefinition] = Field(default_factory=list, description="List of context definitions")
    local_services: LocalServicesConfig | None = Field(default=None, description="Local services config")

    def ensure_context(
        self,
        context_name: str,
        params: ConfigParams,
    ) -> tuple[Cluster, User, ContextDefinition]:
        """Find or create cluster/user/context entities with the given name."""
        existing_context = next((c for c in self.contexts if c.name == context_name), None)

        cluster_name = existing_context.cluster if existing_context else f"{context_name}-cluster"
        user_name = existing_context.user if existing_context else f"{context_name}-user"

        # Find existing or create cluster
        cluster = next((c for c in self.clusters if c.name == cluster_name), None)
        if cluster is None:
            base_url = params.get("base_url")
            if not base_url:
                raise ValueError(f"Cluster '{cluster_name}' does not exist and no base_url provided to create it!")
            cluster = Cluster(name=cluster_name, base_url=HttpUrl(base_url))
            self.clusters.append(cluster)
        elif "base_url" in params:
            cluster.base_url = HttpUrl(params["base_url"])

        # Find existing or create user
        user = next((u for u in self.users if u.name == user_name), None)
        access_token_provided = "access_token" in params
        refresh_token_provided = "refresh_token" in params
        access_token = params.get("access_token")
        refresh_token = params.get("refresh_token")

        if user is None:
            if access_token:
                user = OAuthUser(
                    name=user_name,
                    token=SecretStr(access_token),
                    refresh_token=SecretStr(refresh_token) if refresh_token else None,
                )
            else:
                user = NoAuthUser(name=user_name)
            self.users.append(user)
        elif access_token_provided:
            idx = next(i for i, u in enumerate(self.users) if u.name == user_name)
            if access_token:
                user = OAuthUser(
                    name=user_name,
                    token=SecretStr(access_token),
                    refresh_token=SecretStr(refresh_token) if refresh_token else None,
                )
            else:
                user = NoAuthUser(name=user_name)
            self.users[idx] = user
        elif isinstance(user, OAuthUser) and refresh_token_provided:
            idx = next(i for i, u in enumerate(self.users) if u.name == user_name)
            user = OAuthUser(
                name=user_name,
                token=user.token,
                refresh_token=SecretStr(refresh_token) if refresh_token else None,
            )
            self.users[idx] = user

        # Find existing or create context
        context = existing_context
        if context is None:
            context = ContextDefinition(
                name=context_name,
                cluster=cluster_name,
                user=user_name,
                workspace=params.get("workspace", DEFAULT_WORKSPACE),
                preferences=Preferences(),
            )
            self.contexts.append(context)

        # Apply context-level params
        if "workspace" in params:
            context.workspace = params["workspace"]
        if "default_model" in params:
            context.default_model = params["default_model"]
        if "fast_model" in params:
            context.fast_model = params["fast_model"]
        if "output_format" in params:
            context.preferences.output_format = params["output_format"]
        if "timestamp_format" in params:
            context.preferences.timestamp_format = params["timestamp_format"]
        if "truncate" in params:
            context.preferences.truncate = params["truncate"]
        if "color_output" in params:
            context.preferences.color_output = params["color_output"]

        return cluster, user, context

    @model_validator(mode="after")
    def validate_references(self) -> ConfigFile:
        cluster_names = {cluster.name for cluster in self.clusters}
        user_names = {user.name for user in self.users}
        context_names = {context.name for context in self.contexts}

        for context in self.contexts:
            if context.cluster not in cluster_names:
                raise ValueError(f"Context '{context.name}' references non-existent cluster '{context.cluster}'")
            if context.user not in user_names:
                raise ValueError(f"Context '{context.name}' references non-existent user '{context.user}'")

        if self.current_context and self.contexts and self.current_context not in context_names:
            raise ValueError(f"Current context '{self.current_context}' does not exist")

        return self

    @model_validator(mode="after")
    def validate_unique_names(self) -> ConfigFile:
        cluster_names = [cluster.name for cluster in self.clusters]
        if len(cluster_names) != len(set(cluster_names)):
            raise ValueError("Cluster names must be unique")

        user_names = [user.name for user in self.users]
        if len(user_names) != len(set(user_names)):
            raise ValueError("User names must be unique")

        context_names = [context.name for context in self.contexts]
        if len(context_names) != len(set(context_names)):
            raise ValueError("Context names must be unique")

        return self


class Context(BaseModel):
    """Resolved runtime configuration (what application code should use).

    Unlike :class:`ContextDefinition` (a serialization model from the config file),
    this contains fully resolved objects ready for immediate use.
    """

    context_name: str = Field(..., description="Active context name")
    cluster: Cluster = Field(..., description="Fully resolved cluster configuration")
    user: User | None = Field(default=None, description="Resolved user with authentication credentials")
    workspace: str = Field(..., description="Active workspace")
    default_model: str | None = Field(default=None, description="Default model entity ID for inference")
    fast_model: str | None = Field(default=None, description="Low-latency model entity ID for agent workloads")
    preferences: Preferences = Field(..., description="Effective preferences")
