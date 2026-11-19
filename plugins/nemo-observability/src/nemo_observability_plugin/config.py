# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the observability plugin.

Helix never hosts the observability platform: each connection points at an
instance the customer already runs. Operators declare connections in the
platform config file::

    observability:
      connections:
        - name: mlflow
          provider: mlflow
          base_url: http://localhost:5000
          default_project: "0"        # MLflow experiment id

or as JSON in ``NEMO_OBSERVABILITY_CONNECTIONS``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from nemo_helix_plugin.config import NemoConfig
from pydantic import BaseModel, Field, SecretStr, field_validator

ProviderName = Literal["mlflow", "phoenix"]


class ObservabilityConnection(BaseModel):
    """One external trace explorer the plugin proxies and embeds."""

    name: str = Field(
        pattern=r"^[a-z][a-z0-9-]{0,62}$",
        description="Connection key. Fixes the proxy prefix /apis/observability/ui/<name>, "
        "which the upstream must be configured to serve under.",
    )
    provider: ProviderName
    base_url: str = Field(
        pattern=r"^https?://[^/?#\s]+$",
        description="Origin the platform uses to reach the upstream UI, e.g. http://localhost:5000. No path.",
    )
    display_name: str | None = Field(default=None, description="Label shown in Studio; defaults to the provider.")
    otlp_endpoint: str | None = Field(
        default=None,
        description="OTLP/HTTP traces URL agents export to, when it differs from what the adapter derives "
        "from base_url (e.g. agents run in containers and reach the upstream by another host name).",
    )
    default_project: str | None = Field(
        default=None,
        description="Where exported traces land and the UI opens: MLflow experiment id, Phoenix project name.",
    )
    secret_env: str | None = Field(
        default=None,
        pattern=r"^[A-Z_][A-Z0-9_]*$",
        description="Environment variable holding the upstream credential the proxy injects "
        "(MLflow basic auth 'user:password', Phoenix API key). Never sent to the browser.",
    )


class ObservabilityConfig(NemoConfig):
    plugin_name: ClassVar[str] = "observability"
    plugin_description: ClassVar[str] = "External trace explorer connections embedded in NeMo Studio."

    connections: list[ObservabilityConnection] = Field(default_factory=list)
    session_ttl_seconds: int = Field(
        default=900,
        ge=60,
        le=86400,
        description="Lifetime of the cookie that authorizes the embedded UI; Studio re-mints before expiry.",
    )
    session_secret: SecretStr | None = Field(
        default=None,
        description="HMAC key for UI session cookies. Unset generates a per-process key, so sessions do not "
        "survive restarts or span replicas; set it when running more than one platform replica.",
    )
    upstream_timeout_seconds: float = Field(default=300.0, gt=0, description="Read timeout for proxied requests.")

    @field_validator("connections")
    @classmethod
    def _unique_names(cls, connections: list[ObservabilityConnection]) -> list[ObservabilityConnection]:
        names = [connection.name for connection in connections]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate observability connection names: {duplicates}")
        return connections

    def connection(self, name: str) -> ObservabilityConnection | None:
        return next((connection for connection in self.connections if connection.name == name), None)
