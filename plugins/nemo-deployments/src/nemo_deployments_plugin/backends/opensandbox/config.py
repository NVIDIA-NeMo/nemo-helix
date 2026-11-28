# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor-level OpenSandbox configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from nemo_deployments_plugin.backends.opensandbox.endpoint import OpenSandboxEndpointTransport
from nemo_deployments_plugin.endpoint_transport import EndpointTransport
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

HostPort = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?(:[0-9]{1,5})?$")]
EnvVarName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Host = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$")]


class PlatformEgressConfig(BaseModel):
    """NeMo Helix endpoint a sandbox is allowed to reach."""

    model_config = ConfigDict(extra="forbid")

    host: Host = Field(description="Platform host as seen from a sandbox: an IP address or a DNS name.")
    port: int = Field(default=8080, ge=1, le=65535, description="Platform port.")
    scheme: Literal["http", "https"] = Field(default="http", description="Scheme of the platform endpoint.")

    def url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"


class OpenSandboxExecutorConfig(BaseModel):
    """Knobs for a named opensandbox executor instance."""

    model_config = ConfigDict(extra="forbid")

    domain: HostPort = Field(description="OpenSandbox server as host[:port], with no scheme.")
    gateway_address: HostPort = Field(
        description="OpenSandbox ingress gateway as host[:port], with no scheme. Requires header route mode.",
    )
    protocol: Literal["http", "https"] = Field(
        default="http", description="Scheme for the OpenSandbox server and ingress gateway."
    )
    api_key_env: EnvVarName = Field(
        default="OPEN_SANDBOX_API_KEY",
        description="Environment variable that holds the OpenSandbox API key.",
    )
    platform_egress: PlatformEgressConfig | None = Field(
        default=None,
        description=(
            "Platform endpoint sandboxes reach. Sandbox egress is denied except to this endpoint. "
            "Null denies all egress. The OpenSandbox server enforces this only when its egress mode is dns+nft."
        ),
    )
    default_resources: dict[str, str] = Field(
        default_factory=lambda: {"cpu": "1", "memory": "2Gi"},
        min_length=1,
        description="Sandbox resource limits for a deployment whose container sets none.",
    )
    request_timeout_seconds: float = Field(
        default=60.0, gt=0, description="Timeout for calls to the OpenSandbox server and to execd."
    )
    create_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        description=(
            "Timeout for the create-sandbox call. The OpenSandbox server holds the call until the sandbox pod "
            "is running, for up to its kubernetes.sandbox_create_timeout_seconds, and then reports why the "
            "pod did not start. Set this longer than that server setting."
        ),
    )

    def service_transport(self) -> EndpointTransport:
        """Transport for sandbox endpoints advertised by this executor."""
        return OpenSandboxEndpointTransport(
            connect_base=f"{self.protocol}://{self.gateway_address}",
            server_base=f"{self.protocol}://{self.domain}",
            api_key_env=self.api_key_env,
        )
