# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor-level OpenShell backend configuration.

These knobs configure a named ``openshell`` executor instance (how to reach the
OpenShell gateway), not per-deployment entity ``backend_config``.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OpenShellTLSConfig(BaseModel):
    """mTLS material for an https gateway. Unused for local plaintext gateways."""

    model_config = ConfigDict(extra="forbid")

    ca_cert_path: str | None = Field(default=None, description="Path to the CA bundle that signed the gateway cert.")
    client_cert_path: str | None = Field(default=None, description="Path to the client certificate (mTLS).")
    client_key_path: str | None = Field(default=None, description="Path to the client private key (mTLS).")


class HelixEgressConfig(BaseModel):
    """The NeMo Helix endpoint a sandbox must always be able to reach.

    Environment-specific: on the docker driver a sandbox reaches the platform at
    host.docker.internal:8080; an in-cluster (k8s) driver uses the platform Service
    address. This becomes the single allowed rule in a generated default-deny
    ``SandboxPolicy`` and is re-injected into any static or
    user-supplied policy so a deployment can never lose its path home.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="host.docker.internal", description="Platform host reachable from inside a sandbox.")
    port: int = Field(default=8080, ge=1, description="Platform port (the inference gateway / API listener).")
    # Value sets track openshell/proto/sandbox.proto (NetworkAccessRule), which is broader than
    # the summaries above imply: protocol also allows "graphql" and "" (L4-only), tls allows
    # "passthrough".
    protocol: Literal["rest", "websocket", "graphql", "sql", ""] = Field(
        default="rest",
        description='OpenShell L7 protocol: "rest", "websocket", "graphql", "sql", or "" for L4-only.',
    )
    tls: Literal["terminate", "passthrough", ""] = Field(
        default="",
        description='TLS handling: "terminate" for HTTPS L7 intercept, "passthrough" (or "") for no L7 interception.',
    )
    access: Literal["read-only", "read-write", "full"] = Field(
        default="full",
        description='OpenShell access preset: "read-only", "read-write", or "full".',
    )
    binaries: list[str] = Field(
        default_factory=list,
        description=(
            "Binaries permitted to open the egress connection (the venv/uv python that makes LLM "
            "calls, not curl). Empty uses the backend default set."
        ),
    )


class OpenShellExecutorConfig(BaseModel):
    """Knobs for a named openshell executor instance."""

    model_config = ConfigDict(extra="forbid")

    gateway_endpoint: str = Field(
        default="http://127.0.0.1:17670",
        description=(
            "OpenShell gateway endpoint as a URL (http://host:port or https://host:port). "
            "The gRPC target is the same host:port; http implies plaintext, https implies TLS."
        ),
    )
    insecure: bool | None = Field(
        default=None,
        description="Force plaintext (True) or TLS (False). When None, derived from the endpoint scheme.",
    )
    tls: OpenShellTLSConfig | None = Field(default=None, description="mTLS material for an https gateway.")
    request_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Per-RPC deadline for control-plane calls (create/get/expose/delete/logs).",
    )
    default_policy_path: str | None = Field(
        default=None,
        description=(
            "Path to a hand-written SandboxPolicy YAML applied to created sandboxes. When unset, a "
            "default-deny policy is generated from platform_egress + the sandbox filesystem defaults. "
            "When platform_egress is set it is injected as mandatory; set platform_egress: null to "
            "disable it entirely (e.g. gateway-managed inference via inference.local)."
        ),
    )
    platform_egress: HelixEgressConfig | None = Field(
        default_factory=HelixEgressConfig,
        description=(
            "Platform endpoint a sandbox reaches directly. Drives the generated default policy "
            "(when default_policy_path is unset) and is injected into every policy as a mandatory "
            "egress rule. Set to null to grant the sandbox NO direct egress, correct when the agent "
            "reaches models through the gateway-managed inference.local route, which needs no "
            "sandbox egress rule."
        ),
    )
    serve_workdir: str = Field(
        default="/home/sandbox",
        description=(
            "Working directory for the detached serve command. Must be writable by the sandbox "
            "user: a packaged agent image chowns /workspace to its own 'agent' user, so NAT's "
            "per-run temp dir is written here instead. Empty string disables the chdir."
        ),
    )
    serve_path: str = Field(
        default="/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        description=(
            "PATH injected into the sandbox environment so exec'd processes find "
            "image-installed binaries. The supervisor resets PATH to a system default, so an "
            "image's ENV PATH is not seen by exec'd children. Ignored when the deployment "
            "config already declares PATH."
        ),
    )
    serve_virtual_env: str = Field(
        default="/workspace/.venv",
        description=(
            "VIRTUAL_ENV injected into the sandbox environment. PATH alone only fixes the "
            "processes the supervisor execs directly; a child that re-launches an interpreter "
            "from sys.base_prefix (Fabric spawns its adapter host this way) escapes the venv "
            "and loses site-packages. Ignored when the deployment config already declares "
            "VIRTUAL_ENV; empty string disables the injection."
        ),
    )
    landlock_compatibility: Literal["best_effort", "hard_requirement"] = Field(
        default="best_effort",
        description=(
            "Landlock mode for the generated policy and for hand-written policies that omit a "
            "landlock block. 'best_effort' runs on kernels without Landlock but ships NO "
            "filesystem confinement there (fail-open); 'hard_requirement' fails the sandbox "
            "closed on such kernels (fail-closed) at the cost of breaking hosts without Landlock."
        ),
    )

    @model_validator(mode="after")
    def _validate_endpoint(self) -> OpenShellExecutorConfig:
        parsed = urlparse(self.gateway_endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("gateway_endpoint must be a URL like http://host:port or https://host:port")
        return self

    def grpc_target(self) -> str:
        """host:port for the gRPC channel, derived from the endpoint URL."""
        parsed = urlparse(self.gateway_endpoint)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return f"{host}:{port}"

    def use_insecure(self) -> bool:
        """Whether to open a plaintext channel (explicit override, else scheme-derived)."""
        if self.insecure is not None:
            return self.insecure
        return urlparse(self.gateway_endpoint).scheme == "http"
