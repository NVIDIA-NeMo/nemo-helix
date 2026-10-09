# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderName = Literal["authentik", "zitadel"]
DeploymentRuntime = Literal["compose", "kind", "k3d"]


class PortForwardContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pid_file: Path
    log_file: Path
    kubernetes_context: str
    namespace: str
    service: str = "nemo-helix-envoy"
    local_port: int = Field(ge=1, le=65535)
    remote_port: int = Field(default=8080, ge=1, le=65535)


class AuthIdpDeploymentContext(BaseModel):
    """Immutable, non-secret facts for a deployed auth IdP test environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    provider: ProviderName
    runtime: DeploymentRuntime
    runtime_id: str
    gateway_url: str
    workload_gateway_url: str
    provider_discovery_url: str
    provider_token_endpoint: str
    ca_bundle: Path
    provider_manifest: Path
    capabilities: tuple[str, ...]
    namespace: str | None = None
    credential_env_files: tuple[Path, ...] = ()
    workload_token_file: Path | None = None
    deployment_ca_file: str
    port_forward: PortForwardContext | None = None

    @field_validator("gateway_url", "workload_gateway_url", "provider_discovery_url", "provider_token_endpoint")
    @classmethod
    def validate_https_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("must be an absolute HTTPS URL")
        if parsed.query or parsed.fragment:
            raise ValueError("must not contain a query string or fragment")
        return value.rstrip("/") if parsed.path in {"", "/"} else value

    @model_validator(mode="after")
    def validate_runtime_boundary(self) -> Self:
        if self.runtime == "compose" and self.port_forward is not None:
            raise ValueError("compose context must not contain a Kubernetes port-forward")
        if self.runtime != "compose" and self.port_forward is None:
            raise ValueError("Kubernetes context requires port-forward ownership metadata")
        return self

    @property
    def backend(self) -> Literal["compose", "kubernetes"]:
        return "compose" if self.runtime == "compose" else "kubernetes"


def load_deployment_context(path: Path) -> AuthIdpDeploymentContext:
    context = AuthIdpDeploymentContext.model_validate_json(path.read_text(encoding="utf-8"))
    for required_path in (context.ca_bundle, context.provider_manifest, *context.credential_env_files):
        if not required_path.is_file():
            raise ValueError(f"deployment context references missing file: {required_path}")
    if context.workload_token_file is not None and not context.workload_token_file.is_file():
        raise ValueError(f"deployment context references missing workload token: {context.workload_token_file}")
    return context


def load_credential_environment(paths: tuple[Path, ...]) -> None:
    for path in paths:
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if not separator or not name:
                raise ValueError(f"invalid credential environment entry at {path}:{line_number}")
            os.environ[name] = value


def _runtime_capabilities(manifest: Path, runtime_id: str) -> tuple[str, ...]:
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    runtimes = document.get("test_runtimes", [])
    matches = [runtime for runtime in runtimes if runtime.get("id") == runtime_id]
    if len(matches) != 1:
        raise ValueError(f"expected one {runtime_id!r} runtime in {manifest}; found {len(matches)}")
    capabilities = matches[0].get("capabilities", [])
    if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
        raise ValueError(f"runtime {runtime_id!r} has invalid capabilities")
    return tuple(capabilities)


def write_deployment_context(args: argparse.Namespace) -> None:
    gateway_url = args.gateway_url.rstrip("/")
    gateway_port = urlsplit(gateway_url).port
    if gateway_port is None:
        raise ValueError("gateway URL must include the selected external port")
    provider_paths = {
        "authentik": ("/application/o/nemo/.well-known/openid-configuration", "/application/o/token/"),
        "zitadel": ("/.well-known/openid-configuration", "/oauth/v2/token"),
    }
    discovery_path, token_path = provider_paths[args.provider]
    port_forward = None
    if args.runtime != "compose":
        port_forward = PortForwardContext(
            pid_file=args.port_forward_pid_file,
            log_file=args.port_forward_log_file,
            kubernetes_context=args.kubernetes_context,
            namespace=args.namespace,
            local_port=gateway_port,
        )
    context = AuthIdpDeploymentContext(
        provider=args.provider,
        runtime=args.runtime,
        runtime_id=args.runtime_id,
        gateway_url=gateway_url,
        workload_gateway_url=args.workload_gateway_url,
        provider_discovery_url=f"{gateway_url}{discovery_path}",
        provider_token_endpoint=f"{gateway_url}{token_path}",
        ca_bundle=args.ca_bundle.resolve(),
        provider_manifest=args.provider_manifest.resolve(),
        capabilities=_runtime_capabilities(args.provider_manifest, args.runtime_id),
        namespace=args.namespace,
        credential_env_files=tuple(path.resolve() for path in args.credential_env_file),
        workload_token_file=args.workload_token_file.resolve() if args.workload_token_file else None,
        deployment_ca_file=args.deployment_ca_file,
        port_forward=port_forward,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(json.dumps(context.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a host auth-idp deployment context")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("authentik", "zitadel"), required=True)
    parser.add_argument("--runtime", choices=("compose", "kind", "k3d"), required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--workload-gateway-url", required=True)
    parser.add_argument("--ca-bundle", type=Path, required=True)
    parser.add_argument("--provider-manifest", type=Path, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--credential-env-file", type=Path, action="append", default=[])
    parser.add_argument("--workload-token-file", type=Path)
    parser.add_argument("--deployment-ca-file", required=True)
    parser.add_argument("--port-forward-pid-file", type=Path)
    parser.add_argument("--port-forward-log-file", type=Path)
    parser.add_argument("--kubernetes-context")
    args = parser.parse_args()
    if args.runtime != "compose" and not all(
        (args.namespace, args.port_forward_pid_file, args.port_forward_log_file, args.kubernetes_context)
    ):
        parser.error("Kubernetes contexts require namespace, context, and port-forward PID/log files")
    write_deployment_context(args)


if __name__ == "__main__":
    main()
