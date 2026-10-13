# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.auth_idp.deployment_context import (
    AuthIdpDeploymentContext,
    DeploymentRuntime,
    PortForwardContext,
    load_credential_environment,
    load_deployment_context,
    write_deployment_context,
)
from tests.auth_idp.providers import load_provider_config
from tests.auth_idp.runtime_host import HostAuthIdpRuntime

pytestmark = [pytest.mark.auth_idp]


def test_deployment_context_enforces_runtime_boundary(tmp_path: Path) -> None:
    port_forward = _port_forward(tmp_path)

    with pytest.raises(ValidationError, match="compose context must not contain"):
        _context(tmp_path, runtime="compose", port_forward=port_forward)
    with pytest.raises(ValidationError, match="requires port-forward ownership metadata"):
        _context(tmp_path, runtime="kind", port_forward=None)


def test_deployment_context_writer_requires_explicit_gateway_port(tmp_path: Path) -> None:
    manifest = Path("contrib/auth/authentik/manifest.yaml")
    arguments = argparse.Namespace(
        output=tmp_path / "context.json",
        provider="authentik",
        runtime="compose",
        runtime_id="authentik-compose",
        gateway_url="https://127.0.0.1",
        workload_gateway_url="https://nemo-gateway:8080",
        ca_bundle=tmp_path / "ca.crt",
        provider_manifest=manifest,
        namespace=None,
        credential_env_file=[],
        workload_token_file=None,
        deployment_ca_file="/etc/nhx/gateway-tls/tls.crt",
        port_forward_pid_file=None,
        port_forward_log_file=None,
        kubernetes_context=None,
    )

    with pytest.raises(ValueError, match="must include the selected external port"):
        write_deployment_context(arguments)


def test_load_deployment_context_rejects_missing_files(tmp_path: Path) -> None:
    context = _context(tmp_path, runtime="compose", port_forward=None)
    context_file = tmp_path / "context.json"
    context_file.write_text(context.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="references missing file"):
        load_deployment_context(context_file)


def test_host_runtime_keeps_provider_resolved_interactive_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential_file = tmp_path / "credentials.env"
    credential_file.write_text("AUTHENTIK_INTERACTIVE_USER_PASSWORD=authentik-secret\n", encoding="utf-8")
    context = _context(tmp_path, runtime="compose", port_forward=None).model_copy(
        update={"credential_env_files": (credential_file,)}
    )
    monkeypatch.setenv("ZITADEL_INTERACTIVE_USER_PASSWORD", "unrelated-zitadel-secret")

    runtime = HostAuthIdpRuntime(context)

    assert runtime.provider.interactive_user_password == "authentik-secret"


def test_collection_can_resolve_credentials_before_loading_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential_file = tmp_path / "credentials.env"
    credential_file.write_text("AUTHENTIK_INTERACTIVE_USER_PASSWORD=authentik-secret\n", encoding="utf-8")
    monkeypatch.delenv("AUTHENTIK_INTERACTIVE_USER_PASSWORD", raising=False)

    load_credential_environment((credential_file,))
    provider = load_provider_config(Path("contrib/auth/authentik/manifest.yaml"))

    assert provider.interactive_user_password == "authentik-secret"


def test_host_runtime_targets_workload_reachable_gateway(tmp_path: Path) -> None:
    runtime = HostAuthIdpRuntime(_context(tmp_path, runtime="compose", port_forward=None))

    config = runtime.deployment_workload_runtime_config()

    assert config.env == (
        {"name": "NHX_BASE_URL", "value": "https://nemo-gateway:8080"},
        {"name": "NHX_CLIENT_SSL_CERT_FILE", "value": "/etc/nhx/gateway-tls/tls.crt"},
        {"name": "SSL_CERT_FILE", "value": "/etc/nhx/gateway-tls/tls.crt"},
        {"name": "REQUESTS_CA_BUNDLE", "value": "/etc/nhx/gateway-tls/tls.crt"},
    )


def test_host_runtime_accepts_exact_owned_port_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port_forward = _port_forward(tmp_path)
    context = _context(tmp_path, runtime="kind", port_forward=port_forward)
    port_forward.pid_file.write_text("1234\n", encoding="utf-8")
    runtime = HostAuthIdpRuntime(context)
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "kubectl --kubeconfig /tmp/test --context kind-auth-idp "
                "-n nemo-auth port-forward svc/nemo-helix-envoy 19082:8080"
            ),
        ),
    )

    runtime.assert_available()


def test_host_runtime_rejects_pid_owned_by_another_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port_forward = _port_forward(tmp_path)
    context = _context(tmp_path, runtime="kind", port_forward=port_forward)
    port_forward.pid_file.write_text("1234\n", encoding="utf-8")
    runtime = HostAuthIdpRuntime(context)
    monkeypatch.setattr(os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout="kubectl --context kind-other -n other port-forward svc/other 19082:8080",
        ),
    )

    with pytest.raises(AssertionError, match="does not match deployment context"):
        runtime.assert_available()


def test_host_runtime_includes_port_forward_log_when_process_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    port_forward = _port_forward(tmp_path)
    port_forward.pid_file.write_text("1234\n", encoding="utf-8")
    port_forward.log_file.write_text("address already in use\n", encoding="utf-8")
    runtime = HostAuthIdpRuntime(_context(tmp_path, runtime="kind", port_forward=port_forward))

    def process_missing(pid: int, signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", process_missing)

    with pytest.raises(AssertionError, match="address already in use"):
        runtime.assert_available()


def _port_forward(tmp_path: Path) -> PortForwardContext:
    return PortForwardContext(
        pid_file=tmp_path / "port-forward.pid",
        log_file=tmp_path / "port-forward.log",
        kubernetes_context="kind-auth-idp",
        namespace="nemo-auth",
        local_port=19082,
    )


def _context(
    tmp_path: Path,
    *,
    runtime: DeploymentRuntime,
    port_forward: PortForwardContext | None,
) -> AuthIdpDeploymentContext:
    return AuthIdpDeploymentContext.model_validate(
        {
            "provider": "authentik",
            "runtime": runtime,
            "runtime_id": "authentik-compose" if runtime == "compose" else "authentik-kubernetes",
            "gateway_url": "https://127.0.0.1:19082",
            "workload_gateway_url": (
                "https://nemo-gateway:8080"
                if runtime == "compose"
                else "https://nemo-helix-envoy.nemo-auth.svc.cluster.local:8080"
            ),
            "provider_discovery_url": ("https://127.0.0.1:19082/application/o/nemo/.well-known/openid-configuration"),
            "provider_token_endpoint": "https://127.0.0.1:19082/application/o/token/",
            "ca_bundle": tmp_path / "missing-ca.crt",
            "provider_manifest": Path("contrib/auth/authentik/manifest.yaml").resolve(),
            "capabilities": ["gateway_discovery"],
            "deployment_ca_file": (
                "/etc/nhx/gateway-tls/tls.crt" if runtime == "compose" else "/etc/nhx/workload-token-ca/ca.crt"
            ),
            "namespace": "nemo-auth" if runtime != "compose" else None,
            "port_forward": port_forward,
        }
    )
