# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for auth-idp pytest fixtures and fixture-only helper functions."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from cryptography import x509
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR

from e2e.services_pool import RunningServices
from tests.auth_idp import conftest, runtime
from tests.auth_idp.authentik_live import authentik_gateway_tls_ca_bundle, prepare_authentik_compose_inputs
from tests.auth_idp.common import jwt_claims, require_capability
from tests.auth_idp.conftest import _token_request_body
from tests.auth_idp.providers import ProviderConfig, load_provider_configs
from tests.auth_idp.runtime_contract import AuthIdpCase
from tests.auth_idp.xdist import append_xdist_group_suffix

pytestmark = [pytest.mark.auth_idp]


def test_authentik_stack_fixture_uses_pooled_gateway_metadata():
    provider = ProviderConfig(
        name="authentik",
        mode="compose-ci",
        compose_file=Path("docker-compose.yml"),
        gateway_base_url="https://127.0.0.1:18080",
        issuer_url="http://authentik-server:9000/application/o/nemo/",
        discovery_url="https://127.0.0.1:18080/application/o/nemo/.well-known/openid-configuration",
        token_endpoint="https://127.0.0.1:18080/application/o/token/",
        nemo_config=Path("config/platform-compose-authentik.yaml"),
        interactive_user_username="nemo-user",
        interactive_user_password="nemo-user-password-dev",
        interactive_user_expected_email="nemo-user@example.com",
        workload_principal_id="svc-nemo",
        workload_expected_groups=["nemo-workloads"],
        workload_audience="nemo-helix",
        workload_principal_claim="sub",
        workload_groups_claim="groups",
        workload_groups_format="comma_string",
        workload_token_env_vars=[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR],
        workload_forwarded_headers={
            "principal_id": "X-NHX-Principal-Id",
            "principal_groups": "X-NHX-Principal-Groups",
        },
        e2e_setup_password_grant={
            "grant_type": "password",
            "client_id": "nemo-helix",
            "username": "nemo-setup",
            "password": "nemo-setup-token-secret-dev",
            "scope": "openid email groups",
        },
        interactive_user_password_grant=None,
        workload_provider_password_grant={
            "grant_type": "password",
            "username": "svc-nemo",
            "password": "shared-secret",
        },
        healthchecks=[],
        startup_timeouts={},
    )
    fixture_fn = cast(Any, conftest.authentik_stack).__wrapped__
    stack = fixture_fn(None, provider, "https://127.0.0.1:28080")

    assert stack.gateway_base_url == "https://127.0.0.1:28080"
    assert stack.discovery_url == "https://127.0.0.1:28080/application/o/nemo/.well-known/openid-configuration"
    assert stack.token_endpoint == "https://127.0.0.1:28080/application/o/token/"
    assert stack.nemo_config == provider.nemo_config


def test_auth_idp_runtime_event_line_includes_compose_instance_metadata(tmp_path):
    case = AuthIdpCase(
        id="authentik-compose",
        provider=ProviderConfig(
            name="authentik",
            mode="compose-ci",
            compose_file=Path("docker-compose.yml"),
            gateway_base_url="https://127.0.0.1:18080",
            issuer_url="http://authentik-server:9000/application/o/nemo/",
            discovery_url="https://127.0.0.1:18080/application/o/nemo/.well-known/openid-configuration",
            token_endpoint="https://127.0.0.1:18080/application/o/token/",
            nemo_config=Path("config/platform-compose-authentik.yaml"),
            interactive_user_username="nemo-user",
            interactive_user_password="nemo-user-password-dev",
            interactive_user_expected_email="nemo-user@example.com",
            workload_principal_id="svc-nemo",
            workload_expected_groups=["nemo-workloads"],
            workload_audience="nemo-helix",
            workload_principal_claim="sub",
            workload_groups_claim="groups",
            workload_groups_format="comma_string",
            workload_token_env_vars=[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR],
            workload_forwarded_headers={},
            e2e_setup_password_grant=None,
            interactive_user_password_grant=None,
            workload_provider_password_grant=None,
            healthchecks=[],
            startup_timeouts={},
        ),
        backend="compose",
        capabilities=frozenset(),
    )
    services = RunningServices(
        url="https://127.0.0.1:49123",
        compose_project_name="authentik-e2e-abc123-deadbeef",
        log_path=tmp_path / "pytest.log",
        proc=None,
        config_path=None,
    )
    runtime_obj = SimpleNamespace(gateway_base_url="https://127.0.0.1:49123")

    summary = conftest._auth_idp_runtime_event_line("available", case, runtime_obj, services)

    assert summary == (
        "Auth-idp runtime available: id=authentik-compose backend=compose "
        "url=https://127.0.0.1:49123 port=49123 "
        f"compose_project=authentik-e2e-abc123-deadbeef log={tmp_path / 'pytest.log'}"
    )


def test_auth_idp_runtime_event_line_includes_kubernetes_instance_metadata():
    case = AuthIdpCase(
        id="authentik-kubernetes",
        provider=ProviderConfig(
            name="authentik",
            mode="kubernetes-ci",
            compose_file=None,
            gateway_base_url="https://127.0.0.1:18081",
            issuer_url="http://authentik-server:9000/application/o/nemo/",
            discovery_url="https://127.0.0.1:18081/application/o/nemo/.well-known/openid-configuration",
            token_endpoint="https://127.0.0.1:18081/application/o/token/",
            nemo_config=Path("config/platform-compose-authentik.yaml"),
            interactive_user_username="nemo-user",
            interactive_user_password="nemo-user-password-dev",
            interactive_user_expected_email="nemo-user@example.com",
            workload_principal_id="svc-nemo",
            workload_expected_groups=["nemo-workloads"],
            workload_audience="nemo-helix",
            workload_principal_claim="sub",
            workload_groups_claim="groups",
            workload_groups_format="comma_string",
            workload_token_env_vars=[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR],
            workload_forwarded_headers={},
            e2e_setup_password_grant=None,
            interactive_user_password_grant=None,
            workload_provider_password_grant=None,
            healthchecks=[],
            startup_timeouts={},
        ),
        backend="kubernetes",
        capabilities=frozenset(),
    )
    runtime_obj = SimpleNamespace(
        gateway_base_url="https://127.0.0.1:39001",
        cluster=SimpleNamespace(
            name="ci",
            context="kind-ci",
            runtime="kind",
            kubeconfig=Path("/tmp/nhx-authentik-kubeconfig.yaml"),
        ),
        namespace="nemo-authentik",
        helm_release="authentik-demo",
    )

    summary = conftest._auth_idp_runtime_event_line("teardown_complete", case, runtime_obj)

    assert summary == (
        "Auth-idp runtime teardown_complete: id=authentik-kubernetes backend=kubernetes "
        "url=https://127.0.0.1:39001 port=39001 "
        "cluster=ci context=kind-ci runtime=kind kubeconfig=/tmp/nhx-authentik-kubeconfig.yaml "
        "namespace=nemo-authentik helm_release=authentik-demo"
    )


def test_auth_idp_runtime_result_event_line_includes_status():
    case = AuthIdpCase(
        id="authentik-compose",
        provider=load_provider_configs()[0],
        backend="compose",
        capabilities=frozenset(),
    )

    summary = conftest._auth_idp_runtime_event_line("result", case, status="passed")

    assert summary == "Auth-idp runtime result: id=authentik-compose backend=compose status=passed"


def test_write_terminal_line_bypasses_pytest_capture(capsys):
    events: list[str] = []

    class FakeCaptureManager:
        def global_and_fixture_disabled(self):
            class CaptureDisabled:
                def __enter__(self):
                    events.append("enter")

                def __exit__(self, exc_type, exc, traceback):
                    events.append("exit")

            return CaptureDisabled()

    class FakePluginManager:
        def get_plugin(self, name: str):
            assert name == "capturemanager"
            return FakeCaptureManager()

    request = SimpleNamespace(config=SimpleNamespace(pluginmanager=FakePluginManager()))

    conftest._write_terminal_line(cast(Any, request), "Auth-idp runtime available: id=authentik-compose")

    assert events == ["enter", "exit"]
    assert capsys.readouterr().out == "Auth-idp runtime available: id=authentik-compose\n"


def test_token_request_body_for_password_grant_includes_username_and_password():
    assert _token_request_body(
        {
            "grant_type": "password",
            "client_id": "nemo-helix",
            "client_secret": "secret",
            "username": "akadmin",
            "password": "akadmin-dev",
            "scope": "openid profile email groups",
        }
    ) == {
        "grant_type": "password",
        "client_id": "nemo-helix",
        "client_secret": "secret",
        "username": "akadmin",
        "password": "akadmin-dev",
        "scope": "openid profile email groups",
    }


def test_token_request_body_for_workload_password_grant_includes_username_and_password():
    assert _token_request_body(
        {
            "grant_type": "password",
            "client_id": "nemo-helix",
            "client_secret": "secret",
            "username": "svc-nemo",
            "password": "shared-secret",
            "scope": "openid email groups",
        }
    ) == {
        "grant_type": "password",
        "client_id": "nemo-helix",
        "client_secret": "secret",
        "username": "svc-nemo",
        "password": "shared-secret",
        "scope": "openid email groups",
    }


def test_jwt_claims_decodes_payload() -> None:
    token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJwcm9qZWN0LXN1YmplY3QiLCJncm91cHMiOiJuZW1vLXdvcmtsb2FkcyJ9."

    claims = jwt_claims(token)

    assert claims["sub"] == "project-subject"
    assert claims["groups"] == "nemo-workloads"


def test_require_capability_skips_when_missing() -> None:
    case = AuthIdpCase(
        id="provider-smoke",
        provider=load_provider_configs()[0],
        backend="external",
        capabilities=frozenset({"gateway_discovery"}),
    )

    with pytest.raises(pytest.skip.Exception):
        require_capability(case, "workspace_rbac")


def test_prepare_authentik_compose_inputs_creates_generated_assets(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AUTHENTIK_BLUEPRINT_DIR", raising=False)
    monkeypatch.delenv("AUTHENTIK_GATEWAY_TLS_DIR", raising=False)
    source_blueprint = tmp_path / "helm/files/blueprints/nemo.yaml"
    source_blueprint.parent.mkdir(parents=True)
    source_blueprint.write_text("version: 1\n", encoding="utf-8")

    prepare_authentik_compose_inputs(root=tmp_path)

    generated = tmp_path / ".generated"
    assert (generated / "blueprints/nemo.yaml").read_text(encoding="utf-8") == "version: 1\n"
    key_path = generated / "workload-token-private-key.pem"
    assert key_path.read_text(encoding="utf-8").startswith("-----BEGIN RSA PRIVATE KEY-----")
    assert key_path.stat().st_mode & 0o777 == 0o600

    tls_cert = generated / "gateway-tls/tls.crt"
    tls_key = generated / "gateway-tls/tls.key"
    assert authentik_gateway_tls_ca_bundle(root=tmp_path) == tls_cert
    assert tls_key.stat().st_mode & 0o777 == 0o600
    assert tls_cert.stat().st_mode & 0o777 == 0o644
    assert "DNS.2 = nemo-gateway" in (generated / "gateway-tls/openssl.cnf").read_text(encoding="utf-8")
    cert = x509.load_pem_x509_certificate(tls_cert.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == ["localhost", "nemo-gateway"]
    assert [str(value) for value in san.get_values_for_type(x509.IPAddress)] == ["127.0.0.1"]


def test_authentik_gateway_tls_ca_bundle_honors_tls_dir_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTHENTIK_GATEWAY_TLS_DIR", "./custom-gateway-tls")

    assert authentik_gateway_tls_ca_bundle(root=tmp_path) == tmp_path / "custom-gateway-tls/tls.crt"


def test_authentik_docker_runtime_defaults_workload_identity_password(monkeypatch):
    monkeypatch.delenv(runtime.AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD_ENVVAR, raising=False)
    runtime.get_authentik_docker_test_runtime.cache_clear()
    try:
        try:
            provider = runtime.get_authentik_docker_test_runtime()
        finally:
            runtime.get_authentik_docker_test_runtime.cache_clear()

        assert os.environ[runtime.AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD_ENVVAR] == (
            runtime.AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD_DEFAULT
        )
        assert provider.workload_provider_password_grant is not None
        assert (
            provider.workload_provider_password_grant["password"]
            == runtime.AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD_DEFAULT
        )
    finally:
        os.environ.pop(runtime.AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD_ENVVAR, None)


def test_append_xdist_group_suffix_only_appends_once_and_sorts_groups():
    nodeid = "tests/auth_idp/contracts/test_tokens.py::test_provider_workload_provider_token_is_real"
    assert append_xdist_group_suffix(nodeid, {"idp-live"}) == f"{nodeid}@idp-live"
    assert append_xdist_group_suffix(nodeid, {"b", "a"}) == f"{nodeid}@a_b"
    assert append_xdist_group_suffix(f"{nodeid}@idp-live", {"idp-live"}) == f"{nodeid}@idp-live"
