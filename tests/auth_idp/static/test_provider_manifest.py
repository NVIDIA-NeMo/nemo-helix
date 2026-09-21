# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validator_for
from nemo_platform_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR

from tests.auth_idp.providers import load_provider_config, load_provider_configs

pytestmark = [pytest.mark.auth_idp]


def _load_provider_manifest_schema() -> dict:
    return yaml.safe_load(Path("contrib/auth/manifest.schema.yaml").read_text())


def test_all_provider_manifests_share_the_same_contract(monkeypatch):
    monkeypatch.setenv("AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", "shared-secret")
    schema = _load_provider_manifest_schema()
    validator = validator_for(schema)(schema)
    for provider in load_provider_configs():
        manifest = yaml.safe_load(Path(f"contrib/auth/{provider.name}/manifest.yaml").read_text())
        validator.validate(manifest)
        assert manifest["provider"] == provider.name


@pytest.mark.parametrize(
    ("grant_name", "base_grant", "extra_credential"),
    [
        (
            "e2e_setup_password_grant",
            None,
            {"password_env_var": "AUTHENTIK_SETUP_PASSWORD"},
        ),
        (
            "interactive_user_password_grant",
            {
                "grant_type": "password",
                "client_id": "nemo-platform",
                "username": "nemo-user",
                "password": "shared-secret",
                "scope": "openid email groups",
            },
            {"password_env_var": "AUTHENTIK_USER_PASSWORD"},
        ),
        ("workload_provider_password_grant", None, {"password": "shared-secret"}),
    ],
)
def test_provider_manifest_rejects_multiple_grant_credential_sources(grant_name, base_grant, extra_credential):
    schema = _load_provider_manifest_schema()
    validator = validator_for(schema)(schema)
    manifest = yaml.safe_load(Path("contrib/auth/authentik/manifest.yaml").read_text())
    if base_grant is not None:
        manifest["token_acquisition"][grant_name] = base_grant
    manifest["token_acquisition"][grant_name].update(extra_credential)

    with pytest.raises(ValidationError):
        validator.validate(manifest)


def test_provider_manifest_allows_generic_grant_with_password_and_client_secret():
    schema = _load_provider_manifest_schema()
    validator = validator_for(schema)(schema)
    manifest = yaml.safe_load(Path("contrib/auth/zitadel/manifest.yaml").read_text())
    manifest["token_acquisition"]["e2e_setup_grant"].update(
        {
            "grant_type": "password",
            "username": "nemo-setup",
            "password_env_var": "ZITADEL_E2E_SETUP_PASSWORD",
        }
    )

    validator.validate(manifest)


def test_authentik_manifest_declares_real_token_acquisition_contract():
    manifest = yaml.safe_load(Path("contrib/auth/authentik/manifest.yaml").read_text())
    token_acquisition = manifest["token_acquisition"]
    interactive_user_identity = manifest["interactive_user_identity"]
    principal_contract = manifest["principal_contract"]
    workload_identity = manifest["workload_identity"]
    workload_contract = manifest["workload_contract"]

    assert interactive_user_identity["username"] == "nemo-user"
    assert interactive_user_identity["password"] == "nemo-user-password-dev"
    assert interactive_user_identity["expected_email"] == "nemo-user@example.com"
    assert token_acquisition["token_endpoint"]
    setup_grant = token_acquisition["e2e_setup_password_grant"]
    assert setup_grant["grant_type"] == "password"
    assert setup_grant["client_id"] == "nemo-platform"
    assert setup_grant["username"] == "nemo-setup"
    assert setup_grant["password"] == "nemo-setup-token-secret-dev"
    assert "password_env_var" not in setup_grant
    assert "interactive_user_password_grant" not in token_acquisition
    workload_provider_grant = token_acquisition["workload_provider_password_grant"]
    assert workload_provider_grant["grant_type"] == "password"
    assert workload_provider_grant["client_id"]
    assert workload_provider_grant["password_env_var"] == "AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD"
    assert "password" not in workload_provider_grant
    assert workload_identity["principal_id"]
    assert not workload_identity["principal_id"].startswith(principal_contract["internal_service_prefix_reserved"])
    assert workload_identity["expected_groups"] == ["nemo-workloads"]
    assert workload_contract["audience"] == "nemo-platform"
    assert workload_contract["groups_format"] == "comma_string"
    assert workload_contract["token_env_vars"] == [WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR]
    assert workload_contract["forwarded_headers"]["principal_id"] == "X-NMP-Principal-Id"
    assert workload_contract["forwarded_headers"]["principal_groups"] == "X-NMP-Principal-Groups"


def test_authentik_manifest_declares_provider_test_runtimes():
    manifest = yaml.safe_load(Path("contrib/auth/authentik/manifest.yaml").read_text())

    runtime_ids = {runtime["id"] for runtime in manifest["test_runtimes"]}

    assert "authentik-compose" in runtime_ids
    assert "authentik-kubernetes" in runtime_ids


def test_zitadel_manifest_declares_kubernetes_test_runtime():
    manifest = yaml.safe_load(Path("contrib/auth/zitadel/manifest.yaml").read_text())

    assert manifest["compose_file"] is None
    assert {runtime["id"] for runtime in manifest["test_runtimes"]} == {"zitadel-kubernetes"}
    token_acquisition = manifest["token_acquisition"]
    assert token_acquisition["e2e_setup_grant"]["grant_type"] == "client_credentials"
    assert token_acquisition["workload_provider_grant"]["grant_type"] == "client_credentials"


def test_authentik_manifest_compose_and_kubernetes_runtime_capabilities_stay_in_parity():
    manifest = yaml.safe_load(Path("contrib/auth/authentik/manifest.yaml").read_text())
    runtimes = {runtime["id"]: set(runtime["capabilities"]) for runtime in manifest["test_runtimes"]}

    compose = runtimes["authentik-compose"]
    kubernetes = runtimes["authentik-kubernetes"]

    assert compose == kubernetes - {"kubernetes_token_review"}
    assert "interactive_user_token" not in compose
    assert "interactive_user_token" not in kubernetes
    assert "workload_provider_token" in compose
    assert "workload_provider_token" in kubernetes
    assert "workload_deployment" in compose
    assert "workload_deployment" in kubernetes
    assert "managed_workload_deployment_obo" in compose
    assert "managed_workload_deployment_obo" in kubernetes


def test_authentik_common_contracts_do_not_require_removed_interactive_user_token_capability():
    contract_text = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("tests/auth_idp/contracts").glob("*.py")
    )

    assert '"interactive_user_token"' not in contract_text


def test_all_provider_test_runtimes_declare_capabilities(monkeypatch):
    monkeypatch.setenv("AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", "shared-secret")

    for provider in load_provider_configs():
        assert provider.test_runtimes
        for runtime in provider.test_runtimes:
            assert runtime.id
            assert runtime.backend in {"compose", "kubernetes", "external"}
            assert runtime.capabilities


def test_authentik_manifest_declares_extended_startup_timeouts_for_real_oidc():
    manifest = yaml.safe_load(Path("contrib/auth/authentik/manifest.yaml").read_text())
    startup_timeouts = manifest["startup_timeouts"]

    assert startup_timeouts["healthchecks_seconds"] >= 240
    assert startup_timeouts["gateway_seconds"] >= 30
    assert startup_timeouts["token_endpoint_seconds"] >= 60


def test_authentik_provider_config_loads_token_acquisition_fields(monkeypatch):
    monkeypatch.setenv("AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", "shared-secret")
    provider = next(config for config in load_provider_configs() if config.name == "authentik")

    assert provider.compose_file == Path("contrib/auth/authentik/compose/docker-compose.yml")
    assert provider.nemo_config == Path("contrib/auth/authentik/config/platform-compose-authentik.yaml")
    assert provider.gateway_base_url == "https://127.0.0.1:18080"
    assert provider.discovery_url == "https://127.0.0.1:18080/application/o/nemo/.well-known/openid-configuration"
    assert provider.token_endpoint == "https://127.0.0.1:18080/application/o/token/"
    assert provider.interactive_user_username == "nemo-user"
    assert provider.interactive_user_password == "nemo-user-password-dev"
    assert provider.interactive_user_expected_email == "nemo-user@example.com"
    assert provider.e2e_setup_password_grant is not None
    assert provider.e2e_setup_password_grant["grant_type"] == "password"
    assert provider.e2e_setup_password_grant["password"] == "nemo-setup-token-secret-dev"
    assert provider.interactive_user_password_grant is None
    assert provider.workload_provider_password_grant is not None
    assert provider.workload_provider_password_grant["grant_type"] == "password"
    assert provider.workload_audience == "nemo-platform"
    assert provider.workload_principal_claim == "sub"
    assert provider.workload_groups_claim == "groups"
    assert provider.workload_groups_format == "comma_string"
    assert provider.workload_token_env_vars == [WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR]
    assert provider.startup_timeouts == {
        "healthchecks_seconds": 600,
        "gateway_seconds": 30,
        "token_endpoint_seconds": 180,
    }


def test_authentik_provider_config_resolves_workload_provider_password_grant_env_var(monkeypatch):
    monkeypatch.setenv("AUTHENTIK_WORKLOAD_IDENTITY_PASSWORD", "shared-secret")

    provider = load_provider_config(Path("contrib/auth/authentik/manifest.yaml"))

    assert provider.workload_provider_password_grant is not None
    assert provider.workload_provider_password_grant["password"] == "shared-secret"
    assert "password_env_var" not in provider.workload_provider_password_grant
    assert provider.e2e_setup_password_grant is not None
    assert provider.e2e_setup_password_grant["password"] == "nemo-setup-token-secret-dev"
    assert "password_env_var" not in provider.e2e_setup_password_grant


def test_zitadel_provider_config_loads_generic_client_credentials_grants(monkeypatch):
    monkeypatch.setenv("ZITADEL_E2E_SETUP_CLIENT_SECRET", "setup-secret")
    monkeypatch.setenv("ZITADEL_WORKLOAD_CLIENT_SECRET", "workload-secret")

    provider = load_provider_config(Path("contrib/auth/zitadel/manifest.yaml"))

    assert provider.compose_file is None
    assert provider.nemo_config == Path("contrib/auth/zitadel/helm/values.yaml")
    assert provider.token_endpoint == "https://127.0.0.1:18084/oauth/v2/token"
    assert provider.e2e_setup_password_grant == {
        "grant_type": "client_credentials",
        "client_id": "__ZITADEL_SETUP_CLIENT_ID__",
        "client_secret": "setup-secret",
        "client_auth_method": "client_secret_basic",
        "expected_subject": "nemo-setup",
        "scope": "openid profile email groups urn:zitadel:iam:org:project:id:__ZITADEL_PROJECT_ID__:aud",
    }
    assert provider.workload_provider_password_grant == {
        "grant_type": "client_credentials",
        "client_id": "__ZITADEL_WORKLOAD_CLIENT_ID__",
        "client_secret": "workload-secret",
        "client_auth_method": "client_secret_basic",
        "expected_subject": "svc-nemo",
        "scope": "openid profile email groups urn:zitadel:iam:org:project:id:__ZITADEL_PROJECT_ID__:aud",
    }
