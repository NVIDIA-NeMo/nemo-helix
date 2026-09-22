# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.auth_idp]

ZITADEL_DIR = Path("contrib/auth/zitadel")
HELM_DIR = ZITADEL_DIR / "helm"
ZITADEL_SCRIPT_TIMEOUT_SECONDS = 30


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_zitadel_script(*args: str, env: dict[str, str] | None = None) -> str:
    process_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("NMP_ZITADEL_K8S_") and key != "NEMO_ZITADEL_STATE_DIR"
    }
    process_env.update(env or {})
    completed = subprocess.run(
        [str(ZITADEL_DIR / "run.sh"), *args],
        text=True,
        capture_output=True,
        check=False,
        env=process_env,
        timeout=ZITADEL_SCRIPT_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def _gateway_port_from_script_output(output: str) -> str:
    match = re.search(
        r"\b(?:NMP_ZITADEL_K8S_GATEWAY_PORT|nemo-platform\.zitadelPublicGateway\.port)=(\d+)\b",
        output,
    )
    assert match is not None, output
    return match.group(1)


def test_zitadel_manifest_declares_kubernetes_only_runtime() -> None:
    manifest = _load_yaml(ZITADEL_DIR / "manifest.yaml")

    assert manifest["provider"] == "zitadel"
    assert manifest["mode"] == "reference-only"
    assert manifest["compose_file"] is None
    assert {runtime["id"] for runtime in manifest["test_runtimes"]} == {"zitadel-kubernetes"}
    [runtime] = manifest["test_runtimes"]
    assert runtime["backend"] == "kubernetes"
    assert "device_flow" in runtime["capabilities"]
    assert "kubernetes_token_review" in runtime["capabilities"]


def test_zitadel_readme_documents_kubernetes_runner() -> None:
    readme = (ZITADEL_DIR / "README.md").read_text(encoding="utf-8")

    assert "contrib/auth/zitadel/run.sh --help" in readme
    assert "contrib/auth/zitadel/run.sh up k8s" in readme
    assert "contrib/auth/zitadel/run.sh test k8s" in readme
    assert "contrib/auth/zitadel/run.sh down k8s" in readme
    assert "NMP_ZITADEL_K8S_*" in readme


def test_zitadel_kubernetes_runner_is_provider_specific() -> None:
    run_sh = (ZITADEL_DIR / "run.sh").read_text(encoding="utf-8")

    assert "NMP_ZITADEL_K8S_HELM_RELEASE" in run_sh
    assert 'K8S_RUNTIME="${NMP_ZITADEL_K8S_RUNTIME:-kind}"' in run_sh
    assert "uv run --frozen pytest tests/auth_idp/contracts" in run_sh
    assert "--auth-idp-runtime zitadel-kubernetes" in run_sh
    assert "-m auth_idp_runtime" in run_sh
    assert "--runtime RUNTIME" in run_sh
    assert "--reuse" in run_sh
    assert "--skip-image-load" in run_sh
    assert "--export-kubeconfig" in run_sh
    assert 'DEFAULT_K8S_GATEWAY_PORT="18084"' in run_sh
    assert "helm repo add zitadel https://charts.zitadel.com --force-update" in run_sh
    assert "helm dependency build contrib/auth/zitadel/helm" in run_sh
    assert "nemo-platform.zitadelPublicGateway.port=${K8S_GATEWAY_PORT}" in run_sh
    assert "zitadel nemo-platform-api nemo-platform-core-controller nemo-platform-envoy" in run_sh
    assert "NMP_ZITADEL_K8S_WORKLOAD_TOKEN_PRIVATE_KEY_FILE" in run_sh
    assert "NMP_AUTHENTIK" not in run_sh
    assert "compose" not in run_sh
    assert "render-blueprint" not in run_sh


def test_zitadel_runner_rejects_non_kubernetes_targets() -> None:
    completed = subprocess.run(
        [str(ZITADEL_DIR / "run.sh"), "test", "compose", "--dry-run"],
        text=True,
        capture_output=True,
        check=False,
        timeout=ZITADEL_SCRIPT_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 2
    assert "test target must be k8s" in completed.stderr


def test_zitadel_kubernetes_up_starts_reusable_stack_without_pytest() -> None:
    output = _run_zitadel_script("up", "k8s", "--dry-run", "--skip-image-load")
    gateway_port = _gateway_port_from_script_output(output)

    assert "kind create cluster --name nmp-zitadel-reuse" in output
    assert "kind export kubeconfig --name nmp-zitadel-reuse" not in output
    assert "helm --kubeconfig" in output
    assert "upgrade --install zitadel-demo" in output
    assert "--timeout 20m" in output
    assert f"nemo-platform.zitadelPublicGateway.port={gateway_port}" in output
    assert f"port-forward svc/nemo-platform-envoy {gateway_port}:8080" in output
    assert f"https://127.0.0.1:{gateway_port}/health/gateway/ready" in output
    assert "kubectl -n nemo-zitadel get pods" in output
    assert "uv run --frozen nemo config set --context zitadel-k8s" in output
    assert "--certificate-authority" in output
    assert "write lifecycle state" in output
    assert "uv run --frozen pytest" not in output
    assert "--auth-idp-runtime zitadel-kubernetes" not in output


def test_zitadel_kubernetes_up_key_derives_managed_instance_names() -> None:
    output = _run_zitadel_script(
        "up",
        "k8s",
        "--key",
        "dev",
        "--dry-run",
        "--skip-image-load",
        env={"NMP_ZITADEL_K8S_GATEWAY_PORT": "19084"},
    )

    assert "kind create cluster --name nmp-zitadel-dev" in output
    assert "nemo-platform.zitadelPublicGateway.port=19084" in output
    assert "port-forward svc/nemo-platform-envoy 19084:8080" in output
    assert "uv run --frozen nemo config set --context zitadel-k8s-dev" in output
    assert "write lifecycle state" in output
    assert "run.sh down k8s --key dev" in output


def test_zitadel_kubernetes_test_action_runs_contract_pytest() -> None:
    output = _run_zitadel_script(
        "test",
        "k8s",
        "--dry-run",
        env={"NMP_ZITADEL_K8S_GATEWAY_PORT": "19084"},
    )

    assert "NMP_ZITADEL_K8S_HELM_WAIT_TIMEOUT=20m" in output
    assert "NMP_ZITADEL_K8S_NAMESPACE=nemo-zitadel" in output
    assert "NMP_ZITADEL_K8S_GATEWAY_PORT=19084" in output
    assert "NMP_ZITADEL_K8S_WORKLOAD_TOKEN_PRIVATE_KEY_FILE=" in output
    assert "uv run --frozen pytest tests/auth_idp/contracts" in output
    assert "--auth-idp-runtime zitadel-kubernetes" in output


def test_zitadel_kubernetes_runner_builds_with_direct_buildx_bake() -> None:
    output = _run_zitadel_script(
        "test",
        "k8s",
        "--dry-run",
        "--platform",
        "linux/arm64",
    )

    assert "docker buildx bake -f docker-bake.hcl nmp-api-docker --set \\*.platform=linux/arm64 --load" in output
    assert "make docker-load" not in output


def test_zitadel_down_cleans_kubernetes_resources(tmp_path: Path) -> None:
    output = _run_zitadel_script("down", "k8s", "--dry-run", env={"NEMO_ZITADEL_STATE_DIR": str(tmp_path)})

    assert "kind delete cluster --name nmp-zitadel-reuse" in output
    assert "port-forward.pid" in output
    assert "nemo config delete-context zitadel-k8s --prune-orphans" in output


def test_zitadel_down_key_cleans_derived_kubernetes_context(tmp_path: Path) -> None:
    output = _run_zitadel_script(
        "down",
        "k8s",
        "--key",
        "dev",
        "--dry-run",
        env={"NEMO_ZITADEL_STATE_DIR": str(tmp_path)},
    )

    assert "kind delete cluster --name nmp-zitadel-dev" in output
    assert "nemo config delete-context zitadel-k8s-dev --prune-orphans" in output


def test_zitadel_manifest_uses_supported_non_password_grants() -> None:
    manifest = _load_yaml(ZITADEL_DIR / "manifest.yaml")
    token_acquisition = manifest["token_acquisition"]

    assert "password" not in manifest["interactive_user_identity"]
    assert manifest["interactive_user_identity"]["password_env_var"] == "ZITADEL_INTERACTIVE_USER_PASSWORD"
    assert "e2e_setup_password_grant" not in token_acquisition
    assert "workload_provider_password_grant" not in token_acquisition
    assert token_acquisition["e2e_setup_grant"] == {
        "grant_type": "client_credentials",
        "client_id": "__ZITADEL_SETUP_CLIENT_ID__",
        "client_secret_env_var": "ZITADEL_E2E_SETUP_CLIENT_SECRET",
        "client_auth_method": "client_secret_basic",
        "expected_subject": "nemo-setup",
        "scope": "openid profile email groups urn:zitadel:iam:org:project:id:__ZITADEL_PROJECT_ID__:aud",
    }
    assert token_acquisition["workload_provider_grant"] == {
        "grant_type": "client_credentials",
        "client_id": "__ZITADEL_WORKLOAD_CLIENT_ID__",
        "client_secret_env_var": "ZITADEL_WORKLOAD_CLIENT_SECRET",
        "client_auth_method": "client_secret_basic",
        "expected_subject": "svc-nemo",
        "scope": "openid profile email groups urn:zitadel:iam:org:project:id:__ZITADEL_PROJECT_ID__:aud",
    }


def test_zitadel_chart_declares_expected_dependencies() -> None:
    chart = _load_yaml(HELM_DIR / "Chart.yaml")
    dependencies = {dependency["name"]: dependency for dependency in chart["dependencies"]}

    assert chart["name"] == "nemo-platform-zitadel"
    assert dependencies["zitadel"] == {
        "name": "zitadel",
        "version": "10.0.6",
        "repository": "https://charts.zitadel.com",
    }
    assert dependencies["nemo-platform"] == {
        "name": "nemo-platform",
        "version": "0.0.0",
        "repository": "file://../../../../k8s/helm",
    }


def test_zitadel_values_use_introspection_for_opaque_tokens() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    oidc = values["nemo-platform"]["platformConfig"]["auth"]["oidc"]

    assert oidc["introspect_opaque_tokens"] is True
    assert (
        oidc["jwks_uri"]
        == '{{ include "nemo-platform-zitadel.serviceUrl" (dict "root" . "serviceName" "nemo-platform-envoy" "namespace" .Values.envoyProxy.serviceNamespace "scheme" "https" "port" 8080) }}/oauth/v2/keys'
    )
    assert (
        oidc["introspection_endpoint"]
        == '{{ include "nemo-platform-zitadel.serviceUrl" (dict "root" . "serviceName" "nemo-platform-envoy" "namespace" .Values.envoyProxy.serviceNamespace "scheme" "https" "port" 8080) }}/oauth/v2/introspect'
    )
    assert oidc["client_id"] == "__ZITADEL_NEMO_CLIENT_ID__"
    assert oidc["introspection_client_id"] == "__ZITADEL_INTROSPECTION_CLIENT_ID__"
    assert oidc["introspection_client_secret_env_var"] == "NMP_AUTH_OIDC_INTROSPECTION_CLIENT_SECRET"
    assert "introspection_client_secret" not in oidc
    assert values["nemo-platform"]["api"]["env"]["NMP_AUTH_OIDC_INTROSPECTION_CLIENT_SECRET"] == {
        "valueFrom": {
            "secretKeyRef": {
                "name": "nemo-zitadel-seed-state",
                "key": "nemo_client_secret",
                "optional": True,
            }
        }
    }
    assert oidc["resolve_opaque_tokens_via_userinfo"] is False
    assert oidc["userinfo_endpoint"] == '{{ include "nemo-platform-zitadel.publicGatewayUrl" . }}/oidc/v1/userinfo'
    assert "__ZITADEL_PROJECT_ID__" in oidc["default_scopes"]


def test_zitadel_values_keep_nemo_groups_claim_provider_neutral() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    oidc = values["nemo-platform"]["platformConfig"]["auth"]["oidc"]

    assert values["zitadelDemo"]["groupsClaim"] == "groups"
    assert values["integration"]["zitadel"]["groupsClaim"] == "groups"
    assert oidc["groups_claim"] == "groups"
    assert "urn:zitadel:iam:org:project:roles" not in yaml.safe_dump(values)


def test_zitadel_envoy_strips_trusted_headers_and_checks_zitadel_discovery() -> None:
    envoy_template = (HELM_DIR / "templates" / "_envoy-config.tpl").read_text(encoding="utf-8")

    assert "headers:remove({{ $header | quote }})" in envoy_template
    assert '"/.well-known/openid-configuration"' in envoy_template
    assert '"zitadel"' in envoy_template
    assert 'string.format(\'{"status":"not_ready","nemo":"%s","zitadel":"%s"}\'' in envoy_template
    assert 'prefix: "/.well-known/nemo-platform/"' in envoy_template
    assert 'path: "/apis/auth/discovery"' in envoy_template
    assert 'path: "/apis/auth/authenticate"' in envoy_template
    assert 'path_prefix: "/apis/auth/ext-authz"' in envoy_template
    assert 'path: "/apis/auth/jwks"' in envoy_template
    assert 'path: "/apis/auth/token"' in envoy_template
    assert "host_rewrite_literal: {{ $publicGatewayAuthority | quote }}" in envoy_template
    assert "x-forwarded-proto" in envoy_template


def test_zitadel_chart_seeds_generated_clients_and_patches_nemo_config() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    seed_template = (HELM_DIR / "templates" / "seed-job.yaml").read_text(encoding="utf-8")

    assert values["zitadelSeedJob"]["stateSecretName"] == "nemo-zitadel-seed-state"
    assert values["zitadelSeedJob"]["patSecretName"] == "iam-admin-pat"
    assert values["zitadelSeedJob"]["nemoConfigMapName"] == "nemo-platform-config"
    assert values["zitadel"]["initJob"]["activeDeadlineSeconds"] == 900
    assert values["zitadel"]["setupJob"]["activeDeadlineSeconds"] == 900
    assert values["zitadelSeedJob"]["nemoDeployments"] == [
        "nemo-platform-api",
        "nemo-platform-core-controller",
    ]
    assert values["zitadelDemo"]["loginClientMachine"] == {
        "userId": "nemo-login-client",
        "name": "NeMo Login Client",
    }
    assert "OIDC_TOKEN_TYPE_BEARER" in seed_template
    assert "ACCESS_TOKEN_TYPE_BEARER" in seed_template
    assert "API_AUTH_METHOD_TYPE_BASIC" in seed_template
    assert "OIDC_APP_TYPE_NATIVE" in seed_template
    assert "OIDC_AUTH_METHOD_TYPE_NONE" in seed_template
    assert "/management/v1/projects" in seed_template
    assert "/management/v1/projects/{}/apps/api" in seed_template
    assert "/management/v1/projects/{}/apps/{}/api_config/_generate_client_secret" in seed_template
    assert "/management/v1/users/machine" in seed_template
    assert "/admin/v1/members" in seed_template
    assert "/management/v1/users/{}/pats" in seed_template
    assert '"IAM_LOGIN_CLIENT"' in seed_template
    assert "login_client_pat" in seed_template
    assert "/management/v1/users/human/_import" in seed_template
    assert "/management/v1/global/users/_by_login_name" in seed_template
    assert "/management/v1/users/{}/password" in seed_template
    assert '"noChangeRequired": True' in seed_template
    assert "/management/v1/projects/_search" in seed_template
    assert "/management/v1/projects/{}/apps/_search" in seed_template
    assert "__ZITADEL_NEMO_CLIENT_SECRET__" not in seed_template
    assert values["zitadelDemo"]["interactiveUser"]["userName"] == "nemo-user"
    assert values["zitadelDemo"]["interactiveUser"]["email"] == "nemo-user@example.com"
    assert "password" not in values["zitadelDemo"]["interactiveUser"]
    assert '"interactive_user_password": INTERACTIVE_USER_PASSWORD' in seed_template
    assert 'missing == {"interactive_user_password"}' in seed_template
    assert "migrated legacy ZITADEL seed state" in seed_template
    assert "valueFrom:" in seed_template
    assert "secretKeyRef:" in seed_template
    assert "zitadelSecrets.demo.secretName" in seed_template
    assert "zitadelSecrets.demo.interactiveUserPasswordKey" in seed_template
    assert 'for trigger in ("4", "5")' in seed_template
    assert '"/management/v1/flows/2/trigger/{}".format(trigger)' in seed_template
    assert "__ZITADEL_INTROSPECTION_CLIENT_ID__" in seed_template
    assert "__ZITADEL_NEMO_CLIENT_ID__" in seed_template
    assert "__ZITADEL_PROJECT_ID__" in seed_template
    assert "nemo-platform.nvidia.com/zitadel-seeded-at" in seed_template


def test_zitadel_values_use_generated_secrets_for_sensitive_defaults() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    generated_secrets = (HELM_DIR / "templates" / "generated-secrets.yaml").read_text(encoding="utf-8")

    assert values["zitadelSecrets"]["masterkey"] == {
        "create": True,
        "secretName": "zitadel-masterkey",
        "key": "masterkey",
    }
    assert values["zitadelSecrets"]["demo"] == {
        "create": True,
        "secretName": "nemo-zitadel-demo-credentials",
        "interactiveUserPasswordKey": "interactive-user-password",
    }
    assert values["zitadelSecrets"]["postgresql"] == {
        "create": True,
        "secretName": "zitadel-postgresql",
        "adminPasswordKey": "postgres-password",
        "userPasswordKey": "password",
        "dsnKey": "dsn",
    }
    assert values["nemoPlatformSecrets"]["ngc"] == {
        "create": True,
        "secretName": "nemo-platform-ngc-api",
        "key": "NGC_API_KEY",
    }
    assert values["nemoPlatformSecrets"]["postgresql"] == {
        "create": True,
        "secretName": "nemo-platform-postgres",
        "passwordKey": "password",
    }
    assert values["zitadel"]["zitadel"]["masterkey"] == ""
    assert values["zitadel"]["zitadel"]["masterkeySecretName"] == "zitadel-masterkey"
    assert values["zitadel"]["zitadel"]["configmapConfig"]["Database"]["Postgres"]["Host"] == "zitadel-postgresql"
    assert "Human" not in values["zitadel"]["zitadel"]["configmapConfig"]["FirstInstance"]["Org"]
    assert values["zitadel"]["env"] == [
        {
            "name": "ZITADEL_DATABASE_POSTGRES_DSN",
            "valueFrom": {
                "secretKeyRef": {
                    "name": "zitadel-postgresql",
                    "key": "dsn",
                },
            },
        },
    ]
    assert values["zitadel"]["postgresql"]["fullnameOverride"] == "zitadel-postgresql"
    assert "password" not in values["zitadel"]["postgresql"]["auth"]
    assert "postgresPassword" not in values["zitadel"]["postgresql"]["auth"]
    assert values["zitadel"]["postgresql"]["auth"]["existingSecret"] == "zitadel-postgresql"
    assert values["zitadel"]["postgresql"]["auth"]["secretKeys"] == {
        "adminPasswordKey": "postgres-password",
        "userPasswordKey": "password",
    }
    assert values["nemo-platform"]["existingSecret"] == "nemo-platform-ngc-api"
    assert values["nemo-platform"]["ngcAPIKey"] == ""
    assert values["nemo-platform"]["postgresql"]["auth"]["existingSecret"] == "nemo-platform-postgres"
    assert "nemo-platform-zitadel.secretValue" in generated_secrets
    assert "randAlphaNum 32" in generated_secrets
    assert "randAlphaNum 20" in generated_secrets
    assert "zitadelSecrets.demo.interactiveUserPasswordKey" in generated_secrets
    assert "host=zitadel-postgresql" in generated_secrets
    assert "$adminPassword" in generated_secrets
    assert "dbname=zitadel sslmode=disable" in generated_secrets
    assert '"helm.sh/hook": pre-install,pre-upgrade' in generated_secrets


def test_zitadel_demo_files_avoid_empty_password_placeholders() -> None:
    manifest = (ZITADEL_DIR / "manifest.yaml").read_text(encoding="utf-8")
    values = (HELM_DIR / "values.yaml").read_text(encoding="utf-8")

    assert 'password: ""' not in manifest
    assert 'password: ""' not in values
    assert 'postgresPassword: ""' not in values


def test_zitadel_chart_creates_workload_token_secrets_and_tokenreview_rbac() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    signing_template = (HELM_DIR / "templates" / "workload-token-signing-key-secret.yaml").read_text(encoding="utf-8")
    helpers_template = (HELM_DIR / "templates" / "_helpers.tpl").read_text(encoding="utf-8")
    tls_template = (HELM_DIR / "templates" / "workload-token-tls.yaml").read_text(encoding="utf-8")
    tokenreview_template = (HELM_DIR / "templates" / "tokenreview-rbac.yaml").read_text(encoding="utf-8")

    assert values["workloadTokenSigningKey"]["create"] is True
    assert values["workloadTokenSigningKey"]["secretName"] == "nemo-workload-token-signing-key"
    assert values["workloadTokenSigningKey"]["key"] == "private-key.pem"
    assert values["workloadTokenTls"]["create"] is True
    assert values["workloadTokenTls"]["secretName"] == "nemo-platform-envoy-tls"
    assert values["nemo-platform"]["api"]["env"]["SSL_CERT_FILE"] == "/etc/nmp/workload-token-ca/ca.crt"
    assert values["nemo-platform"]["api"]["env"]["REQUESTS_CA_BUNDLE"] == "/etc/nmp/workload-token-ca/ca.crt"
    assert any(volume["name"] == "workload-token-tls-ca" for volume in values["nemo-platform"]["api"]["extraVolumes"])
    assert any(
        mount["name"] == "workload-token-tls-ca" for mount in values["nemo-platform"]["api"]["extraVolumeMounts"]
    )
    jobs_config = values["nemo-platform"]["platformConfig"]["jobs"]
    assert jobs_config["executor_defaults"]["kubernetes_job"]["env"] == {
        "SSL_CERT_FILE": "/etc/nmp/workload-token-ca/ca.crt",
        "REQUESTS_CA_BUNDLE": "/etc/nmp/workload-token-ca/ca.crt",
    }
    workload_executor = next(
        executor
        for executor in jobs_config["executors"]
        if executor["provider"] == "cpu" and executor["profile"] == "workload"
    )
    workload_config = workload_executor["config"]
    assert workload_config["env"] == {
        "SSL_CERT_FILE": "/etc/nmp/workload-token-ca/ca.crt",
        "REQUESTS_CA_BUNDLE": "/etc/nmp/workload-token-ca/ca.crt",
    }
    assert workload_config["storage"] == jobs_config["executor_defaults"]["kubernetes_job"]["storage"]
    assert "nemo-platform-zitadel.workloadTokenSigningKey.privateKeyPem" in signing_template
    assert 'genPrivateKey "rsa"' in helpers_template
    assert "genSignedCert" in tls_template
    assert 'resources: ["tokenreviews"]' in tokenreview_template
    assert 'verbs: ["create"]' in tokenreview_template
