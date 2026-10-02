# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.auth_idp]

ZITADEL_DIR = Path("contrib/auth/zitadel")
HELM_DIR = ZITADEL_DIR / "helm"
ZITADEL_SCRIPT_TIMEOUT_SECONDS = 30
ENVOY_SERVICE_URL_TEMPLATE = (
    '{{ include "nemo-helix-zitadel.serviceUrl" '
    '(dict "root" . "serviceName" "nemo-helix-envoy" '
    '"namespace" .Values.envoyProxy.serviceNamespace "scheme" "https" "port" 8080) }}'
)


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_zitadel_script(*args: str, env: dict[str, str] | None = None) -> str:
    process_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("NHX_ZITADEL_K8S_") and key != "NEMO_ZITADEL_STATE_DIR"
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
        r"\b(?:NHX_ZITADEL_K8S_GATEWAY_PORT|nemo-helix\.zitadelPublicGateway\.port)=(\d+)\b",
        output,
    )
    assert match is not None, output
    return match.group(1)


def _workflow_job_block(workflow: str, job_name: str) -> str:
    lines = workflow.splitlines()
    start = next(index for index, line in enumerate(lines) if line == f"  {job_name}:")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            end = index
            break
    return "\n".join(lines[start:end])


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
    assert "NHX_ZITADEL_K8S_*" in readme


def test_zitadel_kubernetes_runner_is_provider_specific() -> None:
    run_sh = (ZITADEL_DIR / "run.sh").read_text(encoding="utf-8")

    assert "NHX_ZITADEL_K8S_HELM_RELEASE" in run_sh
    assert 'K8S_RUNTIME="${NHX_ZITADEL_K8S_RUNTIME:-kind}"' in run_sh
    assert "tests/auth_idp/contracts" in run_sh
    assert "--auth-idp-context" in run_sh
    assert "--run-e2e" in run_sh
    assert "deployment_context.py" in run_sh
    assert "--runtime RUNTIME" in run_sh
    assert "--reuse" in run_sh
    assert "--skip-image-load" in run_sh
    assert "--export-kubeconfig" in run_sh
    assert 'DEFAULT_K8S_GATEWAY_PORT="18084"' in run_sh
    assert "helm repo add zitadel https://charts.zitadel.com --force-update" in run_sh
    assert "helm dependency build contrib/auth/zitadel/helm" in run_sh
    assert "nemo-helix.zitadelPublicGateway.port=${K8S_GATEWAY_PORT}" in run_sh
    assert "zitadel nemo-helix-api nemo-helix-core-controller nemo-helix-envoy" in run_sh
    assert "NHX_AUTHENTIK" not in run_sh
    assert "compose" not in run_sh
    assert "render-blueprint" not in run_sh


def test_zitadel_kubernetes_runtime_is_in_auth_idp_ci_matrix() -> None:
    workflow = Path(".github/workflows/ci.yaml").read_text(encoding="utf-8")
    job = _workflow_job_block(workflow, "python-auth-idp-e2e-test")

    assert "runtime: authentik-compose" in job
    assert "runtime: authentik-kubernetes" in job
    assert "runtime: zitadel-kubernetes" in job
    assert "provider: zitadel" in job
    assert "namespace: nemo-zitadel" in job
    assert "NHX_ZITADEL_K8S_RUNTIME: kind" in job
    assert (
        "NHX_ZITADEL_K8S_CLUSTER_NAME: gha-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.runtime }}" in job
    )
    assert 'NHX_ZITADEL_K8S_REUSE_CLUSTER: "1"' in job
    assert "NHX_ZITADEL_K8S_JUNIT_XML: report-auth-idp-${{ matrix.runtime }}.xml" in job
    assert "helm repo add zitadel https://charts.zitadel.com --force-update" in job
    assert "helm dependency build contrib/auth/zitadel/helm" in job
    assert "helm lint --strict contrib/auth/zitadel/helm" in job
    assert "helm show chart zitadel --repo https://charts.zitadel.com --version 10.0.6" in job
    assert 'kind delete cluster --name "${AUTH_IDP_K8S_CLUSTER_NAME}"' in job

    workflow_document = _load_yaml(Path(".github/workflows/ci.yaml"))
    commands = {
        entry["runtime"]: entry["command"]
        for entry in workflow_document["jobs"]["python-auth-idp-e2e-test"]["strategy"]["matrix"]["include"]
        if entry.get("provider") == "zitadel"
    }
    assert commands == {"zitadel-kubernetes": "test k8s --runtime kind --gateway-port 18084"}


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

    assert "kind create cluster --name nhx-zitadel-reuse" in output
    assert "kind export kubeconfig --name nhx-zitadel-reuse" not in output
    assert "helm --kubeconfig" in output
    assert "upgrade --install zitadel-demo" in output
    assert "--timeout 20m" in output
    assert f"nemo-helix.zitadelPublicGateway.port={gateway_port}" in output
    assert f"port-forward svc/nemo-helix-envoy {gateway_port}:8080" in output
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
        env={"NHX_ZITADEL_K8S_GATEWAY_PORT": "19084"},
    )

    assert "kind create cluster --name nhx-zitadel-dev" in output
    assert "nemo-helix.zitadelPublicGateway.port=19084" in output
    assert "port-forward svc/nemo-helix-envoy 19084:8080" in output
    assert "uv run --frozen nemo config set --context zitadel-k8s-dev" in output
    assert "write lifecycle state" in output
    assert "run.sh down k8s --key dev" in output


def test_zitadel_kubernetes_test_action_runs_host_contracts() -> None:
    output = _run_zitadel_script(
        "test",
        "k8s",
        "--dry-run",
        env={"NHX_ZITADEL_K8S_GATEWAY_PORT": "19084"},
    )

    assert "--timeout 20m" in output
    assert "port-forward svc/nemo-helix-envoy 19084:8080" in output
    assert "uv run --frozen nemo config set" in output
    assert "tests/auth_idp/contracts" in output
    assert "--auth-idp-context" in output


def test_zitadel_kubernetes_test_action_uses_k3d_compatible_fresh_cluster_name() -> None:
    output = _run_zitadel_script("test", "k8s", "--dry-run", "--runtime", "k3d")
    match = re.search(r"\bk3d cluster create ([a-z0-9-]+)\b", output)

    assert match is not None, output
    assert match.group(1).startswith("nhx-zt-e2e-")
    assert len(match.group(1)) <= 32


def test_zitadel_kubernetes_runner_builds_with_direct_buildx_bake() -> None:
    output = _run_zitadel_script(
        "test",
        "k8s",
        "--dry-run",
        "--platform",
        "linux/arm64",
    )

    assert "docker buildx bake -f docker-bake.hcl nhx-api-docker --set \\*.platform=linux/arm64 --load" in output
    assert "make docker-load" not in output


def test_zitadel_down_cleans_kubernetes_resources(tmp_path: Path) -> None:
    output = _run_zitadel_script("down", "k8s", "--dry-run", env={"NEMO_ZITADEL_STATE_DIR": str(tmp_path)})

    assert "kind delete cluster --name nhx-zitadel-reuse" in output
    assert "port-forward.pid" in output
    assert "nemo config delete-context zitadel-k8s --prune-orphans" in output


def test_zitadel_kubernetes_port_forward_cleanup_uses_saved_namespace() -> None:
    run_sh = (ZITADEL_DIR / "run.sh").read_text(encoding="utf-8")

    assert "k8s_port_forward_namespace_file_for_cluster" in run_sh
    assert 'expected_namespace="$(<"${namespace_file}")"' in run_sh
    assert 'k8s_port_forward_pid_is_running "${pid}" "${K8S_GATEWAY_PORT}"' in run_sh
    assert '"${expected_namespace}"; then' in run_sh
    assert 'rm -f "${pid_file}" "${namespace_file}"' in run_sh


def test_zitadel_down_key_cleans_derived_kubernetes_context(tmp_path: Path) -> None:
    output = _run_zitadel_script(
        "down",
        "k8s",
        "--key",
        "dev",
        "--dry-run",
        env={"NEMO_ZITADEL_STATE_DIR": str(tmp_path)},
    )

    assert "kind delete cluster --name nhx-zitadel-dev" in output
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

    assert chart["name"] == "nemo-helix-zitadel"
    assert dependencies["zitadel"] == {
        "name": "zitadel",
        "version": "10.0.6",
        "repository": "https://charts.zitadel.com",
    }
    assert dependencies["nemo-helix"] == {
        "name": "nemo-helix",
        "version": "0.0.0",
        "repository": "file://../../../../k8s/helm",
    }


def test_zitadel_values_use_introspection_for_opaque_tokens() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    assert "studio" in values["nemo-helix"]["api"]["services"]
    oidc = values["nemo-helix"]["platformConfig"]["auth"]["oidc"]
    public_client = oidc["public_client"]
    confidential_client = oidc["confidential_client"]

    assert oidc["introspect_opaque_tokens"] is True
    assert (
        oidc["jwks_uri"]
        == '{{ include "nemo-helix-zitadel.serviceUrl" (dict "root" . "serviceName" "nemo-helix-envoy" "namespace" .Values.envoyProxy.serviceNamespace "scheme" "https" "port" 8080) }}/oauth/v2/keys'
    )
    assert (
        oidc["introspection_endpoint"]
        == '{{ include "nemo-helix-zitadel.serviceUrl" (dict "root" . "serviceName" "nemo-helix-envoy" "namespace" .Values.envoyProxy.serviceNamespace "scheme" "https" "port" 8080) }}/oauth/v2/introspect'
    )
    assert public_client["client_id"] == "__ZITADEL_NEMO_CLIENT_ID__"
    assert public_client["server_side_sessions"] is False
    assert public_client["authorization_endpoint"] == (
        '{{ include "nemo-helix-zitadel.publicGatewayUrl" . }}/oauth/v2/authorize'
    )
    assert public_client["token_endpoint"] == '{{ include "nemo-helix-zitadel.publicGatewayUrl" . }}/oauth/v2/token'
    assert confidential_client["client_id"] == "__ZITADEL_USER_LOGIN_CLIENT_ID__"
    assert confidential_client["authorization_endpoint"] == (
        '{{ include "nemo-helix-zitadel.publicGatewayUrl" . }}/oauth/v2/authorize'
    )
    assert confidential_client["token_endpoint"] == (
        '{{ include "nemo-helix-zitadel.serviceUrl" (dict "root" . "serviceName" "nemo-helix-envoy" "namespace" .Values.envoyProxy.serviceNamespace "scheme" "https" "port" 8080) }}/oauth/v2/token'
    )
    assert confidential_client["client_secret_env_var"] == "NHX_OIDC_CLIENT_SECRET"
    assert oidc["server_sessions"] == {"encryption_key_env_var": "NHX_AUTH_SESSION_ENCRYPTION_KEY"}
    assert "client_secret" not in confidential_client
    assert oidc["introspection_client_id"] == "__ZITADEL_INTROSPECTION_CLIENT_ID__"
    assert oidc["introspection_client_secret_env_var"] == "NHX_AUTH_OIDC_INTROSPECTION_CLIENT_SECRET"
    assert "introspection_client_secret" not in oidc
    assert values["nemo-helix"]["api"]["env"]["NHX_AUTH_OIDC_INTROSPECTION_CLIENT_SECRET"] == {
        "valueFrom": {
            "secretKeyRef": {
                "name": "nemo-zitadel-seed-state",
                "key": "nemo_client_secret",
                "optional": True,
            }
        }
    }
    assert oidc["resolve_opaque_tokens_via_userinfo"] is False
    assert oidc["userinfo_endpoint"] == '{{ include "nemo-helix-zitadel.publicGatewayUrl" . }}/oidc/v1/userinfo'
    assert "__ZITADEL_PROJECT_ID__" in public_client["default_scopes"]
    assert "__ZITADEL_PROJECT_ID__" in confidential_client["default_scopes"]


def test_zitadel_values_route_internal_bearer_clients_through_tls_envoy() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    nemo_values = values["nemo-helix"]

    platform_config = nemo_values["platformConfig"]["platform"]
    assert platform_config["base_url"] == ENVOY_SERVICE_URL_TEMPLATE

    controller = nemo_values["core"]["controller"]
    assert controller["env"] == {"NHX_CLIENT_SSL_CERT_FILE": "/etc/nhx/workload-token-ca/ca.crt"}
    assert "NHX_PLATFORM_URL" not in controller["env"]
    assert "NHX_AUTH_URL" not in controller["env"]
    assert controller["extraVolumes"] == [
        {
            "name": "workload-token-signing-key",
            "secret": {"secretName": "nemo-workload-token-signing-key"},
        },
        {
            "name": "workload-token-tls-ca",
            "secret": {
                "secretName": "nemo-helix-envoy-tls",
                "items": [{"key": "ca.crt", "path": "ca.crt"}],
            },
        },
    ]
    assert controller["extraVolumeMounts"] == [
        {
            "name": "workload-token-signing-key",
            "mountPath": "/etc/nhx/workload-token",
            "readOnly": True,
        },
        {
            "name": "workload-token-tls-ca",
            "mountPath": "/etc/nhx/workload-token-ca",
            "readOnly": True,
        },
    ]

    seed_job = nemo_values["platformSeedJob"]
    assert seed_job["extraEnv"] == [{"name": "NHX_CLIENT_SSL_CERT_FILE", "value": "/etc/nhx/workload-token-ca/ca.crt"}]
    assert seed_job["extraVolumes"] == controller["extraVolumes"]
    assert seed_job["extraVolumeMounts"] == controller["extraVolumeMounts"]


def test_zitadel_values_keep_nemo_groups_claim_provider_neutral() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    oidc = values["nemo-helix"]["platformConfig"]["auth"]["oidc"]

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
    assert 'prefix: "/.well-known/nemo-helix/"' in envoy_template
    assert 'path: "/apis/auth/discovery"' in envoy_template
    assert 'regex: "^/apis/auth/v2/(login(/callback)?|authorize(/[^/]+)?|token|logout|session)$"' in envoy_template
    assert 'path: "/apis/auth/authenticate"' in envoy_template
    assert 'path_prefix: "/apis/auth/ext-authz"' in envoy_template
    assert "direct_response:\n                            status: 404" in envoy_template
    assert 'path: "/apis/auth/jwks"' in envoy_template
    assert 'path: "/apis/auth/token"' in envoy_template
    assert "host_rewrite_literal: {{ $publicGatewayAuthority | quote }}" in envoy_template
    assert "x-forwarded-proto" in envoy_template


def test_zitadel_chart_seeds_generated_clients_and_patches_nemo_config() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    seed_template = (HELM_DIR / "templates" / "seed-job.yaml").read_text(encoding="utf-8")

    assert values["zitadelSeedJob"]["stateSecretName"] == "nemo-zitadel-seed-state"
    assert values["zitadelSeedJob"]["patSecretName"] == "iam-admin-pat"
    assert values["zitadelSeedJob"]["nemoConfigMapName"] == "nemo-helix-config"
    assert values["zitadel"]["initJob"]["activeDeadlineSeconds"] == 900
    assert values["zitadel"]["setupJob"]["activeDeadlineSeconds"] == 900
    assert values["zitadelSeedJob"]["nemoDeployments"] == [
        "nemo-helix-api",
        "nemo-helix-core-controller",
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
    assert "interactive_user_password" in seed_template
    assert "migrated ZITADEL seed state" in seed_template
    assert "USER_LOGIN_APP_NAME" in seed_template
    assert "__ZITADEL_USER_LOGIN_CLIENT_ID__" in seed_template
    assert "valueFrom:" in seed_template
    assert "secretKeyRef:" in seed_template
    assert "zitadelSecrets.demo.secretName" in seed_template
    assert "zitadelSecrets.demo.interactiveUserPasswordKey" in seed_template
    assert 'for trigger in ("4", "5")' in seed_template
    assert '"/management/v1/flows/2/trigger/{}".format(trigger)' in seed_template
    assert "__ZITADEL_INTROSPECTION_CLIENT_ID__" in seed_template
    assert "__ZITADEL_NEMO_CLIENT_ID__" in seed_template
    assert "__ZITADEL_PROJECT_ID__" in seed_template
    assert "nemo-helix.nvidia.com/zitadel-seeded-at" in seed_template


def test_zitadel_seed_migrates_legacy_cli_client_id_independently() -> None:
    seed_template = (HELM_DIR / "templates" / "seed-job.yaml").read_text(encoding="utf-8")
    migration = seed_template.split("    def _migrate_state", maxsplit=1)[1].split(
        "    def _upsert_secret", maxsplit=1
    )[0]

    legacy_client_migration = 'if "public_client_id" not in migrated and "cli_client_id" in migrated:'
    confidential_migration = 'if "user_login_client_secret" not in migrated:'
    assert legacy_client_migration in migration
    assert 'migrated["public_client_id"] = migrated["cli_client_id"]' in migration
    assert migration.index(legacy_client_migration) < migration.index(confidential_migration)


def test_zitadel_seed_does_not_overwrite_existing_public_client_id() -> None:
    seed_template = (HELM_DIR / "templates" / "seed-job.yaml").read_text(encoding="utf-8")
    migration = seed_template.split("    def _migrate_state", maxsplit=1)[1].split(
        "    def _upsert_secret", maxsplit=1
    )[0]

    assert migration.count('migrated["public_client_id"] = migrated["cli_client_id"]') == 1
    assert 'if "public_client_id" not in migrated and "cli_client_id" in migrated:' in migration


def test_zitadel_values_reference_precreated_secrets_for_sensitive_defaults() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")

    assert values["zitadelSecrets"]["prepared"] == {"secretName": "nemo-zitadel-prepared"}
    assert values["zitadelSecrets"]["masterkey"] == {
        "secretName": "zitadel-masterkey",
        "key": "masterkey",
    }
    assert values["zitadelSecrets"]["demo"] == {
        "secretName": "nemo-zitadel-prepared",
        "interactiveUserPasswordKey": "ZITADEL_INTERACTIVE_USER_PASSWORD",
    }
    assert values["zitadelSecrets"]["postgresql"] == {
        "secretName": "nemo-zitadel-prepared",
        "adminPasswordKey": "ZITADEL_POSTGRES_ADMIN_PASSWORD",
        "userPasswordKey": "ZITADEL_POSTGRES_USER_PASSWORD",
        "dsnKey": "ZITADEL_POSTGRES_DSN",
    }
    assert values["nemoHelixSecrets"]["ngc"] == {
        "secretName": "nemo-zitadel-prepared",
        "key": "NGC_API_KEY",
    }
    assert values["nemoHelixSecrets"]["postgresql"] == {
        "secretName": "nemo-zitadel-prepared",
        "passwordKey": "NEMO_POSTGRES_PASSWORD",
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
                    "name": "nemo-zitadel-prepared",
                    "key": "ZITADEL_POSTGRES_DSN",
                },
            },
        },
    ]
    assert values["zitadel"]["postgresql"]["fullnameOverride"] == "zitadel-postgresql"
    assert "password" not in values["zitadel"]["postgresql"]["auth"]
    assert "postgresPassword" not in values["zitadel"]["postgresql"]["auth"]
    assert values["zitadel"]["postgresql"]["auth"]["existingSecret"] == "nemo-zitadel-prepared"
    assert values["zitadel"]["postgresql"]["auth"]["secretKeys"] == {
        "adminPasswordKey": "ZITADEL_POSTGRES_ADMIN_PASSWORD",
        "userPasswordKey": "ZITADEL_POSTGRES_USER_PASSWORD",
    }
    assert values["nemo-helix"]["existingSecret"] == "nemo-zitadel-prepared"
    assert values["nemo-helix"]["ngcAPIKey"] == ""
    assert values["nemo-helix"]["postgresql"]["auth"]["existingSecret"] == "nemo-zitadel-prepared"
    assert values["nemo-helix"]["postgresql"]["auth"]["existingSecretPasswordKey"] == "NEMO_POSTGRES_PASSWORD"
    assert values["nemo-helix"]["secrets"]["defaultEncryptionKey"]["existingSecret"] == {
        "name": "nemo-zitadel-prepared",
        "key": "NHX_SECRETS_DEFAULT_ENCRYPTION_KEY",
    }
    assert not (HELM_DIR / "templates" / "generated-secrets.yaml").exists()
    assert "randAlphaNum" not in (HELM_DIR / "templates" / "_helpers.tpl").read_text(encoding="utf-8")


def test_helix_embedded_postgresql_honors_precreated_secret_password_key() -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    completed = subprocess.run(
        [
            "helm",
            "template",
            "nemo",
            "k8s/helm",
            "--set",
            "postgresql.auth.existingSecret=prepared",
            "--set",
            "postgresql.auth.existingSecretPasswordKey=NEMO_POSTGRES_PASSWORD",
            "--show-only",
            "templates/api/api-deployment.yaml",
            "--show-only",
            "templates/core/controller-deployment.yaml",
            "--show-only",
            "templates/postgres/postgres-statefulset.yaml",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=ZITADEL_SCRIPT_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    documents = [document for document in yaml.safe_load_all(completed.stdout) if document]
    password_secret_refs = []
    for document in documents:
        container = document["spec"]["template"]["spec"]["containers"][0]
        password_secret_refs.extend(
            env["valueFrom"]["secretKeyRef"]
            for env in container["env"]
            if env["name"] in {"DATABASE_PASSWORD", "POSTGRES_PASSWORD"}
        )

    assert password_secret_refs == [
        {"name": "prepared", "key": "NEMO_POSTGRES_PASSWORD"},
        {"name": "prepared", "key": "NEMO_POSTGRES_PASSWORD"},
        {"name": "prepared", "key": "NEMO_POSTGRES_PASSWORD"},
    ]


def test_zitadel_demo_files_avoid_empty_password_placeholders() -> None:
    manifest = (ZITADEL_DIR / "manifest.yaml").read_text(encoding="utf-8")
    values = (HELM_DIR / "values.yaml").read_text(encoding="utf-8")

    assert 'password: ""' not in manifest
    assert 'password: ""' not in values
    assert 'postgresPassword: ""' not in values


def test_zitadel_chart_references_precreated_workload_token_secrets_and_tokenreview_rbac() -> None:
    values = _load_yaml(HELM_DIR / "values.yaml")
    tokenreview_template = (HELM_DIR / "templates" / "tokenreview-rbac.yaml").read_text(encoding="utf-8")

    assert values["workloadTokenSigningKey"]["secretName"] == "nemo-workload-token-signing-key"
    assert values["workloadTokenSigningKey"]["key"] == "private-key.pem"
    assert values["workloadTokenTls"]["secretName"] == "nemo-helix-envoy-tls"
    assert values["nemo-helix"]["api"]["env"]["SSL_CERT_FILE"] == "/etc/nhx/workload-token-ca/ca.crt"
    assert values["nemo-helix"]["api"]["env"]["REQUESTS_CA_BUNDLE"] == "/etc/nhx/workload-token-ca/ca.crt"
    assert any(volume["name"] == "workload-token-tls-ca" for volume in values["nemo-helix"]["api"]["extraVolumes"])
    assert any(mount["name"] == "workload-token-tls-ca" for mount in values["nemo-helix"]["api"]["extraVolumeMounts"])
    jobs_config = values["nemo-helix"]["platformConfig"]["jobs"]
    assert jobs_config["executor_defaults"]["kubernetes_job"]["env"] == {
        "SSL_CERT_FILE": "/etc/nhx/workload-token-ca/ca.crt",
        "REQUESTS_CA_BUNDLE": "/etc/nhx/workload-token-ca/ca.crt",
    }
    workload_executor = next(
        executor
        for executor in jobs_config["executors"]
        if executor["provider"] == "cpu" and executor["profile"] == "workload"
    )
    workload_config = workload_executor["config"]
    assert workload_config["env"] == {
        "SSL_CERT_FILE": "/etc/nhx/workload-token-ca/ca.crt",
        "REQUESTS_CA_BUNDLE": "/etc/nhx/workload-token-ca/ca.crt",
    }
    assert workload_config["storage"] == jobs_config["executor_defaults"]["kubernetes_job"]["storage"]
    assert not (HELM_DIR / "templates" / "workload-token-signing-key-secret.yaml").exists()
    assert not (HELM_DIR / "templates" / "workload-token-tls.yaml").exists()
    assert 'resources: ["tokenreviews"]' in tokenreview_template
    assert 'verbs: ["create"]' in tokenreview_template
