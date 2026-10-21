# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"
WORKLOAD_VALUES = HELM_DIR / "ci" / "21-api-extra-volumes.yaml"


def render(*args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")
    return subprocess.run(
        ["helm", "template", "transport-test", str(HELM_DIR), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def container_env(resource: dict) -> dict[str, str]:
    container = resource["spec"]["template"]["spec"]["containers"][0]
    return {item["name"]: item["value"] for item in container.get("env", []) if "value" in item}


@pytest.mark.parametrize("workload", [False, True])
def test_internal_url_reaches_all_platform_callers(workload: bool) -> None:
    args = ["-f", str(WORKLOAD_VALUES)] if workload else []
    url = "https://helix.internal.example.test"
    result = render(*args, "--set", f"api.internalBaseUrl={url}/", "--set", "platformSeedJob.enabled=true")
    assert result.returncode == 0, result.stderr
    resources = {(doc["kind"], doc["metadata"]["name"]): doc for doc in yaml.safe_load_all(result.stdout) if doc}
    configmap = resources["ConfigMap", "transport-test-nemo-helix-config"]
    config = yaml.safe_load(configmap["data"]["config.yaml"])
    assert config["platform"]["base_url"] == url
    assert config["platform"]["advertised_base_url"] == url
    assert config["auth"]["policy_decision_point_base_url"] == url

    controller = container_env(resources["Deployment", "transport-test-nemo-helix-core-controller"])
    assert controller["NHX_BASE_URL"] == url
    assert controller["NHX_INTERNAL_BASE_URL"] == url
    api = container_env(resources["Deployment", "transport-test-nemo-helix-api"])
    assert api["NHX_INTERNAL_BASE_URL"] == url
    if workload:
        assert controller["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == url
        assert api["NHX_BASE_URL"] == "http://localhost:8080"
        assert api["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "http://localhost:8080"
    else:
        assert api["NHX_BASE_URL"] == url
    seed = next(doc for (kind, _), doc in resources.items() if kind == "Job" and "seed" in doc["metadata"]["name"])
    assert container_env(seed)["NHX_BASE_URL"] == url


@pytest.mark.parametrize("url", ["", "http://nemo-helix-api:8080", "http://localhost:8080", "https://"])
def test_workload_exchange_rejects_missing_or_cleartext_internal_url(url: str) -> None:
    result = render("-f", str(WORKLOAD_VALUES), "--set-string", f"api.internalBaseUrl={url}")
    assert result.returncode != 0
    assert "auth.oidc.workload requires api.internalBaseUrl" in result.stderr


@pytest.mark.parametrize("field", ["base_url", "advertised_base_url"])
def test_workload_exchange_rejects_cleartext_platform_override(field: str) -> None:
    result = render("-f", str(WORKLOAD_VALUES), "--set", f"platformConfig.platform.{field}=http://nemo-helix-api:8080")
    assert result.returncode != 0
    assert f"platformConfig.platform.{field} to use HTTPS" in result.stderr


@pytest.mark.parametrize(
    "auth_args",
    [
        [],
        ["--set", "platformConfig.auth.enabled=true"],
        ["--set", "platformConfig.auth.enabled=true,platformConfig.auth.oidc.workload=null"],
        ["--set", "platformConfig.auth.enabled=false,platformConfig.auth.oidc.workload.client_id=unused"],
    ],
)
def test_http_defaults_remain_supported_without_workload_exchange(auth_args: list[str]) -> None:
    result = render(*auth_args)
    assert result.returncode == 0, result.stderr
    resources = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    controller = next(
        doc for doc in resources if doc["kind"] == "Deployment" and doc["metadata"]["name"].endswith("-core-controller")
    )
    assert container_env(controller)["NHX_BASE_URL"] == "http://transport-test-nemo-helix-api:8080"


def test_internal_url_supports_release_name_templates() -> None:
    result = render(
        "-f", str(WORKLOAD_VALUES), "--set-string", "api.internalBaseUrl=https://{{ .Release.Name }}.example.test"
    )
    assert result.returncode == 0, result.stderr
    resources = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    controller = next(
        doc for doc in resources if doc["kind"] == "Deployment" and doc["metadata"]["name"].endswith("-core-controller")
    )
    assert container_env(controller)["NHX_BASE_URL"] == "https://transport-test.example.test"
