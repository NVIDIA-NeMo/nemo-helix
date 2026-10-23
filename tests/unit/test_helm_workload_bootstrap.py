# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s/helm"
BASE_URL = "https://platform.example.test"


def render(tmp_path: Path, values: dict) -> subprocess.CompletedProcess[str]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required")
    values_file = tmp_path / "values.yaml"
    values_file.write_text(yaml.safe_dump(values))
    return subprocess.run(
        ["helm", "template", "bootstrap", str(HELM_DIR), "-f", str(values_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def workload_values(base_url: str = BASE_URL) -> dict:
    return {
        "platformConfig": {
            "platform": {"base_url": base_url},
            "auth": {
                "enabled": True,
                "oidc": {"workload": {"client_id": "bootstrap"}},
            },
        }
    }


def env_for(document: dict) -> dict:
    container = document["spec"]["template"]["spec"]["containers"][0]
    names = [entry["name"] for entry in container["env"]]
    assert len(names) == len(set(names)), "Helm 4 server-side apply rejects duplicate environment entries"
    return {entry["name"]: entry.get("value") for entry in container["env"]}


def test_exchange_disabled_keeps_http_default(tmp_path: Path) -> None:
    result = render(tmp_path, {"platformConfig": {"auth": {"enabled": True}}})
    assert result.returncode == 0, result.stderr
    assert 'value: "http://bootstrap-nemo-helix-api:8080"' in result.stdout


def test_exchange_rejects_default_http_endpoint(tmp_path: Path) -> None:
    values = workload_values()
    del values["platformConfig"]["platform"]
    result = render(tmp_path, values)
    assert result.returncode != 0
    assert "set platformConfig.platform.base_url" in result.stderr


@pytest.mark.parametrize("advertised", [None, "https://public.example.test"])
def test_one_https_url_reaches_platform_callers(tmp_path: Path, advertised: str | None) -> None:
    values = workload_values(BASE_URL + "/")
    values["platformSeedJob"] = {"enabled": True}
    if advertised:
        values["platformConfig"]["platform"]["advertised_base_url"] = advertised
    result = render(tmp_path, values)
    assert result.returncode == 0, result.stderr
    documents = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    deployments = {doc["metadata"]["name"]: doc for doc in documents if doc["kind"] == "Deployment"}
    for suffix in ("api", "core-controller"):
        deployment = deployments[f"bootstrap-nemo-helix-{suffix}"]
        env = env_for(deployment)
        assert env["NHX_BASE_URL"] == BASE_URL
        assert env["NHX_INTERNAL_BASE_URL"] == BASE_URL
        expected_pdp = "http://localhost:8080" if suffix == "api" else BASE_URL
        assert env["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == expected_pdp
        # Exercise the default startup path, not a selected subset of controllers.
        if suffix == "core-controller":
            assert "--controller-group=all" in deployment["spec"]["template"]["spec"]["containers"][0]["args"]
    seed = next(doc for doc in documents if doc["kind"] == "Job" and doc["metadata"]["name"].endswith("-platform-seed"))
    assert env_for(seed)["NHX_BASE_URL"] == BASE_URL
    configmap = next(doc for doc in documents if doc["kind"] == "ConfigMap" and "config.yaml" in doc.get("data", {}))
    config = yaml.safe_load(configmap["data"]["config.yaml"])
    assert config["platform"]["base_url"].rstrip("/") == BASE_URL
    assert config["auth"]["policy_decision_point_base_url"].rstrip("/") == BASE_URL
    if advertised:
        assert config["platform"]["advertised_base_url"] == advertised
    else:
        assert not config["platform"].get("advertised_base_url")


def test_exchange_rejects_explicit_http_advertised_url(tmp_path: Path) -> None:
    values = workload_values()
    values["platformConfig"]["platform"]["advertised_base_url"] = "http://public.example.test"
    result = render(tmp_path, values)
    assert result.returncode != 0
    assert "HTTPS platform.advertised_base_url" in result.stderr


def test_explicit_pdp_config_is_preserved(tmp_path: Path) -> None:
    values = workload_values()
    values["platformConfig"]["auth"].update(
        policy_decision_point_provider="opa",
        policy_decision_point_base_url="https://policy.example.test",
    )
    result = render(tmp_path, values)
    assert result.returncode == 0, result.stderr
    assert "policy_decision_point_base_url: https://policy.example.test" in result.stdout


def test_kubernetes_regression_keeps_generic_startup_and_discovery_fallback(tmp_path: Path) -> None:
    root = HELM_DIR.parents[1]
    values = yaml.safe_load((root / "e2e/k8s/values/workload-bootstrap.yaml").read_text())
    assert values["platformConfig"]["platform"] == {"base_url": "https://workload-tls-gateway"}
    assert "issuer" not in values["platformConfig"]["auth"]["token_signing"]
    assert values["platformConfig"]["auth"]["access_keys"]["enabled"] is True
    assert "controllerGroup" not in values["core"]["controller"]
    assert "controllers" not in values["core"]["controller"]
    assert "serviceGroup" not in values["api"]
    assert "services" not in values["api"]
    result = render(tmp_path, values)
    assert result.returncode == 0, result.stderr
    documents = [doc for doc in yaml.safe_load_all(result.stdout) if doc]
    controller = next(
        doc
        for doc in documents
        if doc["kind"] == "Deployment" and doc["metadata"]["name"] == "workload-bootstrap-core-controller"
    )
    assert "--controller-group=all" in controller["spec"]["template"]["spec"]["containers"][0]["args"]
    assert env_for(controller)["NHX_AUTH_POLICY_DECISION_POINT_BASE_URL"] == "https://workload-tls-gateway"
    for doc in documents:
        if doc["kind"] in {"Deployment", "Job"} and doc["metadata"]["name"] in {
            "workload-bootstrap-api",
            "workload-bootstrap-core-controller",
            "workload-bootstrap-platform-seed",
        }:
            env = env_for(doc)
            assert env["NHX_BASE_URL"] == "https://workload-tls-gateway"
            assert env["NHX_CLIENT_SSL_CERT_FILE"] == "/etc/workload-ca/ca.crt"
            assert env["SSL_CERT_FILE"] == "/etc/workload-ca/ca.crt"
    jobs = yaml.safe_load((root / ".github/workflows/ci.yaml").read_text())["jobs"]
    assert "kind-workload-bootstrap" in jobs["ci-status"]["needs"]
    steps = jobs["kind-workload-bootstrap"]["steps"]
    assert any("test_workload_bootstrap.sh" in step.get("run", "") for step in steps)
