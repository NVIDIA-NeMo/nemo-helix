# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent.parent
HELM_DIR = ROOT / "k8s" / "helm"
HELM_TEMPLATE_TIMEOUT_SECONDS = 60


def _helm_template(*args: str) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    completed = subprocess.run(
        ["helm", "template", "nemo-helix", str(HELM_DIR), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=HELM_TEMPLATE_TIMEOUT_SECONDS,
    )
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def _platform_config(documents: list[dict]) -> dict:
    config_map = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap" and "config.yaml" in document.get("data", {})
    )
    return yaml.safe_load(config_map["data"]["config.yaml"])


def _has_opensandbox_deployment(documents: list[dict]) -> bool:
    return any(
        document.get("kind") in {"Deployment", "StatefulSet"}
        and "opensandbox" in document.get("metadata", {}).get("name", "")
        for document in documents
    )


def test_default_sandbox_cluster_capable_is_false_and_does_not_install_opensandbox() -> None:
    documents = _helm_template()
    config = _platform_config(documents)

    assert config["platform"]["sandbox_cluster_capable"] is False
    assert _has_opensandbox_deployment(documents) is False


def test_sandbox_cluster_capable_wires_domain_protocol_and_secret() -> None:
    documents = _helm_template(
        "--set",
        "sandboxClusterCapable=true",
        "--set",
        "opensandbox.domain=opensandbox-server-kata.opensandbox-system.svc.cluster.local",
        "--set",
        "opensandbox.protocol=http",
        "--set",
        "opensandbox.apiKeySecret=opensandbox-server-api-key",
        "--set",
        "opensandbox.apiKeySecretKey=api-key",
    )
    config = _platform_config(documents)
    platform = config["platform"]

    assert platform["sandbox_cluster_capable"] is True
    assert platform["sandbox_server_domain"] == "opensandbox-server-kata.opensandbox-system.svc.cluster.local"
    assert platform["sandbox_server_protocol"] == "http"
    assert platform["sandbox_api_key_secret"] == "opensandbox-server-api-key"
    assert platform["sandbox_api_key_secret_key"] == "api-key"
    assert "sandbox_cluster_capable" not in config.get("rl", {})
    assert _has_opensandbox_deployment(documents) is False


SANDBOX_POLICIES = ("--set", "sandboxClusterCapable=true")


def _sandbox_policies(documents: list[dict]) -> list[dict]:
    return [
        document
        for document in documents
        if document["kind"] == "NetworkPolicy" and document["metadata"]["name"].endswith("-opensandbox-sandbox-ingress")
    ]


def test_sandbox_pods_admit_ingress_only_from_the_opensandbox_server_namespace() -> None:
    """Every sandbox pod, on every port, admits only the server's namespace: execd there takes commands unauthenticated."""
    (policy,) = _sandbox_policies(_helm_template(*SANDBOX_POLICIES))

    assert policy["metadata"]["namespace"] == "default"
    assert policy["spec"] == {
        "podSelector": {"matchExpressions": [{"key": "opensandbox.io/id", "operator": "Exists"}]},
        "policyTypes": ["Ingress"],
        "ingress": [
            {"from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "opensandbox-system"}}}]}
        ],
    }


def test_sandbox_policy_goes_to_the_sandbox_namespace_and_narrows_to_the_server_pods() -> None:
    """The policy lands where the server creates sandboxes, and a server pod selector narrows the namespace."""
    (policy,) = _sandbox_policies(
        _helm_template(
            *SANDBOX_POLICIES,
            "--set",
            "opensandbox.composeServices.sandboxNamespace=nemo-sandboxes",
            "--set",
            "opensandbox.sandboxNetworkPolicy.serverPodSelector.matchLabels.app=opensandbox-server",
        )
    )

    assert policy["metadata"]["namespace"] == "nemo-sandboxes"
    assert policy["spec"]["ingress"] == [
        {
            "from": [
                {
                    "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "opensandbox-system"}},
                    "podSelector": {"matchLabels": {"app": "opensandbox-server"}},
                }
            ]
        }
    ]


def test_sandbox_policy_is_rendered_whatever_the_chart_wide_network_policy_switch_says() -> None:
    """OpenSandbox alone turns the policy on: execd takes commands from anyone who reaches it."""
    assert _sandbox_policies(_helm_template(*SANDBOX_POLICIES, "--set", "networkPolicies.enabled=false"))


def test_sandbox_policy_is_not_rendered_without_opensandbox() -> None:
    assert _sandbox_policies(_helm_template("--set", "networkPolicies.enabled=true")) == []


def test_extra_ingress_can_admit_everything() -> None:
    """With no off switch, an empty rule is how an operator opens sandboxes up."""
    (policy,) = _sandbox_policies(
        _helm_template(*SANDBOX_POLICIES, "--set-json", "opensandbox.sandboxNetworkPolicy.extraIngress=[{}]")
    )

    assert policy["spec"]["ingress"][-1] == {}


def test_sandbox_policy_refuses_to_admit_the_whole_sandbox_namespace() -> None:
    """A server in the sandbox namespace needs a pod selector, or the policy would admit the sandboxes themselves."""
    with pytest.raises(subprocess.CalledProcessError) as refused:
        _helm_template(
            *SANDBOX_POLICIES,
            "--set",
            "opensandbox.namespace=default",
            "--set",
            "opensandbox.domain=opensandbox-server.default.svc.cluster.local",
        )

    assert "opensandbox.sandboxNetworkPolicy.serverPodSelector is required" in refused.value.stderr


def test_sandbox_policy_refuses_a_namespace_the_server_domain_contradicts() -> None:
    """A wrong opensandbox.namespace would shut the server out of every sandbox, so the chart refuses it."""
    with pytest.raises(subprocess.CalledProcessError) as refused:
        _helm_template(*SANDBOX_POLICIES, "--set", "opensandbox.namespace=elsewhere")

    assert "doesn't match the namespace in opensandbox.domain" in refused.value.stderr


def test_sandbox_policy_skips_the_domain_check_for_an_external_domain() -> None:
    """A gateway or external hostname names no namespace, so only opensandbox.namespace decides."""
    (policy,) = _sandbox_policies(
        _helm_template(
            *SANDBOX_POLICIES,
            "--set",
            "opensandbox.domain=sandbox.example.com",
            "--set",
            "opensandbox.namespace=elsewhere",
        )
    )

    assert policy["spec"]["ingress"][0]["from"][0]["namespaceSelector"]["matchLabels"] == {
        "kubernetes.io/metadata.name": "elsewhere"
    }
