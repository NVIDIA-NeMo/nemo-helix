# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"
COMPONENT_LABEL = "app.kubernetes.io/component"


@pytest.mark.parametrize(
    ("api_label", "controller_label", "override"),
    [
        pytest.param("nhx-api", "nhx-core-controller", False, id="defaults"),
        pytest.param("nmp-api", "nmp-core-controller", True, id="upgrade"),
        pytest.param("custom.api-v1", "custom.controller-v1", True, id="custom"),
    ],
)
def test_component_labels_match_resource_selectors(api_label: str, controller_label: str, override: bool) -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    # Existing releases also retain the name and instance parts of their selectors.
    args = [
        "--set",
        "nameOverride=existing-platform,fullnameOverride=existing-release,"
        "api.serviceMonitor.enabled=true,core.serviceMonitor.enabled=true,"
        "networkPolicies.enabled=true,api.podDisruptionBudget.enabled=true,"
        "api.autoscaling.enabled=true,api.resources.requests.cpu=100m",
    ]
    if override:
        args += ["--set-string", f"api.componentLabel={api_label},core.controller.componentLabel={controller_label}"]
    result = subprocess.run(
        ["helm", "template", "upgrade-test", str(HELM_DIR), "--no-hooks", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    resources = {(doc["kind"], doc["metadata"]["name"]): doc for doc in yaml.safe_load_all(result.stdout) if doc}
    for suffix, component in [("api", api_label), ("core-controller", controller_label)]:
        name = f"existing-release-{suffix}"
        expected_selector = {
            "app.kubernetes.io/name": "existing-platform",
            "app.kubernetes.io/instance": "upgrade-test",
            COMPONENT_LABEL: component,
        }
        deployment = resources["Deployment", name]
        pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
        service = resources["Service", name]
        assert deployment["spec"]["selector"]["matchLabels"] == expected_selector
        assert expected_selector.items() <= pod_labels.items()
        assert service["spec"]["selector"] == expected_selector

        monitor = resources["ServiceMonitor", name]
        assert monitor["spec"]["selector"]["matchLabels"][COMPONENT_LABEL] == component
        assert monitor["spec"]["selector"]["matchLabels"].items() <= service["metadata"]["labels"].items()
        policy = resources["NetworkPolicy", f"{name}-ingress"]
        assert policy["spec"]["podSelector"]["matchLabels"] == expected_selector
        if suffix == "api":
            pdb = resources["PodDisruptionBudget", name]
            assert pdb["spec"]["selector"]["matchLabels"] == expected_selector
            jobs_policy = resources["NetworkPolicy", "existing-release-jobs-egress"]
            destinations = [peer for rule in jobs_policy["spec"]["egress"] for peer in rule["to"]]
            assert {"podSelector": {"matchLabels": expected_selector}} in destinations
