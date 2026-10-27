# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"
PULLER_LABELS = {
    "nhx.nvidia.com/managed-by": "models-controller",
    "nhx.nvidia.com/models-role": "puller",
}


def render_api_policy(tmp_path: Path, values: dict) -> dict | None:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")
    values_path = tmp_path / "values.yaml"
    values_path.write_text(yaml.safe_dump(values))
    result = subprocess.run(
        ["helm", "template", "test", str(HELM_DIR), "--no-hooks", "-f", str(values_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return next(
        (
            doc
            for doc in yaml.safe_load_all(result.stdout)
            if doc and doc["kind"] == "NetworkPolicy" and doc["metadata"]["name"].endswith("-api-ingress")
        ),
        None,
    )


@pytest.mark.parametrize("api_values,port", [({}, 8080), ({"service": {"port": 9090}}, 9090)])
def test_api_policy_allows_same_namespace_model_pullers(tmp_path: Path, api_values: dict, port: int) -> None:
    policy = render_api_policy(tmp_path, {"api": api_values, "networkPolicies": {"enabled": True}})
    assert policy is not None
    assert policy["spec"]["policyTypes"] == ["Ingress"]
    rules = policy["spec"]["ingress"]
    # Exact peer shape matters: adding namespaceSelector: {} would allow other namespaces.
    assert {
        "from": [{"podSelector": {"matchLabels": PULLER_LABELS}}],
        "ports": [{"protocol": "TCP", "port": port}],
    } in rules


@pytest.mark.parametrize(
    "labels",
    [
        {},
        {"nhx.nvidia.com/managed-by": "models-controller"},
        {"nhx.nvidia.com/models-role": "puller"},
        {**PULLER_LABELS, "nhx.nvidia.com/models-role": "server"},
        {**PULLER_LABELS, "nhx.nvidia.com/managed-by": "sandbox-controller"},
        {"app": "opensandbox"},
    ],
    ids=["unlabeled", "controller-only", "role-only", "model-server", "other-controller", "sandbox"],
)
def test_default_api_peers_do_not_admit_untrusted_pods(tmp_path: Path, labels: dict[str, str]) -> None:
    policy = render_api_policy(tmp_path, {"networkPolicies": {"enabled": True}})
    assert policy is not None
    for rule in policy["spec"]["ingress"]:
        for peer in rule["from"]:
            assert set(peer) == {"podSelector"}
            selector = peer["podSelector"]
            assert set(selector) == {"matchLabels"}
            assert selector["matchLabels"]
            assert not selector["matchLabels"].items() <= labels.items()


def test_model_puller_rule_can_be_the_only_api_allowance(tmp_path: Path) -> None:
    policy = render_api_policy(
        tmp_path,
        {
            "builder": {"enabled": False},
            "networkPolicies": {
                "enabled": True,
                "api": {"sameReleasePods": {"enabled": False}, "managedJobs": {"enabled": False}, "port": 8443},
            },
        },
    )
    assert policy is not None
    assert policy["spec"]["ingress"] == [
        {
            "from": [{"podSelector": {"matchLabels": PULLER_LABELS}}],
            "ports": [{"protocol": "TCP", "port": 8443}],
        }
    ]


def test_model_puller_allowance_can_be_disabled(tmp_path: Path) -> None:
    values = {"networkPolicies": {"enabled": True}}
    default_policy = render_api_policy(tmp_path, values)
    policy = render_api_policy(
        tmp_path, {"networkPolicies": {"enabled": True, "api": {"modelPullers": {"enabled": False}}}}
    )
    assert default_policy is not None and policy is not None
    rules = default_policy["spec"]["ingress"]
    rules.remove(
        {"from": [{"podSelector": {"matchLabels": PULLER_LABELS}}], "ports": [{"protocol": "TCP", "port": 8080}]}
    )
    assert policy == default_policy


def test_api_policy_deny_all_without_allowances(tmp_path: Path) -> None:
    policy = render_api_policy(
        tmp_path,
        {
            "builder": {"enabled": False},
            "networkPolicies": {
                "enabled": True,
                "api": {
                    "sameReleasePods": {"enabled": False},
                    "managedJobs": {"enabled": False},
                    "modelPullers": {"enabled": False},
                },
            },
        },
    )
    assert policy is not None
    assert not policy["spec"].get("ingress")
    assert policy["spec"]["policyTypes"] == ["Ingress"]


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"networkPolicies": {"enabled": True, "api": {"enabled": False}}},
        {"api": {"enabled": False}, "networkPolicies": {"enabled": True}},
    ],
    ids=["network-policies-disabled", "api-policy-disabled", "api-disabled"],
)
def test_disabled_api_policy_does_not_render(tmp_path: Path, values: dict) -> None:
    assert render_api_policy(tmp_path, values) is None
