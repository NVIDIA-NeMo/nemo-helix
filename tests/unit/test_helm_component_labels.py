# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

HELM_DIR = Path(__file__).resolve().parents[2] / "k8s" / "helm"
COMPONENT_LABEL = "app.kubernetes.io/component"

# volumeClaimTemplates rendered by published chart 0.3.1 for release "customer".
# Chart archive SHA256: d158f976dfe19bb17812ae86576800049f9640bf622654df92bf635a001abbe4
LEGACY_POSTGRES_CLAIMS = [
    {
        "metadata": {
            "name": "data",
            "labels": {
                COMPONENT_LABEL: "postgres",
                "helm.sh/chart": "nemo-platform-0.3.1",
                "app.kubernetes.io/name": "nemo-platform",
                "app.kubernetes.io/instance": "customer",
                "app.kubernetes.io/version": "0.3.0",
                "app.kubernetes.io/managed-by": "Helm",
            },
        },
        "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "5Gi"}}},
    }
]


def render_postgres(tmp_path: Path, values: dict, chart: Path = HELM_DIR) -> tuple[str, dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")
    values_path = tmp_path / "values.yaml"
    values_path.write_text(yaml.safe_dump(values))
    result = subprocess.run(
        ["helm", "template", "customer", str(chart), "--no-hooks", "-f", str(values_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    postgres = next(
        (
            doc
            for doc in yaml.safe_load_all(result.stdout)
            if doc and doc["kind"] == "StatefulSet" and doc["metadata"]["name"].endswith("-postgres")
        ),
        {},
    )
    return result.stdout, postgres


@pytest.mark.parametrize("persistence", [{}, {"claimTemplateLabels": {}}], ids=["omitted", "empty"])
def test_postgres_default_claim_template_is_unchanged(tmp_path: Path, persistence: dict) -> None:
    _, postgres = render_postgres(tmp_path, {"postgresql": {"persistence": persistence}})
    expected = deepcopy(LEGACY_POSTGRES_CLAIMS)
    expected[0]["metadata"]["labels"] = {COMPONENT_LABEL: "postgres"}
    assert postgres["spec"]["volumeClaimTemplates"] == expected


def test_postgres_preserves_published_legacy_claim_template(tmp_path: Path) -> None:
    values = {"postgresql": {"persistence": {"claimTemplateLabels": LEGACY_POSTGRES_CLAIMS[0]["metadata"]["labels"]}}}
    output, postgres = render_postgres(tmp_path, values)
    assert postgres["spec"]["volumeClaimTemplates"] == LEGACY_POSTGRES_CLAIMS
    # A duplicate component key can be silently discarded by YAML loaders.
    postgres_yaml = output.split("# Source: nemo-helix/templates/postgres/postgres-statefulset.yaml\n", 1)[1]
    claim_yaml = postgres_yaml.split("\n---", 1)[0].split("  volumeClaimTemplates:\n", 1)[1]
    assert claim_yaml.count(f"{COMPONENT_LABEL}:") == 1


def test_postgres_claim_labels_are_literal_and_scoped(tmp_path: Path) -> None:
    _, original = render_postgres(tmp_path, {})
    labels = {COMPONENT_LABEL: "existing-postgres", "example.com/literal": "{{ .Chart.Version }}"}
    _, postgres = render_postgres(tmp_path, {"postgresql": {"persistence": {"claimTemplateLabels": labels}}})
    assert postgres["spec"]["volumeClaimTemplates"][0]["metadata"]["labels"] == labels
    postgres["spec"]["volumeClaimTemplates"] = original["spec"]["volumeClaimTemplates"]
    assert postgres == original


def test_postgres_legacy_claim_labels_survive_version_changes(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    shutil.copytree(HELM_DIR, chart)
    values = {"postgresql": {"persistence": {"claimTemplateLabels": LEGACY_POSTGRES_CLAIMS[0]["metadata"]["labels"]}}}
    metadata_path = chart / "Chart.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    for version in ("0.7.99", "0.8.99"):
        metadata.update(version=version, appVersion=version)
        metadata_path.write_text(yaml.safe_dump(metadata))
        _, postgres = render_postgres(tmp_path, values, chart)
        assert postgres["spec"]["volumeClaimTemplates"] == LEGACY_POSTGRES_CLAIMS


@pytest.mark.parametrize("enabled", [False, True], ids=["postgres-disabled", "persistence-disabled"])
def test_postgres_disabled_storage_ignores_claim_labels(tmp_path: Path, enabled: bool) -> None:
    values = yaml.safe_load((HELM_DIR / "ci/25-postgres-claim-template-labels.yaml").read_text())
    values["postgresql"]["enabled"] = enabled
    values["postgresql"]["persistence"]["enabled"] = False
    # Disabling embedded Postgres requires an external database connection.
    values["externalDatabase"] = {"host": "postgres.example.com", "password": "test-password"}
    _, postgres = render_postgres(tmp_path, values)
    if enabled:
        assert "volumeClaimTemplates" not in postgres["spec"]
    else:
        assert not postgres


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
