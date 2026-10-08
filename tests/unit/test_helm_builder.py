# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The chart's builder: its build namespace, and the config and Jobs profiles it renders."""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from nemo_builder_plugin.config import BuilderConfig
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile

ROOT = Path(__file__).parent.parent.parent
HELM_DIR = ROOT / "k8s" / "helm"
HELM_TEMPLATE_TIMEOUT_SECONDS = 60
BUILD_PROFILES = ["build-fetch", "build-control", "build-push"]
ENABLED = ("--set", "builder.enabled=true")


def _helm_template(*args: str, values: dict | None = None) -> list[dict]:
    if shutil.which("helm") is None:
        pytest.skip("helm is required to render the NeMo Helix chart")

    completed = subprocess.run(
        ["helm", "template", "nemo-helix", str(HELM_DIR), "--namespace", "nhx", *args, "--values", "-"],
        input=yaml.safe_dump(values or {}),
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


def _profiles(config: dict) -> dict[str, KubernetesJobExecutionProfile]:
    """Every profile these tests render runs on kubernetes_job."""
    executors = config.get("jobs", {}).get("executors") or []
    profiles = [KubernetesJobExecutionProfile.model_validate(executor) for executor in executors]
    return {profile.profile: profile for profile in profiles}


def _in_namespace(documents: list[dict], namespace: str) -> dict[tuple[str, str], dict]:
    return {
        (document["kind"], document["metadata"]["name"]): document
        for document in documents
        if document["metadata"].get("namespace") == namespace
    }


def test_off_by_default() -> None:
    documents = _helm_template()
    config = _platform_config(documents)

    assert "builder" not in config
    assert not set(_profiles(config)) & set(BUILD_PROFILES)
    assert not any(document["kind"] == "Namespace" for document in documents)


def test_the_build_namespace_and_what_runs_there() -> None:
    documents = _helm_template(*ENABLED)

    namespace = next(document for document in documents if document["kind"] == "Namespace")
    assert namespace["metadata"]["name"] == "nhx-builds"
    assert namespace["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "baseline"
    resources = _in_namespace(documents, "nhx-builds")
    assert {name for kind, name in resources if kind == "ServiceAccount"} == {
        "nhx-build-fetch",
        "nhx-build-control",
        "nhx-build-push",
    }
    assert resources["PersistentVolumeClaim", "nhx-build-work"]["spec"]["accessModes"] == ["ReadWriteMany"]
    assert ("LimitRange", "nhx-builds") in resources
    assert ("ResourceQuota", "nhx-builds") in resources
    # The jobs controller runs the build jobs there, with a Role of its own.
    binding = resources["RoleBinding", "nemo-helix-core-controller"]
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "nemo-helix-core-controller", "namespace": "nhx"}]


def test_the_build_steps_profiles_and_config() -> None:
    config = _platform_config(_helm_template(*ENABLED))

    profiles = _profiles(config)
    for name in BUILD_PROFILES:
        profile = profiles[name]
        assert (profile.provider, profile.backend) == ("cpu", "kubernetes_job")
        assert profile.config.namespace == "nhx-builds"
        assert profile.config.service_account_name == name.replace("build-", "nhx-build-")
        assert profile.config.storage.pvc_name == "nhx-build-work"
        assert profile.config.cleanup_completed_jobs_immediately is True
    # Build pods run in another namespace, so the platform's URL must name its own.
    assert config["platform"]["base_url"] == "http://nemo-helix-api.nhx.svc:8080"


def test_a_users_profiles_keep_the_builders() -> None:
    values = {
        "builder": {"enabled": True},
        "platformConfig": {
            "jobs": {
                "executors": [
                    {"provider": "cpu", "profile": "extra", "backend": "kubernetes_job", "config": {}},
                    {
                        "provider": "cpu",
                        "profile": "build-push",
                        "backend": "kubernetes_job",
                        "config": {"namespace": "elsewhere"},
                    },
                ]
            }
        },
    }
    config = _platform_config(_helm_template(values=values))

    profiles = _profiles(config)
    assert list(profiles) == ["extra", "build-push", "build-fetch", "build-control"]
    # The user's own build-push replaces the chart's.
    assert profiles["build-push"].config.namespace == "elsewhere"


def test_the_development_registry() -> None:
    documents = _helm_template(
        *ENABLED, "--set", "builder.devRegistry.enabled=true", "--set", "builder.devRegistry.password=test-only"
    )

    builder = BuilderConfig.model_validate(_platform_config(documents)["builder"])
    assert builder.registry == "nemo-helix-builder-registry.nhx.svc.cluster.local:5000"
    assert builder.registry_plain_http
    assert any(
        document["kind"] == "Deployment" and document["metadata"]["name"] == "nemo-helix-builder-registry"
        for document in documents
    )


def test_the_development_registry_needs_a_password() -> None:
    with pytest.raises(subprocess.CalledProcessError) as refused:
        _helm_template(*ENABLED, "--set", "builder.devRegistry.enabled=true")
    assert "builder.devRegistry.password is required" in refused.value.stderr


def test_network_policies_let_the_build_steps_reach_the_api() -> None:
    documents = _helm_template(*ENABLED, "--set", "networkPolicies.enabled=true")

    policy = next(
        document
        for document in documents
        if document["kind"] == "NetworkPolicy" and document["metadata"]["name"] == "nemo-helix-api-ingress"
    )
    sources = [peer for rule in policy["spec"]["ingress"] for peer in rule["from"]]
    assert {
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "nhx-builds"}},
        "podSelector": {"matchLabels": {"app": "nemo-job", "nhx.nvidia.com/managed_by": "jobs-controller"}},
    } in sources


def test_the_build_namespace_takes_extra_labels_but_keeps_its_pod_security() -> None:
    labels = {"nvcr-imagepull": "enabled", "pod-security.kubernetes.io/enforce": "privileged"}
    documents = _helm_template(values={"builder": {"enabled": True, "namespaceLabels": labels}})

    namespace = next(document for document in documents if document["kind"] == "Namespace")
    assert namespace["metadata"]["labels"]["nvcr-imagepull"] == "enabled"
    assert namespace["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "baseline"


def test_the_jobs_controllers_role_carries_its_component_label() -> None:
    documents = _helm_template(*ENABLED, "--set", "core.controller.componentLabel=custom-controller")

    role = _in_namespace(documents, "nhx-builds")["Role", "nemo-helix-core-controller"]
    assert role["metadata"]["labels"]["app.kubernetes.io/component"] == "custom-controller"


class TestWithOpenSandboxOnTheCluster:
    """Jobs gives every job pod a required reference to the OpenSandbox key, which no build step uses."""

    def test_the_build_namespace_gets_a_placeholder_not_the_key(self) -> None:
        documents = _helm_template(*ENABLED, "--set", "sandboxClusterCapable=true")

        secret = _in_namespace(documents, "nhx-builds")["Secret", "opensandbox-server-api-key"]
        assert secret["stringData"] == {"api-key": "not-the-opensandbox-key"}

    def test_under_the_name_and_key_the_platform_config_gives(self) -> None:
        platform = {
            "sandbox_cluster_capable": True,
            "sandbox_api_key_secret": "os-key",
            "sandbox_api_key_secret_key": "k",
        }
        documents = _helm_template(values={"builder": {"enabled": True}, "platformConfig": {"platform": platform}})

        secret = _in_namespace(documents, "nhx-builds")["Secret", "os-key"]
        assert list(secret["stringData"]) == ["k"]

    def test_none_without_it(self) -> None:
        documents = _helm_template(*ENABLED)
        assert not [kind for kind, _ in _in_namespace(documents, "nhx-builds") if kind == "Secret"]


@pytest.mark.parametrize(
    ("args", "refusal"),
    [
        pytest.param(("--set", "builder.namespace=nhx"), "must not be the release namespace", id="release-namespace"),
        pytest.param(
            ("--set", "builder.limitRange.enabled=false"), "builder.resourceQuota needs builder.limitRange", id="quota"
        ),
    ],
)
def test_what_the_chart_refuses(args: tuple[str, ...], refusal: str) -> None:
    with pytest.raises(subprocess.CalledProcessError) as refused:
        _helm_template(*ENABLED, *args)
    assert refusal in refused.value.stderr
