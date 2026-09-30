# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from backends.k8s.k8s_helpers import sample_always_config, sample_config, with_workload_identity
from kubernetes.client import ApiClient
from nemo_deployments_plugin.backends.k8s.compiler import (
    DeploymentConfigError,
    ExecutorK8sDefaults,
    _build_probe,
    build_configmap_body,
    build_env_vars,
    build_secret_body,
    build_secret_env_from,
    compile_workload,
    configmap_data_key,
    validate_config_for_deployment,
    validate_config_for_job,
)
from nemo_deployments_plugin.backends.k8s.deployments import build_deployment_body
from nemo_deployments_plugin.backends.k8s.jobs import build_job_body
from nemo_deployments_plugin.backends.labels import k8s_deployment_secret_name
from nemo_deployments_plugin.entities import (
    ConfigFile,
    Container,
    ContainerPort,
    EnvVar,
    ExecAction,
    HTTPGetAction,
    K8sDeploymentConfig,
    Probe,
    SecretRef,
    VolumeMount,
    WorkloadIdentitySpec,
)
from nemo_helix_plugin.auth.workload_identity import (
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    WORKLOAD_IDENTITY_VOLUME_NAME,
    WORKLOAD_IDENTITY_VOLUME_PATH,
)
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nemo_helix_plugin.config import ImagePullSecret


def _serialized(obj: object) -> dict:
    return ApiClient().sanitize_for_serialization(obj)


def test_build_probe_exec_action() -> None:
    # V1Probe requires the `_exec` kwarg (exec is a Python keyword); regression
    # for the exec-probe path used by loopback-only sidecars.
    probe = _build_probe(Probe(exec=ExecAction(command=["sh", "-c", "curl -sf http://127.0.0.1:8090/healthz"])))
    serialized = _serialized(probe)
    assert serialized["exec"]["command"] == ["sh", "-c", "curl -sf http://127.0.0.1:8090/healthz"]


def test_build_probe_http_get_action() -> None:
    probe = _build_probe(Probe(httpGet=HTTPGetAction(path="/health", port=8000)))
    serialized = _serialized(probe)
    assert serialized["httpGet"]["path"] == "/health"
    assert serialized["httpGet"]["port"] == 8000


def test_configmap_data_key_sanitizes_paths() -> None:
    assert configmap_data_key("/etc/app/config.yaml") == "etc__app__config.yaml"


def test_compile_job_pod_spec_single_container() -> None:
    config = sample_config(restart_policy="Never")
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Never",
    )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert pod_spec["restart_policy"] == "Never"
    assert len(pod_spec["containers"]) == 1
    assert pod_spec["containers"][0]["name"] == "main"
    assert compiled.configmap_body is None


def test_compile_deployment_includes_init_and_sidecar() -> None:
    config = sample_always_config().model_copy(
        update={
            "init_containers": [
                Container(name="bootstrap", image="busybox", command=["sh", "-c", "echo hi"]),
                Container.model_validate(
                    {
                        "name": "sidecar",
                        "image": "nginx:alpine",
                        "restartPolicy": "Always",
                        "ports": [{"name": "proxy", "containerPort": 8081}],
                        "livenessProbe": {"httpGet": {"path": "/healthz", "port": 8081}},
                    }
                ),
            ],
            "containers": [
                Container(name="main", image="nginx:alpine", ports=[ContainerPort(name="http", containerPort=8080)]),
                Container(
                    name="metrics",
                    image="prom/node-exporter",
                    ports=[ContainerPort(name="metrics", containerPort=9100)],
                ),
            ],
        }
    )
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Always",
    )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert pod_spec == {
        "restart_policy": "Always",
        "init_containers": [
            {
                "name": "bootstrap",
                "image": "busybox",
                "command": ["sh", "-c", "echo hi"],
            },
            {
                "name": "sidecar",
                "image": "nginx:alpine",
                "restartPolicy": "Always",
                "ports": [{"name": "proxy", "containerPort": 8081, "protocol": "TCP"}],
                "livenessProbe": {
                    "httpGet": {"path": "/healthz", "port": 8081, "scheme": "HTTP"},
                    "initialDelaySeconds": 0,
                    "timeoutSeconds": 1,
                    "periodSeconds": 10,
                    "failureThreshold": 3,
                },
            },
        ],
        "containers": [
            {
                "name": "main",
                "image": "nginx:alpine",
                "ports": [{"name": "http", "containerPort": 8080, "protocol": "TCP"}],
            },
            {
                "name": "metrics",
                "image": "prom/node-exporter",
                "ports": [{"name": "metrics", "containerPort": 9100, "protocol": "TCP"}],
            },
        ],
    }
    assert len(compiled.service_containers) == 2


def test_compile_applies_k8s_deployment_config() -> None:
    config = sample_always_config()
    k8s_config = K8sDeploymentConfig.model_validate(
        {
            "serviceAccount": "deploy-sa",
            "tolerations": [{"key": "gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}],
            "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": []}}},
            "securityContext": {"runAsUser": 1000, "fsGroup": 2000},
        }
    )
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=k8s_config,
        pod_restart_policy="Always",
    )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert pod_spec["service_account_name"] == "deploy-sa"
    assert pod_spec["tolerations"][0]["key"] == "gpu"
    affinity = compiled.pod_spec_kwargs["affinity"]
    assert affinity.node_affinity is not None
    security_context = compiled.pod_spec_kwargs["security_context"]
    assert security_context.run_as_user == 1000


def test_compile_workload_identity_prefers_workload_service_account() -> None:
    config = with_workload_identity(sample_always_config(), service_account_name="workload-sa")
    k8s_config = K8sDeploymentConfig.model_validate({"serviceAccount": "pod-sa"})
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=k8s_config,
        pod_restart_policy="Always",
    )

    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert pod_spec["service_account_name"] == "workload-sa"
    assert any(volume["name"] == WORKLOAD_IDENTITY_VOLUME_NAME for volume in pod_spec["volumes"])


def test_compile_applies_node_selector() -> None:
    config = sample_always_config()
    k8s_config = K8sDeploymentConfig.model_validate({"nodeSelector": {"gpu": "a100", "zone": "us-west1-a"}})
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=k8s_config,
        pod_restart_policy="Always",
    )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert pod_spec["node_selector"] == {"gpu": "a100", "zone": "us-west1-a"}


def test_compile_applies_topology_spread_constraints() -> None:
    config = sample_always_config()
    k8s_config = K8sDeploymentConfig.model_validate(
        {
            "topologySpreadConstraints": [
                {
                    "maxSkew": 1,
                    "topologyKey": "kubernetes.io/hostname",
                    "whenUnsatisfiable": "DoNotSchedule",
                    "labelSelector": {"matchLabels": {"app": "x"}},
                }
            ]
        }
    )
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=k8s_config,
        pod_restart_policy="Always",
    )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    tsc = pod_spec["topology_spread_constraints"]
    assert len(tsc) == 1
    assert tsc[0]["maxSkew"] == 1
    assert tsc[0]["topologyKey"] == "kubernetes.io/hostname"
    assert tsc[0]["whenUnsatisfiable"] == "DoNotSchedule"


def test_compile_carries_pod_annotations() -> None:
    config = sample_config(restart_policy="Never")
    k8s_config = K8sDeploymentConfig.model_validate({"podAnnotations": {"sidecar.istio.io/nativeSidecar": "true"}})
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=k8s_config,
        pod_restart_policy="Never",
    )
    assert compiled.pod_annotations == {"sidecar.istio.io/nativeSidecar": "true"}
    # Annotations live on the pod-template metadata, not the pod spec.
    assert "annotations" not in compiled.pod_spec_kwargs


def test_compile_applies_executor_defaults() -> None:
    # Executor-level defaults (shared by every consumer: models, agents, ...) land
    # on a workload with no per-entity k8s config.
    config = sample_always_config()
    executor_defaults = ExecutorK8sDefaults(
        pod_annotations={"sidecar.istio.io/nativeSidecar": "true"},
        node_selector={"gpu": "a100"},
        tolerations=[{"key": "gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"}],
        affinity={
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [{"matchExpressions": [{"key": "gpu", "operator": "In", "values": ["a100"]}]}]
                }
            }
        },
        topology_spread_constraints=[
            {"maxSkew": 1, "topologyKey": "kubernetes.io/hostname", "whenUnsatisfiable": "DoNotSchedule"}
        ],
    )
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Always",
        executor_defaults=executor_defaults,
    )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert compiled.pod_annotations == {"sidecar.istio.io/nativeSidecar": "true"}
    assert pod_spec["node_selector"] == {"gpu": "a100"}
    assert pod_spec["tolerations"][0]["key"] == "gpu"
    assert compiled.pod_spec_kwargs["affinity"].node_affinity is not None
    assert pod_spec["topology_spread_constraints"][0]["topologyKey"] == "kubernetes.io/hostname"


def test_compile_per_entity_wins_over_executor_defaults() -> None:
    # A per-entity K8sDeploymentConfig overrides the executor default: annotations
    # merge key-wise (entity key wins), and node_selector / tolerations are applied
    # from the entity wholesale (executor default not additionally applied).
    config = sample_always_config()
    k8s_config = K8sDeploymentConfig.model_validate(
        {
            "podAnnotations": {"sidecar.istio.io/nativeSidecar": "false", "team": "a"},
            "nodeSelector": {"zone": "us-west1-a"},
            "tolerations": [{"key": "entity", "operator": "Exists"}],
        }
    )
    executor_defaults = ExecutorK8sDefaults(
        pod_annotations={"sidecar.istio.io/nativeSidecar": "true", "platform": "nhx"},
        node_selector={"gpu": "a100"},
        tolerations=[{"key": "platform", "operator": "Exists"}],
    )
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=k8s_config,
        pod_restart_policy="Always",
        executor_defaults=executor_defaults,
    )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    # Annotations: entity value wins for the shared key; non-conflicting keys from
    # both sides are retained.
    assert compiled.pod_annotations == {
        "sidecar.istio.io/nativeSidecar": "false",
        "team": "a",
        "platform": "nhx",
    }
    # node_selector / tolerations: entity wins wholesale.
    assert pod_spec["node_selector"] == {"zone": "us-west1-a"}
    assert [t["key"] for t in pod_spec["tolerations"]] == ["entity"]


def test_compile_empty_executor_defaults_are_noop() -> None:
    config = sample_config(restart_policy="Never")
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Never",
        executor_defaults=ExecutorK8sDefaults(),
    )
    assert compiled.pod_annotations == {}
    assert "node_selector" not in compiled.pod_spec_kwargs
    assert "tolerations" not in compiled.pod_spec_kwargs
    assert "affinity" not in compiled.pod_spec_kwargs
    assert "topology_spread_constraints" not in compiled.pod_spec_kwargs


def test_compile_pod_annotations_default_empty_without_k8s_config() -> None:
    config = sample_config(restart_policy="Never")
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Never",
    )
    assert compiled.pod_annotations == {}


def test_build_job_body_stamps_pod_annotations_on_template() -> None:
    config = sample_config(restart_policy="Never")
    k8s_config = K8sDeploymentConfig.model_validate({"podAnnotations": {"sidecar.istio.io/nativeSidecar": "true"}})
    built = build_job_body(
        job_name="default-task",
        labels={"managed-by": "nemo-deployments"},
        config=config,
        workspace="default",
        deployment_name="task",
        k8s_config=k8s_config,
    )
    template_meta = _serialized(built.job)["spec"]["template"]["metadata"]
    assert template_meta["annotations"] == {"sidecar.istio.io/nativeSidecar": "true"}


def test_build_job_body_omits_annotations_when_none() -> None:
    config = sample_config(restart_policy="Never")
    built = build_job_body(
        job_name="default-task",
        labels={"managed-by": "nemo-deployments"},
        config=config,
        workspace="default",
        deployment_name="task",
        k8s_config=None,
    )
    template_meta = _serialized(built.job)["spec"]["template"]["metadata"]
    assert "annotations" not in template_meta


def test_build_deployment_body_stamps_pod_annotations_on_template() -> None:
    config = sample_always_config()
    k8s_config = K8sDeploymentConfig.model_validate({"podAnnotations": {"sidecar.istio.io/nativeSidecar": "true"}})
    built = build_deployment_body(
        resource_name="default-task",
        labels={"managed-by": "nemo-deployments"},
        config=config,
        workspace="default",
        deployment_name="task",
        k8s_config=k8s_config,
    )
    template_meta = _serialized(built.deployment)["spec"]["template"]["metadata"]
    assert template_meta["annotations"] == {"sidecar.istio.io/nativeSidecar": "true"}


def test_build_deployment_body_omits_annotations_when_none() -> None:
    config = sample_always_config()
    built = build_deployment_body(
        resource_name="default-task",
        labels={"managed-by": "nemo-deployments"},
        config=config,
        workspace="default",
        deployment_name="task",
        k8s_config=None,
    )
    template_meta = _serialized(built.deployment)["spec"]["template"]["metadata"]
    assert "annotations" not in template_meta


def test_compile_workload_emits_image_pull_secrets() -> None:
    config = sample_config(restart_policy="Never")
    platform_secret = ImagePullSecret(name="platform-secret")
    executor_secret = ImagePullSecret(name="executor-secret")
    with patch(
        "nemo_deployments_plugin.backends.k8s.compiler.get_platform_config",
        return_value=SimpleNamespace(image_pull_secrets=[platform_secret]),
    ):
        compiled = compile_workload(
            config=config,
            workspace="default",
            deployment_name="task",
            labels={"managed-by": "nemo-deployments"},
            k8s_config=None,
            pod_restart_policy="Never",
            executor_image_pull_secrets=[executor_secret],
        )
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    secret_names = {secret["name"] for secret in pod_spec["image_pull_secrets"]}
    assert secret_names == {"platform-secret", "executor-secret"}


def test_compile_config_files_emit_configmap_and_mounts() -> None:
    config = sample_always_config().model_copy(
        update={"config_files": [ConfigFile(path="/etc/app/config.yaml", content="key: value")]}
    )
    labels = {"managed-by": "nemo-deployments"}
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels=labels,
        k8s_config=None,
        pod_restart_policy="Always",
    )
    assert compiled.configmap_body is not None
    configmap = _serialized(compiled.configmap_body)
    assert configmap["data"]["etc__app__config.yaml"] == "key: value"
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert any(volume["name"] == "config-files" for volume in pod_spec["volumes"])
    main_container = compiled.pod_spec_kwargs["containers"][0]
    mount_paths = [mount.mount_path for mount in main_container.volume_mounts or []]
    assert "/etc/app/config.yaml" in mount_paths


def test_build_job_body_returns_compiled_workload() -> None:
    config = sample_config(restart_policy="OnFailure")
    built = build_job_body(
        job_name="dep-default-task-abc",
        labels={"managed-by": "nemo-deployments"},
        config=config,
        workspace="default",
        deployment_name="task",
        k8s_config=None,
    )
    assert built.job.kind == "Job"
    assert built.compiled.pod_spec_kwargs["restart_policy"] == "OnFailure"


def test_build_deployment_body_returns_compiled_workload() -> None:
    config = sample_always_config()
    built = build_deployment_body(
        resource_name="dep-default-task-abc",
        labels={"managed-by": "nemo-deployments"},
        config=config,
        workspace="default",
        deployment_name="task",
        k8s_config=None,
    )
    assert built.deployment.kind == "Deployment"
    assert built.compiled.service_containers[0].name == "main"


def test_validate_rejects_main_container_restart_policy() -> None:
    config = sample_always_config().model_copy(
        update={"containers": [Container.model_validate({"name": "main", "image": "nginx", "restartPolicy": "Always"})]}
    )
    with pytest.raises(DeploymentConfigError, match="only init_containers"):
        validate_config_for_deployment(config)


def test_validate_job_rejects_always() -> None:
    with pytest.raises(DeploymentConfigError, match="Deployment"):
        validate_config_for_job(sample_always_config())


def test_validate_rejects_duplicate_port_names() -> None:
    config = sample_always_config().model_copy(
        update={
            "containers": [
                Container(
                    name="main",
                    image="nginx",
                    ports=[ContainerPort(name="http", containerPort=8080)],
                ),
                Container(
                    name="side",
                    image="nginx",
                    ports=[ContainerPort(name="http", containerPort=9090)],
                ),
            ],
        }
    )
    with pytest.raises(DeploymentConfigError, match="duplicate container port name"):
        validate_config_for_deployment(config)


def test_validate_rejects_duplicate_listen_ports() -> None:
    config = sample_always_config().model_copy(
        update={
            "containers": [
                Container(
                    name="main",
                    image="nginx",
                    ports=[ContainerPort(name="http", containerPort=8080)],
                ),
                Container(
                    name="side",
                    image="nginx",
                    ports=[ContainerPort(name="alt", containerPort=8080)],
                ),
            ],
        }
    )
    with pytest.raises(DeploymentConfigError, match="duplicate container port 8080"):
        validate_config_for_deployment(config)


def test_build_configmap_body_none_when_empty() -> None:
    assert build_configmap_body(workspace="default", deployment_name="task", labels={}, config_files=[]) is None


def test_build_secret_body_none_when_empty() -> None:
    assert build_secret_body(workspace="default", deployment_name="task", labels={}, secret_env={}) is None


def test_build_secret_body_holds_values_and_labels() -> None:
    labels = {"managed-by": "nemo-deployments", "nemo.nvidia.com/deployment-name": "task"}
    secret = build_secret_body(
        workspace="default",
        deployment_name="task",
        labels=labels,
        secret_env={"APP_TOKEN": "value-a", "OTHER": "value-b"},
    )
    serialized = _serialized(secret)
    assert serialized["kind"] == "Secret"
    assert serialized["type"] == "Opaque"
    assert serialized["metadata"]["name"] == k8s_deployment_secret_name("default", "task")
    assert serialized["metadata"]["labels"] == labels
    assert serialized["stringData"] == {"APP_TOKEN": "value-a", "OTHER": "value-b"}


def test_build_env_vars_skips_secret_ref_entries() -> None:
    container = Container(
        name="main",
        image="alpine",
        env=[
            EnvVar(name="PLAIN", value="v"),
            EnvVar(name="APP_TOKEN", secretRef=SecretRef(workspace="default", name="app-token")),
        ],
    )
    env = build_env_vars(container)
    names = {item.name for item in env}
    assert names == {"PLAIN"}


def test_build_secret_env_from_empty_without_secret() -> None:
    assert build_secret_env_from(None) == []


def test_build_secret_env_from_projects_secret_ref() -> None:
    env_from = build_secret_env_from("dep-sec-abc")
    serialized = [_serialized(item) for item in env_from]
    assert serialized == [{"secretRef": {"name": "dep-sec-abc"}}]


def test_compile_workload_mounts_secret_via_env_from() -> None:
    config = sample_always_config().model_copy(
        update={
            "containers": [
                Container(
                    name="main",
                    image="nginx:alpine",
                    ports=[ContainerPort(name="http", containerPort=8080)],
                    env=[EnvVar(name="APP_TOKEN", secretRef=SecretRef(workspace="default", name="app-token"))],
                )
            ]
        }
    )
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Always",
        secret_env={"APP_TOKEN": "app-token-value"},
    )
    assert compiled.secret_body is not None
    assert compiled.secret_name == k8s_deployment_secret_name("default", "task")
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    main = pod_spec["containers"][0]
    assert main["envFrom"] == [{"secretRef": {"name": k8s_deployment_secret_name("default", "task")}}]
    # The secret value never appears as a plaintext env var in the pod spec.
    assert "env" not in main or all(entry.get("value") != "app-token-value" for entry in main["env"])


def test_compile_workload_no_secret_when_secret_env_empty() -> None:
    config = sample_always_config()
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Always",
        secret_env={},
    )
    assert compiled.secret_body is None
    assert compiled.secret_name is None
    pod_spec = _serialized(compiled.pod_spec_kwargs)
    assert "envFrom" not in pod_spec["containers"][0]


def test_compile_workload_projects_workload_identity_token() -> None:
    config = sample_always_config().model_copy(
        update={
            "workload_identity": WorkloadIdentitySpec(
                enabled=True,
                workloadKind="agent_deployment",
                workloadId="task",
                tokenAudience="nemo-helix",
                tokenExpirationSeconds=900,
            )
        }
    )
    compiled = compile_workload(
        config=config,
        workspace="default",
        deployment_name="task",
        labels={"managed-by": "nemo-deployments"},
        k8s_config=None,
        pod_restart_policy="Always",
    )

    pod_spec = _serialized(compiled.pod_spec_kwargs)
    volume = next(item for item in pod_spec["volumes"] if item["name"] == WORKLOAD_IDENTITY_VOLUME_NAME)
    projection = volume["projected"]["sources"][0]["serviceAccountToken"]
    assert projection["path"] == "token"
    assert projection["audience"] == "nemo-helix"
    assert projection["expirationSeconds"] == 900

    main = pod_spec["containers"][0]
    env = {item["name"]: item["value"] for item in main["env"]}
    assert env[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR] == WORKLOAD_IDENTITY_TOKEN_FILE_PATH
    mount = next(item for item in main["volumeMounts"] if item["name"] == WORKLOAD_IDENTITY_VOLUME_NAME)
    assert mount["mountPath"] == WORKLOAD_IDENTITY_VOLUME_PATH
    assert mount["readOnly"] is True


def test_compile_workload_rejects_user_workload_identity_volume_mount_name() -> None:
    config = with_workload_identity(
        sample_always_config().model_copy(
            update={
                "volume_mounts": [
                    VolumeMount(
                        name=WORKLOAD_IDENTITY_VOLUME_NAME,
                        mountPath="/data",
                    )
                ]
            }
        )
    )

    with pytest.raises(DeploymentConfigError, match=WORKLOAD_IDENTITY_VOLUME_NAME):
        validate_config_for_deployment(config)


def test_compile_workload_rejects_user_workload_identity_volume_mount_path() -> None:
    config = with_workload_identity(
        sample_always_config().model_copy(
            update={
                "volume_mounts": [
                    VolumeMount(
                        name="data",
                        mountPath=WORKLOAD_IDENTITY_VOLUME_PATH,
                    )
                ]
            }
        )
    )

    with pytest.raises(DeploymentConfigError, match=WORKLOAD_IDENTITY_VOLUME_PATH):
        validate_config_for_deployment(config)


def test_compile_workload_allows_user_workload_identity_volume_mount_name_when_disabled() -> None:
    config = sample_always_config().model_copy(
        update={
            "volume_mounts": [
                VolumeMount(
                    name=WORKLOAD_IDENTITY_VOLUME_NAME,
                    mountPath="/data",
                )
            ]
        }
    )

    validate_config_for_deployment(config)


def test_compile_workload_rejects_mismatched_workload_identity_token_audience() -> None:
    config = sample_always_config().model_copy(
        update={
            "workload_identity": WorkloadIdentitySpec(
                enabled=True,
                tokenAudience="other-audience",
            )
        }
    )

    with (
        patch(
            "nemo_deployments_plugin.backends.k8s.compiler.get_workload_identity_token_audience",
            return_value="platform-audience",
        ),
        pytest.raises(DeploymentConfigError, match="tokenAudience"),
    ):
        compile_workload(
            config=config,
            workspace="default",
            deployment_name="task",
            labels={"managed-by": "nemo-deployments"},
            k8s_config=None,
            pod_restart_policy="Always",
        )


def test_compile_workload_does_not_mount_workload_identity_on_auth_proxy() -> None:
    config = sample_always_config().model_copy(
        update={
            "auth_proxy_sidecar": True,
            "auth_proxy_sidecar_identity": "agents",
            "auth_proxy_sidecar_on_behalf_of": "user:alice",
            "workload_identity": WorkloadIdentitySpec(enabled=True, workloadKind="agent_deployment", workloadId="task"),
        }
    )
    with patch("nemo_deployments_plugin.auth_proxy.platform_auth_enabled", return_value=True):
        compiled = compile_workload(
            config=config,
            workspace="default",
            deployment_name="task",
            labels={"managed-by": "nemo-deployments"},
            k8s_config=None,
            pod_restart_policy="Always",
        )

    pod_spec = _serialized(compiled.pod_spec_kwargs)
    auth_proxy = next(item for item in pod_spec["init_containers"] if item["name"] == "auth-proxy")
    auth_proxy_env = {item["name"]: item["value"] for item in auth_proxy.get("env", [])}
    assert auth_proxy_env["XDG_STATE_HOME"] == "/tmp"
    assert all(env["name"] != WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR for env in auth_proxy.get("env", []))
    assert all(mount["name"] != WORKLOAD_IDENTITY_VOLUME_NAME for mount in auth_proxy.get("volumeMounts", []))
