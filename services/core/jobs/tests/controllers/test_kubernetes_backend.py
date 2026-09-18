# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from kubernetes import client
from kubernetes.client.rest import ApiException
from nmp.common.auth import (
    KUBERNETES_POD_UID_REFERENCE_NAME,
    NMP_PRINCIPAL_ENVVAR,
    WorkloadDelegationConflictError,
    reference_delegation_name,
)
from nmp.common.config import ImagePullSecret, PlatformConfig
from nmp.common.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_FILESET_ENVVAR,
    NEMO_JOB_WORKSPACE_ENVVAR,
    PERSISTENT_JOB_STORAGE_PATH_ENVVAR,
)
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.api.v2.jobs.schemas import PlatformJobStepWithContext
from nmp.core.jobs.app.constants import (
    JOB_EXECUTION_BACKEND_LABEL,
    JOB_EXECUTION_PROFILE_LABEL,
    JOB_ID_LABEL,
    JOB_MANAGED_BY_JOBS_CONTROLLER,
    JOB_MANAGED_BY_LABEL,
    JOB_STEP_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_USES_PERSISTENT_STORAGE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
    KUBE_JOB_SELECTOR_LABELS,
)
from nmp.core.jobs.app.providers import ContainerSpec, CPUExecutionProvider, GPUExecutionProvider
from nmp.core.jobs.app.schemas import (
    PlatformJobEnvironmentVariable,
    PlatformJobSecretEnvironmentVariableRef,
    PlatformJobStepSpec,
)
from nmp.core.jobs.controllers.backends.base import (
    NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    WORKLOAD_IDENTITY_VOLUME_NAME,
    WORKLOAD_IDENTITY_VOLUME_PATH,
)
from nmp.core.jobs.controllers.backends.kubernetes import (
    CPUKubernetesJobBackend,
    GPUKubernetesJobBackend,
    KubernetesJobExecutionProfileConfig,
)
from nmp.core.jobs.controllers.backends.kubernetes.common import (
    JOB_DSHM_VOLUME_NAME,
    JOB_STORAGE_VOLUME_NAME,
    KubernetesConfigMapVolume,
    KubernetesEmptyDirVolume,
    KubernetesJobStorageConfig,
    KubernetesKeyToPath,
    KubernetesObjectMetadata,
    KubernetesPersistentVolumeClaim,
    KubernetesSecretVolume,
    KubernetesVolume,
    KubernetesVolumeMount,
    PodStatus,
    build_affinity,
    build_image_pull_secrets,
    build_metadata,
    build_pod_security_context,
    build_tolerations,
    cleanup_job_persistent_storage,
    common_labels_for_step,
    create_configmap,
    delete_configmap,
    name_for_step,
)
from nmp.core.jobs.controllers.backends.workload_tokens import WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS
from pydantic import ValidationError

DEFAULT_STORAGE = KubernetesJobStorageConfig(pvc_name="job-storage-pvc")
DEFAULT_JOB_METADATA = KubernetesObjectMetadata(
    labels={"owner": "alpha"}, annotations={"example.com/annotation": "bravo"}
)
DEFAULT_POD_METADATA = KubernetesObjectMetadata(labels={"foo": "bar"}, annotations={"example.com/annotation": "value"})


def _kubernetes_job_for_step(
    step: PlatformJobStepWithContext,
    *,
    status: client.V1JobStatus | None = None,
    service_account_name: str = "default",
) -> client.V1Job:
    labels = common_labels_for_step(step)
    labels[JOB_EXECUTION_BACKEND_LABEL] = "kubernetes_job"
    labels[JOB_EXECUTION_PROFILE_LABEL] = "default"
    return client.V1Job(
        metadata=client.V1ObjectMeta(
            name=name_for_step(step),
            namespace="test-namespace",
            labels=labels,
        ),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    containers=[],
                    service_account_name=service_account_name,
                )
            )
        ),
        status=status
        or client.V1JobStatus(
            active=1,
            completion_time=None,
            failed=None,
            succeeded=None,
        ),
    )


def _kubernetes_pod(uid: str, *, phase: str = "Running") -> client.V1Pod:
    return client.V1Pod(
        metadata=client.V1ObjectMeta(
            name=f"pod-{uid}",
            namespace="test-namespace",
            uid=uid,
        ),
        status=client.V1PodStatus(
            phase=phase,
            container_statuses=[
                client.V1ContainerStatus(
                    image="test-image",
                    image_id="test-image-id",
                    name="nemo-job-task",
                    ready=phase == "Running",
                    restart_count=0,
                    state=client.V1ContainerState(
                        running=client.V1ContainerStateRunning(started_at=datetime.datetime.now(datetime.timezone.utc))
                        if phase == "Running"
                        else None,
                        terminated=client.V1ContainerStateTerminated(
                            exit_code=0,
                            reason="Completed",
                        )
                        if phase == "Succeeded"
                        else None,
                    ),
                )
            ],
        ),
    )


@pytest.fixture
def workload_exchange_auth_config():
    return SimpleNamespace(
        oidc=SimpleNamespace(
            workload_token_exchange_enabled=True,
            workload_audience="nemo-platform",
            audience=None,
        )
    )


@pytest.fixture
def kubernetes_client_mock():
    """Mock Kubernetes client for testing."""

    batch_v1_mock = MagicMock()
    core_v1_mock = MagicMock()
    with (
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.client.BatchV1Api",
            return_value=batch_v1_mock,
        ),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.client.CoreV1Api",
            return_value=core_v1_mock,
        ),
    ):
        yield {"batch_v1": batch_v1_mock, "core_v1": core_v1_mock}


@pytest.fixture
def kubernetes_execution_profile_config():
    """Create a test Kubernetes execution profile."""
    return KubernetesJobExecutionProfileConfig(
        namespace="test-namespace",
        ttl_seconds_after_finished=300,
        tolerations=[
            {"key": "nvidia.com/gpu", "operator": "Equal", "value": "true", "effect": "NoSchedule"},
            {"key": "node-type", "operator": "Equal", "value": "gpu", "effect": "NoSchedule"},
        ],
        node_selector={"accelerator": "nvidia-tesla-v100", "node-type": "gpu"},
        affinity={
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {"key": "kubernetes.io/arch", "operator": "In", "values": ["amd64"]},
                                {
                                    "key": "node.kubernetes.io/instance-type",
                                    "operator": "In",
                                    "values": ["p3.2xlarge", "p3.8xlarge"],
                                },
                            ]
                        }
                    ]
                }
            }
        },
        job_metadata=DEFAULT_JOB_METADATA,
        pod_metadata=DEFAULT_POD_METADATA,
        storage=DEFAULT_STORAGE,
    )


@pytest.fixture
def cpu_execution_provider():
    """Create a test CPU execution provider."""
    return CPUExecutionProvider(
        container=ContainerSpec(
            image="nvidia/cuda:11.8-runtime-ubuntu20.04",
            command=["python", "-c", "print('Hello World')"],
        ),
    )


@pytest.fixture
def kubernetes_job(
    mock_nmp_client,
    kubernetes_client_mock,
    kubernetes_execution_profile_config,
    mock_platform_config,
):
    """Create a KubernetesJob instance with mocked clients."""
    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        # Convert the Pydantic model to dict format expected by the base class
        k8s_job = CPUKubernetesJobBackend(mock_nmp_client, kubernetes_execution_profile_config, profile_name="default")
        k8s_job._batch_v1 = kubernetes_client_mock["batch_v1"]
        k8s_job._core_v1 = kubernetes_client_mock["core_v1"]
        yield k8s_job


def test_kubernetes_volume_to_k8s_pvc():
    """Test conversion of KubernetesVolume with PVC to V1Volume."""

    pvc_config = KubernetesPersistentVolumeClaim(claim_name="test-pvc")
    volume_config = KubernetesVolume(name="test-volume", persistent_volume_claim=pvc_config)

    k8s_volume = volume_config.to_k8s()

    assert k8s_volume is not None
    assert k8s_volume.name == "test-volume"
    assert k8s_volume.persistent_volume_claim is not None
    assert k8s_volume.persistent_volume_claim.claim_name == "test-pvc"


def test_kubernetes_volume_to_k8s_empty_dir():
    """Test conversion of KubernetesVolume with EmptyDir to V1Volume."""

    empty_dir_config = KubernetesEmptyDirVolume(medium="Memory", size_limit="1Gi")
    volume_config = KubernetesVolume(name="test-volume", empty_dir=empty_dir_config)

    k8s_volume = volume_config.to_k8s()

    assert k8s_volume is not None
    assert k8s_volume.name == "test-volume"
    assert k8s_volume.empty_dir is not None
    assert k8s_volume.empty_dir.medium == "Memory"
    assert k8s_volume.empty_dir.size_limit == "1Gi"


def test_kubernetes_volume_to_k8s_secret():
    """Test conversion of KubernetesVolume with Secret to V1Volume."""

    secret_config = KubernetesSecretVolume(
        secret_name="test-secret",
        items=[KubernetesKeyToPath(key="ca.crt", path="ca.crt")],
    )
    volume_config = KubernetesVolume(name="test-volume", secret=secret_config)

    k8s_volume = volume_config.to_k8s()

    assert k8s_volume is not None
    assert k8s_volume.name == "test-volume"
    assert k8s_volume.secret is not None
    assert k8s_volume.secret.secret_name == "test-secret"
    assert k8s_volume.secret.items[0].key == "ca.crt"
    assert k8s_volume.secret.items[0].path == "ca.crt"


def test_kubernetes_volume_to_k8s_config_map():
    """Test conversion of KubernetesVolume with ConfigMap to V1Volume."""

    config_map_config = KubernetesConfigMapVolume(
        name="test-config-map",
        items=[KubernetesKeyToPath(key="trust-bundle.pem", path="ca.crt")],
    )
    volume_config = KubernetesVolume(name="test-volume", config_map=config_map_config)

    k8s_volume = volume_config.to_k8s()

    assert k8s_volume is not None
    assert k8s_volume.name == "test-volume"
    assert k8s_volume.config_map is not None
    assert k8s_volume.config_map.name == "test-config-map"
    assert k8s_volume.config_map.items[0].key == "trust-bundle.pem"
    assert k8s_volume.config_map.items[0].path == "ca.crt"


def test_kubernetes_volume_invalid_configuration():
    """Test that invalid KubernetesVolume configuration raises ValueError."""

    # Neither PVC nor EmptyDir specified
    with pytest.raises(ValueError, match="Exactly one of"):
        KubernetesVolume(name="invalid-volume")

    # Both PVC and EmptyDir specified
    pvc_config = KubernetesPersistentVolumeClaim(claim_name="test-pvc")
    empty_dir_config = KubernetesEmptyDirVolume()
    with pytest.raises(ValueError, match="Exactly one of"):
        KubernetesVolume(name="invalid-volume", persistent_volume_claim=pvc_config, empty_dir=empty_dir_config)


def test_kubernetes_volume_json_schema_requires_exactly_one_source():
    """Test OpenAPI schema requires exactly one non-null KubernetesVolume source."""

    schema = KubernetesVolume.model_json_schema()
    validator = Draft202012Validator(schema)

    assert "name" in schema["required"]
    assert schema["oneOf"] == [
        {
            "required": ["persistent_volume_claim"],
            "properties": {"persistent_volume_claim": {"not": {"type": "null"}}},
        },
        {"required": ["empty_dir"], "properties": {"empty_dir": {"not": {"type": "null"}}}},
        {"required": ["secret"], "properties": {"secret": {"not": {"type": "null"}}}},
        {"required": ["config_map"], "properties": {"config_map": {"not": {"type": "null"}}}},
    ]

    validator.validate({"name": "pvc-volume", "persistent_volume_claim": {"claim_name": "test-pvc"}})
    validator.validate({"name": "empty-dir-volume", "empty_dir": {}})
    validator.validate({"name": "secret-volume", "secret": {"secret_name": "test-secret"}})
    validator.validate({"name": "config-map-volume", "config_map": {"name": "test-config-map"}})

    with pytest.raises(JsonSchemaValidationError):
        validator.validate({"name": "missing-source"})

    with pytest.raises(JsonSchemaValidationError):
        validator.validate(
            {
                "name": "multiple-sources",
                "persistent_volume_claim": {"claim_name": "test-pvc"},
                "empty_dir": {},
            }
        )

    with pytest.raises(JsonSchemaValidationError):
        validator.validate({"name": "null-source", "persistent_volume_claim": None})


def test_build_metadata():
    """Test building Kubernetes metadata from configuration."""

    labels = {"app": "test-app", "env": "production"}
    metadata_config = KubernetesObjectMetadata(labels={"owner": "team-a"}, annotations={"annotation1": "value1"})

    # Test the happy path
    metadata = build_metadata(labels, metadata_config)

    assert metadata is not None
    assert metadata.labels is not None
    assert metadata.annotations is not None

    # Check labels
    assert metadata.labels["app"] == "test-app"
    assert metadata.labels["env"] == "production"
    assert metadata.labels["owner"] == "team-a"

    # Check annotations
    assert metadata.annotations["annotation1"] == "value1"

    # Test no labels provided
    metadata_no_labels = build_metadata(None, metadata_config)
    assert metadata_no_labels is not None
    assert metadata_no_labels.labels == {"owner": "team-a"}
    assert metadata_no_labels.annotations == {"annotation1": "value1"}

    # Test no metadata provided
    metadata_no_config = build_metadata(labels, None)
    assert metadata_no_config is not None
    assert metadata_no_config.labels == labels
    assert metadata_no_config.annotations is None

    # Test to ensure that the original labels dict is not modified
    original_labels = {"original": "label"}
    metadata = build_metadata(original_labels, metadata_config)
    assert "original" in original_labels
    assert "owner" not in original_labels  # Ensure original dict is unchanged


def test_build_image_pull_secrets(mock_platform_config):
    """Test building Kubernetes image pull secrets from configuration."""
    with patch(
        "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
        return_value=mock_platform_config,
    ):
        # Test with empty list - should return global secrets only
        secrets = build_image_pull_secrets([])
        assert len(secrets) == 1
        assert secrets[0].name == "global-pull-secret"

        # Test with additional secrets
        additional_secrets = [
            ImagePullSecret(name="custom-secret-1"),
            ImagePullSecret(name="custom-secret-2"),
        ]
        secrets = build_image_pull_secrets(additional_secrets)
        assert len(secrets) == 3
        secret_names = {s.name for s in secrets}
        assert secret_names == {"global-pull-secret", "custom-secret-1", "custom-secret-2"}

        # Test deduplication - when local secret has same name as global
        duplicate_secrets = [ImagePullSecret(name="global-pull-secret")]
        secrets = build_image_pull_secrets(duplicate_secrets)
        assert len(secrets) == 1
        assert secrets[0].name == "global-pull-secret"


def test_create_configmap_includes_managed_by_labels(test_step_pending):
    """ConfigMaps created for job steps must include the correct labels so only jobs-controller ConfigMaps are cleaned up.

    Verifies that create_configmap uses labels from KUBE_JOB_SELECTOR_LABELS (app, JOB_MANAGED_BY_LABEL)
    so list/delete by label_selector only targets our ConfigMaps.
    """
    mock_core_v1 = MagicMock()
    create_configmap(mock_core_v1, "test-namespace", test_step_pending)
    mock_core_v1.create_namespaced_config_map.assert_called_once()
    call_kwargs = mock_core_v1.create_namespaced_config_map.call_args.kwargs
    body = call_kwargs["body"]
    labels = body.metadata.labels

    # Must include both selector labels so cleanup can find and safely delete only our ConfigMaps
    for key, value in KUBE_JOB_SELECTOR_LABELS.items():
        assert labels.get(key) == value, f"ConfigMap missing label {key}={value}"
    assert labels[JOB_MANAGED_BY_LABEL] == JOB_MANAGED_BY_JOBS_CONTROLLER
    assert labels.get("app") == "nemo-job"

    # Step/job labels from common_labels_for_step should also be present
    assert labels.get(JOB_ID_LABEL) == test_step_pending.job
    assert labels.get(JOB_STEP_NAME_LABEL) == test_step_pending.name
    assert labels.get(JOB_WORKSPACE_ID_LABEL) == test_step_pending.workspace
    assert call_kwargs["namespace"] == "test-namespace"


def test_common_labels_for_step_includes_required_cleanup_labels(test_step_pending):
    """common_labels_for_step returns labels required by cleanup_steps to identify and process jobs."""
    labels = common_labels_for_step(test_step_pending)
    assert labels[JOB_ID_LABEL] == test_step_pending.job
    assert labels[JOB_STEP_NAME_LABEL] == test_step_pending.name
    assert labels[JOB_WORKSPACE_ID_LABEL] == test_step_pending.workspace


def test_delete_configmap_only_deletes_when_managed_by_jobs_controller():
    """delete_configmap must only delete ConfigMaps that have JOB_MANAGED_BY_LABEL=JOB_MANAGED_BY_JOBS_CONTROLLER."""
    mock_core_v1 = MagicMock()
    namespace, name = "test-namespace", "test-step-id"

    # Case 1: ConfigMap exists and has the label -> delete is called
    mock_core_v1.read_namespaced_config_map.return_value = MagicMock(
        metadata=MagicMock(labels={JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER})
    )
    delete_configmap(mock_core_v1, namespace, name)
    mock_core_v1.read_namespaced_config_map.assert_called_once_with(name=name, namespace=namespace)
    mock_core_v1.delete_namespaced_config_map.assert_called_once_with(name=name, namespace=namespace)

    mock_core_v1.reset_mock()

    # Case 2: ConfigMap exists but does not have the label -> delete is NOT called
    mock_core_v1.read_namespaced_config_map.return_value = MagicMock(metadata=MagicMock(labels={"other": "owner"}))
    delete_configmap(mock_core_v1, namespace, name)
    mock_core_v1.read_namespaced_config_map.assert_called_once_with(name=name, namespace=namespace)
    mock_core_v1.delete_namespaced_config_map.assert_not_called()

    mock_core_v1.reset_mock()

    # Case 3: ConfigMap not found (404) -> delete is NOT called
    from kubernetes.client.rest import ApiException

    mock_core_v1.read_namespaced_config_map.side_effect = ApiException(status=404)
    delete_configmap(mock_core_v1, namespace, name)
    mock_core_v1.read_namespaced_config_map.assert_called_once_with(name=name, namespace=namespace)
    mock_core_v1.delete_namespaced_config_map.assert_not_called()


def test_build_tolerations_with_tolerations(kubernetes_execution_profile_config):
    """Test building tolerations from configuration."""
    tolerations = build_tolerations(kubernetes_execution_profile_config.tolerations)

    assert tolerations is not None
    assert len(tolerations) == 2

    # Check first toleration
    assert tolerations[0].key == "nvidia.com/gpu"
    assert tolerations[0].operator == "Equal"
    assert tolerations[0].value == "true"
    assert tolerations[0].effect == "NoSchedule"

    # Check second toleration
    assert tolerations[1].key == "node-type"
    assert tolerations[1].operator == "Equal"
    assert tolerations[1].value == "gpu"
    assert tolerations[1].effect == "NoSchedule"


def test_build_tolerations_without_tolerations():
    """Test building tolerations when none are configured."""
    profile = KubernetesJobExecutionProfileConfig(namespace="test", storage=DEFAULT_STORAGE)
    tolerations = build_tolerations(profile.tolerations)

    assert tolerations == []


def test_build_affinity_with_node_affinity(kubernetes_execution_profile_config):
    """Test building affinity from configuration."""
    affinity = build_affinity(kubernetes_execution_profile_config.affinity)

    assert affinity is not None
    assert affinity.node_affinity is not None
    assert affinity.node_affinity.required_during_scheduling_ignored_during_execution is not None

    # Check node selector terms
    node_selector = affinity.node_affinity.required_during_scheduling_ignored_during_execution
    assert len(node_selector.node_selector_terms) == 1

    term = node_selector.node_selector_terms[0]
    assert len(term.match_expressions) == 2

    # Check first expression
    expr1 = term.match_expressions[0]
    assert expr1.key == "kubernetes.io/arch"
    assert expr1.operator == "In"
    assert expr1.values == ["amd64"]

    # Check second expression
    expr2 = term.match_expressions[1]
    assert expr2.key == "node.kubernetes.io/instance-type"
    assert expr2.operator == "In"
    assert expr2.values == ["p3.2xlarge", "p3.8xlarge"]


def test_build_security_context():
    """Test building pod security context from configuration."""
    profile = KubernetesJobExecutionProfileConfig(namespace="test", storage=DEFAULT_STORAGE)
    security_context = build_pod_security_context(profile.pod_security_context)

    assert security_context is None


def test_build_security_context_with_values():
    """Test building pod security context with specific values."""
    profile = KubernetesJobExecutionProfileConfig(
        namespace="test",
        pod_security_context={
            "runAsNonRoot": True,
            "runAsUser": 2000,
            "fsGroup": 3000,
        },
        storage=DEFAULT_STORAGE,
    )
    security_context = build_pod_security_context(profile.pod_security_context)

    assert security_context is not None
    assert security_context.run_as_non_root is True
    assert security_context.run_as_user == 2000
    assert security_context.fs_group == 3000


def test_cleanup_job_persistent_storage_applies_pod_security_context():
    """Cleanup job pods use the same pod security context as workload jobs (e.g. NFS ownership)."""
    batch_v1 = MagicMock()
    cleanup_job_persistent_storage(
        namespace="ns",
        batch_v1=batch_v1,
        pvc_name="pvc",
        workspace="ws",
        job_id="jid",
        step_name="step",
        permissions_image="perm:img",
        execution_backend="kubernetes_job",
        execution_profile="default",
        job_metadata=KubernetesObjectMetadata(),
        pod_metadata=KubernetesObjectMetadata(),
        pod_security_context={"runAsUser": 2000, "fsGroup": 3000, "runAsNonRoot": True},
    )
    batch_v1.create_namespaced_job.assert_called_once()
    body = batch_v1.create_namespaced_job.call_args.kwargs["body"]
    sc = body.spec.template.spec.security_context
    assert sc is not None
    assert sc.run_as_user == 2000
    assert sc.fs_group == 3000
    assert sc.run_as_non_root is True


def test_cleanup_job_persistent_storage_omits_security_context_when_unconfigured():
    batch_v1 = MagicMock()
    cleanup_job_persistent_storage(
        namespace="ns",
        batch_v1=batch_v1,
        pvc_name="pvc",
        workspace="ws",
        job_id="jid",
        step_name="step",
        permissions_image="perm:img",
        execution_backend="kubernetes_job",
        execution_profile="default",
        job_metadata=KubernetesObjectMetadata(),
        pod_metadata=KubernetesObjectMetadata(),
    )
    body = batch_v1.create_namespaced_job.call_args.kwargs["body"]
    assert body.spec.template.spec.security_context is None


def test_build_affinity_without_affinity():
    """Test building affinity when none is configured."""
    profile = KubernetesJobExecutionProfileConfig(namespace="test", storage=DEFAULT_STORAGE)
    affinity = build_affinity(profile.affinity)

    assert affinity == client.V1Affinity()


def test_build_affinity_with_pod_anti_affinity():
    """Test building affinity with pod anti-affinity configuration."""
    profile = KubernetesJobExecutionProfileConfig(
        namespace="test",
        affinity={
            "podAntiAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "podAffinityTerm": {
                            "labelSelector": {
                                "matchExpressions": [{"key": "app", "operator": "In", "values": ["my-app"]}]
                            },
                            "topologyKey": "kubernetes.io/hostname",
                        },
                    }
                ]
            }
        },
        storage=DEFAULT_STORAGE,
    )

    affinity = build_affinity(profile.affinity)

    assert affinity is not None
    assert affinity.pod_anti_affinity is not None
    assert affinity.pod_anti_affinity.preferred_during_scheduling_ignored_during_execution is not None

    preferred_terms = affinity.pod_anti_affinity.preferred_during_scheduling_ignored_during_execution
    assert len(preferred_terms) == 1

    term = preferred_terms[0]
    assert term.weight == 100
    assert term.pod_affinity_term.topology_key == "kubernetes.io/hostname"
    assert term.pod_affinity_term.label_selector.match_expressions[0].key == "app"


def test_build_affinity_with_pod_affinity():
    """Test building affinity with pod affinity configuration."""
    profile = KubernetesJobExecutionProfileConfig(
        namespace="test",
        affinity={
            "podAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {"matchLabels": {"app": "database"}},
                        "topologyKey": "topology.kubernetes.io/zone",
                    }
                ]
            }
        },
        storage=DEFAULT_STORAGE,
    )

    affinity = build_affinity(profile.affinity)

    assert affinity is not None
    assert affinity.pod_affinity is not None
    assert affinity.pod_affinity.required_during_scheduling_ignored_during_execution is not None

    required_terms = affinity.pod_affinity.required_during_scheduling_ignored_during_execution
    assert len(required_terms) == 1

    term = required_terms[0]
    assert term.topology_key == "topology.kubernetes.io/zone"
    assert term.label_selector.match_labels["app"] == "database"


def test_build_affinity_with_complex_configuration():
    """Test building affinity with complex multi-type configuration."""
    profile = KubernetesJobExecutionProfileConfig(
        namespace="test",
        affinity={
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {"matchExpressions": [{"key": "node-type", "operator": "In", "values": ["worker"]}]}
                    ]
                }
            },
            "podAntiAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 80,
                        "podAffinityTerm": {
                            "labelSelector": {
                                "matchExpressions": [{"key": "app", "operator": "In", "values": ["my-app"]}]
                            },
                            "topologyKey": "kubernetes.io/hostname",
                        },
                    }
                ]
            },
        },
        storage=DEFAULT_STORAGE,
    )

    affinity = build_affinity(profile.affinity)

    assert affinity is not None
    assert affinity.node_affinity is not None
    assert affinity.pod_anti_affinity is not None
    assert affinity.pod_affinity is None  # Not configured


def test_build_affinity_with_invalid_pod_affinity_structure():
    """Test that invalid pod affinity structure raises ValueError."""
    profile = KubernetesJobExecutionProfileConfig(
        namespace="test",
        affinity={"podAntiAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": "this should be a list"}},
        storage=DEFAULT_STORAGE,
    )

    with pytest.raises(ValueError, match="Invalid affinity configuration"):
        build_affinity(profile.affinity)


def test_build_affinity_with_missing_required_fields():
    """Test that missing required fields raise ValueError."""
    profile = KubernetesJobExecutionProfileConfig(
        namespace="test",
        affinity={
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "labelSelector": {"matchLabels": {"app": "test"}}
                        # Missing required topologyKey
                    }
                ]
            }
        },
        storage=DEFAULT_STORAGE,
    )

    with pytest.raises(ValueError, match="Invalid affinity configuration"):
        build_affinity(profile.affinity)


def test_schedule_job_success(kubernetes_job, cpu_execution_provider, test_step_pending):
    """Test successful job scheduling."""
    # Mock successful job creation
    kubernetes_job._batch_v1.create_namespaced_job.return_value = MagicMock()

    # Schedule the job
    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    # Verify job creation was called
    kubernetes_job._batch_v1.create_namespaced_job.assert_called_once()
    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args

    # Check namespace
    assert call_args.kwargs["namespace"] == "test-namespace"

    # Check job object
    job_body = call_args.kwargs["body"]
    assert job_body.api_version == "batch/v1"
    assert job_body.kind == "Job"

    # Check metadata
    assert "test-step-id" in job_body.metadata.name
    # Labels required by cleanup_steps to identify and process the job
    assert job_body.metadata.labels[JOB_ID_LABEL] == "test-job-id"
    assert job_body.metadata.labels[JOB_STEP_NAME_LABEL] == "test-step"
    assert job_body.metadata.labels[JOB_WORKSPACE_ID_LABEL] == "default"
    assert job_body.metadata.labels[JOB_STEP_ID_LABEL] == "test-step-id"
    assert job_body.metadata.labels["app"] == "nemo-job"
    assert job_body.metadata.labels["owner"] == "alpha"  # From execution profile
    assert job_body.metadata.labels[JOB_EXECUTION_BACKEND_LABEL] == "kubernetes_job"
    assert job_body.metadata.labels[JOB_EXECUTION_PROFILE_LABEL] == "default"
    assert job_body.metadata.annotations["example.com/annotation"] == "bravo"  # From execution profile

    # Check job spec
    assert job_body.spec.backoff_limit == 0
    assert job_body.spec.ttl_seconds_after_finished == 300

    # Check pod template
    pod_template = job_body.spec.template
    pod_metadata = job_body.spec.template.metadata
    pod_spec = pod_template.spec

    # Check pod metadata
    assert pod_metadata is not None
    assert pod_metadata.labels is not None
    assert pod_metadata.labels[JOB_ID_LABEL] == "test-job-id"
    assert pod_metadata.labels[JOB_STEP_NAME_LABEL] == "test-step"
    assert pod_metadata.labels[JOB_WORKSPACE_ID_LABEL] == "default"
    assert pod_metadata.labels["app"] == "nemo-job"
    assert pod_metadata.labels["foo"] == "bar"  # From execution profile
    assert pod_metadata.labels[JOB_EXECUTION_BACKEND_LABEL] == "kubernetes_job"
    assert pod_metadata.labels[JOB_EXECUTION_PROFILE_LABEL] == "default"
    assert pod_metadata.annotations is not None
    assert pod_metadata.annotations["example.com/annotation"] == "value"  # From execution

    # Check basic pod settings
    assert pod_spec.restart_policy == "Never"
    assert pod_spec.active_deadline_seconds == 86400
    assert pod_spec.service_account_name == "default"

    # Check tolerations
    assert pod_spec.tolerations is not None
    assert len(pod_spec.tolerations) == 2
    assert pod_spec.tolerations[0].key == "nvidia.com/gpu"

    # Check node selector
    assert pod_spec.node_selector is not None
    assert pod_spec.node_selector["accelerator"] == "nvidia-tesla-v100"
    assert pod_spec.node_selector["node-type"] == "gpu"

    # Check affinity
    assert pod_spec.affinity is not None
    assert pod_spec.affinity.node_affinity is not None

    # Check containers
    assert len(pod_spec.containers) == 1
    main_container = pod_spec.containers[0]
    assert main_container.name == "nemo-job-task"
    assert main_container.image == "nvidia/cuda:11.8-runtime-ubuntu20.04"

    # Check environment variables
    env_vars = {env.name: env.value for env in main_container.env}
    assert "ENV_VAR" in env_vars
    assert env_vars["ENV_VAR"] == "test_value"
    assert env_vars[NEMO_JOB_FILESET_ENVVAR] == "test-logs-fileset"
    assert env_vars[PERSISTENT_JOB_STORAGE_PATH_ENVVAR] == "/var/test"
    assert env_vars[EPHEMERAL_TASK_STORAGE_PATH_ENVVAR] == "/var/tmp"
    assert "NMP_BASE_URL" in env_vars

    # Ensure that config warnings are disabled
    assert env_vars["NMP_CONFIG_WARNINGS_DISABLED"] == "1"


def test_created_step_does_not_ttl_before_backend_acceptance(kubernetes_job, cpu_execution_provider, test_step_pending):
    """CREATED age should not fail a step before the Kubernetes backend accepts it."""
    kubernetes_job._batch_v1.create_namespaced_job.return_value = MagicMock()
    ttl_seconds = kubernetes_job._execution_profile_config.ttl_seconds_before_active
    old_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=ttl_seconds + 300)
    test_step_pending.created_at = old_timestamp
    test_step_pending.updated_at = old_timestamp
    test_step_pending.status = PlatformJobStatus.CREATED

    update = kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    assert update.status == PlatformJobStatus.PENDING
    kubernetes_job._batch_v1.create_namespaced_job.assert_called_once()


def test_kubernetes_job_profile_environment_applied(
    mock_nmp_client,
    kubernetes_client_mock,
    kubernetes_execution_profile_config,
    mock_platform_config,
    cpu_execution_provider,
    test_step_pending,
):
    """Profile environment (e.g. HOME=/tmp) is applied to scheduled job pod containers."""
    profile_config = KubernetesJobExecutionProfileConfig(
        **{**kubernetes_execution_profile_config.model_dump(), "env": {"HOME": "/tmp"}}
    )
    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        backend = CPUKubernetesJobBackend(mock_nmp_client, profile_config, profile_name="default")
        backend._batch_v1 = kubernetes_client_mock["batch_v1"]
        backend._core_v1 = kubernetes_client_mock["core_v1"]

    backend._batch_v1.create_namespaced_job.return_value = MagicMock()
    backend.schedule(cpu_execution_provider, test_step_pending)

    call_args = backend._batch_v1.create_namespaced_job.call_args
    job_body = call_args.kwargs["body"]
    main_container = job_body.spec.template.spec.containers[0]
    env_vars = {env.name: env.value for env in main_container.env}
    assert env_vars.get("HOME") == "/tmp"
    assert env_vars.get("ENV_VAR") == "test_value"


def test_kubernetes_job_uses_service_discovery_urls_for_job_runtime(
    mock_nmp_client,
    kubernetes_client_mock,
    kubernetes_execution_profile_config,
    cpu_execution_provider,
    test_step_pending,
):
    """Job pods use routable service_discovery URLs instead of local in-process service URLs."""
    platform_config = PlatformConfig(  # type: ignore[abstract]
        base_url="http://127.0.0.1:8080",
        services="jobs,files,models,secrets",
        service_discovery={
            "platform": "https://nemo-gateway:8080",
            "auth": "https://nemo-auth:8080",
        },
        loopback_address="nemo-gateway",
    )
    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=platform_config,
        ),
    ):
        backend = CPUKubernetesJobBackend(
            mock_nmp_client,
            kubernetes_execution_profile_config,
            profile_name="default",
        )
        backend._batch_v1 = kubernetes_client_mock["batch_v1"]
        backend._core_v1 = kubernetes_client_mock["core_v1"]

        backend._batch_v1.create_namespaced_job.return_value = MagicMock()
        backend.schedule(cpu_execution_provider, test_step_pending)

        call_args = backend._batch_v1.create_namespaced_job.call_args
    job_body = call_args.kwargs["body"]
    main_container = job_body.spec.template.spec.containers[0]
    env_vars = {env.name: env.value for env in main_container.env}

    assert env_vars["NMP_BASE_URL"] == "https://nemo-gateway:8080"
    assert env_vars["NMP_AUTH_URL"] == "https://nemo-auth:8080"
    assert env_vars["NMP_JOBS_URL"] == "https://nemo-gateway:8080"
    assert env_vars["NMP_FILES_URL"] == "https://nemo-gateway:8080"
    assert env_vars["NMP_MODELS_URL"] == "https://nemo-gateway:8080"
    assert env_vars["NMP_SECRETS_URL"] == "https://nemo-gateway:8080"
    assert env_vars[NMP_JOB_LAUNCHER_OTLP_LOGS_ENDPOINT_ENVVAR].startswith("https://nemo-gateway:8080/apis/files/")


def test_kubernetes_job_execution_profile_config_rejects_reserved_env_vars():
    """KubernetesJobExecutionProfileConfig raises when environment contains reserved names."""
    with pytest.raises(ValidationError) as exc_info:
        KubernetesJobExecutionProfileConfig(
            namespace="test",
            storage=DEFAULT_STORAGE,
            env={"NEMO_JOB_ID": "y"},
        )
    assert "NEMO_JOB_ID" in str(exc_info.value)
    assert "reserved" in str(exc_info.value).lower()


def test_kubernetes_job_rejects_reserved_step_auth_env_vars(kubernetes_job, cpu_execution_provider, test_step_pending):
    test_step_pending.step_spec.environment.append(
        PlatformJobEnvironmentVariable(name="NEMO_WORKFLOW_TOKEN", value="token")
    )

    with pytest.raises(ValueError, match="NEMO_WORKFLOW_TOKEN"):
        kubernetes_job.schedule(cpu_execution_provider, test_step_pending)


def test_kubernetes_job_injects_projected_workload_identity_token_when_exchange_enabled(
    kubernetes_job, cpu_execution_provider, test_step_pending_with_auth_context
):
    kubernetes_job._execution_profile_config.workload_identity.token_expiration_seconds = 600
    kubernetes_job._execution_profile_config.workload_identity.token_audience = "test-audience"
    auth_config = SimpleNamespace(oidc=SimpleNamespace(workload_token_exchange_enabled=True))

    with patch("nmp.common.config.get_auth_config", return_value=auth_config):
        kubernetes_job.schedule(cpu_execution_provider, test_step_pending_with_auth_context)

    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args
    job_body = call_args.kwargs["body"]
    pod_spec = job_body.spec.template.spec
    main_container = pod_spec.containers[0]

    env_vars = {env.name: env.value for env in main_container.env}
    assert env_vars[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR] == WORKLOAD_IDENTITY_TOKEN_FILE_PATH
    assert NMP_PRINCIPAL_ENVVAR not in env_vars

    workload_identity_volume = next(
        volume for volume in pod_spec.volumes if volume.name == WORKLOAD_IDENTITY_VOLUME_NAME
    )
    projection = workload_identity_volume.projected.sources[0].service_account_token
    assert projection.path == "token"
    assert projection.expiration_seconds == 600
    assert projection.audience == "test-audience"

    mount = next(vm for vm in main_container.volume_mounts if vm.name == WORKLOAD_IDENTITY_VOLUME_NAME)
    assert mount.mount_path == WORKLOAD_IDENTITY_VOLUME_PATH
    assert mount.read_only is True


def test_kubernetes_job_does_not_mount_workload_identity_without_auth_context(
    kubernetes_job, cpu_execution_provider, test_step_pending
):
    auth_config = SimpleNamespace(oidc=SimpleNamespace(workload_token_exchange_enabled=True))

    with patch("nmp.common.config.get_auth_config", return_value=auth_config):
        kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args
    job_body = call_args.kwargs["body"]
    pod_spec = job_body.spec.template.spec
    main_container = pod_spec.containers[0]

    env_vars = {env.name: env.value for env in main_container.env}
    assert WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR not in env_vars
    assert all(volume.name != WORKLOAD_IDENTITY_VOLUME_NAME for volume in pod_spec.volumes)
    assert all(mount.name != WORKLOAD_IDENTITY_VOLUME_NAME for mount in main_container.volume_mounts)


def test_schedule_job_with_args(kubernetes_job, cpu_execution_provider, test_step_pending):
    """Test job scheduling with custom args."""

    # Mock successful job creation
    kubernetes_job._batch_v1.create_namespaced_job.return_value = MagicMock()

    # Schedule the job
    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    # Get the created job
    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args
    job_body = call_args.kwargs["body"]
    main_container = job_body.spec.template.spec.containers[0]

    # Check args was set
    assert main_container.args == ["python", "-c", "print('Hello World')"]


def test_schedule_job_with_custom_service_account_name(kubernetes_job, cpu_execution_provider, test_step_pending):
    """Test that a custom service_account_name from config is set on the job pod spec."""
    kubernetes_job._execution_profile_config.service_account_name = "nmp-jobs-sa"
    kubernetes_job._batch_v1.create_namespaced_job.return_value = MagicMock()

    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args
    job_body = call_args.kwargs["body"]
    pod_spec = job_body.spec.template.spec

    assert pod_spec.service_account_name == "nmp-jobs-sa"


def test_schedule_job_api_exception(kubernetes_job, cpu_execution_provider, test_step_pending):
    """Test job scheduling with API exception."""

    # Mock API exception
    kubernetes_job._batch_v1.create_namespaced_job.side_effect = ApiException("Test error")

    # Scheduling should raise the exception
    with pytest.raises(ApiException):
        kubernetes_job.schedule(cpu_execution_provider, test_step_pending)


def test_schedule_from_resuming_existing_job(kubernetes_job, cpu_execution_provider, test_step_resuming):
    """Test scheduling when resuming from an existing job."""
    # Mock existing job found
    kubernetes_job._batch_v1.read_namespaced_job.return_value = MagicMock()

    # Schedule the job
    kubernetes_job.schedule(cpu_execution_provider, test_step_resuming)

    # Verify that no new job was created
    kubernetes_job._batch_v1.create_namespaced_job.assert_not_called()


def test_schedule_from_resuming_no_existing_job(kubernetes_job, cpu_execution_provider, test_step_resuming):
    """Test scheduling when resuming but no existing job is found."""
    # Mock existing job not found
    kubernetes_job._batch_v1.read_namespaced_job.side_effect = ApiException(status=404)

    # Schedule the job
    kubernetes_job.schedule(cpu_execution_provider, test_step_resuming)

    # Verify job creation was called
    kubernetes_job._batch_v1.create_namespaced_job.assert_called_once()


def test_sync_job_active(kubernetes_job, test_step_pending):
    """Test syncing an active job."""
    # Mock active job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = 1
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock pod status to return running pods
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={},
                completed=set(),
                active={"test-container"},
                waiting={},
                phase="Running",
            )
        ]
        # Sync the job
        job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.ACTIVE


def test_sync_job_completed(kubernetes_job, test_step_pending):
    """Test syncing a completed job."""
    # Mock completed job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = 1
    mock_job_status.failed = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Sync the job
    job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.COMPLETED


def test_sync_job_pausing(kubernetes_job, test_step_pending):
    """Test syncing a pausing job."""
    # Mock pausing job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = True

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = 1
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock pod status to return running pods (still terminating)
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={},
                completed=set(),
                active={"test-container"},
                waiting={},
                phase="Running",
            )
        ]
        # Sync the job
        job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.PAUSING


def test_sync_job_paused(kubernetes_job, test_step_pending):
    """Test syncing a paused job."""
    # Mock paused job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = True

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock pod status to return no running pods (paused)
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = []
        # Sync the job
        job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.PAUSED


def test_sync_job_active_to_paused_to_resumed(kubernetes_job, test_step_active, test_step_pausing, test_step_pending):
    """Test syncing a job that was paused and then resumed."""
    # Mock resumed job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = True  # Job is suspended so the active step should now be pausing

    mock_job_status = MagicMock()
    mock_job_status.active = 1
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock pod status to return running pods (still active)
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={},
                completed=set(),
                active={"test-container"},
                waiting={},
                phase="Running",
            )
        ]
        # Sync the job
        job_update = kubernetes_job.sync(test_step_active)

    assert job_update.status == PlatformJobStatus.PAUSING

    # The Job should now be paused after the status is no longer active
    mock_job_status.active = None
    mock_job.status = mock_job_status
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock pod status to return no running pods (paused)
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = []
        # Sync the job again
        job_update = kubernetes_job.sync(test_step_pausing)
    assert job_update.status == PlatformJobStatus.PAUSED

    # Externally we resume the job, so the job spec is no longer suspended
    mock_job_spec.suspend = False
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock pod status to return pending pods (resuming)
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={},
                completed=set(),
                active=set(),
                waiting={"test-container": "ContainerCreating"},
                phase="Pending",
            )
        ]
        # Sync the job again, it should now be in pending state
        job_update = kubernetes_job.sync(test_step_pending)
    assert job_update.status == PlatformJobStatus.PENDING


def test_sync_job_paused_with_errored_pods_from_sigterm(kubernetes_job, test_step_pending):
    """Test that a suspended job with errored pods (killed by SIGTERM) reports PAUSED, not ERROR.

    When K8s suspends a Job it sends SIGTERM to running pods. The terminated pods
    show up with errors (non-zero exit code). The reconciler must recognise this as
    a normal part of suspension and return PAUSED rather than ERROR.

    Regression test for AIRCORE-853.
    """
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = True

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Pod was killed by SIGTERM during suspension — has errors but no running containers
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={"test-container": 137},
                completed=set(),
                active=set(),
                waiting={},
                phase="Failed",
            )
        ]
        job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.PAUSED


def test_sync_job_paused_ignores_task_errors_from_suspend(kubernetes_job, test_step_pending):
    """Task-level pod errors observed during suspension must not override PAUSED."""
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = True

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks") as mock_update_all_tasks,
    ):
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={"test-container": 137},
                completed=set(),
                active=set(),
                waiting={},
                phase="Failed",
            )
        ]
        mock_update_all_tasks.return_value = True

        job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.PAUSED


def test_sync_job_pausing_with_errored_pods_from_sigterm(kubernetes_job, test_step_pending):
    """Test that a suspended job with both running and errored pods reports PAUSING, not ERROR.

    During suspension, some pods may already be terminated (errored) while others
    are still running. The reconciler should report PAUSING.

    Regression test for AIRCORE-853.
    """
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = True

    mock_job_status = MagicMock()
    mock_job_status.active = 1
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task-1",
                name="test-pod-1",
                errors={"test-container": 137},
                completed=set(),
                active=set(),
                waiting={},
                phase="Failed",
            ),
            PodStatus(
                task_id="test-task-2",
                name="test-pod-2",
                errors={},
                completed=set(),
                active={"test-container"},
                waiting={},
                phase="Running",
            ),
        ]
        job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.PAUSING


def test_sync_job_cancelling_with_errored_pods(kubernetes_job, test_step_cancelling):
    """Test that a cancelling job with errored pods reports CANCELLED, not ERROR.

    Same race as suspension — K8s kills pods during cancellation.

    Regression test for AIRCORE-853.
    """
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={"test-container": 137},
                completed=set(),
                active=set(),
                waiting={},
                phase="Failed",
            )
        ]
        job_update = kubernetes_job.sync(test_step_cancelling)

    assert job_update.status == PlatformJobStatus.CANCELLED


def test_sync_job_cancelled_ignores_task_errors_from_termination(kubernetes_job, test_step_cancelling):
    """Task-level pod errors observed during termination must not override CANCELLED."""
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks") as mock_update_all_tasks,
    ):
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={"test-container": 137},
                completed=set(),
                active=set(),
                waiting={},
                phase="Failed",
            )
        ]
        mock_update_all_tasks.return_value = True

        job_update = kubernetes_job.sync(test_step_cancelling)

    assert job_update.status == PlatformJobStatus.CANCELLED


def test_sync_job_cancelling(kubernetes_job, test_step_cancelling):
    """Test syncing a cancelling job that isn't ready to be cancelled."""
    # Mock cancelling job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = 1
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock pod status to return running pods (still terminating)
    with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status") as mock_list_pod_status:
        mock_list_pod_status.return_value = [
            PodStatus(
                task_id="test-task",
                name="test-pod",
                errors={},
                completed=set(),
                active={"test-container"},
                waiting={},
                phase="Running",
            )
        ]
        # Sync the job
        job_update = kubernetes_job.sync(test_step_cancelling)

    assert job_update.status == PlatformJobStatus.CANCELLING


def test_sync_job_cancelling_to_cancelled(kubernetes_job, test_step_cancelling):
    """Test syncing a cancelling job."""
    # Mock cancelling job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = None
    mock_job_status.terminating = None
    mock_job_status.completion_time = None

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Sync the job
    job_update = kubernetes_job.sync(test_step_cancelling)

    assert job_update.status == PlatformJobStatus.CANCELLED


def test_sync_job_failed(kubernetes_job, test_step_pending):
    """Test syncing a failed job."""
    # Mock failed job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = None
    mock_job_status.failed = 1
    mock_job_status.completion_time = None
    mock_job_status.conditions = [MagicMock(type="Failed", status="True", message="Job has failed")]

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Sync the job
    job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.ERROR


def test_sync_job_not_found(kubernetes_job, test_step_pending):
    """Test syncing a job that doesn't exist."""
    # Mock job not found
    kubernetes_job._batch_v1.read_namespaced_job.side_effect = ApiException(status=404)

    # Sync the job
    job_update = kubernetes_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.ERROR


def test_sync_active_when_k8s_job_not_found(kubernetes_job, test_step_active):
    """Test syncing an ACTIVE step when the k8s job is already gone (e.g. deleted).

    Ensures we do not call terminate_job(None), which would raise
    'NoneType' object has no attribute 'metadata'. Instead we fall through to
    sync_active(step, None) which returns ERROR.
    """
    test_step_active.status = PlatformJobStatus.ACTIVE
    kubernetes_job._batch_v1.read_namespaced_job.side_effect = ApiException(status=404)

    job_update = kubernetes_job.sync(test_step_active)

    assert job_update.status == PlatformJobStatus.ERROR
    assert job_update.error_details is not None
    assert "Job not found" in job_update.error_details.get("message", "")
    # Must not attempt to delete a non-existent job (would have been terminate_job(None))
    kubernetes_job._batch_v1.delete_namespaced_job.assert_not_called()


def test_terminate_job_skips_delete_when_not_managed_by_jobs_controller(kubernetes_job):
    """terminate_job must not delete the job or configmap if the job is not managed by jobs-controller."""
    mock_job = MagicMock()
    mock_job.metadata.name = "unmanaged-job"
    mock_job.metadata.labels = {}  # No JOB_MANAGED_BY_LABEL

    kubernetes_job.terminate_job(mock_job)

    kubernetes_job._batch_v1.delete_namespaced_job.assert_not_called()
    # delete_configmap is only called after delete_namespaced_job; we never reach it
    kubernetes_job._core_v1.delete_namespaced_config_map.assert_not_called()


def test_get_kubernetes_job_list_by_labels_success(kubernetes_job):
    """get_kubernetes_job_list_by_labels calls list_namespaced_job with correct namespace and label_selector and returns items."""
    mock_job1 = MagicMock()
    mock_job1.metadata.name = "job-1"
    mock_job2 = MagicMock()
    mock_job2.metadata.name = "job-2"
    mock_list = MagicMock()
    mock_list.items = [mock_job1, mock_job2]

    kubernetes_job._batch_v1.list_namespaced_job.return_value = mock_list

    labels = KUBE_JOB_SELECTOR_LABELS
    result = kubernetes_job.get_kubernetes_job_list_by_labels(labels=labels)

    kubernetes_job._batch_v1.list_namespaced_job.assert_called_once_with(
        namespace=kubernetes_job.namespace,
        label_selector=",".join([f"{k}={v}" for k, v in labels.items()]),
    )
    assert result == [mock_job1, mock_job2]


def test_get_kubernetes_job_list_by_labels_api_error_returns_empty(kubernetes_job):
    """get_kubernetes_job_list_by_labels returns empty list on ApiException."""
    kubernetes_job._batch_v1.list_namespaced_job.side_effect = ApiException("API error")

    result = kubernetes_job.get_kubernetes_job_list_by_labels(labels=KUBE_JOB_SELECTOR_LABELS)

    assert result == []


def test_get_kubernetes_job_list_by_labels_exception_returns_empty(kubernetes_job):
    """get_kubernetes_job_list_by_labels returns empty list on any other Exception."""
    kubernetes_job._batch_v1.list_namespaced_job.side_effect = RuntimeError("unexpected error")

    result = kubernetes_job.get_kubernetes_job_list_by_labels(labels=KUBE_JOB_SELECTOR_LABELS)

    assert result == []


def test_name_for_job_truncation(kubernetes_job):
    """Test job name truncation for very long job IDs."""
    # Create a job with a very long ID
    long_job = PlatformJobStepWithContext(
        id="jobstep-a" * 100 + "-",  # Very long ID with trailing dash
        job="test-job-id",
        attempt_id="test-job-attempt-id",
        fileset="test-logs-fileset",
        workspace="default",
        name="test-step-",  # Job name with trailing dash.
        step_spec=PlatformJobStepSpec(
            name="test-step",
            executor=CPUExecutionProvider(
                provider="cpu", profile="k8s_profile", container=ContainerSpec(image="test-image")
            ),
            config={"command": ["echo", "Hello"]},
            environment=[PlatformJobEnvironmentVariable(name="TEST_VAR", value="test_value")],
        ),
        status_details={},
        status=PlatformJobStatus.CREATED,
    )

    job_name = name_for_step(long_job)

    # Should be truncated to 63 characters and not end with dash
    assert len(job_name) <= 63
    assert not job_name.endswith("-")


def test_schedule_kubernetes_gpu(mock_nmp_client, kubernetes_execution_profile_config):
    """Test successful job scheduling."""

    gpu_executor_config = GPUExecutionProvider.model_validate(
        {
            "provider": "gpu",
            "profile": "default",
            "container": {
                "image": "hello-world:latest",
                "command": ["c1"],
            },
            "resources": {
                "num_gpus": 2,
            },
        }
    )
    step = PlatformJobStepWithContext.model_validate(
        {
            "id": "test-step-id",
            "job": "test-job-id",
            "attempt_id": "test-job-attempt-id",
            "name": "test-step",
            "fileset": "test-logs-fileset",
            "workspace": "default",
            "step_spec": {
                "name": "test-step",
                "executor": gpu_executor_config.model_dump(),
                "config": {"foo": "test", "bar": 1},
                "environment": [{"name": "ENV_VAR", "value": "test_value"}],
            },
            "status": "pending",
        }
    )

    # patch the check for kubernetes
    with patch("kubernetes.config.load_incluster_config"):
        assert step is not None
        executor = GPUKubernetesJobBackend(
            nmp_sdk=mock_nmp_client,
            execution_profile_config=kubernetes_execution_profile_config,
            profile_name="default",
        )

        # Mock successful job creation
        executor._batch_v1 = MagicMock()
        executor._batch_v1.create_namespaced_job.return_value = MagicMock()
        executor._core_v1 = MagicMock()
        executor._core_v1.read_namespaced_persistent_volume_claim.return_value = MagicMock()

        executor.schedule(executor_config=gpu_executor_config, step=step)

        # Verify job creation was called
        executor._batch_v1.create_namespaced_job.assert_called_once()
        call_args = executor._batch_v1.create_namespaced_job.call_args

        # Check namespace
        assert call_args.kwargs["namespace"] == "test-namespace"

        # Check job object
        job_body = call_args.kwargs["body"]
        assert job_body.api_version == "batch/v1"
        assert job_body.kind == "Job"

        # Check metadata
        assert "test-step-id" in job_body.metadata.name
        assert job_body.metadata.labels[JOB_ID_LABEL] == "test-job-id"
        assert job_body.metadata.labels[JOB_STEP_NAME_LABEL] == "test-step"
        assert job_body.metadata.labels["app"] == "nemo-job"
        assert job_body.metadata.labels[JOB_EXECUTION_BACKEND_LABEL] == "kubernetes_job"
        assert job_body.metadata.labels[JOB_EXECUTION_PROFILE_LABEL] == "default"

        # Check job spec
        assert job_body.spec.backoff_limit == 0
        assert job_body.spec.ttl_seconds_after_finished == 300

        # Check pod template
        pod_template = job_body.spec.template
        pod_spec = pod_template.spec
        pod_metadata = pod_template.metadata
        assert pod_metadata.labels[JOB_EXECUTION_BACKEND_LABEL] == "kubernetes_job"
        assert pod_metadata.labels[JOB_EXECUTION_PROFILE_LABEL] == "default"

        # Check basic pod settings
        assert pod_spec.restart_policy == "Never"
        assert pod_spec.active_deadline_seconds == 86400
        assert pod_spec.service_account_name == "default"

        # Check tolerations
        assert pod_spec.tolerations is not None
        assert len(pod_spec.tolerations) == 2
        assert pod_spec.tolerations[0].key == "nvidia.com/gpu"

        # Check node selector
        assert pod_spec.node_selector is not None
        assert pod_spec.node_selector["accelerator"] == "nvidia-tesla-v100"
        assert pod_spec.node_selector["node-type"] == "gpu"

        # Check affinity
        assert pod_spec.affinity is not None
        assert pod_spec.affinity.node_affinity is not None

        # Check containers
        assert len(pod_spec.containers) == 1
        main_container = pod_spec.containers[0]
        assert main_container.name == "nemo-job-task"
        assert main_container.image == "hello-world:latest"

        # Check resources
        num_gpus = pod_spec.containers[0].resources.limits["nvidia.com/gpu"]
        assert int(num_gpus) == 2

        # GPU steps keep whatever the image sets for device visibility
        assert "NVIDIA_VISIBLE_DEVICES" not in {env.name for env in main_container.env}

        # GPU jobs get a memory-backed /dev/shm (default 1Gi per GPU)
        dshm_vol = next((v for v in pod_spec.volumes if v.name == JOB_DSHM_VOLUME_NAME), None)
        assert dshm_vol is not None
        assert dshm_vol.empty_dir is not None
        assert dshm_vol.empty_dir.medium == "Memory"
        assert dshm_vol.empty_dir.size_limit == "2Gi"
        dshm_mount = next((vm for vm in main_container.volume_mounts if vm.name == JOB_DSHM_VOLUME_NAME), None)
        assert dshm_mount is not None
        assert dshm_mount.mount_path == "/dev/shm"

        # Check environment variables
        env_vars = {env.name: env.value for env in main_container.env}
        assert "ENV_VAR" in env_vars
        assert env_vars["ENV_VAR"] == "test_value"
        assert env_vars[NEMO_JOB_WORKSPACE_ENVVAR] == "default"
        assert env_vars[NEMO_JOB_FILESET_ENVVAR] == "test-logs-fileset"
        assert "NMP_BASE_URL" in env_vars


def test_schedule_with_storage_integration(kubernetes_job, cpu_execution_provider, test_step_pending):
    """Test that scheduling with storage integrates PVC mount correctly."""
    kubernetes_job._execution_profile_config.storage.pvc_name = "job-storage-pvc"
    mock_create_job = kubernetes_job._batch_v1.create_namespaced_job

    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    # Verify job was created with storage mount
    mock_create_job.assert_called_once()
    call_args = mock_create_job.call_args
    job_body = call_args.kwargs["body"]

    # Check that the persistent storage label is set
    assert JOB_USES_PERSISTENT_STORAGE_LABEL in job_body.metadata.labels
    assert job_body.metadata.labels[JOB_USES_PERSISTENT_STORAGE_LABEL] == "true"

    # Check volumes
    volumes = job_body.spec.template.spec.volumes
    job_storage_volume = next((v for v in volumes if v.name == JOB_STORAGE_VOLUME_NAME), None)
    assert job_storage_volume is not None
    assert job_storage_volume.persistent_volume_claim.claim_name == "job-storage-pvc"

    # Check volume mounts
    main_container = job_body.spec.template.spec.containers[0]
    job_storage_mount = next((vm for vm in main_container.volume_mounts if vm.name == JOB_STORAGE_VOLUME_NAME), None)
    assert job_storage_mount is not None
    assert job_storage_mount.mount_path == "/var/test"
    assert job_storage_mount.sub_path == f"jobs/{test_step_pending.workspace}/{test_step_pending.job}"

    # Check permissions init container (mounts full volume, creates subpath and chmods in one shot)
    init_containers = job_body.spec.template.spec.init_containers
    fix_permissions_container = next((c for c in init_containers if c.name == "fix-permissions"), None)
    assert fix_permissions_container is not None
    fix_permissions_mount = fix_permissions_container.volume_mounts[0]
    assert fix_permissions_mount.name == JOB_STORAGE_VOLUME_NAME
    assert fix_permissions_mount.mount_path == "/vol"
    assert fix_permissions_mount.sub_path is None  # full volume mount; container creates subpath via mkdir
    expected_subpath = f"jobs/{test_step_pending.workspace}/{test_step_pending.job}"
    assert f"/vol/{expected_subpath}" in " ".join(fix_permissions_container.command or [])
    assert "mkdir -p" in " ".join(fix_permissions_container.command or [])
    assert "chmod -R 777" in " ".join(fix_permissions_container.command or [])

    # Check the job container has a termination message policy set
    assert main_container.termination_message_policy == "FallbackToLogsOnError"


def test_schedule_nemo_job_secrets_format_same_and_cross_workspace(kubernetes_job, cpu_execution_provider):
    """NEMO_JOB_SECRETS env var is correctly formatted for same-workspace and cross-workspace secret refs.

    Jobs can reference secrets from other workspaces when the user has permissions.
    Format must be ENV_VAR=workspace/secret_name; cross-workspace refs use explicit workspace/secret_name.
    """
    test_step = PlatformJobStepWithContext(
        id="test-step-id",
        job="test-job-id",
        workspace="default",
        attempt_id="test-job-attempt-id",
        name="test-step",
        fileset="test-logs-fileset",
        step_spec=PlatformJobStepSpec(
            name="test-step",
            executor=CPUExecutionProvider(
                provider="cpu", profile="default", container=ContainerSpec(image="test-image")
            ),
            config={},
            environment=[
                PlatformJobEnvironmentVariable(
                    name="LOCAL_SECRET",
                    from_secret=PlatformJobSecretEnvironmentVariableRef(name="local-secret"),
                ),
                PlatformJobEnvironmentVariable(
                    name="CROSS_WORKSPACE_SECRET",
                    from_secret=PlatformJobSecretEnvironmentVariableRef(name="other-ws/shared-secret"),
                ),
            ],
        ),
        status=PlatformJobStatus.PENDING,
    )

    kubernetes_job.schedule(cpu_execution_provider, test_step)
    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args
    job_body = call_args.kwargs["body"]
    main_container = job_body.spec.template.spec.containers[0]
    env_vars = {env.name: env.value for env in main_container.env}

    nemo_secrets = env_vars.get("NEMO_JOB_SECRETS", "")
    assert "LOCAL_SECRET=default/local-secret" in nemo_secrets
    assert "CROSS_WORKSPACE_SECRET=other-ws/shared-secret" in nemo_secrets
    parts = [p.strip() for p in nemo_secrets.split(",")]
    assert len(parts) == 2
    assert set(parts) == {
        "LOCAL_SECRET=default/local-secret",
        "CROSS_WORKSPACE_SECRET=other-ws/shared-secret",
    }


def test_schedule_injects_opensandbox_secret_env_when_cluster_capable(
    kubernetes_job, cpu_execution_provider, test_step_pending, mock_platform_config
):
    mock_platform_config.sandbox_cluster_capable = True
    mock_platform_config.sandbox_server_domain = "opensandbox-server.opensandbox-system.svc.cluster.local"
    mock_platform_config.sandbox_api_key_secret = "opensandbox-server-api-key"
    mock_platform_config.sandbox_api_key_secret_key = "api-key"

    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)
    job_body = kubernetes_job._batch_v1.create_namespaced_job.call_args.kwargs["body"]
    env_by_name = {env.name: env for env in job_body.spec.template.spec.containers[0].env}

    assert env_by_name["OPEN_SANDBOX_DOMAIN"].value == ("opensandbox-server.opensandbox-system.svc.cluster.local")
    secret_ref = env_by_name["OPEN_SANDBOX_API_KEY"].value_from.secret_key_ref
    assert secret_ref.name == "opensandbox-server-api-key"
    assert secret_ref.key == "api-key"
    assert env_by_name["OPEN_SANDBOX_API_KEY"].value is None


def test_schedule_hides_gpus_from_cpu_steps(kubernetes_job, cpu_execution_provider, test_step_pending):
    """CPU steps run NGC-derived images that bake in NVIDIA_VISIBLE_DEVICES=all."""
    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    job_body = kubernetes_job._batch_v1.create_namespaced_job.call_args.kwargs["body"]
    env_vars = {env.name: env.value for env in job_body.spec.template.spec.containers[0].env}
    assert env_vars["NVIDIA_VISIBLE_DEVICES"] == "void"


def test_schedule_lets_profile_override_gpu_visibility_for_cpu_steps(
    kubernetes_job, cpu_execution_provider, test_step_pending
):
    kubernetes_job._execution_profile_config.env = {"NVIDIA_VISIBLE_DEVICES": "all"}

    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    job_body = kubernetes_job._batch_v1.create_namespaced_job.call_args.kwargs["body"]
    visibility = [
        env.value for env in job_body.spec.template.spec.containers[0].env if env.name == "NVIDIA_VISIBLE_DEVICES"
    ]
    assert visibility == ["all"]


def test_schedule_omits_opensandbox_env_when_cluster_not_capable(
    kubernetes_job, cpu_execution_provider, test_step_pending, mock_platform_config
):
    mock_platform_config.sandbox_cluster_capable = False
    mock_platform_config.sandbox_server_domain = "opensandbox-server.opensandbox-system.svc.cluster.local"
    mock_platform_config.sandbox_api_key_secret = "opensandbox-server-api-key"

    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)
    job_body = kubernetes_job._batch_v1.create_namespaced_job.call_args.kwargs["body"]
    env_names = {env.name for env in job_body.spec.template.spec.containers[0].env}
    assert "OPEN_SANDBOX_DOMAIN" not in env_names
    assert "OPEN_SANDBOX_API_KEY" not in env_names


def test_schedule_without_storage_no_label(kubernetes_job, cpu_execution_provider):
    """Test that scheduling without persistent storage sets the persistent storage label to 'false'."""
    # Create a step without PERSISTENT_JOB_STORAGE_PATH_ENVVAR
    test_step_no_storage = PlatformJobStepWithContext(
        id="test-step-id",
        job="test-job-id",
        workspace="default",
        attempt_id="test-job-attempt-id",
        name="test-step",
        fileset="test-logs-fileset",
        step_spec=PlatformJobStepSpec(
            name="test-step",
            executor=CPUExecutionProvider(
                provider="cpu", profile="default", container=ContainerSpec(image="test-image")
            ),
            config={},
            environment=[
                PlatformJobEnvironmentVariable(name="ENV_VAR", value="test_value"),
                # Note: NO PERSISTENT_JOB_STORAGE_PATH_ENVVAR here
                PlatformJobEnvironmentVariable(name=EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, value="/var/tmp"),
            ],
        ),
        status=PlatformJobStatus.PENDING,
    )

    mock_create_job = kubernetes_job._batch_v1.create_namespaced_job

    kubernetes_job.schedule(cpu_execution_provider, test_step_no_storage)

    # Verify job was created
    mock_create_job.assert_called_once()
    call_args = mock_create_job.call_args
    job_body = call_args.kwargs["body"]

    # Check that the persistent storage label is always set; "false" when not using persistent storage
    assert JOB_USES_PERSISTENT_STORAGE_LABEL in job_body.metadata.labels
    assert job_body.metadata.labels[JOB_USES_PERSISTENT_STORAGE_LABEL] == "false"


def test_schedule_with_additional_volumes(kubernetes_job, cpu_execution_provider, test_step_pending):
    """Test that scheduling with additional volumes integrates correctly."""
    kubernetes_job._execution_profile_config.storage.pvc_name = "job-storage-pvc"
    kubernetes_job._execution_profile_config.storage.additional_volumes = [
        KubernetesVolume(
            name="extra-volume-1",
            persistent_volume_claim=KubernetesPersistentVolumeClaim(claim_name="extra-pvc-1"),
        ),
        KubernetesVolume(
            name="extra-volume-2",
            empty_dir=KubernetesEmptyDirVolume(medium="Memory"),
        ),
        KubernetesVolume(
            name="extra-volume-3",
            secret=KubernetesSecretVolume(
                secret_name="extra-secret",
                items=[KubernetesKeyToPath(key="ca.crt", path="ca.crt")],
            ),
        ),
    ]
    kubernetes_job._execution_profile_config.storage.additional_volume_mounts = [
        KubernetesVolumeMount(name="extra-volume-1", mount_path="/mnt/extra-1"),
        KubernetesVolumeMount(name="extra-volume-2", mount_path="/mnt/extra-2"),
        KubernetesVolumeMount(name="extra-volume-3", mount_path="/mnt/extra-3", read_only=True),
    ]
    mock_create_job = kubernetes_job._batch_v1.create_namespaced_job

    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    # Verify job was created with storage mount
    mock_create_job.assert_called_once()
    call_args = mock_create_job.call_args
    job_body = call_args.kwargs["body"]

    # Check that the extra volumes are mounted
    volumes = job_body.spec.template.spec.volumes
    extra_volume_1 = next((v for v in volumes if v.name == "extra-volume-1"), None)
    assert extra_volume_1 is not None
    assert extra_volume_1.persistent_volume_claim.claim_name == "extra-pvc-1"
    extra_volume_2 = next((v for v in volumes if v.name == "extra-volume-2"), None)
    assert extra_volume_2 is not None
    assert extra_volume_2.empty_dir.medium == "Memory"
    extra_volume_3 = next((v for v in volumes if v.name == "extra-volume-3"), None)
    assert extra_volume_3 is not None
    assert extra_volume_3.secret.secret_name == "extra-secret"
    assert extra_volume_3.secret.items[0].key == "ca.crt"

    # Check volume mounts
    main_container = job_body.spec.template.spec.containers[0]
    volume_mounts = {mount.name: mount for mount in main_container.volume_mounts}
    assert volume_mounts["extra-volume-3"].mount_path == "/mnt/extra-3"
    assert volume_mounts["extra-volume-3"].read_only is True
    job_storage_mount = next((vm for vm in main_container.volume_mounts if vm.name == "extra-volume-1"), None)
    assert job_storage_mount is not None
    assert job_storage_mount.mount_path == "/mnt/extra-1"
    job_storage_mount = next((vm for vm in main_container.volume_mounts if vm.name == "extra-volume-2"), None)
    assert job_storage_mount is not None
    assert job_storage_mount.mount_path == "/mnt/extra-2"


@pytest.mark.parametrize("cleanup_completed_jobs_immediately", [True, False])
def test_cleanup_steps_by_ttl(kubernetes_job, cleanup_completed_jobs_immediately):
    kubernetes_job._execution_profile_config.cleanup_completed_jobs_immediately = cleanup_completed_jobs_immediately

    # Step cleanup proceeds for terminal steps or when the step entity is gone.
    kubernetes_job.check_step_is_terminal = MagicMock(return_value=True)

    # Mock active job status
    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False

    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = 1
    mock_job_status.failed = None
    mock_job_status.completion_time = datetime.datetime.now(datetime.UTC)

    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    mock_job.metadata.name = "test-job-completed"
    mock_job.metadata.labels = {
        JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        JOB_WORKSPACE_ID_LABEL: "default",
        JOB_ID_LABEL: "test-job-id",
        JOB_STEP_NAME_LABEL: "test-step",
    }

    mock_failed_job_spec = MagicMock()
    mock_failed_job_spec.suspend = False

    mock_failed_job_status = MagicMock()
    mock_failed_job_status.active = None
    mock_failed_job_status.succeeded = None
    mock_failed_job_status.failed = 1
    mock_failed_job_status.completion_time = None
    mock_failed_job_status.conditions = [MagicMock(type="Failed", status="True", message="Job has failed")]

    mock_failed_job = MagicMock()
    mock_failed_job.status = mock_failed_job_status
    mock_failed_job.spec = mock_failed_job_spec
    mock_failed_job.metadata.name = "test-job-failed"
    mock_failed_job.metadata.labels = {
        JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        JOB_WORKSPACE_ID_LABEL: "default",
        JOB_ID_LABEL: "test-job-id",
        JOB_STEP_NAME_LABEL: "test-step",
    }

    mock_active_job_spec = MagicMock()
    mock_active_job_spec.suspend = False

    mock_active_job_status = MagicMock()
    mock_active_job_status.active = 1
    mock_active_job_status.succeeded = None
    mock_active_job_status.failed = None
    mock_active_job_status.completion_time = None
    mock_active_job_status.conditions = None

    mock_active_job = MagicMock()
    mock_active_job.status = mock_active_job_status
    mock_active_job.spec = mock_active_job_spec
    mock_active_job.metadata.name = "test-job-active"
    mock_active_job.metadata.labels = {
        JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        JOB_WORKSPACE_ID_LABEL: "default",
        JOB_ID_LABEL: "test-job-id",
        JOB_STEP_NAME_LABEL: "test-step",
    }

    mock_job_list = [mock_job, mock_failed_job, mock_active_job]
    mock_jobv1list = MagicMock()
    mock_jobv1list.items = mock_job_list
    kubernetes_job._batch_v1.list_namespaced_job.return_value = mock_jobv1list

    kubernetes_job.cleanup_steps()

    # Cleanup must list jobs using the selector that includes JOB_MANAGED_BY so only jobs-controller jobs are cleaned
    list_call = kubernetes_job._batch_v1.list_namespaced_job.call_args
    label_selector = list_call.kwargs["label_selector"]
    expected_selector = ",".join([f"{k}={v}" for k, v in KUBE_JOB_SELECTOR_LABELS.items()])
    assert label_selector == expected_selector
    assert f"{JOB_MANAGED_BY_LABEL}={JOB_MANAGED_BY_JOBS_CONTROLLER}" in label_selector

    if cleanup_completed_jobs_immediately:
        kubernetes_job._batch_v1.delete_namespaced_job.assert_called_once()
        assert kubernetes_job._batch_v1.delete_namespaced_job.call_args.kwargs["name"] == "test-job-completed"
    else:
        assert kubernetes_job._batch_v1.delete_namespaced_job.call_count == 0


def test_cleanup_pending_by_ttl(kubernetes_job, test_step_pending):
    """Test that sync of a PENDING step transitions to an ERROR state when step's created_at exceeds TTL."""
    # Get the TTL configuration (default is 30 minutes)
    ttl_seconds = kubernetes_job._execution_profile_config.ttl_seconds_before_active

    # Set the step to PENDING status
    test_step_pending.status = PlatformJobStatus.PENDING

    # Create a step with an created_at timestamp that exceeds the TTL (35 minutes ago)
    old_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=ttl_seconds + 300)
    test_step_pending.created_at = old_timestamp
    test_step_pending.updated_at = old_timestamp

    # Create a mock Kubernetes Job that the sync method will find (with managed-by label so terminate_job deletes it)
    mock_job = MagicMock()
    mock_job.metadata.name = "test-job-pending"
    mock_job.metadata.labels = {JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER}
    mock_job.status.active = None
    mock_job.status.completion_time = None

    # Mock get_job_by_name to return our job
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock get_kube_job_events to return some events
    with patch.object(kubernetes_job, "get_kube_job_events", return_value=[]):
        # Mock update_all_tasks
        with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks"):
            # Call sync which should detect the TTL timeout
            result = kubernetes_job.sync(test_step_pending)

    # Verify that it returns an ERROR status with timeout message
    assert result.status == PlatformJobStatus.ERROR.value
    assert result.status_details["message"] == "Job timed out after reaching max TTL of 1800 seconds"
    assert result.error_details == {"message": "Job timed out after reaching max TTL of 1800 seconds"}

    # Verify that the job was terminated (delete_namespaced_job was called)
    kubernetes_job._batch_v1.delete_namespaced_job.assert_called_once()
    call_args = kubernetes_job._batch_v1.delete_namespaced_job.call_args
    assert call_args.kwargs["name"] == "test-job-pending"
    assert call_args.kwargs["namespace"] == "test-namespace"


def test_cleanup_active_by_ttl(kubernetes_job, test_step_active):
    """Test that sync of an ACTIVE step transitions to an ERROR state when step's created_at exceeds TTL."""
    # Get the TTL configuration for active jobs (default is 24 hours)
    ttl_seconds = kubernetes_job._execution_profile_config.ttl_seconds_active

    # Set the step to ACTIVE status
    test_step_active.status = PlatformJobStatus.ACTIVE

    # Create a step with an created_at timestamp that exceeds the TTL (25 hours ago)
    old_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=ttl_seconds + 3600)
    test_step_active.created_at = old_timestamp

    # Create a mock Kubernetes Job that the sync method will find (with managed-by label so terminate_job deletes it)
    mock_job = MagicMock()
    mock_job.metadata.name = "test-job-active"
    mock_job.metadata.labels = {JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER}
    mock_job.status.active = 1
    mock_job.status.completion_time = None

    # Mock get_job_by_name to return our job
    kubernetes_job._batch_v1.read_namespaced_job.return_value = mock_job

    # Mock get_kube_job_events to return some events
    with patch.object(kubernetes_job, "get_kube_job_events", return_value=[]):
        # Mock update_all_tasks
        with patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks"):
            # Call sync which should detect the TTL timeout
            result = kubernetes_job.sync(test_step_active)

    # Verify that it returns an ERROR status with timeout message
    assert result.status == PlatformJobStatus.ERROR.value
    assert result.status_details["message"] == "Job timed out after reaching max TTL of 86400 seconds"
    assert result.error_details == {"message": "Job timed out after reaching max TTL of 86400 seconds"}

    # Verify that the job was terminated (delete_namespaced_job was called)
    kubernetes_job._batch_v1.delete_namespaced_job.assert_called_once()
    call_args = kubernetes_job._batch_v1.delete_namespaced_job.call_args
    assert call_args.kwargs["name"] == "test-job-active"
    assert call_args.kwargs["namespace"] == "test-namespace"


# =============================================================================
# Auth Context Tests
# =============================================================================


@pytest.fixture
def test_step_pending_with_auth_context() -> PlatformJobStepWithContext:
    """Create a test job step with auth context for testing."""
    from nmp.common.auth import AuthContext

    return PlatformJobStepWithContext(
        id="test-step-id",
        job="test-job-id",
        attempt_id="test-job-attempt-id",
        fileset="test-logs-fileset",
        workspace="default",
        name="test-step",
        step_spec=PlatformJobStepSpec(
            name="test-step",
            executor=CPUExecutionProvider(
                provider="cpu",
                profile="default",
                container=ContainerSpec(image="nvidia/cuda:11.8-runtime-ubuntu20.04"),
            ),
            config={},
            environment=[
                PlatformJobEnvironmentVariable(name="ENV_VAR", value="test_value"),
                PlatformJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/var/test"),
                PlatformJobEnvironmentVariable(name=EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, value="/var/tmp"),
            ],
        ),
        status=PlatformJobStatus.PENDING,
        auth_context=AuthContext(
            principal_id="creator@example.com",
            principal_email="creator@example.com",
            principal_groups=["engineering", "ml-team"],
        ),
    )


def test_kubernetes_job_schedule_with_auth_context(
    kubernetes_job, cpu_execution_provider, test_step_pending_with_auth_context
):
    """Test that scheduling sets NMP_PRINCIPAL and launcher OTLP headers when auth_context is present.

    Verifies GitLab issue #3390 Gap 2: job tasks should run with the creating
    user's auth context, propagated via the NMP_PRINCIPAL environment variable
    and private launcher OTLP headers for authenticated telemetry export.
    """
    import json

    from nmp.common.auth.models import NMP_PRINCIPAL_ENVVAR

    # Mock successful job creation
    kubernetes_job._batch_v1.create_namespaced_job.return_value = MagicMock()

    # Schedule the job
    kubernetes_job.schedule(cpu_execution_provider, test_step_pending_with_auth_context)

    # Verify job creation was called
    kubernetes_job._batch_v1.create_namespaced_job.assert_called_once()
    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args

    # Get container env vars
    job_body = call_args.kwargs["body"]
    pod_spec = job_body.spec.template.spec
    main_container = pod_spec.containers[0]
    env_vars = {env.name: env.value for env in main_container.env if env.value is not None}
    env_var_names = {env.name for env in main_container.env}

    # Verify NMP_PRINCIPAL env var is set
    assert NMP_PRINCIPAL_ENVVAR in env_vars
    principal_json = env_vars[NMP_PRINCIPAL_ENVVAR]
    principal_data = json.loads(principal_json)

    assert principal_data == {
        "id": "creator@example.com",
        "email": "creator@example.com",
        "groups": ["engineering", "ml-team"],
    }

    # Verify launcher application log auth is not configured through env headers.
    assert "NMP_JOB_LAUNCHER_OTLP_LOGS_HEADERS" not in env_var_names
    assert "NMP_JOB_LAUNCHER_LOGS_EXPORTER" not in env_var_names
    assert "NMP_JOB_LAUNCHER_OTLP_LOGS_PROTOCOL" not in env_var_names

    # Verify no globally scoped OTEL header environment variables are set
    assert "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT" not in env_var_names
    assert "OTEL_LOGS_EXPORTER" not in env_var_names
    assert "OTEL_SERVICE_NAME" not in env_var_names
    assert "OTEL_EXPORTER_OTLP_LOGS_HEADERS" not in env_var_names


def test_kubernetes_job_schedule_without_auth_context(kubernetes_job, cpu_execution_provider, test_step_pending):
    """Test that auth env vars are NOT set when auth_context is absent."""
    # Mock successful job creation
    kubernetes_job._batch_v1.create_namespaced_job.return_value = MagicMock()

    # Schedule the job (test_step_pending has no auth_context)
    kubernetes_job.schedule(cpu_execution_provider, test_step_pending)

    # Verify job creation was called
    kubernetes_job._batch_v1.create_namespaced_job.assert_called_once()
    call_args = kubernetes_job._batch_v1.create_namespaced_job.call_args

    # Get container env vars
    job_body = call_args.kwargs["body"]
    pod_spec = job_body.spec.template.spec
    main_container = pod_spec.containers[0]
    env_vars = {env.name: env.value for env in main_container.env if env.value is not None}

    env_var_names = {env.name for env in main_container.env}

    # Verify auth env vars are NOT set
    assert NMP_PRINCIPAL_ENVVAR not in env_vars
    assert "NMP_JOB_LAUNCHER_OTLP_LOGS_HEADERS" not in env_var_names
    assert "NMP_JOB_LAUNCHER_LOGS_EXPORTER" not in env_var_names
    assert "NMP_JOB_LAUNCHER_OTLP_LOGS_PROTOCOL" not in env_var_names


def test_kubernetes_job_registers_pod_uid_workload_delegation(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.PENDING
    ttl_seconds_active = 900
    kubernetes_job._execution_profile_config.ttl_seconds_active = ttl_seconds_active
    pod = _kubernetes_pod("pod-uid-123")
    k8s_job = _kubernetes_job_for_step(test_step_pending_with_auth_context)
    kubernetes_job._batch_v1.read_namespaced_job.return_value = k8s_job
    kubernetes_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])
    before_sync = datetime.datetime.now(datetime.timezone.utc)

    expected_name = reference_delegation_name(
        workload_audience="nemo-platform",
        workload_subject="system:serviceaccount:test-namespace:default",
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value="pod-uid-123",
    )

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(kubernetes_job._workload_delegations, "_register_workload_delegation") as register_delegation,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks", return_value=False),
        patch.object(kubernetes_job, "get_kube_job_events", return_value=[]),
    ):
        update = kubernetes_job.sync(test_step_pending_with_auth_context)

    assert update.status == PlatformJobStatus.ACTIVE
    register_delegation.assert_called_once()
    delegation = register_delegation.call_args.args[0]
    assert delegation.name == expected_name
    assert delegation.workload_subject == "system:serviceaccount:test-namespace:default"
    assert delegation.workload_audience == "nemo-platform"
    assert delegation.workload_workspace == test_step_pending_with_auth_context.workspace
    assert delegation.workload_kind == "job"
    assert delegation.workload_id == test_step_pending_with_auth_context.job
    assert delegation.workload_generation == (
        f"{test_step_pending_with_auth_context.attempt_id}/{test_step_pending_with_auth_context.id}/pod-uid-123"
    )
    assert delegation.job_id == test_step_pending_with_auth_context.job
    assert delegation.attempt_id == test_step_pending_with_auth_context.attempt_id
    assert delegation.step_id == test_step_pending_with_auth_context.id
    assert delegation.auth_context.principal_id == "creator@example.com"
    assert delegation.bound_reference_name == KUBERNETES_POD_UID_REFERENCE_NAME
    assert delegation.bound_reference_value == "pod-uid-123"
    assert delegation.opaque_subject_token_hash is None
    expected_ttl_seconds = ttl_seconds_active + WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS
    assert (
        before_sync + datetime.timedelta(seconds=expected_ttl_seconds)
        <= delegation.expires_at
        <= datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expected_ttl_seconds)
    )


def test_kubernetes_job_skips_workload_delegation_when_pod_uid_is_missing(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.PENDING
    pod = _kubernetes_pod("pod-uid-123")
    pod.metadata.uid = None
    k8s_job = _kubernetes_job_for_step(test_step_pending_with_auth_context)
    kubernetes_job._batch_v1.read_namespaced_job.return_value = k8s_job
    kubernetes_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(kubernetes_job._workload_delegations, "_register_workload_delegation") as register_delegation,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks", return_value=False),
        patch.object(kubernetes_job, "get_kube_job_events", return_value=[]),
    ):
        update = kubernetes_job.sync(test_step_pending_with_auth_context)

    assert update.status == PlatformJobStatus.ACTIVE
    register_delegation.assert_not_called()


def test_kubernetes_job_skips_workload_delegation_when_job_labels_are_missing(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.PENDING
    k8s_job = _kubernetes_job_for_step(test_step_pending_with_auth_context)
    del k8s_job.metadata.labels[JOB_STEP_ID_LABEL]
    kubernetes_job._batch_v1.read_namespaced_job.return_value = k8s_job

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(kubernetes_job._workload_delegations, "_register_workload_delegation") as register_delegation,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks", return_value=False),
        patch.object(kubernetes_job, "get_kube_job_events", return_value=[]),
    ):
        update = kubernetes_job.sync(test_step_pending_with_auth_context)

    assert update.status == PlatformJobStatus.PENDING
    register_delegation.assert_not_called()


def test_kubernetes_job_workload_delegation_conflict_is_not_recreated(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.PENDING
    pod = _kubernetes_pod("pod-uid-123")
    k8s_job = _kubernetes_job_for_step(test_step_pending_with_auth_context)
    kubernetes_job._batch_v1.read_namespaced_job.return_value = k8s_job
    kubernetes_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(
            kubernetes_job._workload_delegations,
            "_register_workload_delegation",
            side_effect=WorkloadDelegationConflictError("exists"),
        ) as register_delegation,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks", return_value=False),
        patch.object(kubernetes_job, "get_kube_job_events", return_value=[]),
    ):
        first_update = kubernetes_job.sync(test_step_pending_with_auth_context)
        second_update = kubernetes_job.sync(test_step_pending_with_auth_context)

    assert first_update.status == PlatformJobStatus.ACTIVE
    assert second_update.status == PlatformJobStatus.ACTIVE
    register_delegation.assert_called_once()


def test_kubernetes_job_registration_failure_does_not_block_status_and_retries(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.PENDING
    pod = _kubernetes_pod("pod-uid-123")
    k8s_job = _kubernetes_job_for_step(test_step_pending_with_auth_context)
    kubernetes_job._batch_v1.read_namespaced_job.return_value = k8s_job
    kubernetes_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(
            kubernetes_job._workload_delegations,
            "_register_workload_delegation",
            side_effect=RuntimeError("store offline"),
        ) as register_delegation,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks", return_value=False),
        patch.object(kubernetes_job, "get_kube_job_events", return_value=[]),
    ):
        first_update = kubernetes_job.sync(test_step_pending_with_auth_context)
        second_update = kubernetes_job.sync(test_step_pending_with_auth_context)

    assert first_update.status == PlatformJobStatus.ACTIVE
    assert second_update.status == PlatformJobStatus.ACTIVE
    assert register_delegation.call_count == 2
    job_key = f"test-namespace/{name_for_step(test_step_pending_with_auth_context)}"
    assert kubernetes_job._workload_delegations._delegations_by_target_key.get(job_key, set()) == set()


def test_kubernetes_job_revokes_pod_uid_workload_delegation_when_job_finishes(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.ACTIVE
    pod = _kubernetes_pod("pod-uid-123", phase="Succeeded")
    k8s_job = _kubernetes_job_for_step(
        test_step_pending_with_auth_context,
        status=client.V1JobStatus(
            active=None,
            completion_time=datetime.datetime.now(datetime.timezone.utc),
            failed=None,
            succeeded=1,
        ),
    )
    kubernetes_job._batch_v1.read_namespaced_job.return_value = k8s_job
    kubernetes_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])

    expected_name = reference_delegation_name(
        workload_audience="nemo-platform",
        workload_subject="system:serviceaccount:test-namespace:default",
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value="pod-uid-123",
    )

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(kubernetes_job._workload_delegations, "_revoke_workload_delegation") as revoke_delegation,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks", return_value=False),
        patch.object(kubernetes_job, "get_kube_job_events", return_value=[]),
    ):
        update = kubernetes_job.sync(test_step_pending_with_auth_context)

    assert update.status == PlatformJobStatus.COMPLETED
    revoke_delegation.assert_called_once_with(expected_name)


def test_kubernetes_job_revokes_recorded_workload_delegation_when_pods_are_gone(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.ACTIVE
    k8s_job = _kubernetes_job_for_step(
        test_step_pending_with_auth_context,
        status=client.V1JobStatus(
            active=None,
            completion_time=datetime.datetime.now(datetime.timezone.utc),
            failed=None,
            succeeded=1,
        ),
    )
    job_key = f"test-namespace/{name_for_step(test_step_pending_with_auth_context)}"
    kubernetes_job._workload_delegations._delegations_by_target_key[job_key] = {"ref:recorded-delegation"}
    kubernetes_job._batch_v1.read_namespaced_job.return_value = k8s_job
    kubernetes_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[])

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(kubernetes_job._workload_delegations, "_revoke_workload_delegation") as revoke_delegation,
        patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.update_all_tasks", return_value=False),
        patch.object(kubernetes_job, "get_kube_job_events", return_value=[]),
    ):
        update = kubernetes_job.sync(test_step_pending_with_auth_context)

    assert update.status == PlatformJobStatus.COMPLETED
    revoke_delegation.assert_called_once_with("ref:recorded-delegation")
    assert job_key not in kubernetes_job._workload_delegations._delegations_by_target_key


def test_kubernetes_job_revokes_pod_uid_workload_delegation_when_job_deleted(
    kubernetes_job, test_step_pending_with_auth_context, workload_exchange_auth_config
):
    pod = _kubernetes_pod("pod-uid-123", phase="Succeeded")
    k8s_job = _kubernetes_job_for_step(test_step_pending_with_auth_context)
    kubernetes_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])

    expected_name = reference_delegation_name(
        workload_audience="nemo-platform",
        workload_subject="system:serviceaccount:test-namespace:default",
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value="pod-uid-123",
    )

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(kubernetes_job._workload_delegations, "_revoke_workload_delegation") as revoke_delegation,
    ):
        kubernetes_job.terminate_job(k8s_job)

    revoke_delegation.assert_called_once_with(expected_name)
    kubernetes_job._batch_v1.delete_namespaced_job.assert_called_once()


def test_cleanup_steps_with_multi_step_job_only_first_step_complete(kubernetes_job):
    """Test cleanup_steps with a multi-step job where only the first step is complete.

    Simulates a job with 2 steps where:
    - Step 1 job has completed successfully (terminal)
    - Step 2 job is still active (non-terminal)
    - The overall job itself is not terminal (active)

    Persistent storage should NOT be cleaned up.
    """
    kubernetes_job._execution_profile_config.cleanup_completed_jobs_immediately = True
    kubernetes_job._execution_profile_config.storage = MagicMock()
    kubernetes_job._execution_profile_config.storage.pvc_name = "job-storage-pvc"
    kubernetes_job._execution_profile_config.storage.volume_permissions_image = "busybox"

    # Mock check_step_is_terminal: step1 terminal (True), step2 not terminal (False)
    def check_step_side_effect(job, step_name, workspace):
        return step_name == "step1"

    kubernetes_job.check_step_is_terminal = MagicMock(side_effect=check_step_side_effect)

    # Mock persistent storage cleanup eligibility to return False (job has more steps)
    kubernetes_job.check_job_persistent_storage_cleanup_allowed = MagicMock(return_value=False)

    # Create mock Kubernetes job for step 1 that completed successfully
    mock_job_step1_spec = MagicMock()
    mock_job_step1_spec.suspend = False

    mock_job_step1_status = MagicMock()
    mock_job_step1_status.active = None
    mock_job_step1_status.succeeded = 1
    mock_job_step1_status.failed = None
    mock_job_step1_status.completion_time = datetime.datetime.now(datetime.UTC)

    mock_job_step1 = MagicMock()
    mock_job_step1.status = mock_job_step1_status
    mock_job_step1.spec = mock_job_step1_spec
    mock_job_step1.metadata.name = "multi-step-job-step1"
    mock_job_step1.metadata.labels = {
        JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        JOB_WORKSPACE_ID_LABEL: "default",
        JOB_ID_LABEL: "multi-step-job",
        JOB_STEP_NAME_LABEL: "step1",
        JOB_USES_PERSISTENT_STORAGE_LABEL: "true",
    }

    # Create mock Kubernetes job for step 2 that is still active
    mock_job_step2_spec = MagicMock()
    mock_job_step2_spec.suspend = False

    mock_job_step2_status = MagicMock()
    mock_job_step2_status.active = 1
    mock_job_step2_status.succeeded = None
    mock_job_step2_status.failed = None
    mock_job_step2_status.completion_time = None
    mock_job_step2_status.conditions = None

    mock_job_step2 = MagicMock()
    mock_job_step2.status = mock_job_step2_status
    mock_job_step2.spec = mock_job_step2_spec
    mock_job_step2.metadata.name = "multi-step-job-step2"
    mock_job_step2.metadata.labels = {
        JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        JOB_WORKSPACE_ID_LABEL: "default",
        JOB_ID_LABEL: "multi-step-job",
        JOB_STEP_NAME_LABEL: "step2",
        JOB_USES_PERSISTENT_STORAGE_LABEL: "true",
    }

    # Mock list_namespaced_job to return both jobs
    mock_job_list = [mock_job_step1, mock_job_step2]
    mock_jobv1list = MagicMock()
    mock_jobv1list.items = mock_job_list
    kubernetes_job._batch_v1.list_namespaced_job.return_value = mock_jobv1list

    # Mock the cleanup_job_persistent_storage function to track if it's called
    from nmp.core.jobs.controllers.backends.kubernetes import kubernetes_job as k8s_job_module

    with patch.object(k8s_job_module, "cleanup_job_persistent_storage") as mock_cleanup_storage:
        # Run cleanup
        kubernetes_job.cleanup_steps()

        # Cleanup must list jobs using the selector that includes JOB_MANAGED_BY so only jobs-controller jobs are cleaned
        list_call = kubernetes_job._batch_v1.list_namespaced_job.call_args
        label_selector = list_call.kwargs["label_selector"]
        expected_selector = ",".join([f"{k}={v}" for k, v in KUBE_JOB_SELECTOR_LABELS.items()])
        assert label_selector == expected_selector
        assert f"{JOB_MANAGED_BY_LABEL}={JOB_MANAGED_BY_JOBS_CONTROLLER}" in label_selector

        # Verify step 1 job was deleted (it's completed and terminal)
        kubernetes_job._batch_v1.delete_namespaced_job.assert_called_once()
        assert kubernetes_job._batch_v1.delete_namespaced_job.call_args.kwargs["name"] == "multi-step-job-step1"

        # Verify check_step_is_terminal was called for step 1 (and step 2, but we skip step 2)
        assert kubernetes_job.check_step_is_terminal.call_count >= 1
        kubernetes_job.check_step_is_terminal.assert_any_call(
            job="multi-step-job", step_name="step1", workspace="default"
        )

        # Verify persistent storage cleanup eligibility was checked for step 1
        kubernetes_job.check_job_persistent_storage_cleanup_allowed.assert_called_once_with(
            job="multi-step-job", step_name="step1", workspace="default"
        )

        # Verify persistent storage cleanup was NOT called
        # because the job is not terminal yet (step 2 still active)
        mock_cleanup_storage.assert_not_called()


def test_cleanup_steps_proceeds_when_entity_not_found(kubernetes_job):
    """When step/job entities are gone (e.g. workspace deleted) but the backend job is terminal and ours, still clean up.

    check_step_is_terminal returns True when terminal or when the entity is not found (404).
    """
    kubernetes_job._execution_profile_config.cleanup_completed_jobs_immediately = True

    # Simulate entity-not-found: both return True so cleanup proceeds
    kubernetes_job.check_step_is_terminal = MagicMock(return_value=True)

    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False
    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = 1
    mock_job_status.completion_time = datetime.datetime.now(datetime.UTC)
    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    mock_job.metadata.name = "orphan-step-job"
    mock_job.metadata.labels = {
        JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        JOB_WORKSPACE_ID_LABEL: "deleted-workspace",
        JOB_ID_LABEL: "orphan-job",
        JOB_STEP_NAME_LABEL: "step1",
    }

    mock_jobv1list = MagicMock()
    mock_jobv1list.items = [mock_job]
    kubernetes_job._batch_v1.list_namespaced_job.return_value = mock_jobv1list

    kubernetes_job.cleanup_steps()

    # Should still delete the backend job and configmap (entity-not-found is treated as "proceed with cleanup")
    kubernetes_job._batch_v1.delete_namespaced_job.assert_called_once()
    assert kubernetes_job._batch_v1.delete_namespaced_job.call_args.kwargs["name"] == "orphan-step-job"


def test_cleanup_steps_proceeds_when_job_entity_not_found_with_persistent_storage(kubernetes_job):
    """When job entity is not found (404) but backend job is completed and uses persistent storage, still run full cleanup.

    check_job_persistent_storage_cleanup_allowed returns True when cleanup should proceed.
    """
    kubernetes_job._execution_profile_config.cleanup_completed_jobs_immediately = True
    kubernetes_job._execution_profile_config.storage = MagicMock()
    kubernetes_job._execution_profile_config.storage.pvc_name = "job-storage-pvc"
    kubernetes_job._execution_profile_config.storage.volume_permissions_image = "busybox"

    kubernetes_job.check_step_is_terminal = MagicMock(return_value=True)
    kubernetes_job.check_job_persistent_storage_cleanup_allowed = MagicMock(return_value=True)

    mock_job_spec = MagicMock()
    mock_job_spec.suspend = False
    mock_job_status = MagicMock()
    mock_job_status.active = None
    mock_job_status.succeeded = 1
    mock_job_status.completion_time = datetime.datetime.now(datetime.UTC)
    mock_job = MagicMock()
    mock_job.status = mock_job_status
    mock_job.spec = mock_job_spec
    mock_job.metadata.name = "orphan-job-with-storage"
    mock_job.metadata.labels = {
        JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
        JOB_WORKSPACE_ID_LABEL: "deleted-workspace",
        JOB_ID_LABEL: "orphan-job",
        JOB_STEP_NAME_LABEL: "step1",
        JOB_USES_PERSISTENT_STORAGE_LABEL: "true",
    }

    mock_jobv1list = MagicMock()
    mock_jobv1list.items = [mock_job]
    kubernetes_job._batch_v1.list_namespaced_job.return_value = mock_jobv1list

    from nmp.core.jobs.controllers.backends.kubernetes import kubernetes_job as k8s_job_module

    with patch.object(k8s_job_module, "cleanup_job_persistent_storage") as mock_cleanup_storage:
        kubernetes_job.cleanup_steps()

        # Backend job and configmap are deleted
        kubernetes_job._batch_v1.delete_namespaced_job.assert_called_once()
        assert kubernetes_job._batch_v1.delete_namespaced_job.call_args.kwargs["name"] == "orphan-job-with-storage"
        # Persistent storage is also cleaned (job not found is treated as terminal)
        mock_cleanup_storage.assert_called_once()


def test_scheduler_name_applied_to_pod_spec(
    mock_nmp_client,
    kubernetes_client_mock,
    mock_platform_config,
    cpu_execution_provider,
    test_step_pending,
):
    """When scheduler_name is set in the execution profile, it is applied to the pod spec.

    This enables custom schedulers such as KAI Scheduler to be used with Kubernetes Jobs.
    """
    config = KubernetesJobExecutionProfileConfig(
        namespace="test-namespace",
        storage=DEFAULT_STORAGE,
        scheduler_name="kai-scheduler",
        pod_metadata=KubernetesObjectMetadata(
            labels={"kai.scheduler/queue-name": "my-queue"},
        ),
        job_metadata=KubernetesObjectMetadata(
            labels={"kai.scheduler/queue-name": "my-queue"},
        ),
    )
    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        backend = CPUKubernetesJobBackend(mock_nmp_client, config, profile_name="default")
        backend._batch_v1 = kubernetes_client_mock["batch_v1"]
        backend._core_v1 = kubernetes_client_mock["core_v1"]
        backend.schedule(cpu_execution_provider, test_step_pending)

    backend._batch_v1.create_namespaced_job.assert_called_once()
    job_body = backend._batch_v1.create_namespaced_job.call_args.kwargs["body"]
    assert job_body.spec.template.spec.scheduler_name == "kai-scheduler"


def test_scheduler_name_not_set_by_default(
    mock_nmp_client,
    kubernetes_client_mock,
    mock_platform_config,
    cpu_execution_provider,
    test_step_pending,
):
    """When scheduler_name is not configured, the pod spec does not set a custom scheduler.

    This preserves the default Kubernetes scheduler behavior.
    """
    config = KubernetesJobExecutionProfileConfig(
        namespace="test-namespace",
        storage=DEFAULT_STORAGE,
    )
    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        backend = CPUKubernetesJobBackend(mock_nmp_client, config, profile_name="default")
        backend._batch_v1 = kubernetes_client_mock["batch_v1"]
        backend._core_v1 = kubernetes_client_mock["core_v1"]
        backend.schedule(cpu_execution_provider, test_step_pending)

    backend._batch_v1.create_namespaced_job.assert_called_once()
    job_body = backend._batch_v1.create_namespaced_job.call_args.kwargs["body"]
    assert job_body.spec.template.spec.scheduler_name is None
