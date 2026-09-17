# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import SimpleNamespace
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client
from kubernetes.client.rest import ApiException
from nmp.common.auth import KUBERNETES_POD_UID_REFERENCE_NAME, AuthContext, reference_delegation_name
from nmp.common.jobs.constants import (
    EPHEMERAL_TASK_STORAGE_PATH_ENVVAR,
    NEMO_JOB_FILESET_ENVVAR,
    NEMO_JOB_SECRETS_ENVVAR,
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
    JOB_MULTINODE_NETWORKING_ANNOTATION,
    JOB_NUM_NODES_ANNOTATION,
    JOB_STEP_ID_LABEL,
    JOB_STEP_NAME_LABEL,
    JOB_USES_PERSISTENT_STORAGE_LABEL,
    JOB_WORKSPACE_ID_LABEL,
    KUBE_JOB_SELECTOR_LABELS,
)
from nmp.core.jobs.app.providers import ComputeResources, ContainerSpec, DistributedGPUExecutionProvider
from nmp.core.jobs.app.schemas import (
    PlatformJobEnvironmentVariable,
    PlatformJobSecretEnvironmentVariableRef,
    PlatformJobStepSpec,
)
from nmp.core.jobs.controllers.backends.base import (
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    WORKLOAD_IDENTITY_VOLUME_NAME,
)
from nmp.core.jobs.controllers.backends.kubernetes.common import (
    KubernetesJobStorageConfig,
    KubernetesObjectMetadata,
    common_labels_for_step,
    name_for_step,
)
from nmp.core.jobs.controllers.backends.kubernetes.volcano_job import (
    VolcanoJobBackend,
    VolcanoJobExecutionProfileConfig,
)
from nmp.core.jobs.controllers.backends.workload_tokens import WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS
from pydantic import ValidationError

DEFAULT_STORAGE = KubernetesJobStorageConfig(pvc_name="job-storage-pvc")
DEFAULT_JOB_METADATA = KubernetesObjectMetadata(
    labels={"owner": "alpha"}, annotations={"example.com/annotation": "bravo"}
)
DEFAULT_POD_METADATA = KubernetesObjectMetadata(labels={"foo": "bar"}, annotations={"example.com/annotation": "value"})


@pytest.fixture
def kubernetes_client_mock():
    """Mock Kubernetes client for testing."""

    core_v1_mock = MagicMock()
    custom_v1_mock = MagicMock()
    with (
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.volcano_job.client.CoreV1Api",
            return_value=core_v1_mock,
        ),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.volcano_job.client.CustomObjectsApi",
            return_value=custom_v1_mock,
        ),
    ):
        yield {"core_v1": core_v1_mock, "custom_v1": custom_v1_mock}


@pytest.fixture
def volcano_execution_profile_config():
    """Create a test Volcano execution profile."""
    return VolcanoJobExecutionProfileConfig(
        queue="default",  # default, but want explicitly for visibility
        scheduler_name="volcano",  # default, but want explicitly for visibility
        namespace="test-namespace",
        plugins={"pytorch": ["--master=leader", "--worker=worker", "--port=23456"]},
        max_retry=0,
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
        # image_pull_secrets,
        num_gpus=2,  # TODO test permutations of this vs distributed_gpu_execution_provider
    )


@pytest.fixture
def distributed_gpu_execution_provider():
    """Create a test Distributed GPU execution provider."""
    return DistributedGPUExecutionProvider(
        container=ContainerSpec(
            image="nvidia/cuda:11.8-runtime-ubuntu20.04",
            command=["python", "-c", "print('Hello World')"],
        ),
        resources=ComputeResources(
            num_nodes=3,
            num_gpus=5,
        ),
    )


@pytest.fixture
def volcano_job(
    mock_nmp_client,
    kubernetes_client_mock,
    volcano_execution_profile_config,
    mock_platform_config,
) -> Generator[VolcanoJobBackend, None, None]:
    """Create a namespaced custom object (Volcano Job) instance with mocked clients."""
    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        # Convert the Pydantic model to dict format expected by the base class
        volcano_job = VolcanoJobBackend(
            nmp_sdk=mock_nmp_client,
            execution_profile_config=volcano_execution_profile_config,
            profile_name="default",
        )
        volcano_job._core_v1 = kubernetes_client_mock["core_v1"]
        volcano_job._custom_v1 = kubernetes_client_mock["custom_v1"]
        yield volcano_job


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
def test_step_pending_with_auth_context(test_step_pending: PlatformJobStepWithContext) -> PlatformJobStepWithContext:
    step = test_step_pending.model_copy(deep=True)
    step.auth_context = AuthContext(
        principal_id="creator@example.com",
        principal_email="creator@example.com",
        principal_groups=["engineering", "ml-team"],
    )
    return step


@pytest.fixture
def test_step_active_with_auth_context(test_step_active: PlatformJobStepWithContext) -> PlatformJobStepWithContext:
    step = test_step_active.model_copy(deep=True)
    step.auth_context = AuthContext(
        principal_id="creator@example.com",
        principal_email="creator@example.com",
        principal_groups=["engineering", "ml-team"],
    )
    return step


def _volcano_job_for_step(
    step: PlatformJobStepWithContext,
    *,
    phase: str = "Running",
    service_account_name: str = "default",
) -> dict:
    labels = common_labels_for_step(step)
    labels[JOB_EXECUTION_BACKEND_LABEL] = "volcano_job"
    labels[JOB_EXECUTION_PROFILE_LABEL] = "default"
    return {
        "metadata": {
            "name": name_for_step(step),
            "namespace": "test-namespace",
            "labels": labels,
        },
        "spec": {
            "tasks": [
                {
                    "name": "worker",
                    "template": {
                        "spec": {
                            "serviceAccountName": service_account_name,
                        }
                    },
                }
            ]
        },
        "status": {
            "state": {
                "phase": phase,
                "lastTransitionTime": datetime.datetime.now(datetime.UTC).isoformat(),
            }
        },
    }


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
                        running=client.V1ContainerStateRunning(started_at=datetime.datetime.now(datetime.UTC))
                        if phase == "Running"
                        else None,
                        terminated=client.V1ContainerStateTerminated(exit_code=0, reason="Completed")
                        if phase == "Succeeded"
                        else None,
                    ),
                )
            ],
        ),
    )


# test_kubernetes_backend has tests for build_affinity, build_metadata, build_tolerations. These have been refactored to
# jobs.controller.backends.kubernetes.common.  As common code, not testing a second time here.


def test_schedule_job_success(
    volcano_job: VolcanoJobBackend, distributed_gpu_execution_provider, test_step_pending: PlatformJobStepWithContext
):
    """Test successful job scheduling."""
    # Mock successful job creation
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]

    # Schedule the job
    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Verify job creation was called
    volcano_job._custom_v1.create_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]

    # Check namespace
    assert call_args.kwargs["namespace"] == "test-namespace"

    # Check job object
    job_body = call_args.kwargs["body"]
    assert job_body["apiVersion"] == "batch.volcano.sh/v1alpha1"
    assert job_body["kind"] == "Job"

    # Check metadata
    assert "test-step-id" in job_body["metadata"]["name"]
    # Labels required by cleanup_steps to identify and process the job
    assert job_body["metadata"]["labels"][JOB_ID_LABEL] == "test-job-id"
    assert job_body["metadata"]["labels"][JOB_STEP_NAME_LABEL] == "test-step"
    assert job_body["metadata"]["labels"][JOB_WORKSPACE_ID_LABEL] == "default"
    assert job_body["metadata"]["labels"][JOB_STEP_ID_LABEL] == "test-step-id"
    assert job_body["metadata"]["labels"]["app"] == "nemo-job"
    assert (
        job_body["metadata"]["labels"][JOB_USES_PERSISTENT_STORAGE_LABEL] == "true"
    )  # test_step_pending has persistent storage
    assert job_body["metadata"]["labels"][JOB_MANAGED_BY_LABEL] == JOB_MANAGED_BY_JOBS_CONTROLLER
    assert job_body["metadata"]["labels"][JOB_EXECUTION_BACKEND_LABEL] == "volcano_job"
    assert job_body["metadata"]["labels"][JOB_EXECUTION_PROFILE_LABEL] == "default"

    assert job_body["metadata"]["labels"]["owner"] == "alpha"  # From execution profile
    assert job_body["metadata"]["annotations"]["example.com/annotation"] == "bravo"  # From execution profile

    # Check job spec
    assert job_body["spec"]["maxRetry"] == 0
    assert job_body["spec"]["plugins"] == {"pytorch": ["--master=leader", "--worker=worker", "--port=23456"]}
    assert volcano_job._execution_profile_config.ttl_seconds_after_finished == 3600
    assert "ttl_seconds_after_finished" not in job_body["spec"].keys()

    # Check task templates
    for task in job_body["spec"]["tasks"]:
        # Sniff out leader/worker tasks and check that their specific fields look good
        if task["replicas"] == 1:
            task["name"] == "leader"
            assert task["policies"] == [
                {
                    "event": "TaskCompleted",
                    "action": "CompleteJob",
                }
            ]
        elif task["replicas"] == 2:
            task["name"] == "worker"
            assert "policies" not in task
        else:
            assert False, f"Found unexpected number of replicas {task.get('replicas')}"

        pod_template = task["template"]
        pod_metadata = pod_template["metadata"]
        pod_spec = pod_template["spec"]

        # Check pod metadata
        assert pod_metadata is not None
        assert pod_metadata["labels"] is not None
        assert pod_metadata["labels"][JOB_ID_LABEL] == "test-job-id"
        assert pod_metadata["labels"][JOB_STEP_NAME_LABEL] == "test-step"
        assert pod_metadata["labels"][JOB_WORKSPACE_ID_LABEL] == "default"
        assert pod_metadata["labels"]["app"] == "nemo-job"
        assert pod_metadata["labels"][JOB_USES_PERSISTENT_STORAGE_LABEL] == "true"
        assert pod_metadata["labels"][JOB_MANAGED_BY_LABEL] == JOB_MANAGED_BY_JOBS_CONTROLLER
        assert pod_metadata["labels"]["foo"] == "bar"  # From execution profile
        assert pod_metadata["labels"][JOB_EXECUTION_BACKEND_LABEL] == "volcano_job"
        assert pod_metadata["labels"][JOB_EXECUTION_PROFILE_LABEL] == "default"
        assert pod_metadata["annotations"] is not None
        assert pod_metadata["annotations"]["example.com/annotation"] == "value"  # From execution

        # Check basic pod settings
        assert pod_spec["restartPolicy"] == "Never"
        assert pod_spec["activeDeadlineSeconds"] == 86400
        assert pod_spec["serviceAccountName"] == "default"

        # Check tolerations
        assert pod_spec["tolerations"] is not None
        assert len(pod_spec["tolerations"]) == 2
        assert pod_spec["tolerations"][0]["key"] == "nvidia.com/gpu"

        # Check node selector
        assert pod_spec["nodeSelector"] is not None
        assert pod_spec["nodeSelector"]["accelerator"] == "nvidia-tesla-v100"
        assert pod_spec["nodeSelector"]["node-type"] == "gpu"

        # Check affinity
        assert pod_spec["affinity"] is not None
        assert pod_spec["affinity"]["nodeAffinity"] is not None

        # Check containers
        assert len(pod_spec["containers"]) == 1
        main_container = pod_spec["containers"][0]
        assert main_container["name"] == "nemo-job-task"
        assert main_container["image"] == "nvidia/cuda:11.8-runtime-ubuntu20.04"

        # Check environment variables
        env_vars = {env["name"]: env["value"] for env in main_container["env"]}
        assert "ENV_VAR" in env_vars
        assert env_vars["ENV_VAR"] == "test_value"
        assert env_vars[NEMO_JOB_FILESET_ENVVAR] == "test-logs-fileset"
        assert env_vars[PERSISTENT_JOB_STORAGE_PATH_ENVVAR] == "/var/test"
        assert env_vars[EPHEMERAL_TASK_STORAGE_PATH_ENVVAR] == "/var/tmp"
        assert "NMP_BASE_URL" in env_vars

        # Ensure that config warnings are disabled
        assert env_vars["NMP_CONFIG_WARNINGS_DISABLED"] == "1"


def test_schedule_job_with_auth_context_mounts_projected_workload_identity_token(
    volcano_job: VolcanoJobBackend,
    distributed_gpu_execution_provider,
    test_step_pending_with_auth_context: PlatformJobStepWithContext,
    workload_exchange_auth_config,
):
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]
    volcano_job._execution_profile_config.workload_identity.token_expiration_seconds = 600
    volcano_job._execution_profile_config.workload_identity.token_audience = "test-audience"

    with patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config):
        volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending_with_auth_context)

    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]
    job_body = call_args.kwargs["body"]

    for task in job_body["spec"]["tasks"]:
        pod_spec = task["template"]["spec"]
        main_container = pod_spec["containers"][0]
        env_vars = {env["name"]: env.get("value") for env in main_container["env"]}
        assert env_vars[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR] == WORKLOAD_IDENTITY_TOKEN_FILE_PATH

        volume_mounts = {mount["name"]: mount for mount in main_container["volumeMounts"]}
        assert WORKLOAD_IDENTITY_VOLUME_NAME in volume_mounts

        volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
        projection = volumes[WORKLOAD_IDENTITY_VOLUME_NAME]["projected"]["sources"][0]["serviceAccountToken"]
        assert projection["audience"] == "test-audience"
        assert projection["expirationSeconds"] == 600


def test_created_step_does_not_ttl_before_backend_acceptance(
    volcano_job: VolcanoJobBackend,
    distributed_gpu_execution_provider,
    test_step_pending: PlatformJobStepWithContext,
):
    """CREATED age should not fail a step before the Volcano backend accepts it."""
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]
    ttl_seconds = volcano_job._execution_profile_config.ttl_seconds_before_active
    old_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=ttl_seconds + 300)
    test_step_pending.created_at = old_timestamp
    test_step_pending.updated_at = old_timestamp
    test_step_pending.status = PlatformJobStatus.CREATED

    update = volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    assert update.status == PlatformJobStatus.PENDING
    volcano_job._custom_v1.create_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]


def test_volcano_job_profile_environment_applied(
    kubernetes_client_mock,
    mock_nmp_client,
    volcano_execution_profile_config,
    mock_platform_config,
    distributed_gpu_execution_provider,
    test_step_pending: PlatformJobStepWithContext,
):
    """Profile environment (e.g. HOME=/tmp) is applied to scheduled Volcano job pod containers."""
    profile_config = VolcanoJobExecutionProfileConfig(
        **{**volcano_execution_profile_config.model_dump(), "env": {"HOME": "/tmp"}}
    )
    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        backend = VolcanoJobBackend(
            nmp_sdk=mock_nmp_client,
            execution_profile_config=profile_config,
            profile_name="default",
        )
        backend._core_v1 = kubernetes_client_mock["core_v1"]
        backend._custom_v1 = kubernetes_client_mock["custom_v1"]

    backend._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[assignment]
    backend.schedule(distributed_gpu_execution_provider, test_step_pending)

    call_args = backend._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[union-attr]
    job_body = call_args.kwargs["body"]
    main_container = job_body["spec"]["tasks"][0]["template"]["spec"]["containers"][0]
    env_vars = {env["name"]: env["value"] for env in main_container["env"]}
    assert env_vars.get("HOME") == "/tmp"
    assert env_vars.get("ENV_VAR") == "test_value"


def test_volcano_job_execution_profile_config_rejects_reserved_env_vars():
    """VolcanoJobExecutionProfileConfig raises when environment contains reserved names."""
    with pytest.raises(ValidationError) as exc_info:
        VolcanoJobExecutionProfileConfig(
            namespace="test",
            storage=DEFAULT_STORAGE,
            env={"NEMO_JOB_ID": "x"},
        )
    assert "NEMO_JOB_ID" in str(exc_info.value)
    assert "reserved" in str(exc_info.value).lower()


def test_schedule_job_single_node_success(
    volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext
):
    """Test successful job scheduling."""
    # Mock successful job creation
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]

    # Tweak the distributed_gpu_execution_provider for this one
    distributed_gpu_execution_provider = DistributedGPUExecutionProvider(
        container=ContainerSpec(
            image="nvidia/cuda:11.8-runtime-ubuntu20.04",
            command=["python", "-c", "print('Hello World')"],
        ),
        resources=ComputeResources(
            num_nodes=1,
            num_gpus=1,
        ),
    )

    # Schedule the job
    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Verify job creation was called
    volcano_job._custom_v1.create_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]

    # Check namespace
    assert call_args.kwargs["namespace"] == "test-namespace"

    # Check job object
    job_body = call_args.kwargs["body"]
    assert job_body["apiVersion"] == "batch.volcano.sh/v1alpha1"
    assert job_body["kind"] == "Job"

    # Check metadata
    assert "test-step-id" in job_body["metadata"]["name"]
    # Labels required by cleanup_steps to identify and process the job
    assert job_body["metadata"]["labels"][JOB_ID_LABEL] == "test-job-id"
    assert job_body["metadata"]["labels"][JOB_STEP_NAME_LABEL] == "test-step"
    assert job_body["metadata"]["labels"][JOB_WORKSPACE_ID_LABEL] == "default"
    assert job_body["metadata"]["labels"][JOB_STEP_ID_LABEL] == "test-step-id"
    assert job_body["metadata"]["labels"]["app"] == "nemo-job"
    assert job_body["metadata"]["labels"][JOB_USES_PERSISTENT_STORAGE_LABEL] == "true"
    assert job_body["metadata"]["labels"][JOB_MANAGED_BY_LABEL] == JOB_MANAGED_BY_JOBS_CONTROLLER
    assert job_body["metadata"]["labels"][JOB_EXECUTION_BACKEND_LABEL] == "volcano_job"
    assert job_body["metadata"]["labels"][JOB_EXECUTION_PROFILE_LABEL] == "default"

    assert job_body["metadata"]["labels"]["owner"] == "alpha"  # From execution profile
    assert job_body["metadata"]["annotations"]["example.com/annotation"] == "bravo"  # From execution profile

    # Check job spec
    assert job_body["spec"]["maxRetry"] == 0

    assert job_body["spec"]["plugins"] == {"pytorch": ["--master=leader", "--worker=worker", "--port=23456"]}

    # Check task templates
    assert len(job_body["spec"]["tasks"]) == 1

    for task in job_body["spec"]["tasks"]:
        assert task["name"] == "worker"
        assert task["policies"] == [
            {
                "event": "TaskCompleted",
                "action": "CompleteJob",
            }
        ]
        assert task["replicas"] == 1

        pod_template = task["template"]
        pod_metadata = pod_template["metadata"]
        pod_spec = pod_template["spec"]

        # Check pod metadata
        assert pod_metadata is not None
        assert pod_metadata["labels"] is not None
        assert pod_metadata["labels"][JOB_ID_LABEL] == "test-job-id"
        assert pod_metadata["labels"][JOB_STEP_NAME_LABEL] == "test-step"
        assert pod_metadata["labels"][JOB_WORKSPACE_ID_LABEL] == "default"
        assert pod_metadata["labels"]["app"] == "nemo-job"
        assert pod_metadata["labels"][JOB_USES_PERSISTENT_STORAGE_LABEL] == "true"
        assert pod_metadata["labels"][JOB_MANAGED_BY_LABEL] == JOB_MANAGED_BY_JOBS_CONTROLLER
        assert pod_metadata["labels"]["foo"] == "bar"  # From execution profile
        assert pod_metadata["labels"][JOB_EXECUTION_BACKEND_LABEL] == "volcano_job"
        assert pod_metadata["labels"][JOB_EXECUTION_PROFILE_LABEL] == "default"
        assert pod_metadata["annotations"] is not None
        assert pod_metadata["annotations"]["example.com/annotation"] == "value"  # From execution

        # Check basic pod settings
        assert pod_spec["restartPolicy"] == "Never"
        assert pod_spec["activeDeadlineSeconds"] == 86400
        assert pod_spec["serviceAccountName"] == "default"

        # Check tolerations
        assert pod_spec["tolerations"] is not None
        assert len(pod_spec["tolerations"]) == 2
        assert pod_spec["tolerations"][0]["key"] == "nvidia.com/gpu"

        # Check node selector
        assert pod_spec["nodeSelector"] is not None
        assert pod_spec["nodeSelector"]["accelerator"] == "nvidia-tesla-v100"
        assert pod_spec["nodeSelector"]["node-type"] == "gpu"

        # Check affinity
        assert pod_spec["affinity"] is not None
        assert pod_spec["affinity"]["nodeAffinity"] is not None

        # Check containers
        assert len(pod_spec["containers"]) == 1
        main_container = pod_spec["containers"][0]
        assert main_container["name"] == "nemo-job-task"
        assert main_container["image"] == "nvidia/cuda:11.8-runtime-ubuntu20.04"

        # Check environment variables
        env_vars = {env["name"]: env["value"] for env in main_container["env"]}
        assert "ENV_VAR" in env_vars
        assert env_vars["ENV_VAR"] == "test_value"
        assert env_vars[NEMO_JOB_FILESET_ENVVAR] == "test-logs-fileset"
        assert env_vars[PERSISTENT_JOB_STORAGE_PATH_ENVVAR] == "/var/test"
        assert env_vars[EPHEMERAL_TASK_STORAGE_PATH_ENVVAR] == "/var/tmp"
        assert "NMP_BASE_URL" in env_vars

        # Ensure that config warnings are disabled
        assert env_vars["NMP_CONFIG_WARNINGS_DISABLED"] == "1"


def test_volcano_job_nemo_job_secrets_format_same_and_cross_workspace(
    volcano_job: VolcanoJobBackend, distributed_gpu_execution_provider
):
    """NEMO_JOB_SECRETS is correctly formatted for same-workspace and cross-workspace secret refs.

    Jobs can reference secrets from other workspaces when the user has permissions.
    Format must be ENV_VAR=workspace/secret_name; cross-workspace refs use explicit workspace/secret_name.
    """
    step_with_secrets = PlatformJobStepWithContext(
        id="test-step-id",
        job="test-job-id",
        workspace="default",
        attempt_id="test-job-attempt-id",
        name="test-step",
        fileset="test-logs-fileset",
        step_spec=PlatformJobStepSpec(
            name="test-step",
            executor=DistributedGPUExecutionProvider(
                provider="gpu_distributed",
                profile="default",
                container=ContainerSpec(image="test-image"),
            ),
            config={},
            environment=[
                PlatformJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/var/test"),
                PlatformJobEnvironmentVariable(name=EPHEMERAL_TASK_STORAGE_PATH_ENVVAR, value="/var/tmp"),
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

    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]
    volcano_job.schedule(distributed_gpu_execution_provider, step_with_secrets)

    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]
    job_body = call_args.kwargs["body"]
    main_container = job_body["spec"]["tasks"][0]["template"]["spec"]["containers"][0]
    env_vars = {env["name"]: env.get("value") for env in main_container["env"]}

    nemo_secrets = env_vars.get(NEMO_JOB_SECRETS_ENVVAR, "")
    assert "LOCAL_SECRET=default/local-secret" in nemo_secrets
    assert "CROSS_WORKSPACE_SECRET=other-ws/shared-secret" in nemo_secrets
    parts = [p.strip() for p in nemo_secrets.split(",")]
    assert len(parts) == 2
    assert set(parts) == {
        "LOCAL_SECRET=default/local-secret",
        "CROSS_WORKSPACE_SECRET=other-ws/shared-secret",
    }


def test_schedule_injects_opensandbox_secret_env_when_cluster_capable(
    volcano_job: VolcanoJobBackend,
    distributed_gpu_execution_provider,
    test_step_pending: PlatformJobStepWithContext,
    mock_platform_config,
):
    mock_platform_config.sandbox_cluster_capable = True
    mock_platform_config.sandbox_server_domain = "opensandbox-server.opensandbox-system.svc.cluster.local"
    mock_platform_config.sandbox_api_key_secret = "opensandbox-server-api-key"
    mock_platform_config.sandbox_api_key_secret_key = "api-key"

    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]
    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    job_body = volcano_job._custom_v1.create_namespaced_custom_object.call_args.kwargs["body"]  # ty: ignore[possibly-unbound-attribute]
    for task in job_body["spec"]["tasks"]:
        env_by_name = {env["name"]: env for env in task["template"]["spec"]["containers"][0]["env"]}
        assert env_by_name["OPEN_SANDBOX_DOMAIN"]["value"] == (
            "opensandbox-server.opensandbox-system.svc.cluster.local"
        )
        secret_ref = env_by_name["OPEN_SANDBOX_API_KEY"]["valueFrom"]["secretKeyRef"]
        assert secret_ref["name"] == "opensandbox-server-api-key"
        assert secret_ref["key"] == "api-key"


def test_schedule_job_with_args(
    volcano_job: VolcanoJobBackend, distributed_gpu_execution_provider, test_step_pending: PlatformJobStepWithContext
):
    """Test job scheduling with custom args."""

    # Mock successful job creation
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]

    # Schedule the job
    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Get the created job
    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]
    job_body = call_args.kwargs["body"]
    main_container = job_body["spec"]["tasks"][0]["template"]["spec"]["containers"][0]

    # Check args was set
    assert main_container["args"] == ["python", "-c", "print('Hello World')"]


def test_schedule_job_api_exception(
    volcano_job: VolcanoJobBackend, distributed_gpu_execution_provider, test_step_pending
):
    """Test job scheduling with API exception."""

    # Mock API exception
    volcano_job._custom_v1.create_namespaced_custom_object.side_effect = ApiException("Test error")  # ty: ignore[invalid-assignment]

    # Scheduling should raise the exception
    with pytest.raises(ApiException):
        volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)


def test_multi_node_networking_annotations_added(
    volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext
):
    """Test that enable-multi-node-networking annotations are added for multi-node jobs."""
    # Mock successful job creation
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]

    # Create multi-node job (num_nodes > 1)
    distributed_gpu_execution_provider = DistributedGPUExecutionProvider(
        container=ContainerSpec(
            image="nvidia/cuda:11.8-runtime-ubuntu20.04",
            command=["python", "-c", "print('Hello World')"],
        ),
        resources=ComputeResources(
            num_nodes=2,
            num_gpus=4,
        ),
    )

    # Schedule the job
    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Verify job creation was called
    volcano_job._custom_v1.create_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]

    # Check job object
    job_body = call_args.kwargs["body"]

    # Check that all tasks have the multi-node networking annotations
    for task in job_body["spec"]["tasks"]:
        pod_template = task["template"]
        pod_metadata = pod_template["metadata"]

        # Verify pod template has annotations
        assert pod_metadata["annotations"] is not None
        assert pod_metadata["annotations"][JOB_MULTINODE_NETWORKING_ANNOTATION] == "true"
        assert pod_metadata["annotations"][JOB_NUM_NODES_ANNOTATION] == "2"


def test_single_node_no_networking_annotations(
    volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext
):
    """Test that networking annotations are NOT added for single-node jobs."""
    # Mock successful job creation
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]

    # Create single-node job (num_nodes = 1)
    distributed_gpu_execution_provider = DistributedGPUExecutionProvider(
        container=ContainerSpec(
            image="nvidia/cuda:11.8-runtime-ubuntu20.04",
            command=["python", "-c", "print('Hello World')"],
        ),
        resources=ComputeResources(
            num_nodes=1,
            num_gpus=1,
        ),
    )

    # Schedule the job
    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Verify job creation was called
    volcano_job._custom_v1.create_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]

    # Check job object
    job_body = call_args.kwargs["body"]

    # Check that tasks do NOT have the multi-node networking annotations
    for task in job_body["spec"]["tasks"]:
        pod_template = task["template"]
        pod_metadata = pod_template["metadata"]

        # Verify pod template does NOT have networking annotations
        assert JOB_MULTINODE_NETWORKING_ANNOTATION not in pod_metadata["annotations"]
        assert JOB_NUM_NODES_ANNOTATION not in pod_metadata["annotations"]


def test_networking_annotations_disabled_via_config(
    kubernetes_client_mock,
    mock_nmp_client,
    mock_platform_config,
    test_step_pending: PlatformJobStepWithContext,
):
    """Test that annotations are not added when enable_multi_node_networking=False."""
    # Create execution profile with enable_multi_node_networking disabled
    volcano_execution_profile_config = VolcanoJobExecutionProfileConfig(
        queue="default",
        scheduler_name="volcano",
        namespace="test-namespace",
        enable_multi_node_networking=False,  # Disable networking
        storage=DEFAULT_STORAGE,
        num_gpus=2,
    )

    with (
        patch("nmp.core.jobs.controllers.backends.kubernetes.common.config.load_incluster_config"),
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.common.get_platform_config",
            return_value=mock_platform_config,
        ),
    ):
        volcano_job = VolcanoJobBackend(
            nmp_sdk=mock_nmp_client,
            execution_profile_config=volcano_execution_profile_config,
            profile_name="default",
        )
        volcano_job._core_v1 = kubernetes_client_mock["core_v1"]
        volcano_job._custom_v1 = kubernetes_client_mock["custom_v1"]

    # Mock successful job creation
    volcano_job._custom_v1.create_namespaced_custom_object.return_value = MagicMock()  # ty: ignore[invalid-assignment]

    # Create multi-node job (num_nodes > 1)
    distributed_gpu_execution_provider = DistributedGPUExecutionProvider(
        container=ContainerSpec(
            image="nvidia/cuda:11.8-runtime-ubuntu20.04",
            command=["python", "-c", "print('Hello World')"],
        ),
        resources=ComputeResources(
            num_nodes=2,
            num_gpus=4,
        ),
    )

    # Schedule the job
    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Verify job creation was called
    volcano_job._custom_v1.create_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.create_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]

    # Check job object
    job_body = call_args.kwargs["body"]

    # Check that tasks do NOT have the multi-node networking annotations (even though multi-node)
    for task in job_body["spec"]["tasks"]:
        pod_template = task["template"]
        pod_metadata = pod_template["metadata"]

        # Verify NO networking annotations
        assert JOB_MULTINODE_NETWORKING_ANNOTATION not in pod_metadata["annotations"]
        assert JOB_NUM_NODES_ANNOTATION not in pod_metadata["annotations"]


def test_schedule_job_volcano_not_installed(
    volcano_job: VolcanoJobBackend, distributed_gpu_execution_provider, test_step_pending
):
    """Test job scheduling when Volcano is not installed (404 error)."""

    # Mock API exception with 404 status (Volcano CRD not found)
    volcano_job._custom_v1.create_namespaced_custom_object.side_effect = ApiException(status=404)  # ty: ignore[invalid-assignment]

    # delete_configmap reads first and only deletes if managed by jobs-controller; mock so cleanup proceeds
    volcano_job._core_v1.read_namespaced_config_map.return_value = MagicMock(  # ty: ignore[attr-defined]
        metadata=MagicMock(labels={JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER})
    )

    # Schedule the job - should not raise exception but return error status
    result = volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Verify error status is returned
    assert result.status == PlatformJobStatus.ERROR
    assert result.error_details is not None
    assert "Volcano is not available" in result.error_details["message"]
    assert "contact your platform administrator" in result.error_details["message"]
    assert result.error_details["reason"] == "VolcanoNotInstalled"

    # Verify configmap cleanup was attempted
    volcano_job._core_v1.delete_namespaced_config_map.assert_called_once()  # ty: ignore[possibly-unbound-attribute]


def test_sync_job_active(volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext):
    """Test syncing an active job."""
    # Mock active job status
    mock_job = {"status": {"state": {"phase": "Running"}}, "metadata": {"name": "test-job"}}
    mock_job_list = {"items": [mock_job]}
    volcano_job._custom_v1.get_namespaced_custom_object.return_value = mock_job  # ty: ignore[invalid-assignment]
    volcano_job._custom_v1.list_namespaced_custom_object.return_value = mock_job_list  # ty: ignore[invalid-assignment]

    # Sync the job
    job_update = volcano_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.ACTIVE


def test_sync_job_active_registers_pod_uid_workload_delegation(
    volcano_job: VolcanoJobBackend,
    test_step_pending_with_auth_context: PlatformJobStepWithContext,
    workload_exchange_auth_config,
):
    test_step_pending_with_auth_context.status = PlatformJobStatus.PENDING
    ttl_seconds_active = 900
    volcano_job._execution_profile_config.ttl_seconds_active = ttl_seconds_active
    pod = _kubernetes_pod("pod-uid-123")
    mock_job = _volcano_job_for_step(test_step_pending_with_auth_context, phase="Running")
    volcano_job._custom_v1.get_namespaced_custom_object.return_value = mock_job  # ty: ignore[invalid-assignment]
    volcano_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])  # ty: ignore[attr-defined]
    before_sync = datetime.datetime.now(datetime.UTC)

    expected_name = reference_delegation_name(
        workload_audience="nemo-platform",
        workload_subject="system:serviceaccount:test-namespace:default",
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value="pod-uid-123",
    )

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(volcano_job._workload_delegations, "_register_workload_delegation") as register_delegation,
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.volcano_job.update_all_tasks",
            return_value=False,
        ),
        patch.object(volcano_job, "get_volcano_job_events", return_value=[]),
        patch.object(volcano_job, "get_volcano_pod_group_events", return_value=[]),
    ):
        update = volcano_job.sync(test_step_pending_with_auth_context)

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
        <= datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=expected_ttl_seconds)
    )


def test_sync_job_completed(volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext):
    """Test syncing a completed job."""
    # Mock completed job status
    mock_job = {"status": {"state": {"phase": "Completed"}}, "metadata": {"name": "test-job"}}
    mock_job_list = {"items": [mock_job]}
    volcano_job._custom_v1.get_namespaced_custom_object.return_value = mock_job  # ty: ignore[invalid-assignment]
    volcano_job._custom_v1.list_namespaced_custom_object.return_value = mock_job_list  # ty: ignore[invalid-assignment]

    # Sync the job
    job_update = volcano_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.COMPLETED


def test_sync_job_completed_revokes_pod_uid_workload_delegation(
    volcano_job: VolcanoJobBackend,
    test_step_active_with_auth_context: PlatformJobStepWithContext,
    workload_exchange_auth_config,
):
    test_step_active_with_auth_context.status = PlatformJobStatus.ACTIVE
    pod = _kubernetes_pod("pod-uid-123", phase="Succeeded")
    mock_job = _volcano_job_for_step(test_step_active_with_auth_context, phase="Completed")
    volcano_job._custom_v1.get_namespaced_custom_object.return_value = mock_job  # ty: ignore[invalid-assignment]
    volcano_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])  # ty: ignore[attr-defined]

    expected_name = reference_delegation_name(
        workload_audience="nemo-platform",
        workload_subject="system:serviceaccount:test-namespace:default",
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value="pod-uid-123",
    )

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(volcano_job._workload_delegations, "_revoke_workload_delegation") as revoke_delegation,
        patch(
            "nmp.core.jobs.controllers.backends.kubernetes.volcano_job.update_all_tasks",
            return_value=False,
        ),
        patch.object(volcano_job, "get_volcano_job_events", return_value=[]),
        patch.object(volcano_job, "get_volcano_pod_group_events", return_value=[]),
    ):
        update = volcano_job.sync(test_step_active_with_auth_context)

    assert update.status == PlatformJobStatus.COMPLETED
    revoke_delegation.assert_called_once_with(expected_name)


def test_sync_job_failed(volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext):
    """Test syncing a failed job."""
    # Mock failed job status
    mock_job = {"status": {"state": {"phase": "Failed"}}, "metadata": {"name": "test-job"}}
    mock_job_list = {"items": [mock_job]}
    volcano_job._custom_v1.get_namespaced_custom_object.return_value = mock_job  # ty: ignore[invalid-assignment]
    volcano_job._custom_v1.list_namespaced_custom_object.return_value = mock_job_list  # ty: ignore[invalid-assignment]

    # Sync the job
    job_update = volcano_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.ERROR


def test_sync_job_not_found(volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext):
    """Test syncing a job that doesn't exist."""
    # Mock job not found (get_volcano_job_by_name uses get_namespaced_custom_object)
    volcano_job._custom_v1.get_namespaced_custom_object.side_effect = ApiException(status=404)  # ty: ignore[invalid-assignment]

    # Sync the job
    job_update = volcano_job.sync(test_step_pending)

    assert job_update.status == PlatformJobStatus.PENDING


def test_sync_active_when_volcano_job_not_found(
    volcano_job: VolcanoJobBackend, test_step_active: PlatformJobStepWithContext
):
    """Test syncing an ACTIVE step when the Volcano job is already gone (e.g. deleted).

    Ensures we do not call enforce_sync_ttl with None; we fall through to
    sync_active(step, None) which returns ERROR with 'Job not found'.
    """
    test_step_active.status = PlatformJobStatus.ACTIVE
    volcano_job._custom_v1.get_namespaced_custom_object.side_effect = ApiException(status=404)  # ty: ignore[invalid-assignment]

    job_update = volcano_job.sync(test_step_active)

    assert job_update.status == PlatformJobStatus.ERROR
    assert job_update.error_details is not None
    assert "Job not found" in job_update.error_details.get("message", "")
    # Must not attempt to delete a non-existent job
    volcano_job._custom_v1.delete_namespaced_custom_object.assert_not_called()  # ty: ignore[possibly-unbound-attribute]


def test_sync_active_when_volcano_job_not_found_revokes_recorded_workload_delegation(
    volcano_job: VolcanoJobBackend,
    test_step_active_with_auth_context: PlatformJobStepWithContext,
    workload_exchange_auth_config,
):
    test_step_active_with_auth_context.status = PlatformJobStatus.ACTIVE
    job_key = f"test-namespace/{name_for_step(test_step_active_with_auth_context)}"
    volcano_job._workload_delegations._delegations_by_target_key[job_key] = {"ref:recorded-delegation"}
    volcano_job._custom_v1.get_namespaced_custom_object.side_effect = ApiException(status=404)  # ty: ignore[invalid-assignment]

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(volcano_job._workload_delegations, "_revoke_workload_delegation") as revoke_delegation,
    ):
        job_update = volcano_job.sync(test_step_active_with_auth_context)

    assert job_update.status == PlatformJobStatus.ERROR
    revoke_delegation.assert_called_once_with("ref:recorded-delegation")
    assert job_key not in volcano_job._workload_delegations._delegations_by_target_key


def test_name_for_job_truncation(volcano_job: VolcanoJobBackend):
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
            executor=DistributedGPUExecutionProvider(
                provider="gpu_distributed", profile="volcano_profile", container=ContainerSpec(image="test-image")
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


def test_schedule_with_storage_integration(
    volcano_job: VolcanoJobBackend, distributed_gpu_execution_provider, test_step_pending
):
    """Test that scheduling with storage integrates PVC mount correctly."""
    volcano_job._execution_profile_config.storage.pvc_name = "job-storage-pvc"
    mock_create_custom_object = volcano_job._custom_v1.create_namespaced_custom_object

    volcano_job.schedule(distributed_gpu_execution_provider, test_step_pending)

    # Verify job was created with storage mount
    mock_create_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = mock_create_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]
    job_body = call_args.kwargs["body"]

    for task in job_body["spec"]["tasks"]:
        # Check volumes
        volumes = task["template"]["spec"]["volumes"]
        job_storage_volume = next((v for v in volumes if v["name"] == "job-storage"), None)  # type: ignore[arg-type]
        assert job_storage_volume is not None
        assert job_storage_volume["persistentVolumeClaim"]["claimName"] == "job-storage-pvc"

        # Check volume mounts
        main_container = task["template"]["spec"]["containers"][0]
        job_storage_mount = next((vm for vm in main_container["volumeMounts"] if vm["name"] == "job-storage"), None)
        assert job_storage_mount is not None
        assert job_storage_mount["mountPath"] == "/var/test"
        assert job_storage_mount["subPath"] == f"jobs/{test_step_pending.workspace}/{test_step_pending.job}"

        # Check permissions init container (mounts full volume, creates subpath and chmods in one shot)
        init_containers = task["template"]["spec"]["initContainers"]
        fix_permissions_container = next((c for c in init_containers if c["name"] == "fix-permissions"), None)
        assert fix_permissions_container is not None
        fix_permissions_mount = fix_permissions_container["volumeMounts"][0]
        assert fix_permissions_mount["name"] == "job-storage"
        assert fix_permissions_mount["mountPath"] == "/vol"
        assert fix_permissions_mount.get("subPath") is None  # full volume mount; container creates subpath via mkdir
        expected_subpath = f"jobs/{test_step_pending.workspace}/{test_step_pending.job}"
        fix_permissions_cmd = " ".join(fix_permissions_container.get("command", []) or [])
        assert f"/vol/{expected_subpath}" in fix_permissions_cmd
        assert "mkdir -p" in fix_permissions_cmd
        assert "chmod -R 777" in fix_permissions_cmd


@pytest.mark.parametrize("status", ["Completed", "Failed"])
def test_cleanup_steps_by_ttl(volcano_job: VolcanoJobBackend, status):
    """Test job cleanup with one active, one recently completed, one harvestable completed."""

    # Step cleanup proceeds for terminal steps or when the step entity is gone.
    volcano_job.check_step_is_terminal = MagicMock(return_value=True)  # type: ignore[assignment]

    status_key = status.lower()
    two_hours_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=7200)
    mock_old_completed_job = {
        "status": {status_key: 1, "state": {"lastTransitionTime": two_hours_ago.isoformat(), "phase": status}},
        "metadata": {
            "name": "test-job-old",
            "labels": {
                JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
                JOB_WORKSPACE_ID_LABEL: "default",
                JOB_ID_LABEL: "test-job-id",
                JOB_STEP_NAME_LABEL: "test-step",
            },
        },
    }

    two_minutes_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=120)
    mock_recently_completed_job = {
        "status": {status_key: 1, "state": {"lastTransitionTime": two_minutes_ago.isoformat(), "phase": status}},
        "metadata": {
            "name": "test-job-recent",
            "labels": {
                JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
                JOB_WORKSPACE_ID_LABEL: "default",
                JOB_ID_LABEL: "test-job-id",
                JOB_STEP_NAME_LABEL: "test-step",
            },
        },
    }

    mock_active_job = {
        "status": {"running": 1, "state": {"lastTransitionTime": two_minutes_ago.isoformat(), "phase": "Running"}},
        "metadata": {
            "name": "test-job-active",
            "labels": {
                JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
                JOB_WORKSPACE_ID_LABEL: "default",
                JOB_ID_LABEL: "test-job-id",
                JOB_STEP_NAME_LABEL: "test-step",
            },
        },
    }

    mock_job_list = {"items": [mock_old_completed_job, mock_recently_completed_job, mock_active_job]}
    volcano_job._custom_v1.list_namespaced_custom_object.return_value = mock_job_list  # ty: ignore[invalid-assignment]

    volcano_job.cleanup_steps()

    # Cleanup must list jobs using the selector that includes JOB_MANAGED_BY so only jobs-controller jobs are cleaned
    list_call = volcano_job._custom_v1.list_namespaced_custom_object.call_args  # type: ignore[union-attr]
    label_selector = list_call.kwargs["label_selector"]
    expected_selector = ",".join([f"{k}={v}" for k, v in KUBE_JOB_SELECTOR_LABELS.items()])
    assert label_selector == expected_selector
    assert f"{JOB_MANAGED_BY_LABEL}={JOB_MANAGED_BY_JOBS_CONTROLLER}" in label_selector

    if status == "Completed":
        assert volcano_job._custom_v1.delete_namespaced_custom_object.call_count == 2  # ty: ignore[possibly-unbound-attribute]
    else:
        volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.delete_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]
    assert call_args.kwargs["group"] == "batch.volcano.sh"
    assert call_args.kwargs["version"] == "v1alpha1"
    assert call_args.kwargs["namespace"] == "test-namespace"
    assert call_args.kwargs["plural"] == "jobs"
    assert call_args.kwargs["name"] != "test-job-active"
    assert call_args.kwargs["propagation_policy"] == "Foreground"


def test_cleanup_pending_by_ttl(volcano_job: VolcanoJobBackend, test_step_pending: PlatformJobStepWithContext):
    """Test that sync of a PENDING step transitions to an ERROR state when step's created_at exceeds TTL."""
    # Get the TTL configuration (default is 30 minutes)
    ttl_seconds = volcano_job._execution_profile_config.ttl_seconds_before_active

    # Set the step to PENDING status
    test_step_pending.status = PlatformJobStatus.PENDING

    # Create a step with an created_at timestamp that exceeds the TTL (35 minutes ago)
    old_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=ttl_seconds + 300)
    test_step_pending.created_at = old_timestamp
    test_step_pending.updated_at = old_timestamp

    # Create a mock Volcano Job that the sync method will find (with managed-by label so terminate_job deletes it)
    mock_job = {
        "status": {"state": {"phase": "Pending"}},
        "metadata": {
            "name": "test-job-pending",
            "labels": {JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER},
        },
    }

    # Mock get_volcano_job_by_name to return our job
    volcano_job._custom_v1.get_namespaced_custom_object.return_value = mock_job  # ty: ignore[invalid-assignment]

    # Mock update_all_tasks
    with patch("nmp.core.jobs.controllers.backends.kubernetes.volcano_job.update_all_tasks"):
        # Call sync which should detect the TTL timeout
        result = volcano_job.sync(test_step_pending)

    # Verify that it returns an ERROR status with timeout message
    assert result.status == PlatformJobStatus.ERROR.value
    assert result.status_details["message"] == "Job timed out after reaching max TTL of 1800 seconds"  # type: ignore[index]
    assert result.error_details == {"message": "Job timed out after reaching max TTL of 1800 seconds"}

    # Verify that the job was terminated (delete_namespaced_custom_object was called)
    volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.delete_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]
    assert call_args.kwargs["group"] == "batch.volcano.sh"
    assert call_args.kwargs["version"] == "v1alpha1"
    assert call_args.kwargs["namespace"] == "test-namespace"
    assert call_args.kwargs["plural"] == "jobs"
    assert call_args.kwargs["name"] == "test-job-pending"
    assert call_args.kwargs["propagation_policy"] == "Foreground"


def test_cleanup_active_by_ttl(volcano_job: VolcanoJobBackend, test_step_active):
    """Test that sync of an ACTIVE step transitions to an ERROR state when step's created_at exceeds TTL."""
    # Get the TTL configuration for active jobs (default is 24 hours)
    ttl_seconds = volcano_job._execution_profile_config.ttl_seconds_active

    # Set the step to ACTIVE status
    test_step_active.status = PlatformJobStatus.ACTIVE

    # Create a step with an created_at timestamp that exceeds the TTL (25 hours ago)
    old_timestamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=ttl_seconds + 3600)
    test_step_active.created_at = old_timestamp

    # Create a mock Volcano Job that the sync method will find (with managed-by label so terminate_job deletes it)
    mock_job = {
        "status": {"state": {"phase": "Running"}},
        "metadata": {
            "name": "test-job-active",
            "labels": {JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER},
        },
    }

    # Mock get_volcano_job_by_name to return our job
    volcano_job._custom_v1.get_namespaced_custom_object.return_value = mock_job  # ty: ignore[invalid-assignment]

    # Mock update_all_tasks
    with patch("nmp.core.jobs.controllers.backends.kubernetes.volcano_job.update_all_tasks"):
        # Call sync which should detect the TTL timeout
        result = volcano_job.sync(test_step_active)

    # Verify that it returns an ERROR status with timeout message
    assert result.status == PlatformJobStatus.ERROR.value
    assert result.status_details["message"] == "Job timed out after reaching max TTL of 86400 seconds"  # type: ignore[index]
    assert result.error_details == {"message": "Job timed out after reaching max TTL of 86400 seconds"}

    # Verify that the job was terminated (delete_namespaced_custom_object was called)
    volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.delete_namespaced_custom_object.call_args  # ty: ignore[possibly-unbound-attribute]
    assert call_args.kwargs["group"] == "batch.volcano.sh"
    assert call_args.kwargs["version"] == "v1alpha1"
    assert call_args.kwargs["namespace"] == "test-namespace"
    assert call_args.kwargs["plural"] == "jobs"
    assert call_args.kwargs["name"] == "test-job-active"
    assert call_args.kwargs["propagation_policy"] == "Foreground"


def test_terminate_job_skips_delete_when_not_managed_by_jobs_controller(volcano_job: VolcanoJobBackend):
    """terminate_job must not delete the Volcano job or configmap if the job is not managed by jobs-controller."""
    mock_job = {
        "metadata": {"name": "unmanaged-job", "labels": {}},  # No JOB_MANAGED_BY_LABEL
    }

    volcano_job.terminate_job(mock_job)

    volcano_job._custom_v1.delete_namespaced_custom_object.assert_not_called()  # type: ignore[possibly-unbound-attribute]
    volcano_job._core_v1.delete_namespaced_config_map.assert_not_called()  # type: ignore[possibly-unbound-attribute]


def test_terminate_job_revokes_pod_uid_workload_delegation(
    volcano_job: VolcanoJobBackend,
    test_step_pending_with_auth_context: PlatformJobStepWithContext,
    workload_exchange_auth_config,
):
    pod = _kubernetes_pod("pod-uid-123", phase="Succeeded")
    mock_job = _volcano_job_for_step(test_step_pending_with_auth_context, phase="Running")
    volcano_job._core_v1.list_namespaced_pod.return_value = client.V1PodList(items=[pod])  # ty: ignore[attr-defined]
    volcano_job._core_v1.read_namespaced_config_map.return_value = MagicMock(  # ty: ignore[attr-defined]
        metadata=MagicMock(labels={JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER})
    )

    expected_name = reference_delegation_name(
        workload_audience="nemo-platform",
        workload_subject="system:serviceaccount:test-namespace:default",
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value="pod-uid-123",
    )

    with (
        patch("nmp.common.config.get_auth_config", return_value=workload_exchange_auth_config),
        patch.object(volcano_job._workload_delegations, "_revoke_workload_delegation") as revoke_delegation,
    ):
        volcano_job.terminate_job(mock_job)

    revoke_delegation.assert_called_once_with(expected_name)
    volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]


def test_terminate_job_volcano_not_installed(volcano_job: VolcanoJobBackend):
    """Test job termination when Volcano is not installed (404 error)."""
    # Create a mock job (with managed-by label so delete is attempted)
    mock_job = {
        "metadata": {
            "name": "test-job-terminate",
            "labels": {JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER},
        },
    }

    # Mock API exception with 404 status (Volcano CRD not found or job already deleted)
    volcano_job._custom_v1.delete_namespaced_custom_object.side_effect = ApiException(status=404)  # ty: ignore[invalid-assignment]

    # delete_configmap reads first and only deletes if managed by jobs-controller; mock so cleanup proceeds
    volcano_job._core_v1.read_namespaced_config_map.return_value = MagicMock(  # ty: ignore[attr-defined]
        metadata=MagicMock(labels={JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER})
    )

    # Terminate the job - should not raise exception
    volcano_job.terminate_job(mock_job)

    # Verify delete was attempted
    volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # ty: ignore[possibly-unbound-attribute]

    # Verify configmap cleanup was still called
    volcano_job._core_v1.delete_namespaced_config_map.assert_called_once()  # ty: ignore[possibly-unbound-attribute]


def test_terminate_job_other_api_error(volcano_job: VolcanoJobBackend):
    """Test job termination with non-404 API error should raise exception."""
    # Create a mock job (with managed-by label so delete is attempted)
    mock_job = {
        "metadata": {
            "name": "test-job-terminate",
            "labels": {JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER},
        },
    }

    # Mock API exception with non-404 status (e.g., 403 Forbidden)
    volcano_job._custom_v1.delete_namespaced_custom_object.side_effect = ApiException(status=403)  # ty: ignore[invalid-assignment]

    # Terminate the job - should raise exception
    with pytest.raises(ApiException):
        volcano_job.terminate_job(mock_job)


def test_cleanup_steps_with_multi_step_job_only_first_step_complete(volcano_job: VolcanoJobBackend):
    """Test cleanup_steps with a multi-step job where only the first step is complete.

    Simulates a job with 2 steps where:
    - Step 1 job has completed successfully (terminal)
    - Step 2 job is still active (non-terminal)
    - The overall job itself is not terminal (active)

    Persistent storage should NOT be cleaned up.
    """
    volcano_job._execution_profile_config.cleanup_completed_jobs_immediately = True
    volcano_job._execution_profile_config.storage = MagicMock()
    volcano_job._execution_profile_config.storage.pvc_name = "job-storage-pvc"
    volcano_job._execution_profile_config.storage.volume_permissions_image = "busybox"

    # Mock check_step_is_terminal: step1 terminal (True), step2 not terminal (False)
    def check_step_side_effect(job, step_name, workspace):
        return step_name == "step1"

    volcano_job.check_step_is_terminal = MagicMock(side_effect=check_step_side_effect)  # type: ignore[assignment]

    # Mock persistent storage cleanup eligibility to return False (job has more steps)
    volcano_job.check_job_persistent_storage_cleanup_allowed = MagicMock(return_value=False)  # type: ignore[assignment]

    # Create mock Volcano job for step 1 that completed successfully
    two_minutes_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=120)
    mock_job_step1 = {
        "status": {
            "succeeded": 1,
            "state": {"lastTransitionTime": two_minutes_ago.isoformat(), "phase": "Completed"},
        },
        "metadata": {
            "name": "multi-step-job-step1",
            "labels": {
                JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
                JOB_WORKSPACE_ID_LABEL: "default",
                JOB_ID_LABEL: "multi-step-job",
                JOB_STEP_NAME_LABEL: "step1",
                JOB_USES_PERSISTENT_STORAGE_LABEL: "true",
            },
        },
    }

    # Create mock Volcano job for step 2 that is still active
    mock_job_step2 = {
        "status": {"running": 1, "state": {"lastTransitionTime": two_minutes_ago.isoformat(), "phase": "Running"}},
        "metadata": {
            "name": "multi-step-job-step2",
            "labels": {
                JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
                JOB_WORKSPACE_ID_LABEL: "default",
                JOB_ID_LABEL: "multi-step-job",
                JOB_STEP_NAME_LABEL: "step2",
                JOB_USES_PERSISTENT_STORAGE_LABEL: "true",
            },
        },
    }

    # Mock list_namespaced_custom_object to return both jobs
    mock_job_list = {"items": [mock_job_step1, mock_job_step2]}
    volcano_job._custom_v1.list_namespaced_custom_object.return_value = mock_job_list  # type: ignore[invalid-assignment]

    # Mock the cleanup_job_persistent_storage function to track if it's called
    from nmp.core.jobs.controllers.backends.kubernetes import common as k8s_common

    with patch.object(k8s_common, "cleanup_job_persistent_storage") as mock_cleanup_storage:
        # Run cleanup
        volcano_job.cleanup_steps()

        # Cleanup must list jobs using the selector that includes JOB_MANAGED_BY so only jobs-controller jobs are cleaned
        list_call = volcano_job._custom_v1.list_namespaced_custom_object.call_args  # type: ignore[union-attr]
        label_selector = list_call.kwargs["label_selector"]
        expected_selector = ",".join([f"{k}={v}" for k, v in KUBE_JOB_SELECTOR_LABELS.items()])
        assert label_selector == expected_selector
        assert f"{JOB_MANAGED_BY_LABEL}={JOB_MANAGED_BY_JOBS_CONTROLLER}" in label_selector

        # Verify step 1 job was deleted (it's completed and terminal)
        volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # type: ignore[possibly-unbound-attribute]
        call_args = volcano_job._custom_v1.delete_namespaced_custom_object.call_args  # type: ignore[possibly-unbound-attribute]
        assert call_args.kwargs["name"] == "multi-step-job-step1"

        # Verify check_step_is_terminal was called for both steps (loop checks all jobs)
        assert volcano_job.check_step_is_terminal.call_count == 2
        volcano_job.check_step_is_terminal.assert_any_call(job="multi-step-job", step_name="step1", workspace="default")
        volcano_job.check_step_is_terminal.assert_any_call(job="multi-step-job", step_name="step2", workspace="default")

        # Verify persistent storage cleanup eligibility was checked once for step 1
        volcano_job.check_job_persistent_storage_cleanup_allowed.assert_called_once_with(
            job="multi-step-job", step_name="step1", workspace="default"
        )

        # Verify persistent storage cleanup was NOT called
        # because the job is not terminal yet (step 2 still active)
        mock_cleanup_storage.assert_not_called()


def test_cleanup_steps_proceeds_when_entity_not_found(volcano_job: VolcanoJobBackend):
    """When step/job entities are gone (e.g. workspace deleted) but the backend job is terminal and ours, still clean up.

    check_step_is_terminal returns True when terminal or when the entity is not found (404).
    """
    volcano_job._execution_profile_config.cleanup_completed_jobs_immediately = True

    # Simulate entity-not-found: both return True so cleanup proceeds
    volcano_job.check_step_is_terminal = MagicMock(return_value=True)  # type: ignore[assignment]

    two_minutes_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=120)
    mock_job = {
        "status": {
            "succeeded": 1,
            "state": {"lastTransitionTime": two_minutes_ago.isoformat(), "phase": "Completed"},
        },
        "metadata": {
            "name": "orphan-step-job",
            "labels": {
                JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
                JOB_WORKSPACE_ID_LABEL: "deleted-workspace",
                JOB_ID_LABEL: "orphan-job",
                JOB_STEP_NAME_LABEL: "step1",
            },
        },
    }

    volcano_job._custom_v1.list_namespaced_custom_object.return_value = {"items": [mock_job]}  # type: ignore[invalid-assignment]

    volcano_job.cleanup_steps()

    # Should still delete the backend job and configmap (entity-not-found is treated as "proceed with cleanup")
    volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # type: ignore[possibly-unbound-attribute]
    call_args = volcano_job._custom_v1.delete_namespaced_custom_object.call_args  # type: ignore[possibly-unbound-attribute]
    assert call_args.kwargs["name"] == "orphan-step-job"


def test_cleanup_steps_proceeds_when_job_entity_not_found_with_persistent_storage(volcano_job: VolcanoJobBackend):
    """When job entity is not found (404) but backend job is completed and uses persistent storage, still run full cleanup.

    check_job_persistent_storage_cleanup_allowed returns True when cleanup should proceed.
    """
    volcano_job._execution_profile_config.cleanup_completed_jobs_immediately = True
    volcano_job._execution_profile_config.storage = MagicMock()
    volcano_job._execution_profile_config.storage.pvc_name = "job-storage-pvc"
    volcano_job._execution_profile_config.storage.volume_permissions_image = "busybox"

    volcano_job.check_step_is_terminal = MagicMock(return_value=True)  # type: ignore[assignment]
    volcano_job.check_job_persistent_storage_cleanup_allowed = MagicMock(return_value=True)  # type: ignore[assignment]

    two_minutes_ago = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=120)
    mock_job = {
        "status": {
            "succeeded": 1,
            "state": {"lastTransitionTime": two_minutes_ago.isoformat(), "phase": "Completed"},
        },
        "metadata": {
            "name": "orphan-job-with-storage",
            "labels": {
                JOB_MANAGED_BY_LABEL: JOB_MANAGED_BY_JOBS_CONTROLLER,
                JOB_WORKSPACE_ID_LABEL: "deleted-workspace",
                JOB_ID_LABEL: "orphan-job",
                JOB_STEP_NAME_LABEL: "step1",
                JOB_USES_PERSISTENT_STORAGE_LABEL: "true",
            },
        },
    }

    volcano_job._custom_v1.list_namespaced_custom_object.return_value = {"items": [mock_job]}  # type: ignore[invalid-assignment]

    from nmp.core.jobs.controllers.backends.kubernetes import volcano_job as volcano_job_module

    with patch.object(volcano_job_module, "cleanup_job_persistent_storage") as mock_cleanup_storage:
        volcano_job.cleanup_steps()

        # Backend job and configmap are deleted
        volcano_job._custom_v1.delete_namespaced_custom_object.assert_called_once()  # type: ignore[possibly-unbound-attribute]
        call_args = volcano_job._custom_v1.delete_namespaced_custom_object.call_args  # type: ignore[possibly-unbound-attribute]
        assert call_args.kwargs["name"] == "orphan-job-with-storage"
        # Persistent storage is also cleaned (job not found is treated as terminal)
        mock_cleanup_storage.assert_called_once()
