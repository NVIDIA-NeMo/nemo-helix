# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for map_kubernetes_job_status_to_step_status and aggregate_pod_statuses_for_job_step."""

from unittest.mock import MagicMock, patch

import pytest
from kubernetes import client
from nmp.common.jobs.schemas import PlatformJobStatus
from nmp.core.jobs.api.v2.jobs.schemas import PlatformJobStepWithContext
from nmp.core.jobs.controllers.backends.kubernetes.common import (
    PodStatus,
    aggregate_pod_statuses_for_job_step,
    map_pod_status_to_platform_status,
    map_pod_to_pod_status,
    update_all_tasks,
)
from nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job import map_kubernetes_job_status_to_step_status


def _job(
    *,
    completion_time=None,
    failed=None,
    conditions=None,
    suspend: bool = False,
) -> MagicMock:
    mock_job = MagicMock()
    mock_job.metadata.namespace = "ns"
    mock_job.metadata.name = "jobname"
    mock_job.spec = MagicMock()
    mock_job.spec.suspend = suspend
    mock_job.status = MagicMock()
    mock_job.status.completion_time = completion_time
    mock_job.status.failed = failed
    mock_job.status.conditions = conditions
    mock_job.status.active = None
    mock_job.status.succeeded = None
    return mock_job


def _pod(
    *,
    phase: str,
    errors: dict | None = None,
    active: set | None = None,
    completed: set | None = None,
    waiting: dict | None = None,
) -> PodStatus:
    return PodStatus(
        task_id="task-1",
        name="pod-1",
        errors=errors or {},
        completed=completed or set(),
        active=active or set(),
        waiting=waiting or {},
        phase=phase,
    )


@patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
def test_map_status_empty_pods_returns_pending_waiting_message(
    mock_list_pods: MagicMock, test_step_pending: PlatformJobStepWithContext
) -> None:
    mock_list_pods.return_value = []
    job = _job()
    core_v1 = MagicMock()

    status, details = map_kubernetes_job_status_to_step_status(job, core_v1, test_step_pending)

    assert status == PlatformJobStatus.PENDING
    assert "Waiting for pods" in details["message"]


@patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
def test_map_status_succeeded_pods_without_job_completion_time(
    mock_list_pods: MagicMock, test_step_pending: PlatformJobStepWithContext
) -> None:
    """Pods can report Succeeded before batch Job.completion_time is set."""
    mock_list_pods.return_value = [_pod(phase="Succeeded")]
    job = _job()
    core_v1 = MagicMock()

    status, details = map_kubernetes_job_status_to_step_status(job, core_v1, test_step_pending)

    assert status == PlatformJobStatus.COMPLETED
    assert "completion_time" in details["message"].lower() or "transient" in details["message"].lower()


@patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
def test_map_status_failed_count_without_failed_condition_true(
    mock_list_pods: MagicMock, test_step_pending: PlatformJobStepWithContext
) -> None:
    cond = MagicMock()
    cond.type = "Progressing"
    cond.status = "True"
    cond.message = "ReplicaSet updated"

    mock_list_pods.return_value = []
    job = _job(failed=1, conditions=[cond])
    core_v1 = MagicMock()

    status, details = map_kubernetes_job_status_to_step_status(job, core_v1, test_step_pending)

    assert status == PlatformJobStatus.ERROR
    assert "failure" in details["message"].lower() or "Failed" in details["message"]
    assert "kubernetes_conditions" in details
    assert len(details["kubernetes_conditions"]) == 1


@patch("nmp.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
def test_map_status_unknown_phase_pods_fallback_pending(
    mock_list_pods: MagicMock, test_step_pending: PlatformJobStepWithContext
) -> None:
    """Phase Unknown with no container signals maps to PENDING via aggregate."""
    mock_list_pods.return_value = [_pod(phase="Unknown")]
    job = _job()
    core_v1 = MagicMock()

    status, details = map_kubernetes_job_status_to_step_status(job, core_v1, test_step_pending)

    assert status == PlatformJobStatus.PENDING
    assert "unclear" in details["message"].lower() or "reconciling" in details["message"].lower()


def test_aggregate_pod_statuses_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_pod_statuses_for_job_step([])


def test_aggregate_pod_statuses_error_wins() -> None:
    pods = [
        _pod(phase="Succeeded"),
        _pod(phase="Failed"),
    ]
    status, details = aggregate_pod_statuses_for_job_step(pods)
    assert status == PlatformJobStatus.ERROR
    assert "error" in details["message"].lower()


def test_aggregate_pod_statuses_all_completed() -> None:
    pods = [_pod(phase="Succeeded"), _pod(phase="Succeeded")]
    status, details = aggregate_pod_statuses_for_job_step(pods)
    assert status == PlatformJobStatus.COMPLETED
    assert "transient" in details["message"].lower()


def _waiting_pod(reason: str) -> client.V1Pod:
    return client.V1Pod(
        metadata=client.V1ObjectMeta(name="pod-1", uid="uid-1"),
        status=client.V1PodStatus(
            phase="Pending",
            container_statuses=[
                client.V1ContainerStatus(
                    image="test-image",
                    image_id="test-image-id",
                    name="nemo-job-task",
                    ready=False,
                    restart_count=0,
                    state=client.V1ContainerState(waiting=client.V1ContainerStateWaiting(reason=reason)),
                )
            ],
        ),
    )


@pytest.mark.parametrize("reason", ["ImagePullBackOff", "ErrImagePull"])
def test_retried_image_pull_is_pending_not_error(reason: str) -> None:
    """The kubelet keeps retrying these, and the pull often succeeds, so the pod is still starting."""
    pod_status = map_pod_to_pod_status(_waiting_pod(reason))

    assert pod_status.errors == {}
    assert pod_status.waiting == {"nemo-job-task": reason}
    assert map_pod_status_to_platform_status(pod_status) == PlatformJobStatus.PENDING


@pytest.mark.parametrize("reason", ["InvalidImageName", "CreateContainerConfigError"])
def test_unrecoverable_waiting_reason_is_error(reason: str) -> None:
    pod_status = map_pod_to_pod_status(_waiting_pod(reason))

    assert pod_status.errors == {"nemo-job-task": reason}
    assert map_pod_status_to_platform_status(pod_status) == PlatformJobStatus.ERROR


@patch("nmp.core.jobs.controllers.backends.kubernetes.common.client_from_platform")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.get_pod_details")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.list_pod_status")
def test_update_all_tasks_ignores_warning_events_for_a_running_pod(
    mock_list_pod_status: MagicMock,
    mock_get_pod_details: MagicMock,
    mock_client_from_platform: MagicMock,
    test_step_active: PlatformJobStepWithContext,
) -> None:
    """A pod that recovered keeps reporting active; its stale Warning event is not the current state."""
    mock_list_pod_status.return_value = [_pod(phase="Running", active={"nemo-job-task"})]
    mock_get_pod_details.return_value = ({"phase": "Running"}, {"failed": "Error: ImagePullBackOff"}, "")
    jobs_client = MagicMock()
    mock_client_from_platform.return_value = jobs_client

    has_errors = update_all_tasks(MagicMock(), MagicMock(), "ns", test_step_active)

    assert has_errors is False
    body = jobs_client.update_job_step_task.call_args.kwargs["body"]
    assert body.status == PlatformJobStatus.ACTIVE
    assert body.error_details == {}


@patch("nmp.core.jobs.controllers.backends.kubernetes.common.client_from_platform")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.get_pod_details")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.list_pod_status")
def test_update_all_tasks_keeps_a_retrying_pull_pending(
    mock_list_pod_status: MagicMock,
    mock_get_pod_details: MagicMock,
    mock_client_from_platform: MagicMock,
    test_step_active: PlatformJobStepWithContext,
) -> None:
    """A backing-off pull emits a Failed event per attempt; the task must not go terminal on it."""
    mock_list_pod_status.return_value = [_pod(phase="Pending", waiting={"nemo-job-task": "ImagePullBackOff"})]
    mock_get_pod_details.return_value = ({"phase": "Pending"}, {"failed": "Error: ImagePullBackOff"}, "")
    jobs_client = MagicMock()
    mock_client_from_platform.return_value = jobs_client

    has_errors = update_all_tasks(MagicMock(), MagicMock(), "ns", test_step_active)

    assert has_errors is False
    body = jobs_client.update_job_step_task.call_args.kwargs["body"]
    assert body.status == PlatformJobStatus.PENDING
    assert body.error_details == {}


@patch("nmp.core.jobs.controllers.backends.kubernetes.common.client_from_platform")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.get_pod_details")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.list_pod_status")
def test_update_all_tasks_reports_error_when_a_sibling_container_failed(
    mock_list_pod_status: MagicMock,
    mock_get_pod_details: MagicMock,
    mock_client_from_platform: MagicMock,
    test_step_active: PlatformJobStepWithContext,
) -> None:
    """A retrying pull must not mask a container that has already failed."""
    mock_list_pod_status.return_value = [
        _pod(phase="Pending", errors={"sidecar": 1}, waiting={"nemo-job-task": "ImagePullBackOff"})
    ]
    mock_get_pod_details.return_value = ({"phase": "Pending"}, {"failed": "sidecar exited 1"}, "")
    jobs_client = MagicMock()
    mock_client_from_platform.return_value = jobs_client

    has_errors = update_all_tasks(MagicMock(), MagicMock(), "ns", test_step_active)

    assert has_errors is True
    body = jobs_client.update_job_step_task.call_args.kwargs["body"]
    assert body.status == PlatformJobStatus.ERROR
    assert body.error_details["failed"] == "sidecar exited 1"


@patch("nmp.core.jobs.controllers.backends.kubernetes.common.client_from_platform")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.get_pod_details")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.list_pod_status")
def test_update_all_tasks_reports_error_when_an_active_sibling_container_failed(
    mock_list_pod_status: MagicMock,
    mock_get_pod_details: MagicMock,
    mock_client_from_platform: MagicMock,
    test_step_active: PlatformJobStepWithContext,
) -> None:
    mock_list_pod_status.return_value = [_pod(phase="Running", errors={"sidecar": 1}, active={"nemo-job-task"})]
    mock_get_pod_details.return_value = ({"phase": "Running"}, {"failed": "sidecar exited 1"}, "")
    jobs_client = MagicMock()
    mock_client_from_platform.return_value = jobs_client

    has_errors = update_all_tasks(MagicMock(), MagicMock(), "ns", test_step_active)

    assert has_errors is True
    body = jobs_client.update_job_step_task.call_args.kwargs["body"]
    assert body.status == PlatformJobStatus.ERROR
    assert body.error_details["failed"] == "sidecar exited 1"


@patch("nmp.core.jobs.controllers.backends.kubernetes.common.client_from_platform")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.get_pod_details")
@patch("nmp.core.jobs.controllers.backends.kubernetes.common.list_pod_status")
def test_update_all_tasks_reports_error_for_a_pending_pod_that_is_not_retrying(
    mock_list_pod_status: MagicMock,
    mock_get_pod_details: MagicMock,
    mock_client_from_platform: MagicMock,
    test_step_active: PlatformJobStepWithContext,
) -> None:
    mock_list_pod_status.return_value = [_pod(phase="Pending", waiting={"nemo-job-task": "waiting"})]
    mock_get_pod_details.return_value = ({"phase": "Pending"}, {"inspect_failed": "no such image"}, "")
    jobs_client = MagicMock()
    mock_client_from_platform.return_value = jobs_client

    has_errors = update_all_tasks(MagicMock(), MagicMock(), "ns", test_step_active)

    assert has_errors is True
    body = jobs_client.update_job_step_task.call_args.kwargs["body"]
    assert body.status == PlatformJobStatus.ERROR
    assert body.error_details["inspect_failed"] == "no such image"
