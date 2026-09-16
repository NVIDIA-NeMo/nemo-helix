# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for map_kubernetes_job_status_to_step_status and aggregate_pod_statuses_for_job_step."""

from unittest.mock import MagicMock, patch

import pytest
from nhx.common.jobs.schemas import PlatformJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import PlatformJobStepWithContext
from nhx.core.jobs.controllers.backends.kubernetes.common import (
    PodStatus,
    aggregate_pod_statuses_for_job_step,
)
from nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job import map_kubernetes_job_status_to_step_status


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


@patch("nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
def test_map_status_empty_pods_returns_pending_waiting_message(
    mock_list_pods: MagicMock, test_step_pending: PlatformJobStepWithContext
) -> None:
    mock_list_pods.return_value = []
    job = _job()
    core_v1 = MagicMock()

    status, details = map_kubernetes_job_status_to_step_status(job, core_v1, test_step_pending)

    assert status == PlatformJobStatus.PENDING
    assert "Waiting for pods" in details["message"]


@patch("nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
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


@patch("nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
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


@patch("nhx.core.jobs.controllers.backends.kubernetes.kubernetes_job.list_pod_status")
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
