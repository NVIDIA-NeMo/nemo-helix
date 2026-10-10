# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from nemo_helix_plugin.client.errors import NemoTransportError
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import HelixJobStepWithContext
from nhx.core.jobs.app.providers import SubprocessExecutionProvider
from nhx.core.jobs.config import JobsStorageConfig
from nhx.core.jobs.controllers.backends import JobUpdate
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.backends.test import MockDockerCPUJobBackend
from nhx.core.jobs.controllers.reconciler import JobReconciler

from services.core.jobs.tests.controllers.client_mocks import data_response, paginated_response


def test_job_reconciler_syncs_active_job(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_active: HelixJobStepWithContext,
):
    job_reconciler = JobReconciler(backend_registry, mock_nemo_client)

    # Mock the jobs list response
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_active])

    # Get the test backend from the registry
    test_backend = job_reconciler._backend_registry.get_backend(provider="cpu", profile="default")
    assert isinstance(test_backend, MockDockerCPUJobBackend)

    # Run reconciler step
    job_reconciler.step()

    # Verify the typed Jobs client was called with the correct deepObject filter for
    # active/pending steps. The status list is encoded as a comma-joined filter value.
    assert mock_jobs_client.list_steps.call_args_list[0].kwargs == {
        "workspace": "-",
        "name": "-",
        "query_params": {
            "filter[status]": "pending,active,cancelling,pausing",
            "sort": "updated_at",
        },
    }
    assert (
        mock_jobs_client.list_steps.call_args_list[1].kwargs["query_params"]["filter[status]"]
        == "paused,error,cancelled"
    )

    # Test backend should have received one sync call for our test job
    assert len(test_backend.mock.sync_calls) == 1
    assert test_backend.mock.sync_calls[0]["step"].id == test_step_active.id

    # Verify the status update was called with the correct job ID and status
    mock_jobs_client.update_job_step_status.assert_called()
    update_call = mock_jobs_client.update_job_step_status.call_args
    assert update_call.kwargs["name"] == "test-step"
    assert update_call.kwargs["workspace"] == "default"
    assert update_call.kwargs["job"] == "test-job-id"
    assert update_call.kwargs["body"].status == HelixJobStatus.COMPLETED


def test_job_reconciler_logs_diagnostics_for_error_transition_in_debug_mode(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_active: HelixJobStepWithContext,
):
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_active])

    job_reconciler = JobReconciler(backend_registry, mock_nemo_client)
    test_backend = job_reconciler._backend_registry.get_backend(provider="cpu", profile="default")
    assert isinstance(test_backend, MockDockerCPUJobBackend)

    with (
        patch.object(test_backend, "sync", return_value=JobUpdate(status=HelixJobStatus.ERROR)),
        patch("nhx.core.jobs.controllers.reconciler.log_job_diagnostics_if_debug") as log_diagnostics,
    ):
        job_reconciler.step()

    log_diagnostics.assert_called_once_with(
        mock_nemo_client,
        test_step_active,
        logger=job_reconciler._logger,
        context="step transitioned to error during reconciliation",
    )


def test_job_reconciler_marks_itself_unhealthy_after_transport_failure(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
):
    job_reconciler = JobReconciler(backend_registry, mock_nemo_client)
    mock_jobs_client.list_steps.return_value = paginated_response([])
    job_reconciler.step()
    assert job_reconciler.is_healthy

    request = httpx.Request("GET", "http://localhost/apis/jobs/v2/workspaces/-/jobs/-/steps")
    mock_jobs_client.list_steps.side_effect = NemoTransportError(
        httpx.ConnectError("Connection refused", request=request)
    )
    job_reconciler.step()

    assert not job_reconciler.is_healthy


def _aged_step(
    step: HelixJobStepWithContext, status: HelixJobStatus, age: datetime.timedelta
) -> HelixJobStepWithContext:
    aged = step.model_copy(deep=True)
    aged.status = status
    # The retention clock is stopped_at. A later updated_at must not extend the window.
    now = datetime.datetime.now(datetime.timezone.utc)
    aged.updated_at = now
    aged.status_details = {"stopped_at": (now - age).isoformat()}
    aged.step_spec.executor = SubprocessExecutionProvider(provider="subprocess", profile="default", command=["true"])
    return aged


def _job(
    pause_ttl_seconds: int | None = None,
    status_details: dict | None = None,
    attempt_id: str | None = None,
):
    return SimpleNamespace(
        attempt_id=attempt_id,
        status_details=status_details or {},
        control=SimpleNamespace(pause_ttl_seconds=pause_ttl_seconds),
    )


def test_storage_retention_uses_live_job_ttl(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_paused: HelixJobStepWithContext,
    test_step_error: HelixJobStepWithContext,
):
    platform_defaults = JobsStorageConfig()
    reconciler = JobReconciler(backend_registry, mock_nemo_client, storage_config=platform_defaults)
    backend = reconciler._backend_registry.get_backend(provider="subprocess", profile="default")
    paused_inside = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(days=6))
    paused_default = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(days=8))
    paused_extended = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(days=8))
    paused_short = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(hours=2))
    error_inside = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(hours=3))
    error_default = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(hours=5))
    zero_ttl = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(seconds=1))
    short_failed = JobsStorageConfig(failed_storage_ttl_seconds=3600)
    error_short = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(hours=2))
    cancelled = _aged_step(test_step_paused, HelixJobStatus.CANCELLED, datetime.timedelta(seconds=1))
    old_attempt = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(days=8))

    cases = [
        (paused_inside, _job(), platform_defaults, False, False),
        (paused_default, _job(), platform_defaults, True, True),
        (paused_extended, _job(pause_ttl_seconds=14 * 24 * 3600), platform_defaults, False, False),
        (
            paused_extended,
            _job(pause_ttl_seconds=14 * 24 * 3600),
            JobsStorageConfig(maximum_pause_ttl_seconds=5 * 24 * 3600),
            True,
            True,
        ),
        (paused_short, _job(pause_ttl_seconds=3600), platform_defaults, True, True),
        (error_inside, _job(), platform_defaults, False, False),
        (error_default, _job(), platform_defaults, True, False),
        (error_short, _job(), short_failed, True, False),
        (zero_ttl, _job(pause_ttl_seconds=0), platform_defaults, True, True),
        (cancelled, _job(), platform_defaults, True, False),
        (old_attempt, _job(attempt_id="some-other-attempt"), platform_defaults, False, False),
    ]

    for step, job, storage, should_reclaim, should_cancel in cases:
        reconciler._storage_config = storage
        mock_jobs_client.list_steps.side_effect = [paginated_response([]), paginated_response([step])]
        mock_jobs_client.get_job.return_value = data_response(job)
        mock_jobs_client.update_job_status_details.reset_mock()
        mock_jobs_client.cancel_job.reset_mock()
        with patch.object(backend, "reclaim_persistent_storage") as reclaim:
            reconciler.step()
        if should_reclaim:
            reclaim.assert_called_once_with(step.workspace, step.job)
            mock_jobs_client.update_job_status_details.assert_called_once()
        else:
            reclaim.assert_not_called()
            mock_jobs_client.update_job_status_details.assert_not_called()
        if should_cancel:
            mock_jobs_client.cancel_job.assert_called_once_with(name=step.job, workspace=step.workspace)
        else:
            mock_jobs_client.cancel_job.assert_not_called()
