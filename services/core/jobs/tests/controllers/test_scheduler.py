# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import pytest
import time_machine
from nemo_helix_plugin.client.errors import ConflictError, NemoTransportError
from nemo_helix_plugin.jobs.types import AuthContext
from nemo_helix_plugin.jobs.types import HelixJobStepWithContext as SDKJobStepWithContext
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import HelixJobStepWithContext
from nhx.core.jobs.app.constants import RESUMED_AT
from nhx.core.jobs.controllers import scheduler as scheduler_module
from nhx.core.jobs.controllers.admission import AdmissionDecision
from nhx.core.jobs.controllers.backends.exceptions import ResourceAllocationError, SchedulingDeferred
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.backends.test import MockDockerCPUJobBackend
from nhx.core.jobs.controllers.scheduler import JobScheduler
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Gauge, InMemoryMetricReader
from pytest import fixture

from services.core.jobs.tests.controllers.client_mocks import data_response, paginated_response


def _conflict_error(detail: str) -> ConflictError:
    """Build a client ``ConflictError`` (HTTP 409) with the given detail message."""
    request = httpx.Request(
        "PATCH", "http://localhost/apis/jobs/v2/workspaces/default/jobs/test-job-id/steps/test-step/status"
    )
    response = httpx.Response(409, request=request, json={"detail": detail})
    return ConflictError(response)


@fixture
def job_scheduler(backend_registry: BackendRegistry, mock_nemo_client) -> JobScheduler:
    return JobScheduler(backend_registry, mock_nemo_client)


@fixture
def test_step_created(test_step_pending: HelixJobStepWithContext) -> HelixJobStepWithContext:
    return test_step_pending.model_copy(update={"status": HelixJobStatus.CREATED})


def test_does_schedule_job(
    job_scheduler: JobScheduler,
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    # Mock the jobs list response
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created])

    # Get the test backend from the registry
    backend = backend_registry.get_backend(provider="cpu", profile="default")
    assert isinstance(backend, MockDockerCPUJobBackend)
    test_backend = backend

    # Run scheduler step
    job_scheduler.step()

    # Verify the typed Jobs client was called with the correct deepObject filter.
    # The status list is encoded as ``filter[status]=created,resuming`` (comma form).
    mock_jobs_client.list_steps.assert_any_call(
        workspace="-",
        name="-",
        query_params={
            "filter[status]": "created,queued,resuming",
            "sort": "created_at",
        },
    )

    # Test backend should have received one schedule call for our test job
    assert len(test_backend.mock.schedule_calls) == 1
    assert test_backend.mock.schedule_calls[0]["step"].id == test_step_created.id
    assert test_backend.mock.sync_calls == []


def test_scheduling_deferred_keeps_step_created_with_visible_status_details(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created])

    with patch.object(job_scheduler, "schedule_step", side_effect=SchedulingDeferred("capacity full")):
        job_scheduler.step()

    mock_jobs_client.update_job_step_status.assert_called_once()
    call = mock_jobs_client.update_job_step_status.call_args
    assert call.kwargs["name"] == test_step_created.name
    assert call.kwargs["workspace"] == test_step_created.workspace
    assert call.kwargs["job"] == test_step_created.job
    body = call.kwargs["body"]
    assert body.status == HelixJobStatus.CREATED
    assert body.status_details == {"message": "capacity full"}


def test_scheduling_deferred_preserves_step_resuming_with_visible_status_details(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_resuming: HelixJobStepWithContext,
):
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_resuming])

    with patch.object(job_scheduler, "schedule_step", side_effect=SchedulingDeferred("capacity full")):
        job_scheduler.step()

    mock_jobs_client.update_job_step_status.assert_called_once()
    call = mock_jobs_client.update_job_step_status.call_args
    assert call.kwargs["name"] == test_step_resuming.name
    assert call.kwargs["workspace"] == test_step_resuming.workspace
    assert call.kwargs["job"] == test_step_resuming.job
    body = call.kwargs["body"]
    assert body.status == HelixJobStatus.RESUMING
    assert body.status_details == {"message": "capacity full"}


def test_scheduling_deferred_resends_unchanged_resuming_step(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_resuming: HelixJobStepWithContext,
):
    resuming_step = test_step_resuming.model_copy(update={"status_details": {"message": "capacity full"}})
    mock_jobs_client.list_steps.return_value = paginated_response([resuming_step])

    with patch.object(job_scheduler, "schedule_step", side_effect=SchedulingDeferred("capacity full")):
        job_scheduler.step()

    # The dispatcher no-ops the step write but still retries job and attempt propagation.
    mock_jobs_client.update_job_step_status.assert_called_once()


def test_scheduling_deferred_status_write_failure_does_not_skip_later_steps(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    next_step = test_step_created.model_copy(update={"id": "next-step-id", "name": "next-step", "job": "next-job"})
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created, next_step])
    mock_jobs_client.update_job_step_status.side_effect = [RuntimeError("persist failed"), data_response(None)]

    with patch.object(job_scheduler, "schedule_step", side_effect=SchedulingDeferred("capacity full")) as schedule_step:
        job_scheduler.step()

    assert schedule_step.call_count == 2
    assert mock_jobs_client.update_job_step_status.call_count == 2
    first_body = mock_jobs_client.update_job_step_status.call_args_list[0].kwargs["body"]
    second_body = mock_jobs_client.update_job_step_status.call_args_list[1].kwargs["body"]
    assert first_body.status == HelixJobStatus.CREATED
    assert second_body.status == HelixJobStatus.CREATED
    assert second_body.status_details == {"message": "capacity full"}


def test_resource_allocation_error_marks_step_as_error(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    """When ResourceAllocationError is raised (e.g. no GPUs), scheduler marks step as error with error_details."""
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created])
    error_message = "No GPUs available on this system. GPU jobs require a system with NVIDIA GPUs."

    with patch.object(job_scheduler, "schedule_step", side_effect=ResourceAllocationError(error_message)):
        job_scheduler.step()

    mock_jobs_client.update_job_step_status.assert_called_once()
    call = mock_jobs_client.update_job_step_status.call_args
    assert call.kwargs["name"] == test_step_created.name
    assert call.kwargs["workspace"] == test_step_created.workspace
    assert call.kwargs["job"] == test_step_created.job
    body = call.kwargs["body"]
    assert body.status == HelixJobStatus.ERROR
    assert body.status_details == {"message": error_message}
    assert body.error_details == {"message": error_message}


def test_scheduler_logs_diagnostics_for_unexpected_schedule_error_in_debug_mode(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created])

    with (
        patch.object(job_scheduler, "schedule_step", side_effect=RuntimeError("boom")),
        patch("nhx.core.jobs.controllers.scheduler.logger.isEnabledFor", return_value=True),
        patch("nhx.core.jobs.controllers.scheduler.log_job_diagnostics_if_debug") as log_diagnostics,
    ):
        job_scheduler.step()

    log_diagnostics.assert_called_once_with(
        mock_nemo_client,
        test_step_created,
        logger=job_scheduler._logger,
        context="unexpected scheduling error",
    )


def test_scheduler_does_not_mark_step_error_when_pending_update_conflicts_with_concurrent_advance(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created])

    conflict = _conflict_error(
        "Invalid status transition from HelixJobStatus.ACTIVE to HelixJobStatus.PENDING for step test-step-id"
    )
    active_step = test_step_created.model_copy(update={"status": HelixJobStatus.ACTIVE})
    mock_jobs_client.update_job_step_status.side_effect = [conflict]
    get_step_resp = active_step
    mock_jobs_client.get_job_step.return_value.data.return_value = get_step_resp

    job_scheduler.step()

    mock_jobs_client.update_job_step_status.assert_called_once()
    update_call = mock_jobs_client.update_job_step_status.call_args
    assert update_call.kwargs["name"] == test_step_created.name
    assert update_call.kwargs["workspace"] == test_step_created.workspace
    assert update_call.kwargs["job"] == test_step_created.job
    assert update_call.kwargs["body"].status == HelixJobStatus.PENDING

    mock_jobs_client.get_job_step.assert_called_once_with(
        name=test_step_created.name,
        workspace=test_step_created.workspace,
        job=test_step_created.job,
        attempt=test_step_created.attempt_id,
    )


def test_scheduler_does_not_ignore_pending_update_conflict_when_step_remains_resuming(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_pending: HelixJobStepWithContext,
):
    resuming_step = test_step_pending.model_copy(update={"status": HelixJobStatus.RESUMING})
    mock_jobs_client.list_steps.return_value = paginated_response([resuming_step])

    conflict = _conflict_error(
        "Invalid status transition from HelixJobStatus.RESUMING to HelixJobStatus.PENDING for step test-step-id"
    )
    # First call (CREATED->PENDING) conflicts; the second call (marking ERROR) succeeds
    # and its response is chained with ``.data()`` by the scheduler.
    mock_jobs_client.update_job_step_status.side_effect = [conflict, data_response(None)]
    mock_jobs_client.get_job_step.return_value.data.return_value = resuming_step

    job_scheduler.step()

    assert mock_jobs_client.update_job_step_status.call_count == 2
    error_call = mock_jobs_client.update_job_step_status.call_args_list[1]
    error_body = error_call.kwargs["body"]
    assert error_body.status == HelixJobStatus.ERROR
    assert "409" in error_body.status_details["message"]
    mock_jobs_client.get_job_step.assert_called_once_with(
        name=resuming_step.name,
        workspace=resuming_step.workspace,
        job=resuming_step.job,
        attempt=resuming_step.attempt_id,
    )


def test_scheduler_marks_itself_unhealthy_after_transport_failure(
    job_scheduler: JobScheduler,
    mock_jobs_client,
):
    mock_jobs_client.list_steps.return_value = paginated_response([])
    job_scheduler.step()
    assert job_scheduler.is_healthy

    request = httpx.Request("GET", "http://localhost/apis/jobs/v2/workspaces/-/jobs/-/steps")
    mock_jobs_client.list_steps.side_effect = NemoTransportError(
        httpx.ConnectError("Connection refused", request=request)
    )
    job_scheduler.step()

    assert not job_scheduler.is_healthy


class _DeferringAdmission:
    """Admission stub that defers steps by name and records the order of checks."""

    def __init__(self, deferred_names: set[str]) -> None:
        self._deferred_names = deferred_names
        self.checked: list[str] = []

    def check(self, step: SDKJobStepWithContext) -> AdmissionDecision:
        self.checked.append(step.name)
        if step.name in self._deferred_names:
            return AdmissionDecision.defer("waiting for capacity")
        return AdmissionDecision.admit()


def test_default_admission_dispatches_every_ready_step(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    queued_step = test_step_created.model_copy(
        update={"id": "queued-step-id", "name": "queued-step", "status": HelixJobStatus.QUEUED}
    )
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created, queued_step])

    with patch.object(job_scheduler, "schedule_step", wraps=job_scheduler.schedule_step) as schedule_step:
        job_scheduler.step()

    assert schedule_step.call_count == 2


def test_admission_deferral_queues_step_without_dispatching(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created])
    admission = _DeferringAdmission({test_step_created.name})
    scheduler = JobScheduler(backend_registry, mock_nemo_client, admission=admission)

    with patch.object(scheduler, "schedule_step") as schedule_step:
        scheduler.step()

    schedule_step.assert_not_called()
    mock_jobs_client.update_job_step_status.assert_called_once()
    body = mock_jobs_client.update_job_step_status.call_args.kwargs["body"]
    assert body.status == HelixJobStatus.QUEUED
    assert body.status_details == {"message": "waiting for capacity"}


def test_admission_deferral_keeps_resuming_step_resuming(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_resuming: HelixJobStepWithContext,
):
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_resuming])
    scheduler = JobScheduler(
        backend_registry, mock_nemo_client, admission=_DeferringAdmission({test_step_resuming.name})
    )

    scheduler.step()

    body = mock_jobs_client.update_job_step_status.call_args.kwargs["body"]
    assert body.status == HelixJobStatus.RESUMING
    assert body.status_details == {"message": "waiting for capacity"}


def test_admission_checks_in_fifo_order_and_deferral_does_not_block_later_steps(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    next_step = test_step_created.model_copy(update={"id": "next-step-id", "name": "next-step", "job": "next-job"})
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created, next_step])
    admission = _DeferringAdmission({test_step_created.name})
    scheduler = JobScheduler(backend_registry, mock_nemo_client, admission=admission)

    with patch.object(scheduler, "schedule_step", wraps=scheduler.schedule_step) as schedule_step:
        scheduler.step()

    assert admission.checked == [test_step_created.name, "next-step"]
    schedule_step.assert_called_once_with(next_step)


def test_queued_step_is_dispatched_once_admitted(
    job_scheduler: JobScheduler,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    queued_step = test_step_created.model_copy(update={"status": HelixJobStatus.QUEUED})
    mock_jobs_client.list_steps.return_value = paginated_response([queued_step])

    job_scheduler.step()

    body = mock_jobs_client.update_job_step_status.call_args.kwargs["body"]
    assert body.status == HelixJobStatus.PENDING


def test_admission_deferral_status_write_failure_does_not_skip_later_steps(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    next_step = test_step_created.model_copy(update={"id": "next-step-id", "name": "next-step", "job": "next-job"})
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created, next_step])
    mock_jobs_client.update_job_step_status.side_effect = [RuntimeError("persist failed"), data_response(None)]
    admission = _DeferringAdmission({test_step_created.name, "next-step"})
    scheduler = JobScheduler(backend_registry, mock_nemo_client, admission=admission)

    scheduler.step()

    assert admission.checked == [test_step_created.name, "next-step"]
    assert mock_jobs_client.update_job_step_status.call_count == 2


def test_admission_deferral_resends_when_step_already_queued_with_same_message(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    queued_step = test_step_created.model_copy(
        update={"status": HelixJobStatus.QUEUED, "status_details": {"message": "waiting for capacity"}}
    )
    mock_jobs_client.list_steps.return_value = paginated_response([queued_step])
    scheduler = JobScheduler(backend_registry, mock_nemo_client, admission=_DeferringAdmission({queued_step.name}))

    scheduler.step()

    # The dispatcher no-ops the step write but still retries job and attempt propagation.
    mock_jobs_client.update_job_step_status.assert_called_once()


def test_admission_deferral_rewrites_when_reason_changes(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    queued_step = test_step_created.model_copy(
        update={"status": HelixJobStatus.QUEUED, "status_details": {"message": "an older reason"}}
    )
    mock_jobs_client.list_steps.return_value = paginated_response([queued_step])
    scheduler = JobScheduler(backend_registry, mock_nemo_client, admission=_DeferringAdmission({queued_step.name}))

    scheduler.step()

    body = mock_jobs_client.update_job_step_status.call_args.kwargs["body"]
    assert body.status == HelixJobStatus.QUEUED
    assert body.status_details == {"message": "waiting for capacity"}


def _gauge_points(reader) -> dict[str, dict[tuple[str, str], int]]:
    """Collect gauge data points keyed by metric name, then by (workspace, user)."""
    points: dict[str, dict[tuple[str, str], int]] = {}
    data = reader.get_metrics_data()
    for rm in data.resource_metrics if data else []:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name not in {"nhx.jobs.scheduler.queued", "nhx.jobs.scheduler.running"}:
                    continue
                points[metric.name] = {
                    (p.attributes["workspace"], p.attributes["user"]): p.value for p in metric.data.data_points
                }
    return points


def test_queued_and_running_gauges_report_counts_by_workspace_and_user(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    with patch.object(scheduler_module, "meter", provider.get_meter("test")):
        scheduler = JobScheduler(
            backend_registry, mock_nemo_client, admission=_DeferringAdmission({"q1", "q2", "q3", "q4", "q5"})
        )

    def _step(name: str, workspace: str, status: HelixJobStatus, user: str | None) -> HelixJobStepWithContext:
        auth_context = AuthContext(principal_id=user) if user is not None else None
        return test_step_created.model_copy(
            update={"id": name, "name": name, "workspace": workspace, "status": status, "auth_context": auth_context}
        )

    waiting = [
        _step("q1", "ws-a", HelixJobStatus.QUEUED, "alice"),
        _step("q2", "ws-a", HelixJobStatus.QUEUED, "bob"),
        _step("q3", "ws-a", HelixJobStatus.QUEUED, "bob"),
        _step("q4", "ws-b", HelixJobStatus.QUEUED, None),
        _step("q5", "ws-a", HelixJobStatus.RESUMING, "alice"),
    ]
    running = [
        _step("r1", "ws-a", HelixJobStatus.ACTIVE, "alice"),
        _step("r2", "ws-a", HelixJobStatus.ACTIVE, "bob"),
        _step("r3", "ws-b", HelixJobStatus.ACTIVE, None),
    ]
    mock_jobs_client.list_steps.side_effect = [paginated_response(waiting), paginated_response(running)]

    scheduler.step()

    points = _gauge_points(reader)
    assert points["nhx.jobs.scheduler.queued"] == {
        ("ws-a", "alice"): 2,
        ("ws-a", "bob"): 2,
        ("ws-b", "unknown"): 1,
    }
    assert points["nhx.jobs.scheduler.running"] == {
        ("ws-a", "alice"): 1,
        ("ws-a", "bob"): 1,
        ("ws-b", "unknown"): 1,
    }


_BACKLOG_OLDEST_AGE = "nhx.jobs.scheduler.backlog.oldest_age"
_BACKLOG_SNAPSHOT_TIMESTAMP = "nhx.jobs.scheduler.backlog.snapshot_timestamp"
_BACKLOG_TEST_TIME = datetime(2026, 8, 15, 12, tzinfo=timezone.utc)


@fixture
def scheduler_with_backlog_metrics(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    test_step_created: HelixJobStepWithContext,
) -> Iterator[tuple[JobScheduler, InMemoryMetricReader]]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    try:
        with patch.object(scheduler_module, "meter", provider.get_meter("test")):
            scheduler = JobScheduler(
                backend_registry,
                mock_nemo_client,
                admission=_DeferringAdmission({test_step_created.name, "resumed-step", "unknown-step"}),
            )
            yield scheduler, reader
    finally:
        provider.shutdown()


def _backlog_points(reader: InMemoryMetricReader) -> dict[str, float]:
    points: dict[str, float] = {}
    data = reader.get_metrics_data()
    for resource in data.resource_metrics if data else []:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                if metric.name not in {_BACKLOG_OLDEST_AGE, _BACKLOG_SNAPSHOT_TIMESTAMP}:
                    continue
                assert metric.unit == "s"
                assert isinstance(metric.data, Gauge)
                (point,) = metric.data.data_points
                assert not point.attributes
                points[metric.name] = point.value
    return points


@pytest.mark.parametrize(
    ("waiting_status", "waiting_age", "oldest_age"),
    [(HelixJobStatus.CREATED, 45, 45), (HelixJobStatus.QUEUED, 20, 30)],
)
def test_backlog_age_uses_creation_or_latest_resume(
    scheduler_with_backlog_metrics: tuple[JobScheduler, InMemoryMetricReader],
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
    waiting_status: HelixJobStatus,
    waiting_age: int,
    oldest_age: int,
):
    scheduler, reader = scheduler_with_backlog_metrics
    now = _BACKLOG_TEST_TIME
    waiting = test_step_created.model_copy(
        update={
            "status": waiting_status,
            "created_at": now - timedelta(seconds=waiting_age),
            "updated_at": now - timedelta(seconds=1),
            "workspace": "workspace-one",
            "auth_context": AuthContext(principal_id="user-one"),
        }
    )
    resumed = test_step_created.model_copy(
        update={
            "id": "resumed-step",
            "name": "resumed-step",
            "status": HelixJobStatus.RESUMING,
            "created_at": now - timedelta(hours=1),
            "updated_at": now - timedelta(seconds=2),
            "status_details": {RESUMED_AT: (now - timedelta(seconds=30)).isoformat()},
            "workspace": "workspace-two",
            "auth_context": AuthContext(principal_id="user-two"),
        }
    )
    mock_jobs_client.list_steps.side_effect = [paginated_response([waiting, resumed]), paginated_response([])]

    with time_machine.travel(now, tick=False):
        scheduler.step()
        calls = mock_jobs_client.mock_calls.copy()
        assert _backlog_points(reader) == {
            _BACKLOG_OLDEST_AGE: oldest_age,
            _BACKLOG_SNAPSHOT_TIMESTAMP: now.timestamp(),
        }
        assert mock_jobs_client.mock_calls == calls


def test_empty_backlog_reports_zero_but_failed_scan_retains_previous_snapshot(
    scheduler_with_backlog_metrics: tuple[JobScheduler, InMemoryMetricReader],
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    scheduler, reader = scheduler_with_backlog_metrics
    now = _BACKLOG_TEST_TIME
    request = httpx.Request("GET", "http://localhost/apis/jobs/v2/workspaces/-/jobs/-/steps")
    failed_scan = NemoTransportError(httpx.ConnectError("Connection refused", request=request))
    backlog = test_step_created.model_copy(update={"created_at": now - timedelta(seconds=35)})
    empty_scan = paginated_response([])
    mock_jobs_client.list_steps.side_effect = [
        failed_scan,
        paginated_response([backlog]),
        empty_scan,
        failed_scan,
        empty_scan,
        empty_scan,
    ]

    with time_machine.travel(now, tick=False) as clock:
        assert _backlog_points(reader) == {}
        mock_jobs_client.list_steps.assert_not_called()
        scheduler.step()
        assert _backlog_points(reader) == {}
        clock.shift(timedelta(seconds=10))
        scheduler.step()
        points = {_BACKLOG_OLDEST_AGE: 45, _BACKLOG_SNAPSHOT_TIMESTAMP: now.timestamp() + 10}
        assert _backlog_points(reader) == points
        clock.shift(timedelta(seconds=10))
        scheduler.step()
        assert _backlog_points(reader) == points
        clock.shift(timedelta(seconds=10))
        scheduler.step()
        assert _backlog_points(reader) == {_BACKLOG_OLDEST_AGE: 0, _BACKLOG_SNAPSHOT_TIMESTAMP: now.timestamp() + 30}
        assert mock_jobs_client.list_steps.call_count == 6


def test_backlog_reports_snapshot_without_age_if_a_candidate_has_no_anchor(
    scheduler_with_backlog_metrics: tuple[JobScheduler, InMemoryMetricReader],
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
):
    scheduler, reader = scheduler_with_backlog_metrics
    now = _BACKLOG_TEST_TIME
    waiting = test_step_created.model_copy(update={"created_at": now - timedelta(seconds=30)})
    unknown = test_step_created.model_copy(
        update={
            "id": "unknown-step",
            "name": "unknown-step",
            "created_at": None,
            "updated_at": now,
            "status_details": {},
        }
    )
    mock_jobs_client.list_steps.side_effect = [
        paginated_response([waiting]),
        paginated_response([]),
        paginated_response([waiting, unknown]),
        paginated_response([]),
    ]

    with time_machine.travel(now, tick=False) as clock:
        scheduler.step()
        assert _backlog_points(reader) == {_BACKLOG_OLDEST_AGE: 30, _BACKLOG_SNAPSHOT_TIMESTAMP: now.timestamp()}
        clock.shift(timedelta(seconds=10))
        scheduler.step()
        assert _backlog_points(reader) == {_BACKLOG_SNAPSHOT_TIMESTAMP: now.timestamp() + 10}


@pytest.mark.parametrize("pending_status", HelixJobStatus.lifecycle_pending())
def test_pending_job_cannot_reach_backend(
    job_scheduler: JobScheduler,
    backend_registry: BackendRegistry,
    mock_jobs_client,
    test_step_created: HelixJobStepWithContext,
    pending_status: HelixJobStatus,
):
    job = mock_jobs_client.get_job.return_value.data.return_value
    job.status = pending_status
    backend = backend_registry.get_backend(provider="cpu", profile="default")
    assert isinstance(backend, MockDockerCPUJobBackend)

    with pytest.raises(SchedulingDeferred, match="pending"):
        job_scheduler.schedule_step(test_step_created)

    assert backend.mock.schedule_calls == []


@pytest.mark.parametrize("status_code", [0, 429, 503, 403])
def test_job_lookup_failure_defers_only_retryable_errors(
    job_scheduler, mock_jobs_client, test_step_created, status_code
):
    from nemo_helix_plugin.client.errors import NemoHTTPError

    request = httpx.Request("GET", "http://jobs.test/job")
    error = (
        NemoHTTPError(httpx.Response(status_code, request=request))
        if status_code
        else NemoTransportError(httpx.ConnectTimeout("lookup timed out", request=request))
    )
    mock_jobs_client.get_job.side_effect = error
    expected = NemoHTTPError if status_code == 403 else SchedulingDeferred
    with pytest.raises(expected):
        job_scheduler.schedule_step(test_step_created)


@pytest.mark.parametrize(
    ("status_code", "headers", "deferred"),
    [
        (408, {}, True),
        (409, {}, True),
        (500, {}, True),
        (403, {}, False),
        (403, {"x-should-retry": "true"}, True),
        (503, {"x-should-retry": "false"}, False),
    ],
)
def test_scheduler_pass_uses_controller_retry_policy(
    job_scheduler,
    backend_registry,
    mock_jobs_client,
    test_step_created,
    status_code,
    headers,
    deferred,
):
    from nemo_helix_plugin.client.errors import NemoHTTPError
    from nemo_helix_plugin.client.types import PLATFORM_DEFAULT_RETRY_POLICY

    job_scheduler._nemo_client = job_scheduler._nemo_client.with_retry(PLATFORM_DEFAULT_RETRY_POLICY)
    mock_jobs_client.list_steps.return_value = paginated_response([test_step_created])
    mock_jobs_client.get_job.side_effect = NemoHTTPError(
        httpx.Response(status_code, headers=headers, request=httpx.Request("GET", "http://jobs.test/job"))
    )
    job_scheduler.step()

    backend = backend_registry.get_backend(provider="cpu", profile="default")
    assert backend.mock.schedule_calls == []
    body = mock_jobs_client.update_job_step_status.call_args.kwargs["body"]
    assert body.status == (HelixJobStatus.CREATED if deferred else HelixJobStatus.ERROR)
