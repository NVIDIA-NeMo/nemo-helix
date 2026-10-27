# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import threading
import time
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from kubernetes import client as kubernetes_client
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nhx.common.controller import Loop, TimedLoopWaiter, TrackLastExecutionTime
from nhx.common.entities import EntityConflictError
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.api.v2.jobs.schemas import HelixJobStepWithContext
from nhx.core.jobs.app.providers import SubprocessExecutionProvider
from nhx.core.jobs.app.schemas import HelixJobEnvironmentVariable, HelixJobSpec, HelixJobStepSpec
from nhx.core.jobs.config import JobsStorageConfig
from nhx.core.jobs.controllers.backends import JobUpdate
from nhx.core.jobs.controllers.backends.exceptions import JobStorageError
from nhx.core.jobs.controllers.backends.kubernetes.common import (
    KubernetesObjectMetadata,
    reclaim_job_persistent_storage,
)
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.backends.test import MockDockerCPUJobBackend
from nhx.core.jobs.controllers.reconciler import JobReconciler
from nhx.core.jobs.controllers.retention import StorageRetentionSweeper
from nhx.core.jobs.entities import HelixJobStorageReclaim, JobStorageTarget

from services.core.jobs.tests.controllers.client_mocks import data_response, paginated_response


@pytest.mark.parametrize("status", [HelixJobStatus.CANCELLED, HelixJobStatus.ERROR])
def test_job_reconciler_retries_terminal_child_cleanup_without_backend_sync(
    backend_registry, mock_nemo_client, mock_jobs_client, test_step_active, status
):
    step = test_step_active.model_copy(update={"status": status, "status_details": {"task_cleanup_pending": True}})
    # Active steps, then terminal steps pending task cleanup.
    mock_jobs_client.list_steps.side_effect = [
        paginated_response([]),
        paginated_response([step]),
    ]
    reconciler = JobReconciler(backend_registry, mock_nemo_client)
    backend = backend_registry.get_backend(provider="cpu", profile="default")

    reconciler.step()

    assert backend.mock.sync_calls == []
    mock_jobs_client.update_job_step_status.assert_called_once()
    request = mock_jobs_client.update_job_step_status.call_args.kwargs
    assert request["body"].status == status
    assert request["body"].expected_step_id == step.id
    assert request["job"] == step.job
    assert request["workspace"] == step.workspace
    assert request["name"] == step.name


@pytest.mark.parametrize("status_code", [0, 422, 503])
def test_terminal_cleanup_query_failure_does_not_block_active_reconciliation(
    backend_registry, mock_nemo_client, mock_jobs_client, test_step_active, caplog, status_code
):
    """A failed ancillary cleanup fetch must not discard already-fetched active steps."""
    request = httpx.Request("GET", "http://localhost/apis/jobs/v2/workspaces/-/jobs/-/steps")
    if status_code:
        error = NemoHTTPError(httpx.Response(status_code, request=request))
    else:
        error = NemoTransportError(httpx.ConnectError("cleanup query unavailable", request=request))
    mock_jobs_client.list_steps.side_effect = [paginated_response([test_step_active]), error]
    reconciler = JobReconciler(backend_registry, mock_nemo_client)
    backend = backend_registry.get_backend(provider="cpu", profile="default")

    reconciler.step()

    assert reconciler.is_healthy
    assert len(backend.mock.sync_calls) == 1
    mock_jobs_client.update_job_step_status.assert_called_once()
    assert mock_jobs_client.update_job_step_status.call_args.kwargs["body"].status == HelixJobStatus.COMPLETED
    assert "Could not fetch terminal steps pending task cleanup" in caplog.text


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
    mock_jobs_client.list_steps.assert_any_call(
        workspace="-",
        name="-",
        query_params={
            "filter[status]": "pending,active,cancelling,pausing",
            "sort": "updated_at",
        },
    )
    mock_jobs_client.list_steps.assert_any_call(
        workspace="-",
        name="-",
        query_params={
            "filter[status]": "cancelled,error",
            "filter[task_cleanup_pending]": "true",
            "sort": "updated_at",
        },
    )
    # Stopped steps are listed by the retention sweeper, not on the reconcile tick.
    assert mock_jobs_client.list_steps.call_count == 2

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


def _storage_spec(*, uses_storage: bool = True) -> HelixJobSpec:
    environment = [HelixJobEnvironmentVariable(name=PERSISTENT_JOB_STORAGE_PATH_ENVVAR, value="/storage")]
    return HelixJobSpec(
        steps=[
            HelixJobStepSpec(
                name="train",
                executor=SubprocessExecutionProvider(provider="subprocess", profile="default", command=["true"]),
                environment=environment if uses_storage else None,
            )
        ]
    )


def _job(
    pause_ttl_seconds: int | None = None,
    status_details: dict | None = None,
    attempt_id: str | None = None,
    uses_storage: bool = True,
):
    return SimpleNamespace(
        id="job-id",
        attempt_id=attempt_id,
        status_details=status_details or {},
        control=SimpleNamespace(pause_ttl_seconds=pause_ttl_seconds),
        platform_spec=_storage_spec(uses_storage=uses_storage),
    )


def _reconciler(backend_registry, mock_nemo_client, storage_config: JobsStorageConfig | None = None):
    system_entities = MagicMock()
    system_entities.list.return_value = SimpleNamespace(data=[])
    system_entities.update.side_effect = lambda entity: entity
    reconciler = JobReconciler(
        backend_registry,
        mock_nemo_client,
        storage_config=storage_config or JobsStorageConfig(),
        system_entity_client=system_entities,
    )
    return reconciler, system_entities


def test_storage_retention_uses_live_job_ttl(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_paused: HelixJobStepWithContext,
    test_step_error: HelixJobStepWithContext,
):
    platform_defaults = JobsStorageConfig()
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client, platform_defaults)
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
    ]

    for step, job, storage, should_reclaim, should_cancel in cases:
        reconciler._storage_config = storage
        mock_jobs_client.list_steps.side_effect = [paginated_response([step])]
        mock_jobs_client.get_job.return_value = data_response(job)
        mock_jobs_client.update_job_attempt_status_details.reset_mock()
        mock_jobs_client.update_job_step_status.reset_mock()
        mock_jobs_client.cancel_job.reset_mock()
        system_entities.create.reset_mock()
        with patch.object(backend, "reclaim_persistent_storage") as reclaim:
            reconciler.sweep_stopped_steps()
        # The reclaim pass removes the storage on a later tick, from the queued record.
        reclaim.assert_not_called()
        if should_reclaim:
            mark = mock_jobs_client.update_job_attempt_status_details.call_args.kwargs
            assert mark["body"].root["storage_reclaimed_at"]
            # The jobs API writes the mark only while the step is still current and stopped.
            assert mark["attempt"] == step.attempt_id
            assert mark["query_params"] == {"expected_step_id": step.id, "expected_step_status": step.status.value}
            (record,) = system_entities.create.call_args.args
            assert (record.job_id, record.job_workspace, record.job_name) == ("job-id", step.workspace, step.job)
            assert record.targets == [JobStorageTarget(provider="subprocess", profile="default")]
            # The step is marked so later passes skip it without looking up the job.
            body = mock_jobs_client.update_job_step_status.call_args.kwargs["body"]
            # Cancel moves a paused step to cancelling, so the mark carries that status.
            assert body.status == (HelixJobStatus.CANCELLING if should_cancel else step.status)
            assert body.expected_step_id == step.id
            assert "storage_swept_at" in body.status_details
        else:
            mock_jobs_client.update_job_attempt_status_details.assert_not_called()
            system_entities.create.assert_not_called()
            mock_jobs_client.update_job_step_status.assert_not_called()
        if should_cancel:
            mock_jobs_client.cancel_job.assert_called_once_with(name=step.job, workspace=step.workspace)
        else:
            mock_jobs_client.cancel_job.assert_not_called()


def test_storage_retention_marks_steps_with_nothing_to_reclaim(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_error: HelixJobStepWithContext,
):
    """Jobs without storage are marked once and then skipped."""
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    backend = reconciler._backend_registry.get_backend(provider="subprocess", profile="default")
    expired = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(days=1))
    cases = [
        _job(uses_storage=False),
    ]
    for job in cases:
        mock_jobs_client.list_steps.side_effect = [paginated_response([expired])]
        mock_jobs_client.get_job.return_value = data_response(job)
        mock_jobs_client.update_job_step_status.reset_mock()
        with patch.object(backend, "reclaim_persistent_storage") as reclaim:
            reconciler.sweep_stopped_steps()
        reclaim.assert_not_called()
        system_entities.create.assert_not_called()
        mock_jobs_client.update_job_step_status.assert_called_once()
        # The update names the step, so it carries the step's ID to never land on a rerun's step.
        assert mock_jobs_client.update_job_step_status.call_args.kwargs["body"].expected_step_id == expired.id

    swept = expired.model_copy(deep=True)
    swept.status_details = {**swept.status_details, "storage_swept_at": "2026-10-01T00:00:00+00:00"}
    mock_jobs_client.list_steps.side_effect = [paginated_response([swept])]
    mock_jobs_client.get_job.reset_mock()
    reconciler.sweep_stopped_steps()
    mock_jobs_client.get_job.assert_not_called()


def test_reconcile_tick_does_not_list_stopped_steps(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
):
    """Listing every stopped step takes minutes, so it belongs to the retention sweeper, not the tick."""
    reconciler, _ = _reconciler(backend_registry, mock_nemo_client)
    mock_jobs_client.list_steps.side_effect = lambda **_: paginated_response([])

    reconciler.step()
    reconciler.step()

    statuses = [call.kwargs["query_params"]["filter[status]"] for call in mock_jobs_client.list_steps.call_args_list]
    assert "paused,error,cancelled" not in statuses
    assert statuses.count("pending,active,cancelling,pausing") == 2


def test_deleted_job_storage_is_reclaimed_on_every_backend_then_cleared(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
):
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    record = HelixJobStorageReclaim(
        name=HelixJobStorageReclaim.name_for_job("job-123"),
        workspace="system",
        job_id="job-123",
        job_workspace="team",
        job_name="train-llama",
        targets=[
            JobStorageTarget(provider="subprocess", profile="default"),
            JobStorageTarget(provider="gpu", profile="not-configured"),
        ],
    )
    system_entities.list.return_value = SimpleNamespace(data=[record])
    mock_jobs_client.list_steps.side_effect = lambda **_: paginated_response([])
    backend = reconciler._backend_registry.get_backend(provider="subprocess", profile="default")

    with patch.object(backend, "reclaim_persistent_storage") as reclaim:
        reconciler.step()

    reclaim.assert_called_once_with("team", "train-llama")
    system_entities.delete_by_id.assert_called_once_with(HelixJobStorageReclaim, record.id)


@pytest.mark.parametrize("shared_backend", [False, True])
@pytest.mark.parametrize("retry_failure", [False, True])
def test_reclaim_persists_progress_across_ticks_and_controller_restarts(
    backend_registry, mock_nemo_client, shared_backend, retry_failure
):
    """Exercise restart progress with a fake store, not database serialization or version locking."""
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    first = MagicMock()
    second = first if shared_backend else MagicMock()
    # A completed cleanup is consumed: another call would create a new workload.
    if shared_backend:
        first.reclaim_persistent_storage.side_effect = [
            False,
            True,
            JobStorageError("cleanup failed") if retry_failure else False,
            True,
        ]
    else:
        first.reclaim_persistent_storage.side_effect = [True, AssertionError("completed target restarted")]
        second.reclaim_persistent_storage.side_effect = [
            JobStorageError("cleanup failed") if retry_failure else False,
            False,
            True,
        ]
    reconciler._backend_registry = MagicMock()
    reconciler._backend_registry.get_backend.side_effect = lambda *, provider, profile: {
        "cpu": first,
        "gpu": second,
    }[provider]
    record = HelixJobStorageReclaim(
        name=HelixJobStorageReclaim.name_for_job("deleted-job"),
        workspace="system",
        job_id="deleted-job",
        job_workspace="team",
        job_name="train",
        targets=[
            JobStorageTarget(provider="cpu", profile="default"),
            JobStorageTarget(provider="gpu", profile="default"),
        ],
    )
    stored = {record.name: record}
    system_entities.list.side_effect = lambda *args, **kwargs: SimpleNamespace(data=list(stored.values()))

    def persist(entity):
        stored[entity.name] = entity
        return entity

    system_entities.update.side_effect = persist
    system_entities.delete_by_id.side_effect = lambda *args: stored.clear()

    for _ in range(3):
        # Read the saved record rather than retaining progress only in controller memory.
        reconciler._reclaim_queued_storage()
        if not stored:
            break
        assert len(stored[record.name].targets) == 1
        system_entities.delete_by_id.assert_not_called()
        reconciler = JobReconciler(reconciler._backend_registry, mock_nemo_client, system_entity_client=system_entities)

    assert not stored
    system_entities.delete_by_id.assert_called_once_with(HelixJobStorageReclaim, record.id)
    first.reclaim_persistent_storage.reset_mock()
    second.reclaim_persistent_storage.reset_mock()
    reconciler._reclaim_queued_storage()
    first.reclaim_persistent_storage.assert_not_called()
    second.reclaim_persistent_storage.assert_not_called()


@pytest.mark.parametrize("gpu_pvc", ["cpu-pvc", "gpu-pvc"])
def test_reclaim_drains_kubernetes_cleanup_workloads_for_shared_and_distinct_claims(
    backend_registry, mock_nemo_client, gpu_pvc
):
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    batch = MagicMock()
    workloads = {}
    batch.list_namespaced_job.side_effect = lambda **kwargs: SimpleNamespace(items=list(workloads.values()))

    def create(*, namespace, body):
        workloads[body.metadata.name] = body

    batch.create_namespaced_job.side_effect = create
    batch.delete_namespaced_job.side_effect = lambda *, name, **kwargs: workloads.pop(name)

    def backend(pvc_name):
        return SimpleNamespace(
            reclaim_persistent_storage=partial(
                reclaim_job_persistent_storage,
                namespace="jobs",
                batch_v1=batch,
                pvc_name=pvc_name,
                permissions_image="permissions:latest",
                execution_backend="kubernetes_job",
                execution_profile="default",
                job_metadata=KubernetesObjectMetadata(),
                pod_metadata=KubernetesObjectMetadata(),
            )
        )

    backends = {"cpu": backend("cpu-pvc"), "gpu": backend(gpu_pvc)}
    reconciler._backend_registry = MagicMock()
    reconciler._backend_registry.get_backend.side_effect = lambda *, provider, profile: backends[provider]
    # The backend contract takes workspace and job positionally, unlike the Kubernetes helper.
    for configured in backends.values():
        reclaim = configured.reclaim_persistent_storage
        configured.reclaim_persistent_storage = lambda workspace, job, reclaim=reclaim: reclaim(
            workspace=workspace, job_id=job
        )
    record = HelixJobStorageReclaim(
        name=HelixJobStorageReclaim.name_for_job("deleted-job"),
        workspace="system",
        job_id="deleted-job",
        job_workspace="team",
        job_name="train",
        targets=[JobStorageTarget(provider=provider, profile="default") for provider in backends],
    )
    stored = {record.name: record}
    system_entities.list.side_effect = lambda *args, **kwargs: SimpleNamespace(data=list(stored.values()))

    def persist(entity):
        stored[entity.name] = entity
        return entity

    system_entities.update.side_effect = persist
    system_entities.delete_by_id.side_effect = lambda *args: stored.clear()

    for _ in range(3):
        reconciler._reclaim_queued_storage()
        for workload in workloads.values():
            workload.status = kubernetes_client.V1JobStatus(
                conditions=[kubernetes_client.V1JobCondition(type="Complete", status="True")]
            )

    assert not stored
    assert not workloads
    claims = {
        call.kwargs["body"].spec.template.spec.volumes[0].persistent_volume_claim.claim_name
        for call in batch.create_namespaced_job.call_args_list
    }
    # Every storage claim must be reclaimed, but shared claims may be deduplicated.
    assert claims == {"cpu-pvc", gpu_pvc}
    batch.create_namespaced_job.reset_mock()
    batch.delete_namespaced_job.reset_mock()
    reconciler._reclaim_queued_storage()
    batch.create_namespaced_job.assert_not_called()
    batch.delete_namespaced_job.assert_not_called()


def test_reclaim_stops_if_completed_target_cannot_be_persisted(backend_registry, mock_nemo_client):
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    record = HelixJobStorageReclaim(
        name=HelixJobStorageReclaim.name_for_job("deleted-job"),
        workspace="system",
        job_id="deleted-job",
        job_workspace="team",
        job_name="train",
        targets=[
            JobStorageTarget(provider="cpu", profile="default"),
            JobStorageTarget(provider="gpu", profile="default"),
        ],
    )
    system_entities.list.return_value = SimpleNamespace(data=[record])
    system_entities.update.side_effect = EntityConflictError("concurrent update")
    backend = MagicMock()
    reconciler._backend_registry = MagicMock()
    reconciler._backend_registry.get_backend.return_value = backend

    reconciler._reclaim_queued_storage()

    backend.reclaim_persistent_storage.assert_called_once_with("team", "train")
    assert len(record.targets) == 2
    system_entities.delete_by_id.assert_not_called()


def _run_retention(reconciler, mock_jobs_client, step, job):
    mock_jobs_client.list_steps.side_effect = [paginated_response([step])]
    mock_jobs_client.get_job.return_value = data_response(job)
    reconciler.sweep_stopped_steps()


def test_storage_retention_leaves_a_step_replaced_by_a_rerun_alone(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_error: HelixJobStepWithContext,
):
    """A failed attempt's step is neither reclaimed nor marked once a rerun replaced it.

    Step names repeat across attempts, so a status update by name would land on the rerun's step.
    """
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    replaced = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(days=1))

    _run_retention(reconciler, mock_jobs_client, replaced, _job(attempt_id="rerun-attempt"))

    mock_jobs_client.update_job_attempt_status_details.assert_not_called()
    system_entities.create.assert_not_called()
    mock_jobs_client.update_job_step_status.assert_not_called()
    mock_jobs_client.cancel_job.assert_not_called()


def test_storage_retention_marks_the_job_before_queueing_its_storage(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_paused: HelixJobStepWithContext,
):
    """The mark makes a resume refuse the job, so it is written before the storage is handed off."""
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    calls = MagicMock()
    mock_jobs_client.update_job_attempt_status_details.side_effect = calls.mark
    system_entities.create.side_effect = calls.queue
    mock_jobs_client.cancel_job.side_effect = calls.cancel
    expired = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(days=8))

    _run_retention(reconciler, mock_jobs_client, expired, _job())

    assert [call[0] for call in calls.mock_calls] == ["mark", "queue", "cancel"]


def test_storage_retention_keeps_the_storage_of_a_job_resumed_meanwhile(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_paused: HelixJobStepWithContext,
):
    """The jobs API refuses the mark when a resume or rerun took the job after the step was listed."""
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    request = httpx.Request("PATCH", "http://test")
    mock_jobs_client.update_job_attempt_status_details.side_effect = NemoHTTPError(httpx.Response(409, request=request))
    expired = _aged_step(test_step_paused, HelixJobStatus.PAUSED, datetime.timedelta(days=8))

    _run_retention(reconciler, mock_jobs_client, expired, _job())

    mock_jobs_client.update_job_attempt_status_details.assert_called_once()
    system_entities.create.assert_not_called()
    mock_jobs_client.cancel_job.assert_not_called()
    mock_jobs_client.update_job_step_status.assert_not_called()


def test_storage_retention_queues_a_marked_job_again(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_error: HelixJobStepWithContext,
):
    """An earlier pass may have marked the job and stopped before queueing its storage."""
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    system_entities.create.side_effect = EntityConflictError("already queued")
    marked = _job(status_details={"storage_reclaimed_at": "2026-10-01T00:00:00+00:00"})
    expired = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(days=1))

    _run_retention(reconciler, mock_jobs_client, expired, marked)

    mock_jobs_client.update_job_attempt_status_details.assert_not_called()
    system_entities.create.assert_called_once()
    mock_jobs_client.update_job_step_status.assert_called_once()


@pytest.mark.parametrize("outcome", [False, JobStorageError("cleanup failed")])
def test_queued_storage_stays_queued_until_it_is_gone(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    outcome,
):
    """A backend still removing the storage, or failing to, keeps the record, which holds the job name."""
    reconciler, system_entities = _reconciler(backend_registry, mock_nemo_client)
    record = HelixJobStorageReclaim(
        name=HelixJobStorageReclaim.name_for_job("job-123"),
        workspace="system",
        job_id="job-123",
        job_workspace="team",
        job_name="train-llama",
        targets=[JobStorageTarget(provider="subprocess", profile="default")],
    )
    system_entities.list.return_value = SimpleNamespace(data=[record])
    mock_jobs_client.list_steps.side_effect = lambda **_: paginated_response([])
    backend = reconciler._backend_registry.get_backend(provider="subprocess", profile="default")

    with patch.object(backend, "reclaim_persistent_storage", side_effect=[outcome]):
        reconciler.step()

    system_entities.delete_by_id.assert_not_called()


def test_sweep_reports_progress_for_each_listed_step_and_each_swept_step(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_error: HelixJobStepWithContext,
):
    """A sweep that is working must keep reporting progress, or the loop looks stalled."""
    reconciler, _ = _reconciler(backend_registry, mock_nemo_client)
    expired = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(days=1))
    swept = expired.model_copy(deep=True)
    swept.status_details = {**swept.status_details, "storage_swept_at": "2026-10-01T00:00:00+00:00"}
    mock_jobs_client.list_steps.side_effect = [paginated_response([expired, swept, swept])]
    mock_jobs_client.get_job.return_value = data_response(_job(uses_storage=False))
    on_progress = MagicMock()

    reconciler.sweep_stopped_steps(on_progress=on_progress)

    # Three steps listed, and the one that was not yet swept.
    assert on_progress.call_count == 4


def test_sweep_stops_when_the_controller_is_shutting_down(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_error: HelixJobStepWithContext,
):
    stop_signal = threading.Event()
    stop_signal.set()
    reconciler = JobReconciler(backend_registry, mock_nemo_client, stop_signal=stop_signal)
    expired = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(days=1))
    mock_jobs_client.list_steps.side_effect = [paginated_response([expired])]

    reconciler.sweep_stopped_steps()

    mock_jobs_client.get_job.assert_not_called()


def test_sweep_writes_hold_the_step_lock_but_its_reads_do_not(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
    test_step_active: HelixJobStepWithContext,
    test_step_error: HelixJobStepWithContext,
):
    """The sweep runs on its own thread: its writes are ordered against reconciling, its reads never block it."""
    reconciler, _ = _reconciler(backend_registry, mock_nemo_client)
    held_during: list[str] = []

    def lookup(**_):
        held_during.append("locked lookup" if reconciler._step_lock.locked() else "lookup")
        return MagicMock(data=lambda: MagicMock(attempt_id=None, platform_spec={}, status_details={}))

    def mark(step, status=None):
        held_during.append("mark" if reconciler._step_lock.locked() else "unlocked mark")

    def sync(step):
        held_during.append("reconcile" if reconciler._step_lock.locked() else "unlocked reconcile")
        return JobUpdate(status=HelixJobStatus.ACTIVE)

    active = _aged_step(test_step_active, HelixJobStatus.ACTIVE, datetime.timedelta(seconds=1))
    expired = _aged_step(test_step_error, HelixJobStatus.ERROR, datetime.timedelta(days=1))
    backend = reconciler._backend_registry.get_backend(provider="subprocess", profile="default")
    mock_jobs_client.get_job.side_effect = lookup
    with (
        patch.object(reconciler, "_mark_step_swept", side_effect=mark),
        patch("nhx.core.jobs.controllers.reconciler.storage_targets", return_value=[]),
        patch.object(backend, "sync", sync),
    ):
        mock_jobs_client.list_steps.side_effect = [paginated_response([active]), paginated_response([])]
        reconciler.step()
        mock_jobs_client.list_steps.side_effect = [paginated_response([expired])]
        reconciler.sweep_stopped_steps()

    assert held_during == ["reconcile", "lookup", "mark"]


def test_sweeper_runs_the_sweep_and_reports_progress_through_its_own_heartbeat(
    backend_registry: BackendRegistry,
    mock_nemo_client,
):
    reconciler, _ = _reconciler(backend_registry, mock_nemo_client)
    sweeper = StorageRetentionSweeper(reconciler)
    sweeper.emit_heartbeat = MagicMock()
    with patch.object(reconciler, "sweep_stopped_steps") as sweep:
        sweeper.step()

    sweep.assert_called_once_with(on_progress=sweeper.emit_heartbeat)
    assert sweeper.is_healthy


def test_reconciler_loop_stays_healthy_while_a_slow_sweep_runs(
    backend_registry: BackendRegistry,
    mock_nemo_client,
    mock_jobs_client,
):
    """A sweep far longer than the liveness window must not stall the reconciler or flag it unhealthy."""
    reconciler, _ = _reconciler(backend_registry, mock_nemo_client)
    mock_jobs_client.list_steps.side_effect = lambda **_: paginated_response([])
    stop_signal = threading.Event()
    sweep_started = threading.Event()
    release_sweep = threading.Event()

    def slow_sweep(on_progress=None):
        sweep_started.set()
        release_sweep.wait(timeout=10)

    interval = 0.1  # the loop is unhealthy after 3 intervals without progress
    reconciler_loop = Loop(
        TimedLoopWaiter(interval, stop_signal=stop_signal), TrackLastExecutionTime(reconciler), stop_signal=stop_signal
    )
    sweeper_loop = Loop(
        TimedLoopWaiter(60, stop_signal=stop_signal),
        TrackLastExecutionTime(StorageRetentionSweeper(reconciler)),
        stop_signal=stop_signal,
    )
    with patch.object(reconciler, "sweep_stopped_steps", side_effect=slow_sweep):
        reconciler_loop.start()
        sweeper_loop.start()
        try:
            assert sweep_started.wait(timeout=5)
            # Several liveness windows pass while the sweep is still running.
            time.sleep(10 * 3 * interval)
            assert not release_sweep.is_set()
            assert reconciler_loop.is_healthy, reconciler_loop.unhealthy_reason
            assert sweeper_loop.is_healthy, sweeper_loop.unhealthy_reason
        finally:
            release_sweep.set()
            stop_signal.set()
            reconciler_loop.join(timeout=5)
            sweeper_loop.join(timeout=5)
