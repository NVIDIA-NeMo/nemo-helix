# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import threading
import time
from collections.abc import Callable
from typing import cast

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError
from nemo_helix_plugin.entities.client import EntitiesClient
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import (
    HelixJobStatusUpdateRequest,
    HelixJobStepWithContext,
    JobStatusDetailsQueryParams,
    JobStatusDetailsUpdate,
    ListStepsQueryParams,
)
from nhx.common.controller import Controller, HeartbeatMixin
from nhx.common.entities import SYSTEM_WORKSPACE, EntityConflictError, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.common.observability import scoped_app_ctx, start_span_with_ctx
from nhx.core.jobs.app.constants import (
    JOB_TASK_CLEANUP_PENDING_KEY,
    STOPPED_AT,
    STORAGE_RECLAIMED_AT,
    STORAGE_SWEPT_AT,
)
from nhx.core.jobs.app.ctx import JobBackendContext, JobContext
from nhx.core.jobs.app.lifecycle import (
    effective_pause_ttl_seconds,
    parse_timestamp,
    storage_targets,
    storage_window_expired,
)
from nhx.core.jobs.config import JobsStorageConfig
from nhx.core.jobs.config import config as jobs_config
from nhx.core.jobs.controllers.backends import extract_provider_profile
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.diagnostics import log_job_diagnostics_if_debug
from nhx.core.jobs.entities import HelixJobStorageReclaim, JobStorageTarget
from opentelemetry import metrics, trace

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
logger = logging.getLogger(__name__)


class JobReconciler(HeartbeatMixin, Controller):
    def __init__(
        self,
        backend_registry: BackendRegistry,
        nemo_client: NemoClient,
        stop_signal: threading.Event | None = None,
        storage_config: JobsStorageConfig | None = None,
        system_entity_client: SyncEntityClient | None = None,
    ) -> None:
        self._backend_registry = backend_registry
        self._nemo_client = nemo_client
        self._storage_config = storage_config or jobs_config.storage
        # Typed Jobs client sharing the platform client's transport; every call passes
        # ``workspace=`` explicitly (incl. cross-workspace "-"), so the client's
        # default workspace is never relied upon.
        self._jobs = JobsClient.from_client(nemo_client)
        self._stop_signal = stop_signal
        self._is_healthy = False
        self._logger = logger
        self._last_retention_at: float | None = None
        # Storage reclaim records outlive the jobs they describe, so they are read from the
        # entity store as the jobs service rather than through the jobs API.
        self._system_entities = system_entity_client or SyncEntityClient(
            EntitiesClient.from_client(nemo_client)
        ).as_service("jobs", internal=True)

        self._step_reconciliation_total = meter.create_counter(
            name="nhx.jobs.reconciler.step.reconciliation.total",
            description="Total number of job reconciliation attempts",
        )
        self._step_reconciliation_errors = meter.create_counter(
            name="nhx.jobs.reconciler.step.reconciliation.errors",
            description="Number of job reconciliation errors",
        )

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def step(self):
        # Check stop signal before making any API calls
        if self._stop_signal and self._stop_signal.is_set():
            logger.debug("Stop signal received, skipping reconciliation step")
            return

        # Keep fetch, per-step reconciliation, cleanup, and their logs in one trace.
        with tracer.start_as_current_span("jobs_reconciler/step"):
            self._reconcile()

    def _reconcile(self) -> None:
        fetch_started_at = time.monotonic()
        with tracer.start_as_current_span("jobs_reconciler/fetch_steps_for_reconciliation"):
            try:
                statuses: list[str] = [
                    HelixJobStatus.PENDING.value,
                    HelixJobStatus.ACTIVE.value,
                    HelixJobStatus.CANCELLING.value,
                    HelixJobStatus.PAUSING.value,
                ]
                steps_to_reconcile = self.get_steps_for_reconciliation(statuses)
                self._is_healthy = True
                self.emit_heartbeat()
            except NemoClientError:
                self._is_healthy = False
                logger.exception("Could not fetch job steps for reconciliation", exc_info=True)
                return

        with tracer.start_as_current_span("jobs_reconciler/fetch_summary") as span:
            span.set_attribute("count", len(steps_to_reconcile))
            if len(steps_to_reconcile) > 0:
                logger.info(f"Got {len(steps_to_reconcile)} job steps to reconcile")
            else:
                logger.debug("No job steps to reconcile")
            logger.debug(
                "Reconciler fetched job steps",
                extra={"count": len(steps_to_reconcile), "duration_seconds": time.monotonic() - fetch_started_at},
            )
        for step in steps_to_reconcile:
            with start_span_with_ctx(
                tracer,
                "jobs_reconciler/reconcile_job_step",
                JobContext(id=step.job, step_name=step.name),
            ):
                try:
                    provider, profile = extract_provider_profile(step)
                    if step.status in (HelixJobStatus.CANCELLED, HelixJobStatus.ERROR):
                        self._update_step_status_with_timing(
                            step=step, provider=provider, profile=profile, status=step.status
                        )
                        continue
                    backend = self._backend_registry.get_backend(provider=provider, profile=profile)
                    self._step_reconciliation_total.add(1)
                    with scoped_app_ctx(
                        JobBackendContext(provider=provider, profile=profile, name=str(backend)),
                    ):
                        sync_started_at = time.monotonic()
                        job_update = backend.sync(step)
                        logger.debug(
                            "Reconciler backend sync completed",
                            extra={
                                "job": step.job,
                                "step": step.name,
                                "workspace": step.workspace,
                                "provider": provider,
                                "profile": profile,
                                "from_status": step.status,
                                "to_status": job_update.status,
                                "duration_seconds": time.monotonic() - sync_started_at,
                            },
                        )
                        logger.info(f"Updating job step status from '{step.status}' to '{job_update.status}'")
                        if job_update.status == HelixJobStatus.ERROR and step.status != HelixJobStatus.ERROR:
                            log_job_diagnostics_if_debug(
                                self._nemo_client,
                                step,
                                logger=self._logger,
                                context="step transitioned to error during reconciliation",
                            )
                        self._update_step_status_with_timing(
                            step=step,
                            provider=provider,
                            profile=profile,
                            status=job_update.status,
                            status_details=job_update.status_details,
                            error_details=job_update.error_details,
                        )
                except NemoHTTPError as e:
                    # In cases when attempting to update job step status results in a conflict (409),
                    # log a warning and continue processing other steps.
                    if e.status_code == 409:
                        logger.warning(f"Conflict updating job step': {str(e)}")
                        self._step_reconciliation_errors.add(
                            1,
                            attributes={
                                "error_type": "api_status_conflict",
                            },
                        )
                    else:
                        logger.error(f"Could not reconcile job step': {str(e)}")
                        self._step_reconciliation_errors.add(
                            1,
                            attributes={
                                "error_type": "api_status_error",
                            },
                        )
                except Exception:
                    logger.exception("Unexpected error when reconciling job step")
                    log_job_diagnostics_if_debug(
                        self._nemo_client,
                        step,
                        logger=self._logger,
                        context="unexpected reconciliation error",
                    )
                    self._step_reconciliation_errors.add(
                        1,
                        attributes={
                            "error_type": "unknown",
                        },
                    )
                finally:
                    # Working through the queue is progress whether or not an
                    # individual step could be reconciled.
                    self.emit_heartbeat()

        # Perform any necessary cleanup steps for expired jobs
        with tracer.start_as_current_span("jobs_reconciler/cleanup_summary"):
            logger.debug("Running job cleanup for expired job steps")
        for backend in self._backend_registry.get_all_backends():
            with tracer.start_as_current_span("jobs_reconciler/cleanup_backend_steps"):
                try:
                    backend.cleanup_steps()
                except Exception:
                    logger.exception("Could not complete cleanup steps for backend", exc_info=True)
                finally:
                    self.emit_heartbeat()

        with tracer.start_as_current_span("jobs_reconciler/enforce_storage_retention"):
            self._enforce_storage_retention()

    def _enforce_storage_retention(self) -> None:
        """Reclaim queued storage every tick, and queue stopped jobs' storage on the retention interval."""
        if self._stop_signal and self._stop_signal.is_set():
            return
        with tracer.start_as_current_span("jobs_reconciler/reclaim_queued_storage"):
            self._reclaim_queued_storage()

        # Listing every stopped step is the expensive part, so it runs on a slower interval.
        now = time.monotonic()
        interval = self._storage_config.retention_interval_seconds
        if self._last_retention_at is not None and now - self._last_retention_at < interval:
            return
        self._last_retention_at = now
        try:
            steps = self._list_steps_with_status(
                [HelixJobStatus.PAUSED.value, HelixJobStatus.ERROR.value, HelixJobStatus.CANCELLED.value],
                on_progress=self.emit_heartbeat,
            )
        except NemoClientError:
            logger.exception("Could not fetch paused and failed steps for storage retention")
            return

        for step in steps:
            if (step.status_details or {}).get(STORAGE_SWEPT_AT):
                continue
            try:
                self._retain_step_storage(step)
            except Exception:
                logger.exception(
                    "Could not enforce storage retention",
                    extra={"job": step.job, "step": step.name, "workspace": step.workspace},
                )
            finally:
                self.emit_heartbeat()

    def _reclaim_queued_storage(self) -> None:
        """Reclaim the storage of deleted and stopped jobs, and clear each record once it is gone.

        Runs every tick: a deleted job's record holds its name, and some backends remove storage
        in the background, so a record stays until a later tick finds the removal done.
        """
        try:
            # One page per tick. Anything beyond it is picked up on the next tick.
            reclaims = list(self._system_entities.list(HelixJobStorageReclaim, workspace=SYSTEM_WORKSPACE).data)
        except Exception:
            logger.exception("Could not list job storage to reclaim")
            return
        for reclaim in reclaims:
            extra = {"job": reclaim.job_name, "workspace": reclaim.job_workspace}
            try:
                if not self._reclaim_storage(reclaim):
                    continue
                self._system_entities.delete_by_id(HelixJobStorageReclaim, reclaim.id)
                logger.info("Reclaimed job storage", extra=extra)
            except Exception:
                logger.exception("Could not reclaim job storage", extra=extra)
            finally:
                self.emit_heartbeat()

    def _reclaim_storage(self, reclaim: HelixJobStorageReclaim) -> bool:
        """Persist each completed target before moving on to another one.

        Backends may consume their completed cleanup workload when returning True. Calling
        them again would start a new cleanup, so only unfinished targets may run on later ticks.
        This also lets provider aliases sharing a cleanup workload make forward progress.
        """
        for target in reclaim.targets:
            try:
                backend = self._backend_registry.get_backend(provider=target.provider, profile=target.profile)
            except KeyError:
                # The profile is no longer configured, so nothing here can reach its storage.
                logger.warning(
                    "Skipping storage reclaim for an execution profile this controller does not have",
                    extra={
                        "job": reclaim.job_name,
                        "workspace": reclaim.job_workspace,
                        "provider": target.provider,
                        "profile": target.profile,
                    },
                )
            else:
                if not backend.reclaim_persistent_storage(reclaim.job_workspace, reclaim.job_name):
                    continue
            reclaim = self._system_entities.update(
                reclaim.model_copy(update={"targets": [pending for pending in reclaim.targets if pending != target]})
            )
        return not reclaim.targets

    def _retain_step_storage(self, step: HelixJobStepWithContext) -> None:
        job = self._jobs.get_job(name=step.job, workspace=step.workspace).data()
        # An older attempt's stopped step must not reclaim storage from under a rerun. It is not
        # marked either: step names repeat across attempts, so the update could land on the rerun's.
        current_attempt_id = getattr(job, "attempt_id", None)
        if current_attempt_id is not None and step.attempt_id != current_attempt_id:
            return
        targets = storage_targets(job.platform_spec)
        if not targets:
            self._mark_step_swept(step)
            return
        # A job already marked is queued again: an earlier pass may have marked it and stopped there.
        if not (job.status_details or {}).get(STORAGE_RECLAIMED_AT):
            if step.status == HelixJobStatus.CANCELLED:
                pass
            elif step.status == HelixJobStatus.PAUSED:
                ttl_seconds = effective_pause_ttl_seconds(job.control, self._storage_config)
                if not storage_window_expired(_storage_clock(step), ttl_seconds):
                    return
            elif step.status == HelixJobStatus.ERROR:
                ttl_seconds = self._storage_config.failed_storage_ttl_seconds
                if not storage_window_expired(_storage_clock(step), ttl_seconds):
                    return
            else:
                return
            if not self._mark_storage_reclaimed(step):
                return

        self._queue_storage_reclaim(job.id, step, targets)
        swept_status = step.status
        if step.status == HelixJobStatus.PAUSED:
            self._jobs.cancel_job(name=step.job, workspace=step.workspace)
            # Cancel moves the paused step to cancelling, and the sweep mark must not move it back.
            swept_status = HelixJobStatus.CANCELLING
        self._mark_step_swept(step, swept_status)

    def _mark_storage_reclaimed(self, step: HelixJobStepWithContext) -> bool:
        """Mark the job's storage reclaimed before reclaiming it. Return False if the job moved on.

        The jobs API writes the mark on the step's attempt only while it is the job's current
        attempt and the step has the status it was listed with. It checks under the job's lock,
        as resume and rerun do, so either they refuse the marked job, or one of them took the
        job first and the mark is refused.
        """
        try:
            self._jobs.update_job_attempt_status_details(
                job=step.job,
                attempt=step.attempt_id,
                workspace=step.workspace,
                body=JobStatusDetailsUpdate(
                    {STORAGE_RECLAIMED_AT: datetime.datetime.now(datetime.timezone.utc).isoformat()}
                ),
                query_params=JobStatusDetailsQueryParams(
                    expected_step_id=step.id, expected_step_status=HelixJobStatus(step.status).value
                ),
            )
        except NemoHTTPError as exc:
            if exc.status_code != 409:
                raise
            logger.info(
                "Kept the storage of a job that moved on",
                extra={"job": step.job, "step": step.name, "workspace": step.workspace},
            )
            return False
        return True

    def _queue_storage_reclaim(
        self, job_id: str, step: HelixJobStepWithContext, targets: list[JobStorageTarget]
    ) -> None:
        """Hand the job's storage to the reclaim pass, which runs every tick until the storage is gone."""
        try:
            self._system_entities.create(
                HelixJobStorageReclaim(
                    name=HelixJobStorageReclaim.name_for_job(job_id),
                    workspace=SYSTEM_WORKSPACE,
                    job_id=job_id,
                    job_workspace=step.workspace,
                    job_name=step.job,
                    targets=targets,
                )
            )
        except EntityConflictError:
            pass  # Already queued.

    def _mark_step_swept(self, step: HelixJobStepWithContext, status: HelixJobStatus | None = None) -> None:
        """Record on the step that the sweep is done with it, so later passes skip it without a lookup.

        ``status`` is the step's status now, when the sweep changed it since the step was listed.
        """
        self._jobs.update_job_step_status(
            name=step.name,
            workspace=step.workspace,
            job=step.job,
            body=HelixJobStatusUpdateRequest(
                status=status or step.status,
                status_details={STORAGE_SWEPT_AT: datetime.datetime.now(datetime.timezone.utc).isoformat()},
                # Rejected if a rerun has replaced the step since it was listed.
                expected_step_id=step.id,
            ),
        )

    def _update_step_status_with_timing(
        self,
        *,
        step: HelixJobStepWithContext,
        provider: str,
        profile: str,
        status: HelixJobStatus,
        status_details: dict | None = None,
        error_details: dict | None = None,
    ):
        started_at = time.monotonic()
        update_fields: dict = {"status": status, "expected_step_id": step.id}
        if status_details is not None:
            update_fields["status_details"] = status_details
        if error_details is not None:
            update_fields["error_details"] = error_details
        try:
            response = self._jobs.update_attempt_step_status(
                name=step.name,
                workspace=step.workspace,
                job=step.job,
                attempt=step.attempt_id,
                body=HelixJobStatusUpdateRequest(**update_fields),
            ).data()
        except Exception:
            logger.warning(
                "Reconciler step status update failed",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "workspace": step.workspace,
                    "provider": provider,
                    "profile": profile,
                    "from_status": step.status,
                    "to_status": status,
                    "duration_seconds": time.monotonic() - started_at,
                },
            )
            raise
        logger.debug(
            "Reconciler step status update succeeded",
            extra={
                "job": step.job,
                "step": step.name,
                "workspace": step.workspace,
                "provider": provider,
                "profile": profile,
                "from_status": step.status,
                "to_status": status,
                "duration_seconds": time.monotonic() - started_at,
            },
        )
        return response

    def get_steps_for_reconciliation(self, statuses: list[str]) -> list[HelixJobStepWithContext]:
        """
        Return the list of steps to reconcile.
        """
        steps = self._list_steps_with_status(statuses, on_progress=self.emit_heartbeat)
        cleanup_query = cast(
            ListStepsQueryParams,
            {
                "filter[status]": "cancelled,error",
                "filter[" + JOB_TASK_CLEANUP_PENDING_KEY + "]": "true",
                "sort": "updated_at",
            },
        )
        try:
            for step in self._jobs.list_steps(name="-", workspace="-", query_params=cleanup_query).items():
                steps.append(step)
        except NemoClientError:
            logger.exception("Could not fetch terminal steps pending task cleanup")
            self._step_reconciliation_errors.add(1, attributes={"error_type": "task_cleanup_fetch"})
        return steps

    def _list_steps_with_status(
        self, statuses: list[str], on_progress: Callable[[], None] | None = None
    ) -> list[HelixJobStepWithContext]:
        """Return every step, across jobs and workspaces, in one of ``statuses``.

        ``on_progress`` is called for each step received. Pages arrive one request at a time, so a
        long listing keeps reporting progress instead of looking stalled.
        """
        # Iterate through all pages to get all steps.
        # deepObject query param: sent as ``filter[status]=pending,active,...``
        # (comma form), which the steps-list route splits back into a list. The
        # bracketed key isn't expressible as a TypedDict field, so cast the dict.
        query = cast(
            ListStepsQueryParams,
            {
                "filter[status]": ",".join(statuses),
                "sort": "updated_at",
            },
        )
        steps: list[HelixJobStepWithContext] = []
        for step in self._jobs.list_steps(
            name="-",  # Use "-" to indicate all jobs
            workspace="-",  # Cross-workspace query
            query_params=query,
        ).items():
            steps.append(step)
            if on_progress is not None:
                on_progress()
        return steps


def _storage_clock(step: HelixJobStepWithContext) -> datetime.datetime | None:
    """Prefer ``stopped_at``. Fall back to ``updated_at`` for steps stopped before it was written."""
    stopped_at = parse_timestamp((step.status_details or {}).get(STOPPED_AT))
    if stopped_at is not None:
        return stopped_at
    return step.updated_at
