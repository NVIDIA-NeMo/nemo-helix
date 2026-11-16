# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
import traceback
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import cast

from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import (
    HelixJobStatusUpdateRequest,
    HelixJobStepWithContext,
    ListStepsQueryParams,
)
from nhx.common.controller import Controller, HeartbeatMixin
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.common.observability import create_observable_gauge, start_span_with_ctx
from nhx.core.jobs.app.ctx import JobBackendContext, JobContext
from nhx.core.jobs.controllers.admission import AdmissionController, AlwaysAdmit
from nhx.core.jobs.controllers.backends import JobUpdate, extract_provider_profile
from nhx.core.jobs.controllers.backends.exceptions import (
    AdmissionDeferred,
    ResourceAllocationError,
    SchedulingDeferred,
)
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.concurrency_admission import IN_FLIGHT_STATUSES, principal_of, round_robin_by_user
from nhx.core.jobs.controllers.diagnostics import log_job_diagnostics_if_debug
from opentelemetry import metrics, trace
from opentelemetry.metrics import CallbackOptions, Observation

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "default"
DEFAULT_PROVIDER = "cpu"
UNKNOWN_USER = "unknown"

# Step counts keyed by (workspace, user).
StepCounts = dict[tuple[str, str], int]


def _count_by_workspace_and_user(steps: Iterable[HelixJobStepWithContext]) -> StepCounts:
    """Count steps per (workspace, user); steps without a principal count under ``unknown``."""
    counts: Counter[tuple[str, str]] = Counter()
    for step in steps:
        counts[(step.workspace, principal_of(step) or UNKNOWN_USER)] += 1
    return dict(counts)


class JobScheduler(HeartbeatMixin, Controller):
    def __init__(
        self,
        backend_registry: BackendRegistry,
        nemo_client: NemoClient,
        stop_signal: threading.Event | None = None,
        admission: AdmissionController | None = None,
    ) -> None:
        self._backend_registry = backend_registry
        self._admission = admission or AlwaysAdmit()
        self._nemo_client = nemo_client
        # Typed Jobs client sharing the platform client's transport; every call passes
        # ``workspace=`` explicitly (incl. cross-workspace "-"), so the client's
        # default workspace is never relied upon.
        self._jobs = JobsClient.from_client(nemo_client)
        self._stop_signal = stop_signal
        self._is_healthy = False
        self._logger = logger

        self._step_scheduled_total = meter.create_counter(
            name="nhx.jobs.scheduler.step.scheduled.total",
            description="Total number of job scheduling attempts",
        )
        self._step_scheduling_errors = meter.create_counter(
            name="nhx.jobs.scheduler.step.scheduling.errors",
            description="Number of job scheduling errors",
        )

        # Snapshots refreshed each scheduling pass. The gauge callbacks only read them (never call the
        # API), and each refresh replaces the whole dict so the metric reader thread sees a consistent
        # view. Keys with no steps are absent, so their series stop being reported.
        self._queued_counts: StepCounts = {}
        self._running_counts: StepCounts = {}
        create_observable_gauge(
            meter,
            subsystem="jobs.scheduler",
            name="queued",
            description="Number of job steps queued or resuming, waiting for capacity, by workspace and user",
            callbacks=[self._observe_queued],
        )
        create_observable_gauge(
            meter,
            subsystem="jobs.scheduler",
            name="running",
            description="Number of job steps currently running, by workspace and user",
            callbacks=[self._observe_running],
        )

    @property
    def is_healthy(self) -> bool:
        """Whether the last attempt to fetch steps for scheduling succeeded."""
        return self._is_healthy

    @staticmethod
    def _observations(counts: StepCounts) -> Iterator[Observation]:
        """Turn per-(workspace, user) counts into gauge observations."""
        for (workspace, user), count in counts.items():
            yield Observation(count, {"workspace": workspace, "user": user})

    def _observe_queued(self, options: CallbackOptions) -> Iterator[Observation]:
        """Gauge callback for the queued snapshot."""
        return self._observations(self._queued_counts)

    def _observe_running(self, options: CallbackOptions) -> Iterator[Observation]:
        """Gauge callback for the running snapshot."""
        return self._observations(self._running_counts)

    def _refresh_running_counts(self) -> None:
        """Recount in-flight steps for the ``running`` gauge; keep the last value if the fetch fails."""
        query = cast(
            ListStepsQueryParams,
            {"filter[status]": ",".join(s.value for s in IN_FLIGHT_STATUSES)},
        )
        try:
            steps = list(self._jobs.list_steps(name="-", workspace="-", query_params=query).items())
        except NemoClientError:
            logger.warning("Could not count running job steps for metrics", exc_info=True)
            return
        self._running_counts = _count_by_workspace_and_user(steps)

    def step(self):
        """Run one scheduling pass: fetch schedulable steps, refresh gauges, and dispatch each step."""
        # Check stop signal before making any API calls
        if self._stop_signal and self._stop_signal.is_set():
            logger.debug("Stop signal received, skipping scheduling step")
            return

        steps = []
        fetch_started_at = time.monotonic()
        with tracer.start_as_current_span("jobs_scheduler/fetch_steps_for_scheduling"):
            try:
                steps = self.get_steps_for_scheduling()
                self._is_healthy = True
                self.emit_heartbeat()
            except NemoClientError:
                self._is_healthy = False
                logger.exception("Could not fetch job steps for scheduling", exc_info=True)
                return

        self._queued_counts = _count_by_workspace_and_user(
            # Resumed steps deferred by admission stay RESUMING (they cannot move to QUEUED),
            # so count them as waiting too or they would be invisible to both gauges.
            s
            for s in steps
            if HelixJobStatus(s.status) in (HelixJobStatus.QUEUED, HelixJobStatus.RESUMING)
        )
        self._refresh_running_counts()

        if len(steps) > 0:
            logger.info(f"Got {len(steps)} job steps to schedule")
        else:
            logger.debug("No job steps to schedule")
        logger.debug(
            "Scheduler fetched job steps",
            extra={"count": len(steps), "duration_seconds": time.monotonic() - fetch_started_at},
        )
        # Snapshot current load once per pass for admission controllers that count in-flight work.
        begin_pass = getattr(self._admission, "begin_pass", None)
        if len(steps) > 0 and callable(begin_pass):
            begin_pass()
        # Round-robin across users so one user's backlog cannot starve others when capacity is scarce.
        for step in round_robin_by_user(steps):
            with start_span_with_ctx(
                tracer, "jobs_scheduler/schedule_step", JobContext(id=step.job, step_name=step.name)
            ):
                try:
                    schedule_started_at = time.monotonic()
                    try:
                        decision = self._admission.check(step)
                    except Exception:
                        # Admission reads capacity state that can fail transiently; a step that is
                        # only waiting must be retried, not failed.
                        logger.exception(
                            "Admission check failed; deferring step",
                            extra={"job": step.job, "step": step.name, "workspace": step.workspace},
                        )
                        raise AdmissionDeferred("Admission check failed; will retry")
                    if not decision.admitted:
                        raise AdmissionDeferred(decision.reason or "Waiting for capacity")
                    update = self.schedule_step(step)
                    logger.debug(
                        "Scheduled job step",
                        extra={
                            "job": step.job,
                            "step": step.name,
                            "workspace": step.workspace,
                            "duration_seconds": time.monotonic() - schedule_started_at,
                            "status": update.status,
                        },
                    )
                    try:
                        self._update_step_status_with_timing(
                            step=step,
                            phase="schedule",
                            status=update.status,
                            status_details=update.status_details,
                            error_details=update.error_details,
                        )
                    except NemoHTTPError as e:
                        # Stopgap for a scheduler/reconciler race: by the time the scheduler persists
                        # CREATED -> PENDING, another controller pass may already have advanced the
                        # step to ACTIVE (or later). In that case, treating the stale PENDING write
                        # as fatal incorrectly marks a healthy job as ERROR. The real fix is to
                        # properly serialize step state transitions so stale controller writes do not
                        # happen in the first place.
                        if self._should_ignore_conflicting_pending_update(step, update, e):
                            logger.info(
                                "Ignoring stale pending update for job step that already advanced",
                                extra={
                                    "job": step.job,
                                    "step": step.name,
                                    "workspace": step.workspace,
                                },
                            )
                            continue
                        raise

                except ResourceAllocationError as e:
                    logger.info(
                        f"Could not schedule job '{step.job}' step '{step.name}' due to resource constraints: {e.message}. Marking step as error."
                    )
                    log_job_diagnostics_if_debug(
                        self._nemo_client,
                        step,
                        logger=self._logger,
                        context="resource allocation error during scheduling",
                    )
                    self._step_scheduling_errors.add(1, attributes={"error_type": "resource_allocation"})
                    self._update_step_status_with_timing(
                        step=step,
                        phase="resource_allocation_error",
                        status=HelixJobStatus.ERROR,
                        status_details={"message": e.message},
                        error_details={"message": e.message},
                    )
                except SchedulingDeferred as e:
                    logger.debug(
                        "Scheduling deferred for job step",
                        extra={
                            "job": step.job,
                            "step": step.name,
                            "workspace": step.workspace,
                            "reason": e.message,
                        },
                    )
                    deferred_status = self._deferred_status(step, e)
                    if (
                        HelixJobStatus(step.status) == deferred_status
                        and deferred_status not in (HelixJobStatus.QUEUED, HelixJobStatus.RESUMING)
                        and (step.status_details or {}).get("message") == e.message
                    ):
                        # Already recorded on an earlier pass; skip the write so API and DB load
                        # does not grow with queue depth times poll frequency. Queued and resuming
                        # steps are excluded: re-sending lets the dispatcher retry job and attempt
                        # status propagation after a partial failure. The dispatcher treats the
                        # unchanged step write as a no-op.
                        continue
                    try:
                        self._update_step_status_with_timing(
                            step=step,
                            phase="scheduling_deferred",
                            status=deferred_status,
                            status_details={"message": e.message},
                        )
                    except Exception:
                        logger.exception(
                            "Could not persist scheduling deferral details",
                            extra={
                                "job": step.job,
                                "step": step.name,
                                "workspace": step.workspace,
                            },
                        )
                except Exception as e:
                    logger.exception("Could not schedule job step", exc_info=True)
                    log_job_diagnostics_if_debug(
                        self._nemo_client,
                        step,
                        logger=self._logger,
                        context="unexpected scheduling error",
                    )
                    self._step_scheduling_errors.add(1, attributes={"error_type": "unknown"})
                    self._update_step_status_with_timing(
                        step=step,
                        phase="unexpected_error",
                        status=HelixJobStatus.ERROR,
                        status_details={"message": str(e)},
                        error_details={"message": str(e), "error": traceback.format_exc()},
                    )
                finally:
                    # Working through the queue is progress whether or not an
                    # individual step could be scheduled.
                    self.emit_heartbeat()

    @staticmethod
    def _deferred_status(step: HelixJobStepWithContext, error: SchedulingDeferred) -> HelixJobStatus:
        """Status to persist for a deferred step.

        Admission deferrals queue the step. Backend deferrals, and steps with no valid
        transition to ``queued`` (e.g. ``resuming``), keep their current status.
        """
        current = HelixJobStatus(step.status)
        if isinstance(error, AdmissionDeferred) and current.can_transition_to(HelixJobStatus.QUEUED):
            return HelixJobStatus.QUEUED
        return current

    def _update_step_status_with_timing(
        self,
        *,
        step: HelixJobStepWithContext,
        phase: str,
        status: HelixJobStatus,
        status_details: dict[str, object] | None = None,
        error_details: dict[str, object] | None = None,
    ):
        started_at = time.monotonic()
        update_fields: dict = {"status": status}
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
                "Scheduler step status update failed",
                extra={
                    "job": step.job,
                    "step": step.name,
                    "workspace": step.workspace,
                    "phase": phase,
                    "from_status": step.status,
                    "to_status": status,
                    "duration_seconds": time.monotonic() - started_at,
                },
            )
            raise
        logger.debug(
            "Scheduler step status update succeeded",
            extra={
                "job": step.job,
                "step": step.name,
                "workspace": step.workspace,
                "phase": phase,
                "from_status": step.status,
                "to_status": status,
                "duration_seconds": time.monotonic() - started_at,
            },
        )
        return response

    def get_steps_for_scheduling(self) -> list[HelixJobStepWithContext]:
        """
        Return the oldest set of steps to schedule. We using the
        set of pending steps as our queue for what to schedule next.
        """
        # The steps-list route parses ``filter`` as a deepObject query param, so
        # the schedulable statuses are sent as a comma-joined list (comma form),
        # which the server splits back into a list.
        steps = []
        query = cast(
            ListStepsQueryParams,
            {
                "filter[status]": ",".join(
                    s.value for s in (HelixJobStatus.CREATED, HelixJobStatus.QUEUED, HelixJobStatus.RESUMING)
                ),
                "sort": "created_at",
            },
        )
        for step in self._jobs.list_steps(
            name="-",  # Use "-" to indicate all jobs
            workspace="-",  # Cross-workspace query
            query_params=query,
        ).items():
            steps.append(step)
        return steps

    def schedule_step(self, step: HelixJobStepWithContext) -> JobUpdate:
        """Hand the step to its provider/profile backend and return the resulting status update."""
        provider, profile = extract_provider_profile(step)
        backend = self._backend_registry.get_backend(profile=profile, provider=provider)
        self._step_scheduled_total.add(1, attributes={"provider": provider, "profile": profile})
        with start_span_with_ctx(
            tracer,
            "job_scheduler/schedule_step_with_backend",
            JobBackendContext(provider=provider, profile=profile, name=str(backend)),
        ):
            assert step.step_spec is not None
            return backend.schedule(step.step_spec.executor, step)

    def _should_ignore_conflicting_pending_update(
        self,
        step: HelixJobStepWithContext,
        update: JobUpdate,
        error: NemoHTTPError,
    ) -> bool:
        if error.status_code != 409 or update.status != HelixJobStatus.PENDING:
            return False

        current_step = self._jobs.get_attempt_step(
            name=step.name,
            workspace=step.workspace,
            job=step.job,
            attempt=step.attempt_id,
        ).data()
        original_status = HelixJobStatus(step.status)
        current_status = HelixJobStatus(current_step.status)
        return current_status != original_status and original_status.can_transition_to(current_status)
