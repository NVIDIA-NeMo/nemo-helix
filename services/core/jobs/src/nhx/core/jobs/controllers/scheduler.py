# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
import traceback
from typing import cast

from nemo_helix import NeMoHelix
from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.client.errors import NemoClientError, NemoHTTPError
from nemo_helix_plugin.jobs.client import JobsClient
from nemo_helix_plugin.jobs.types import (
    ListStepsQueryParams,
    PlatformJobStatusUpdateRequest,
    PlatformJobStepWithContext,
)
from nhx.common.controller import Controller, HeartbeatMixin
from nhx.common.jobs.schemas import PlatformJobStatus
from nhx.common.observability import start_span_with_ctx
from nhx.core.jobs.app.ctx import JobBackendContext, JobContext
from nhx.core.jobs.controllers.backends import JobUpdate, extract_provider_profile
from nhx.core.jobs.controllers.backends.exceptions import ResourceAllocationError, SchedulingDeferred
from nhx.core.jobs.controllers.backends.registry import BackendRegistry
from nhx.core.jobs.controllers.diagnostics import log_job_diagnostics_if_debug
from opentelemetry import metrics, trace

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)
logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "default"
DEFAULT_PROVIDER = "cpu"


class JobScheduler(HeartbeatMixin, Controller):
    def __init__(
        self,
        backend_registry: BackendRegistry,
        nhx_sdk: NeMoHelix,
        stop_signal: threading.Event | None = None,
    ) -> None:
        self._backend_registry = backend_registry
        self._nhx_sdk = nhx_sdk
        # Typed Jobs client sharing the SDK's transport; every call passes
        # ``workspace=`` explicitly (incl. cross-workspace "-"), so the client's
        # default workspace is never relied upon.
        self._jobs = client_from_platform(nhx_sdk, JobsClient)
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

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def step(self):
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

        if len(steps) > 0:
            logger.info(f"Got {len(steps)} job steps to schedule")
        else:
            logger.debug("No job steps to schedule")
        logger.debug(
            "Scheduler fetched job steps",
            extra={"count": len(steps), "duration_seconds": time.monotonic() - fetch_started_at},
        )
        for step in steps:
            with start_span_with_ctx(
                tracer, "jobs_scheduler/schedule_step", JobContext(id=step.job, step_name=step.name)
            ):
                try:
                    schedule_started_at = time.monotonic()
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
                        self._nhx_sdk,
                        step,
                        logger=self._logger,
                        context="resource allocation error during scheduling",
                    )
                    self._step_scheduling_errors.add(1, attributes={"error_type": "resource_allocation"})
                    self._update_step_status_with_timing(
                        step=step,
                        phase="resource_allocation_error",
                        status=PlatformJobStatus.ERROR,
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
                    try:
                        self._update_step_status_with_timing(
                            step=step,
                            phase="scheduling_deferred",
                            status=PlatformJobStatus(step.status),
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
                        self._nhx_sdk,
                        step,
                        logger=self._logger,
                        context="unexpected scheduling error",
                    )
                    self._step_scheduling_errors.add(1, attributes={"error_type": "unknown"})
                    self._update_step_status_with_timing(
                        step=step,
                        phase="unexpected_error",
                        status=PlatformJobStatus.ERROR,
                        status_details={"message": str(e)},
                        error_details={"message": str(e), "error": traceback.format_exc()},
                    )
                finally:
                    # Working through the queue is progress whether or not an
                    # individual step could be scheduled.
                    self.emit_heartbeat()

    def _update_step_status_with_timing(
        self,
        *,
        step: PlatformJobStepWithContext,
        phase: str,
        status: PlatformJobStatus,
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
            response = self._jobs.update_job_step_status(
                name=step.name,
                workspace=step.workspace,
                job=step.job,
                body=PlatformJobStatusUpdateRequest(**update_fields),
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

    def get_steps_for_scheduling(self) -> list[PlatformJobStepWithContext]:
        """
        Return the oldest set of steps to schedule. We using the
        set of pending steps as our queue for what to schedule next.
        """
        # The steps-list route parses ``filter`` as a deepObject query param, so
        # the status list is sent as ``filter[status]=created,resuming`` (comma
        # form), which the server splits back into a list.
        steps = []
        query = cast(
            ListStepsQueryParams,
            {
                "filter[status]": f"{PlatformJobStatus.CREATED.value},{PlatformJobStatus.RESUMING.value}",
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

    def schedule_step(self, step: PlatformJobStepWithContext) -> JobUpdate:
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
        step: PlatformJobStepWithContext,
        update: JobUpdate,
        error: NemoHTTPError,
    ) -> bool:
        if error.status_code != 409 or update.status != PlatformJobStatus.PENDING:
            return False

        current_step = self._jobs.get_job_step(
            name=step.name,
            workspace=step.workspace,
            job=step.job,
        ).data()
        original_status = PlatformJobStatus(step.status)
        current_status = PlatformJobStatus(current_step.status)
        return current_status != original_status and original_status.can_transition_to(current_status)
