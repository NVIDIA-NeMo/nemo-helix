# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-owned observations and retryable terminal lifecycle publication."""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import monotonic
from typing import Literal, TypeVar

from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.jobs.events import JobEventType, JobLifecycleEvent
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nhx.common.entities import SYSTEM_WORKSPACE, EntityNotFoundError, SyncEntityClient
from nhx.common.jobs.schemas import HelixJobStatus
from nhx.core.jobs.app.event_stream import LifecycleChange, StoredJobEvent, SyncJobEventStream, condition
from nhx.core.jobs.entities import HelixJob, HelixJobAttempt, HelixJobStep
from pydantic import JsonValue

logger = logging.getLogger(__name__)
LifecycleResource = HelixJob | HelixJobAttempt | HelixJobStep


@dataclass(frozen=True)
class LifecycleImage:
    kind: Literal["job", "attempt", "step"]
    workspace_id: str
    job_id: str
    attempt_id: str | None
    step_id: str | None
    state: str
    occurred_at: datetime
    snapshot: dict[str, JsonValue]

    def change(self, previous: str | None, action: str) -> LifecycleChange:
        return LifecycleChange(
            event_type=JobEventType(self.kind + "." + action),
            workspace_id=self.workspace_id,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            step_id=self.step_id,
            previous_state=previous,
            new_state=None if action == "deleted" else self.state,
            occurred_at=self.occurred_at,
            snapshot=self.snapshot,
        )


def image(resource: LifecycleResource, job: HelixJob, workspace_id: str, job_state: HelixJobStatus) -> LifecycleImage:
    snapshot: dict[str, JsonValue] = {"name": resource.name, "source": job.source, "project": resource.project}
    if isinstance(resource, HelixJob):
        kind: Literal["job", "attempt", "step"] = "job"
        attempt_id = resource.current_attempt_id
        step_id = None
    elif isinstance(resource, HelixJobAttempt):
        kind, attempt_id, step_id = "attempt", resource.id, None
        snapshot["sequence"] = resource.seq
    else:
        kind, attempt_id, step_id = "step", resource.attempt_id, resource.id
    timestamp = resource.updated_at or resource.created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return LifecycleImage(
        kind,
        workspace_id,
        job.id,
        attempt_id,
        step_id,
        (job_state if isinstance(resource, HelixJob) else resource.status).value,
        timestamp,
        snapshot,
    )


T = TypeVar("T", bound=LifecycleResource)
TERMINAL_STATES = [state.value for state in HelixJobStatus.terminals()]


def equals(field: str, value: str | bool | None) -> ComparisonOperation:
    return ComparisonOperation(field="data." + field, operator=FilterOperator.EQ, value=value)


def terminal_filter(*conditions: FilterOperation) -> LogicalOperation:
    return LogicalOperation(
        operator=FilterOperator.AND,
        operations=[
            ComparisonOperation(field="data.status", operator=FilterOperator.IN, value=TERMINAL_STATES),
            *conditions,
        ],
    )


TerminalResource = HelixJobAttempt | HelixJobStep


@dataclass
class DeletionScan:
    batches: Iterator[Sequence[TerminalResource]]
    current: deque[TerminalResource] = field(default_factory=deque)


class JobEventPublisher:
    def __init__(self, store: SyncEntityClient, workspaces: WorkspacesClient):
        self.store = store
        self.workspaces = workspaces
        self.stream = SyncJobEventStream(store)
        self._stream_states: dict[str, HelixJobStatus] = {}
        self._job_attempt_states: dict[tuple[str, str | None], HelixJobStatus] = {}
        self._position = 0
        self._pages: dict[str, int] = {}
        self._offsets: dict[str, int] = {}
        self._deleting: dict[str, DeletionScan] = {}
        self._healthy = True

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    def _apply(self, event: JobLifecycleEvent) -> None:
        kind = event.event_type.value.split(".")[0]
        identity = event.step_id if kind == "step" else event.attempt_id if kind == "attempt" else event.job_id
        if identity is None:
            return
        if event.new_state is None:
            self._stream_states.pop(identity, None)
            if kind == "job":
                for key in list(self._job_attempt_states):
                    if key[0] == event.job_id:
                        self._job_attempt_states.pop(key)
            return
        state = HelixJobStatus(event.new_state)
        if kind == "job":
            self._job_attempt_states[event.job_id, event.attempt_id] = state
        self._stream_states[identity] = state

    def _append(self, change: LifecycleChange, on_progress: Callable[[], None]) -> None:
        self._apply(self.stream.append(change, on_progress))

    def _replay(self, on_progress: Callable[[], None]) -> bool:
        state = self.stream.recover(on_progress)
        rows = self.store.list(
            StoredJobEvent,
            workspace=SYSTEM_WORKSPACE,
            sort="name",
            page_size=100,
            filter_operation=LogicalOperation(
                operator=FilterOperator.AND,
                operations=[
                    condition("number", FilterOperator.GT, self._position),
                    condition("number", FilterOperator.LTE, state.position),
                ],
            ),
        )
        on_progress()
        for row in rows.data:
            if row.event is not None:
                self._apply(row.event)
            self._position = row.number
        return rows.pagination.total_results <= len(rows.data)

    def _announce(
        self, resource: LifecycleResource, job: HelixJob, workspace_id: str, on_progress: Callable[[], None]
    ) -> None:
        if resource.id not in self._stream_states:
            self._append(
                image(resource, job, workspace_id, HelixJobStatus.CREATED).change(None, "created"), on_progress
            )

    def _publish_pending_jobs(self, on_progress: Callable[[], None], stop_signal: threading.Event | None) -> None:
        page = self._pages.get("creation", 1)
        jobs = self.store.list(
            HelixJob,
            workspace="-",
            filter_obj={"status": HelixJobStatus.CREATION_PENDING.value},
            sort="created_at",
            page=page,
            page_size=100,
        )
        on_progress()
        for job in jobs.data:
            if stop_signal is not None and stop_signal.is_set():
                return
            try:
                self._publish_pending_job(job, on_progress)
            except Exception:
                self._healthy = False
                logger.exception("Job creation publication will retry", extra={"job": job.id})
        self._pages["creation"] = page + 1 if page < jobs.pagination.total_pages else 1

    def _publish_pending_job(self, job: HelixJob, on_progress: Callable[[], None]) -> None:
        if job.output_location is None and job.fileset_id is None:
            return
        attempt = None
        first_step = None
        if job.platform_spec.steps:
            if job.current_attempt_id is None:
                return
            attempt = self.store.get_by_id(HelixJobAttempt, job.current_attempt_id)
            on_progress()
            steps = self.store.list(
                HelixJobStep, workspace=job.workspace, filter_obj={"attempt_id": attempt.id}, page_size=1
            )
            on_progress()
            if not steps.data:
                return
            first_step = steps.data[0]
        workspace_id = self.workspaces.get_workspace(name=job.workspace).data().id
        on_progress()
        published = job.model_copy(update={"status": HelixJobStatus.CREATED})
        self._announce(published, published, workspace_id, on_progress)
        if attempt is not None and first_step is not None:
            self._announce(attempt, published, workspace_id, on_progress)
            self._announce(first_step, published, workspace_id, on_progress)
        self.store.update(published)
        on_progress()

    def _image(self, resource: LifecycleResource, on_progress: Callable[[], None]) -> LifecycleImage:
        if isinstance(resource, HelixJob):
            job = resource
            state = job.status or HelixJobStatus.CREATED
            if job.current_attempt_id and job.status not in HelixJobStatus.lifecycle_pending():
                state = self.store.get_by_id(HelixJobAttempt, job.current_attempt_id).status
                on_progress()
        elif isinstance(resource, HelixJobAttempt):
            job = self.store.get_by_id(HelixJob, resource.job)
            on_progress()
            state = resource.status
        else:
            attempt = self.store.get_by_id(HelixJobAttempt, resource.attempt_id)
            on_progress()
            job = self.store.get_by_id(HelixJob, attempt.job)
            on_progress()
            state = resource.status
        workspace = self.workspaces.get_workspace(name=resource.workspace).data()
        on_progress()
        return image(resource, job, workspace.id, state)

    def _publish_terminal(self, resource: TerminalResource, on_progress: Callable[[], None]) -> None:
        captured = self._image(resource, on_progress)
        previous = self._stream_states.get(resource.id)
        self._append(captured.change(previous.value if previous is not None else None, "state_changed"), on_progress)
        if isinstance(resource, HelixJobAttempt):
            job = self.store.get_by_id(HelixJob, resource.job)
            on_progress()
            if job.current_attempt_id == resource.id or (job.id, resource.id) in self._job_attempt_states:
                projected = replace(
                    image(job, job, captured.workspace_id, resource.status),
                    attempt_id=resource.id,
                    occurred_at=captured.occurred_at,
                )
                self._append(projected.change(None, "state_changed"), on_progress)

    def _observe_page(
        self, resource_type: type[T], on_progress: Callable[[], None], stop_signal: threading.Event | None
    ) -> None:
        key = resource_type.__entity_type__
        page = self._pages.get(key, 1)
        records = self.store.list(resource_type, workspace="-", sort="created_at", page=page, page_size=100)
        on_progress()
        deadline = monotonic() + 1
        offset = self._offsets.get(key, 0)
        for index, resource in enumerate(records.data):
            if index < offset:
                continue
            if (index > offset and monotonic() >= deadline) or (stop_signal is not None and stop_signal.is_set()):
                self._offsets[key] = index
                return
            state = resource.status or HelixJobStatus.CREATED
            if (
                state in HelixJobStatus.lifecycle_pending()
                or state.is_terminal()
                or self._stream_states.get(resource.id) == state
            ):
                continue
            try:
                captured = self._image(resource, on_progress)
                if not isinstance(resource, HelixJob) and captured.job_id not in self._stream_states:
                    continue
                previous = self._stream_states.get(resource.id)
                if previous is not None and previous.value == captured.state:
                    continue
                self._append(
                    captured.change(
                        previous.value if previous is not None else None,
                        "created" if previous is None else "state_changed",
                    ),
                    on_progress,
                )
            except Exception:
                logger.warning("Intermediate Jobs lifecycle observation was not published", exc_info=True)
        self._offsets.pop(key, None)
        self._pages[key] = page + 1 if page < records.pagination.total_pages else 1

    def publish_observations(self, on_progress: Callable[[], None], stop_signal: threading.Event | None) -> None:
        self._healthy = True
        try:
            if not self._replay(on_progress):
                return
            self._publish_pending_jobs(on_progress, stop_signal)
            for resource_type in (HelixJobStep, HelixJobAttempt):
                key = "terminal:" + resource_type.__entity_type__
                page = self._pages.get(key, 1)
                records = self.store.list(
                    resource_type,
                    workspace="-",
                    filter_operation=terminal_filter(),
                    sort="created_at",
                    page=page,
                    page_size=100,
                )
                on_progress()
                deadline = monotonic() + 1
                complete = True
                offset = self._offsets.get(key, 0)
                for index, resource in enumerate(records.data):
                    if index < offset:
                        continue
                    if (index > offset and monotonic() >= deadline) or (
                        stop_signal is not None and stop_signal.is_set()
                    ):
                        self._offsets[key] = index
                        complete = False
                        break
                    if self._stream_states.get(resource.id) == resource.status:
                        if isinstance(resource, HelixJobStep):
                            continue
                        projected = self._job_attempt_states.get((resource.job, resource.id))
                        if projected == resource.status:
                            continue
                        job = self.store.get_by_id(HelixJob, resource.job)
                        on_progress()
                        if job.current_attempt_id != resource.id and projected is None:
                            continue
                    try:
                        self._publish_terminal(resource, on_progress)
                    except Exception:
                        self._healthy = False
                        complete = False
                        self._offsets[key] = index
                        logger.exception(
                            "Terminal Jobs lifecycle publication will be retried", extra={"resource": resource.id}
                        )
                        break
                if complete:
                    self._offsets.pop(key, None)
                    self._pages[key] = page + 1 if page < records.pagination.total_pages else 1
            self._observe_page(HelixJob, on_progress, stop_signal)
            self._observe_page(HelixJobAttempt, on_progress, stop_signal)
            self._observe_page(HelixJobStep, on_progress, stop_signal)
        except Exception:
            self._healthy = False
            logger.exception("Jobs lifecycle discovery will be retried")

    def _terminal_batches(self, job: HelixJob, on_progress: Callable[[], None]) -> Iterator[Sequence[TerminalResource]]:
        page = 1
        while True:
            attempts = self.store.list(
                HelixJobAttempt,
                workspace=job.workspace,
                filter_obj={"job": job.id},
                sort="created_at",
                page=page,
                page_size=100,
            )
            on_progress()
            yield []
            for attempt in attempts.data:
                step_page = 1
                while True:
                    steps = self.store.list(
                        HelixJobStep,
                        workspace=job.workspace,
                        page=step_page,
                        page_size=100,
                        sort="created_at",
                        filter_operation=terminal_filter(equals("attempt_id", attempt.id)),
                    )
                    on_progress()
                    yield steps.data
                    if step_page >= steps.pagination.total_pages:
                        break
                    step_page += 1
                yield [attempt] if attempt.status.is_terminal() else []
            if page >= attempts.pagination.total_pages:
                return
            page += 1

    def flush_job(self, job: HelixJob, on_progress: Callable[[], None], stop_signal: threading.Event | None) -> bool:
        """Republish terminal outcomes before allowing a deleting Job's cascade."""
        if stop_signal is not None and stop_signal.is_set():
            return False
        try:
            if not self._replay(on_progress):
                return False
        except Exception:
            self._healthy = False
            logger.exception("Keeping Job until stream recovery completes", extra={"job": job.id})
            return False
        scan = self._deleting.get(job.id)
        if scan is None:
            scan = DeletionScan(self._terminal_batches(job, on_progress))
            self._deleting[job.id] = scan
        deadline = monotonic() + 1
        try:
            for _ in range(20):
                if monotonic() >= deadline or (stop_signal is not None and stop_signal.is_set()):
                    return False
                if not scan.current:
                    try:
                        scan.current.extend(next(scan.batches))
                    except StopIteration:
                        self._deleting.pop(job.id, None)
                        return True
                    if not scan.current:
                        continue
                # Cleanup announces final outcomes even when the stream already contains them.
                self._publish_terminal(scan.current[0], on_progress)
                scan.current.popleft()
            return False
        except Exception:
            self._healthy = False
            # A failed page fetch closes a generator; start a fresh scan on retry.
            # Replaying earlier outcomes is safe under at-least-once delivery.
            self._deleting.pop(job.id, None)
            logger.exception("Keeping Job until terminal events are published", extra={"job": job.id})
            return False

    def deletion_images(self, job: HelixJob, stop_signal: threading.Event | None) -> list[LifecycleImage]:
        images: list[LifecycleImage] = []
        deadline = monotonic() + 2

        def collecting() -> bool:
            return monotonic() < deadline and not (stop_signal is not None and stop_signal.is_set())

        try:
            if not collecting():
                return images
            workspace_id = self.workspaces.get_workspace(name=job.workspace).data().id
            if not collecting():
                return images
            job_state = job.status or HelixJobStatus.CREATED
            if job.current_attempt_id and job.status not in HelixJobStatus.lifecycle_pending():
                try:
                    job_state = self.store.get_by_id(HelixJobAttempt, job.current_attempt_id).status
                except EntityNotFoundError:
                    pass
            page = 1
            while collecting():
                attempts = self.store.list(
                    HelixJobAttempt, workspace=job.workspace, filter_obj={"job": job.id}, page=page, page_size=100
                )
                for attempt in attempts.data:
                    step_page = 1
                    while collecting():
                        steps = self.store.list(
                            HelixJobStep,
                            workspace=job.workspace,
                            filter_obj={"attempt_id": attempt.id},
                            page=step_page,
                            page_size=100,
                        )
                        for step in steps.data:
                            if not collecting():
                                return images
                            images.append(image(step, job, workspace_id, job_state))
                        if step_page >= steps.pagination.total_pages:
                            break
                        step_page += 1
                    if not collecting():
                        return images
                    images.append(image(attempt, job, workspace_id, job_state))
                if not collecting():
                    return images
                if page >= attempts.pagination.total_pages:
                    images.append(image(job, job, workspace_id, job_state))
                    return images
                page += 1
        except Exception:
            logger.warning("Could not capture Jobs deletion preimages", exc_info=True)
        return images

    def deleted(self, images: list[LifecycleImage], on_progress: Callable[[], None]) -> None:
        deadline = time.monotonic() + 2
        for before in images:
            if time.monotonic() >= deadline:
                logger.warning("Remaining Jobs deletion events were dropped after the recording deadline")
                break
            try:
                change = before.change(before.state, "deleted")
                change.occurred_at = datetime.now(timezone.utc)
                self.stream.append(change, on_progress)
            except Exception:
                logger.warning("Jobs lifecycle deletion event was dropped", exc_info=True)
