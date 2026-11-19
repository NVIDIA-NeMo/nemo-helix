# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jobs-owned event publication and paging through the generic EntityClient API."""

import asyncio
import time
from datetime import datetime, timezone
from time import time as wall_time
from typing import ClassVar

from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.jobs.events import (
    ExpiredJobEventCursor,
    InvalidJobEventCursor,
    JobEventCursor,
    JobEventFilters,
    JobEventPage,
    JobEventRequest,
    JobEventType,
    JobLifecycleEvent,
)
from nhx.common.entities import (
    SYSTEM_WORKSPACE,
    EntityBase,
    EntityClient,
    EntityConflictError,
    EntityNotFoundError,
    SyncEntityClient,
)
from pydantic import BaseModel, JsonValue

RETENTION_SECONDS = 7 * 24 * 60 * 60
PUBLICATION_LEASE_SECONDS = 30
STREAM_NAME = "lifecycle-events"


class JobEventSequence(EntityBase):
    __entity_type__: ClassVar[str] = "platform_job_event_sequence"
    position: int = 0
    expired_through: int = 0
    recorded_at_epoch: float = 0
    pending_until: float | None = None


class StoredJobEvent(EntityBase):
    __entity_type__: ClassVar[str] = "platform_job_lifecycle_event"
    number: int
    recorded_at_epoch: float
    event_type: str = ""
    workspace_identity: str = ""
    event: JobLifecycleEvent | None = None


class LifecycleChange(BaseModel):
    event_type: JobEventType
    workspace_id: str
    job_id: str
    attempt_id: str | None
    step_id: str | None
    previous_state: str | None
    new_state: str | None
    occurred_at: datetime
    snapshot: dict[str, JsonValue]

    def stored(self, state: JobEventSequence) -> StoredJobEvent:
        number = state.position + 1
        event = JobLifecycleEvent(
            id=str(number),
            event_type=self.event_type,
            workspace_id=self.workspace_id,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            step_id=self.step_id,
            previous_state=self.previous_state,
            new_state=self.new_state,
            occurred_at=self.occurred_at,
            recorded_at=datetime.fromtimestamp(state.recorded_at_epoch, timezone.utc),
            snapshot=self.snapshot,
        )
        return StoredJobEvent(
            name=slot_name(number),
            workspace=SYSTEM_WORKSPACE,
            number=number,
            recorded_at_epoch=state.recorded_at_epoch,
            event_type=self.event_type.value,
            workspace_identity=self.workspace_id,
            event=event,
        )


def slot_name(number: int) -> str:
    return f"e{number:019d}"


def condition(field: str, operator: FilterOperator, value: str | int | float | list[str]) -> ComparisonOperation:
    if field == "number":
        assert isinstance(value, int)
        return ComparisonOperation(field="name", operator=operator, value=slot_name(value))
    return ComparisonOperation(field="data." + field, operator=operator, value=value)


def conjunction(*operations: FilterOperation) -> LogicalOperation:
    return LogicalOperation(operator=FilterOperator.AND, operations=list(operations))


def reservation(state: JobEventSequence) -> JobEventSequence:
    if state.position >= 2**63 - 2:
        raise RuntimeError("Lifecycle event ID space exhausted")
    now = wall_time()
    return state.model_copy(
        update={
            "pending_until": now + PUBLICATION_LEASE_SECONDS,
            "recorded_at_epoch": max(now, state.recorded_at_epoch),
        }
    )


def abandoned_slot(state: JobEventSequence) -> StoredJobEvent:
    return StoredJobEvent(
        name=slot_name(state.position + 1),
        workspace=SYSTEM_WORKSPACE,
        number=state.position + 1,
        recorded_at_epoch=state.recorded_at_epoch,
    )


class JobEventStream:
    def __init__(self, store: EntityClient):
        self.store = store

    async def state(self) -> JobEventSequence:
        try:
            return await self.store.get(JobEventSequence, STREAM_NAME, workspace=SYSTEM_WORKSPACE)
        except EntityNotFoundError:
            try:
                return await self.store.create(JobEventSequence(name=STREAM_NAME, workspace=SYSTEM_WORKSPACE))
            except EntityConflictError:
                return await self.store.get(JobEventSequence, STREAM_NAME, workspace=SYSTEM_WORKSPACE)

    async def complete(self, state: JobEventSequence, record: StoredJobEvent) -> StoredJobEvent:
        # Closing an abandoned slot creates a tombstone with the same unique name.
        # A delayed writer then conflicts instead of appearing behind an advanced cursor.
        try:
            saved = await self.store.create(record)
        except EntityConflictError:
            saved = await self.store.get(StoredJobEvent, record.name, workspace=SYSTEM_WORKSPACE)
        for _ in range(32):
            if state.position >= record.number:
                return saved
            try:
                await self.store.update(state.model_copy(update={"position": record.number, "pending_until": None}))
                return saved
            except EntityConflictError:
                state = await self.state()
        raise RuntimeError("Lifecycle publication contention")

    async def append(self, change: LifecycleChange) -> JobLifecycleEvent | None:
        while True:
            state = await self.state()
            if state.pending_until is not None:
                if state.pending_until <= wall_time():
                    await self.complete(state, abandoned_slot(state))
                else:
                    await asyncio.sleep(0.01)
                continue
            try:
                claimed = await self.store.update(reservation(state))
            except EntityConflictError:
                continue
            return (await self.complete(claimed, change.stored(claimed))).event

    async def floor(self, state: JobEventSequence, cutoff: float) -> int:
        expired = await self.store.list(
            StoredJobEvent,
            workspace=SYSTEM_WORKSPACE,
            sort="-name",
            page_size=1,
            filter_operation=conjunction(
                condition("number", FilterOperator.LTE, state.position),
                condition("recorded_at_epoch", FilterOperator.LTE, cutoff),
            ),
        )
        return max(state.expired_through, expired.data[0].number if expired.data else 0)

    async def read(self, request: JobEventRequest) -> JobEventPage:
        state = await self.state()
        if state.pending_until is not None and state.pending_until <= wall_time():
            await self.complete(state, abandoned_slot(state))
            state = await self.state()
        highwater = state.position
        cutoff = wall_time() - RETENTION_SECONDS
        floor = await self.floor(state, cutoff)
        if request.cursor is not None:
            cursor = JobEventCursor.decode(request.cursor)
            if request.filters is not None:
                cursor = cursor.model_copy(update={"filters": request.filters})
            position = int(cursor.position)
            if position > highwater + (cursor.direction == "backward"):
                raise InvalidJobEventCursor("cursor position is beyond the stream")
            if position < floor or (cursor.direction == "backward" and position <= floor + 1):
                raise ExpiredJobEventCursor("cursor history is outside the seven-day window")
        else:
            direction = request.direction or "forward"
            position = highwater + (direction == "backward") if request.start == "latest" else floor
            cursor = JobEventCursor(
                position=str(position), direction=direction, filters=request.filters or JobEventFilters()
            )
        forward = cursor.direction == "forward"
        operations: list[FilterOperation] = [
            condition("number", FilterOperator.LTE, highwater),
            condition("number", FilterOperator.GT, floor),
            condition("recorded_at_epoch", FilterOperator.GT, cutoff),
            condition("number", FilterOperator.GT if forward else FilterOperator.LT, position),
            condition(
                "event_type", FilterOperator.IN, [event.value for event in (cursor.filters.event_types or JobEventType)]
            ),
        ]
        if cursor.filters.workspace_id is not None:
            operations.append(condition("workspace_identity", FilterOperator.EQ, cursor.filters.workspace_id))
        result = await self.store.list(
            StoredJobEvent,
            workspace=SYSTEM_WORKSPACE,
            sort="name" if forward else "-name",
            filter_operation=conjunction(*operations),
            page_size=request.limit,
        )
        rows = result.data
        has_more = result.pagination.total_results > len(rows)
        current = await self.state()
        if request.cursor is not None and (
            position < current.expired_through or (not forward and position <= current.expired_through + 1)
        ):
            raise ExpiredJobEventCursor("cursor history expired during the read")
        if forward:
            next_position = rows[-1].number if rows and has_more else highwater
            prev_position = rows[0].number if rows else position + 1
            previous = (
                cursor.model_copy(update={"position": str(prev_position), "direction": "backward"}).encode()
                if prev_position > floor + 1
                else None
            )
        else:
            next_position = rows[0].number if rows else max(floor, position - 1)
            previous = (
                cursor.model_copy(update={"position": str(rows[-1].number), "direction": "backward"}).encode()
                if rows and has_more
                else None
            )
        return JobEventPage(
            data=[row.event for row in rows if row.event is not None],
            next_cursor=cursor.model_copy(update={"position": str(next_position), "direction": "forward"}).encode(),
            prev_cursor=previous,
            has_more=has_more,
        )


class SyncJobEventStream:
    """Synchronous publication and maintenance for the existing Jobs controller."""

    def __init__(self, store: SyncEntityClient):
        self.store = store

    def state(self) -> JobEventSequence:
        try:
            return self.store.get(JobEventSequence, STREAM_NAME, workspace=SYSTEM_WORKSPACE)
        except EntityNotFoundError:
            try:
                return self.store.create(JobEventSequence(name=STREAM_NAME, workspace=SYSTEM_WORKSPACE))
            except EntityConflictError:
                return self.store.get(JobEventSequence, STREAM_NAME, workspace=SYSTEM_WORKSPACE)

    def complete(self, state: JobEventSequence, record: StoredJobEvent) -> StoredJobEvent:
        try:
            saved = self.store.create(record)
        except EntityConflictError:
            saved = self.store.get(StoredJobEvent, record.name, workspace=SYSTEM_WORKSPACE)
        for _ in range(32):
            if state.position >= record.number:
                return saved
            try:
                self.store.update(state.model_copy(update={"position": record.number, "pending_until": None}))
                return saved
            except EntityConflictError:
                state = self.state()
        raise RuntimeError("Lifecycle publication contention")

    def append(self, change: LifecycleChange) -> JobLifecycleEvent | None:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = self.state()
            if state.pending_until is not None:
                if state.pending_until <= wall_time():
                    self.complete(state, abandoned_slot(state))
                else:
                    time.sleep(0.01)
                continue
            try:
                claimed = self.store.update(reservation(state))
            except EntityConflictError:
                continue
            return self.complete(claimed, change.stored(claimed)).event
        raise TimeoutError("Lifecycle publication busy")

    def cleanup(self, batch_size: int = 100) -> int:
        for _ in range(32):
            state = self.state()
            expired = self.store.list(
                StoredJobEvent,
                workspace=SYSTEM_WORKSPACE,
                sort="-name",
                page_size=1,
                filter_operation=conjunction(
                    condition("number", FilterOperator.LTE, state.position),
                    condition("recorded_at_epoch", FilterOperator.LTE, wall_time() - RETENTION_SECONDS),
                ),
            )
            floor = max(state.expired_through, expired.data[0].number if expired.data else 0)
            if floor == state.expired_through:
                break
            try:
                self.store.update(state.model_copy(update={"expired_through": floor}))
                break
            except EntityConflictError:
                continue
        else:
            raise RuntimeError("Lifecycle cleanup contention")
        rows = self.store.list(
            StoredJobEvent,
            workspace=SYSTEM_WORKSPACE,
            sort="name",
            page_size=batch_size,
            filter_operation=condition("number", FilterOperator.LTE, floor),
        ).data
        for row in rows:
            try:
                self.store.delete(StoredJobEvent, row.name, workspace=SYSTEM_WORKSPACE)
            except EntityNotFoundError:
                pass
        return len(rows)
