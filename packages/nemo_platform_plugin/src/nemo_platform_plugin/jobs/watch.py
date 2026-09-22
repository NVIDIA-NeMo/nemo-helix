# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Literal, TypeAlias

import httpx
from nemo_platform_plugin.client.errors import NemoClientError, NemoHTTPError, NemoTransportError
from nemo_platform_plugin.jobs.client import (
    AsyncJobLogsClient,
    AsyncJobStatusClient,
    AsyncJobsWatchClient,
    JobLogsClient,
    JobStatusClient,
    JobsWatchClient,
)
from nemo_platform_plugin.jobs.schemas import PlatformJobLog, PlatformJobStatusResponse
from nemo_platform_plugin.jobs.types import JobLogsQueryParams
from nemo_platform_plugin.jobs.watch_types import (
    JobLogEvent,
    JobStatusEvent,
    JobWarningEvent,
    JobWatchEvent,
    JobWatchTimeoutError,
)

_SUCCESSFUL_TERMINAL_STATUSES = {"completed"}
# A paused job is resumable, but it is no longer making progress. Treat it as
# failure-terminal so wait/watch callers do not report incomplete work as done.
_FAILED_TERMINAL_STATUSES = {"cancelled", "error", "paused"}
_TERMINAL_STATUSES = _SUCCESSFUL_TERMINAL_STATUSES | _FAILED_TERMINAL_STATUSES
_TRANSIENT_STATUS_CODES = {429, 502, 503, 504}
_TRANSPORT_ERRORS = (NemoTransportError,)
_INVALID_PAGE_CURSOR_STATUS_CODE = 422
_INVALID_PAGE_CURSOR_ERROR_CODE = "invalid_page_cursor"
_PAGE_CURSOR_DECODE_ERROR_DETAIL = "Invalid page cursor"
_LogKey: TypeAlias = tuple[str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class _SyncWatchClients:
    status: JobStatusClient
    logs: JobLogsClient | None


@dataclass(frozen=True, slots=True)
class _AsyncWatchClients:
    status: AsyncJobStatusClient
    logs: AsyncJobLogsClient | None


@dataclass(frozen=True, slots=True)
class _WatchOptions:
    workspace: str | None
    attempt_id: int | None
    step_id: str | None
    task_id: str | None
    limit: int | None


@dataclass(slots=True)
class _WatchState:
    history_seen: bool
    log_cursor: str | None
    last_status: str | None = None
    previous_drain_seen_logs: dict[_LogKey, int] = field(default_factory=dict)
    current_drain_seen_logs: dict[_LogKey, int] = field(default_factory=dict)

    def start_log_drain(self) -> None:
        if self.current_drain_seen_logs:
            self.previous_drain_seen_logs = self.current_drain_seen_logs
            self.current_drain_seen_logs = {}

    def complete_log_drain(self) -> None:
        self.previous_drain_seen_logs = self.current_drain_seen_logs
        self.current_drain_seen_logs = {}

    def seen_log_occurrences(self, key: _LogKey) -> int:
        return max(
            self.previous_drain_seen_logs.get(key, 0),
            self.current_drain_seen_logs.get(key, 0),
        )

    def record_log_occurrence(self, key: _LogKey, occurrences: int) -> None:
        self.current_drain_seen_logs[key] = occurrences


@dataclass(slots=True)
class _TerminalLogDrainRetryBudget:
    RETRY_CAP: ClassVar[int] = 3

    _failures_before_warning: Iterator[int] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._failures_before_warning = iter(range(1, self.RETRY_CAP))

    def can_retry_after_failure(self) -> bool:
        try:
            next(self._failures_before_warning)
        except StopIteration:
            return False
        return True

    def warning(self, job_name: str) -> JobWarningEvent:
        message = f"Terminal log drain retry cap reached ({self.RETRY_CAP}); stopping watch for job {job_name!r}"
        return JobWarningEvent(kind="warning", job_name=job_name, message=message)


@dataclass(slots=True)
class _TransientRetryBackoff:
    MAX_SLEEP_SECONDS: ClassVar[float] = 30.0
    MIN_SLEEP_SECONDS: ClassVar[float] = 1.0

    _next_sleep_interval: float | None = None
    _last_warning_key: tuple[Literal["status", "log"], str] | None = None

    def reset(self) -> None:
        self._next_sleep_interval = None
        self._last_warning_key = None

    def next_sleep_interval(self, poll_interval: float) -> float:
        base_interval = max(poll_interval, self.MIN_SLEEP_SECONDS)
        max_sleep = max(base_interval, self.MAX_SLEEP_SECONDS)
        sleep_interval = base_interval if self._next_sleep_interval is None else self._next_sleep_interval
        sleep_interval = min(sleep_interval, max_sleep)
        self._next_sleep_interval = min(sleep_interval * 2, max_sleep)
        return sleep_interval

    def warning(
        self,
        operation: Literal["status", "log"],
        job_name: str,
        exc: Exception,
        *,
        transient: bool = True,
    ) -> JobWarningEvent | None:
        warning = (
            _transient_failure_warning(operation, job_name, exc) if transient else _log_failure_warning(job_name, exc)
        )
        warning_key = (operation, warning.message)
        if warning_key == self._last_warning_key:
            return None
        self._last_warning_key = warning_key
        return warning


@dataclass(frozen=True, slots=True)
class _TransientRetry:
    warning: JobWarningEvent | None
    sleep_interval: float


@dataclass(frozen=True, slots=True)
class _WatchDeadline:
    job_name: str
    expires_at: float | None

    @classmethod
    def from_timeout(cls, job_name: str, timeout: float | None) -> _WatchDeadline:
        expires_at = None if timeout is None else time.monotonic() + timeout
        return cls(job_name=job_name, expires_at=expires_at)

    def raise_if_expired(self) -> None:
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise JobWatchTimeoutError(f"Timed out watching job {self.job_name!r}")

    def remaining(self) -> float | None:
        if self.expires_at is None:
            return None
        return self.expires_at - time.monotonic()

    def sleep_seconds(self, poll_interval: float) -> float:
        remaining = self.remaining()
        if remaining is None:
            return poll_interval
        if remaining <= 0:
            raise JobWatchTimeoutError(f"Timed out watching job {self.job_name!r}")
        return min(poll_interval, remaining)


def watch_job(
    client: JobsWatchClient,
    name: str,
    *,
    workspace: str | None = None,
    poll_interval: float = 3,
    timeout: float | None = None,
    include_history: bool = True,
    include_logs: bool = True,
    attempt_id: int | None = None,
    step_id: str | None = None,
    task_id: str | None = None,
    limit: int | None = None,
    page_cursor: str | None = None,
) -> Iterator[JobWatchEvent]:
    """Watch a platform job until completion.

    The iterator yields typed status, log, and warning events. It accepts a
    ``JobsClient``-compatible client from the source-owned Jobs service.
    """
    if poll_interval < 0:
        raise ValueError("poll_interval must be greater than or equal to 0")

    jobs = _SyncWatchClients(status=client, logs=client if include_logs else None)
    options = _WatchOptions(
        workspace=workspace,
        attempt_id=attempt_id,
        step_id=step_id,
        task_id=task_id,
        limit=limit,
    )
    state = _WatchState(history_seen=include_history, log_cursor=page_cursor)
    return _watch_job(
        jobs,
        name,
        options=options,
        state=state,
        deadline=_WatchDeadline.from_timeout(name, timeout),
        poll_interval=poll_interval,
    )


def async_watch_job(
    client: AsyncJobsWatchClient,
    name: str,
    *,
    workspace: str | None = None,
    poll_interval: float = 3,
    timeout: float | None = None,
    include_history: bool = True,
    include_logs: bool = True,
    attempt_id: int | None = None,
    step_id: str | None = None,
    task_id: str | None = None,
    limit: int | None = None,
    page_cursor: str | None = None,
) -> AsyncIterator[JobWatchEvent]:
    """Async variant of :func:`watch_job`.

    This has the same poll-based log pagination limitation as
    :func:`watch_job`.
    """
    if poll_interval < 0:
        raise ValueError("poll_interval must be greater than or equal to 0")

    jobs = _AsyncWatchClients(status=client, logs=client if include_logs else None)
    options = _WatchOptions(
        workspace=workspace,
        attempt_id=attempt_id,
        step_id=step_id,
        task_id=task_id,
        limit=limit,
    )
    state = _WatchState(history_seen=include_history, log_cursor=page_cursor)
    return _async_watch_job(
        jobs,
        name,
        options=options,
        state=state,
        deadline=_WatchDeadline.from_timeout(name, timeout),
        poll_interval=poll_interval,
    )


def _watch_job(
    jobs: _SyncWatchClients,
    name: str,
    *,
    options: _WatchOptions,
    state: _WatchState,
    deadline: _WatchDeadline,
    poll_interval: float,
) -> Iterator[JobWatchEvent]:
    terminal_log_drain_budget = _TerminalLogDrainRetryBudget()
    status_retry = _TransientRetryBackoff()
    log_retry = _TransientRetryBackoff()
    while True:
        deadline.raise_if_expired()
        try:
            status_response = jobs.status.get_job_status(workspace=options.workspace, name=name)
            status_event = _status_event(status_response.data(), name)
        except Exception as exc:
            if not _is_transient_failure(exc):
                raise
            retry = _transient_retry(status_retry, "status", name, exc, poll_interval)
            if retry.warning is not None:
                yield retry.warning
            _sleep(retry.sleep_interval, deadline)
            continue
        status_retry.reset()

        if status_event.status != state.last_status:
            yield status_event
            state.last_status = status_event.status

        logs_drained = True
        sleep_interval = poll_interval
        if jobs.logs is not None:
            logs_drained = False
            try:
                yield from _drain_logs_with_cursor_recovery(
                    jobs.logs,
                    name,
                    options=options,
                    state=state,
                    deadline=deadline,
                )
                logs_drained = True
                log_retry.reset()
            except Exception as exc:
                if _is_transient_failure(exc):
                    retry = _transient_retry(log_retry, "log", name, exc, poll_interval)
                elif _is_http_status_error(exc):
                    retry = _log_failure_retry(log_retry, name, exc, poll_interval)
                else:
                    raise
                if retry.warning is not None:
                    yield retry.warning
                sleep_interval = retry.sleep_interval

        if status_event.terminal:
            if logs_drained:
                return
            if not terminal_log_drain_budget.can_retry_after_failure():
                yield terminal_log_drain_budget.warning(name)
                return
        else:
            terminal_log_drain_budget.reset()

        _sleep(sleep_interval, deadline)


async def _async_watch_job(
    jobs: _AsyncWatchClients,
    name: str,
    *,
    options: _WatchOptions,
    state: _WatchState,
    deadline: _WatchDeadline,
    poll_interval: float,
) -> AsyncIterator[JobWatchEvent]:
    terminal_log_drain_budget = _TerminalLogDrainRetryBudget()
    status_retry = _TransientRetryBackoff()
    log_retry = _TransientRetryBackoff()
    while True:
        deadline.raise_if_expired()
        try:
            status_response = await jobs.status.get_job_status(workspace=options.workspace, name=name)
            status_event = _status_event(status_response.data(), name)
        except Exception as exc:
            if not _is_transient_failure(exc):
                raise
            retry = _transient_retry(status_retry, "status", name, exc, poll_interval)
            if retry.warning is not None:
                yield retry.warning
            await _async_sleep(retry.sleep_interval, deadline)
            continue
        status_retry.reset()

        if status_event.status != state.last_status:
            yield status_event
            state.last_status = status_event.status

        logs_drained = True
        sleep_interval = poll_interval
        if jobs.logs is not None:
            logs_drained = False
            try:
                async for event in _async_drain_logs_with_cursor_recovery(
                    jobs.logs,
                    name,
                    options=options,
                    state=state,
                    deadline=deadline,
                ):
                    yield event
                logs_drained = True
                log_retry.reset()
            except Exception as exc:
                if _is_transient_failure(exc):
                    retry = _transient_retry(log_retry, "log", name, exc, poll_interval)
                elif _is_http_status_error(exc):
                    retry = _log_failure_retry(log_retry, name, exc, poll_interval)
                else:
                    raise
                if retry.warning is not None:
                    yield retry.warning
                sleep_interval = retry.sleep_interval

        if status_event.terminal:
            if logs_drained:
                return
            if not terminal_log_drain_budget.can_retry_after_failure():
                yield terminal_log_drain_budget.warning(name)
                return
        else:
            terminal_log_drain_budget.reset()

        await _async_sleep(sleep_interval, deadline)


def _sleep(poll_interval: float, deadline: _WatchDeadline) -> None:
    sleep_for = deadline.sleep_seconds(poll_interval)
    if sleep_for > 0:
        time.sleep(sleep_for)


async def _async_sleep(poll_interval: float, deadline: _WatchDeadline) -> None:
    sleep_for = deadline.sleep_seconds(poll_interval)
    if sleep_for > 0:
        await asyncio.sleep(sleep_for)


def _status_event(status_response: PlatformJobStatusResponse, job_name: str) -> JobStatusEvent:
    status = _normalized_status(status_response.status)
    terminal = status in _TERMINAL_STATUSES
    successful = (
        True if status in _SUCCESSFUL_TERMINAL_STATUSES else False if status in _FAILED_TERMINAL_STATUSES else None
    )
    error_details = status_response.error_details
    return JobStatusEvent(
        kind="status",
        job_name=job_name,
        status=status,
        status_details=dict(status_response.status_details),
        terminal=terminal,
        successful=successful,
        error_details=dict(error_details) if error_details is not None else None,
    )


def _normalized_status(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value).lower()
    return value.lower()


def _drain_logs_with_cursor_recovery(
    jobs: JobLogsClient,
    name: str,
    *,
    options: _WatchOptions,
    state: _WatchState,
    deadline: _WatchDeadline,
) -> Iterator[JobLogEvent]:
    state.start_log_drain()
    try:
        yield from _drain_logs(jobs, name, options=options, state=state, deadline=deadline)
    except NemoHTTPError as exc:
        if not _can_retry_log_scan_from_start(exc, state.log_cursor):
            raise
        state.log_cursor = None
        yield from _drain_logs(jobs, name, options=options, state=state, deadline=deadline)
    state.complete_log_drain()


async def _async_drain_logs_with_cursor_recovery(
    jobs: AsyncJobLogsClient,
    name: str,
    *,
    options: _WatchOptions,
    state: _WatchState,
    deadline: _WatchDeadline,
) -> AsyncIterator[JobLogEvent]:
    state.start_log_drain()
    try:
        async for event in _async_drain_logs(jobs, name, options=options, state=state, deadline=deadline):
            yield event
    except NemoHTTPError as exc:
        if not _can_retry_log_scan_from_start(exc, state.log_cursor):
            raise
        state.log_cursor = None
        async for event in _async_drain_logs(jobs, name, options=options, state=state, deadline=deadline):
            yield event
    state.complete_log_drain()


def _drain_logs(
    jobs: JobLogsClient,
    name: str,
    *,
    options: _WatchOptions,
    state: _WatchState,
    deadline: _WatchDeadline,
) -> Iterator[JobLogEvent]:
    current_cursor = state.log_cursor
    emit_logs = state.history_seen
    visited_cursors = {current_cursor}
    occurrence_counts: dict[_LogKey, int] = {}
    while True:
        deadline.raise_if_expired()
        page = jobs.list_job_logs(
            workspace=options.workspace,
            name=name,
            query_params=_log_query_params(options, page_cursor=current_cursor),
        ).page()
        yield from _new_log_events(
            page.items,
            state=state,
            occurrence_counts=occurrence_counts,
            name=name,
            emit=emit_logs,
        )

        next_cursor = page.metadata["next_page"]
        if next_cursor is None or next_cursor in visited_cursors:
            state.history_seen = True
            return
        visited_cursors.add(next_cursor)
        state.log_cursor = next_cursor
        current_cursor = next_cursor


async def _async_drain_logs(
    jobs: AsyncJobLogsClient,
    name: str,
    *,
    options: _WatchOptions,
    state: _WatchState,
    deadline: _WatchDeadline,
) -> AsyncIterator[JobLogEvent]:
    current_cursor = state.log_cursor
    emit_logs = state.history_seen
    visited_cursors = {current_cursor}
    occurrence_counts: dict[_LogKey, int] = {}
    while True:
        deadline.raise_if_expired()
        page_response = await jobs.list_job_logs(
            workspace=options.workspace,
            name=name,
            query_params=_log_query_params(options, page_cursor=current_cursor),
        )
        page = page_response.page()
        for event in _new_log_events(
            page.items,
            state=state,
            occurrence_counts=occurrence_counts,
            name=name,
            emit=emit_logs,
        ):
            yield event

        next_cursor = page.metadata["next_page"]
        if next_cursor is None or next_cursor in visited_cursors:
            state.history_seen = True
            return
        visited_cursors.add(next_cursor)
        state.log_cursor = next_cursor
        current_cursor = next_cursor


def _new_log_events(
    logs: Iterable[PlatformJobLog],
    *,
    state: _WatchState,
    occurrence_counts: dict[_LogKey, int],
    name: str,
    emit: bool,
) -> Iterator[JobLogEvent]:
    for log in logs:
        key = _log_key(log)
        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
        seen_occurrences = state.seen_log_occurrences(key)
        state.record_log_occurrence(key, occurrence_counts[key])
        if occurrence_counts[key] <= seen_occurrences:
            continue
        if emit:
            yield _log_event(log, name)


def _log_query_params(options: _WatchOptions, *, page_cursor: str | None) -> JobLogsQueryParams | None:
    params: JobLogsQueryParams = {}
    if options.attempt_id is not None:
        params["attempt_id"] = options.attempt_id
    if options.step_id is not None:
        params["step_id"] = options.step_id
    if options.task_id is not None:
        params["task_id"] = options.task_id
    if options.limit is not None:
        params["limit"] = options.limit
    if page_cursor is not None:
        params["page_cursor"] = page_cursor
    return params or None


def _log_event(log: PlatformJobLog, job_name: str) -> JobLogEvent:
    return JobLogEvent(
        kind="log",
        job_name=job_name,
        timestamp=log.timestamp,
        step_id=log.job_step,
        task_id=log.job_task,
        message=log.message,
    )


def _log_key(log: PlatformJobLog) -> _LogKey:
    return (log.job, log.timestamp.isoformat(), log.job_step, log.job_task, log.message)


def _transient_retry(
    retry_backoff: _TransientRetryBackoff,
    operation: Literal["status", "log"],
    job_name: str,
    exc: Exception,
    poll_interval: float,
) -> _TransientRetry:
    return _TransientRetry(
        warning=retry_backoff.warning(operation, job_name, exc),
        sleep_interval=retry_backoff.next_sleep_interval(poll_interval),
    )


def _log_failure_retry(
    retry_backoff: _TransientRetryBackoff,
    job_name: str,
    exc: Exception,
    poll_interval: float,
) -> _TransientRetry:
    return _TransientRetry(
        warning=retry_backoff.warning("log", job_name, exc, transient=False),
        sleep_interval=retry_backoff.next_sleep_interval(poll_interval),
    )


def _transient_failure_warning(
    operation: Literal["status", "log"],
    job_name: str,
    exc: Exception,
) -> JobWarningEvent:
    return JobWarningEvent(kind="warning", job_name=job_name, message=f"Transient {operation} check failed: {exc}")


def _log_failure_warning(job_name: str, exc: Exception) -> JobWarningEvent:
    return JobWarningEvent(kind="warning", job_name=job_name, message=f"Log check failed: {exc}")


def _http_status_code(exc: BaseException) -> int | None:
    """Return the HTTP status carried by *exc*, or ``None`` when it is not an HTTP status error.

    :class:`NemoHTTPError` is the typed-client status error. Any other
    non-client exception exposing an ``int`` ``status_code`` attribute (the
    generated SDK's ``APIStatusError`` shape) is treated the same way, so
    callers still on that SDK keep identical retry behaviour.
    """
    if isinstance(exc, NemoHTTPError):
        return exc.status_code
    if isinstance(exc, NemoClientError):
        return None
    status_code = getattr(exc, "status_code", None)
    return status_code if type(status_code) is int else None


def _is_http_status_error(exc: BaseException) -> bool:
    return _http_status_code(exc) is not None


def _is_transport_error(exc: BaseException) -> bool:
    """Typed-client transport failures, plus foreign wrappers chaining an httpx transport error as their cause."""
    if isinstance(exc, _TRANSPORT_ERRORS):
        return True
    if isinstance(exc, NemoClientError) or _is_http_status_error(exc):
        return False
    return isinstance(exc.__cause__, httpx.TransportError)


def _is_retryable_http_error(exc: BaseException) -> bool:
    return _http_status_code(exc) in _TRANSIENT_STATUS_CODES


def _is_transient_failure(exc: BaseException) -> bool:
    return _is_transport_error(exc) or _is_retryable_http_error(exc)


def _can_retry_log_scan_from_start(exc: NemoHTTPError, page_cursor: str | None) -> bool:
    return page_cursor is not None and _is_invalid_page_cursor_error(exc)


def _is_invalid_page_cursor_error(exc: NemoHTTPError) -> bool:
    if exc.status_code != _INVALID_PAGE_CURSOR_STATUS_CODE:
        return False

    match exc.body:
        case {"code": code} | {"detail": {"code": code}} | {"error": {"code": code}}:
            return code == _INVALID_PAGE_CURSOR_ERROR_CODE
        case {"detail": detail}:
            return detail == _PAGE_CURSOR_DECODE_ERROR_DETAIL
        case _:
            return False
