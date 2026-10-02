# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timezone
from typing import TypedDict, TypeVar

import httpx
import pytest
from nemo_helix import APIStatusError
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.client.response import AsyncNemoPaginatedResponse, NemoPaginatedResponse, NemoResponse
from nemo_helix_plugin.client.types import CursorPagination, PreparedRequest
from nemo_helix_plugin.jobs import watch as watch_module
from nemo_helix_plugin.jobs.client import AsyncJobsClient, JobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobLog, HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.jobs.types import JobLogsQueryParams
from nemo_helix_plugin.jobs.watch import async_watch_job, watch_job
from nemo_helix_plugin.jobs.watch_types import (
    JobLogEvent,
    JobStatusEvent,
    JobWarningEvent,
    JobWatchTimeoutError,
)

ResponseT = TypeVar("ResponseT")


class _StatusCall(TypedDict):
    workspace: str | None
    name: str


class _LogCall(TypedDict):
    workspace: str | None
    name: str
    query_params: JobLogsQueryParams | None


def _prepared_request(response_type: type[ResponseT] | None = None) -> PreparedRequest[ResponseT]:
    return PreparedRequest(
        path_template="/test",
        path_params={},
        method="GET",
        content=None,
        content_type=None,
        response_type=response_type,
    )


def _status_response(body: HelixJobStatusResponse) -> NemoResponse[HelixJobStatusResponse]:
    return NemoResponse(
        http_response=httpx.Response(200),
        body=body,
        request=_prepared_request(HelixJobStatusResponse),
    )


def _warning_message(event: JobStatusEvent | JobLogEvent | JobWarningEvent) -> str:
    assert isinstance(event, JobWarningEvent)
    return event.message


async def _fake_async_sleep(_: float) -> None:
    return None


def _page_cursor(call: _LogCall) -> str | None:
    query_params = call["query_params"]
    if query_params is None:
        return None
    return query_params.get("page_cursor")


class _PageResponse(
    NemoPaginatedResponse[HelixJobLog, CursorPagination],
    AsyncNemoPaginatedResponse[HelixJobLog, CursorPagination],
):
    def __init__(self, items: list[HelixJobLog], next_page: str | None = None) -> None:
        response = httpx.Response(
            200,
            json={
                "data": [item.model_dump(mode="json") for item in items],
                "total": len(items),
                "next_page": next_page,
                "prev_page": None,
            },
        )
        super().__init__(
            first_http_response=response,
            model_type=HelixJobLog,
            request=_prepared_request(),
            fetch_page=_unexpected_page_fetch,
            strategy=CursorPagination,
        )


def _unexpected_page_fetch(_request: PreparedRequest[object], _page: object) -> httpx.Response:
    raise AssertionError("Unexpected paginated fetch")


class _JobsClientState:
    def __init__(
        self,
        *,
        statuses: Iterable[HelixJobStatusResponse | Exception],
        log_results: Iterable[_PageResponse | Exception],
    ) -> None:
        self._statuses = deque(statuses)
        self._last_status: HelixJobStatusResponse | None = None
        self._log_results = deque(log_results)
        self.status_calls: list[_StatusCall] = []
        self.log_calls: list[_LogCall] = []

    def _next_status(self, *, workspace: str | None, name: str) -> HelixJobStatusResponse:
        self.status_calls.append({"workspace": workspace, "name": name})
        if self._statuses:
            result = self._statuses.popleft()
            if isinstance(result, Exception):
                raise result
            self._last_status = result
        if self._last_status is None:
            raise AssertionError("No status result configured")
        return self._last_status

    def _next_logs(
        self,
        *,
        workspace: str | None,
        name: str,
        query_params: JobLogsQueryParams | None,
    ) -> _PageResponse:
        self.log_calls.append({"workspace": workspace, "name": name, "query_params": query_params})
        if not self._log_results:
            return _PageResponse([])
        result = self._log_results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


class _SyncJobsClient(_JobsClientState):
    def get_job_status(self, *, workspace: str | None = None, name: str) -> NemoResponse[HelixJobStatusResponse]:
        return _status_response(self._next_status(workspace=workspace, name=name))

    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> NemoPaginatedResponse[HelixJobLog, CursorPagination]:
        return self._next_logs(workspace=workspace, name=name, query_params=query_params)


class _AsyncJobsClient(_JobsClientState):
    async def get_job_status(
        self,
        *,
        workspace: str | None = None,
        name: str,
    ) -> NemoResponse[HelixJobStatusResponse]:
        return _status_response(self._next_status(workspace=workspace, name=name))

    async def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> AsyncNemoPaginatedResponse[HelixJobLog, CursorPagination]:
        return self._next_logs(workspace=workspace, name=name, query_params=query_params)


def _status(status: str, status_details: dict[str, object] | None = None) -> HelixJobStatusResponse:
    timestamp = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
    return HelixJobStatusResponse(
        id="job-a",
        name="job-a",
        status=HelixJobStatus(status),
        status_details=status_details or {},
        error_details=None,
        steps=[],
        created_at=timestamp,
        updated_at=timestamp,
    )


def _log(
    message: str,
    *,
    timestamp: datetime | None = None,
    job_step: str = "step-a",
    job_task: str = "task-a",
) -> HelixJobLog:
    return HelixJobLog(
        job="job-a",
        timestamp=timestamp or datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc),
        job_step=job_step,
        job_task=job_task,
        message=message,
    )


def _record_completed_log_drain(
    state: watch_module._WatchState,
    logs: list[HelixJobLog],
) -> list[JobLogEvent]:
    state.start_log_drain()
    events = list(
        watch_module._new_log_events(
            logs,
            state=state,
            occurrence_counts={},
            name="job-a",
            emit=True,
        )
    )
    state.complete_log_drain()
    return events


def _invalid_page_cursor_error() -> NemoHTTPError:
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(422, request=request, json={"detail": "Invalid page cursor"})
    return NemoHTTPError(response)


def _http_error_body(status_code: int, body: object) -> NemoHTTPError:
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(status_code, request=request, json=body)
    return NemoHTTPError(response)


def _http_text_error(status_code: int, text: str) -> NemoHTTPError:
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(status_code, request=request, text=text)
    return NemoHTTPError(response)


def _http_error(status_code: int, detail: str) -> NemoHTTPError:
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(status_code, request=request, json={"detail": detail})
    return NemoHTTPError(response)


def test_can_retry_log_scan_from_start_accepts_invalid_cursor_code() -> None:
    exc = _http_error_body(
        422,
        {"detail": {"code": "invalid_page_cursor", "message": "The saved cursor is no longer valid"}},
    )

    assert watch_module._can_retry_log_scan_from_start(exc, "cursor-0") is True


def test_can_retry_log_scan_from_start_keeps_page_cursor_decode_compatibility() -> None:
    assert watch_module._can_retry_log_scan_from_start(_invalid_page_cursor_error(), "cursor-0") is True


def test_can_retry_log_scan_from_start_requires_saved_cursor() -> None:
    assert watch_module._can_retry_log_scan_from_start(_invalid_page_cursor_error(), None) is False


def test_can_retry_log_scan_from_start_ignores_plain_text_detail_fallback() -> None:
    exc = _http_text_error(422, "Invalid page cursor")

    assert watch_module._can_retry_log_scan_from_start(exc, "cursor-0") is False


def test_can_retry_log_scan_from_start_rejects_other_422_errors() -> None:
    exc = _http_error(422, "Invalid page size")

    assert watch_module._can_retry_log_scan_from_start(exc, "cursor-0") is False


def test_watch_job_yields_status_logs_terminal_and_passes_log_query_params() -> None:
    client = _SyncJobsClient(
        statuses=[
            _status("active", {"phase": "training"}),
            _status("completed", {"phase": "done"}),
        ],
        log_results=[
            _PageResponse([_log("starting")], next_page="cursor-1"),
            _PageResponse([_log("still running")]),
            _PageResponse([_log("starting"), _log("still running"), _log("done")]),
        ],
    )

    events = list(
        watch_job(
            client,
            "job-a",
            workspace="default",
            poll_interval=0,
            attempt_id=1,
            step_id="step-1",
            task_id="task-1",
            limit=2,
            page_cursor="cursor-0",
        )
    )

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("log", None, "starting"),
        ("log", None, "still running"),
        ("status", "completed", None),
        ("log", None, "done"),
    ]
    assert isinstance(events[0], JobStatusEvent)
    assert events[0].terminal is False
    assert events[0].successful is None
    assert events[0].status_details == {"phase": "training"}
    assert isinstance(events[3], JobStatusEvent)
    assert events[3].terminal is True
    assert events[3].successful is True

    assert client.status_calls == [
        {"workspace": "default", "name": "job-a"},
        {"workspace": "default", "name": "job-a"},
    ]
    assert client.log_calls == [
        {
            "workspace": "default",
            "name": "job-a",
            "query_params": {
                "attempt_id": 1,
                "step_id": "step-1",
                "task_id": "task-1",
                "limit": 2,
                "page_cursor": "cursor-0",
            },
        },
        {
            "workspace": "default",
            "name": "job-a",
            "query_params": {
                "attempt_id": 1,
                "step_id": "step-1",
                "task_id": "task-1",
                "limit": 2,
                "page_cursor": "cursor-1",
            },
        },
        {
            "workspace": "default",
            "name": "job-a",
            "query_params": {
                "attempt_id": 1,
                "step_id": "step-1",
                "task_id": "task-1",
                "limit": 2,
                "page_cursor": "cursor-1",
            },
        },
    ]


def test_watch_job_rejects_negative_poll_interval_eagerly() -> None:
    client = _SyncJobsClient(statuses=[], log_results=[])

    with pytest.raises(ValueError, match="poll_interval"):
        watch_job(client, "job-a", poll_interval=-1)


def test_jobs_client_watch_job_delegates_to_source_owned_watcher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_watch_job(client: JobsClient, name: str, **kwargs: object) -> Iterable[JobWarningEvent]:
        calls["client"] = client
        calls["name"] = name
        calls["kwargs"] = kwargs
        return iter([JobWarningEvent(kind="warning", job_name=name, message="delegated")])

    monkeypatch.setattr("nemo_helix_plugin.jobs.watch.watch_job", fake_watch_job)
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    client = JobsClient(base_url="http://test", http_client=http_client)

    try:
        events = list(
            client.watch_job(
                "job-a",
                workspace="default",
                poll_interval=0,
                timeout=5,
                include_history=False,
                include_logs=False,
                attempt_id=1,
                step_id="step-1",
                task_id="task-1",
                limit=2,
                page_cursor="cursor-0",
            )
        )
    finally:
        http_client.close()

    assert calls == {
        "client": client,
        "name": "job-a",
        "kwargs": {
            "workspace": "default",
            "poll_interval": 0,
            "timeout": 5,
            "include_history": False,
            "include_logs": False,
            "attempt_id": 1,
            "step_id": "step-1",
            "task_id": "task-1",
            "limit": 2,
            "page_cursor": "cursor-0",
        },
    }
    assert events == [JobWarningEvent(kind="warning", job_name="job-a", message="delegated")]


def test_watch_job_can_skip_existing_log_history() -> None:
    client = _SyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _PageResponse([_log("old")]),
            _PageResponse([_log("old"), _log("new")]),
        ],
    )

    events = list(watch_job(client, "job-a", include_history=False, poll_interval=0))

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("status", "completed", None),
        ("log", None, "new"),
    ]


def test_watch_state_replaces_retained_log_window_after_successful_drain() -> None:
    state = watch_module._WatchState(history_seen=True, log_cursor=None)
    first_log = _log("first")
    second_log = _log("second")

    assert [event.message for event in _record_completed_log_drain(state, [first_log])] == ["first"]
    assert state.previous_drain_seen_logs == {watch_module._log_key(first_log): 1}

    assert [event.message for event in _record_completed_log_drain(state, [second_log])] == ["second"]
    assert state.previous_drain_seen_logs == {watch_module._log_key(second_log): 1}
    assert state.current_drain_seen_logs == {}


def test_new_log_events_suppresses_seen_occurrences_after_partial_drain_failure() -> None:
    state = watch_module._WatchState(history_seen=True, log_cursor=None)
    duplicate_log = _log("duplicate")

    state.start_log_drain()
    first_events = list(
        watch_module._new_log_events(
            [duplicate_log, duplicate_log],
            state=state,
            occurrence_counts={},
            name="job-a",
            emit=True,
        )
    )

    state.start_log_drain()
    retry_events = list(
        watch_module._new_log_events(
            [duplicate_log, duplicate_log, duplicate_log],
            state=state,
            occurrence_counts={},
            name="job-a",
            emit=True,
        )
    )
    state.complete_log_drain()

    assert [event.message for event in first_events] == ["duplicate", "duplicate"]
    assert [event.message for event in retry_events] == ["duplicate"]
    assert state.previous_drain_seen_logs == {watch_module._log_key(duplicate_log): 3}


def test_watch_job_can_poll_status_without_logs() -> None:
    client = _SyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[AssertionError("logs should not be fetched")],
    )

    events = list(watch_job(client, "job-a", include_logs=False, poll_interval=0))

    assert [(event.kind, getattr(event, "status", None)) for event in events] == [
        ("status", "active"),
        ("status", "completed"),
    ]
    assert client.log_calls == []


def test_watch_job_stops_when_status_is_paused() -> None:
    client = _SyncJobsClient(
        statuses=[_status("paused")],
        log_results=[_PageResponse([_log("paused")])],
    )

    events = list(watch_job(client, "job-a", poll_interval=0))

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "paused", None),
        ("log", None, "paused"),
    ]
    assert isinstance(events[0], JobStatusEvent)
    assert events[0].terminal is True
    assert events[0].successful is False
    assert len(client.status_calls) == 1
    assert len(client.log_calls) == 1


def test_watch_job_retries_sdk_transient_status_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(503, request=request)
    client = _SyncJobsClient(
        statuses=[
            APIStatusError("service unavailable", response=response, body=None),
            _status("completed"),
        ],
        log_results=[AssertionError("logs should not be fetched")],
    )
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)

    events = list(watch_job(client, "job-a", include_logs=False, poll_interval=0))

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("warning", None, "Transient status check failed: service unavailable"),
        ("status", "completed", None),
    ]
    assert client.log_calls == []


def test_watch_job_backs_off_and_deduplicates_consecutive_transient_status_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "http://test")
    response = httpx.Response(503, request=request)
    client = _SyncJobsClient(
        statuses=[
            APIStatusError("service unavailable", response=response, body=None),
            APIStatusError("service unavailable", response=response, body=None),
            APIStatusError("service unavailable", response=response, body=None),
            APIStatusError("service unavailable", response=response, body=None),
            APIStatusError("service unavailable", response=response, body=None),
            APIStatusError("service unavailable", response=response, body=None),
            APIStatusError("service unavailable", response=response, body=None),
            _status("completed"),
        ],
        log_results=[AssertionError("logs should not be fetched")],
    )
    sleeps: list[float] = []
    monkeypatch.setattr(watch_module.time, "sleep", sleeps.append)

    events = list(watch_job(client, "job-a", include_logs=False, poll_interval=1))

    assert sleeps == [1, 2, 4, 8, 16, 30.0, 30.0]
    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("warning", None, "Transient status check failed: service unavailable"),
        ("status", "completed", None),
    ]
    assert client.log_calls == []


def test_transient_retry_backoff_uses_minimum_sleep_interval() -> None:
    backoff = watch_module._TransientRetryBackoff()

    assert backoff.next_sleep_interval(0) == 1.0
    assert backoff.next_sleep_interval(0) == 2.0


def test_watch_job_suppresses_unread_history_after_partial_history_drain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _SyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _PageResponse([_log("old")], next_page="cursor-1"),
            NemoTransportError(httpx.TransportError("temporary log failure")),
            _PageResponse([_log("old"), _log("also-old")]),
        ],
    )
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)

    events = list(watch_job(client, "job-a", include_history=False, poll_interval=0))

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("warning", None, "Transient log check failed: temporary log failure"),
        ("status", "completed", None),
    ]


def test_watch_job_continues_status_polling_after_non_retryable_log_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _SyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _http_error(500, "log store unavailable"),
            _PageResponse([_log("done")]),
        ],
    )
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)

    events = list(watch_job(client, "job-a", poll_interval=0))

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("warning", None, "Log check failed: HTTP 500: log store unavailable"),
        ("status", "completed", None),
        ("log", None, "done"),
    ]


def test_watch_job_retries_terminal_status_until_logs_drain_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _SyncJobsClient(
        statuses=[_status("completed"), _status("completed")],
        log_results=[
            NemoTransportError(httpx.TransportError("temporary log failure")),
            _PageResponse([_log("done")]),
        ],
    )
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)

    events = list(watch_job(client, "job-a", poll_interval=0))

    assert [event.kind for event in events] == ["status", "warning", "log"]
    assert _warning_message(events[1]) == "Transient log check failed: temporary log failure"
    assert isinstance(events[2], JobLogEvent)
    assert events[2].message == "done"


def test_watch_job_stops_after_terminal_log_drain_retry_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    retry_cap = watch_module._TerminalLogDrainRetryBudget.RETRY_CAP
    client = _SyncJobsClient(
        statuses=[_status("completed")],
        log_results=[
            NemoTransportError(httpx.TransportError(f"temporary log failure {attempt}")) for attempt in range(retry_cap)
        ],
    )
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)

    events = list(watch_job(client, "job-a", poll_interval=0))

    assert [event.kind for event in events] == ["status"] + ["warning"] * (retry_cap + 1)
    assert [getattr(event, "message", None) for event in events[1:-1]] == [
        f"Transient log check failed: temporary log failure {attempt}" for attempt in range(retry_cap)
    ]
    assert _warning_message(events[-1]) == (
        f"Terminal log drain retry cap reached ({retry_cap}); stopping watch for job 'job-a'"
    )
    assert len(client.status_calls) == retry_cap
    assert len(client.log_calls) == retry_cap


def test_watch_job_falls_back_to_full_rescan_when_saved_cursor_is_invalid() -> None:
    client = _SyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _PageResponse([_log("old")], next_page="cursor-1"),
            _PageResponse([_log("new")]),
            _invalid_page_cursor_error(),
            _PageResponse([_log("old"), _log("new"), _log("done")]),
        ],
    )

    events = list(watch_job(client, "job-a", poll_interval=0))

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("log", None, "old"),
        ("log", None, "new"),
        ("status", "completed", None),
        ("log", None, "done"),
    ]
    assert [_page_cursor(call) for call in client.log_calls] == [None, "cursor-1", "cursor-1", None]


def test_watch_job_stops_log_pagination_when_cursor_does_not_advance() -> None:
    client = _SyncJobsClient(
        statuses=[_status("completed")],
        log_results=[_PageResponse([_log("done")], next_page="cursor-0")],
    )

    events = list(watch_job(client, "job-a", poll_interval=0, page_cursor="cursor-0"))

    assert [(event.kind, getattr(event, "message", None)) for event in events] == [
        ("status", None),
        ("log", "done"),
    ]
    assert [_page_cursor(call) for call in client.log_calls] == ["cursor-0"]


def test_watch_job_enforces_timeout_between_log_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _SyncJobsClient(
        statuses=[_status("completed")],
        log_results=[
            _PageResponse([], next_page="cursor-1"),
            _PageResponse([]),
        ],
    )
    monotonic_values = iter([0.0, 0.0, 1.0, 10.0])
    monkeypatch.setattr(watch_module.time, "monotonic", lambda: next(monotonic_values, 10.0))

    with pytest.raises(JobWatchTimeoutError, match="job-a"):
        list(watch_job(client, "job-a", timeout=5, poll_interval=0))

    assert len(client.log_calls) == 1


async def test_async_watch_job_suppresses_unread_history_after_partial_history_drain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _AsyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _PageResponse([_log("old")], next_page="cursor-1"),
            NemoTransportError(httpx.TransportError("temporary log failure")),
            _PageResponse([_log("old"), _log("also-old")]),
        ],
    )
    monkeypatch.setattr(watch_module.asyncio, "sleep", _fake_async_sleep)

    events = [event async for event in async_watch_job(client, "job-a", include_history=False, poll_interval=0)]

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("warning", None, "Transient log check failed: temporary log failure"),
        ("status", "completed", None),
    ]


async def test_async_watch_job_backs_off_and_deduplicates_consecutive_transient_log_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _AsyncJobsClient(
        statuses=[_status("active"), _status("active"), _status("completed")],
        log_results=[
            NemoTransportError(httpx.TransportError("temporary log failure")),
            NemoTransportError(httpx.TransportError("temporary log failure")),
            _PageResponse([_log("done")]),
        ],
    )
    sleeps: list[float] = []

    async def fake_sleep(sleep_for: float) -> None:
        sleeps.append(sleep_for)

    monkeypatch.setattr(watch_module.asyncio, "sleep", fake_sleep)

    events = [event async for event in async_watch_job(client, "job-a", poll_interval=1)]

    assert sleeps == [1, 2]
    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("warning", None, "Transient log check failed: temporary log failure"),
        ("status", "completed", None),
        ("log", None, "done"),
    ]


async def test_async_watch_job_continues_status_polling_after_non_retryable_log_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _AsyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _http_error(500, "log store unavailable"),
            _PageResponse([_log("done")]),
        ],
    )
    monkeypatch.setattr(watch_module.asyncio, "sleep", _fake_async_sleep)

    events = [event async for event in async_watch_job(client, "job-a", poll_interval=0)]

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("warning", None, "Log check failed: HTTP 500: log store unavailable"),
        ("status", "completed", None),
        ("log", None, "done"),
    ]


async def test_async_watch_job_can_poll_status_without_logs() -> None:
    client = _AsyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[AssertionError("logs should not be fetched")],
    )

    events = [event async for event in async_watch_job(client, "job-a", include_logs=False, poll_interval=0)]

    assert [(event.kind, getattr(event, "status", None)) for event in events] == [
        ("status", "active"),
        ("status", "completed"),
    ]
    assert client.log_calls == []


async def test_async_watch_job_stops_when_status_is_paused() -> None:
    client = _AsyncJobsClient(
        statuses=[_status("paused")],
        log_results=[_PageResponse([_log("paused")])],
    )

    events = [event async for event in async_watch_job(client, "job-a", poll_interval=0)]

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "paused", None),
        ("log", None, "paused"),
    ]
    assert isinstance(events[0], JobStatusEvent)
    assert events[0].terminal is True
    assert events[0].successful is False
    assert len(client.status_calls) == 1
    assert len(client.log_calls) == 1


async def test_async_watch_job_falls_back_to_full_rescan_when_saved_cursor_is_invalid() -> None:
    client = _AsyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _PageResponse([_log("old")], next_page="cursor-1"),
            _PageResponse([_log("new")]),
            _invalid_page_cursor_error(),
            _PageResponse([_log("old"), _log("new"), _log("done")]),
        ],
    )

    events = [event async for event in async_watch_job(client, "job-a", poll_interval=0)]

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("log", None, "old"),
        ("log", None, "new"),
        ("status", "completed", None),
        ("log", None, "done"),
    ]
    assert [_page_cursor(call) for call in client.log_calls] == [None, "cursor-1", "cursor-1", None]


async def test_async_watch_job_stops_log_pagination_when_cursor_repeats() -> None:
    client = _AsyncJobsClient(
        statuses=[_status("completed")],
        log_results=[
            _PageResponse([_log("one")], next_page="cursor-1"),
            _PageResponse([_log("two")], next_page="cursor-0"),
        ],
    )

    events = [event async for event in async_watch_job(client, "job-a", poll_interval=0, page_cursor="cursor-0")]

    assert [(event.kind, getattr(event, "message", None)) for event in events] == [
        ("status", None),
        ("log", "one"),
        ("log", "two"),
    ]
    assert [_page_cursor(call) for call in client.log_calls] == ["cursor-0", "cursor-1"]


async def test_async_watch_job_enforces_timeout_between_log_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _AsyncJobsClient(
        statuses=[_status("completed")],
        log_results=[
            _PageResponse([], next_page="cursor-1"),
            _PageResponse([]),
        ],
    )
    monotonic_values = iter([0.0, 0.0, 1.0, 10.0])
    monkeypatch.setattr(watch_module.time, "monotonic", lambda: next(monotonic_values, 10.0))

    with pytest.raises(JobWatchTimeoutError, match="job-a"):
        [event async for event in async_watch_job(client, "job-a", timeout=5, poll_interval=0)]

    assert len(client.log_calls) == 1


async def test_async_watch_job_stops_after_terminal_log_drain_retry_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    retry_cap = watch_module._TerminalLogDrainRetryBudget.RETRY_CAP
    client = _AsyncJobsClient(
        statuses=[_status("completed")],
        log_results=[
            NemoTransportError(httpx.TransportError(f"temporary log failure {attempt}")) for attempt in range(retry_cap)
        ],
    )
    monkeypatch.setattr(watch_module.asyncio, "sleep", _fake_async_sleep)

    events = [event async for event in async_watch_job(client, "job-a", poll_interval=0)]

    assert [event.kind for event in events] == ["status"] + ["warning"] * (retry_cap + 1)
    assert [getattr(event, "message", None) for event in events[1:-1]] == [
        f"Transient log check failed: temporary log failure {attempt}" for attempt in range(retry_cap)
    ]
    assert _warning_message(events[-1]) == (
        f"Terminal log drain retry cap reached ({retry_cap}); stopping watch for job 'job-a'"
    )
    assert len(client.status_calls) == retry_cap
    assert len(client.log_calls) == retry_cap


def test_watch_job_raises_timeout_with_job_name(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _SyncJobsClient(statuses=[_status("active")], log_results=[_PageResponse([])])
    monotonic_values = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(watch_module.time, "monotonic", lambda: next(monotonic_values, 10.0))

    with pytest.raises(JobWatchTimeoutError, match="job-a"):
        list(watch_job(client, "job-a", timeout=5, poll_interval=1))


async def test_async_watch_job_is_async_iterator_and_uses_async_jobs_client() -> None:
    client = _AsyncJobsClient(
        statuses=[_status("active"), _status("completed")],
        log_results=[
            _PageResponse([_log("starting")]),
            _PageResponse([_log("starting"), _log("done")]),
        ],
    )

    iterator = async_watch_job(client, "job-a", workspace="default", poll_interval=0)
    assert hasattr(iterator, "__aiter__")

    events = [event async for event in iterator]

    assert [(event.kind, getattr(event, "status", None), getattr(event, "message", None)) for event in events] == [
        ("status", "active", None),
        ("log", None, "starting"),
        ("status", "completed", None),
        ("log", None, "done"),
    ]
    assert client.status_calls == [
        {"workspace": "default", "name": "job-a"},
        {"workspace": "default", "name": "job-a"},
    ]


def test_async_watch_job_rejects_negative_poll_interval_eagerly() -> None:
    client = _AsyncJobsClient(statuses=[], log_results=[])

    with pytest.raises(ValueError, match="poll_interval"):
        async_watch_job(client, "job-a", poll_interval=-1)


async def test_async_jobs_client_watch_job_delegates_to_source_owned_watcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    async def fake_events(job_name: str) -> AsyncIterator[JobWarningEvent]:
        yield JobWarningEvent(kind="warning", job_name=job_name, message="delegated")

    def fake_async_watch_job(client: AsyncJobsClient, name: str, **kwargs: object) -> AsyncIterator[JobWarningEvent]:
        calls["client"] = client
        calls["name"] = name
        calls["kwargs"] = kwargs
        return fake_events(name)

    monkeypatch.setattr("nemo_helix_plugin.jobs.watch.async_watch_job", fake_async_watch_job)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    client = AsyncJobsClient(base_url="http://test", http_client=http_client)

    try:
        events = [
            event
            async for event in client.watch_job(
                "job-a",
                workspace="default",
                poll_interval=0,
                timeout=5,
                include_history=False,
                include_logs=False,
                attempt_id=1,
                step_id="step-1",
                task_id="task-1",
                limit=2,
                page_cursor="cursor-0",
            )
        ]
    finally:
        await http_client.aclose()

    assert calls == {
        "client": client,
        "name": "job-a",
        "kwargs": {
            "workspace": "default",
            "poll_interval": 0,
            "timeout": 5,
            "include_history": False,
            "include_logs": False,
            "attempt_id": 1,
            "step_id": "step-1",
            "task_id": "task-1",
            "limit": 2,
            "page_cursor": "cursor-0",
        },
    }
    assert events == [JobWarningEvent(kind="warning", job_name="job-a", message="delegated")]
