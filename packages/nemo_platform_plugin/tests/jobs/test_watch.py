# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from nemo_platform_plugin.client.errors import NemoHTTPError, NemoResponseValidationError, NemoTransportError
from nemo_platform_plugin.client.response import NemoPaginatedResponse, NemoResponse
from nemo_platform_plugin.client.types import CursorPagination, PreparedRequest
from nemo_platform_plugin.jobs import watch as watch_module
from nemo_platform_plugin.jobs.schemas import PlatformJobLog, PlatformJobStatus, PlatformJobStatusResponse
from nemo_platform_plugin.jobs.types import JobLogsQueryParams
from nemo_platform_plugin.jobs.watch import _status_event, watch_job


def test_status_event_preserves_error_details_for_failed_job() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    status_response = PlatformJobStatusResponse(
        id="job-id",
        name="job-name",
        status=PlatformJobStatus.ERROR,
        status_details={"phase": "failed"},
        error_details={"reason": "container exited"},
        steps=[],
        created_at=timestamp,
        updated_at=timestamp,
    )

    event = _status_event(status_response, "job-name")

    assert event.status == "error"
    assert event.status_details == {"phase": "failed"}
    assert event.error_details == {"reason": "container exited"}
    assert event.terminal is True
    assert event.successful is False


class _ForeignStatusError(Exception):
    """Generated-SDK ``APIStatusError`` shape: any exception carrying an ``int`` ``status_code``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _ForeignConnectionError(Exception):
    """Generated-SDK ``APIConnectionError`` shape: chains the httpx transport failure as its cause."""


def _foreign_connection_error() -> _ForeignConnectionError:
    try:
        raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://test"))
    except httpx.ConnectError as cause:
        error = _ForeignConnectionError("connection error")
        error.__cause__ = cause
        return error


def _nemo_http_error(status_code: int) -> NemoHTTPError:
    request = httpx.Request("GET", "http://test")
    return NemoHTTPError(httpx.Response(status_code, request=request, json={"detail": "x"}))


def _nemo_transport_error() -> NemoTransportError:
    return NemoTransportError(httpx.ConnectError("refused", request=httpx.Request("GET", "http://test")))


@pytest.mark.parametrize(
    ("exc", "http_status", "transport", "transient"),
    [
        pytest.param(_nemo_http_error(503), True, False, True, id="typed-http-503"),
        pytest.param(_nemo_http_error(404), True, False, False, id="typed-http-404"),
        pytest.param(_nemo_transport_error(), False, True, True, id="typed-transport"),
        pytest.param(_ForeignStatusError(503), True, False, True, id="foreign-status-503"),
        pytest.param(_ForeignStatusError(404), True, False, False, id="foreign-status-404"),
        pytest.param(_foreign_connection_error(), False, True, True, id="foreign-connection"),
        pytest.param(_ForeignConnectionError("no cause"), False, False, False, id="foreign-without-httpx-cause"),
        pytest.param(RuntimeError("boom"), False, False, False, id="plain-exception"),
        pytest.param(
            NemoResponseValidationError(httpx.Response(200, request=httpx.Request("GET", "http://t")), ValueError()),
            False,
            False,
            False,
            id="typed-validation-error-is-not-a-status-error",
        ),
    ],
)
def test_failure_classification_covers_typed_and_foreign_error_shapes(
    exc: Exception, http_status: bool, transport: bool, transient: bool
) -> None:
    assert watch_module._is_http_status_error(exc) is http_status
    assert watch_module._is_transport_error(exc) is transport
    assert watch_module._is_transient_failure(exc) is transient


def test_http_status_code_ignores_non_int_status_codes() -> None:
    class _Odd(Exception):
        status_code = "503"

    class _Bool(Exception):
        status_code = True

    assert watch_module._http_status_code(_Odd()) is None
    assert watch_module._http_status_code(_Bool()) is None
    assert watch_module._http_status_code(_ForeignStatusError(429)) == 429


class _StatusOnlyClient:
    """Minimal ``JobsWatchClient`` whose status calls replay *results* in order."""

    def __init__(self, results: list[PlatformJobStatusResponse | Exception]) -> None:
        self._results = list(results)
        self.calls = 0

    def get_job_status(self, *, workspace: str | None = None, name: str) -> NemoResponse[PlatformJobStatusResponse]:
        self.calls += 1
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return NemoResponse(
            http_response=httpx.Response(200),
            body=result,
            request=PreparedRequest(
                path_template="/test",
                path_params={},
                method="GET",
                content=None,
                content_type=None,
                response_type=PlatformJobStatusResponse,
            ),
        )

    def list_job_logs(
        self,
        *,
        workspace: str | None = None,
        name: str,
        query_params: JobLogsQueryParams | None = None,
    ) -> NemoPaginatedResponse[PlatformJobLog, CursorPagination]:
        raise AssertionError("logs are disabled in this test")


def _completed() -> PlatformJobStatusResponse:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return PlatformJobStatusResponse(
        id="job-a",
        name="job-a",
        status=PlatformJobStatus.COMPLETED,
        status_details={},
        error_details=None,
        steps=[],
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_watch_job_retries_foreign_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry semantics hold for both error families without importing the generated SDK."""
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)
    client = _StatusOnlyClient([_ForeignStatusError(503), _foreign_connection_error(), _completed()])

    events = list(watch_job(client, "job-a", include_logs=False, poll_interval=0))

    assert [event.kind for event in events] == ["warning", "warning", "status"]
    assert [getattr(event, "message", None) for event in events[:2]] == [
        "Transient status check failed: HTTP 503",
        "Transient status check failed: connection error",
    ]
    assert client.calls == 3


def test_watch_job_raises_foreign_non_transient_status_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)
    client = _StatusOnlyClient([_ForeignStatusError(404)])

    with pytest.raises(_ForeignStatusError):
        list(watch_job(client, "job-a", include_logs=False, poll_interval=0))


def test_watch_job_propagates_unclassified_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watch_module.time, "sleep", lambda _: None)
    client = _StatusOnlyClient([RuntimeError("boom")])

    with pytest.raises(RuntimeError, match="boom"):
        list(watch_job(client, "job-a", include_logs=False, poll_interval=0))
