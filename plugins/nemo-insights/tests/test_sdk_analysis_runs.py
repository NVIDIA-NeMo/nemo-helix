# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``client.insights.analysis_runs`` SDK sub-resource."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from nemo_insights_plugin.schema import CreateAnalysisRunRequest
from nemo_insights_plugin.sdk_resources.analysis_runs import (
    AnalysisRunNotSubmittedError,
    AnalysisRunTimeoutError,
    _AnalysisRunResource,
    _AsyncAnalysisRunResource,
)
from nemo_insights_plugin.types import ListAnalysisRunsQueryParams

RUN_NAME = "insights-run-0123456789abcdef0123456789abcdef"
DEFAULT_MODEL = "default/big"
FAST_MODEL = "default/small"


def _run_body(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "entity-123",
        "name": RUN_NAME,
        "workspace": "default",
        "agent": "demo-agent",
        "since": None,
        "evaluation_id": "",
        "default_model": DEFAULT_MODEL,
        "fast_model": FAST_MODEL,
        **overrides,
    }


def _response_body(status: str | None = "created", **run_overrides: Any) -> dict[str, Any]:
    job = None if status is None else {"name": RUN_NAME, "status": status}
    return {"run": _run_body(**run_overrides), "job": job}


class _StubHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubTypedResponse:
    def __init__(self, payload: dict[str, Any], *, page_error: Exception | None = None) -> None:
        self.http_response = _StubHTTPResponse(payload)
        self._page_error = page_error

    def page(self) -> object:
        if self._page_error is not None:
            raise self._page_error
        return None


class _StubInsightsClient:
    """Records typed client calls and replays a queued sequence of response bodies.

    The last body repeats, so a polling test only has to queue the states it
    cares about.
    """

    def __init__(self, *bodies: dict[str, Any], page_error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._bodies = list(bodies) or [_response_body()]
        self._page_error = page_error

    def _next(self) -> _StubTypedResponse:
        body = self._bodies[0] if len(self._bodies) == 1 else self._bodies.pop(0)
        return _StubTypedResponse(body, page_error=self._page_error)

    def create_analysis_run(self, *, workspace: str, body: CreateAnalysisRunRequest) -> _StubTypedResponse:
        self.calls.append(
            {
                "method": "POST",
                "url": f"http://platform/apis/insights/v2/workspaces/{workspace}/analysis-runs",
                "json": body.model_dump(mode="json", exclude_unset=True),
            }
        )
        return self._next()

    def list_analysis_runs(self, *, workspace: str, query_params: ListAnalysisRunsQueryParams) -> _StubTypedResponse:
        self.calls.append(
            {
                "method": "GET",
                "url": f"http://platform/apis/insights/v2/workspaces/{workspace}/analysis-runs",
                "params": query_params,
            }
        )
        return self._next()

    def get_analysis_run(self, *, workspace: str, name: str) -> _StubTypedResponse:
        self.calls.append(
            {
                "method": "GET",
                "url": f"http://platform/apis/insights/v2/workspaces/{workspace}/analysis-runs/{name}",
            }
        )
        return self._next()


class _AsyncStubInsightsClient:
    """Async mirror of :class:`_StubInsightsClient`, delegating to one for recording."""

    def __init__(self, *bodies: dict[str, Any]) -> None:
        self._sync = _StubInsightsClient(*bodies)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._sync.calls

    async def create_analysis_run(self, *, workspace: str, body: CreateAnalysisRunRequest) -> _StubTypedResponse:
        return self._sync.create_analysis_run(workspace=workspace, body=body)

    async def list_analysis_runs(
        self, *, workspace: str, query_params: ListAnalysisRunsQueryParams
    ) -> _StubTypedResponse:
        return self._sync.list_analysis_runs(workspace=workspace, query_params=query_params)

    async def get_analysis_run(self, *, workspace: str, name: str) -> _StubTypedResponse:
        return self._sync.get_analysis_run(workspace=workspace, name=name)


class _StubParent:
    def __init__(self, client: _StubInsightsClient) -> None:
        self._client = client


class _AsyncStubParent:
    def __init__(self, client: _AsyncStubInsightsClient) -> None:
        self._client = client


def _resource(client: _StubInsightsClient) -> _AnalysisRunResource:
    return _AnalysisRunResource(_StubParent(client))


def _async_resource(client: _AsyncStubInsightsClient) -> _AsyncAnalysisRunResource:
    return _AsyncAnalysisRunResource(_AsyncStubParent(client))


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_posts_the_request_to_the_analysis_runs_route() -> None:
    http = _StubInsightsClient()

    _resource(http).create(
        workspace="team-a",
        agent="demo-agent",
        default_model=DEFAULT_MODEL,
        fast_model=FAST_MODEL,
    )

    assert http.calls[0]["url"] == "http://platform/apis/insights/v2/workspaces/team-a/analysis-runs"
    assert http.calls[0]["json"] == {
        "agent": "demo-agent",
        "default_model": DEFAULT_MODEL,
        "fast_model": FAST_MODEL,
    }


def test_create_sends_optional_read_scope_when_given() -> None:
    """Unset optionals stay off the wire so the server's defaults apply."""
    http = _StubInsightsClient()

    _resource(http).create(
        workspace="default",
        agent="demo-agent",
        default_model=DEFAULT_MODEL,
        fast_model=FAST_MODEL,
        since=datetime(2026, 8, 1, tzinfo=timezone.utc),
        evaluation_id="eval-123",
        timeout_seconds=60.0,
    )

    assert http.calls[0]["json"] == {
        "agent": "demo-agent",
        "default_model": DEFAULT_MODEL,
        "fast_model": FAST_MODEL,
        "since": "2026-08-01T00:00:00Z",
        "evaluation_id": "eval-123",
        "timeout_seconds": 60.0,
    }


def test_create_sends_the_ethos_inline() -> None:
    """The Fabric adapter has no Files access, so the Markdown travels in the body."""
    http = _StubInsightsClient()

    _resource(http).create(
        workspace="default",
        agent="demo-agent",
        default_model=DEFAULT_MODEL,
        fast_model=FAST_MODEL,
        ethos="# Ethos\n\nBe careful.",
    )

    assert http.calls[0]["json"]["ethos"] == "# Ethos\n\nBe careful."


def test_create_omits_the_ethos_when_unset() -> None:
    http = _StubInsightsClient()

    _resource(http).create(workspace="default", agent="demo-agent", default_model=DEFAULT_MODEL, fast_model=FAST_MODEL)

    assert "ethos" not in http.calls[0]["json"]


async def test_async_create_sends_the_ethos_inline() -> None:
    http = _AsyncStubInsightsClient()

    await _async_resource(http).create(
        workspace="default",
        agent="demo-agent",
        default_model=DEFAULT_MODEL,
        fast_model=FAST_MODEL,
        ethos="# Ethos",
    )

    assert http.calls[0]["json"]["ethos"] == "# Ethos"


def test_create_returns_the_run_with_its_store_assigned_id() -> None:
    response = _resource(_StubInsightsClient()).create(
        workspace="default",
        agent="demo-agent",
        default_model=DEFAULT_MODEL,
        fast_model=FAST_MODEL,
    )

    assert response.run.name == RUN_NAME
    assert response.run.id == "entity-123"
    assert response.job == {"name": RUN_NAME, "status": "created"}


# ---------------------------------------------------------------------------
# List and get
# ---------------------------------------------------------------------------


def test_list_runs_omits_the_agent_filter_when_not_given() -> None:
    http = _StubInsightsClient({"data": [], "pagination": None, "sort": "-created_at", "filter": None})

    _resource(http).list_runs(workspace="default")

    assert http.calls[0]["params"] == {"page": 1, "page_size": 20, "sort": "-created_at"}


def test_list_runs_filters_by_agent_and_hydrates_items() -> None:
    http = _StubInsightsClient(
        {
            "data": [_run_body()],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "current_page_size": 1,
                "total_pages": 1,
                "total_results": 1,
            },
            "sort": "-created_at",
            "filter": {"agent": "demo-agent"},
        }
    )

    page = _resource(http).list_runs(workspace="default", agent="demo-agent")

    assert http.calls[0]["params"]["agent"] == "demo-agent"
    assert page.data[0].id == "entity-123"


def test_list_runs_uses_paginated_response_status_handling() -> None:
    http = _StubInsightsClient(
        {"detail": "permission denied"},
        page_error=RuntimeError("paginated status error"),
    )

    with pytest.raises(RuntimeError, match="paginated status error"):
        _resource(http).list_runs(workspace="default")


def test_get_reads_one_run_joined_with_its_job() -> None:
    http = _StubInsightsClient(_response_body(status="active"))

    response = _resource(http).get(workspace="default", name=RUN_NAME)

    assert http.calls[0]["url"].endswith(f"/analysis-runs/{RUN_NAME}")
    assert response.job_status == "active"
    assert response.job_is_terminal is False


def test_a_run_with_no_job_has_no_status_and_is_not_terminal() -> None:
    """A missing job means submission never landed — not that the run finished."""
    response = _resource(_StubInsightsClient(_response_body(status=None))).get(workspace="default", name=RUN_NAME)

    assert response.job is None
    assert response.job_status is None
    assert response.job_is_terminal is False


@pytest.mark.parametrize(
    ("status", "terminal"),
    [("completed", True), ("error", True), ("cancelled", True), ("pending", False), ("active", False)],
)
def test_job_terminality_follows_the_platform_job_states(status: str, terminal: bool) -> None:
    response = _resource(_StubInsightsClient(_response_body(status=status))).get(workspace="default", name=RUN_NAME)

    assert response.job_is_terminal is terminal


# ---------------------------------------------------------------------------
# Wait
# ---------------------------------------------------------------------------


def test_wait_polls_until_the_job_is_terminal() -> None:
    http = _StubInsightsClient(
        _response_body(status="created"),
        _response_body(status="active"),
        _response_body(status="completed"),
    )
    seen: list[str | None] = []

    response = _resource(http).wait(
        workspace="default",
        name=RUN_NAME,
        poll_interval=0,
        on_status=seen.append,
    )

    assert response.job_status == "completed"
    assert seen == ["created", "active", "completed"]
    assert len(http.calls) == 3


def test_wait_returns_a_failed_job_rather_than_raising() -> None:
    """A job that ran and failed is an answer; the caller decides what it means."""
    response = _resource(_StubInsightsClient(_response_body(status="error"))).wait(
        workspace="default", name=RUN_NAME, poll_interval=0
    )

    assert response.job_status == "error"
    assert response.job_is_terminal is True


def test_wait_refuses_to_poll_a_run_that_was_never_submitted() -> None:
    """Its job will never appear, so polling would only burn the timeout."""
    http = _StubInsightsClient(_response_body(status=None))

    with pytest.raises(AnalysisRunNotSubmittedError) as excinfo:
        _resource(http).wait(workspace="default", name=RUN_NAME, poll_interval=0)

    assert RUN_NAME in str(excinfo.value)
    assert len(http.calls) == 1


def test_wait_times_out_with_the_last_status_it_saw() -> None:
    http = _StubInsightsClient(_response_body(status="active"))

    with pytest.raises(AnalysisRunTimeoutError) as excinfo:
        _resource(http).wait(workspace="default", name=RUN_NAME, timeout=0, poll_interval=0)

    assert "'active'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Async parity
# ---------------------------------------------------------------------------


async def test_async_create_matches_the_sync_request() -> None:
    http = _AsyncStubInsightsClient()

    response = await _async_resource(http).create(
        workspace="team-a",
        agent="demo-agent",
        default_model=DEFAULT_MODEL,
        fast_model=FAST_MODEL,
    )

    assert http.calls[0]["url"] == "http://platform/apis/insights/v2/workspaces/team-a/analysis-runs"
    assert response.run.id == "entity-123"


async def test_async_wait_polls_until_terminal() -> None:
    http = _AsyncStubInsightsClient(
        _response_body(status="active"),
        _response_body(status="completed"),
    )

    response = await _async_resource(http).wait(workspace="default", name=RUN_NAME, poll_interval=0)

    assert response.job_status == "completed"
    assert len(http.calls) == 2


async def test_async_get_reads_one_run() -> None:
    http = _AsyncStubInsightsClient(_response_body(status="active"))

    response = await _async_resource(http).get(workspace="default", name=RUN_NAME)

    assert response.job_status == "active"
