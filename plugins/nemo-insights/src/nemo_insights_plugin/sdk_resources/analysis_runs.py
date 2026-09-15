# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK sub-resources for on-demand Insights analysis runs.

Mounted as ``client.insights.analysis_runs``. Each method maps 1:1 onto the
FastAPI routes in :mod:`nemo_insights_plugin.analysis_runs`, plus one
convenience verb — :meth:`_AnalysisRunResource.wait` — that polls ``get``
until the backing job reaches a terminal state. Waiting belongs here rather
than in each caller because the run/job link is derived from the shared name:
a caller polling on its own would have to know that rule.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import NotRequired, Protocol, TypedDict

from nemo_insights_plugin.entities import AnalysisRun
from nemo_insights_plugin.schema import (
    AnalysisRunPage,
    AnalysisRunResponse,
    CreateAnalysisRunRequest,
)
from nemo_insights_plugin.sdk_resources._entity import entity_from_response, hydrate_page, object_dict
from nemo_insights_plugin.sdk_resources._errors import httpx_status_errors
from nemo_insights_plugin.types import ListAnalysisRunsQueryParams

DEFAULT_WAIT_TIMEOUT = 900.0
DEFAULT_POLL_INTERVAL = 5.0


class AnalysisRunNotSubmittedError(RuntimeError):
    """No job exists under the run's name, so waiting on it would never end.

    Insights names a run before submitting its job and the job takes that same
    name, so a missing job is proof the submission never landed — a resubmit
    case, not a slow one.
    """


class AnalysisRunTimeoutError(TimeoutError):
    """The backing job did not reach a terminal state within the wait budget."""


class _HTTPResponse(Protocol):
    def json(self) -> dict[str, object]: ...


class _TypedResponse(Protocol):
    http_response: _HTTPResponse


class _PaginatedTypedResponse(_TypedResponse, Protocol):
    def page(self) -> object: ...


class _CreateAnalysisRunFields(TypedDict):
    agent: str
    default_model: str
    fast_model: str
    ethos: NotRequired[str]
    since: NotRequired[datetime]
    evaluation_id: NotRequired[str]
    timeout_seconds: NotRequired[float]


class _AnalysisRunsClient(Protocol):
    def create_analysis_run(self, *, workspace: str, body: CreateAnalysisRunRequest) -> _TypedResponse: ...

    def list_analysis_runs(
        self, *, workspace: str, query_params: ListAnalysisRunsQueryParams
    ) -> _PaginatedTypedResponse: ...

    def get_analysis_run(self, *, workspace: str, name: str) -> _TypedResponse: ...


class _AsyncAnalysisRunsClient(Protocol):
    def create_analysis_run(self, *, workspace: str, body: CreateAnalysisRunRequest) -> Awaitable[_TypedResponse]: ...

    def list_analysis_runs(
        self, *, workspace: str, query_params: ListAnalysisRunsQueryParams
    ) -> Awaitable[_PaginatedTypedResponse]: ...

    def get_analysis_run(self, *, workspace: str, name: str) -> Awaitable[_TypedResponse]: ...


class _ResourceParent(Protocol):
    """The slice of the insights SDK namespace this sub-resource needs."""

    @property
    def _client(self) -> _AnalysisRunsClient: ...


class _AsyncResourceParent(Protocol):
    @property
    def _client(self) -> _AsyncAnalysisRunsClient: ...


def _build_create_body(
    *,
    agent: str,
    default_model: str,
    fast_model: str,
    ethos: str | None,
    since: datetime | None,
    evaluation_id: str | None,
    timeout_seconds: float | None,
) -> CreateAnalysisRunRequest:
    body: _CreateAnalysisRunFields = {
        "agent": agent,
        "default_model": default_model,
        "fast_model": fast_model,
    }
    if ethos is not None:
        body["ethos"] = ethos
    if since is not None:
        body["since"] = since
    if evaluation_id is not None:
        body["evaluation_id"] = evaluation_id
    if timeout_seconds is not None:
        body["timeout_seconds"] = timeout_seconds
    return CreateAnalysisRunRequest(**body)


def _list_params(
    *,
    page: int,
    page_size: int,
    sort: str,
    agent: str | None,
) -> ListAnalysisRunsQueryParams:
    params: ListAnalysisRunsQueryParams = {"page": page, "page_size": page_size, "sort": sort}
    if agent is not None:
        params["agent"] = agent
    return params


def _run_response_from_response(data: dict[str, object]) -> AnalysisRunResponse:
    """Parse a run-plus-job body, preserving the run's store-assigned metadata."""
    response = AnalysisRunResponse.model_validate(data)
    raw_run = object_dict(data.get("run"))
    if raw_run is not None:
        response.run = entity_from_response(AnalysisRun, raw_run)
    return response


def _page_from_response(data: dict[str, object]) -> AnalysisRunPage:
    page = AnalysisRunPage.model_validate(data)
    hydrate_page(page.data, data.get("data"))
    return page


def _wait_deadline(timeout: float) -> float:
    return time.monotonic() + timeout


def _timed_out(deadline: float) -> bool:
    return time.monotonic() >= deadline


def _check_waitable(response: AnalysisRunResponse) -> None:
    if response.job is None:
        raise AnalysisRunNotSubmittedError(
            f"Analysis run '{response.run.name}' has no backing job — its submission never "
            "landed, so it will never reach a terminal state. Resubmit it."
        )


def _wait_timeout_error(response: AnalysisRunResponse, timeout: float) -> AnalysisRunTimeoutError:
    return AnalysisRunTimeoutError(
        f"Analysis run '{response.run.name}' did not finish within {timeout}s "
        f"(last job status: {response.job_status!r})."
    )


class _AnalysisRunResource:
    """Sync ``analysis_runs`` sub-resource."""

    def __init__(self, parent: _ResourceParent) -> None:
        self._parent = parent

    @property
    def _client(self) -> _AnalysisRunsClient:
        return self._parent._client

    def create(
        self,
        *,
        workspace: str,
        agent: str,
        default_model: str,
        fast_model: str,
        ethos: str | None = None,
        since: datetime | None = None,
        evaluation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AnalysisRunResponse:
        """Submit an analysis run. The model pair is required — see the route."""
        with httpx_status_errors():
            response = self._client.create_analysis_run(
                workspace=workspace,
                body=_build_create_body(
                    agent=agent,
                    default_model=default_model,
                    fast_model=fast_model,
                    ethos=ethos,
                    since=since,
                    evaluation_id=evaluation_id,
                    timeout_seconds=timeout_seconds,
                ),
            )
            return _run_response_from_response(response.http_response.json())

    def list_runs(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
        agent: str | None = None,
    ) -> AnalysisRunPage:
        """List analysis runs. Job state is not joined — read one run to get it."""
        with httpx_status_errors():
            response = self._client.list_analysis_runs(
                workspace=workspace,
                query_params=_list_params(page=page, page_size=page_size, sort=sort, agent=agent),
            )
            response.page()
            return _page_from_response(response.http_response.json())

    def get(self, *, workspace: str, name: str) -> AnalysisRunResponse:
        """Get one analysis run joined with the live state of its backing job."""
        with httpx_status_errors():
            response = self._client.get_analysis_run(workspace=workspace, name=name)
            return _run_response_from_response(response.http_response.json())

    def wait(
        self,
        *,
        workspace: str,
        name: str,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        on_status: Callable[[str | None], None] | None = None,
    ) -> AnalysisRunResponse:
        """Poll a run until its job finishes, then return the final state.

        Raises :class:`AnalysisRunNotSubmittedError` if the run has no job and
        :class:`AnalysisRunTimeoutError` if *timeout* elapses first.
        """
        deadline = _wait_deadline(timeout)
        last_status: str | None = ""
        while True:
            response = self.get(workspace=workspace, name=name)
            _check_waitable(response)
            if on_status is not None and response.job_status != last_status:
                on_status(response.job_status)
            last_status = response.job_status
            if response.job_is_terminal:
                return response
            if _timed_out(deadline):
                raise _wait_timeout_error(response, timeout)
            time.sleep(poll_interval)


class _AsyncAnalysisRunResource:
    """Async ``analysis_runs`` sub-resource — mirrors :class:`_AnalysisRunResource`."""

    def __init__(self, parent: _AsyncResourceParent) -> None:
        self._parent = parent

    @property
    def _client(self) -> _AsyncAnalysisRunsClient:
        return self._parent._client

    async def create(
        self,
        *,
        workspace: str,
        agent: str,
        default_model: str,
        fast_model: str,
        ethos: str | None = None,
        since: datetime | None = None,
        evaluation_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AnalysisRunResponse:
        """Submit an analysis run. The model pair is required — see the route."""
        with httpx_status_errors():
            response = await self._client.create_analysis_run(
                workspace=workspace,
                body=_build_create_body(
                    agent=agent,
                    default_model=default_model,
                    fast_model=fast_model,
                    ethos=ethos,
                    since=since,
                    evaluation_id=evaluation_id,
                    timeout_seconds=timeout_seconds,
                ),
            )
            return _run_response_from_response(response.http_response.json())

    async def list_runs(
        self,
        *,
        workspace: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
        agent: str | None = None,
    ) -> AnalysisRunPage:
        """List analysis runs. Job state is not joined — read one run to get it."""
        with httpx_status_errors():
            response = await self._client.list_analysis_runs(
                workspace=workspace,
                query_params=_list_params(page=page, page_size=page_size, sort=sort, agent=agent),
            )
            response.page()
            return _page_from_response(response.http_response.json())

    async def get(self, *, workspace: str, name: str) -> AnalysisRunResponse:
        """Get one analysis run joined with the live state of its backing job."""
        with httpx_status_errors():
            response = await self._client.get_analysis_run(workspace=workspace, name=name)
            return _run_response_from_response(response.http_response.json())

    async def wait(
        self,
        *,
        workspace: str,
        name: str,
        timeout: float = DEFAULT_WAIT_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        on_status: Callable[[str | None], None] | None = None,
    ) -> AnalysisRunResponse:
        """Poll a run until its job finishes, then return the final state.

        Raises :class:`AnalysisRunNotSubmittedError` if the run has no job and
        :class:`AnalysisRunTimeoutError` if *timeout* elapses first.
        """
        deadline = _wait_deadline(timeout)
        last_status: str | None = ""
        while True:
            response = await self.get(workspace=workspace, name=name)
            _check_waitable(response)
            if on_status is not None and response.job_status != last_status:
                on_status(response.job_status)
            last_status = response.job_status
            if response.job_is_terminal:
                return response
            if _timed_out(deadline):
                raise _wait_timeout_error(response, timeout)
            await asyncio.sleep(poll_interval)


__all__ = [
    "AnalysisRunNotSubmittedError",
    "AnalysisRunTimeoutError",
    "_AnalysisRunResource",
    "_AsyncAnalysisRunResource",
]
