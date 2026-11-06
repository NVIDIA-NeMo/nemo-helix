# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Safe Synthesizer plugin API."""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import get, post
from nemo_helix_plugin.client.types import Paginated, PreparedRequest
from nemo_helix_plugin.jobs.selectors import JobReferenceKind
from nemo_helix_plugin.jobs.types import JobSelectorQueryParams
from nemo_safe_synthesizer_plugin.sdk.types import (
    CreateSafeSynthesizerJobRequest,
    ListSafeSynthesizerJobsQueryParams,
    SafeSynthesizerJobResponse,
)

_JOBS = "/apis/safe-synthesizer/v2/workspaces/{workspace}/jobs"


@post(_JOBS)
@abstractmethod
def create_job(
    *, workspace: str | None = None, body: CreateSafeSynthesizerJobRequest
) -> SafeSynthesizerJobResponse: ...


@get(_JOBS)
@abstractmethod
def list_jobs(
    *, workspace: str | None = None, query_params: ListSafeSynthesizerJobsQueryParams | None = None
) -> Paginated[SafeSynthesizerJobResponse]: ...


@get(_JOBS + "/{name}")
@abstractmethod
def _raw_get_job(
    *, workspace: str | None = None, name: str, query_params: JobSelectorQueryParams | None = None
) -> SafeSynthesizerJobResponse: ...


def get_job(
    *,
    workspace: str | None = None,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[SafeSynthesizerJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_by_id(
    *, workspace: str | None = None, job_id: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[SafeSynthesizerJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_by_name(
    *, workspace: str | None = None, job_name: str, query_params: JobSelectorQueryParams | None = None
) -> PreparedRequest[SafeSynthesizerJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


def _compose_job_selector_query_params(
    query: JobSelectorQueryParams | None, mode: JobReferenceKind | None
) -> JobSelectorQueryParams:
    result: JobSelectorQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result
