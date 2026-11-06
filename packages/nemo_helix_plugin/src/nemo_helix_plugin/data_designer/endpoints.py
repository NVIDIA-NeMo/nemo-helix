# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Data Designer service.

The Data Designer plugin exposes:
- ``POST /preview`` and ``POST /retrieval-preview`` streaming NDJSON functions
- four job collections under ``/jobs/{job_collection}``: ``create``,
  ``retrieval-generate``, ``retrieval-prepare``, and ``retrieval-run``

The service prefix is ``/apis/data-designer/v2/workspaces/{workspace}``.
"""

from __future__ import annotations

from abc import abstractmethod

from nemo_helix_plugin.client.endpoint import delete, get, post
from nemo_helix_plugin.client.types import BinaryContent, Paginated, PreparedRequest, Stream
from nemo_helix_plugin.data_designer.types import (
    DataDesignerJobCollection,
    DataDesignerJobLogsQueryParams,
    DataDesignerJobRequest,
    DataDesignerJobResponse,
    ListDataDesignerJobsQueryParams,
    PreviewFrameData,
    PreviewRequest,
    RetrievalPreviewRequest,
)
from nemo_helix_plugin.jobs.schemas import (
    HelixJobListResultResponse,
    HelixJobLogPage,
    HelixJobResultResponse,
    HelixJobStatusResponse,
)
from nemo_helix_plugin.jobs.selectors import JobReferenceKind
from nemo_helix_plugin.jobs.types import JobSelectorQueryParams

_DATA_DESIGNER = "/apis/data-designer/v2/workspaces/{workspace}"

# ---------------------------------------------------------------------------
# Preview (streaming NDJSON)
# ---------------------------------------------------------------------------


@post("/apis/data-designer/v2/workspaces/{workspace}/preview")
@abstractmethod
def preview(*, workspace: str | None = None, body: PreviewRequest) -> Stream[PreviewFrameData]: ...


@post("/apis/data-designer/v2/workspaces/{workspace}/retrieval-preview")
@abstractmethod
def retrieval_preview(*, workspace: str | None = None, body: RetrievalPreviewRequest) -> Stream[PreviewFrameData]: ...


# ---------------------------------------------------------------------------
# Job CRUD (collections: /jobs/{job_collection})
# ---------------------------------------------------------------------------


@post(_DATA_DESIGNER + "/jobs/{job_collection}")
@abstractmethod
def create_job(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    body: DataDesignerJobRequest,
) -> DataDesignerJobResponse: ...


@get(_DATA_DESIGNER + "/jobs/{job_collection}")
@abstractmethod
def list_jobs(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    query_params: ListDataDesignerJobsQueryParams | None = None,
) -> Paginated[DataDesignerJobResponse]: ...


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{name}")
@abstractmethod
def _raw_get_job(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> DataDesignerJobResponse: ...


def get_job(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[DataDesignerJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        job_collection=job_collection,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[DataDesignerJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        job_collection=job_collection,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[DataDesignerJobResponse]:
    return _raw_get_job(
        workspace=workspace,
        job_collection=job_collection,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@delete(_DATA_DESIGNER + "/jobs/{job_collection}/{name}")
@abstractmethod
def _raw_delete_job(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> None: ...


def delete_job(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[None]:
    return _raw_delete_job(
        workspace=workspace,
        job_collection=job_collection,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def delete_job_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[None]:
    return _raw_delete_job(
        workspace=workspace,
        job_collection=job_collection,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def delete_job_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[None]:
    return _raw_delete_job(
        workspace=workspace,
        job_collection=job_collection,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@post(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/cancel")
@abstractmethod
def _raw_cancel_job(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> DataDesignerJobResponse: ...


def cancel_job(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[DataDesignerJobResponse]:
    return _raw_cancel_job(
        workspace=workspace,
        job_collection=job_collection,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def cancel_job_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[DataDesignerJobResponse]:
    return _raw_cancel_job(
        workspace=workspace,
        job_collection=job_collection,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def cancel_job_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[DataDesignerJobResponse]:
    return _raw_cancel_job(
        workspace=workspace,
        job_collection=job_collection,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/status")
@abstractmethod
def _raw_get_job_status(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobStatusResponse: ...


def get_job_status(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_job_status(
        workspace=workspace,
        job_collection=job_collection,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_status_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_job_status(
        workspace=workspace,
        job_collection=job_collection,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_status_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobStatusResponse]:
    return _raw_get_job_status(
        workspace=workspace,
        job_collection=job_collection,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/logs")
@abstractmethod
def _raw_get_job_logs(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: DataDesignerJobLogsQueryParams | None = None,
) -> HelixJobLogPage: ...


def get_job_logs(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: DataDesignerJobLogsQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobLogPage]:
    return _raw_get_job_logs(
        workspace=workspace,
        job_collection=job_collection,
        name=name,
        query_params=_compose_data_designer_job_logs_query_params(query_params, job_reference_kind),
    )


def get_job_logs_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    query_params: DataDesignerJobLogsQueryParams | None = None,
) -> PreparedRequest[HelixJobLogPage]:
    return _raw_get_job_logs(
        workspace=workspace,
        job_collection=job_collection,
        name=job_id,
        query_params=_compose_data_designer_job_logs_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_logs_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    query_params: DataDesignerJobLogsQueryParams | None = None,
) -> PreparedRequest[HelixJobLogPage]:
    return _raw_get_job_logs(
        workspace=workspace,
        job_collection=job_collection,
        name=job_name,
        query_params=_compose_data_designer_job_logs_query_params(query_params, JobReferenceKind.NAME),
    )


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/results")
@abstractmethod
def _raw_list_job_results(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobListResultResponse: ...


def list_job_results(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_job_results(
        workspace=workspace,
        job_collection=job_collection,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def list_job_results_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_job_results(
        workspace=workspace,
        job_collection=job_collection,
        name=job_id,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def list_job_results_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobListResultResponse]:
    return _raw_list_job_results(
        workspace=workspace,
        job_collection=job_collection,
        name=job_name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{job}/results/{name}")
@abstractmethod
def _raw_get_job_result(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> HelixJobResultResponse: ...


def get_job_result(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_job_result(
        workspace=workspace,
        job_collection=job_collection,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def get_job_result_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_job_result(
        workspace=workspace,
        job_collection=job_collection,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def get_job_result_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[HelixJobResultResponse]:
    return _raw_get_job_result(
        workspace=workspace,
        job_collection=job_collection,
        job=job_name,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.NAME),
    )


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{job}/results/{name}/download")
@abstractmethod
def _raw_download_job_result(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> BinaryContent: ...


def download_job_result(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
    job_reference_kind: JobReferenceKind | None = None,
) -> PreparedRequest[BinaryContent]:
    return _raw_download_job_result(
        workspace=workspace,
        job_collection=job_collection,
        job=job,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, job_reference_kind),
    )


def download_job_result_by_id(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_id: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[BinaryContent]:
    return _raw_download_job_result(
        workspace=workspace,
        job_collection=job_collection,
        job=job_id,
        name=name,
        query_params=_compose_job_selector_query_params(query_params, JobReferenceKind.ID),
    )


def download_job_result_by_name(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    job_name: str,
    name: str,
    query_params: JobSelectorQueryParams | None = None,
) -> PreparedRequest[BinaryContent]:
    return _raw_download_job_result(
        workspace=workspace,
        job_collection=job_collection,
        job=job_name,
        name=name,
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


def _compose_data_designer_job_logs_query_params(
    query: DataDesignerJobLogsQueryParams | None, mode: JobReferenceKind | None
) -> DataDesignerJobLogsQueryParams:
    result: DataDesignerJobLogsQueryParams = {} if query is None else query.copy()
    existing = result.pop("job_reference_kind", None)
    existing_mode = JobReferenceKind(existing) if existing is not None else None
    selected_mode = JobReferenceKind(mode) if mode is not None else None
    if existing_mode is not None and selected_mode is not None and existing_mode != selected_mode:
        raise ValueError("Conflicting Job reference modes")
    selected_mode = selected_mode if selected_mode is not None else existing_mode
    if selected_mode is not None:
        result["job_reference_kind"] = selected_mode
    return result
