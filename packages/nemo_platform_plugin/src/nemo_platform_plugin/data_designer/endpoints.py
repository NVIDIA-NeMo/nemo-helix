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

from nemo_platform_plugin.client.endpoint import delete, get, post
from nemo_platform_plugin.client.types import BinaryContent, Paginated, Stream
from nemo_platform_plugin.data_designer.types import (
    DataDesignerJobCollection,
    DataDesignerJobLogsQueryParams,
    DataDesignerJobRequest,
    DataDesignerJobResponse,
    ListDataDesignerJobsQueryParams,
    PreviewFrameData,
    PreviewRequest,
    RetrievalPreviewRequest,
)
from nemo_platform_plugin.jobs.schemas import (
    PlatformJobListResultResponse,
    PlatformJobLogPage,
    PlatformJobResultResponse,
    PlatformJobStatusResponse,
)

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
def get_job(
    *, workspace: str | None = None, job_collection: DataDesignerJobCollection = "create", name: str
) -> DataDesignerJobResponse: ...


@delete(_DATA_DESIGNER + "/jobs/{job_collection}/{name}")
@abstractmethod
def delete_job(
    *, workspace: str | None = None, job_collection: DataDesignerJobCollection = "create", name: str
) -> None: ...


@post(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/cancel")
@abstractmethod
def cancel_job(
    *, workspace: str | None = None, job_collection: DataDesignerJobCollection = "create", name: str
) -> DataDesignerJobResponse: ...


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/status")
@abstractmethod
def get_job_status(
    *, workspace: str | None = None, job_collection: DataDesignerJobCollection = "create", name: str
) -> PlatformJobStatusResponse: ...


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/logs")
@abstractmethod
def get_job_logs(
    *,
    workspace: str | None = None,
    job_collection: DataDesignerJobCollection = "create",
    name: str,
    query_params: DataDesignerJobLogsQueryParams | None = None,
) -> PlatformJobLogPage: ...


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{name}/results")
@abstractmethod
def list_job_results(
    *, workspace: str | None = None, job_collection: DataDesignerJobCollection = "create", name: str
) -> PlatformJobListResultResponse: ...


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{job}/results/{name}")
@abstractmethod
def get_job_result(
    *, workspace: str | None = None, job_collection: DataDesignerJobCollection = "create", job: str, name: str
) -> PlatformJobResultResponse: ...


@get(_DATA_DESIGNER + "/jobs/{job_collection}/{job}/results/{name}/download")
@abstractmethod
def download_job_result(
    *, workspace: str | None = None, job_collection: DataDesignerJobCollection = "create", job: str, name: str
) -> BinaryContent: ...
