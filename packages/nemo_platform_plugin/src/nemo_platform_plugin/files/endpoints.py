# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for the Files service.

These are the single source of truth for the HTTP contract.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterable, Iterable

from nemo_platform_plugin.client.endpoint import delete, get, patch, post, put
from nemo_platform_plugin.client.types import BinaryContent, Paginated, PreparedRequest
from nemo_platform_plugin.files.types import (
    CreateFilesetRequest,
    FilesetFileOutput,
    FilesetOutput,
    ListFilesetFilesResponse,
    ListFilesetsQueryParams,
    ListFilesQueryParams,
    OtlpExportLogsResponse,
    OtlpLogQueryRequest,
    UpdateFilesetRequest,
    UploadOtlpLogsQueryParams,
)
from nemo_platform_plugin.jobs.schemas import PlatformJobLogPage

# ---------------------------------------------------------------------------
# Fileset CRUD
# ---------------------------------------------------------------------------


@get("/apis/files/v2/workspaces/{workspace}/filesets/{name}")
@abstractmethod
def get_fileset(*, workspace: str | None = None, name: str) -> FilesetOutput: ...


@get("/apis/files/v2/workspaces/{workspace}/filesets")
@abstractmethod
def list_filesets(
    *, workspace: str | None = None, query_params: ListFilesetsQueryParams | None = None
) -> Paginated[FilesetOutput]: ...


def _get_fileset_on_conflict(body: CreateFilesetRequest, workspace: str | None) -> PreparedRequest[FilesetOutput]:
    """Build the retrieve request replayed when ``create_fileset(exist_ok=True)`` 409s."""
    return get_fileset(name=body.name, workspace=workspace)


@post("/apis/files/v2/workspaces/{workspace}/filesets", get_on_conflict=_get_fileset_on_conflict)
@abstractmethod
def create_fileset(
    *, workspace: str | None = None, body: CreateFilesetRequest, exist_ok: bool = False
) -> FilesetOutput: ...


@patch("/apis/files/v2/workspaces/{workspace}/filesets/{name}")
@abstractmethod
def update_fileset(*, workspace: str | None = None, name: str, body: UpdateFilesetRequest) -> FilesetOutput: ...


@delete("/apis/files/v2/workspaces/{workspace}/filesets/{name}")
@abstractmethod
def delete_fileset(*, workspace: str | None = None, name: str) -> FilesetOutput: ...


@post("/apis/files/v2/workspaces/{workspace}/filesets/{name}/refresh")
@abstractmethod
def refresh_fileset(*, workspace: str | None = None, name: str) -> FilesetOutput:
    """Re-resolve a fileset's tracked mutable ref and repoint it at the current immutable id."""
    ...


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------


@get("/apis/files/v2/workspaces/{workspace}/filesets/{name}/files")
@abstractmethod
def list_files(
    *, workspace: str | None = None, name: str, query_params: ListFilesQueryParams | None = None
) -> ListFilesetFilesResponse: ...


@put("/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}")
@abstractmethod
def upload_file(
    *, workspace: str | None = None, name: str, path: str, content: bytes | Iterable[bytes] | AsyncIterable[bytes]
) -> FilesetFileOutput: ...


@get("/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}")
@abstractmethod
def download_file(*, workspace: str | None = None, name: str, path: str) -> BinaryContent: ...


@delete("/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{path}")
@abstractmethod
def delete_file(*, workspace: str | None = None, name: str, path: str) -> FilesetFileOutput: ...


# ---------------------------------------------------------------------------
# OTLP log operations
# ---------------------------------------------------------------------------


@post("/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs")
@abstractmethod
def upload_otlp_logs(
    *,
    workspace: str | None = None,
    name: str,
    content: bytes | Iterable[bytes] | AsyncIterable[bytes],
    query_params: UploadOtlpLogsQueryParams | None = None,
) -> OtlpExportLogsResponse: ...


@post("/apis/files/v2/workspaces/{workspace}/filesets/{name}/otlp/v1/logs/query")
@abstractmethod
def query_otlp_logs(*, workspace: str | None = None, name: str, body: OtlpLogQueryRequest) -> PlatformJobLogPage: ...
