# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and response schemas for filesets API.

Response types (FilesetOutput, FilesetFileOutput, ListFilesetFilesResponse) and
request types (CreateFilesetRequest, UpdateFilesetRequest) are imported from
``nemo_helix_plugin.files.types`` — the shared single source of truth.

This module adds server-specific concerns: converter functions that map domain
entities to response DTOs, the FilesetFilter schema, and the FilesetPage alias.
"""

from typing import Annotated, Optional

from nemo_helix_plugin.files.types import CacheStatus
from nemo_helix_plugin.files.types import CreateFilesetRequest as CreateFilesetRequest
from nemo_helix_plugin.files.types import FilesetFileOutput as FilesetFileOutput
from nemo_helix_plugin.files.types import FilesetOutput as FilesetOutput
from nemo_helix_plugin.files.types import ListFilesetFilesResponse as ListFilesetFilesResponse
from nemo_helix_plugin.files.types import UpdateFilesetRequest as UpdateFilesetRequest
from nhx.common.api.common import Page
from nhx.common.entities.values import DatetimeFilter, Filter, StringFilter, map_entity_field
from nhx.core.files.app.backends import FileInfo
from nhx.core.files.app.backends.base import StorageConfigType
from nhx.core.files.entities import Fileset, FilesetPurpose
from pydantic import Field

FilesetPage = Page[FilesetOutput]


# ---------------------------------------------------------------------------
# Entity → DTO converters
# ---------------------------------------------------------------------------


def fileset_output_from_entity(entity: Fileset) -> FilesetOutput:
    """Convert a Fileset domain entity to a FilesetOutput response DTO."""
    return FilesetOutput(
        id=entity.id,
        name=entity.name,
        workspace=entity.workspace,
        description=entity.description or "",
        purpose=entity.purpose,
        storage=entity.storage,
        metadata=entity.metadata,
        custom_fields=entity.custom_fields,
        project=entity.project or "",
        created_at=entity.created_at.isoformat() if entity.created_at else "",
        updated_at=entity.updated_at.isoformat() if entity.updated_at else "",
    )


def fileset_file_output_from_info(
    workspace: str,
    name: str,
    file_info: FileInfo,
    cache_status: CacheStatus | None = None,
) -> FilesetFileOutput:
    """Convert a FileInfo to a FilesetFileOutput response DTO."""
    return FilesetFileOutput(
        file_url=f"/apis/files/v2/workspaces/{workspace}/filesets/{name}/-/{file_info.path}",
        file_ref=f"{workspace}/{name}#{file_info.path}",
        path=file_info.path,
        size=file_info.size,
        cache_status=cache_status,
    )


def list_fileset_files_from_infos(
    fileset: Fileset,
    file_infos: list[FileInfo],
    cache_status_map: dict[str, CacheStatus] | None = None,
) -> ListFilesetFilesResponse:
    """Convert a list of FileInfos to a ListFilesetFilesResponse."""
    cache_status_map = cache_status_map or {}
    return ListFilesetFilesResponse(
        data=[
            fileset_file_output_from_info(
                fileset.workspace,
                fileset.name,
                fi,
                cache_status=cache_status_map.get(fi.path),
            )
            for fi in file_infos
        ]
    )


# ---------------------------------------------------------------------------
# Filter schema (server-only, not shared with client)
# ---------------------------------------------------------------------------


class FilesetFilter(Filter):
    """Filter schema for listing filesets."""

    name: StringFilter | str | None = Field(default=None, description="Filter by fileset name.")
    description: StringFilter | str | None = Field(default=None, description="Filter by fileset description.")
    purpose: Optional[FilesetPurpose] = Field(
        default=None,
        description="Filter by the purpose of the fileset (e.g., 'dataset', 'generic').",
    )
    storage_type: Annotated[Optional[StorageConfigType], map_entity_field("data.storage.type")] = Field(
        default=None,
        description="Filter by the storage backend type (e.g., 'local', 'ngc').",
    )
    created_at: Optional[DatetimeFilter] = Field(
        default=None,
        description="Filter by creation date. Supports '$gte' (on or after) and '$lte' (on or before) datetime filters.",
    )
    updated_at: Optional[DatetimeFilter] = Field(
        default=None,
        description="Filter by update date. Supports '$gte' (on or after) and '$lte' (on or before) datetime filters.",
    )
