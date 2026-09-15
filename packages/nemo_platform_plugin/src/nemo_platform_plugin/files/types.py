# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Files service.

These types define the HTTP contract for filesets and file operations.
Both the server (FastAPI routes) and the client (NemoClient endpoints)
import from here — one source of truth, no Stainless-generated duplicates.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, NotRequired, TypedDict

from nemo_platform_plugin.entity_naming import NAME_MAX_LENGTH, NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from nemo_platform_plugin.files.metadata import FilesetMetadata
from nemo_platform_plugin.files.storage_config import StorageConfig
from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, ConfigDict, Field


class FilesetPurpose(StrEnum):
    DATASET = "dataset"
    ENVIRONMENT = "environment"
    GENERIC = "generic"
    MODEL = "model"


class CacheStatus(StrEnum):
    """Cache status for files in external storage backends."""

    CACHED = "cached"
    CACHING = "caching"
    NOT_CACHED = "not_cached"
    NOT_CACHEABLE = "not_cacheable"


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class FilesetOutput(BaseModel):
    """Response DTO for fileset operations."""

    id: str
    name: str
    workspace: str
    description: str
    purpose: FilesetPurpose
    storage: StorageConfig
    metadata: FilesetMetadata
    custom_fields: dict[str, Any]
    project: str
    created_at: str
    updated_at: str


class FilesetFileOutput(BaseModel):
    file_ref: str
    file_url: str
    path: str
    size: int
    cache_status: CacheStatus | None = None


class ListFilesetFilesResponse(BaseModel):
    data: list[FilesetFileOutput]


FilesetPage = Page[FilesetOutput]


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------

MAX_LENGTH = 255


class CreateFilesetRequest(BaseModel):
    model_config = ConfigDict(regex_engine="python-re")

    name: str = Field(
        description=f"The name of the fileset. {NAME_PATTERN_DESCRIPTION}",
        max_length=NAME_MAX_LENGTH,
        pattern=NAME_PATTERN,
        examples=["training-data-v1", "llama-checkpoint"],
    )
    description: str | None = Field(
        default=None,
        description="The description of the fileset.",
        max_length=MAX_LENGTH,
    )
    project: str | None = Field(
        default=None,
        description="The name of the project associated with this fileset.",
    )
    storage: StorageConfig | None = Field(
        default=None,
        description="The storage configuration for the fileset. If not provided, uses default storage.",
    )
    purpose: FilesetPurpose = Field(
        default=FilesetPurpose.GENERIC,
        description="The purpose of the fileset.",
    )
    metadata: FilesetMetadata = Field(
        default_factory=FilesetMetadata,
        description="Purpose-specific metadata. Use the purpose as the key (e.g., {dataset: {...}}).",
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom fields for the fileset.",
    )
    cache: bool = Field(
        default=False,
        description="Cache all files after creation. Only applies to external storage.",
    )


class UpdateFilesetRequest(BaseModel):
    description: str | None = Field(
        default=None,
        description="The description of the fileset.",
        max_length=MAX_LENGTH,
    )
    project: str | None = Field(
        default=None,
        description="The name of the project associated with this fileset.",
    )
    purpose: FilesetPurpose | None = Field(
        default=None,
        description="The purpose of the fileset.",
    )
    metadata: FilesetMetadata | None = Field(
        default=None,
        description="Purpose-specific metadata. Use the purpose as the key (e.g., {dataset: {...}}).",
    )
    custom_fields: dict[str, Any] | None = Field(
        default=None,
        description="Custom fields for the fileset.",
    )


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListFilesetsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListFilesQueryParams(TypedDict, total=False):
    path: NotRequired[str]
    include_cache_status: NotRequired[bool]


class UploadOtlpLogsQueryParams(TypedDict, total=False):
    artifact_base_path: NotRequired[str]


# ---------------------------------------------------------------------------
# OTLP types
# ---------------------------------------------------------------------------


class OtlpLogQueryRequest(BaseModel):
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int | None = None
    page_cursor: str | None = None
    tail: int | None = None
    artifact_base_path: str | None = None


class OtlpExportLogsPartialSuccess(BaseModel):
    error_message: str | None = Field(default=None, alias="errorMessage")
    rejected_log_records: int | None = Field(default=None, alias="rejectedLogRecords")


class OtlpExportLogsResponse(BaseModel):
    partial_success: OtlpExportLogsPartialSuccess | None = Field(default=None, alias="partialSuccess")
