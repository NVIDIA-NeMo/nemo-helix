# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common structures used across multiple API endpoints/schemas."""

from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Generic, List, Optional, TypeVar

from nemo_helix_plugin.schema import Page as Page
from nemo_helix_plugin.schema import PaginationData as PaginationData
from nemo_helix_plugin.schema import SecretRef as SecretRef
from nemo_helix_plugin.schema import Value
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.responses import StreamingResponse

T = TypeVar("T")

JSONL_CONTENT_TYPE = "application/ndjson"


class URN(str):
    """An absolute or relative URN for a NeMo Helix resource.

    e.g.
    meta/llama3-8b-instruct
    models/meta/llama3-8b-instruct
    urn:nemo:models/meta/llama3-8b-instruct
    urn:nemo:models/meta/llama3-8b-instruct@v2
    """


class PaginatedResult(BaseModel, Generic[T]):
    """Generic container for paginated results from the repository layer.

    This is an intermediate representation used by repositories to return
    paginated data with metadata. The service layer typically converts this
    to a full Page response by adding filter/sort context.

    Example:
        ```python
        # Repository returns PaginatedResult
        result = await repo.list_items(page=1, page_size=100)
        # result.data contains List[Item]
        # result.pagination contains PaginationData

        # Service converts to Page with additional context
        return Page(
            data=result.data,
            pagination=result.pagination,
            sort=sort,
            filter=filter_obj.model_dump(mode="json", exclude_none=True) if filter_obj else None,
        )
        ```
    """

    data: List[T]
    pagination: PaginationData


class GenericSortField(StrEnum):
    """Sort options for entity-store-backed list endpoints.

    Members map to entity base columns (see ``BASE_FIELDS``), so any
    entity-backed resource (filesets, guardrail configs, workspaces, projects)
    can share this without advertising a sort it cannot honor. Prefix a field with ``-`` for
    descending order.
    """

    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"
    NAME_ASC = "name"
    NAME_DESC = "-name"


class DeleteResponse(Value):
    message: str = Field(default="Resource deleted successfully.")
    id: str = Field(..., description="The ID of the deleted resource.")


class ErrorResponse(Value):
    detail: str = Field(
        description="A human-readable error message describing what went wrong.",
        json_schema_extra={"example": "Error message"},
    )


class FileUploadResponse(BaseModel):
    sha: str = Field(..., title="Sha", description="The SHA hash of the uploaded file content.")
    message: str = Field(..., title="Message", description="The result of the file upload.")
    path: str = Field(..., title="Path", description="The fully qualified path to the uploaded file in the repository.")
    size: int = Field(..., title="Size", description="The size of the uploaded file in bytes.")


class UploadMode(str, Enum):
    LFS = "lfs"


class SortByColumn(str, Enum):
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    NAME = "name"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class File(Value):
    path: str = Field()
    size: int = Field()
    sha: str = Field()


class FileCommitResponse(Value):
    sha: str
    message: str


class StreamingJSONLResponse(StreamingResponse):
    """
    Streaming response for JSON-Lines (`application/ndjson`).
    Accepts an async iterator / generator that already yields bytes or
    JSON-encoded strings separated by `\n`.
    """

    media_type = JSONL_CONTENT_TYPE


class DateRange(BaseModel):
    start: Optional[datetime] = Field(
        default=None,
        description="Start of the date range.",
        json_schema_extra={"format": "date-time"},
    )
    end: Optional[datetime] = Field(
        default=None,
        description="End of the date range.",
        json_schema_extra={"format": "date-time"},
    )

    @field_validator("start", "end")
    @classmethod
    def convert_to_naive_utc(cls, value):
        if value is None:
            return value

        if value.tzinfo is not None:
            # Convert to UTC first, then make naive
            return value.utctimetuple() and value.astimezone(timezone.utc).replace(tzinfo=None)

        return value

    @model_validator(mode="after")
    def validate_date_range(self):
        if self.start is not None and self.end is not None:
            if self.start >= self.end:
                raise ValueError("Start date must be before end date")
        return self
