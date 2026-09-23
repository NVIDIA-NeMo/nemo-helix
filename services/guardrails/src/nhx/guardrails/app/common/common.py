# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common structures used across multiple API endpoints/schemas."""

from enum import Enum

from nhx.common.entities.values import Value
from pydantic import BaseModel, Field


class URN(str):
    """An absolute or relative URN for a NeMo Helix resource.

    e.g.
    meta/llama3-8b-instruct
    models/meta/llama3-8b-instruct
    urn:nemo:models/meta/llama3-8b-instruct
    urn:nemo:models/meta/llama3-8b-instruct@v2
    """


class ErrorResponse(Value):
    detail: str = Field(json_schema_extra={"example": "Error message"})


class FileUploadResponse(BaseModel):
    sha: str = Field(..., title="Sha")
    message: str = Field(..., title="Message")
    path: str = Field(..., title="Path")
    size: int = Field(..., title="Size")


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
