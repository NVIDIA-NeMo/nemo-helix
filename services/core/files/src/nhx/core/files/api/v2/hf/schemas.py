# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Hub compatible response schemas."""

from pydantic import BaseModel, Field


class HfSibling(BaseModel):
    """File entry in HF repo info response."""

    rfilename: str
    size: int | None = None


class HfRepoInfo(BaseModel):
    """Repository info response compatible with HF Hub API."""

    id: str  # repo_id (workspace_id/name)
    sha: str  # Commit hash (40 hex chars)
    lastModified: str  # ISO 8601 datetime
    private: bool = False
    siblings: list[HfSibling] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # For model repos
    modelId: str | None = None


class HfTreeEntry(BaseModel):
    """Tree entry for file listing."""

    type: str  # "file" or "directory"
    oid: str  # Object ID
    size: int
    path: str


class PathsInfoRequest(BaseModel):
    """Request body for paths-info endpoint."""

    paths: list[str]


class PathInfo(BaseModel):
    """Response item for paths-info endpoint."""

    path: str
    type: str  # "file"
    size: int
    oid: str
