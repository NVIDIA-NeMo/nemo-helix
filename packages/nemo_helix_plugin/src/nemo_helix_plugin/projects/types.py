# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for Projects (Entity Store).

Single source of truth for the HTTP contract. Replaces the Stainless-generated
``nemo_helix.types.projects`` module.
"""

from __future__ import annotations

from datetime import datetime
from typing import NotRequired, TypedDict

from nemo_helix_plugin.schema import Page
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class Project(BaseModel):
    """Schema for Project responses."""

    id: str
    name: str
    workspace: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime


ProjectPage = Page[Project]


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class CreateProjectRequest(BaseModel):
    """Schema for creating a new project."""

    name: str
    description: str | None = None


class UpdateProjectRequest(BaseModel):
    """Schema for updating a project."""

    description: str | None = None


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListProjectsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
