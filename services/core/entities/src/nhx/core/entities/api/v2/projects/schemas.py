# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project API Schemas for v2."""

from datetime import datetime
from typing import Optional

from nhx.common.entities.constants import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from pydantic import BaseModel, ConfigDict, Field


class ProjectInput(BaseModel):
    """Schema for creating a new project."""

    name: str = Field(
        ...,
        description=f"Project name (unique within workspace). {NAME_PATTERN_DESCRIPTION}",
        pattern=NAME_PATTERN,
        examples=["ml-project", "nlp-research"],
    )
    description: Optional[str] = Field(None, description="Optional description of the project")

    model_config = ConfigDict(regex_engine="python-re")


class ProjectUpdate(BaseModel):
    """Schema for updating a project."""

    description: Optional[str] = Field(None, description="Updated description")


class Project(BaseModel):
    """Schema for Project responses."""

    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Project name")
    workspace: str = Field(..., description="Workspace identifier")
    description: Optional[str] = Field(None, description="Project description")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ProjectFilter(BaseModel):
    """Filter for Projects."""

    name: Optional[str] = Field(None, description="Filter by project name.")
