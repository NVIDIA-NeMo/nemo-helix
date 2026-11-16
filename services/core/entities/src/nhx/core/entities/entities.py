# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for entities service.

These are the domain models used by the API and service layers.
Separate from SQLAlchemy database models.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class WorkspaceDeletionStage(StrEnum):
    PENDING = "pending"
    DELETING = "deleting"
    FAILED = "failed"


class WorkspaceLimits(BaseModel):
    """Optional per-workspace resource limits.

    A field left unset means "use the platform default"; the consumer applies
    that default, it is never written back to the workspace record.
    """

    max_concurrent_jobs: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of jobs the workspace may have admitted concurrently. "
        "When unset, the Jobs service default applies.",
    )

    model_config = ConfigDict(extra="forbid")


class Workspace(BaseModel):
    """Workspace schema for API responses."""

    id: str = Field(..., description="System-generated UUID")
    name: str = Field(..., description="Workspace name (user-provided)")
    description: Optional[str] = Field(None, description="Optional description")
    limits: Optional[WorkspaceLimits] = Field(None, description="Optional per-workspace resource limits")

    created_at: datetime = Field(..., description="Timestamp of workspace creation")
    created_by: str | None = Field(default=None, description="Principal id for workspace creator")
    updated_at: datetime = Field(..., description="Timestamp of last workspace update")
    updated_by: str | None = Field(default=None, description="Principal id for last workspace update")

    _deletion_stage: WorkspaceDeletionStage | None = PrivateAttr(None)
    _deletion_error: str | None = PrivateAttr(None)

    model_config = ConfigDict(extra="forbid", from_attributes=True)


class Entity(BaseModel):
    """Entity schema for API responses."""

    entity_type: str = Field(..., description="Entity type identifier")
    id: str = Field(..., description="UUID identifier")
    workspace: str = Field(..., description="Workspace identifier")
    parent: Optional[str] = Field(None, description="Parent entity ID for nested entities")
    project: Optional[str] = Field(None, description="The name of the project associated with this entity")

    name: str = Field(..., description="Entity name")
    data: Dict[str, Any] = Field(..., description="Entity data")
    created_at: datetime = Field(..., description="Timestamp of entity creation")
    created_by: str | None = Field(default=None, description="Principal id for entity creator")
    updated_at: datetime = Field(..., description="Timestamp of last entity update")
    updated_by: str | None = Field(default=None, description="Principal id for last entity update")
    db_version: int = Field(..., description="Database version of the entity for optimistic locking.")

    model_config = ConfigDict(extra="forbid", from_attributes=True)
