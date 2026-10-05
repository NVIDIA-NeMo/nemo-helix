# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity API Schemas for v2."""

from typing import Any, Dict

from nhx.common.entities.constants import NAME_PATTERN, NAME_PATTERN_DESCRIPTION
from pydantic import BaseModel, ConfigDict, Field


class EntityCreateInput(BaseModel):
    """Schema for creating a new entity (name-based routes).

    Name is optional - if not provided, it will be auto-generated.
    Workspace and entity_type come from the URL path parameters.
    """

    name: str | None = Field(
        default=None,
        description=f"Entity name (optional - auto-generated if not provided). {NAME_PATTERN_DESCRIPTION}",
        pattern=NAME_PATTERN,
        examples=["my-config", "baseline-model-v1"],
    )
    parent: str | None = Field(
        default=None,
        description="Parent entity ID for nested entities",
    )
    project: str | None = Field(
        default=None,
        description="The name of the project associated with this entity",
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Entity-specific data (schema is opaque to entity store, validated by client SDK)",
    )

    model_config = ConfigDict(regex_engine="python-re")


class EntityUpdate(BaseModel):
    """Schema for updating an entity."""

    new_name: str | None = Field(
        default=None,
        description=f"Updated entity name (optional). {NAME_PATTERN_DESCRIPTION}",
        pattern=NAME_PATTERN,
        examples=["my-config", "baseline-model-v1"],
    )
    project: str | None = Field(
        default=None,
        description="The name of the project associated with this entity",
    )
    data: Dict[str, Any] = Field(
        ...,
        description="Updated entity-specific data",
    )
    expected_db_version: int | None = Field(
        default=None,
        description="Optional database version for optimistic locking. Update only succeeds if current version matches.",
    )

    model_config = ConfigDict(regex_engine="python-re")
