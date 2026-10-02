# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for Guardrails config endpoints."""

from typing import Any, Dict, Optional

from nhx.common.entities.values import DatetimeFilter, Filter, StringFilter
from pydantic import BaseModel, Field


class GuardrailConfigInput(BaseModel):
    """Input schema for creating a guardrail config."""

    name: str = Field(..., description="The name of the guardrail config")
    # Note: namespace comes from workspace_id in the URL path, not the body
    description: Optional[str] = Field(default=None, description="Description of the guardrail config")
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Guardrail configuration data",
        json_schema_extra={"type": "object"},
    )


class GuardrailConfigUpdate(BaseModel):
    """Input schema for updating a guardrail config."""

    description: Optional[str] = Field(default=None, description="Description of the guardrail config")
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Guardrail configuration data",
        json_schema_extra={"type": "object"},
    )


class GuardrailConfigFilter(Filter):
    """Filter schema for listing guardrail configs."""

    name: StringFilter | str | None = Field(default=None, description="Filter by config name.")
    description: StringFilter | str | None = Field(default=None, description="Filter by config description.")
    project: Optional[str] = Field(default=None, description="Filter by project name.")
    created_at: Optional[DatetimeFilter] = Field(
        default=None,
        description="Filter by creation date. Supports '$gte' (on or after) and '$lte' (on or before) datetime filters.",
    )
    updated_at: Optional[DatetimeFilter] = Field(
        default=None,
        description="Filter by update date. Supports '$gte' (on or after) and '$lte' (on or before) datetime filters.",
    )
