# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed DTOs for the Customizer SDK routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, NotRequired, Self, TypedDict

from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.types import validate_output_location
from pydantic import BaseModel, Field, field_validator


class CustomizationHealthResponse(BaseModel):
    """Health payload returned by the Customizer router."""

    plugin: str
    status: str
    contributors: list[str] = Field(default_factory=list)


class CustomizationJobCreateRequest(BaseModel):
    """Body accepted by each customization backend job collection."""

    name: str | None = None
    description: str | None = None
    project: str | None = None
    spec: dict[str, Any]
    profile: str | None = None
    options: dict[str, Any] | None = None
    ownership: dict[str, Any] | None = None
    custom_fields: dict[str, Any] | None = None
    output_location: str | None = None

    _validate_output_location = field_validator("output_location")(validate_output_location)


class CustomizationJob(BaseModel):
    """Backend-specific Customizer job response."""

    id: str | None = None
    name: str
    description: str | None = None
    project: str | None = None
    workspace: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    spec: dict[str, Any] = Field(default_factory=dict)
    status: HelixJobStatus | None = None
    status_details: dict[str, Any] | None = None
    error_details: dict[str, Any] | None = None
    ownership: dict[str, Any] | None = None
    custom_fields: dict[str, Any] | None = None

    @property
    def job(self) -> Self:
        """Compatibility alias for the pre-typed-SDK job handle shape."""
        return self


class ListCustomizationJobsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


__all__ = [
    "CustomizationHealthResponse",
    "CustomizationJob",
    "CustomizationJobCreateRequest",
    "ListCustomizationJobsQueryParams",
]
