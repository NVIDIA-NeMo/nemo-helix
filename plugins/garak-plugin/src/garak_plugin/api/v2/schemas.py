# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Garak Plugin API request schemas.

Entity definitions (subclasses of ``NemoEntity`` stored in the entity store)
live in :mod:`garak_plugin.entities`. This module holds only the request bodies
clients send to the plugin's CRUD routes. The nested data models (system / run /
plugins / reporting) are imported directly from the entity module — they are
the same shape on the wire as at rest.
"""

from __future__ import annotations

from typing import Any

from garak_plugin.entities import (
    ScanPluginsData,
    ScanReportData,
    ScanRunData,
    ScanSystemData,
)
from nemo_helix_plugin.schema import DatetimeFilter, NemoFilter
from pydantic import BaseModel, Field


class CreateScanConfigRequest(BaseModel):
    """Request body for ``POST /v2/workspaces/{workspace}/configs``."""

    name: str = Field(description="Unique config name within the workspace.")
    description: str | None = Field(default=None, description="Human-readable description.")
    system: ScanSystemData = Field(default_factory=ScanSystemData)
    run: ScanRunData = Field(default_factory=ScanRunData)
    plugins: ScanPluginsData = Field(default_factory=ScanPluginsData)
    reporting: ScanReportData = Field(default_factory=ScanReportData)


class UpdateScanConfigRequest(BaseModel):
    """Request body for ``PUT /v2/workspaces/{workspace}/configs/{name}``."""

    description: str | None = Field(default=None)
    system: ScanSystemData = Field(default_factory=ScanSystemData)
    run: ScanRunData = Field(default_factory=ScanRunData)
    plugins: ScanPluginsData = Field(default_factory=ScanPluginsData)
    reporting: ScanReportData = Field(default_factory=ScanReportData)


class CreateScanTargetRequest(BaseModel):
    """Request body for ``POST /v2/workspaces/{workspace}/targets``."""

    name: str = Field(description="Unique target name within the workspace.")
    description: str | None = Field(default=None)
    type: str = Field(description="Target type (e.g., 'nim', 'openai').")
    model: str = Field(description="Model identifier.")
    options: dict[str, Any] = Field(default_factory=dict)


class UpdateScanTargetRequest(BaseModel):
    """Request body for ``PUT /v2/workspaces/{workspace}/targets/{name}``."""

    description: str | None = Field(default=None)
    type: str = Field(description="Target type (e.g., 'nim', 'openai').")
    model: str = Field(description="Model identifier.")
    options: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filters — extend NemoFilter so extra fields are rejected (extra="forbid")
# ---------------------------------------------------------------------------


class TargetFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/targets``."""

    type: str | None = Field(
        default=None,
        description="Filter to targets of this type (e.g., 'nim', 'openai').",
    )
    model: str | None = Field(
        default=None,
        description="Filter to targets using this model identifier.",
    )
    description: str | None = Field(
        default=None,
        description="Filter to targets with this description.",
    )
    project: str | None = Field(
        default=None,
        description="Filter to targets in this project.",
    )
    created_at: DatetimeFilter | None = Field(
        default=None,
        description="Filter on creation time (supports $gte / $lte).",
    )
    updated_at: DatetimeFilter | None = Field(
        default=None,
        description="Filter on last-update time (supports $gte / $lte).",
    )


class ConfigFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/configs``."""

    description: str | None = Field(
        default=None,
        description="Filter to configs with this description.",
    )
    project: str | None = Field(
        default=None,
        description="Filter to configs in this project.",
    )
    created_at: DatetimeFilter | None = Field(
        default=None,
        description="Filter on creation time (supports $gte / $lte).",
    )
    updated_at: DatetimeFilter | None = Field(
        default=None,
        description="Filter on last-update time (supports $gte / $lte).",
    )
