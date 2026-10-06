# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Garak Plugin service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
garak_plugin resource from ``garak_plugin.sdk``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, NotRequired, TypedDict

from nemo_helix_plugin.schema import Page
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class ScanConfig(BaseModel):
    """Scan configuration entity response."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    workspace: str = ""
    project: str | None = None
    description: str | None = None
    system: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)
    plugins: dict[str, Any] = Field(default_factory=dict)
    reporting: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


ScanConfigPage = Page[ScanConfig]


class ScanTarget(BaseModel):
    """Scan target entity response."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    workspace: str = ""
    project: str | None = None
    description: str | None = None
    type: str = ""
    model: str = ""
    options: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


ScanTargetPage = Page[ScanTarget]


class ScanJobResponse(BaseModel):
    """Scan job submission/list response."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    status: str = ""


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class CreateScanConfigRequest(BaseModel):
    """Request body for POST /configs."""

    name: str
    description: str | None = None
    system: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)
    plugins: dict[str, Any] = Field(default_factory=dict)
    reporting: dict[str, Any] = Field(default_factory=dict)


class UpdateScanConfigRequest(BaseModel):
    """Request body for PUT /configs/{name}."""

    description: str | None = None
    system: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)
    plugins: dict[str, Any] = Field(default_factory=dict)
    reporting: dict[str, Any] = Field(default_factory=dict)


class CreateScanTargetRequest(BaseModel):
    """Request body for POST /targets."""

    name: str
    description: str | None = None
    type: str
    model: str
    options: dict[str, Any] = Field(default_factory=dict)


class UpdateScanTargetRequest(BaseModel):
    """Request body for PUT /targets/{name}."""

    description: str | None = None
    type: str
    model: str
    options: dict[str, Any] = Field(default_factory=dict)


class SubmitScanRequest(BaseModel):
    """Request body for POST /jobs/scan."""

    spec: dict[str, Any]


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListScanConfigsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListScanTargetsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]


class ListScanJobsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
