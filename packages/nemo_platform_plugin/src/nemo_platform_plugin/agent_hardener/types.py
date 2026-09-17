# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request/response types for the Agent Hardener service.

Single source of truth for the HTTP contract. Replaces the Stainless-generated
agent_hardener resource from ``nemo_agent_hardener_plugin.sdk``.
"""

from __future__ import annotations

from datetime import datetime
from typing import NotRequired, TypedDict

from nemo_platform_plugin.schema import Page
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class AgentHardenerManifest(BaseModel):
    """Named war-game target manifest."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    workspace: str = ""
    project: str | None = None
    agent: str = ""
    source_type: str = "agent"
    project_fileset: str = ""
    agent_fileset: str = ""
    workflow: str = ""
    launch_mode: str = ""
    dockerfile: str = ""
    binaries: list[str] = Field(default_factory=list)
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


AgentHardenerManifestPage = Page[AgentHardenerManifest]


class AgentHardenerRun(BaseModel):
    """A record of one Agent Hardener war-game run."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    workspace: str = ""
    project: str | None = None
    agent: str = ""
    job_id: str = ""
    port: int = 0
    manifest: str = ""
    manifest_id: str = ""
    status: str = "failed"
    returncode: int = -1
    summary: str = ""
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


AgentHardenerRunPage = Page[AgentHardenerRun]


class ValidateModelResponse(BaseModel):
    """Verdict for a model choice."""

    ok: bool = False
    reason: str = ""
    available: list[str] = Field(default_factory=list)
    detail: str = ""


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class CreateManifestRequest(BaseModel):
    """Request body for POST /manifests."""

    model_config = ConfigDict(extra="allow")

    agent: str = ""
    source_type: str = "agent"
    project_fileset: str = ""
    agent_fileset: str = ""
    workflow: str = ""
    launch_mode: str = ""
    dockerfile: str = ""
    binaries: list[str] = Field(default_factory=list)


class UpdateManifestRequest(BaseModel):
    """Request body for PATCH /manifests/{name}."""

    model_config = ConfigDict(extra="allow")


class CreateRunRequest(BaseModel):
    """Request body for POST /runs (war-game or sanity-check)."""

    model_config = ConfigDict(extra="allow")

    manifest: str = ""
    validate_only: bool = False


class ValidateModelRequest(BaseModel):
    """Request body for POST /model-config/validate."""

    model_config = ConfigDict(extra="allow")

    model: str = ""
    base_url: str = ""
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Query parameter types
# ---------------------------------------------------------------------------


class ListManifestsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    limit: NotRequired[int]


class ListRunsQueryParams(TypedDict, total=False):
    page: NotRequired[int]
    page_size: NotRequired[int]
    limit: NotRequired[int]
    manifest_id: NotRequired[str]
