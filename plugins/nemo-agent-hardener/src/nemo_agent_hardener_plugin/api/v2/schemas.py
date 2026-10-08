# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent Hardener plugin API request/query schemas.

The persisted entities are the same shape on the wire as at rest, so the read routes return them
directly. This module holds the list-endpoint query filters (extending ``NemoFilter``,
``extra="forbid"`` so a misspelled key 422s) plus the ``POST /manifests`` init request body.
"""

from __future__ import annotations

from nemo_platform_plugin.agent_hardener.types import (
    ApplyMitigationRequest,
    ApplyMitigationResponse,
    ComposeDefenseRequest,
    ComposeDefenseResponse,
    InspectAgentRequest,
    InspectAgentResponse,
    InspectProjectRequest,
    InspectProjectResponse,
    ManifestFilter,
    ManifestInit,
    ManifestUpdate,
    RunFilter,
    ValidateModelRequest,
    ValidateModelResponse,
)

__all__ = [
    "ApplyMitigationRequest",
    "ApplyMitigationResponse",
    "ComposeDefenseRequest",
    "ComposeDefenseResponse",
    "InspectAgentRequest",
    "InspectAgentResponse",
    "InspectProjectRequest",
    "InspectProjectResponse",
    "ManifestFilter",
    "ManifestInit",
    "ManifestUpdate",
    "RunFilter",
    "ValidateModelRequest",
    "ValidateModelResponse",
]
