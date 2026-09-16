# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and response payloads for the customization job template routes."""

from __future__ import annotations

from typing import Any

from nemo_customizer.entities import CustomizationJobTemplate
from nemo_platform_plugin.schema import NemoListResponse
from pydantic import BaseModel, Field

#: Paginated list of :class:`~nemo_customizer.entities.CustomizationJobTemplate` objects.
CustomizationJobTemplatePage = NemoListResponse[CustomizationJobTemplate]


class CreateCustomizationJobTemplateRequest(BaseModel):
    """Request body for ``POST /v2/workspaces/{workspace}/job-templates``."""

    name: str = Field(description="Unique template name within the workspace.")
    backend: str = Field(description="Customization backend the config targets, e.g. 'automodel'.")
    config: dict[str, Any] = Field(
        description="Job input to replay, in the same shape the backend's submit accepts.",
    )
    description: str = Field(default="", description="What this template is for.")


class UpdateCustomizationJobTemplateRequest(BaseModel):
    """Request body for ``PATCH /v2/workspaces/{workspace}/job-templates/{name}``."""

    backend: str | None = Field(default=None, description="Customization backend the config targets.")
    config: dict[str, Any] | None = Field(
        default=None,
        description="Job input to replay, in the same shape the backend's submit accepts.",
    )
    description: str | None = Field(default=None, description="What this template is for.")
