# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wire shapes for the Inference Gateway proxy routes.

The gateway forwards arbitrary OpenAI-compatible (or provider-native) payloads,
so request and response bodies are open JSON objects rather than fixed models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel

JsonObject = dict[str, Any]


class JsonBody(RootModel[JsonObject]):
    """Arbitrary JSON object request body for proxied inference calls."""


class OpenAIModel(BaseModel):
    """One entry of an OpenAI-compatible ``/v1/models`` listing."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str | None = None


class OpenAIModelList(BaseModel):
    """OpenAI-compatible ``/v1/models`` response."""

    model_config = ConfigDict(extra="allow")

    data: list[OpenAIModel] = Field(default_factory=list)
    object: str = "list"


class ProviderReadyResponse(BaseModel):
    """Gateway readiness payload for a registered provider."""

    model_config = ConfigDict(extra="allow")

    workspace: str | None = None
    name: str | None = None
