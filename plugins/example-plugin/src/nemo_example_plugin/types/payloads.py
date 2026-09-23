# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request and response payloads for the example plugin API.

These are plain Pydantic models with no knowledge of the HTTP layer.  They
can be used anywhere in the codebase — domain logic, tests, serialization —
without pulling in transport concerns.
"""

from __future__ import annotations

from nemo_example_plugin.entities import ExampleItem
from nemo_helix_plugin.schema import NemoListResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HelloResponse(BaseModel):
    message: str


#: Paginated list of :class:`~nemo_example_plugin.entities.ExampleItem` objects.
ExampleItemPage = NemoListResponse[ExampleItem]


# ---------------------------------------------------------------------------
# Request bodies — plain Pydantic BaseModel, no shared base
# ---------------------------------------------------------------------------


class CreateExampleItemRequest(BaseModel):
    """Request body for ``POST /v2/workspaces/{workspace}/items``."""

    name: str = Field(description="Unique item name within the workspace.")
    title: str = Field(description="Short title for the item.")
    body: str = Field(default="", description="Long-form body text.")
    tags: list[str] = Field(default_factory=list, description="Searchable tags.")


class GreetRequest(BaseModel):
    """Request body for the greet function."""

    name: str = Field(default="world", description="Name to greet.")


class GreetResponse(BaseModel):
    """Response from the greet function."""

    message: str
    workspace: str


class CountRequest(BaseModel):
    """Request body for the streaming count function."""

    upto: int = Field(default=3, description="How many tick frames to emit.")


class Tick(BaseModel):
    """A frame from the count stream (tick or done)."""

    kind: str
    n: int | None = None


class BlobUploadResponse(BaseModel):
    """Response from uploading a binary blob."""

    name: str
    size: int


class UpdateExampleItemRequest(BaseModel):
    """Request body for ``PATCH /v2/workspaces/{workspace}/items/{name}``.

    All fields are optional — omitted fields are left unchanged.
    """

    title: str | None = Field(default=None, description="Updated title.")
    body: str | None = Field(default=None, description="Updated body text.")
    tags: list[str] | None = Field(default=None, description="Replacement tag list.")
