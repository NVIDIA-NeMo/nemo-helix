# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared request and query types for Inference Gateway VirtualModel CRUD.

The concrete VirtualModel response and nested middleware models live in
``inference_middleware_models`` for CRUD/API schema code. Middleware hook code
imports the lightweight ``VirtualModel`` protocol from ``inference_middleware``.
This module re-exports the concrete CRUD types alongside the request contract
used by both the Inference Gateway router and the typed client.

Entity-store filter models remain server-side. Clients pass the public filter
expression as a string through :class:`ListVirtualModelsQueryParams`.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from nemo_platform_plugin.inference_middleware_models import (
    _AUTOPROVISIONED_DESC,
    MiddlewareCall,
    VirtualModel,
    VirtualModelInferenceConfig,
)
from pydantic import BaseModel, Field

__all__ = [
    "CreateVirtualModelRequest",
    "DeleteVirtualModelQueryParams",
    "ListVirtualModelsQueryParams",
    "MiddlewareCall",
    "UpdateVirtualModelRequest",
    "VirtualModel",
    "VirtualModelInferenceConfig",
]


class _VirtualModelFields(BaseModel):
    """Mutable fields shared by create and update requests."""

    default_model_entity: str | None = Field(
        default=None,
        description=(
            'Model entity to route to, in "workspace/name" format. Written into request["model"] '
            "before the request middleware pipeline runs. If omitted, a request middleware plugin "
            "must handle backend routing itself. Set to null to clear an existing value."
        ),
    )
    autoprovisioned: bool = Field(
        default=False,
        description=_AUTOPROVISIONED_DESC,
    )
    models: list[VirtualModelInferenceConfig] = Field(
        default_factory=list,
        description=(
            "Model entity references used by this VirtualModel. A per-entry backend_format overrides the referenced "
            "ModelEntity backend_format when IGW resolves the backend format for a request."
        ),
    )
    request_middleware: list[MiddlewareCall] = Field(
        default_factory=list,
        description=(
            "Ordered list of middleware plugins applied before proxying to the backend. "
            'Each entry is a MiddlewareCall with a "name" (plugin identifier) and optional '
            '"config_type" and "config_id" fields that reference a stored plugin configuration.'
        ),
    )
    response_middleware: list[MiddlewareCall] = Field(
        default_factory=list,
        description=(
            "Ordered list of middleware plugins applied after the backend response is received, "
            "before returning it to the caller."
        ),
    )
    post_response_middleware: list[MiddlewareCall] = Field(
        default_factory=list,
        description=(
            "Ordered list of middleware plugins invoked after the response has been returned to "
            "the caller. Intended for fire-and-forget work (logging, analytics) that must not "
            "block or modify the response."
        ),
    )
    override_proxy: str | None = Field(
        default=None,
        description=(
            "Plugin-provided proxy implementation for IGW to use instead of its default aiohttp proxy. "
            'Format: "plugin-name.proxy-name". Leave unset to use the default IGW proxy. '
            "Set to null to clear an existing value."
        ),
    )


class CreateVirtualModelRequest(_VirtualModelFields):
    """Request body for creating a new VirtualModel."""

    name: str = Field(
        description="Name of the virtual model within the workspace. Must be unique per workspace.",
    )


class UpdateVirtualModelRequest(_VirtualModelFields):
    """Request body for partially updating an existing VirtualModel (PATCH).

    Only fields present in the request body are updated.  Omitted fields
    retain their current values.  ``model_fields_set`` is used in the handler
    to distinguish an intentional ``[]`` (clear the list) from a missing field
    (leave unchanged).  Set ``default_model_entity`` or ``override_proxy`` to
    ``null`` explicitly to clear them.
    """


class ListVirtualModelsQueryParams(TypedDict, total=False):
    """Query parameters accepted by the VirtualModel list operation."""

    page: NotRequired[int]
    page_size: NotRequired[int]
    sort: NotRequired[str]
    filter: NotRequired[str]
    exclude_autoprovisioned: NotRequired[bool]


class DeleteVirtualModelQueryParams(TypedDict, total=False):
    """Query parameters accepted by the VirtualModel delete operation."""

    expected_db_version: NotRequired[int]
