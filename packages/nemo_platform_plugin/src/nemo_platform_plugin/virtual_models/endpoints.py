# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for Inference Gateway VirtualModel CRUD."""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.client.endpoint import delete, get, patch, post
from nemo_platform_plugin.client.types import Paginated, PreparedRequest
from nemo_platform_plugin.virtual_models.types import (
    CreateVirtualModelRequest,
    DeleteVirtualModelQueryParams,
    ListVirtualModelsQueryParams,
    UpdateVirtualModelRequest,
    VirtualModel,
)

_VIRTUAL_MODELS = "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models"


@get(_VIRTUAL_MODELS + "/{name}")
@abstractmethod
def get_virtual_model(*, workspace: str | None = None, name: str) -> VirtualModel: ...


def _get_virtual_model_on_conflict(
    body: CreateVirtualModelRequest, workspace: str | None
) -> PreparedRequest[VirtualModel]:
    """Retrieve request replayed when ``create_virtual_model(exist_ok=True)`` 409s."""
    return get_virtual_model(name=body.name, workspace=workspace)


@post(_VIRTUAL_MODELS, get_on_conflict=_get_virtual_model_on_conflict)
@abstractmethod
def create_virtual_model(
    *, workspace: str | None = None, body: CreateVirtualModelRequest, exist_ok: bool = False
) -> VirtualModel: ...


@get(_VIRTUAL_MODELS)
@abstractmethod
def list_virtual_models(
    *, workspace: str | None = None, query_params: ListVirtualModelsQueryParams | None = None
) -> Paginated[VirtualModel]: ...


@patch(_VIRTUAL_MODELS + "/{name}")
@abstractmethod
def update_virtual_model(
    *, workspace: str | None = None, name: str, body: UpdateVirtualModelRequest
) -> VirtualModel: ...


@delete(_VIRTUAL_MODELS + "/{name}")
@abstractmethod
def delete_virtual_model(
    *, workspace: str | None = None, name: str, query_params: DeleteVirtualModelQueryParams | None = None
) -> None: ...
