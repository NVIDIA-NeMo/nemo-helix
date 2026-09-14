# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed endpoint definitions for Inference Gateway VirtualModel CRUD."""

from __future__ import annotations

from abc import abstractmethod

from nemo_platform_plugin.client.endpoint import delete, get, patch, post
from nemo_platform_plugin.client.types import Paginated
from nemo_platform_plugin.virtual_models.types import (
    CreateVirtualModelRequest,
    DeleteVirtualModelQueryParams,
    ListVirtualModelsQueryParams,
    UpdateVirtualModelRequest,
    VirtualModel,
)

_VIRTUAL_MODELS = "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models"


@post(_VIRTUAL_MODELS)
@abstractmethod
def create_virtual_model(*, workspace: str | None = None, body: CreateVirtualModelRequest) -> VirtualModel: ...


@get(_VIRTUAL_MODELS)
@abstractmethod
def list_virtual_models(
    *, workspace: str | None = None, query_params: ListVirtualModelsQueryParams | None = None
) -> Paginated[VirtualModel]: ...


@get(_VIRTUAL_MODELS + "/{name}")
@abstractmethod
def get_virtual_model(*, workspace: str | None = None, name: str) -> VirtualModel: ...


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
