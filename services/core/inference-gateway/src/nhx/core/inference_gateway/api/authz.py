# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve inference authorization against declared PDP operations in every target workspace."""

from __future__ import annotations

import logging

from fastapi import HTTPException, status
from nemo_helix_plugin.auth.decisions import DecisionRequest, ResourceTarget
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.config import get_auth_config
from nhx.common.entities.utils import ParsedEntityRef

logger = logging.getLogger(__name__)

_LORA_ADAPTER_SEPARATOR = "&adapters/"


def _enabled_auth_client() -> AuthClient | None:
    auth_client = auth_client_context.get()
    if auth_client is None:
        if get_auth_config().enabled:
            raise HTTPException(status_code=503, detail="Missing authenticated inference context")
        return None
    if not auth_client.auth_enabled:
        return None
    return auth_client


async def can_run_inference_in(request_workspace: str, workspace: str) -> bool:
    auth_client = _enabled_auth_client()
    if auth_client is None:
        return True
    return await auth_client.has_operation(
        DecisionRequest(
            operation="inference-gateway.model.exec",
            target=ResourceTarget(workspace=workspace, resource_type="inference_target", name=""),
        )
    )


def model_ref_workspaces(model_ref: ParsedEntityRef) -> list[str]:
    workspaces = [model_ref.workspace]
    _, separator, adapter_ref = model_ref.name.partition(_LORA_ADAPTER_SEPARATOR)
    adapter_workspace, adapter_separator, _ = adapter_ref.partition("/")
    if separator and adapter_separator and adapter_workspace not in workspaces:
        workspaces.append(adapter_workspace)
    return workspaces


async def enforce_model_ref_access(request_workspace: str, model_ref: ParsedEntityRef) -> None:
    for workspace in model_ref_workspaces(model_ref):
        if await can_run_inference_in(request_workspace, workspace):
            continue
        logger.warning(
            "Denying cross-workspace model access: request_workspace=%s model=%s/%s workspace=%s",
            request_workspace,
            model_ref.workspace,
            model_ref.name,
            workspace,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized to run inference on models in workspace '{workspace}'.",
        )
