# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-time authorization for the inference proxy path.

The shared ``AuthMiddleware`` route gate authorizes every request against the
PDP, but for a **service principal** the PDP takes the ServiceSystem bypass
(wildcard permission) and does not narrow on the ``on-behalf-of`` identity. That
is correct for genuine internal callers that act only as themselves, but a
service principal that delegates (e.g. an agent deployment acting as its creator
via ``X-NMP-Principal-On-Behalf-Of``) must not inherit that platform-wide reach:
its access should be scoped to what the delegated user can reach.

The proxy handlers themselves do no per-caller access control (they resolve
models/providers from in-memory caches), so this module adds the missing checks:
a delegated service principal is scoped to what its on-behalf-of user can reach,
and any caller whose request resolves to a model or LoRA adapter in another
workspace must hold inference permission in that workspace too.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from fastapi import HTTPException, status
from nmp.common.auth.client import AuthClient
from nmp.common.auth.dependencies import auth_client_context
from nmp.common.auth.models import Principal
from nmp.common.entities.utils import ParsedEntityRef, parse_model_entity_ref

logger = logging.getLogger(__name__)

# Permission gating the inference proxy endpoints in a workspace. These mirror the
# central endpoint definitions in the auth service's static-authz.yaml for the
# inference-gateway ``.../{openai,model,provider}/...`` proxy routes.
OPENAI_EXEC_PERMISSION = "inference.gateway.openai.exec"
MODEL_EXEC_PERMISSION = "inference.gateway.model.exec"
PROVIDER_EXEC_PERMISSION = "inference.gateway.provider.exec"
# The provider readiness probe is gated by the provider read permission, not exec.
PROVIDER_READ_PERMISSION = "inference.providers.read"

_LORA_ADAPTER_SEPARATOR = "&adapters/"


def _enabled_auth_client() -> AuthClient | None:
    auth_client = auth_client_context.get()
    if auth_client is None or not auth_client.auth_enabled:
        return None
    return auth_client


def _is_delegated_service(principal: Principal) -> bool:
    return principal.is_privileged and principal.is_delegated


async def _caller_has_permission(auth_client: AuthClient, workspace: str, permission: str) -> bool:
    # A delegated service principal would pass any check as itself via the ServiceSystem wildcard.
    if _is_delegated_service(auth_client.principal):
        return await auth_client.on_behalf_of_has_permissions(workspace, [permission])
    return await auth_client.has_permissions(workspace, [permission])


async def enforce_delegated_workspace_access(workspace: str, permission: str) -> None:
    """Scope a delegated service-principal request to the on-behalf-of user.

    No-op unless the current principal is a *delegated* service principal
    (privileged id ``service:*`` with ``on_behalf_of`` set). In that case the
    request is allowed only if the on-behalf-of user holds *permission* in
    *workspace*; otherwise a 403 is raised.

    This is deliberately narrow: it never *grants* access the route gate denied,
    it only *removes* the service-principal bypass for delegated calls so a
    deployed workload cannot reach workspaces its creator cannot.

    Args:
        workspace: Target workspace from the request path.
        permission: Required permission (one of the ``inference.gateway.*.exec``
            constants in this module).

    Raises:
        HTTPException: 403 when the on-behalf-of user lacks *permission* in
            *workspace*.
    """
    auth_client = _enabled_auth_client()
    if auth_client is None:
        return

    principal = auth_client.principal
    # Only delegated service principals need narrowing. A plain user was already
    # gated by the route gate as themselves; a non-delegated service principal
    # keeps its existing (intended) internal bypass.
    if not _is_delegated_service(principal):
        return

    if not await _caller_has_permission(auth_client, workspace, permission):
        logger.info(
            "Denying delegated inference request: on-behalf-of=%s lacks %s in workspace=%s (service=%s)",
            principal.on_behalf_of,
            permission,
            workspace,
            principal.id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"On-behalf-of principal '{principal.on_behalf_of}' is not authorized "
                f"for inference in workspace '{workspace}'."
            ),
        )


async def can_run_inference_in(request_workspace: str, workspace: str) -> bool:
    if workspace == request_workspace:
        return True
    auth_client = _enabled_auth_client()
    if auth_client is None:
        return True
    return await _caller_has_permission(auth_client, workspace, MODEL_EXEC_PERMISSION)


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
        logger.info(
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


async def enforce_model_refs_access(request_workspace: str, model_refs: Iterable[str]) -> None:
    for model_ref in model_refs:
        try:
            parsed = parse_model_entity_ref(model_ref, default_workspace=request_workspace)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid model entity reference {model_ref!r}: {exc}",
            ) from exc
        await enforce_model_ref_access(request_workspace, parsed)
