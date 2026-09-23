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
models/providers from in-memory caches), so this module adds the missing check:
when the caller is a *delegated* service principal, verify the on-behalf-of user
holds the endpoint's required permission in the target workspace. Non-delegated
callers are untouched — plain users are already gated by the route gate, and a
non-delegated service principal keeps the existing bypass.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, status
from nmp.common.auth.client import AuthClient
from nmp.common.auth.dependencies import auth_client_context

logger = logging.getLogger(__name__)

# Permission gating the inference proxy endpoints in a workspace. These mirror the
# central endpoint definitions in the auth service's static-authz.yaml for the
# inference-gateway ``.../{openai,model,provider}/...`` proxy routes.
OPENAI_EXEC_PERMISSION = "inference.gateway.openai.exec"
MODEL_EXEC_PERMISSION = "inference.gateway.model.exec"
PROVIDER_EXEC_PERMISSION = "inference.gateway.provider.exec"
# The provider readiness probe is gated by the provider read permission, not exec.
PROVIDER_READ_PERMISSION = "inference.providers.read"
# Permission required in a model entity's own workspace to resolve it there, mirroring
# GET /apis/models/v2/workspaces/{workspace}/models/{name} in the models service.
MODEL_READ_PERMISSION = "models.read"


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
    auth_client = auth_client_context.get()
    # No auth context (auth disabled / not configured) or auth disabled: nothing
    # to scope — the route gate already made the allow/deny decision.
    if auth_client is None or not auth_client.auth_enabled:
        return

    principal = auth_client.principal
    # Only delegated service principals need narrowing. A plain user was already
    # gated by the route gate as themselves; a non-delegated service principal
    # keeps its existing (intended) internal bypass.
    if not principal.is_privileged or not principal.is_delegated:
        return

    allowed = await _on_behalf_of_has_permission(auth_client, workspace, permission)
    if not allowed:
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


async def enforce_resolved_model_workspace_access(request_workspace: str, resolved_workspace: str) -> None:
    """Deny a proxy request whose resolved model entity lives outside the request's workspace.

    A VirtualModel's ``default_model_entity``/``models``, or a request-middleware rewrite (e.g.
    switchyard routing), can name a model entity in a workspace other than the one in the request
    path. The route gate only authorizes *request_workspace*, so without this check any caller who
    can create a VirtualModel in their own workspace could reach a served model in any other
    workspace by name. No-op when the resolved entity is in the request's own workspace.

    Raises:
        HTTPException: 403 when the caller lacks ``models.read`` in *resolved_workspace*.
    """
    if resolved_workspace == request_workspace:
        return

    auth_client = auth_client_context.get()
    if auth_client is None or not auth_client.auth_enabled:
        return

    if not await auth_client.has_permissions(resolved_workspace, [MODEL_READ_PERMISSION]):
        logger.info(
            "Denying cross-workspace model resolution: principal=%s request_workspace=%s resolved_workspace=%s",
            auth_client.principal.id,
            request_workspace,
            resolved_workspace,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized to access models in workspace '{resolved_workspace}'.",
        )


async def _on_behalf_of_has_permission(auth_client: AuthClient, workspace: str, permission: str) -> bool:
    """Return whether the on-behalf-of user holds *permission* in *workspace*.

    Uses :meth:`AuthClient.on_behalf_of_has_permissions`, which evaluates the PDP
    as the delegated user (not the service principal), so the ServiceSystem
    wildcard does not apply.
    """
    return await auth_client.on_behalf_of_has_permissions(workspace, [permission])
