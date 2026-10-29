# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router key route under /apis/evals/v2/workspaces/{workspace}/router-keys."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from nemo_evals.api.dependencies import delegated_service_client
from nemo_evals.authz import scope
from nemo_evals.router_keys import RouterKey, RouterKeyCreate, RouterKeyModelError, create_router_key
from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.authz import CallerKind, PermissionSet, path_rule, perm
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.dependencies import get_nemo_client


class RouterKeyPerms(PermissionSet, namespace="evaluator.router-keys"):
    """Permissions for minting router keys."""

    CREATE = perm("Mint a router key for closed-book models")


router = APIRouter()


@router.post(
    "/router-keys",
    summary="Create Router Key",
    response_description="The secret the router key was stored in",
    status_code=status.HTTP_201_CREATED,
    response_model=RouterKey,
)
@scope.write
@path_rule(callers=[CallerKind.PRINCIPAL], permissions=[RouterKeyPerms.CREATE])
async def create_router_key_route(
    workspace: str,
    body: RouterKeyCreate,
    caller_client: AsyncNemoClient = Depends(get_nemo_client),
) -> RouterKey:
    """Mint a router key limited to closed-book VirtualModels and store it in a platform secret.

    Give the secret to a router's ModelProvider as its `api_key_secret_name`. The router then calls models back through
    the gateway with the key, which reaches only the listed models, and closed-book evaluations may use the router.
    """
    try:
        async with delegated_service_client(caller_client) as service_client:
            return await create_router_key(
                workspace=workspace,
                request=body,
                caller_client=caller_client,
                service_client=service_client,
                auth_enabled=platform_auth_enabled(),
            )
    except RouterKeyModelError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except NemoHTTPError as exc:
        raise HTTPException(exc.status_code, detail=exc.detail) from exc
