# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse
from nemo_helix_plugin.auth.access_keys.issuer import (
    AccessKeyFeatureDisabledError,
    AccessKeyOperationNotImplementedError,
)
from nemo_helix_plugin.auth.access_keys.types import AccessKeyReversibleStatus
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.auth.access_keys import (
    ACCESS_KEY_JTI_PATTERN,
    AccessKeyValidationError,
)
from nhx.common.config import get_auth_config
from nhx.common.entities import EntityConflictError
from nhx.common.service.dependencies import get_nemo_client
from nhx.core.auth.app.access_keys import (
    AccessKeyNotFoundError,
    AccessKeyRegistry,
    AccessKeyStateConflictError,
    AdminOverride,
    PersistentAccessKeyIssuer,
    get_access_key_registry,
    resolve_workspace_id,
)

from . import schemas

router = APIRouter(tags=["Scoped Access Keys"])

_ACCESS_KEY_DISABLED_CODE = "access_keys_disabled"
_ACCESS_KEY_DISABLED_DETAIL = "Scoped Access Keys are not enabled"
_AccessKeyJTI = Annotated[
    str,
    Path(
        pattern=ACCESS_KEY_JTI_PATTERN,
        description="Stable JWT ID of the Scoped Access Key for the lifecycle operation.",
    ),
]

_ACCESS_KEY_DISABLED_ERROR_RESPONSE: dict[str, Any] = {
    "description": "Scoped Access Keys are not enabled",
    "model": schemas.AccessKeyErrorResponse,
}
_ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE: dict[str, Any] = {
    "description": "Scoped Access Keys are not enabled or the key was not found",
    "model": schemas.AccessKeyErrorResponse,
}
_ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE: dict[str, Any] = {
    "description": "Not Implemented",
    "model": schemas.AccessKeyNotImplementedErrorResponse,
}
_ACCESS_KEY_CONFLICT_ERROR_RESPONSE: dict[str, Any] = {
    "description": "Concurrent access-key update conflict",
    "model": schemas.AccessKeyErrorResponse,
}
_ACCESS_KEY_STATE_CONFLICT_ERROR_RESPONSE: dict[str, Any] = {
    "description": "Invalid or concurrent access-key state transition",
    "model": schemas.AccessKeyErrorResponse,
}
_ACCESS_KEY_CREATE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "Scoped Access Key creation error",
        "model": schemas.AccessKeyErrorResponse,
    },
    403: {
        "description": "Service-bound Scoped Access Keys require HelixAdmin or Admin of the bound workspace",
        "model": schemas.AccessKeyErrorResponse,
    },
    404: _ACCESS_KEY_DISABLED_ERROR_RESPONSE,
    409: _ACCESS_KEY_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_LIFECYCLE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: _ACCESS_KEY_DISABLED_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_REVOKE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: _ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE,
    409: _ACCESS_KEY_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_SUSPENSION_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: _ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE,
    409: _ACCESS_KEY_STATE_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_ROTATE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "Scoped Access Key rotation error",
        "model": schemas.AccessKeyErrorResponse,
    },
    404: _ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE,
    409: _ACCESS_KEY_STATE_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}


def _service_key_admin_check(auth_client: AuthClient, workspaces_client: AsyncWorkspacesClient) -> AdminOverride:
    """Build the per-request check for managing service-bound Scoped Access Keys.

    Only humans may create or manage service-bound keys, matching
    AccessKeyIssuerService._target_principal. A service account never qualifies, even if
    granted HelixAdmin or workspace Admin, so it can't manage service credentials. A
    HelixAdmin qualifies for any key; a workspace Admin only for keys bound to their own
    workspace. The HelixAdmin role is resolved once per request, however many workspaces
    are checked.
    """
    is_helix_admin: bool | None = None

    async def can_administer(workspace: str | None, workspace_id: str | None = None) -> bool:
        """Whether the caller is a human HelixAdmin, or Admin of the (unchanged) bound workspace."""
        nonlocal is_helix_admin
        if auth_client.principal.effective_principal.is_service_identity():
            return False
        # Deny (rather than has_role's default-allow) when auth is disabled: there is no real
        # identity to check.
        if not auth_client.auth_enabled:
            return False
        if is_helix_admin is None:
            is_helix_admin = await auth_client.has_role("system", "HelixAdmin")
        if is_helix_admin:
            return True
        if workspace is None:
            return False
        if workspace_id is None:
            return await auth_client.has_role(workspace, "Admin")

        # The key was bound to a specific workspace; a deleted and recreated workspace with the
        # same name must not inherit it. has_role resolves the workspace by name, so read the ID
        # before and after it: IDs are unique, so a match on both reads means the same workspace
        # held the name throughout the role check. A workspace that is gone (or unreadable by the
        # caller) fails closed; a transient lookup failure is a 503, like a failed has_role.
        if await resolve_workspace_id(workspaces_client, workspace) != workspace_id:
            return False
        if not await auth_client.has_role(workspace, "Admin"):
            return False
        return await resolve_workspace_id(workspaces_client, workspace) == workspace_id

    return can_administer


def get_workspaces_client(
    nemo_client: AsyncNemoClient = Depends(get_nemo_client),
) -> AsyncWorkspacesClient:
    return AsyncWorkspacesClient.from_client(nemo_client)


def _caller_access_key_scope(auth_client: AuthClient) -> list[str] | None:
    """The caller's own Scoped Access Key scope restriction, if any."""
    resolved = auth_client.resolved_bearer_token
    if resolved is None or resolved.token_kind != "access_key":
        return None
    services = sorted({scope.split(":", 1)[0] for scope in resolved.scopes if ":" in scope})
    return services or None


def get_access_key_issuer(
    auth_client: AuthClient = Depends(get_auth_client),
    registry: AccessKeyRegistry = Depends(get_access_key_registry),
    workspaces_client: AsyncWorkspacesClient = Depends(get_workspaces_client),
) -> PersistentAccessKeyIssuer:
    return PersistentAccessKeyIssuer(
        get_auth_config(),
        auth_client.principal.effective_principal,
        registry,
        workspaces_client,
        admin_override=_service_key_admin_check(auth_client, workspaces_client),
        caller_scope=_caller_access_key_scope(auth_client),
    )


def _not_implemented(exc: AccessKeyOperationNotImplementedError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))


def _disabled_response() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": _ACCESS_KEY_DISABLED_DETAIL, "code": _ACCESS_KEY_DISABLED_CODE},
    )


async def _change_suspension_status(
    jti: str,
    transition: Callable[[str], Awaitable[tuple[bool, AccessKeyReversibleStatus]]],
) -> schemas.AccessKeyStatusChangeResponse | JSONResponse:
    try:
        changed, effective_status = await transition(jti)
    except AccessKeyFeatureDisabledError:
        return _disabled_response()
    except AccessKeyOperationNotImplementedError as exc:
        raise _not_implemented(exc) from exc
    except AccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AccessKeyStateConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Concurrent update conflict; retry.") from exc
    return schemas.AccessKeyStatusChangeResponse(jti=jti, status=effective_status, changed=changed)


@router.post(
    "/v2/access-keys",
    response_model=schemas.AccessKeyCreateResponse,
    responses=_ACCESS_KEY_CREATE_ERROR_RESPONSES,
)
async def create_access_key(
    request: schemas.AccessKeyCreateRequest,
    issuer: PersistentAccessKeyIssuer = Depends(get_access_key_issuer),
) -> schemas.AccessKeyCreateResponse | JSONResponse:
    try:
        if request.service_account_id is not None:
            if not get_auth_config().access_keys.enabled:
                raise AccessKeyFeatureDisabledError("Scoped Access Keys are not enabled")
            # Delegates to the issuer's own memoized admin check (rather than calling
            # _service_key_admin_check(...) directly here) so this pre-check and
            # create_async's defense-in-depth re-check share one PDP has_role round trip.
            if not await issuer.can_administer(request.workspace):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Service-bound Scoped Access Keys require HelixAdmin or Admin of the bound workspace",
                )
            return await issuer.create_async(request, allow_service_account=True)
        return await issuer.create_async(request)
    except AccessKeyFeatureDisabledError:
        return _disabled_response()
    except AccessKeyOperationNotImplementedError as exc:
        raise _not_implemented(exc) from exc
    except AccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Concurrent conflict; retry.") from exc


@router.get(
    "/v2/access-keys",
    response_model=schemas.AccessKeyListResponse,
    responses=_ACCESS_KEY_LIFECYCLE_ERROR_RESPONSES,
)
async def list_access_keys(
    page: int = Query(default=1, ge=1, description="Page number to retrieve."),
    page_size: int = Query(default=100, ge=1, le=100, description="Number of keys to retrieve per page."),
    issuer: PersistentAccessKeyIssuer = Depends(get_access_key_issuer),
) -> schemas.AccessKeyListResponse | JSONResponse:
    try:
        return await issuer.list_async(page=page, page_size=page_size)
    except AccessKeyFeatureDisabledError:
        return _disabled_response()
    except AccessKeyOperationNotImplementedError as exc:
        raise _not_implemented(exc) from exc


@router.delete(
    "/v2/access-keys/{jti}",
    response_model=schemas.AccessKeyRevokeResponse,
    responses=_ACCESS_KEY_REVOKE_ERROR_RESPONSES,
)
async def revoke_access_key(
    jti: _AccessKeyJTI,
    issuer: PersistentAccessKeyIssuer = Depends(get_access_key_issuer),
) -> schemas.AccessKeyRevokeResponse | JSONResponse:
    try:
        revoked = await issuer.revoke_async(jti)
    except AccessKeyFeatureDisabledError:
        return _disabled_response()
    except AccessKeyOperationNotImplementedError as exc:
        raise _not_implemented(exc) from exc
    except AccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Concurrent update conflict; retry.") from exc
    return schemas.AccessKeyRevokeResponse(jti=jti, revoked=revoked)


@router.post(
    "/v2/access-keys/{jti}/suspend",
    response_model=schemas.AccessKeyStatusChangeResponse,
    responses=_ACCESS_KEY_SUSPENSION_ERROR_RESPONSES,
)
async def suspend_access_key(
    jti: _AccessKeyJTI,
    issuer: PersistentAccessKeyIssuer = Depends(get_access_key_issuer),
) -> schemas.AccessKeyStatusChangeResponse | JSONResponse:
    return await _change_suspension_status(jti, issuer.suspend_async)


@router.post(
    "/v2/access-keys/{jti}/unsuspend",
    response_model=schemas.AccessKeyStatusChangeResponse,
    responses=_ACCESS_KEY_SUSPENSION_ERROR_RESPONSES,
)
async def unsuspend_access_key(
    jti: _AccessKeyJTI,
    issuer: PersistentAccessKeyIssuer = Depends(get_access_key_issuer),
) -> schemas.AccessKeyStatusChangeResponse | JSONResponse:
    return await _change_suspension_status(jti, issuer.unsuspend_async)


@router.post(
    "/v2/access-keys/{jti}/rotate",
    response_model=schemas.AccessKeyRotateResponse,
    responses=_ACCESS_KEY_ROTATE_ERROR_RESPONSES,
)
async def rotate_access_key(
    jti: _AccessKeyJTI,
    request: schemas.AccessKeyRotateRequest = schemas.AccessKeyRotateRequest(),
    issuer: PersistentAccessKeyIssuer = Depends(get_access_key_issuer),
) -> schemas.AccessKeyRotateResponse | JSONResponse:
    try:
        return await issuer.rotate_async(jti, grace_period_seconds=request.grace_period_seconds)
    except AccessKeyFeatureDisabledError:
        return _disabled_response()
    except AccessKeyOperationNotImplementedError as exc:
        raise _not_implemented(exc) from exc
    except AccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AccessKeyStateConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Concurrent update conflict; retry.") from exc
