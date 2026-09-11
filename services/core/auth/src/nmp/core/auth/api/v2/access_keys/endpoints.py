# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse
from nemo_platform_plugin.auth.access_keys.issuer import (
    AccessKeyFeatureDisabledError,
    AccessKeyOperationNotImplementedError,
)
from nemo_platform_plugin.auth.access_keys.types import AccessKeyReversibleStatus
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient
from nmp.common.auth import AuthClient, get_auth_client
from nmp.common.auth.access_keys import (
    ACCESS_KEY_JTI_PATTERN,
    AccessKeyValidationError,
)
from nmp.common.config import get_auth_config
from nmp.common.entities import EntityConflictError
from nmp.common.service.dependencies import get_nemo_client
from nmp.core.auth.app.access_keys import (
    AccessKeyNotFoundError,
    AccessKeyRegistry,
    AccessKeyStateConflictError,
    PersistentAccessKeyIssuer,
    get_access_key_registry,
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
        "description": "Service-bound Scoped Access Keys require PlatformAdmin",
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


async def _is_platform_admin(auth_client: AuthClient) -> bool:
    # Only human PlatformAdmins may create or manage service-bound Scoped Access Keys — the
    # same invariant AccessKeyIssuerService._target_principal enforces for creation. A
    # service-account principal must never qualify here, even if it were ever (mis)granted
    # the PlatformAdmin role, since that would let a service credential manage other
    # service-bound credentials, including its own.
    if auth_client.principal.effective_principal.is_service_identity():
        return False
    # Deny (rather than has_role's own default-allow) when auth is globally disabled: there is
    # no real identity to check "is PlatformAdmin" against, so we don't silently grant this
    # highly privileged, service-account-impersonating capability.
    return auth_client.auth_enabled and await auth_client.has_role("system", "PlatformAdmin")


def get_workspaces_client(
    nemo_client: AsyncNemoClient = Depends(get_nemo_client),
) -> AsyncWorkspacesClient:
    return AsyncWorkspacesClient.from_client(nemo_client)


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
        admin_override=lambda: _is_platform_admin(auth_client),
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
            # _is_platform_admin(auth_client) directly here) so this pre-check and
            # create_async's defense-in-depth re-check share one PDP has_role round trip.
            if not await issuer.is_platform_admin():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only PlatformAdmin can create service-bound Scoped Access Keys",
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
