# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
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
    AccessKeyForbiddenError,
    AccessKeyNotFoundError,
    AccessKeyRegistry,
    AccessKeyStateConflictError,
    AdminOverride,
    PersistentAccessKeyIssuer,
    get_access_key_registry,
    resolve_workspace_id,
    resolve_workspace_name,
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
        "description": (
            "Service-bound Scoped Access Keys require HelixAdmin, or Admin of the single workspace named in "
            "`workspaces`, which the key is then bound to and whose Admins manage it. A workspace Admin can "
            "only use a service account whose live keys and role bindings are all within that workspace"
        ),
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
_ACCESS_KEY_UNSUSPENSION_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "The service-bound key's service account is disabled",
        "model": schemas.AccessKeyErrorResponse,
    },
    403: {
        "description": (
            "A workspace Admin can't unsuspend a service-bound key whose service account has access outside "
            "the key's bound workspace; only a HelixAdmin can"
        ),
        "model": schemas.AccessKeyErrorResponse,
    },
    404: _ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE,
    409: _ACCESS_KEY_STATE_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_ROTATE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "Scoped Access Key rotation error",
        "model": schemas.AccessKeyErrorResponse,
    },
    403: {
        "description": (
            "A workspace Admin can't rotate a service-bound key whose service account has access outside "
            "the key's bound workspace; only a HelixAdmin can"
        ),
        "model": schemas.AccessKeyErrorResponse,
    },
    404: _ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE,
    409: _ACCESS_KEY_STATE_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}


def _service_key_admin_check(
    auth_client: AuthClient,
    workspaces_client: AsyncWorkspacesClient,
    workspace_names: dict[str, str | None],
) -> AdminOverride:
    """Build the per-request check for managing service-bound Scoped Access Keys.

    Only humans may create or manage service-bound keys, matching
    AccessKeyIssuerService._target_principal. A service account never qualifies, even if granted
    HelixAdmin or workspace Admin, so it can't manage service credentials, including its own. A
    HelixAdmin qualifies for any key; anyone else only for a key bound to a workspace they are
    Admin of. The HelixAdmin check is resolved once per request, and so is each workspace's Admin
    check: listing many keys bound to one workspace costs one lookup, not one per key.

    ``workspace_names`` (workspace ID -> current name, None if gone or invisible) is a per-request
    cache shared with the issuer: a name it already read, e.g. while listing the caller's
    workspaces, isn't looked up again here. The name is still confirmed to map to the ID after
    the role check.
    """
    # The outcome of the HelixAdmin check (keyed None) and of each workspace's check, once known:
    # each runs once per request, however many keys need it, and a failed one raises its failure
    # again for the rest of the request. Checks running at once (as when listing) wait on the
    # first instead of repeating it; one that's cancelled leaves no outcome, so the next runs it.
    outcomes: dict[str | None, bool | Exception] = {}
    running: dict[str | None, asyncio.Lock] = {}

    async def once(key: str | None, check: Callable[[], Awaitable[bool]]) -> bool:
        lock = running.get(key)
        if lock is None:
            lock = running[key] = asyncio.Lock()
        async with lock:
            if key not in outcomes:
                try:
                    outcomes[key] = await check()
                except Exception as exc:
                    outcomes[key] = exc
        outcome = outcomes[key]
        if isinstance(outcome, HTTPException):
            # A fresh one for each caller, so concurrent ones don't share (and rewrite) its
            # traceback and context; the first is its cause.
            raise HTTPException(outcome.status_code, outcome.detail, outcome.headers) from outcome
        if isinstance(outcome, Exception):
            # Anything else is a misconfiguration (such as a missing PDP URL), failing the request.
            raise outcome
        return outcome

    async def check_admin_of(workspace_id: str) -> bool:
        # Keys are bound to workspace IDs, but has_role takes a name. Resolve the ID's current
        # name, check the role, then confirm the name still maps to the ID: IDs are unique, so a
        # match means the same workspace held the name throughout the role check, and a workspace
        # deleted and recreated under the same name never qualifies. A workspace that is gone (or
        # unreadable by the caller) fails closed; a transient lookup failure is a 503, like a
        # failed has_role.
        if workspace_id not in workspace_names:
            workspace_names[workspace_id] = await resolve_workspace_name(workspaces_client, workspace_id)
        name = workspace_names[workspace_id]
        if name is None:
            return False
        if not await auth_client.has_role(name, "Admin"):
            return False
        return await resolve_workspace_id(workspaces_client, name) == workspace_id

    async def can_administer(workspace_id: str | None) -> bool:
        """Whether the caller is a human HelixAdmin, or Admin of the key's bound workspace."""
        if auth_client.principal.effective_principal.is_service_identity():
            return False
        # Deny (rather than has_role's default-allow) when auth is disabled: there is no real
        # identity to check, so we don't silently grant this service-account-impersonating
        # capability.
        if not auth_client.auth_enabled:
            return False
        if await once(None, lambda: auth_client.has_role("system", "HelixAdmin")):
            return True
        if workspace_id is None:
            return False
        return await once(workspace_id, lambda: check_admin_of(workspace_id))

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
    # One per request, shared by the Admin check and the issuer (see _service_key_admin_check).
    workspace_names: dict[str, str | None] = {}
    # A service identity, or anyone while auth is disabled, never manages service-bound keys (the
    # check would always say no), so it gets none: listing then skips looking for Admin workspaces.
    may_manage_service_keys = (
        auth_client.auth_enabled and not auth_client.principal.effective_principal.is_service_identity()
    )
    return PersistentAccessKeyIssuer(
        get_auth_config(),
        auth_client.principal.effective_principal,
        registry,
        workspaces_client,
        admin_override=(
            _service_key_admin_check(auth_client, workspaces_client, workspace_names)
            if may_manage_service_keys
            else None
        ),
        caller_scope=_caller_access_key_scope(auth_client),
        workspace_names=workspace_names,
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
    except AccessKeyForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except AccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
            # Authorize before a key is minted. Delegates to the issuer's memoized admin check
            # and cached workspace lookups (rather than building the check here) so this and
            # create_async's defense-in-depth re-check share the PDP has_role and workspace round
            # trips, and report the same reason for a refusal.
            await issuer.resolve_bound_workspace([grant.workspace for grant in request.workspaces or []])
            return await issuer.create_async(request, allow_service_account=True)
        return await issuer.create_async(request)
    except AccessKeyFeatureDisabledError:
        return _disabled_response()
    except AccessKeyOperationNotImplementedError as exc:
        raise _not_implemented(exc) from exc
    except AccessKeyForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
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
    responses=_ACCESS_KEY_UNSUSPENSION_ERROR_RESPONSES,
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
    except AccessKeyForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except AccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AccessKeyNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AccessKeyStateConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Concurrent update conflict; retry.") from exc
