# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse
from nemo_helix_plugin.auth.access_keys.issuer import (
    WORKSPACE_BOUND_KEYS_HINT,
    AccessKeyFeatureDisabledError,
    AccessKeyOperationNotImplementedError,
)
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
    ADMIN_REQUIRED_MESSAGE,
    AccessKeyForbiddenError,
    AccessKeyNotAdministeredError,
    AccessKeyNotFoundError,
    AccessKeyRegistry,
    AccessKeyStateConflictError,
    AccessKeyStillActiveError,
    AdminOverride,
    PersistentAccessKeyIssuer,
    WorkspaceBoundKeysWidenedError,
    gather_or_cancel,
    get_access_key_registry,
    resolve_workspace_id,
    resolve_workspace_name,
)

from . import schemas

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Scoped Access Keys"])

_ACCESS_KEY_DISABLED_CODE = "access_keys_disabled"
_ACCESS_KEY_DISABLED_DETAIL = "Scoped Access Keys are not enabled"
_ACCESS_KEY_WORKSPACE_BOUND_KEYS_CODE = "workspace_bound_keys_exist"
_ACCESS_KEY_NOT_ADMINISTERED_CODE = "bound_workspaces_not_administered"
# Most workspaces whose Admin checks (a PDP call and workspace lookups each) run at once per request.
_MAX_CONCURRENT_WORKSPACE_ADMIN_CHECKS = 8
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
_NOT_ADMINISTERED_DESCRIPTION = (
    "(code bound_workspaces_not_administered) the caller is Admin of some of the service-bound key's "
    "bound workspaces but not all of them, which the detail names"
)
_ACCESS_KEY_NOT_ADMINISTERED_ERROR_RESPONSE: dict[str, Any] = {
    "description": f"The key wasn't changed: {_NOT_ADMINISTERED_DESCRIPTION}",
    "model": schemas.AccessKeyErrorResponse,
}
_ACCESS_KEY_CREATE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "Scoped Access Key creation error",
        "model": schemas.AccessKeyErrorResponse,
    },
    403: {
        "description": (
            "Service-bound Scoped Access Keys require HelixAdmin, or Admin of every bound workspace for a "
            "service account named `<workspace>--<name>` after one of them, with no access outside them"
        ),
        "model": schemas.AccessKeyErrorResponse,
    },
    404: _ACCESS_KEY_DISABLED_ERROR_RESPONSE,
    409: {
        "description": (
            "Concurrent access-key update conflict, a bound workspace changed while the key was being "
            "created (retry), or (code workspace_bound_keys_exist) the service account has workspace-bound "
            "keys and the request needs acknowledge_workspace_bound_keys"
        ),
        "model": schemas.AccessKeyErrorResponse,
    },
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_LIFECYCLE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: _ACCESS_KEY_DISABLED_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_REVOKE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: _ACCESS_KEY_NOT_ADMINISTERED_ERROR_RESPONSE,
    404: _ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE,
    409: _ACCESS_KEY_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_SUSPENSION_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: _ACCESS_KEY_NOT_ADMINISTERED_ERROR_RESPONSE,
    404: _ACCESS_KEY_DISABLED_OR_NOT_FOUND_ERROR_RESPONSE,
    409: _ACCESS_KEY_STATE_CONFLICT_ERROR_RESPONSE,
    501: _ACCESS_KEY_NOT_IMPLEMENTED_ERROR_RESPONSE,
}
_ACCESS_KEY_UNSUSPENSION_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    **_ACCESS_KEY_SUSPENSION_ERROR_RESPONSES,
    400: {
        "description": "The service-bound key's service account is disabled",
        "model": schemas.AccessKeyErrorResponse,
    },
    403: {
        "description": (
            "A workspace Admin can't unsuspend a service-bound key whose service account has access outside "
            f"the key's bound workspaces; only a HelixAdmin can. Or {_NOT_ADMINISTERED_DESCRIPTION}"
        ),
        "model": schemas.AccessKeyErrorResponse,
    },
}
_ACCESS_KEY_ROTATE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "Scoped Access Key rotation error",
        "model": schemas.AccessKeyErrorResponse,
    },
    403: {
        "description": (
            "A workspace Admin can't rotate a service-bound key whose service account has access outside "
            f"the key's bound workspaces; only a HelixAdmin can. Or {_NOT_ADMINISTERED_DESCRIPTION}"
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
    workspace_names: dict[str, str | None] | None = None,
) -> AdminOverride:
    """Build the per-request check for managing service-bound Scoped Access Keys.

    Only humans may create or manage service-bound keys, matching
    AccessKeyIssuerService._target_principal. A service account never qualifies, even if
    granted HelixAdmin or workspace Admin, so it can't manage service credentials. A
    HelixAdmin qualifies for any key; a workspace Admin only for keys bound to workspaces
    they are all Admin of. The HelixAdmin role is resolved once per request, however many
    workspaces are checked, and so is each workspace's Admin check: keys bound to overlapping
    sets of workspaces (as when listing) share the lookups for the workspaces they have in common.

    ``workspace_names`` (workspace ID -> current name, None if gone or invisible) is a per-request
    cache shared with the issuer: a name it already read, e.g. while listing the caller's
    workspaces, isn't looked up again here. The name is still confirmed to map to the ID after
    the role check.
    """
    names: dict[str, str | None] = workspace_names if workspace_names is not None else {}
    # The outcome of the HelixAdmin check (keyed None) and of each workspace's check, once known:
    # each runs once per request, however many keys need it, and a failed one raises its failure
    # again for the rest of the request. Checks running at once (as when listing) wait on the
    # first instead of repeating it; one that's cancelled leaves no outcome, so the next runs it.
    outcomes: dict[str | None, bool | Exception] = {}
    running: dict[str | None, asyncio.Lock] = {}
    # Bounds the workspaces checked at once, however many a listed page or a request binds.
    concurrent_checks = asyncio.Semaphore(_MAX_CONCURRENT_WORKSPACE_ADMIN_CHECKS)

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
        async with concurrent_checks:
            return await check_admin_of_unbounded(workspace_id)

    async def check_admin_of_unbounded(workspace_id: str) -> bool:
        if workspace_id not in names:
            names[workspace_id] = await resolve_workspace_name(workspaces_client, workspace_id)
        name = names[workspace_id]
        if name is None:
            return False
        if not await auth_client.has_role(name, "Admin"):
            return False
        return await resolve_workspace_id(workspaces_client, name) == workspace_id

    async def can_administer(workspace_ids: Sequence[str]) -> bool:
        """Whether the caller is a human HelixAdmin, or Admin of every bound workspace."""
        if auth_client.principal.effective_principal.is_service_identity():
            return False
        # Deny (rather than has_role's default-allow) when auth is disabled: there is no real
        # identity to check.
        if not auth_client.auth_enabled:
            return False
        if await once(None, lambda: auth_client.has_role("system", "HelixAdmin")):
            return True
        if not workspace_ids:
            return False
        # Each workspace's check is a chain of dependent lookups, so check the workspaces
        # concurrently rather than paying for their chains one after another. The first failure
        # stops the rest and is raised as is, so a failed lookup still reaches callers (and
        # FastAPI) as the HTTPException it was.
        results = await gather_or_cancel(
            once(workspace_id, lambda workspace_id=workspace_id: check_admin_of(workspace_id))
            for workspace_id in dict.fromkeys(workspace_ids)
        )
        return all(results)

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
    caller_scope = _caller_access_key_scope(auth_client)
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
        caller_scope=caller_scope,
        workspace_names=workspace_names,
    )


def _not_implemented(exc: AccessKeyOperationNotImplementedError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=str(exc))


def _coded_error_response(
    status_code: int, detail: str, code: str, *, headers: dict[str, str] | None = None
) -> JSONResponse:
    """An error response with a ``code`` (see AccessKeyErrorResponse) that lets clients tell it apart."""
    return JSONResponse(status_code=status_code, content={"detail": detail, "code": code}, headers=headers)


def _disabled_response() -> JSONResponse:
    return _coded_error_response(status.HTTP_404_NOT_FOUND, _ACCESS_KEY_DISABLED_DETAIL, _ACCESS_KEY_DISABLED_CODE)


async def _not_administered_response(
    issuer: PersistentAccessKeyIssuer, exc: AccessKeyNotAdministeredError, *, action: str
) -> JSONResponse:
    # Only an Admin of at least one of the key's bound workspaces gets this; anyone else gets a 404.
    # The CLI shows it as a warning.
    logger.info(
        "Refused to %s a service-bound Scoped Access Key bound to workspaces the caller isn't Admin of",
        action,
        extra={"actor_principal": issuer.principal, "access_key_jti": exc.jti, "workspace_ids": exc.workspace_ids},
    )
    return _coded_error_response(
        status.HTTP_403_FORBIDDEN,
        await issuer.describe_not_administered(exc, action=action),
        _ACCESS_KEY_NOT_ADMINISTERED_CODE,
    )


async def _change_suspension_status(
    jti: str,
    issuer: PersistentAccessKeyIssuer,
    action: Literal["suspend", "unsuspend"],
) -> schemas.AccessKeyStatusChangeResponse | JSONResponse:
    transition = issuer.suspend_async if action == "suspend" else issuer.unsuspend_async
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
    except AccessKeyNotAdministeredError as exc:
        return await _not_administered_response(issuer, exc, action=action)
    except AccessKeyForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except AccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AccessKeyStillActiveError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=exc.detail) from exc
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
            # Delegates to the issuer's memoized admin check and cached workspace lookups (rather
            # than calling _service_key_admin_check(...) directly here) so this pre-check and
            # create_async's defense-in-depth re-check share the PDP has_role and workspace
            # round trips.
            if not await issuer.can_create_bound_key([grant.workspace for grant in request.workspaces or []]):
                raise AccessKeyForbiddenError(ADMIN_REQUIRED_MESSAGE)
            return await issuer.create_async(request, allow_service_account=True)
        return await issuer.create_async(request)
    except AccessKeyFeatureDisabledError:
        return _disabled_response()
    except AccessKeyOperationNotImplementedError as exc:
        raise _not_implemented(exc) from exc
    except WorkspaceBoundKeysWidenedError as exc:
        # The detail names the service account and the bound workspaces. Only a HelixAdmin, or an
        # Admin of every workspace the account's keys are bound to, gets this far (see
        # PersistentAccessKeyIssuer._widening_error).
        logger.info("Refused service-bound Scoped Access Key that widens workspace-bound keys: %s", exc.detail)
        return _coded_error_response(
            status.HTTP_409_CONFLICT,
            f"{exc.detail} {WORKSPACE_BOUND_KEYS_HINT}",
            _ACCESS_KEY_WORKSPACE_BOUND_KEYS_CODE,
            # Resending the same request gets the same answer, so clients that retry 409s (as the
            # platform clients do) should hand it to the caller to confirm instead.
            headers={"x-should-retry": "false"},
        )
    except AccessKeyForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except AccessKeyValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EntityConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Concurrent conflict; retry.") from exc
    except AccessKeyStateConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


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
    except AccessKeyNotAdministeredError as exc:
        return await _not_administered_response(issuer, exc, action="revoke")
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
    return await _change_suspension_status(jti, issuer, "suspend")


@router.post(
    "/v2/access-keys/{jti}/unsuspend",
    response_model=schemas.AccessKeyStatusChangeResponse,
    responses=_ACCESS_KEY_UNSUSPENSION_ERROR_RESPONSES,
)
async def unsuspend_access_key(
    jti: _AccessKeyJTI,
    issuer: PersistentAccessKeyIssuer = Depends(get_access_key_issuer),
) -> schemas.AccessKeyStatusChangeResponse | JSONResponse:
    return await _change_suspension_status(jti, issuer, "unsuspend")


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
    except AccessKeyNotAdministeredError as exc:
        return await _not_administered_response(issuer, exc, action="rotate")
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
