# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The caller's profile and preferences (/me), preference defaults, and principal display resolution."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import asdict
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Response, status
from fastapi.responses import JSONResponse
from nemo_helix_plugin.auth.profiles.types import (
    PREFERENCE_NAMESPACE_PATTERN,
    Me,
    PreferenceDefaults,
    PreferenceDefaultsList,
    PreferenceDocument,
    PreferenceDocumentList,
    PreferenceValue,
    Profile,
    ProfileResolveRequest,
    ProfileResolveResponse,
    ProfileUpdate,
    ResolvedProfile,
)
from nhx.common.api.common import ErrorResponse
from nhx.common.auth import AuthClient, get_auth_client
from nhx.common.auth.openapi import BEARER_AUTH_OPENAPI
from nhx.common.config import get_service_config
from nhx.common.entities import EntityClient
from nhx.common.service.dependencies import get_entity_client
from nhx.core.auth.app.account_resolution import get_account_session_maker
from nhx.core.auth.app.directory import accounts_sharing_a_workspace
from nhx.core.auth.app.profiles import (
    LOCAL_DISPLAY_NAME,
    Caller,
    PreferenceDefaultsStore,
    require_editable,
    require_readable_workspace,
    resolve_caller,
)
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.entities.app.repository import (
    PREFERENCE_MAX_BYTES,
    PREFERENCE_MAX_DOCUMENTS,
    AccountPreferenceStore,
    AccountProfileStore,
    PreferenceLimitError,
    PreferenceRecord,
    PreferenceTooLargeError,
    PreferenceVersionConflictError,
    ResolvedPrincipal,
    merge_patch,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

router = APIRouter(tags=["Profiles"])
T = TypeVar("T")

_Namespace = Annotated[
    str,
    Path(pattern=PREFERENCE_NAMESPACE_PATTERN, description="Preference namespace, e.g. studio.ui or plugin.<name>."),
]
_Workspace = Annotated[str | None, Query(description="Workspace for workspace-scoped preferences; omit for global.")]
_IfMatch = Annotated[
    str | None,
    Header(description="Expected db_version. 0 requires that no document exists yet."),
]


class _CodedError(ErrorResponse):
    code: str


_ERRORS: dict[int | str, dict[str, Any]] = {
    403: {"model": ErrorResponse, "description": "Service account or delegated caller"},
    404: {"model": ErrorResponse, "description": "Workspace not found or not accessible"},
}
_WRITE_ERRORS: dict[int | str, dict[str, Any]] = {
    **_ERRORS,
    409: {"model": _CodedError, "description": "Document limit reached, or concurrent writes kept colliding"},
    412: {"model": _CodedError, "description": "If-Match did not match the stored db_version"},
    413: {"model": _CodedError, "description": "Document exceeds the size limit (code preference_too_large)"},
}
_DEFAULTS_WRITE_ERRORS: dict[int | str, dict[str, Any]] = {
    403: {"model": ErrorResponse, "description": "Caller is not a HelixAdmin"},
    409: {"model": _CodedError, "description": "Namespace defaults are managed by platform config"},
    412: _WRITE_ERRORS[412],
    413: _WRITE_ERRORS[413],
}


async def get_session_maker() -> async_sessionmaker[AsyncSession]:
    try:
        return await get_account_session_maker()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Account storage is not available") from exc


def get_config() -> AuthServiceConfig:
    return get_service_config(AuthServiceConfig)


SessionMaker = Annotated[async_sessionmaker[AsyncSession], Depends(get_session_maker)]
Config = Annotated[AuthServiceConfig, Depends(get_config)]
Auth = Annotated[AuthClient, Depends(get_auth_client)]


async def get_caller(auth_client: Auth, config: Config, session_maker: SessionMaker) -> Caller:
    return await resolve_caller(auth_client, config, session_maker)


def get_defaults_store(config: Config, session_maker: SessionMaker) -> PreferenceDefaultsStore:
    return PreferenceDefaultsStore(config, session_maker)


CallerDep = Annotated[Caller, Depends(get_caller)]
DefaultsStore = Annotated[PreferenceDefaultsStore, Depends(get_defaults_store)]


def _parse_if_match(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.removeprefix("W/").strip('"'))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="If-Match must be a db_version integer") from exc


async def _store_call(awaitable: Awaitable[T], *, if_match: int | None) -> T | JSONResponse:
    try:
        return await awaitable
    # Static details: exception text never reaches the response.
    except PreferenceTooLargeError:
        return _coded(413, "preference_too_large", f"Preference documents are limited to {PREFERENCE_MAX_BYTES} bytes")
    except PreferenceLimitError:
        detail = f"Accounts are limited to {PREFERENCE_MAX_DOCUMENTS} preference documents"
        return _coded(409, "preference_limit_exceeded", detail)
    except PreferenceVersionConflictError:
        if if_match is not None:
            return _coded(412, "preference_version_conflict", "Preference version does not match If-Match")
        return _coded(409, "preference_write_conflict", "Preference was modified concurrently; retry")


def _coded(status_code: int, code: str, detail: str, **extra: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, "code": code, **extra})


async def _is_platform_admin(auth_client: AuthClient, caller: Caller) -> bool:
    return caller.can_edit and await auth_client.has_role("system", "HelixAdmin")


async def _me(caller: Caller, auth_client: AuthClient, session_maker: async_sessionmaker[AsyncSession]) -> Me:
    record = await AccountProfileStore(session_maker).get(caller.account_id)
    email = caller.email or (record.primary_email if record else None)
    fallback = email or caller.principal_id or LOCAL_DISPLAY_NAME
    profile = (
        Profile(
            display_name=record.display_name() or fallback,
            display_name_override=record.display_name_override,
            idp_display_name=record.idp_display_name,
            avatar_url=record.avatar_url,
            timezone=record.timezone,
            locale=record.locale,
        )
        if record
        else Profile(display_name=fallback)
    )
    return Me(
        account_id=caller.account_id,
        principal_id=caller.principal_id,
        email=email,
        groups=caller.groups,
        caller_kind=caller.caller_kind,
        auth_method=caller.auth_method,
        profile=profile,
        can_edit_profile=caller.can_edit,
        is_platform_admin=await _is_platform_admin(auth_client, caller),
    )


@router.get("/v2/me", response_model=Me, openapi_extra=BEARER_AUTH_OPENAPI)
async def get_me(caller: CallerDep, auth_client: Auth, session_maker: SessionMaker) -> Me:
    """Return the caller's account, identity, and profile. Works with cookies, bearer tokens, and access keys."""
    return await _me(caller, auth_client, session_maker)


@router.patch("/v2/me/profile", response_model=Me, openapi_extra=BEARER_AUTH_OPENAPI, responses=_ERRORS)
async def update_my_profile(
    body: ProfileUpdate, caller: CallerDep, auth_client: Auth, session_maker: SessionMaker
) -> Me:
    """Change the caller's display name, time zone, or locale. Omitted fields are unchanged; null clears."""
    require_editable(caller)
    await AccountProfileStore(session_maker).update(caller.account_id, body.model_dump(exclude_unset=True))
    return await _me(caller, auth_client, session_maker)


async def _preference_scope(
    caller: Caller, auth_client: AuthClient, session_maker: async_sessionmaker[AsyncSession], workspace: str | None
) -> None:
    if caller.caller_kind != "principal":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Service accounts have no preferences")
    if workspace is not None:
        await require_readable_workspace(auth_client, session_maker, workspace)


def _document(
    namespace: str, workspace: str | None, record: PreferenceRecord | None, defaults: PreferenceDefaults | None
) -> PreferenceDocument:
    value = record.value if record else {}
    default_value = defaults.value if defaults else {}
    return PreferenceDocument(
        namespace=namespace,
        workspace=workspace,
        value=value,
        defaults=default_value,
        # User values win key by key; a key the user removed falls back to the default.
        effective=merge_patch(default_value, value),
        db_version=record.db_version if record else 0,
        updated_at=record.updated_at if record else None,
    )


@router.get(
    "/v2/me/preferences", response_model=PreferenceDocumentList, openapi_extra=BEARER_AUTH_OPENAPI, responses=_ERRORS
)
async def list_my_preferences(
    caller: CallerDep,
    auth_client: Auth,
    session_maker: SessionMaker,
    defaults_store: DefaultsStore,
    namespace: Annotated[str | None, Query(description="Namespace, or a prefix such as studio")] = None,
    workspace: _Workspace = None,
) -> PreferenceDocumentList:
    """List the caller's preference documents in one scope, including namespaces that only have defaults."""
    await _preference_scope(caller, auth_client, session_maker, workspace)
    namespace = namespace.rstrip(".") if namespace else None
    records = await AccountPreferenceStore(session_maker).list_documents(
        caller.account_id, namespace_prefix=namespace, workspace=workspace
    )
    by_namespace: dict[str, PreferenceRecord | None] = {record.namespace: record for record in records}
    defaults = {item.namespace: item for item in await defaults_store.list_defaults(namespace)}
    for item in defaults:
        by_namespace.setdefault(item, None)
    return PreferenceDocumentList(
        data=[_document(ns, workspace, by_namespace[ns], defaults.get(ns)) for ns in sorted(by_namespace)]
    )


@router.get(
    "/v2/me/preferences/{namespace}",
    response_model=PreferenceDocument,
    openapi_extra=BEARER_AUTH_OPENAPI,
    responses=_ERRORS,
)
async def get_my_preference(
    namespace: _Namespace,
    caller: CallerDep,
    auth_client: Auth,
    session_maker: SessionMaker,
    defaults_store: DefaultsStore,
    workspace: _Workspace = None,
) -> PreferenceDocument:
    """Return one preference document. A namespace the caller never wrote returns an empty value, db_version 0."""
    await _preference_scope(caller, auth_client, session_maker, workspace)
    record = await AccountPreferenceStore(session_maker).get(caller.account_id, namespace, workspace)
    return _document(namespace, workspace, record, await defaults_store.get(namespace))


async def _write_preference(
    namespace: str,
    workspace: str | None,
    body: PreferenceValue,
    *,
    merge: bool,
    if_match: str | None,
    caller: Caller,
    auth_client: AuthClient,
    session_maker: async_sessionmaker[AsyncSession],
    defaults_store: PreferenceDefaultsStore,
) -> PreferenceDocument | JSONResponse:
    require_editable(caller)
    await _preference_scope(caller, auth_client, session_maker, workspace)
    version = _parse_if_match(if_match)
    store = AccountPreferenceStore(session_maker)
    write = store.patch if merge else store.put
    result = await _store_call(
        write(caller.account_id, namespace, workspace, body.root, if_match=version), if_match=version
    )
    if isinstance(result, JSONResponse):
        return result
    return _document(namespace, workspace, result, await defaults_store.get(namespace))


@router.put(
    "/v2/me/preferences/{namespace}",
    response_model=PreferenceDocument,
    openapi_extra=BEARER_AUTH_OPENAPI,
    responses=_WRITE_ERRORS,
)
async def replace_my_preference(
    namespace: _Namespace,
    body: PreferenceValue,
    caller: CallerDep,
    auth_client: Auth,
    session_maker: SessionMaker,
    defaults_store: DefaultsStore,
    workspace: _Workspace = None,
    if_match: _IfMatch = None,
) -> PreferenceDocument | JSONResponse:
    """Replace the caller's document for a namespace."""
    return await _write_preference(
        namespace,
        workspace,
        body,
        merge=False,
        if_match=if_match,
        caller=caller,
        auth_client=auth_client,
        session_maker=session_maker,
        defaults_store=defaults_store,
    )


@router.patch(
    "/v2/me/preferences/{namespace}",
    response_model=PreferenceDocument,
    openapi_extra=BEARER_AUTH_OPENAPI,
    responses=_WRITE_ERRORS,
)
async def patch_my_preference(
    namespace: _Namespace,
    body: PreferenceValue,
    caller: CallerDep,
    auth_client: Auth,
    session_maker: SessionMaker,
    defaults_store: DefaultsStore,
    workspace: _Workspace = None,
    if_match: _IfMatch = None,
) -> PreferenceDocument | JSONResponse:
    """Apply an RFC 7396 merge patch to the caller's document. A null value removes that key."""
    return await _write_preference(
        namespace,
        workspace,
        body,
        merge=True,
        if_match=if_match,
        caller=caller,
        auth_client=auth_client,
        session_maker=session_maker,
        defaults_store=defaults_store,
    )


@router.delete(
    "/v2/me/preferences/{namespace}",
    status_code=status.HTTP_204_NO_CONTENT,
    openapi_extra=BEARER_AUTH_OPENAPI,
    responses=_WRITE_ERRORS,
)
async def delete_my_preference(
    namespace: _Namespace,
    caller: CallerDep,
    auth_client: Auth,
    session_maker: SessionMaker,
    workspace: _Workspace = None,
    if_match: _IfMatch = None,
) -> Response:
    """Delete the caller's document for a namespace, falling back to defaults."""
    require_editable(caller)
    await _preference_scope(caller, auth_client, session_maker, workspace)
    version = _parse_if_match(if_match)
    result = await _store_call(
        AccountPreferenceStore(session_maker).delete(caller.account_id, namespace, workspace, if_match=version),
        if_match=version,
    )
    return result if isinstance(result, JSONResponse) else Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/v2/preferences/defaults", response_model=PreferenceDefaultsList, openapi_extra=BEARER_AUTH_OPENAPI)
async def list_preference_defaults(defaults_store: DefaultsStore) -> PreferenceDefaultsList:
    """List deployment-wide preference defaults."""
    return PreferenceDefaultsList(data=await defaults_store.list_defaults())


@router.get(
    "/v2/preferences/defaults/{namespace}", response_model=PreferenceDefaults, openapi_extra=BEARER_AUTH_OPENAPI
)
async def get_preference_defaults(namespace: _Namespace, defaults_store: DefaultsStore) -> PreferenceDefaults:
    """Return the defaults for one namespace (an empty value when none are set)."""
    return await defaults_store.get(namespace) or PreferenceDefaults(
        namespace=namespace, value={}, managed_by="api", db_version=0
    )


async def _require_defaults_writable(
    namespace: str, caller: Caller, auth_client: AuthClient, defaults_store: PreferenceDefaultsStore
) -> JSONResponse | None:
    if not await _is_platform_admin(auth_client, caller):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only HelixAdmin can change defaults")
    if defaults_store.is_config_managed(namespace):
        return _coded(
            409,
            "preference_defaults_config_managed",
            f"Defaults for {namespace} are set in platform config (auth.preference_defaults)",
            managed_by="config",
        )
    return None


async def _write_defaults(
    namespace: str,
    body: PreferenceValue,
    *,
    merge: bool,
    if_match: str | None,
    caller: Caller,
    auth_client: AuthClient,
    defaults_store: PreferenceDefaultsStore,
) -> PreferenceDefaults | JSONResponse:
    if rejected := await _require_defaults_writable(namespace, caller, auth_client, defaults_store):
        return rejected
    version = _parse_if_match(if_match)
    store = defaults_store.preferences
    write = store.patch if merge else store.put
    result = await _store_call(
        write(await defaults_store.account_id(), namespace, None, body.root, if_match=version), if_match=version
    )
    if isinstance(result, JSONResponse):
        return result
    return PreferenceDefaults(namespace=namespace, value=result.value, managed_by="api", db_version=result.db_version)


@router.put(
    "/v2/preferences/defaults/{namespace}",
    response_model=PreferenceDefaults,
    openapi_extra=BEARER_AUTH_OPENAPI,
    responses=_DEFAULTS_WRITE_ERRORS,
)
async def replace_preference_defaults(
    namespace: _Namespace,
    body: PreferenceValue,
    caller: CallerDep,
    auth_client: Auth,
    defaults_store: DefaultsStore,
    if_match: _IfMatch = None,
) -> PreferenceDefaults | JSONResponse:
    """Replace the defaults for a namespace (HelixAdmin only)."""
    return await _write_defaults(
        namespace,
        body,
        merge=False,
        if_match=if_match,
        caller=caller,
        auth_client=auth_client,
        defaults_store=defaults_store,
    )


@router.patch(
    "/v2/preferences/defaults/{namespace}",
    response_model=PreferenceDefaults,
    openapi_extra=BEARER_AUTH_OPENAPI,
    responses=_DEFAULTS_WRITE_ERRORS,
)
async def patch_preference_defaults(
    namespace: _Namespace,
    body: PreferenceValue,
    caller: CallerDep,
    auth_client: Auth,
    defaults_store: DefaultsStore,
    if_match: _IfMatch = None,
) -> PreferenceDefaults | JSONResponse:
    """Merge-patch the defaults for a namespace (HelixAdmin only)."""
    return await _write_defaults(
        namespace,
        body,
        merge=True,
        if_match=if_match,
        caller=caller,
        auth_client=auth_client,
        defaults_store=defaults_store,
    )


@router.delete(
    "/v2/preferences/defaults/{namespace}",
    status_code=status.HTTP_204_NO_CONTENT,
    openapi_extra=BEARER_AUTH_OPENAPI,
    responses=_DEFAULTS_WRITE_ERRORS,
)
async def delete_preference_defaults(
    namespace: _Namespace,
    caller: CallerDep,
    auth_client: Auth,
    defaults_store: DefaultsStore,
    if_match: _IfMatch = None,
) -> Response:
    """Remove the defaults for a namespace (HelixAdmin only)."""
    if rejected := await _require_defaults_writable(namespace, caller, auth_client, defaults_store):
        return rejected
    version = _parse_if_match(if_match)
    result = await _store_call(
        defaults_store.preferences.delete(await defaults_store.account_id(), namespace, None, if_match=version),
        if_match=version,
    )
    return result if isinstance(result, JSONResponse) else Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/v2/profiles/resolve", response_model=ProfileResolveResponse, openapi_extra=BEARER_AUTH_OPENAPI)
async def resolve_profiles(
    body: ProfileResolveRequest,
    caller: CallerDep,
    auth_client: Auth,
    session_maker: SessionMaker,
    entities_client: Annotated[EntityClient, Depends(get_entity_client)],
) -> ProfileResolveResponse:
    """Resolve principal strings (subjects or emails) to public display names and avatars.

    Returns only directory fields: never email beyond the principal string sent, groups, or preferences.

    Only accounts that share an explicitly bound workspace with the caller resolve; platform admins see all.

    Everything else, including unresolvable, ambiguous, group, and wildcard principals, has ``display_name: null``.
    """
    profiles = AccountProfileStore(session_maker)
    resolved = await profiles.resolve(body.principals)
    account_ids = {item.account_id for item in resolved if item.account_id and item.account_id != caller.account_id}
    filterable = account_ids or any(item.ambiguous for item in resolved)
    if filterable and auth_client.auth_enabled and not await _is_platform_admin(auth_client, caller):
        hidden: set[str] = set()
        if account_ids:
            caller_keys = [key for key in (caller.principal_id, caller.email, *caller.groups) if key]
            hidden = account_ids - await accounts_sharing_a_workspace(
                entities_client.as_service("auth", internal=True),
                caller_keys,
                await profiles.principal_keys(account_ids),
            )
        # Hidden accounts, and ambiguous matches (which reveal that accounts exist), look like principals with no account.
        resolved = [
            item if item.account_id and item.account_id not in hidden else ResolvedPrincipal(principal=item.principal)
            for item in resolved
        ]
    return ProfileResolveResponse(data=[ResolvedProfile(**asdict(item)) for item in resolved])
