# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence and issuance services for Scoped Access Keys."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, TypeVar

from fastapi import Depends, HTTPException
from nemo_helix_plugin.auth.access_keys.issuer import AccessKeyFeatureDisabledError, AccessKeyWorkspaceBoundKeysError
from nemo_helix_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyListResponse,
    AccessKeyMetadataResponse,
    AccessKeyReversibleStatus,
    AccessKeyRotateResponse,
    AccessKeyStatus,
    AccessKeyWorkspaceGrant,
)
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, UpdateWorkspaceMemberRequest
from nhx.common.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nhx.common.auth.access_keys import (
    LEGACY_ACCESS_KEY_METADATA_VERSION,
    SERVICE_ACCOUNT_PRINCIPAL_PREFIX,
    AccessKeyIssuerService,
    AccessKeyValidationError,
)
from nhx.common.auth.models import Principal
from nhx.common.auth.token_claims import TokenClaims
from nhx.common.config import AuthConfig
from nhx.common.entities import ALL_WORKSPACES, EntityClient, EntityConflictError, EntityNotFoundError
from nhx.common.service.dependencies import get_entity_client
from nhx.core.auth.app.access_key_credentials import AccessKeyCredentialAdapter
from nhx.core.auth.app.account_resolution import get_account_session_maker
from nhx.core.auth.entities import AccessKeyEntity, RoleBindingEntity
from nhx.core.entities.app.repository import (
    AccountCredentialConflictError,
    AccountCredentialStore,
    AccountIdentityStore,
)

ACCESS_KEY_WORKSPACE = "system"
_OUTSIDE_ACCESS_MESSAGE = (
    "Service account '{account_id}' already has keys or role bindings outside workspace(s) {workspaces}, "
    "or a key bound to an earlier workspace of the same name; "
    "only a HelixAdmin can create or rotate keys for it"
)
_NAMESPACE_MESSAGE = (
    "Workspace Admins must use a service_account_id namespaced under a bound workspace "
    "('<workspace>/<name>', with <workspace> one of {workspaces})"
)
_ADMIN_REQUIRED_MESSAGE = "Service-bound Scoped Access Keys require HelixAdmin or Admin of every bound workspace"
# Page size for the entity-store scans behind the confinement checks (see AccessKeyRegistry._scan):
# the entity store's maximum.
_SCAN_PAGE_SIZE = 1000
logger = logging.getLogger(__name__)


class AccessKeyForbiddenError(AccessKeyValidationError):
    """The caller isn't allowed to create or rotate this service-bound key (reported as 403, not 400)."""


def _format_workspaces(names: Sequence[str]) -> str:
    return ", ".join(f"'{name}'" for name in names)


def _workspace_lookup_failed(workspace: str, exc: NemoClientError) -> HTTPException:
    # Callers treat a None lookup result as definitive, so a transient outage (transport error,
    # 5xx) must surface as a 503 instead of a spurious 403/404 or a client-error 400.
    logger.warning("Failed to read workspace '%s'", workspace, exc_info=True)
    return HTTPException(status_code=503, detail=f"Workspace '{workspace}' could not be read; retry")


async def resolve_workspace_id(workspaces_client: AsyncWorkspacesClient, workspace: str) -> str | None:
    """The workspace's current ID, or None if it doesn't exist or the caller can't read it.

    Any other client failure is raised as a 503 (see _workspace_lookup_failed).
    """
    try:
        return (await workspaces_client.get_workspace(name=workspace)).data().id
    except (NotFoundError, PermissionDeniedError):
        return None
    except NemoClientError as exc:
        raise _workspace_lookup_failed(workspace, exc) from exc


async def resolve_workspace_name(workspaces_client: AsyncWorkspacesClient, workspace_id: str) -> str | None:
    """The current name of the workspace with this ID, or None if it's gone or the caller can't see it.

    Workspaces are listed with the caller's access applied, so an invisible workspace reads as
    gone. Any other client failure is raised as a 503 (see _workspace_lookup_failed).
    """
    try:
        page = await workspaces_client.list_workspaces(
            query_params={"filter": json.dumps({"id": workspace_id}), "page_size": 1}
        )
        async for workspace in page.items():
            if workspace.id == workspace_id:
                return workspace.name
    except (NotFoundError, PermissionDeniedError):
        return None
    except NemoClientError as exc:
        raise _workspace_lookup_failed(workspace_id, exc) from exc
    return None


# Bounds retries of a lifecycle mutation's optimistic-lock update against version
# bumps from concurrent, unrelated writes (chiefly is_active's last_used_at updates
# on every authenticated request) that don't actually conflict with the mutation
# itself. A key with real traffic is exactly the case these operations exist to serve.
_MUTATION_MAX_ATTEMPTS = 3

# Coarsens last_used_at writes in is_active: a key with active traffic doesn't need a
# store write on every single authenticated request, just often enough to tell a caller
# their traffic has moved off a rotated-out key.
_LAST_USED_AT_RESOLUTION = timedelta(minutes=5)

_MutationResultT = TypeVar("_MutationResultT")


class _Retry:
    """Sentinel a `_retry_on_conflict` conflict handler returns to request another attempt."""


_RETRY = _Retry()

_MetadataT = TypeVar("_MetadataT", bound=AccessKeyMetadataResponse)
_ScannedT = TypeVar("_ScannedT", AccessKeyEntity, RoleBindingEntity)


@dataclass(frozen=True)
class _BoundWorkspaces:
    """The workspaces a service-bound key is bound to: their IDs, and current names aligned with them.

    The IDs are the binding; the names are only handles for role checks, membership grants and
    messages, resolved when the key is created or used.
    """

    ids: tuple[str, ...] = ()
    names: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.ids)


# Service-bound keys are machine-to-machine credentials owned by the platform, not
# by the creating individual. Every lifecycle operation therefore
# uses this callback to require a current administrator, including when the caller
# originally created the key. It takes the IDs of the key's bound workspaces (empty if unbound):
# a HelixAdmin qualifies for any key, a workspace Admin only for keys bound to workspaces they
# are Admin of, all of them. A workspace deleted and recreated under the same name has a new ID,
# so its Admins don't qualify.
class AdminOverride(Protocol):
    def __call__(self, workspace_ids: Sequence[str]) -> Awaitable[bool]:
        """Whether the caller may manage keys bound to the workspaces with ``workspace_ids``."""
        ...


def _memoize_admin_override(admin_override: AdminOverride | None) -> AdminOverride | None:
    """Cache admin_override results (per set of workspaces) for the lifetime of one lifecycle call.

    _get_owned() may be invoked twice within one revoke()/suspend()/unsuspend() call
    (initial read, then a re-read after a losing optimistic-lock race). Without this,
    each read would trigger its own PDP has_role round trip for service-bound keys.
    """
    if admin_override is None:
        return None
    results: dict[tuple[str, ...], bool] = {}

    async def cached(workspace_ids: Sequence[str]) -> bool:
        """Return the memoized admin_override result for these workspace IDs."""
        key = tuple(workspace_ids)
        if key not in results:
            results[key] = await admin_override(workspace_ids)
        return results[key]

    return cached


class AccessKeyNotFoundError(Exception):
    """Raised when a key does not exist or is not owned by the caller."""


class AccessKeyStateConflictError(Exception):
    """Raised when an irreversible lifecycle state prevents a transition."""


class AccessKeyRegistry:
    """Durable access-key lifecycle records with a legacy entity fallback."""

    def __init__(
        self,
        entity_client: EntityClient,
        credential_adapter: AccessKeyCredentialAdapter | None = None,
    ) -> None:
        self._entity_client = entity_client
        self._credential_adapter = credential_adapter

    async def add(
        self,
        key: AccessKeyCreateResponse,
        *,
        owner_principal: str | None = None,
        owner_account_id: str | None = None,
        bound_workspace_ids: Sequence[str] = (),
    ) -> None:
        owner = owner_principal or key.principal
        record = AccessKeyEntity(
            name=key.jti,
            workspace=ACCESS_KEY_WORKSPACE,
            key_name=key.name,
            description=key.description,
            principal=owner,
            subject_principal=key.principal if key.principal != owner else None,
            bound_workspace_ids=list(bound_workspace_ids),
            entity_type=key.entity_type,
            issuer=key.issuer,
            audiences=key.audiences,
            scope=key.scope,
            issued_at=key.created_at,
            expires_at=key.expires_at,
        )
        if self._credential_adapter is not None:
            await self._credential_adapter.create(record, owner_account_id=owner_account_id)
            return
        await self._entity_client.create(record)

    async def discard_unreturned(self, jti: str) -> None:
        if self._credential_adapter is not None and await self._credential_adapter.delete(jti):
            return
        try:
            await self._entity_client.delete(
                AccessKeyEntity,
                name=jti,
                workspace=ACCESS_KEY_WORKSPACE,
            )
        except EntityNotFoundError:
            pass

    async def list_for_principal(
        self,
        principal: str,
        *,
        page: int,
        page_size: int,
        include_service_accounts: bool = False,
        owner_account_id: str | None = None,
        admin_override: AdminOverride | None = None,
        service_key_workspace_ids: Sequence[str] = (),
    ) -> AccessKeyListResponse:
        # include_service_accounts (a current HelixAdmin) surfaces every service-bound key, not just
        # the ones the caller created. Otherwise the caller's own records are fetched, plus the
        # service-bound keys bound to any of ``service_key_workspace_ids`` (the workspaces the caller
        # can see), whoever created them, so co-admins see each other's keys. Filtering on the
        # bound IDs in the store keeps other tenants' service-bound keys from crowding the caller's
        # keys out of a page. A service-bound key is kept only while admin_override accepts all of
        # its bound workspaces, so the Admin checks run only for workspaces that have such keys.
        # EntityBase fields (all fields on AccessKeyEntity besides the base name/workspace/etc.)
        # live in the data JSON column, so filter_operation needs the same `data.` prefix
        # that _convert_filter_obj_to_filter_str applies for the filter_obj shorthand.
        own_principal = ComparisonOperation(operator=FilterOperator.EQ, field="data.principal", value=principal)
        service_keys = ComparisonOperation(
            operator=FilterOperator.EQ, field="data.entity_type", value="SERVICE_ACCOUNT"
        )
        if include_service_accounts:
            filter_operation: FilterOperation = LogicalOperation(
                operator=FilterOperator.OR, operations=[own_principal, service_keys]
            )
        elif service_key_workspace_ids:
            # One $elemMatch/$in, so a caller who sees many workspaces still gets a single array scan.
            bound_to_candidates = ComparisonOperation(
                operator=FilterOperator.ELEM_MATCH,
                field="data.bound_workspace_ids",
                value={FilterOperator.IN.value: list(service_key_workspace_ids)},
            )
            filter_operation = LogicalOperation(
                operator=FilterOperator.OR,
                operations=[
                    own_principal,
                    LogicalOperation(
                        operator=FilterOperator.AND,
                        operations=[service_keys, bound_to_candidates],
                    ),
                ],
            )
        else:
            filter_operation = own_principal
        result = await self._entity_client.list(
            AccessKeyEntity,
            workspace=ACCESS_KEY_WORKSPACE,
            filter_operation=filter_operation,
            sort="-issued_at",
            page=page,
            page_size=page_size,
        )
        records = result.data
        credential_has_more = False
        if self._credential_adapter is not None and owner_account_id is not None:
            credential_records, credential_has_more = await self._credential_adapter.list_for_owner(
                owner_account_id,
                # Keys the caller created, service-bound ones included, are already covered by the
                # owner match. Only a HelixAdmin pulls in other owners' service keys: asking for them
                # on behalf of every caller would page every tenant's service keys through each list.
                include_service_accounts=include_service_accounts,
                service_bound_workspace_ids=service_key_workspace_ids,
                offset=(page - 1) * page_size,
                limit=page_size,
            )
            records = sorted(
                [*credential_records, *records],
                key=lambda record: record.issued_at,
                reverse=True,
            )
        # During the dual-store migration window, a page can exceed page_size:
        # both stores already applied the requested offset and limit, so trimming
        # here would permanently hide a fetched row without a merged cursor.
        visible_records = []
        for record in records:
            if (
                record.is_service_account()
                and not include_service_accounts
                and (admin_override is None or not await admin_override(record.bound_workspace_ids))
            ):
                continue
            visible_records.append(record)
        # Short non-admin pages after Python filtering are intentional: favor never hiding valid keys over exact pagination; no fix needed.
        return AccessKeyListResponse(
            data=[self._metadata(record) for record in visible_records],
            has_more=credential_has_more or page < result.pagination.total_pages,
        )

    async def has_service_account_access_outside_workspaces(
        self,
        service_account_principal: str,
        *,
        workspace_ids: Sequence[str],
        workspace_names: Sequence[str],
    ) -> bool:
        """Whether this service account has a key or an active role binding outside the given workspaces.

        Keys are checked in any status, in both the entity store and, when configured, the
        credential store. A key is outside unless it is bound, and only to workspaces among
        ``workspace_ids``: deleting a workspace doesn't revoke its keys, so a recreated workspace of
        the same name must not take over an account whose old key is still live. Role bindings
        catch access granted without a key, e.g. membership a HelixAdmin or another workspace's
        Admin pre-provisioned for the principal. Bindings are stored by workspace name, so they're
        compared against ``workspace_names`` (the current names of ``workspace_ids``): deleting a
        workspace deletes its bindings, so none can outlive the workspace.
        """
        allowed_ids = set(workspace_ids)
        allowed_names = set(workspace_names)

        def is_outside(record: AccessKeyEntity) -> bool:
            return not record.bound_workspace_ids or not set(record.bound_workspace_ids) <= allowed_ids

        if await self._any_service_account_key(service_account_principal, is_outside):
            return True
        bindings = self._scan(
            RoleBindingEntity,
            workspace=ALL_WORKSPACES,
            # Revoked bindings are filtered in Python below: the entity service only supports
            # $exists for registered relationships, not plain fields like revoked_at.
            condition=ComparisonOperation(
                operator=FilterOperator.EQ, field="data.principal", value=service_account_principal
            ),
        )
        async with aclosing(bindings):
            async for page in bindings:
                if any(binding.revoked_at is None and binding.workspace not in allowed_names for binding in page):
                    return True
        return False

    async def workspace_bound_keys_widened_by(
        self, service_account_principal: str, granted_workspace_ids: Sequence[str]
    ) -> list[str] | None:
        """The bound workspace IDs of this account's live keys that a new key's grants reach beyond, if any.

        Roles belong to the service account, not to a key, so a key bound to workspaces (and
        managed by their Admins) gains every role a new key grants the account. A new key's grants
        stay within an existing key's reach only if every workspace it grants roles in (by ID) is
        one that key is bound to; a key that grants nothing widens nothing. Revoked and expired keys
        can't authenticate, so they don't count.

        Returns None when no live bound key would be widened, else the sorted IDs of the workspaces
        such keys are bound to. Callers reject a disabled service account first (see
        is_service_account_disabled), since its credential-store keys can't be listed.
        """
        granted = set(granted_workspace_ids)
        if not granted:
            return None
        widened_ids: set[str] = set()

        def collect_widened(record: AccessKeyEntity) -> bool:
            if not record.bound_workspace_ids or self._status(record) in ("REVOKED", "EXPIRED"):
                return False
            if granted <= set(record.bound_workspace_ids):
                return False
            widened_ids.update(record.bound_workspace_ids)
            # Keep scanning so the result names every affected workspace.
            return False

        await self._any_service_account_key(service_account_principal, collect_widened)
        return sorted(widened_ids) or None

    async def is_service_account_disabled(self, service_account_principal: str) -> bool:
        """Whether the service account's account identity exists but is disabled.

        Always False without a credential store, which is the only place identities are tracked.
        """
        return self._credential_adapter is not None and await self._credential_adapter.is_service_account_disabled(
            service_account_principal
        )

    async def _any_service_account_key(
        self, service_account_principal: str, predicate: Callable[[AccessKeyEntity], bool]
    ) -> bool:
        """Whether any key, in any status, whose subject is this service account satisfies ``predicate``.

        Scans the entity store and, when configured, the credential store. An account whose
        credential-store identity is disabled counts as a match (see
        AccessKeyCredentialAdapter.any_for_service_account).
        """
        if self._credential_adapter is not None and await self._credential_adapter.any_for_service_account(
            service_account_principal, predicate
        ):
            return True
        keys = self._scan(
            AccessKeyEntity,
            workspace=ACCESS_KEY_WORKSPACE,
            condition=ComparisonOperation(
                operator=FilterOperator.EQ, field="data.subject_principal", value=service_account_principal
            ),
        )
        async with aclosing(keys):
            async for page in keys:
                if any(predicate(record) for record in page):
                    return True
        return False

    async def _scan(
        self, entity_type: type[_ScannedT], *, workspace: str, condition: FilterOperation
    ) -> AsyncGenerator[list[_ScannedT], None]:
        """Yield every entity matching ``condition``, a page at a time, paging by entity ID.

        For the confinement checks, where a skipped row would fail open: unlike offset paging,
        each page starts after the last ID read, so a row deleted while the scan runs can't shift
        an unread row onto a page already read.
        """
        after_id: str | None = None
        while True:
            filter_operation = (
                condition
                if after_id is None
                else LogicalOperation(
                    operator=FilterOperator.AND,
                    operations=[
                        condition,
                        ComparisonOperation(operator=FilterOperator.GT, field="id", value=after_id),
                    ],
                )
            )
            result = await self._entity_client.list(
                entity_type,
                workspace=workspace,
                filter_operation=filter_operation,
                sort="id",
                page=1,
                page_size=_SCAN_PAGE_SIZE,
            )
            yield result.data
            if result.pagination.total_pages <= 1 or not result.data:
                return
            after_id = result.data[-1].id
            if not after_id:
                # Can't page past an entity without an ID; fail closed rather than skip the rest.
                raise RuntimeError(f"Cannot page past a {entity_type.__name__} without an ID")

    async def _retry_on_conflict(
        self,
        do_attempt: Callable[[], Awaitable[_MutationResultT]],
        handle_conflict: Callable[[], Awaitable[_MutationResultT | _Retry]],
    ) -> _MutationResultT:
        """Run `do_attempt`, retrying up to `_MUTATION_MAX_ATTEMPTS` times on
        `EntityConflictError`.

        `EntityClient.update` uses db_version optimistic locking, so every conflict
        is handed to `handle_conflict`, which re-reads the current record and either
        returns a final result (an early outcome, or re-raising a state error) or the
        `_RETRY` sentinel to try `do_attempt` again against the fresh state — the
        conflict may be from an unrelated concurrent write (e.g. is_active's
        last_used_at update) rather than a real state clash.
        """
        for attempt in range(_MUTATION_MAX_ATTEMPTS):
            try:
                return await do_attempt()
            except EntityConflictError:
                outcome = await handle_conflict()
                if not isinstance(outcome, _Retry):
                    return outcome
                if attempt == _MUTATION_MAX_ATTEMPTS - 1:
                    raise
        raise AssertionError("unreachable: loop always returns or raises")

    async def revoke(self, jti: str, principal: str, *, admin_override: AdminOverride | None = None) -> bool:
        # Note: unlike is_active, revoke has no backfill path for legacy v1 keys that have
        # never authenticated after migration. Without the original JWT claims we cannot
        # construct a valid entity, so callers receive 404 until the key authenticates once.
        admin_override = _memoize_admin_override(admin_override)
        record = await self._get_owned(jti, principal, admin_override=admin_override)
        if record.status == "REVOKED":
            return False

        async def do_attempt() -> bool:
            await self._update(record.model_copy(update={"status": "REVOKED"}))
            return True

        async def handle_conflict() -> bool | _Retry:
            nonlocal record
            try:
                record = await self._get_owned(jti, principal, admin_override=admin_override)
            except AccessKeyNotFoundError:
                # The key was concurrently hard-deleted between our update and this
                # read. Treat as already-revoked (idempotent outcome).
                return False
            if record.status == "REVOKED":
                return False
            return _RETRY

        return await self._retry_on_conflict(do_attempt, handle_conflict)

    async def get_for_rotation(
        self, jti: str, principal: str, *, admin_override: AdminOverride | None = None
    ) -> AccessKeyEntity:
        """Check the rotation target (the key `rotates` points to) before minting its replacement.

        Raises AccessKeyValidationError (400), not AccessKeyNotFoundError, so an invalid target
        fails the request instead of minting an orphan key. Personal keys only: service-bound
        keys have no stable non-admin owner to rotate against.
        """
        try:
            record = await self._get_owned(jti, principal, admin_override=admin_override)
        except AccessKeyNotFoundError as exc:
            raise AccessKeyValidationError(
                f"rotates references a Scoped Access Key that does not exist or is not owned by the caller: {jti}"
            ) from exc
        if record.is_service_account():
            raise AccessKeyValidationError(
                "rotates only supports personal Scoped Access Keys without a service account"
            )
        return record

    async def suspend(
        self, jti: str, principal: str, *, admin_override: AdminOverride | None = None
    ) -> tuple[bool, AccessKeyReversibleStatus]:
        return await self._set_suspension(jti, principal, suspended=True, admin_override=admin_override)

    async def unsuspend(
        self, jti: str, principal: str, *, admin_override: AdminOverride | None = None
    ) -> tuple[bool, AccessKeyReversibleStatus]:
        return await self._set_suspension(jti, principal, suspended=False, admin_override=admin_override)

    async def _set_suspension(
        self,
        jti: str,
        principal: str,
        *,
        suspended: bool,
        admin_override: AdminOverride | None = None,
    ) -> tuple[bool, AccessKeyReversibleStatus]:
        target_status: Literal["ACTIVE", "SUSPENDED"] = "SUSPENDED" if suspended else "ACTIVE"
        completed_action = "suspended" if suspended else "unsuspended"
        admin_override = _memoize_admin_override(admin_override)
        record = await self._get_owned(jti, principal, admin_override=admin_override)
        self._ensure_not_rotating(jti, record, action=completed_action)
        if record.status == "REVOKED":
            raise AccessKeyStateConflictError(f"Revoked Scoped Access Key {jti} cannot be {completed_action}")
        effective_status = self._reversible_status(record)
        if effective_status == "EXPIRED":
            return False, effective_status
        if record.status == target_status:
            return False, effective_status

        async def do_attempt() -> tuple[bool, AccessKeyReversibleStatus]:
            updated = record.model_copy(update={"status": target_status})
            await self._update(updated)
            return True, self._reversible_status(updated)

        async def handle_conflict() -> tuple[bool, AccessKeyReversibleStatus] | _Retry:
            nonlocal record
            record = await self._get_owned(jti, principal, admin_override=admin_override)
            self._ensure_not_rotating(jti, record, action=completed_action)
            if record.status == "REVOKED":
                raise AccessKeyStateConflictError(f"Revoked Scoped Access Key {jti} cannot be {completed_action}")
            current_status = self._reversible_status(record)
            if current_status == "EXPIRED" or record.status == target_status:
                return False, current_status
            return _RETRY

        return await self._retry_on_conflict(do_attempt, handle_conflict)

    async def get_rotatable(
        self, jti: str, principal: str, *, admin_override: AdminOverride | None = None
    ) -> AccessKeyEntity:
        """Return `jti`'s current record if it is eligible to be rotated, else raise.

        Read-only: callers mint the successor key from the returned record's
        attributes, then call `begin_rotation` to transition this record to ROTATING.
        """
        record = await self._get_owned(jti, principal, admin_override=admin_override)
        self._ensure_rotatable(jti, record)
        return record

    async def get_status(
        self, jti: str, principal: str, *, admin_override: AdminOverride | None = None
    ) -> AccessKeyEntity:
        """Return an owned key record without requiring a particular lifecycle state."""
        return await self._get_owned(jti, principal, admin_override=admin_override)

    async def begin_rotation(
        self,
        jti: str,
        principal: str,
        *,
        grace_period_seconds: int,
        successor_jti: str,
        admin_override: AdminOverride | None = None,
    ) -> AccessKeyEntity:
        admin_override = _memoize_admin_override(admin_override)
        try:
            record = await self._get_owned(jti, principal, admin_override=admin_override)
        except Exception as exc:
            # No update has been attempted in this rotation iteration yet.
            exc.__dict__["write_attempted"] = False
            raise
        self._ensure_rotatable(jti, record)

        async def do_attempt() -> AccessKeyEntity:
            # Recomputed each attempt so a retry's deadline reflects the current
            # time, and capped by the old key's own natural expiry so the reported
            # deadline is never later than the instant the key would die anyway
            # (natural expiry always takes precedence over rotation grace).
            grace_period_expires_at = datetime.now(tz=UTC) + timedelta(seconds=grace_period_seconds)
            if record.expires_at is not None and record.expires_at < grace_period_expires_at:
                grace_period_expires_at = record.expires_at
            updated = record.model_copy(
                update={
                    "status": "ROTATING",
                    "grace_period_expires_at": grace_period_expires_at,
                    "rotation_successor_jti": successor_jti,
                }
            )
            await self._update(updated)
            return updated

        async def handle_conflict() -> AccessKeyEntity | _Retry:
            nonlocal record
            # Re-read so a caller that already validated get_rotatable sees a precise
            # reason if a concurrent lifecycle change (e.g. a revoke) won the race,
            # rather than a bare conflict. If the record is still rotatable, the
            # conflict was from an unrelated concurrent write (e.g. is_active's
            # last_used_at update) — retry against the fresh db_version instead of
            # failing the rotation.
            try:
                record = await self._get_owned(jti, principal, admin_override=admin_override)
            except Exception as exc:
                # The conflict proves the preceding update was rejected, so this
                # re-read can also fail only before a rotation write commits.
                exc.__dict__["write_attempted"] = False
                raise
            self._ensure_rotatable(jti, record)
            return _RETRY

        return await self._retry_on_conflict(do_attempt, handle_conflict)

    @staticmethod
    def _ensure_rotatable(jti: str, record: AccessKeyEntity) -> None:
        if record.status != "ACTIVE":
            raise AccessKeyStateConflictError(
                f"Scoped Access Key {jti} must be ACTIVE to rotate (current status: {record.status})"
            )
        if AccessKeyRegistry._status(record) == "EXPIRED":
            raise AccessKeyStateConflictError(f"Expired Scoped Access Key {jti} cannot be rotated")

    @staticmethod
    def _ensure_not_rotating(jti: str, record: AccessKeyEntity, *, action: str) -> None:
        if record.status == "ROTATING":
            raise AccessKeyStateConflictError(f"Scoped Access Key {jti} is being rotated and cannot be {action}")

    async def is_active(self, jti: str, principal: str, *, claims: TokenClaims | None = None) -> bool:
        record: AccessKeyEntity | None = None
        if self._credential_adapter is not None:
            stored = await self._credential_adapter.get(jti)
            if stored is not None:
                record = await self._credential_adapter.get_active(jti)
                if record is None or (record.subject_principal or record.principal) != principal:
                    return False
        try:
            record = record or await self._get_for_subject(jti, principal)
        except AccessKeyNotFoundError:
            if claims is None:
                return False
            record = await self._backfill_legacy_record(jti, principal, claims)
            if record is None:
                return False
        effective_status = self._status(record, leeway_seconds=30)
        # ROTATING keys skip the coalescing resolution and get a write on every
        # authenticated request: last_used_at is the caller's signal for confirming
        # traffic has moved off a rotated-out key before revoking it, and that key
        # only exists for the bounded grace window, so a stale-by-up-to-5-minutes
        # timestamp right after cutover could read as "no more traffic" when there
        # still is some, prompting a premature revoke.
        if effective_status in {"ACTIVE", "ROTATING"} and (
            effective_status == "ROTATING"
            or record.last_used_at is None
            or datetime.now(tz=UTC) - record.last_used_at >= _LAST_USED_AT_RESOLUTION
        ):
            current_record = record

            async def do_attempt() -> bool:
                await self._update(current_record.model_copy(update={"last_used_at": datetime.now(tz=UTC)}))
                return True

            async def handle_conflict() -> bool | _Retry:
                nonlocal current_record, effective_status
                try:
                    current_record = await self._get_for_subject(jti, principal)
                except Exception:
                    logger.warning(
                        "Failed to refresh Scoped Access Key after a concurrent last-used timestamp update",
                        extra={"access_key_jti": jti, "actor_principal": principal},
                        exc_info=True,
                    )
                    return False
                effective_status = self._status(current_record, leeway_seconds=30)
                if effective_status not in {"ACTIVE", "ROTATING"}:
                    return False
                if (
                    effective_status != "ROTATING"
                    and current_record.last_used_at is not None
                    and datetime.now(tz=UTC) - current_record.last_used_at < _LAST_USED_AT_RESOLUTION
                ):
                    return False
                return _RETRY

            try:
                await self._retry_on_conflict(do_attempt, handle_conflict)
            except EntityConflictError:
                logger.warning(
                    "Failed to update last-used timestamp for Scoped Access Key due to concurrent updates",
                    extra={"access_key_jti": jti, "actor_principal": principal},
                    exc_info=True,
                )
            except Exception:
                logger.warning(
                    "Failed to update last-used timestamp for Scoped Access Key",
                    extra={"access_key_jti": jti, "actor_principal": principal},
                    exc_info=True,
                )
        # ROTATING keys stay usable through their grace period so callers can cut
        # over to the successor key without downtime (dual-active rotation).
        return effective_status in ("ACTIVE", "ROTATING")

    async def _get_for_subject(self, jti: str, principal: str) -> AccessKeyEntity:
        record = await self._get(jti)
        if (record.subject_principal or record.principal) != principal:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        return record

    async def _get_owned(
        self, jti: str, principal: str, *, admin_override: AdminOverride | None = None
    ) -> AccessKeyEntity:
        record = await self._get(jti)
        if record.is_service_account():
            # Service-bound keys belong to the platform, not to the administrator who
            # created them. Require the caller to be a *current* administrator of the
            # bound workspace (or a HelixAdmin) for every lifecycle operation, including
            # when that caller is the recorded creator.
            if admin_override is not None and await admin_override(record.bound_workspace_ids):
                return record
        elif record.principal == principal:
            return record
        raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")

    async def _get(self, jti: str) -> AccessKeyEntity:
        if self._credential_adapter is not None:
            record = await self._credential_adapter.get(jti)
            if record is not None:
                return record
        try:
            return await self._entity_client.get(
                AccessKeyEntity,
                name=jti,
                workspace=ACCESS_KEY_WORKSPACE,
            )
        except EntityNotFoundError as exc:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found") from exc

    async def _update(self, record: AccessKeyEntity) -> None:
        if self._credential_adapter is not None and await self._credential_adapter.get(record.name) is not None:
            try:
                await self._credential_adapter.update(record)
            except AccountCredentialConflictError as exc:
                raise EntityConflictError(str(exc)) from exc
            return
        await self._entity_client.update(record)

    @staticmethod
    def _metadata(record: AccessKeyEntity) -> AccessKeyMetadataResponse:
        effective_status = AccessKeyRegistry._status(record)
        return AccessKeyMetadataResponse(
            jti=record.name,
            name=record.key_name,
            description=record.description,
            principal=record.subject_principal or record.principal,
            entity_type=record.entity_type,
            # IDs until the issuer resolves the current names (see PersistentAccessKeyIssuer._with_bound_names).
            bound_workspaces=list(record.bound_workspace_ids),
            bound_workspace_ids=list(record.bound_workspace_ids),
            # Report lifecycle status against the published expiration instant.
            # Clock-skew leeway applies only while authenticating the JWT.
            status=effective_status,
            issuer=record.issuer,
            audiences=list(dict.fromkeys(record.audiences)),
            scope=list(dict.fromkeys(record.scope)),
            created_at=record.issued_at,
            expires_at=record.expires_at,
            grace_period_expires_at=record.grace_period_expires_at if effective_status == "ROTATING" else None,
            last_used_at=record.last_used_at,
        )

    @staticmethod
    def _status(record: AccessKeyEntity, *, leeway_seconds: int = 0) -> AccessKeyStatus:
        if record.status == "REVOKED":
            return "REVOKED"
        if record.expires_at is not None and datetime.now(tz=UTC) >= record.expires_at + timedelta(
            seconds=leeway_seconds
        ):
            return "EXPIRED"
        if record.status == "ROTATING":
            # Finalization is lazy, mirroring how natural expiry above is derived at
            # read time rather than written by a background job: once grace_period_expires_at
            # passes, the rotated-out key is reported (and authenticates) as REVOKED.
            # leeway_seconds is clock-skew tolerance for validating a JWT's own `exp`
            # claim against the issuing server's clock; the rotation grace deadline is
            # a server-side-only comparison, so it must not get the same extension —
            # otherwise a key could keep authenticating past the deadline metadata
            # already reports as REVOKED.
            if record.grace_period_expires_at is not None and datetime.now(tz=UTC) >= record.grace_period_expires_at:
                return "REVOKED"
            return "ROTATING"
        if record.status == "SUSPENDED":
            return "SUSPENDED"
        return "ACTIVE"

    @staticmethod
    def _reversible_status(record: AccessKeyEntity) -> AccessKeyReversibleStatus:
        effective_status = AccessKeyRegistry._status(record)
        if effective_status in ("REVOKED", "ROTATING"):
            # Callers must guard with _ensure_not_rotating (and never pass a REVOKED
            # record) before reaching here; both are lifecycle states that suspend/
            # unsuspend explicitly reject earlier, not states this helper should report.
            raise AssertionError(f"A reversible access-key transition cannot produce {effective_status} status")
        return effective_status

    async def _backfill_legacy_record(
        self,
        jti: str,
        principal: str,
        claims: TokenClaims,
    ) -> AccessKeyEntity | None:
        record = self._record_from_validated_claims(jti, principal, claims)
        if record is None:
            return None
        try:
            created = await self._entity_client.create(record)
        except EntityConflictError:
            try:
                return await self._get_for_subject(jti, principal)
            except AccessKeyNotFoundError:
                return None
        logger.info(
            "Backfilled legacy Scoped Access Key lifecycle record",
            extra={
                "audit_event": "access_key.backfilled",
                "actor_principal": principal,
                "access_key_jti": jti,
            },
        )
        return created

    @classmethod
    def _record_from_validated_claims(
        cls,
        jti: str,
        principal: str,
        claims: TokenClaims,
    ) -> AccessKeyEntity | None:
        raw_claims = claims.raw_claims
        # Keep the registry boundary defensive even though the authenticate
        # endpoint currently derives ``jti`` and ``principal`` from these claims.
        if raw_claims.get("jti") != jti or claims.subject != principal:
            return None
        issuer = raw_claims.get("iss")
        issued_at = cls._datetime_from_claim(raw_claims.get("iat"))
        if not isinstance(issuer, str) or issued_at is None:
            return None
        audiences = cls._audiences_from_claim(raw_claims.get("aud"))
        if not audiences:
            return None

        metadata = raw_claims.get("nhx_access_key")
        if not isinstance(metadata, dict):
            return None
        if metadata.get("version") != LEGACY_ACCESS_KEY_METADATA_VERSION:
            logger.warning(
                "Access key %s (version=%s) has no registry record and cannot be backfilled; "
                "this key will be rejected until its record is restored",
                jti,
                metadata.get("version"),
            )
            return None
        key_name = metadata.get("name")
        return AccessKeyEntity(
            name=jti,
            workspace=ACCESS_KEY_WORKSPACE,
            key_name=key_name if isinstance(key_name, str) else None,
            # description is not embedded in JWT claims for any version; always None on backfill
            description=None,
            principal=principal,
            issuer=issuer,
            audiences=audiences,
            issued_at=issued_at,
            expires_at=cls._datetime_from_claim(raw_claims.get("exp")),
        )

    @staticmethod
    def _audiences_from_claim(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return list(dict.fromkeys(audience for audience in value if isinstance(audience, str)))
        return []

    @staticmethod
    def _datetime_from_claim(value: object) -> datetime | None:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=UTC)
        if isinstance(value, datetime):
            return value.astimezone(UTC)
        return None


async def get_access_key_registry(entity_client: EntityClient = Depends(get_entity_client)) -> AccessKeyRegistry:
    session_maker = await get_account_session_maker()
    adapter = AccessKeyCredentialAdapter(
        AccountCredentialStore(session_maker),
        AccountIdentityStore(session_maker),
    )
    return AccessKeyRegistry(entity_client.as_service("auth", internal=True), adapter)


class PersistentAccessKeyIssuer:
    """Signs access keys and records their lifecycle before returning them."""

    def __init__(
        self,
        config: AuthConfig,
        principal: Principal,
        registry: AccessKeyRegistry,
        workspaces_client: AsyncWorkspacesClient,
        *,
        admin_override: AdminOverride | None = None,
        caller_scope: list[str] | None = None,
        workspace_names: dict[str, str | None] | None = None,
    ) -> None:
        self._issuer = AccessKeyIssuerService(config=config, principal=principal)
        self._config = config
        self._registry = registry
        self._workspaces_client = workspaces_client
        self.principal = principal.id
        self.account_id = principal.account_id
        # Lets a current HelixAdmin, or an Admin of the bound workspace, manage a
        # service-bound key; see AdminOverride. Memoized for the lifetime of this issuer
        # instance (one per request, see get_access_key_issuer) so that the endpoint-level
        # pre-check (can_create_bound_key) and create_async's own defense-in-depth re-check
        # share a single PDP has_role round trip instead of each paying for one.
        self._admin_override = _memoize_admin_override(admin_override)
        # The caller's own Scoped Access Key scope, if any; stops it from minting a broader
        # replacement. None = unrestricted caller.
        self._caller_scope = frozenset(caller_scope) if caller_scope is not None else None
        # Workspace ID -> current name (None if gone or invisible to the caller), for this request.
        # Shared with admin_override when the caller passes the same dict (see get_access_key_issuer),
        # so names this issuer has already read aren't looked up again by the Admin checks.
        self._workspace_names: dict[str, str | None] = workspace_names if workspace_names is not None else {}
        # Workspace name -> current ID (None if gone or invisible to the caller), for this request.
        # Only for resolving the names a request asks to bind; checks that must see a fresh ID
        # (see _bound_workspaces_unchanged) read it directly.
        self._workspace_ids: dict[str, str | None] = {}

    async def is_platform_admin(self) -> bool:
        """Whether the caller is a current HelixAdmin."""
        return await self.can_administer([])

    async def can_administer(self, workspace_ids: Sequence[str]) -> bool:
        """Whether the caller is a HelixAdmin, or Admin of every workspace in ``workspace_ids``.

        Uses the memoized admin_override, so gating before create_async costs no extra PDP call.
        """
        return self._admin_override is not None and await self._admin_override(workspace_ids)

    async def can_create_bound_key(self, workspace_names: Sequence[str]) -> bool:
        """Whether the caller may create a service-bound key bound to ``workspace_names``.

        A HelixAdmin always may (create_async still rejects a workspace that doesn't exist). Anyone
        else needs at least one bound workspace, each one visible to them, and Admin of all of them.
        """
        if await self.is_platform_admin():
            return True
        if not workspace_names:
            return False
        workspace_ids = []
        for name in workspace_names:
            workspace_id = await self._workspace_id(name)
            if workspace_id is None:
                return False
            workspace_ids.append(workspace_id)
        return await self.can_administer(workspace_ids)

    async def _workspace_id(self, name: str) -> str | None:
        """The current ID of a workspace, by name; cached for this request."""
        if name not in self._workspace_ids:
            self._workspace_ids[name] = await resolve_workspace_id(self._workspaces_client, name)
        return self._workspace_ids[name]

    async def _workspace_name(self, workspace_id: str) -> str | None:
        """The current name of a workspace, by ID; cached for this request."""
        if workspace_id not in self._workspace_names:
            self._workspace_names[workspace_id] = await resolve_workspace_name(self._workspaces_client, workspace_id)
        return self._workspace_names[workspace_id]

    async def _display_names(self, workspace_ids: Sequence[str]) -> list[str]:
        """Current workspace names for display; a workspace that's gone or unreadable is shown by its ID."""
        names = []
        for workspace_id in workspace_ids:
            try:
                name = await self._workspace_name(workspace_id)
            except HTTPException:
                name = None
            names.append(name or workspace_id)
        return names

    async def _with_bound_names(self, key: _MetadataT) -> _MetadataT:
        """``key`` with its bound workspaces reported by current name (see _display_names)."""
        if not key.bound_workspace_ids:
            return key
        return key.model_copy(update={"bound_workspaces": await self._display_names(key.bound_workspace_ids)})

    async def _resolve_bound_workspaces(self, names: Sequence[str]) -> _BoundWorkspaces:
        """The current ID of each named workspace. Fails closed: a missing or unreadable one can't be bound.

        A transient lookup failure surfaces as a 503 (see resolve_workspace_id), not a validation error.
        """
        workspace_ids = []
        for name in names:
            workspace_id = await self._workspace_id(name)
            if workspace_id is None:
                raise AccessKeyValidationError(f"Workspace '{name}' could not be resolved")
            workspace_ids.append(workspace_id)
        return _BoundWorkspaces(ids=tuple(workspace_ids), names=tuple(names))

    async def _bound_workspaces_of(self, record: AccessKeyEntity) -> _BoundWorkspaces:
        """A stored key's bound workspaces with their current names (a workspace that's gone keeps its ID).

        For a caller who isn't a HelixAdmin the names feed the confinement check, which compares
        role bindings by workspace name, so a failed lookup is raised (as a 503) rather than
        replaced by the ID, which would make a binding in the bound workspace look like one
        outside it. A HelixAdmin skips that check, so the names are only for display.
        """
        ids = tuple(record.bound_workspace_ids)
        if await self.is_platform_admin():
            return _BoundWorkspaces(ids=ids, names=tuple(await self._display_names(ids)))
        names = [await self._workspace_name(workspace_id) or workspace_id for workspace_id in ids]
        return _BoundWorkspaces(ids=ids, names=tuple(names))

    async def _require_enabled_service_account(self, service_account_id: str | None) -> None:
        """Reject a disabled service account up front.

        Runs before the checks that inspect the account's keys: a disabled account's keys can't
        be listed, and saving a new key for it would fail anyway.
        """
        if await self._registry.is_service_account_disabled(f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{service_account_id}"):
            raise AccessKeyValidationError(
                f"Service account '{service_account_id}' is disabled; enable it before creating keys for it"
            )

    async def _require_namespaced_service_account(
        self, bound: _BoundWorkspaces, service_account_id: str | None
    ) -> None:
        """Make a workspace Admin name the service account '<workspace>/<name>' after a bound workspace.

        Service-account IDs are global and roles belong to the account, so without a namespace a
        workspace Admin could claim a generic name (say, one a HelixAdmin later provisions with
        broad access) and their key would silently gain whatever access anyone grants that name.
        HelixAdmins are unrestricted. Applies to create and rotate alike.
        """
        if not bound or await self.is_platform_admin():
            return
        account_id = service_account_id or ""
        if not any(account_id.startswith(f"{name}/") and len(account_id) > len(name) + 1 for name in bound.names):
            raise AccessKeyValidationError(_NAMESPACE_MESSAGE.format(workspaces=_format_workspaces(bound.names)))

    async def _require_confined_service_account(self, bound: _BoundWorkspaces, service_account_id: str | None) -> None:
        """Confine a workspace Admin to service accounts that have no access beyond the bound workspaces.

        The JWT authenticates as service-account:<id> everywhere that principal has access, so a
        workspace Admin may only use an account with no key or role binding outside the bound
        workspaces (e.g. one a HelixAdmin provisioned with access to other workspaces), nor a key
        bound to an earlier workspace of the same name, whose live key a recreated workspace must
        not inherit access for. HelixAdmins are unrestricted. Applies to create and rotate alike.
        """
        if await self._confinement_violated(bound, service_account_id):
            raise AccessKeyForbiddenError(
                _OUTSIDE_ACCESS_MESSAGE.format(
                    account_id=service_account_id, workspaces=_format_workspaces(bound.names)
                )
            )

    async def _confinement_violated(self, bound: _BoundWorkspaces, service_account_id: str | None) -> bool:
        """Whether a workspace Admin's service account has access outside ``bound``.

        Run before the new key's record is written, and again after: a key or role binding that a
        HelixAdmin (or another workspace's Admin) adds for the same service account in between
        would slip past the first check, and the second closes that window for whichever request
        commits second. Always False for HelixAdmins and unbound keys.
        """
        if not bound or service_account_id is None or await self.is_platform_admin():
            return False
        return await self._registry.has_service_account_access_outside_workspaces(
            f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{service_account_id}",
            workspace_ids=bound.ids,
            workspace_names=bound.names,
        )

    async def _granted_workspace_ids(self, request: AccessKeyCreateRequest, bound: _BoundWorkspaces) -> list[str]:
        """IDs of every workspace the new key grants its account roles in.

        A `workspaces` grant naming a workspace that can't be resolved keeps a placeholder that no
        key is bound to, so it still counts as reaching outside every key (the grant itself then
        fails anyway).
        """
        granted = list(bound.ids)
        for grant in request.workspaces or []:
            workspace_id = await self._workspace_id(grant.workspace)
            granted.append(workspace_id or f"unresolved:{grant.workspace}")
        return granted

    async def _require_acknowledged_widening(
        self, granted_workspace_ids: Sequence[str], service_account_id: str | None, *, acknowledged: bool
    ) -> None:
        """Make the caller confirm a key whose grants reach beyond the account's workspace-bound keys.

        Roles belong to the service account, so keys bound to workspaces (and managed by those
        workspaces' Admins) silently gain any role the new key grants elsewhere. Applies to
        HelixAdmins and workspace Admins alike; the confinement check already stops a workspace
        Admin from reaching outside their workspaces. Roles granted directly through workspace
        membership, rather than with a key, aren't covered.
        """
        widened = await self._registry.workspace_bound_keys_widened_by(
            f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{service_account_id}", granted_workspace_ids
        )
        if widened is None:
            return
        names = sorted(await self._display_names(widened))
        if not acknowledged:
            raise AccessKeyWorkspaceBoundKeysError(
                f"Service account '{service_account_id}' already has live keys bound to workspace(s) "
                f"{_format_workspaces(names)}, which those workspaces' Admins manage. Roles belong to the "
                "service account, so those keys will also get the roles this key grants outside those workspaces."
            )
        logger.warning(
            "Creating a service-bound Scoped Access Key that widens the account's workspace-bound keys",
            extra={
                "audit_event": "access_key.workspace_bound_keys_widened",
                "actor_principal": self.principal,
                "service_account_id": service_account_id,
                "workspaces": names,
                "workspace_ids": widened,
            },
        )

    async def _bound_workspaces_unchanged(self, bound: _BoundWorkspaces) -> bool:
        """Whether every bound workspace's name still maps to the ID the key was bound to."""
        for workspace_id, name in zip(bound.ids, bound.names, strict=True):
            if await resolve_workspace_id(self._workspaces_client, name) != workspace_id:
                return False
        return True

    async def _discard_unreturned_key(self, jti: str, failure_reason: str) -> None:
        """Delete the record of a key whose JWT was never returned, or revoke it if that fails.

        Deleting rather than revoking leaves no trace for the confinement check to trip over: a
        revoked key outside a workspace would otherwise lock that workspace out of the account.
        """
        try:
            await self._registry.discard_unreturned(jti)
        except Exception:
            logger.warning(
                "Failed to discard an unreturned Scoped Access Key; revoking it instead",
                extra={"access_key_jti": jti, "actor_principal": self.principal, "failure_reason": failure_reason},
                exc_info=True,
            )
            await self._revoke_after_failed_creation(jti, failure_reason)

    def _enforce_caller_scope(self, request: AccessKeyCreateRequest) -> None:
        """Reject a scope-restricted caller minting a key broader than its own access."""
        if self._caller_scope is None:
            return
        if (request.workspaces or request.bound_workspaces) and not {"entities", "platform"} & self._caller_scope:
            raise AccessKeyValidationError(
                "A Scoped Access Key restricted via --scope must include entities or platform "
                "in its own scope to grant workspace memberships"
            )
        requested = set(request.scope or [])
        if request.scope is None or not requested <= self._caller_scope:
            raise AccessKeyValidationError(
                "A Scoped Access Key restricted via --scope can only create replacement keys "
                f"scoped to a subset of its own scope ({', '.join(sorted(self._caller_scope))})"
            )

    async def create_async(
        self, request: AccessKeyCreateRequest, *, allow_service_account: bool = False
    ) -> AccessKeyCreateResponse:
        self._ensure_enabled()
        if request.service_account_id is not None and request.rotates is not None:
            # `rotates` only applies to personal keys; a service-bound replacement would let
            # rotation quietly change the credential's identity type.
            raise AccessKeyValidationError(
                "rotates cannot be combined with service_account_id; rotation is only supported for personal keys"
            )
        bound = _BoundWorkspaces()
        if allow_service_account:
            # Defense-in-depth, mirroring the admin_override re-check that revoke/suspend/
            # unsuspend apply for service-bound keys: don't rely solely on the caller having
            # verified admin status before setting this flag (see AdminOverride).
            requested_names = [workspace.workspace for workspace in request.bound_workspaces or []]
            if not requested_names and not await self.is_platform_admin():
                raise AccessKeyForbiddenError(_ADMIN_REQUIRED_MESSAGE)
            bound = await self._resolve_bound_workspaces(requested_names)
            # Checked against the IDs just read, so the key is bound to the workspaces whose Admin
            # check passed, even if one was recreated under the same name in between.
            if not await self.can_administer(bound.ids):
                raise AccessKeyForbiddenError(_ADMIN_REQUIRED_MESSAGE)
            await self._require_enabled_service_account(request.service_account_id)
            await self._require_namespaced_service_account(bound, request.service_account_id)
            await self._require_confined_service_account(bound, request.service_account_id)
        if request.rotates is not None:
            # Fail fast on an invalid rotation target instead of minting an orphan key.
            old_record = await self._registry.get_for_rotation(
                request.rotates, self.principal, admin_override=self._admin_override
            )
            if request.scope is None:
                # No --scope given: inherit the predecessor's scope instead of widening to unscoped.
                request = request.model_copy(update={"scope": old_record.scope or None})
            elif set(request.scope) != set(old_record.scope or []):
                # `rotates` must preserve the predecessor's scope exactly; callers who want a
                # different scope should create + revoke separately instead.
                raise AccessKeyValidationError(
                    "rotates requires the replacement scope to match the predecessor's scope "
                    f"({', '.join(sorted(old_record.scope)) if old_record.scope else 'unscoped'}); "
                    "omit --scope to inherit it automatically, or create a separate key instead"
                )
        self._enforce_caller_scope(request)
        if allow_service_account:
            # Last before minting, so a request that would fail validation anyway isn't first
            # sent back for confirmation.
            await self._require_acknowledged_widening(
                await self._granted_workspace_ids(request, bound),
                request.service_account_id,
                acknowledged=request.acknowledge_workspace_bound_keys,
            )
        key = await self._issuer.create_async(request, allow_service_account=allow_service_account)
        key = key.model_copy(update={"bound_workspaces": list(bound.names), "bound_workspace_ids": list(bound.ids)})
        try:
            await self._registry.add(
                key,
                owner_principal=self.principal,
                owner_account_id=self.account_id,
                bound_workspace_ids=bound.ids,
            )
        except Exception:
            logger.warning(
                "Failed to persist Scoped Access Key lifecycle record; the signed JWT will not be returned to the caller",
                extra={"access_key_jti": key.jti, "actor_principal": self.principal},
                exc_info=True,
            )
            raise
        if await self._confinement_violated(bound, request.service_account_id):
            # Discarded rather than revoked: two Admins racing for the same account would otherwise
            # each leave a revoked key that locks the other's workspace out of the account for good.
            await self._discard_unreturned_key(key.jti, "service_account_confinement_violated")
            raise AccessKeyForbiddenError(
                _OUTSIDE_ACCESS_MESSAGE.format(
                    account_id=request.service_account_id, workspaces=_format_workspaces(bound.names)
                )
            )
        logger.info(
            "Scoped Access Key created",
            extra={
                "audit_event": "access_key.created",
                "actor_principal": self.principal,
                "access_key_jti": key.jti,
            },
        )
        # Grant requested access before revoking the predecessor, so the key is never left
        # with less access than it had before rotation started.
        try:
            # A bound key's grants are its bound workspaces' roles; `workspaces` is empty then.
            attempted_grants = await self._grant_workspace_memberships(
                key, [*(request.workspaces or []), *(request.bound_workspaces or [])]
            )
        except Exception:
            # Don't return a 200 for a key that's missing the access it was supposed to get.
            await self._revoke_after_failed_creation(key.jti, "workspace_grant_failed")
            raise
        # Memberships are granted by workspace name. If a bound workspace was recreated under the
        # same name since its ID was read, the roles just landed in the new workspace while the key
        # is bound to the old one; undo them rather than hand out access the Admin check never covered.
        if bound and not await self._bound_workspaces_unchanged(bound):
            await self._rollback_workspace_memberships(key, attempted_grants)
            await self._discard_unreturned_key(key.jti, "bound_workspace_replaced")
            raise AccessKeyForbiddenError(
                f"Workspace(s) {_format_workspaces(bound.names)} changed while the key was being created; retry"
            )
        if request.rotates is not None:
            try:
                await self.revoke_async(request.rotates)
            except Exception:
                logger.warning(
                    "Failed to revoke the prior Scoped Access Key after rotation",
                    extra={
                        "audit_event": "access_key.rotation_revoke_failed",
                        "actor_principal": self.principal,
                        "access_key_jti": key.jti,
                        "rotated_from_jti": request.rotates,
                    },
                    exc_info=True,
                )
                # The old key is still active, so compensate by revoking the new key and
                # undoing its workspace grants, leaving the untouched predecessor as the
                # caller's one valid credential to retry rotation with.
                await self._rollback_workspace_memberships(key, attempted_grants)
                await self._revoke_after_failed_creation(key.jti, "rotation_revoke_failed")
                raise
        return key

    async def _revoke_after_failed_creation(self, jti: str, failure_reason: str) -> None:
        try:
            await self.revoke_async(jti)
        except Exception:
            logger.error(
                "Failed to compensate for a failed Scoped Access Key creation step by revoking the new key",
                extra={
                    "audit_event": "access_key.compensating_revoke_failed",
                    "actor_principal": self.principal,
                    "access_key_jti": jti,
                    "failure_reason": failure_reason,
                },
                exc_info=True,
            )

    async def _grant_workspace_memberships(
        self,
        key: AccessKeyCreateResponse,
        workspaces: list[AccessKeyWorkspaceGrant] | None,
    ) -> list[tuple[AccessKeyWorkspaceGrant, list[str] | None]]:
        # Record each attempted grant so a partial failure can be undone precisely; returned
        # to the caller since other failures later in create_async need to undo these too.
        attempted: list[tuple[AccessKeyWorkspaceGrant, list[str] | None]] = []
        for grant in workspaces or []:
            try:
                prior_roles = await self._get_member_roles(grant.workspace, key.principal)
                attempted.append((grant, prior_roles))
                await self._workspaces_client.create_workspace_member(
                    workspace=grant.workspace,
                    body=CreateWorkspaceMemberRequest(
                        principal=key.principal,
                        roles=grant.roles,
                    ),
                )
            except Exception:
                logger.warning(
                    "Failed to grant workspace membership to a new Scoped Access Key",
                    extra={
                        "audit_event": "access_key.workspace_grant_failed",
                        "actor_principal": self.principal,
                        "access_key_jti": key.jti,
                        "workspace": grant.workspace,
                        "roles": grant.roles,
                    },
                    exc_info=True,
                )
                await self._rollback_workspace_memberships(key, attempted)
                raise
        return attempted

    async def _get_member_roles(self, workspace: str, principal: str) -> list[str] | None:
        """Return a principal's current roles in workspace, or None if not yet a member.

        Fails closed: an unreadable workspace can't be distinguished from "no prior
        membership" later, so don't grant membership for it — a rollback could then wrongly
        delete a real pre-existing grant.
        """
        try:
            members = (await self._workspaces_client.list_workspace_members(workspace=workspace)).data()
        except Exception:
            logger.warning(
                "Failed to read prior workspace membership before granting access; "
                "refusing to grant membership for this workspace since a later rollback "
                "could not distinguish no prior membership from an unreadable one",
                extra={"workspace": workspace, "principal": principal},
                exc_info=True,
            )
            raise
        return next((list(member.roles) for member in members.data if member.principal == principal), None)

    async def _rollback_workspace_memberships(
        self,
        key: AccessKeyCreateResponse,
        attempted: list[tuple[AccessKeyWorkspaceGrant, list[str] | None]],
    ) -> None:
        for grant, prior_roles in reversed(attempted):
            try:
                if prior_roles is None:
                    await self._workspaces_client.delete_workspace_member(
                        workspace=grant.workspace, principal_id=key.principal
                    )
                else:
                    await self._workspaces_client.update_workspace_member(
                        workspace=grant.workspace,
                        principal_id=key.principal,
                        body=UpdateWorkspaceMemberRequest(roles=prior_roles),
                    )
            except Exception:
                logger.error(
                    "Failed to roll back a workspace membership granted during a failed "
                    "Scoped Access Key creation; the principal may retain unintended access",
                    extra={
                        "audit_event": "access_key.workspace_grant_rollback_failed",
                        "actor_principal": self.principal,
                        "access_key_jti": key.jti,
                        "workspace": grant.workspace,
                    },
                    exc_info=True,
                )

    def _listing_admin_override(self) -> AdminOverride | None:
        """The admin_override for listing: a failed check hides that service-bound key.

        Listing must not fail outright because one workspace's role or ID lookup did. A key whose
        check failed is treated as not administered, and not asked about again in this listing.
        """
        admin_override = self._admin_override
        if admin_override is None:
            return None
        failed: set[tuple[str, ...]] = set()

        async def check(workspace_ids: Sequence[str]) -> bool:
            key = tuple(workspace_ids)
            if key in failed:
                return False
            try:
                return await admin_override(workspace_ids)
            # The PDP and workspace lookups report outages as HTTPException; anything else, such as
            # a missing PDP URL, is a misconfiguration that should still fail the request.
            except HTTPException:
                logger.warning(
                    "Admin check failed while listing Scoped Access Keys; "
                    "hiding the service-bound keys bound to these workspaces from the list",
                    extra={"actor_principal": self.principal, "workspace_ids": list(workspace_ids)},
                    exc_info=True,
                )
                failed.add(key)
                return False

        return check

    async def _visible_workspace_ids(self) -> list[str]:
        """IDs of the workspaces visible to the caller, recording their names for this request.

        Lets listing include the service-bound keys other Admins created for workspaces the caller
        administers. No role is checked here: the per-key admin_override does that, and only for
        workspaces a listed key is bound to, so a caller pays nothing for workspaces without bound
        keys. Only runs for callers who aren't HelixAdmins. If the workspaces can't be listed,
        returns none: listing then shows just the caller's own keys.
        """
        visible = []
        try:
            page = await self._workspaces_client.list_workspaces(query_params={"page_size": 100})
            async for workspace in page.items():
                self._workspace_names[workspace.id] = workspace.name
                visible.append(workspace.id)
        except NemoClientError:
            logger.warning(
                "Could not list the caller's workspaces while listing Scoped Access Keys; "
                "showing only the service-bound keys the caller created",
                extra={"actor_principal": self.principal},
                exc_info=True,
            )
            return []
        return visible

    async def list_async(self, *, page: int = 1, page_size: int = 100) -> AccessKeyListResponse:
        self._ensure_enabled()
        # A current HelixAdmin sees every service-bound key, while a workspace Admin also sees
        # the service-bound keys bound to workspaces they administer, mirroring
        # the admin_override check lifecycle operations already apply (see
        # AccessKeyRegistry.list_for_principal). If the PDP role check itself fails (e.g.
        # unreachable), degrade to the caller's own keys rather than failing listing outright
        # for every user. Only that check is guarded: a failure listing the keys themselves
        # must still surface.
        try:
            include_service_accounts = await self.is_platform_admin()
        except HTTPException:
            logger.warning(
                "HelixAdmin check failed while listing Scoped Access Keys; "
                "falling back to the caller's own personal keys (service-bound keys are hidden from this list)",
                extra={"actor_principal": self.principal},
                exc_info=True,
            )
            return await self._registry.list_for_principal(
                self.principal,
                page=page,
                page_size=page_size,
                include_service_accounts=False,
                owner_account_id=self.account_id,
            )
        admin_override = self._listing_admin_override()
        service_key_workspace_ids = (
            await self._visible_workspace_ids() if admin_override is not None and not include_service_accounts else []
        )
        listed = await self._registry.list_for_principal(
            self.principal,
            page=page,
            page_size=page_size,
            include_service_accounts=include_service_accounts,
            admin_override=admin_override,
            service_key_workspace_ids=service_key_workspace_ids,
            owner_account_id=self.account_id,
        )
        return listed.model_copy(update={"data": [await self._with_bound_names(key) for key in listed.data]})

    async def revoke_async(self, jti: str) -> bool:
        self._ensure_enabled()
        revoked = await self._registry.revoke(jti, self.principal, admin_override=self._admin_override)
        audit_event = "access_key.revoked" if revoked else "access_key.revoke_noop"
        logger.info(
            "Scoped Access Key revoked" if revoked else "Scoped Access Key revoke requested for already-revoked key",
            extra={
                "audit_event": audit_event,
                "actor_principal": self.principal,
                "access_key_jti": jti,
                "access_key_already_revoked": not revoked,
            },
        )
        return revoked

    async def suspend_async(self, jti: str) -> tuple[bool, AccessKeyReversibleStatus]:
        self._ensure_enabled()
        suspended, effective_status = await self._registry.suspend(
            jti, self.principal, admin_override=self._admin_override
        )
        self._log_suspension(jti, changed=suspended, action="suspend")
        return suspended, effective_status

    async def unsuspend_async(self, jti: str) -> tuple[bool, AccessKeyReversibleStatus]:
        self._ensure_enabled()
        unsuspended, effective_status = await self._registry.unsuspend(
            jti, self.principal, admin_override=self._admin_override
        )
        self._log_suspension(jti, changed=unsuspended, action="unsuspend")
        return unsuspended, effective_status

    async def _discard_orphaned_successor(self, new_key_jti: str, *, context: str) -> None:
        try:
            await self._registry.discard_unreturned(new_key_jti)
        except Exception:
            logger.warning(
                f"Failed to discard {context}",
                extra={"access_key_jti": new_key_jti, "actor_principal": self.principal},
                exc_info=True,
            )

    async def rotate_async(self, jti: str, *, grace_period_seconds: int | None = None) -> AccessKeyRotateResponse:
        self._ensure_enabled()
        max_grace_period_seconds = self._config.access_keys.max_rotation_grace_period_seconds
        if grace_period_seconds is not None and (
            max_grace_period_seconds is not None and grace_period_seconds > max_grace_period_seconds
        ):
            raise AccessKeyValidationError(
                f"grace_period_seconds ({grace_period_seconds}) exceeds "
                f"auth.access_keys.max_rotation_grace_period_seconds ({max_grace_period_seconds})"
            )
        # Read-validate before minting: fail fast on an ineligible key (already
        # revoked/suspended/rotating/expired) instead of issuing a successor JWT we'd
        # then have to discard.
        old_record = await self._registry.get_rotatable(jti, self.principal, admin_override=self._admin_override)
        allow_service_account = old_record.entity_type == "SERVICE_ACCOUNT"
        if allow_service_account:
            # Mirrors create_async's defense-in-depth re-check: don't rely solely on
            # the caller having verified admin status upstream (see AdminOverride).
            if not await self.can_administer(old_record.bound_workspace_ids):
                raise AccessKeyForbiddenError(_ADMIN_REQUIRED_MESSAGE)
            service_account_id = (old_record.subject_principal or "").removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX)
            await self._require_enabled_service_account(service_account_id)
            bound = await self._bound_workspaces_of(old_record)
            await self._require_namespaced_service_account(bound, service_account_id)
            await self._require_confined_service_account(bound, service_account_id)
        else:
            service_account_id = None
            bound = _BoundWorkspaces()
        # Preserve the original key's lifetime characteristic (finite duration, restarted
        # from now, or explicitly non-expiring) rather than the caller's current default.
        expires_in_seconds = (
            int((old_record.expires_at - old_record.issued_at).total_seconds())
            if old_record.expires_at is not None
            else None
        )
        # The preserved lifetime (or non-expiring None) can violate a max_expires_in_seconds
        # policy that has since tightened relative to when the old key was issued. Clamp it
        # to the current maximum rather than let an otherwise-eligible rotation fail outright.
        max_expires_in_seconds = self._config.access_keys.max_expires_in_seconds
        if max_expires_in_seconds is not None and (
            expires_in_seconds is None or expires_in_seconds > max_expires_in_seconds
        ):
            expires_in_seconds = max_expires_in_seconds
        request = AccessKeyCreateRequest(
            name=old_record.key_name,
            description=old_record.description,
            service_account_id=service_account_id,
            # Keep the original scope; map empty (unscoped) to None since an explicit
            # empty scope is rejected.
            scope=old_record.scope or None,
            expires_in_seconds=expires_in_seconds,
        )
        # get_rotatable only checks ownership, not scope — enforce that separately.
        self._enforce_caller_scope(request)
        new_key = await self._issuer.create_async(request, allow_service_account=allow_service_account)
        # Rotation re-issues the key only, so the request names no bound workspaces (their roles are
        # already in place); report them on the successor from the record it inherits.
        new_key = new_key.model_copy(
            update={"bound_workspaces": list(bound.names), "bound_workspace_ids": list(bound.ids)}
        )
        try:
            await self._registry.add(
                new_key,
                owner_principal=self.principal,
                owner_account_id=self.account_id,
                bound_workspace_ids=bound.ids,
            )
        except Exception:
            try:
                await self._registry.get_status(
                    new_key.jti,
                    self.principal,
                    admin_override=self._admin_override,
                )
            except AccessKeyNotFoundError:
                pass
            except Exception:
                logger.warning(
                    "Could not reconcile whether the rotated Scoped Access Key lifecycle record was persisted",
                    extra={"access_key_jti": new_key.jti, "actor_principal": self.principal},
                    exc_info=True,
                )
            else:
                await self._discard_orphaned_successor(
                    new_key.jti,
                    context="unreturned successor Scoped Access Key after an ambiguous persistence failure",
                )
            logger.warning(
                "Failed to persist rotated Scoped Access Key lifecycle record; "
                "the signed JWT will not be returned to the caller",
                extra={"access_key_jti": new_key.jti, "actor_principal": self.principal},
                exc_info=True,
            )
            raise
        if await self._confinement_violated(bound, service_account_id):
            await self._discard_orphaned_successor(
                new_key.jti, context="successor Scoped Access Key rejected by the post-insert confinement check"
            )
            raise AccessKeyForbiddenError(
                _OUTSIDE_ACCESS_MESSAGE.format(
                    account_id=service_account_id, workspaces=_format_workspaces(bound.names)
                )
            )
        if grace_period_seconds is None:
            grace_period_seconds = self._config.access_keys.rotation_grace_period_seconds
        try:
            rotated_record = await self._registry.begin_rotation(
                jti,
                self.principal,
                grace_period_seconds=grace_period_seconds,
                successor_jti=new_key.jti,
                admin_override=self._admin_override,
            )
        except (AccessKeyStateConflictError, EntityConflictError, AccessKeyNotFoundError):
            await self._discard_orphaned_successor(
                new_key.jti, context="orphaned successor Scoped Access Key after a failed rotation"
            )
            raise
        except Exception as rotation_error:
            if not getattr(rotation_error, "write_attempted", True):
                # begin_rotation identified a deterministic pre-write failure. A
                # concurrent winner may still make the old key look ROTATING, but
                # this request's successor was never paired with that transition.
                await self._discard_orphaned_successor(
                    new_key.jti, context="orphaned successor Scoped Access Key after a failed rotation"
                )
                raise
            try:
                reconciled_record = await self._registry.get_status(
                    jti,
                    self.principal,
                    admin_override=self._admin_override,
                )
            except Exception:
                logger.warning(
                    "Could not reconcile Scoped Access Key state after rotation failed; keeping the successor",
                    extra={
                        "access_key_jti": jti,
                        "access_key_new_jti": new_key.jti,
                        "actor_principal": self.principal,
                    },
                    exc_info=True,
                )
                raise rotation_error from None
            if reconciled_record.rotation_successor_jti == new_key.jti:
                # Confirmed: this request's write committed, however the record now
                # reads. Normally that's still ROTATING, but if reconciliation was
                # itself delayed (e.g. past the grace deadline, or a subsequent
                # manual revoke landed first), _status() may already report
                # REVOKED/EXPIRED -- report that faithfully rather than discarding
                # a successor this request is confirmed to own.
                rotated_record = reconciled_record
            else:
                # This request's own transition never took effect: the old key is
                # either still ACTIVE (the write never committed), or it moved on
                # paired with a different successor (a concurrent request's
                # rotation committed instead) -- either way, this request's
                # successor must be discarded rather than misattributed as a
                # success it didn't cause.
                await self._discard_orphaned_successor(
                    new_key.jti, context="orphaned successor Scoped Access Key after a failed rotation"
                )
                raise rotation_error from None
        previous_status = AccessKeyRegistry._status(rotated_record)
        effective_grace_period_expires_at = (
            rotated_record.grace_period_expires_at if previous_status == "ROTATING" else None
        )
        logger.info(
            "Scoped Access Key rotated",
            extra={
                "audit_event": "access_key.rotated",
                "actor_principal": self.principal,
                "access_key_jti": jti,
                "access_key_new_jti": new_key.jti,
            },
        )
        # rotated_record.grace_period_expires_at is already capped by the old key's
        # natural expiry (see begin_rotation), so derive the reported remaining
        # seconds from it rather than echoing the configured grace_period_seconds
        # verbatim -- otherwise a key expiring sooner than the configured grace
        # period would be reported as remaining usable far longer than it actually is.
        effective_grace_period_seconds = 0
        if effective_grace_period_expires_at is not None:
            effective_grace_period_seconds = max(
                0, int((effective_grace_period_expires_at - datetime.now(tz=UTC)).total_seconds())
            )
        return AccessKeyRotateResponse(
            new_key=new_key,
            previous_jti=jti,
            previous_status=previous_status,
            grace_period_seconds=effective_grace_period_seconds,
            grace_period_expires_at=effective_grace_period_expires_at,
        )

    def _log_suspension(
        self,
        jti: str,
        *,
        changed: bool,
        action: Literal["suspend", "unsuspend"],
    ) -> None:
        completed_action = "suspended" if action == "suspend" else "unsuspended"
        logger.info(
            f"Scoped Access Key {completed_action}"
            if changed
            else f"Scoped Access Key {action} requested with no change",
            extra={
                "audit_event": f"access_key.{action}ed" if changed else f"access_key.{action}_noop",
                "actor_principal": self.principal,
                "access_key_jti": jti,
                "access_key_state_changed": changed,
            },
        )

    def _ensure_enabled(self) -> None:
        if not self._config.access_keys.enabled:
            raise AccessKeyFeatureDisabledError("Scoped Access Keys are not enabled")
