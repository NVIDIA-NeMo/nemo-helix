# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence and issuance services for Scoped Access Keys."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeVar

from fastapi import Depends, HTTPException
from nemo_helix_plugin.auth.access_keys.issuer import AccessKeyFeatureDisabledError
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
from nemo_helix_plugin.workspaces.types import (
    CreateWorkspaceMemberRequest,
    UpdateWorkspaceMemberRequest,
    Workspace,
)
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
from nhx.common.entities import (
    ALL_WORKSPACES,
    EntityClient,
    EntityConflictError,
    EntityNotFoundError,
    EntityStoreError,
)
from nhx.common.service.dependencies import get_entity_client
from nhx.core.auth.app.access_key_credentials import AccessKeyCredentialAdapter
from nhx.core.auth.app.account_resolution import get_account_session_maker
from nhx.core.auth.entities import AccessKeyEntity, RoleBindingEntity
from nhx.core.entities.app.repository import (
    AccountCredentialConflictError,
    AccountCredentialStore,
    AccountIdentityRecord,
    AccountIdentityStore,
    AccountIdentityUnavailableError,
)

ACCESS_KEY_WORKSPACE = "system"
logger = logging.getLogger(__name__)

# Bounds retries of a lifecycle mutation's optimistic-lock update against version
# bumps from concurrent, unrelated writes (chiefly is_active's last_used_at updates
# on every authenticated request) that don't actually conflict with the mutation
# itself. A key with real traffic is exactly the case these operations exist to serve.
_MUTATION_MAX_ATTEMPTS = 3

# Coarsens last_used_at writes in is_active: a key with active traffic doesn't need a
# store write on every single authenticated request, just often enough to tell a caller
# their traffic has moved off a rotated-out key.
_LAST_USED_AT_RESOLUTION = timedelta(minutes=5)

# Workspaces read in one batched lookup, and batches read at once. Bounds the query string of a
# single lookup and the concurrent load of resolving many workspaces' names while listing.
_WORKSPACE_NAME_BATCH_SIZE = 50
_MAX_CONCURRENT_WORKSPACE_READS = 8

# Workspaces a caller's Admin role bindings are scanned for when listing. A caller who administers
# more than this sees the keys bound to the first _MAX_ADMINISTERED_WORKSPACES of them, by ID; the
# filter is serialized into a query string, so an unbounded list would not survive the round trip.
_MAX_ADMINISTERED_WORKSPACES = 200

# Entities read per page when scanning role bindings.
_SCAN_PAGE_SIZE = 1000

# Principals whose role bindings are read in one query. A caller can have many identifiers (every
# IdP group they belong to), and the filter goes in the entity store's query string.
_ROLE_BINDING_PRINCIPAL_BATCH_SIZE = 50

_MutationResultT = TypeVar("_MutationResultT")
_MetadataT = TypeVar("_MetadataT", bound=AccessKeyMetadataResponse)
_ScannedT = TypeVar("_ScannedT", bound=AccessKeyEntity | RoleBindingEntity)

_OUTSIDE_ACCESS_MESSAGE = (
    "Service account '{account_id}' has Scoped Access Keys or roles outside workspace '{workspace}', so only a "
    "HelixAdmin can create, rotate or unsuspend its keys"
)
ADMIN_REQUIRED_MESSAGE = (
    "Service-bound Scoped Access Keys require HelixAdmin, or Admin of the single workspace the key binds to"
)
ONE_WORKSPACE_MESSAGE = (
    "A service-bound Scoped Access Key binds to one workspace: name exactly one in workspaces, and be its Admin, "
    "or ask a HelixAdmin to create the key"
)


class _Retry:
    """Sentinel a `_retry_on_conflict` conflict handler returns to request another attempt."""


_RETRY = _Retry()

# Service-bound keys are machine-to-machine credentials owned by the platform, not by the
# creating individual. Every lifecycle operation therefore uses this callback to authorize the
# caller against the key's bound workspace, including when that caller created the key. It takes
# the key's bound workspace ID, or None for a key bound to none, and answers whether the caller
# may manage it: a current HelixAdmin always may, an Admin of the bound workspace may. The
# implementation memoizes per workspace for the request (see _service_key_admin_check), so asking
# once per key costs one PDP round trip per distinct workspace.
AdminOverride = Callable[[str | None], Awaitable[bool]]


def _outside_access_error(service_account_id: str | None, workspace_name: str) -> AccessKeyForbiddenError:
    return AccessKeyForbiddenError(
        _OUTSIDE_ACCESS_MESSAGE.format(account_id=service_account_id, workspace=workspace_name)
    )


def _service_account_disabled(service_account_id: str | None, action: str) -> AccessKeyValidationError:
    return AccessKeyValidationError(f"Service account '{service_account_id}' is disabled; enable it before {action}")


class UnpageableScanError(RuntimeError):
    """An entity-store scan met an entity without an ID, so it can't read past it (see AccessKeyRegistry._scan)."""


@dataclass(frozen=True)
class LiveServiceAccountKey:
    """A key for a service account that can still authenticate, suspended ones included (they can be unsuspended)."""

    jti: str
    bound_workspace_id: str | None


@dataclass(frozen=True)
class ServiceAccountAccess:
    """What a service account can reach, as read in one pass (see AccessKeyRegistry.service_account_access).

    Its live keys, from both stores, and the roles it holds through active role bindings keyed by
    any of ``principals`` (the identifiers the PDP matches its keys by), by workspace name.
    """

    principals: tuple[str, ...]
    live_keys: tuple[LiveServiceAccountKey, ...] = ()
    roles: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def reaches_outside(self, workspace_id: str, workspace_name: str) -> bool:
        """Whether it has a live key or a role binding outside this one workspace.

        A key is outside unless it is bound, and to this workspace's ID: deleting a workspace
        doesn't revoke its keys, so a recreated workspace of the same name must not take over an
        account whose old key is still live. Role bindings catch access granted without a key, e.g.
        membership a HelixAdmin or another workspace's Admin pre-provisioned for the account.
        They're stored by workspace name, so they're compared against ``workspace_name`` (the
        current name of ``workspace_id``): deleting a workspace deletes its bindings, so none can
        outlive the workspace.
        """
        return any(key.bound_workspace_id != workspace_id for key in self.live_keys) or bool(
            set(self.roles) - {workspace_name}
        )


class AccessKeyForbiddenError(AccessKeyValidationError):
    """Raised when the caller may not create or manage a key, rather than having asked for an invalid one.

    A subclass of AccessKeyValidationError so existing handling still catches it; endpoints map it
    to 403 instead of 400.
    """


def _workspace_lookup_failed(workspace: str) -> HTTPException:
    # Callers treat a None lookup result as definitive, so a transient outage (transport error,
    # 5xx) must surface as a 503 instead of a spurious 403/404 or a client-error 400.
    logger.warning("Failed to read workspace '%s'", workspace, exc_info=True)
    return HTTPException(status_code=503, detail=f"Workspace '{workspace}' could not be read; retry")


async def resolve_workspace(workspaces_client: AsyncWorkspacesClient, workspace: str) -> Workspace | None:
    """The workspace named ``workspace``, or None if it doesn't exist or the caller can't read it.

    Any other client failure is raised as a 503 (see _workspace_lookup_failed).
    """
    try:
        return (await workspaces_client.get_workspace(name=workspace)).data()
    except (NotFoundError, PermissionDeniedError):
        return None
    except NemoClientError as exc:
        raise _workspace_lookup_failed(workspace) from exc


async def resolve_workspace_id(workspaces_client: AsyncWorkspacesClient, workspace: str) -> str | None:
    """The workspace's current ID, or None if it doesn't exist or the caller can't read it."""
    found = await resolve_workspace(workspaces_client, workspace)
    return found.id if found is not None else None


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
        raise _workspace_lookup_failed(workspace_id) from exc
    return None


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
        bound_workspace_id: str | None = None,
    ) -> None:
        owner = owner_principal or key.principal
        record = AccessKeyEntity(
            name=key.jti,
            workspace=ACCESS_KEY_WORKSPACE,
            key_name=key.name,
            description=key.description,
            principal=owner,
            subject_principal=key.principal if key.principal != owner else None,
            bound_workspace_id=bound_workspace_id,
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
        # Service-bound keys belong to the workspace they're bound to, or to the platform when they
        # are bound to none: whoever can manage one via _get_owned's admin_override must also see
        # it, rather than only the person who happened to create it. So a HelixAdmin
        # (include_service_accounts) sees every service key, and a workspace Admin sees the ones
        # bound to a workspace they administer (service_key_workspace_ids).
        # EntityBase fields (all fields on AccessKeyEntity besides the base name/workspace/etc.)
        # live in the data JSON column, so filter_operation needs the same `data.` prefix
        # that _convert_filter_obj_to_filter_str applies for the filter_obj shorthand.
        own_principal = ComparisonOperation(operator=FilterOperator.EQ, field="data.principal", value=principal)
        service_keys = ComparisonOperation(
            operator=FilterOperator.EQ, field="data.entity_type", value="SERVICE_ACCOUNT"
        )
        filter_operation: FilterOperation = own_principal
        if include_service_accounts:
            filter_operation = LogicalOperation(operator=FilterOperator.OR, operations=[own_principal, service_keys])
        elif service_key_workspace_ids:
            filter_operation = LogicalOperation(
                operator=FilterOperator.OR,
                operations=[
                    own_principal,
                    LogicalOperation(
                        operator=FilterOperator.AND,
                        operations=[
                            service_keys,
                            ComparisonOperation(
                                operator=FilterOperator.IN,
                                field="data.bound_workspace_id",
                                value=list(service_key_workspace_ids),
                            ),
                        ],
                    ),
                ],
            )
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

        async def is_visible(record: AccessKeyEntity) -> bool:
            # The filters above only narrow candidates. Confirm each service key through the same
            # check its lifecycle operations use, so listing can never show a key the caller
            # couldn't manage -- including one an owner match pulled in from the credential store.
            if not record.is_service_account() or include_service_accounts:
                return True
            return admin_override is not None and await admin_override(record.bound_workspace_id)

        visible = await asyncio.gather(*(is_visible(record) for record in records))
        records = [record for record, shown in zip(records, visible, strict=True) if shown]
        # During the dual-store migration window, a page can exceed page_size:
        # both stores already applied the requested offset and limit, so trimming
        # here would permanently hide a fetched row without a merged cursor.
        # Short non-admin pages after Python filtering are intentional: favor never hiding valid keys over exact pagination; no fix needed.
        return AccessKeyListResponse(
            data=[self._metadata(record) for record in records],
            has_more=credential_has_more or page < result.pagination.total_pages,
        )

    async def service_account_access(self, service_account_principal: str) -> ServiceAccountAccess:
        """Read what this service account can reach: its live keys and its active role bindings.

        Keys come from the credential store, when configured, and from the entity store; revoked
        and expired ones give no access, so they're left out. Role bindings are those keyed by any
        identifier the PDP matches the account's keys by: ``service_account_principal`` (also its
        only alias) and, once the credential store has an identity for it, that identity's account
        ID. The credential store is read first so that a disabled identity is reported as such
        (AccountIdentityUnavailableError) rather than reading as an account with no keys.
        """
        identity: AccountIdentityRecord | None = None
        records: list[AccessKeyEntity] = []
        if self._credential_adapter is not None:
            identity, records = await self._credential_adapter.service_account_keys(service_account_principal)
        from_entity_store = self._scan(
            AccessKeyEntity,
            workspace=ACCESS_KEY_WORKSPACE,
            condition=ComparisonOperation(
                operator=FilterOperator.EQ, field="data.subject_principal", value=service_account_principal
            ),
        )
        async with aclosing(from_entity_store):
            records = [*records, *[record async for page in from_entity_store for record in page]]
        principals = (service_account_principal, identity.account_id) if identity else (service_account_principal,)
        bindings = self._scan(
            RoleBindingEntity,
            workspace=ALL_WORKSPACES,
            # Revoked bindings are filtered in Python below: the entity service only supports
            # $exists for registered relationships, not plain fields like revoked_at.
            condition=ComparisonOperation(operator=FilterOperator.IN, field="data.principal", value=list(principals)),
        )
        roles: dict[str, set[str]] = {}
        async with aclosing(bindings):
            async for page in bindings:
                for binding in page:
                    if binding.revoked_at is None:
                        roles.setdefault(binding.workspace, set()).add(binding.role)
        return ServiceAccountAccess(
            principals=principals,
            live_keys=tuple(
                LiveServiceAccountKey(jti=record.name, bound_workspace_id=record.bound_workspace_id)
                for record in records
                if self._status(record) not in ("REVOKED", "EXPIRED")
            ),
            roles={workspace: frozenset(held) for workspace, held in roles.items()},
        )

    async def workspaces_with_role(self, principals: Sequence[str], role: str) -> set[str]:
        """Names of the workspaces where any of ``principals`` holds an active ``role`` binding.

        Reads role bindings directly, as the PDP's policy data is built from them, but only to
        narrow candidates: callers still confirm the role through the PDP before relying on it.
        """
        principals = list(principals)

        async def workspaces_of(batch: list[str]) -> set[str]:
            bindings = self._scan(
                RoleBindingEntity,
                workspace=ALL_WORKSPACES,
                condition=LogicalOperation(
                    operator=FilterOperator.AND,
                    operations=[
                        ComparisonOperation(operator=FilterOperator.IN, field="data.principal", value=batch),
                        ComparisonOperation(operator=FilterOperator.EQ, field="data.role", value=role),
                    ],
                ),
            )
            async with aclosing(bindings):
                # Revoked bindings are filtered here for the same reason as in service_account_access.
                return {binding.workspace async for page in bindings for binding in page if binding.revoked_at is None}

        batches = await asyncio.gather(
            *(
                workspaces_of(principals[start : start + _ROLE_BINDING_PRINCIPAL_BATCH_SIZE])
                for start in range(0, len(principals), _ROLE_BINDING_PRINCIPAL_BATCH_SIZE)
            )
        )
        return set().union(*batches) if batches else set()

    async def is_service_account_disabled(self, service_account_principal: str) -> bool:
        """Whether the service account's account identity exists but is disabled.

        Always False without a credential store, which is the only place identities are tracked.
        That's safe: without one there is no identity to disable, and every key lives in the entity
        store, which service_account_access scans in full for the confinement check.
        """
        return self._credential_adapter is not None and await self._credential_adapter.is_service_account_disabled(
            service_account_principal
        )

    async def _scan(
        self, entity_type: type[_ScannedT], *, workspace: str, condition: FilterOperation
    ) -> AsyncGenerator[list[_ScannedT], None]:
        """Yield every entity matching ``condition``, a page at a time, paging by entity ID.

        For the confinement check, where a skipped row would fail open: unlike offset paging, each
        page starts after the last ID read, so a row deleted while the scan runs can't shift an
        unread row onto a page already read.
        """
        after_id: str | None = None
        while True:
            filter_operation: FilterOperation = (
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
                raise UnpageableScanError(f"Cannot page past a {entity_type.__name__} without an ID")

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
        self,
        jti: str,
        principal: str,
        *,
        admin_override: AdminOverride | None = None,
        before_unsuspend: Callable[[AccessKeyEntity], Awaitable[None]] | None = None,
    ) -> tuple[bool, AccessKeyReversibleStatus]:
        return await self._set_suspension(
            jti, principal, suspended=False, admin_override=admin_override, before_write=before_unsuspend
        )

    async def _set_suspension(
        self,
        jti: str,
        principal: str,
        *,
        suspended: bool,
        admin_override: AdminOverride | None = None,
        before_write: Callable[[AccessKeyEntity], Awaitable[None]] | None = None,
    ) -> tuple[bool, AccessKeyReversibleStatus]:
        """Move a key to ACTIVE or SUSPENDED.

        ``before_write`` runs on the record about to change, after the no-op cases are ruled out
        and before each write attempt; raising from it leaves the key as it was. Unsuspending uses
        it to re-check that the caller may still put this credential back in service.
        """
        target_status: Literal["ACTIVE", "SUSPENDED"] = "SUSPENDED" if suspended else "ACTIVE"
        completed_action = "suspended" if suspended else "unsuspended"
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
            if before_write is not None:
                await before_write(record)
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
            # Service-bound keys belong to the workspace they're bound to, or to the platform when
            # they're bound to none -- not to the administrator who created them. Authorize every
            # lifecycle operation against the binding, including for that creator.
            if admin_override is not None and await admin_override(record.bound_workspace_id):
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
            # The ID is the binding; the name is a display handle the issuer fills in once it has
            # resolved it, so a record read without a workspace lookup still reports the binding.
            bound_workspace=record.bound_workspace_id,
            bound_workspace_id=record.bound_workspace_id,
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
        # Every identifier a role binding for the caller may be keyed by, as the PDP's has_role
        # matches them (not the wildcard principal, which it ignores); see _administered_workspace_ids.
        self._role_binding_principals = list(
            dict.fromkeys(
                identifier
                for identifier in (
                    principal.id,
                    principal.account_id,
                    *principal.authz_aliases,
                    principal.email,
                    *principal.groups,
                )
                if identifier and identifier != "*"
            )
        )
        # Authorizes managing a service-bound key against its bound workspace; see AdminOverride.
        # It memoizes per workspace for the request (one issuer per request, see
        # get_access_key_issuer), so an endpoint-level pre-check and create_async's own
        # defense-in-depth re-check share a single PDP has_role round trip instead of each
        # paying for one. None when the caller can never manage a service-bound key.
        self._admin_override = admin_override
        # Workspace name <-> ID, resolved once per request. ``workspace_names`` is shared with the
        # admin check, so a name either side reads is not looked up again by the other; None means
        # the workspace is gone or the caller can't see it.
        self._workspace_names: dict[str, str | None] = workspace_names if workspace_names is not None else {}
        self._workspace_ids: dict[str, str | None] = {}
        # The caller's own Scoped Access Key scope, if any; stops it from minting a broader
        # replacement. None = unrestricted caller.
        self._caller_scope = frozenset(caller_scope) if caller_scope is not None else None

    async def is_platform_admin(self) -> bool:
        """Whether the caller is a current HelixAdmin.

        Asking about a key bound to no workspace asks exactly that: only a HelixAdmin manages one.
        Backed by the same memoized admin_override create_async consults, so callers that gate on
        HelixAdmin status before invoking create_async (to return 403 instead of create_async's
        400) do not trigger a second PDP round trip.
        """
        return await self.can_administer(None)

    async def can_administer(self, workspace_id: str | None) -> bool:
        """Whether the caller may manage a service-bound key bound to this workspace (None: to none)."""
        return self._admin_override is not None and await self._admin_override(workspace_id)

    async def resolve_bound_workspace(self, workspaces: Sequence[str]) -> tuple[str | None, str | None]:
        """The workspace a new service-account key binds to, as (ID, name), or (None, None) for none.

        A HelixAdmin may create a key for any service account, and it stays bound to no workspace:
        it's a platform-owned credential, and binding it would hand it to a workspace's Admins.
        Anyone else must name exactly one workspace and be its Admin, and gets a key bound to it.
        Raises AccessKeyForbiddenError otherwise. The endpoint calls this to authorize the request
        before a key is minted, and create_async calls it again to re-derive the binding; the two
        share this request's cached role and workspace lookups.
        """
        if await self.is_platform_admin():
            return None, None
        if len(workspaces) != 1:
            raise AccessKeyForbiddenError(ADMIN_REQUIRED_MESSAGE if not workspaces else ONE_WORKSPACE_MESSAGE)
        name = workspaces[0]
        workspace_id = await self._bound_workspace_id(name)
        if not await self.can_administer(workspace_id):
            raise AccessKeyForbiddenError(ADMIN_REQUIRED_MESSAGE)
        return workspace_id, name

    async def _workspace_id(self, name: str) -> str | None:
        """The current ID of a workspace, by name; cached for this request, along with its name by ID."""
        if name not in self._workspace_ids:
            workspace = await resolve_workspace(self._workspaces_client, name)
            self._workspace_ids[name] = workspace.id if workspace is not None else None
            if workspace is not None:
                # Spares the Admin check the reverse lookup; it still re-confirms the ID after has_role.
                self._workspace_names.setdefault(workspace.id, name)
        return self._workspace_ids[name]

    async def _workspace_name(self, workspace_id: str) -> str | None:
        """The current name of a workspace, by ID; cached for this request."""
        if workspace_id not in self._workspace_names:
            self._workspace_names[workspace_id] = await resolve_workspace_name(self._workspaces_client, workspace_id)
        return self._workspace_names[workspace_id]

    async def _display_names(self, workspace_ids: Sequence[str]) -> None:
        """Record current names for these workspaces, so _with_cached_bound_name can read them.

        One batched lookup instead of one per key; a workspace that's gone or unreadable is
        recorded as None and shown by its ID.
        """
        await self._read_workspaces(
            "id", [workspace_id for workspace_id in workspace_ids if workspace_id not in self._workspace_names]
        )

    async def _read_workspaces(self, field: Literal["id", "name"], values: Sequence[str]) -> list[str]:
        """Find the workspaces whose ``field`` is among ``values``, a batch per call, recording their names.

        So many workspaces cost a lookup per batch rather than one each, up to
        _MAX_CONCURRENT_WORKSPACE_READS batches at once. Returns the IDs found; an ID asked for and
        not found (gone, or not visible to the caller) is recorded as None, as resolve_workspace_name
        would. A batch that fails is logged and left out; the others' results are kept.
        """
        unique = list(dict.fromkeys(values))
        concurrent_reads = asyncio.Semaphore(_MAX_CONCURRENT_WORKSPACE_READS)

        async def read(batch: list[str]) -> list[tuple[str, str]]:
            async with concurrent_reads:
                page = await self._workspaces_client.list_workspaces(
                    query_params={"filter": json.dumps({field: {"$in": batch}}), "page_size": len(batch)}
                )
                return [(workspace.id, workspace.name) async for workspace in page.items()]

        batches = [
            unique[start : start + _WORKSPACE_NAME_BATCH_SIZE]
            for start in range(0, len(unique), _WORKSPACE_NAME_BATCH_SIZE)
        ]
        results = await asyncio.gather(*(read(batch) for batch in batches), return_exceptions=True)
        found_ids: list[str] = []
        for batch, result in zip(batches, results, strict=True):
            if isinstance(result, NemoClientError):
                logger.warning(
                    "Could not read some workspaces; continuing with the others",
                    extra={"actor_principal": self.principal, "workspace_lookup_field": field},
                    exc_info=result,
                )
                continue
            if isinstance(result, BaseException):
                raise result
            for workspace_id, name in result:
                self._workspace_names[workspace_id] = name
                found_ids.append(workspace_id)
            if field == "id":
                for workspace_id in batch:
                    self._workspace_names.setdefault(workspace_id, None)
        return found_ids

    def _with_cached_bound_name(self, key: _MetadataT) -> _MetadataT:
        """``key`` with its bound workspace named from this request's lookups, or by ID if unknown."""
        if key.bound_workspace_id is None:
            return key
        name = self._workspace_names.get(key.bound_workspace_id) or key.bound_workspace_id
        return key.model_copy(update={"bound_workspace": name})

    async def _bound_workspace_id(self, name: str) -> str:
        """The current ID of the workspace a new key binds to. Fails closed: an unreadable one can't be bound.

        A transient lookup failure surfaces as a 503 (see resolve_workspace_id), not a validation error.
        """
        workspace_id = await self._workspace_id(name)
        if workspace_id is None:
            raise AccessKeyValidationError(f"Workspace '{name}' could not be resolved")
        return workspace_id

    async def _bound_workspace_of(self, record: AccessKeyEntity) -> tuple[str | None, str | None]:
        """A stored key's bound workspace ID and its current name (a workspace that's gone keeps its ID).

        For a caller who isn't a HelixAdmin the name feeds the confinement check, which compares
        role bindings by workspace name, so a failed lookup is raised (as a 503) rather than
        replaced by the ID, which would make a binding in the bound workspace look like one
        outside it. A HelixAdmin skips that check, so the name is only for display.
        """
        workspace_id = record.bound_workspace_id
        if workspace_id is None:
            return None, None
        return workspace_id, await self._workspace_name(workspace_id) or workspace_id

    async def _require_enabled_service_account(
        self, service_account_id: str | None, *, action: str = "creating keys for it"
    ) -> None:
        """Reject a disabled service account, for checks that don't read its access.

        Saving a new key for it would fail anyway. Checks that do read it get the same error from
        _service_account_access, since a disabled account's keys can't be listed.
        """
        if await self._registry.is_service_account_disabled(f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{service_account_id}"):
            raise _service_account_disabled(service_account_id, action)

    async def _service_account_access(self, service_account_id: str | None, *, action: str) -> ServiceAccountAccess:
        """What the service account can reach (see AccessKeyRegistry.service_account_access); a 400 if it's disabled."""
        try:
            return await self._registry.service_account_access(
                f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{service_account_id}"
            )
        except AccountIdentityUnavailableError as exc:
            raise _service_account_disabled(service_account_id, action) from exc

    async def _require_confined_service_account(
        self,
        workspace_id: str | None,
        workspace_name: str | None,
        service_account_id: str | None,
        *,
        action: str,
    ) -> None:
        """Raise _confinement_error's error, if there is one."""
        if error := await self._confinement_error(workspace_id, workspace_name, service_account_id, action=action):
            raise error

    async def _require_confined_existing_key(
        self, record: AccessKeyEntity, *, action: str
    ) -> tuple[str | None, str | None]:
        """Apply the enabled-account and confinement checks to a stored service-bound key.

        For rotate and unsuspend, which hand a working credential for the key's account back to the
        caller. A HelixAdmin only needs the account to be enabled. Returns the key's bound
        workspace ID and name, so the caller doesn't resolve them again.
        """
        service_account_id = (record.subject_principal or "").removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX)
        if await self.is_platform_admin():
            await self._require_enabled_service_account(service_account_id, action=action)
            workspace_id = record.bound_workspace_id
            if workspace_id is None:
                return None, None
            # Display only, so a workspace that can't be read is shown by its ID rather than
            # failing the request, as it would for the confinement check below.
            await self._display_names([workspace_id])
            return workspace_id, self._workspace_names.get(workspace_id) or workspace_id
        workspace_id, workspace_name = await self._bound_workspace_of(record)
        await self._require_confined_service_account(workspace_id, workspace_name, service_account_id, action=action)
        return workspace_id, workspace_name

    async def _confinement_error(
        self,
        workspace_id: str | None,
        workspace_name: str | None,
        service_account_id: str | None,
        *,
        action: str,
    ) -> AccessKeyForbiddenError | None:
        """Why a workspace Admin can't use this service account for a key bound to this workspace, if they can't.

        The JWT authenticates as service-account:<id> everywhere that principal has access, so a
        workspace Admin may only use an account with no key or role binding outside the bound
        workspace -- not one a HelixAdmin provisioned with access elsewhere, and not one whose live
        key is bound to an earlier workspace of the same name, whose access a recreated workspace
        must not inherit. Applies to create, rotate and unsuspend: each puts a usable credential
        for the account back in a workspace Admin's hands.

        Run before the new key's record is written, and again after: a key or role binding that a
        HelixAdmin (or another workspace's Admin) adds for the same service account in between
        would slip past the first check, and the second closes that window for whichever request
        commits second. None for HelixAdmins and unbound keys.
        """
        if workspace_id is None or workspace_name is None or service_account_id is None:
            return None
        if await self.is_platform_admin():
            return None
        access = await self._service_account_access(service_account_id, action=action)
        if access.reaches_outside(workspace_id, workspace_name):
            return _outside_access_error(service_account_id, workspace_name)
        return None

    def _enforce_caller_scope(self, request: AccessKeyCreateRequest) -> None:
        """Reject a scope-restricted caller minting a key broader than its own access."""
        if self._caller_scope is None:
            return
        if request.workspaces and not {"entities", "platform"} & self._caller_scope:
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
        bound_workspace_id: str | None = None
        bound_workspace_name: str | None = None
        if request.service_account_id is not None and request.rotates is not None:
            # `rotates` only applies to personal keys; a service-bound replacement would let
            # rotation quietly change the credential's identity type.
            raise AccessKeyValidationError(
                "rotates cannot be combined with service_account_id; rotation is only supported for personal keys"
            )
        if allow_service_account:
            # Defense-in-depth, mirroring the admin_override re-check that revoke/suspend/
            # unsuspend apply for service-bound keys: don't rely solely on the caller having
            # verified their authorization before setting this flag (see AdminOverride). Re-derives
            # the binding from this request's cached role and workspace lookups, so the endpoint's
            # pre-check and this one agree without paying twice.
            bound_workspace_id, bound_workspace_name = await self.resolve_bound_workspace(
                [grant.workspace for grant in request.workspaces or []]
            )
            await self._require_confined_service_account(
                bound_workspace_id,
                bound_workspace_name,
                request.service_account_id,
                action="creating keys for it",
            )
            await self._require_enabled_service_account(request.service_account_id)
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
        key = await self._issuer.create_async(request, allow_service_account=allow_service_account)
        key = key.model_copy(update={"bound_workspace_id": bound_workspace_id, "bound_workspace": bound_workspace_name})
        try:
            await self._registry.add(
                key,
                owner_principal=self.principal,
                owner_account_id=self.account_id,
                bound_workspace_id=bound_workspace_id,
            )
        except Exception:
            logger.warning(
                "Failed to persist Scoped Access Key lifecycle record; the signed JWT will not be returned to the caller",
                extra={"access_key_jti": key.jti, "actor_principal": self.principal},
                exc_info=True,
            )
            raise
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
            attempted_grants = await self._grant_workspace_memberships(key, request.workspaces)
        except Exception:
            # Don't return a 200 for a key that's missing the access it was supposed to get.
            await self._revoke_after_failed_creation(key.jti, "workspace_grant_failed")
            raise
        # Re-check now that the key and its grants are stored: a key or role binding added for the
        # same service account between the first check and here would have slipped past it, and
        # this closes that window for whichever request commits second.
        if (
            error := await self._confinement_error(
                bound_workspace_id,
                bound_workspace_name,
                request.service_account_id,
                action="creating keys for it",
            )
        ) is not None:
            await self._rollback_workspace_memberships(key, attempted_grants)
            await self._revoke_after_failed_creation(key.jti, "confinement_check_failed")
            raise error
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
        check failed is treated as not administered; the failure is logged once per workspace.
        """
        admin_override = self._admin_override
        if admin_override is None:
            return None
        logged: set[str | None] = set()

        async def check(workspace_id: str | None) -> bool:
            try:
                return await admin_override(workspace_id)
            # The PDP and workspace lookups report outages as HTTPException; anything else, such as
            # a missing PDP URL, is a misconfiguration that should still fail the request.
            except HTTPException:
                if workspace_id not in logged:
                    logged.add(workspace_id)
                    logger.warning(
                        "Admin check failed while listing Scoped Access Keys; "
                        "hiding the service-bound keys bound to this workspace from the list",
                        extra={"actor_principal": self.principal, "workspace_id": workspace_id},
                        exc_info=True,
                    )
                return False

        return check

    async def _administered_workspace_ids(self) -> list[str]:
        """IDs of the workspaces the caller holds an Admin binding in, recording their names for this request.

        Lets listing include the service-bound keys other Admins created for workspaces the caller
        administers. Costs one role-binding query plus a workspace lookup per batch of the caller's
        Admin workspaces; a caller who is Admin nowhere pays for no lookups and is shown no
        service-bound keys. The bindings only pick candidates: the per-key admin_override still
        confirms the role through the PDP. Only runs for callers who aren't HelixAdmins. Workspaces
        that can't be read are left out (see _read_workspaces); if the bindings can't be read,
        there are none, and listing shows only the caller's own keys.

        Capped at _MAX_ADMINISTERED_WORKSPACES, sorted by name so the cap is stable across pages:
        the IDs go into the entity store's query string, which an unbounded list would not survive.
        """
        try:
            names = await self._registry.workspaces_with_role(self._role_binding_principals, "Admin")
        except (NemoClientError, EntityStoreError, UnpageableScanError):
            logger.warning(
                "Could not read the caller's Admin workspaces while listing Scoped Access Keys; "
                "showing only the caller's own keys",
                extra={"actor_principal": self.principal},
                exc_info=True,
            )
            return []
        capped = sorted(names)[:_MAX_ADMINISTERED_WORKSPACES]
        if len(names) > len(capped):
            logger.warning(
                "The caller administers more workspaces than one Scoped Access Key listing covers; "
                "listing the keys bound to the first %d by name",
                _MAX_ADMINISTERED_WORKSPACES,
                extra={"actor_principal": self.principal, "administered_workspaces": len(names)},
            )
        return await self._read_workspaces("name", capped)

    async def list_async(self, *, page: int = 1, page_size: int = 100) -> AccessKeyListResponse:
        self._ensure_enabled()
        # A current HelixAdmin sees every service-bound key, while a workspace Admin also sees the
        # service-bound keys bound to workspaces they administer, mirroring the admin_override
        # check lifecycle operations already apply (see AccessKeyRegistry.list_for_principal). If
        # the PDP role check itself fails (e.g. unreachable), degrade to the caller's own personal
        # keys rather than failing listing outright for every user. Only that check is guarded: a
        # failure listing the keys themselves must still surface.
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
            await self._administered_workspace_ids()
            if admin_override is not None and not include_service_accounts
            else []
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
        # Read in batches up front, so each key's name comes from the cache.
        await self._display_names([key.bound_workspace_id for key in listed.data if key.bound_workspace_id])
        return listed.model_copy(update={"data": [self._with_cached_bound_name(key) for key in listed.data]})

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

        async def confine(record: AccessKeyEntity) -> None:
            # Unsuspending returns a usable credential for the account to the caller, so it
            # re-applies the create-time confinement check, as rotation does. Raising here leaves
            # the key suspended: the registry runs this before it writes.
            if record.is_service_account():
                await self._require_confined_existing_key(record, action="unsuspending its keys")

        unsuspended, effective_status = await self._registry.unsuspend(
            jti, self.principal, admin_override=self._admin_override, before_unsuspend=confine
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
        bound_workspace_id: str | None = None
        bound_workspace_name: str | None = None
        if allow_service_account:
            # Rotation hands back a working credential for the key's account, so it re-applies the
            # create-time checks: get_rotatable already authorized the caller against the binding,
            # and the account must still be confined to the bound workspace.
            bound_workspace_id, bound_workspace_name = await self._require_confined_existing_key(
                old_record, action="rotating its keys"
            )
            service_account_id = (old_record.subject_principal or "").removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX)
        else:
            service_account_id = None
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
        # The successor is the same credential with a new secret, so it keeps the binding: the
        # workspace's Admins manage it exactly as they managed its predecessor.
        new_key = new_key.model_copy(
            update={"bound_workspace_id": bound_workspace_id, "bound_workspace": bound_workspace_name}
        )
        try:
            await self._registry.add(
                new_key,
                owner_principal=self.principal,
                owner_account_id=self.account_id,
                bound_workspace_id=bound_workspace_id,
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
