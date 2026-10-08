# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence and issuance services for Scoped Access Keys."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol, TypeVar

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
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, UpdateWorkspaceMemberRequest
from nhx.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
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
    "Service account '{account_id}' already has keys or role bindings outside workspace '{workspace}', "
    "or a key bound to an earlier workspace of the same name; "
    "only a HelixAdmin can create or rotate keys for it"
)
logger = logging.getLogger(__name__)


async def resolve_workspace_id(workspaces_client: AsyncWorkspacesClient, workspace: str) -> str | None:
    """The workspace's current ID, or None if it doesn't exist or the caller can't read it.

    Any other client failure (transport error, 5xx) is raised as a 503 rather than read as "gone":
    callers treat None as a definitive answer, and a transient outage must not turn into a
    spurious 403/404 or a client-error 400.
    """
    try:
        return (await workspaces_client.get_workspace(name=workspace)).data().id
    except (NotFoundError, PermissionDeniedError):
        return None
    except NemoClientError as exc:
        logger.warning("Failed to read workspace '%s'", workspace, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Workspace '{workspace}' could not be read; retry") from exc


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


# Service-bound keys are machine-to-machine credentials owned by the platform, not
# by the creating individual. Every lifecycle operation therefore
# uses this callback to require a current administrator, including when the caller
# originally created the key. It takes the key's bound workspace (None if unbound): a
# HelixAdmin qualifies for any key, a workspace Admin only for keys bound to that workspace.
# The workspace's ID, recorded when the key was bound, lets the check reject a key whose workspace
# was deleted and recreated under the same name.
class AdminOverride(Protocol):
    def __call__(self, workspace: str | None, workspace_id: str | None = None) -> Awaitable[bool]:
        """Whether the caller may manage keys bound to ``workspace`` (and ``workspace_id``, if given)."""
        ...


def _memoize_admin_override(admin_override: AdminOverride | None) -> AdminOverride | None:
    """Cache admin_override results (per workspace) for the lifetime of one lifecycle call.

    _get_owned() may be invoked twice within one revoke()/suspend()/unsuspend() call
    (initial read, then a re-read after a losing optimistic-lock race). Without this,
    each read would trigger its own PDP has_role round trip for service-bound keys.
    """
    if admin_override is None:
        return None
    results: dict[tuple[str | None, str | None], bool] = {}

    async def cached(workspace: str | None, workspace_id: str | None = None) -> bool:
        """Return the memoized admin_override result for this workspace and ID."""
        key = (workspace, workspace_id)
        if key not in results:
            results[key] = await admin_override(workspace, workspace_id)
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
        bound_workspace: str | None = None,
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
            bound_workspace=bound_workspace,
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
    ) -> AccessKeyListResponse:
        # include_service_accounts (a current HelixAdmin) surfaces every service-bound key, not just
        # the ones the caller created. Otherwise only the caller's own records are fetched, so other
        # tenants' service-bound keys can never crowd the caller's keys out of a page, and a
        # service-bound one is kept only while admin_override still accepts its bound workspace.
        # EntityBase fields (all fields on AccessKeyEntity besides the base name/workspace/etc.)
        # live in the data JSON column, so filter_operation needs the same `data.` prefix
        # that _convert_filter_obj_to_filter_str applies for the filter_obj shorthand.
        own_principal = ComparisonOperation(operator=FilterOperator.EQ, field="data.principal", value=principal)
        filter_operation = (
            LogicalOperation(
                operator=FilterOperator.OR,
                operations=[
                    own_principal,
                    ComparisonOperation(operator=FilterOperator.EQ, field="data.entity_type", value="SERVICE_ACCOUNT"),
                ],
            )
            if include_service_accounts
            else own_principal
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
                # Keys the caller created, service-bound ones included, are already covered by the
                # owner match. Only a HelixAdmin pulls in other owners' service keys: asking for them
                # on behalf of every caller would page every tenant's service keys through each list.
                include_service_accounts=include_service_accounts,
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
                and (
                    admin_override is None
                    or not await admin_override(record.bound_workspace, record.bound_workspace_id)
                )
            ):
                continue
            visible_records.append(record)
        # Short non-admin pages after Python filtering are intentional: favor never hiding valid keys over exact pagination; no fix needed.
        return AccessKeyListResponse(
            data=[self._metadata(record) for record in visible_records],
            has_more=credential_has_more or page < result.pagination.total_pages,
        )

    async def has_service_account_access_outside_workspace(
        self, service_account_principal: str, workspace: str, workspace_id: str | None = None
    ) -> bool:
        """Whether this service account has a key or an active role binding outside ``workspace``.

        Keys are checked in any status, in both the entity store and, when configured, the
        credential store. Role bindings catch access granted without a key, e.g. membership a
        HelixAdmin or another workspace's Admin pre-provisioned for the principal. Bindings compare
        by name only: deleting a workspace deletes its bindings, so none can outlive the workspace.

        With ``workspace_id``, a key bound to a different workspace ID also counts as outside, even
        under the same name: deleting a workspace doesn't revoke its keys, so a recreated workspace
        must not take over an account whose old key is still live. Without it, only names compare.
        """

        def is_outside(record: AccessKeyEntity) -> bool:
            if record.bound_workspace != workspace:
                return True
            return workspace_id is not None and record.bound_workspace_id != workspace_id

        if self._credential_adapter is not None and await self._credential_adapter.any_for_service_account(
            service_account_principal, is_outside
        ):
            return True
        page = 1
        while True:
            result = await self._entity_client.list(
                AccessKeyEntity,
                workspace=ACCESS_KEY_WORKSPACE,
                filter_operation=ComparisonOperation(
                    operator=FilterOperator.EQ, field="data.subject_principal", value=service_account_principal
                ),
                page=page,
                page_size=100,
            )
            if any(is_outside(record) for record in result.data):
                return True
            if page >= result.pagination.total_pages:
                break
            page += 1
        page = 1
        while True:
            result = await self._entity_client.list(
                RoleBindingEntity,
                workspace=ALL_WORKSPACES,
                # Revoked bindings are filtered in Python below: the entity service only supports
                # $exists for registered relationships, not plain fields like revoked_at.
                filter_operation=ComparisonOperation(
                    operator=FilterOperator.EQ, field="data.principal", value=service_account_principal
                ),
                page=page,
                page_size=100,
            )
            if any(binding.revoked_at is None and binding.workspace != workspace for binding in result.data):
                return True
            if page >= result.pagination.total_pages:
                return False
            page += 1

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
            if admin_override is not None and await admin_override(record.bound_workspace, record.bound_workspace_id):
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
            workspace=record.bound_workspace,
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
    ) -> None:
        self._issuer = AccessKeyIssuerService(config=config, principal=principal)
        self._config = config
        self._registry = registry
        self._workspaces_client = workspaces_client
        self.principal = principal.id
        self.account_id = principal.account_id
        # Lets a current HelixAdmin, or an Admin of the bound workspace, manage a
        # service-bound key; see AdminOverride. Memoized for the lifetime of this issuer
        # instance (one per request, see get_access_key_issuer) so that an endpoint-level
        # pre-check (is_platform_admin) and create_async's own defense-in-depth re-check
        # share a single PDP has_role round trip instead of each paying for one.
        self._admin_override = _memoize_admin_override(admin_override)
        # The caller's own Scoped Access Key scope, if any; stops it from minting a broader
        # replacement. None = unrestricted caller.
        self._caller_scope = frozenset(caller_scope) if caller_scope is not None else None

    async def is_platform_admin(self) -> bool:
        """Whether the caller is a current HelixAdmin."""
        return await self.can_administer(None)

    async def can_administer(self, workspace: str | None, workspace_id: str | None = None) -> bool:
        """Whether the caller is a HelixAdmin or an Admin of ``workspace``.

        Uses the memoized admin_override, so gating before create_async costs no extra PDP call.
        """
        return self._admin_override is not None and await self._admin_override(workspace, workspace_id)

    async def _get_workspace_id(self, workspace: str | None) -> str | None:
        """The bound workspace's current ID. Fails closed: a missing or unreadable workspace can't be bound.

        A transient lookup failure surfaces as a 503 (see resolve_workspace_id), not a validation error.
        """
        if workspace is None:
            return None
        workspace_id = await resolve_workspace_id(self._workspaces_client, workspace)
        if workspace_id is None:
            raise AccessKeyValidationError(f"Workspace '{workspace}' could not be resolved")
        return workspace_id

    async def _require_namespaced_service_account(
        self, workspace: str | None, workspace_id: str | None, service_account_id: str | None
    ) -> None:
        """Confine a workspace Admin to service accounts that belong to their workspace.

        The JWT authenticates as service-account:<id> everywhere that principal has access, so a
        workspace Admin may only use an id namespaced under the bound workspace that no key
        outside that workspace already uses (e.g. one a HelixAdmin provisioned with access to
        other workspaces, which a newly created workspace of the same name must not take over).
        ``workspace_id`` extends that to a key bound to an earlier workspace of the same name,
        whose live key a recreated workspace must not inherit access for. HelixAdmins are
        unrestricted. Applies to create and rotate alike.
        """
        if workspace is None or await self.is_platform_admin():
            return
        prefix = f"{workspace}/"
        account_id = service_account_id or ""
        if not (account_id.startswith(prefix) and len(account_id) > len(prefix)):
            raise AccessKeyValidationError(
                f"Workspace Admins must use a service_account_id namespaced under the workspace ('{workspace}/<name>')"
            )
        if await self._registry.has_service_account_access_outside_workspace(
            f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{account_id}", workspace, workspace_id
        ):
            raise AccessKeyValidationError(_OUTSIDE_ACCESS_MESSAGE.format(account_id=account_id, workspace=workspace))

    async def _confinement_violated_after_insert(
        self, workspace: str | None, workspace_id: str | None, service_account_id: str | None
    ) -> bool:
        """Re-run the outside-access check once the new key's record exists.

        _require_namespaced_service_account runs before the record is written, so a key or role
        binding that a HelixAdmin (or another workspace's Admin) adds for the same service account
        in between would slip past it. Checking again after the insert closes that window for
        whichever request commits second. Always False for HelixAdmins and unbound keys.
        """
        if workspace is None or service_account_id is None or await self.is_platform_admin():
            return False
        return await self._registry.has_service_account_access_outside_workspace(
            f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{service_account_id}", workspace, workspace_id
        )

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
        if request.service_account_id is not None and request.rotates is not None:
            # `rotates` only applies to personal keys; a service-bound replacement would let
            # rotation quietly change the credential's identity type.
            raise AccessKeyValidationError(
                "rotates cannot be combined with service_account_id; rotation is only supported for personal keys"
            )
        bound_workspace_id: str | None = None
        if allow_service_account:
            # Defense-in-depth, mirroring the admin_override re-check that revoke/suspend/
            # unsuspend apply for service-bound keys: don't rely solely on the caller having
            # verified admin status before setting this flag (see AdminOverride).
            if not await self.can_administer(request.workspace):
                raise AccessKeyValidationError(
                    "Service-bound Scoped Access Keys require HelixAdmin or Admin of the bound workspace"
                )
            bound_workspace_id = await self._get_workspace_id(request.workspace)
            # The check above resolved the workspace by name; repeat it against the ID just read so
            # a workspace recreated in between can't be bound on the old workspace's Admin check.
            if bound_workspace_id is not None and not await self.can_administer(request.workspace, bound_workspace_id):
                raise AccessKeyValidationError(
                    "Service-bound Scoped Access Keys require HelixAdmin or Admin of the bound workspace"
                )
            await self._require_namespaced_service_account(
                request.workspace, bound_workspace_id, request.service_account_id
            )
            if (
                request.workspace is not None
                and not any(grant.workspace == request.workspace for grant in request.workspaces or [])
                # Memberships belong to the service account, not the key, so a default grant would
                # widen every other key of an account that already has a role here (e.g. Viewer).
                and await self._is_not_a_member(
                    request.workspace, f"{SERVICE_ACCOUNT_PRINCIPAL_PREFIX}{request.service_account_id}"
                )
            ):
                # A workspace-bound key's account is always a member of its own workspace.
                request = request.model_copy(
                    update={
                        "workspaces": [
                            *(request.workspaces or []),
                            AccessKeyWorkspaceGrant(workspace=request.workspace),
                        ]
                    }
                )
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
        try:
            await self._registry.add(
                key,
                owner_principal=self.principal,
                owner_account_id=self.account_id,
                bound_workspace=request.workspace,
                bound_workspace_id=bound_workspace_id,
            )
        except Exception:
            logger.warning(
                "Failed to persist Scoped Access Key lifecycle record; the signed JWT will not be returned to the caller",
                extra={"access_key_jti": key.jti, "actor_principal": self.principal},
                exc_info=True,
            )
            raise
        if await self._confinement_violated_after_insert(
            request.workspace, bound_workspace_id, request.service_account_id
        ):
            await self._revoke_after_failed_creation(key.jti, "service_account_confinement_violated")
            raise AccessKeyValidationError(
                _OUTSIDE_ACCESS_MESSAGE.format(account_id=request.service_account_id, workspace=request.workspace)
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
            attempted_grants = await self._grant_workspace_memberships(key, request.workspaces)
        except Exception:
            # Don't return a 200 for a key that's missing the access it was supposed to get.
            await self._revoke_after_failed_creation(key.jti, "workspace_grant_failed")
            raise
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

    async def _is_not_a_member(self, workspace: str, principal: str) -> bool:
        """Whether the principal has no role in workspace yet; an unreadable member list is a 503."""
        try:
            return await self._get_member_roles(workspace, principal) is None
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Members of workspace '{workspace}' could not be read; retry"
            ) from exc

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
        failed: set[tuple[str | None, str | None]] = set()

        async def check(workspace: str | None, workspace_id: str | None = None) -> bool:
            if (workspace, workspace_id) in failed:
                return False
            try:
                return await admin_override(workspace, workspace_id)
            # The PDP and workspace lookups report outages as HTTPException; anything else, such as
            # a missing PDP URL, is a misconfiguration that should still fail the request.
            except HTTPException:
                logger.warning(
                    "Admin check failed while listing Scoped Access Keys; "
                    "hiding the service-bound keys bound to this workspace from the list",
                    extra={"actor_principal": self.principal, "workspace": workspace},
                    exc_info=True,
                )
                failed.add((workspace, workspace_id))
                return False

        return check

    async def list_async(self, *, page: int = 1, page_size: int = 100) -> AccessKeyListResponse:
        self._ensure_enabled()
        # A current HelixAdmin sees every service-bound key, while a workspace Admin also sees
        # the service-bound keys they created for workspaces they still administer, mirroring
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
        return await self._registry.list_for_principal(
            self.principal,
            page=page,
            page_size=page_size,
            include_service_accounts=include_service_accounts,
            admin_override=self._listing_admin_override(),
            owner_account_id=self.account_id,
        )

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
            if not await self.can_administer(old_record.bound_workspace, old_record.bound_workspace_id):
                raise AccessKeyValidationError(
                    "Service-bound Scoped Access Keys require HelixAdmin or Admin of the bound workspace"
                )
            service_account_id = (old_record.subject_principal or "").removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX)
            # can_administer above already confirmed the workspace still has the ID the key was bound to.
            await self._require_namespaced_service_account(
                old_record.bound_workspace, old_record.bound_workspace_id, service_account_id
            )
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
            workspace=old_record.bound_workspace,
            # Keep the original scope; map empty (unscoped) to None since an explicit
            # empty scope is rejected.
            scope=old_record.scope or None,
            expires_in_seconds=expires_in_seconds,
        )
        # get_rotatable only checks ownership, not scope — enforce that separately.
        self._enforce_caller_scope(request)
        new_key = await self._issuer.create_async(request, allow_service_account=allow_service_account)
        try:
            await self._registry.add(
                new_key,
                owner_principal=self.principal,
                owner_account_id=self.account_id,
                bound_workspace=old_record.bound_workspace,
                bound_workspace_id=old_record.bound_workspace_id,
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
        if await self._confinement_violated_after_insert(
            old_record.bound_workspace, old_record.bound_workspace_id, service_account_id
        ):
            await self._discard_orphaned_successor(
                new_key.jti, context="successor Scoped Access Key rejected by the post-insert confinement check"
            )
            raise AccessKeyValidationError(
                _OUTSIDE_ACCESS_MESSAGE.format(account_id=service_account_id, workspace=old_record.bound_workspace)
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
