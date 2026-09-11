# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence and issuance services for Scoped Access Keys."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeVar

from fastapi import Depends, HTTPException
from nemo_platform_plugin.auth.access_keys.issuer import AccessKeyFeatureDisabledError
from nemo_platform_plugin.auth.access_keys.types import (
    AccessKeyCreateRequest,
    AccessKeyCreateResponse,
    AccessKeyListResponse,
    AccessKeyMetadataResponse,
    AccessKeyReversibleStatus,
    AccessKeyRotateResponse,
    AccessKeyStatus,
    AccessKeyWorkspaceGrant,
)
from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceMemberRequest
from nmp.common.api.filter import ComparisonOperation, FilterOperator, LogicalOperation
from nmp.common.auth.access_keys import (
    LEGACY_ACCESS_KEY_METADATA_VERSION,
    SERVICE_ACCOUNT_PRINCIPAL_PREFIX,
    AccessKeyIssuerService,
    AccessKeyValidationError,
)
from nmp.common.auth.models import Principal
from nmp.common.auth.token_claims import TokenClaims
from nmp.common.config import AuthConfig
from nmp.common.entities import EntityClient, EntityConflictError, EntityNotFoundError
from nmp.common.service.dependencies import get_entity_client
from nmp.core.auth.entities import AccessKeyEntity

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

_MutationResultT = TypeVar("_MutationResultT")


class _Retry:
    """Sentinel a `_retry_on_conflict` conflict handler returns to request another attempt."""


_RETRY = _Retry()

# Service-bound keys are machine-to-machine credentials owned by the platform, not
# by the creating individual (AIRCORE-986). Every lifecycle operation therefore
# uses this callback to require a current PlatformAdmin, including when the caller
# originally created the key.
AdminOverride = Callable[[], Awaitable[bool]]


def _memoize_admin_override(admin_override: AdminOverride | None) -> AdminOverride | None:
    """Cache a single admin_override result for the lifetime of one lifecycle call.

    _get_owned() may be invoked twice within one revoke()/suspend()/unsuspend() call
    (initial read, then a re-read after a losing optimistic-lock race). Without this,
    each read would trigger its own PDP has_role round trip for service-bound keys.
    """
    if admin_override is None:
        return None
    result: bool | None = None

    async def cached() -> bool:
        nonlocal result
        if result is None:
            result = await admin_override()
        return result

    return cached


class AccessKeyNotFoundError(Exception):
    """Raised when a key does not exist or is not owned by the caller."""


class AccessKeyStateConflictError(Exception):
    """Raised when an irreversible lifecycle state prevents a transition."""


class AccessKeyRegistry:
    """Durable access-key lifecycle records stored by the entities service."""

    def __init__(self, entity_client: EntityClient) -> None:
        self._entity_client = entity_client

    async def add(self, key: AccessKeyCreateResponse, *, owner_principal: str | None = None) -> None:
        owner = owner_principal or key.principal
        await self._entity_client.create(
            AccessKeyEntity(
                name=key.jti,
                workspace=ACCESS_KEY_WORKSPACE,
                key_name=key.name,
                description=key.description,
                principal=owner,
                subject_principal=key.principal if key.principal != owner else None,
                entity_type=key.entity_type,
                issuer=key.issuer,
                audiences=key.audiences,
                scope=key.scope,
                issued_at=key.created_at,
                expires_at=key.expires_at,
            )
        )

    async def discard_unreturned(self, jti: str) -> None:
        try:
            await self._entity_client.delete(
                AccessKeyEntity,
                name=jti,
                workspace=ACCESS_KEY_WORKSPACE,
            )
        except EntityNotFoundError:
            pass

    async def list_for_principal(
        self, principal: str, *, page: int, page_size: int, include_service_accounts: bool = False
    ) -> AccessKeyListResponse:
        # Service-bound keys are platform-owned, not creator-owned (AIRCORE-986): a
        # PlatformAdmin other than the creator can already revoke/suspend one via
        # _get_owned's admin_override, so listing must surface every service-bound key
        # to any current admin rather than only the one who happened to create it.
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
        if not include_service_accounts:
            records = [record for record in records if record.entity_type != "SERVICE_ACCOUNT"]
        # Short non-admin pages after Python filtering are intentional: favor never hiding valid keys over exact pagination; no fix needed.
        return AccessKeyListResponse(
            data=[self._metadata(record) for record in records],
            has_more=page < result.pagination.total_pages,
        )

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
            await self._entity_client.update(record.model_copy(update={"status": "REVOKED"}))
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
        """Validate a `rotates` target before its replacement is minted.

        Raises AccessKeyValidationError (not AccessKeyNotFoundError) so callers can rely on this
        failing the create request with a 400 before any new key is issued, rather than silently
        minting an unrelated key while the referenced rotation target goes unrevoked. `rotates` is
        scoped to personal keys only: a service-bound predecessor exists independently of any
        single owner and has no stable non-admin identity to key rotation off of.
        """
        try:
            record = await self._get_owned(jti, principal, admin_override=admin_override)
        except AccessKeyNotFoundError as exc:
            raise AccessKeyValidationError(
                f"rotates references a Scoped Access Key that does not exist or is not owned by the caller: {jti}"
            ) from exc
        if record.is_service_account:
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
            await self._entity_client.update(updated)
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
            await self._entity_client.update(updated)
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
        try:
            record = await self._get_for_subject(jti, principal)
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
                await self._entity_client.update(
                    current_record.model_copy(update={"last_used_at": datetime.now(tz=UTC)})
                )
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
        if record.is_service_account:
            # Service-bound keys belong to the platform, not to the administrator who
            # created them. Require the caller to be a *current* PlatformAdmin for every
            # lifecycle operation, including when that caller is the recorded creator.
            if admin_override is not None and await admin_override():
                return record
        elif record.principal == principal:
            return record
        raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")

    async def _get(self, jti: str) -> AccessKeyEntity:
        try:
            return await self._entity_client.get(
                AccessKeyEntity,
                name=jti,
                workspace=ACCESS_KEY_WORKSPACE,
            )
        except EntityNotFoundError as exc:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found") from exc

    @staticmethod
    def _metadata(record: AccessKeyEntity) -> AccessKeyMetadataResponse:
        effective_status = AccessKeyRegistry._status(record)
        return AccessKeyMetadataResponse(
            jti=record.name,
            name=record.key_name,
            description=record.description,
            principal=record.subject_principal or record.principal,
            entity_type=record.entity_type,
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

        metadata = raw_claims.get("nmp_access_key")
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


def get_access_key_registry(entity_client: EntityClient = Depends(get_entity_client)) -> AccessKeyRegistry:
    return AccessKeyRegistry(entity_client.as_service("auth", internal=True))


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
    ) -> None:
        self._issuer = AccessKeyIssuerService(config=config, principal=principal)
        self._config = config
        self._registry = registry
        self._workspaces_client = workspaces_client
        self.principal = principal.id
        # Lets any current PlatformAdmin revoke or suspend a service-bound key;
        # see AdminOverride and AIRCORE-986. Memoized for the lifetime of this issuer
        # instance (one per request, see get_access_key_issuer) so that an endpoint-level
        # pre-check (is_platform_admin) and create_async's own defense-in-depth re-check
        # share a single PDP has_role round trip instead of each paying for one.
        self._admin_override = _memoize_admin_override(admin_override)

    async def is_platform_admin(self) -> bool:
        """Whether the caller is a current PlatformAdmin.

        Backed by the same memoized admin_override create_async consults, so callers that
        need to gate on PlatformAdmin status before invoking create_async (to return 403
        instead of create_async's 400) do not trigger a second PDP round trip.
        """
        return self._admin_override is not None and await self._admin_override()

    async def create_async(
        self, request: AccessKeyCreateRequest, *, allow_service_account: bool = False
    ) -> AccessKeyCreateResponse:
        self._ensure_enabled()
        if request.service_account_id is not None and request.rotates is not None:
            # `rotates` is only meaningful for personal keys: a service-bound predecessor has no
            # stable non-admin owner identity to correlate rotations against (see
            # AccessKeyRegistry.get_for_rotation), and allowing the *replacement* to be
            # service-bound would let a rotation silently change the credential's identity type.
            raise AccessKeyValidationError(
                "rotates cannot be combined with service_account_id; rotation is only supported for personal keys"
            )
        if allow_service_account:
            # Defense-in-depth, mirroring the admin_override re-check that revoke/suspend/
            # unsuspend apply for service-bound keys: don't rely solely on the caller having
            # verified PlatformAdmin status before setting this flag (see AdminOverride).
            if not await self.is_platform_admin():
                raise AccessKeyValidationError("Service-bound Scoped Access Keys require PlatformAdmin")
        if request.rotates is not None:
            # Validate the rotation target before minting its replacement: an invalid `rotates`
            # (not found, not owned, or service-bound) fails the request up front instead of
            # silently minting an orphan key while the referenced key goes unrevoked.
            await self._registry.get_for_rotation(request.rotates, self.principal, admin_override=self._admin_override)
        key = await self._issuer.create_async(request, allow_service_account=allow_service_account)
        try:
            await self._registry.add(key, owner_principal=self.principal)
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
        # Provision the new key's requested access before revoking its predecessor, so a key
        # is never left with less access than it had before rotation started.
        try:
            await self._grant_workspace_memberships(key, request.workspaces)
        except Exception:
            # A key whose requested workspace access could not be provisioned is worse than
            # useless: the caller would believe --workspace replaced their separate
            # `nemo workspaces members create` call when it silently didn't. Revoke it and
            # surface the failure instead of returning a 200 with incomplete access.
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
                # Unlike the workspace-grant case, the new key is fully usable here; the only
                # thing that failed is retiring the old one. The old key is still active and
                # untouched, so compensate by revoking the new key and surfacing the failure so
                # the caller can retry rotation, leaving the untouched predecessor as the
                # caller's one valid credential in the meantime.
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
    ) -> None:
        for grant in workspaces or []:
            try:
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
                raise

    async def list_async(self, *, page: int = 1, page_size: int = 100) -> AccessKeyListResponse:
        self._ensure_enabled()
        # A current PlatformAdmin sees every service-bound key, not just the ones they
        # personally created, mirroring the admin_override check lifecycle operations
        # already apply (see AccessKeyRegistry.list_for_principal). If the PDP role check
        # itself fails (e.g. unreachable), degrade to the caller's own keys rather than
        # failing listing outright for every user.
        try:
            include_service_accounts = await self.is_platform_admin()
        except HTTPException:
            logger.warning(
                "PlatformAdmin check failed while listing Scoped Access Keys; "
                "falling back to listing only the caller's own keys",
                extra={"actor_principal": self.principal},
                exc_info=True,
            )
            include_service_accounts = False
        return await self._registry.list_for_principal(
            self.principal, page=page, page_size=page_size, include_service_accounts=include_service_accounts
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
            # the caller having verified PlatformAdmin status upstream (see AdminOverride).
            if not await self.is_platform_admin():
                raise AccessKeyValidationError("Service-bound Scoped Access Keys require PlatformAdmin")
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
            # Preserve the original key's scope restriction; an empty list means the
            # original key was unscoped, and AccessKeyCreateRequest rejects an explicitly
            # empty scope, so map that case to None (also unscoped) instead.
            scope=old_record.scope or None,
            expires_in_seconds=expires_in_seconds,
        )
        new_key = await self._issuer.create_async(request, allow_service_account=allow_service_account)
        try:
            await self._registry.add(new_key, owner_principal=self.principal)
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
