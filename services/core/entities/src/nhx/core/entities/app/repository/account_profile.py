# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User profiles: IdP-synced identity fields, user overrides, and principal display resolution."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar

from nhx.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountIdentity, DBAccountProfile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.exc import StaleDataError

T = TypeVar("T")

audit_logger = logging.getLogger("nhx.audit.accounts")

# Claims kept in account_identities.claims_snapshot; everything else may be PII we have no use for.
CLAIMS_SNAPSHOT_ALLOWLIST = frozenset({"iss", "sub", "email", "name", "picture", "groups"})
# Identities that never represent a person, so never match a principal string in resolve().
NON_USER_ISSUERS = ("nemo:service", "nemo:system")
MAX_AVATAR_URL_LENGTH = 2048
MAX_IDP_DISPLAY_NAME_LENGTH = 512

ProfileField = Literal["display_name", "timezone", "locale"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def trim_claims_snapshot(claims: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (claims or {}).items() if key in CLAIMS_SNAPSHOT_ALLOWLIST}


def safe_avatar_url(value: str | None) -> str | None:
    """Only https URLs under the length cap are stored; anything else is dropped."""
    if not value or len(value) > MAX_AVATAR_URL_LENGTH or not value.startswith("https://"):
        return None
    return value


@dataclass(frozen=True)
class IdentityClaims:
    """Profile-relevant claims observed for one identity."""

    email: str | None = None
    name: str | None = None
    picture: str | None = None
    # None leaves account_identities.claims_snapshot unchanged (bearer descriptors carry only a subset).
    snapshot: dict[str, Any] | None = None
    login: bool = False


@dataclass(frozen=True)
class AccountProfileRecord:
    account_id: str
    status: str
    primary_email: str | None
    # Name recorded when the account was materialized; predates profiles.
    account_display_name: str | None
    display_name_override: str | None
    idp_display_name: str | None
    avatar_url: str | None
    timezone: str | None
    locale: str | None
    last_login_at: datetime | None

    def display_name(self) -> str | None:
        return self.display_name_override or self.idp_display_name or self.primary_email or self.account_display_name

    def public_display_name(self, principal: str) -> str | None:
        """Display name for other users: never reveals an email the caller didn't send.

        ``account_display_name`` is skipped: sign-in often sets it to the email at the time, which stays
        stale after an IdP email change.
        """
        email = self.primary_email if principal == self.primary_email else None
        return self.display_name_override or self.idp_display_name or email


@dataclass(frozen=True)
class ResolvedPrincipal:
    principal: str
    account_id: str | None = None
    display_name: str | None = None
    idp_display_name: str | None = None
    display_name_is_override: bool = False
    avatar_url: str | None = None
    status: str | None = None
    ambiguous: bool = False


def _record(account: DBAccount, profile: DBAccountProfile | None) -> AccountProfileRecord:
    return AccountProfileRecord(
        account_id=account.id,
        status=account.status,
        primary_email=account.primary_email,
        account_display_name=account.display_name,
        display_name_override=profile.display_name if profile else None,
        idp_display_name=profile.idp_display_name if profile else None,
        avatar_url=profile.idp_avatar_url if profile else None,
        timezone=profile.timezone if profile else None,
        locale=profile.locale if profile else None,
        last_login_at=profile.last_login_at if profile else None,
    )


class AccountProfileStore:
    """Read and write account profiles in the entities database."""

    def __init__(self, session_maker: async_sessionmaker[AsyncSession]) -> None:
        self.session_maker = session_maker

    async def get(self, account_id: str) -> AccountProfileRecord | None:
        async with self.session_maker() as session:
            row = (
                await session.execute(
                    select(DBAccount, DBAccountProfile)
                    .outerjoin(DBAccountProfile, DBAccountProfile.account_id == DBAccount.id)
                    .where(DBAccount.id == account_id)
                )
            ).one_or_none()
        return _record(*row) if row else None

    async def update(self, account_id: str, fields: dict[ProfileField, str | None]) -> AccountProfileRecord:
        """Set user-editable fields; a None value clears the field."""

        async def apply(session: AsyncSession) -> AccountProfileRecord:
            account = await session.get(DBAccount, account_id)
            if account is None:
                raise LookupError(account_id)
            profile = await self._get_or_create_profile(session, account_id)
            for key, value in fields.items():
                setattr(profile, key, value)
            return _record(account, profile)

        return await self._in_transaction(apply)

    async def sync_identity(self, identity_id: str, claims: IdentityClaims) -> None:
        """Record a sighting of ``identity_id`` and copy its claims onto the profile if it is primary."""
        await self._in_transaction(lambda session: self._sync_identity(session, identity_id, claims))

    async def _in_transaction(self, fn: Callable[[AsyncSession], Awaitable[T]]) -> T:
        """Run ``fn`` in a transaction, retried when a concurrent first write created the profile or bumped its version."""

        async def attempt() -> T:
            async with self.session_maker() as session, session.begin():
                return await fn(session)

        for _ in range(2):
            with suppress(IntegrityError, StaleDataError):
                return await attempt()
        return await attempt()

    async def _sync_identity(self, session: AsyncSession, identity_id: str, claims: IdentityClaims) -> None:
        row = (
            await session.execute(
                select(DBAccountIdentity, DBAccount)
                .join(DBAccount, DBAccount.id == DBAccountIdentity.account_id)
                .where(DBAccountIdentity.id == identity_id)
            )
        ).one_or_none()
        if row is None:
            return
        identity, account = row
        now = _utcnow()
        identity.last_seen_at = now
        if claims.snapshot is not None:
            identity.claims_snapshot = trim_claims_snapshot(claims.snapshot)

        profile = await self._get_or_create_profile(session, account.id)
        if claims.login:
            profile.last_login_at = now
        if profile.primary_identity_id != identity.id:
            return
        if claims.name:
            profile.idp_display_name = claims.name[:MAX_IDP_DISPLAY_NAME_LENGTH]
        if avatar := safe_avatar_url(claims.picture):
            profile.idp_avatar_url = avatar
        if claims.email and claims.email != account.primary_email:
            audit_logger.info(
                "account primary email changed",
                extra={"account_id": account.id, "identity_id": identity.id, "issuer": identity.issuer},
            )
            account.primary_email = claims.email

    async def _get_or_create_profile(self, session: AsyncSession, account_id: str) -> DBAccountProfile:
        profile = await session.get(DBAccountProfile, account_id)
        if profile is None:
            profile = DBAccountProfile(account_id=account_id)
            session.add(profile)
        if profile.primary_identity_id is None:
            # First identity linked to the account; also the fallback when the primary was unlinked.
            profile.primary_identity_id = await session.scalar(
                select(DBAccountIdentity.id)
                .where(DBAccountIdentity.account_id == account_id, DBAccountIdentity.status == "active")
                .order_by(DBAccountIdentity.linked_at, DBAccountIdentity.id)
                .limit(1)
            )
        return profile

    async def principal_keys(self, account_ids: Iterable[str]) -> dict[str, set[str]]:
        """Strings a role binding may name each account by: identity subjects, primary email, and groups."""
        wanted = list(dict.fromkeys(account_ids))
        keys: dict[str, set[str]] = {account_id: set() for account_id in wanted}
        if not wanted:
            return keys
        async with self.session_maker() as session:
            rows = await session.execute(
                select(
                    DBAccount.id, DBAccount.primary_email, DBAccountIdentity.subject, DBAccountIdentity.claims_snapshot
                )
                .outerjoin(
                    DBAccountIdentity,
                    (DBAccountIdentity.account_id == DBAccount.id) & (DBAccountIdentity.status == "active"),
                )
                .where(DBAccount.id.in_(wanted))
            )
            for account_id, email, subject, snapshot in rows:
                groups = (snapshot or {}).get("groups")
                keys[account_id].update(
                    key
                    for key in (email, subject, *(groups if isinstance(groups, list) else ()))
                    if isinstance(key, str) and key
                )
        return keys

    async def resolve(self, principals: Iterable[str]) -> list[ResolvedPrincipal]:
        """Map principal strings (subjects or emails) to public profile fields.

        A principal resolves only when exactly one user account matches it, either by an identity subject or by
        primary email. Deactivated accounts still resolve so history stays readable.
        """
        wanted = list(dict.fromkeys(p for p in principals if p and p != "*"))
        matches: dict[str, set[str]] = defaultdict(set)
        rows: dict[str, tuple[DBAccount, DBAccountProfile | None]] = {}
        if wanted:
            async with self.session_maker() as session:
                base = select(DBAccount, DBAccountProfile).outerjoin(
                    DBAccountProfile, DBAccountProfile.account_id == DBAccount.id
                )
                by_subject = await session.execute(
                    base.add_columns(DBAccountIdentity.subject)
                    .join(DBAccountIdentity, DBAccountIdentity.account_id == DBAccount.id)
                    .where(
                        DBAccount.type == "user",
                        DBAccountIdentity.subject.in_(wanted),
                        DBAccountIdentity.issuer.notin_(NON_USER_ISSUERS),
                    )
                )
                for account, profile, subject in by_subject:
                    matches[subject].add(account.id)
                    rows[account.id] = (account, profile)
                by_email = await session.execute(
                    base.where(DBAccount.type == "user", DBAccount.primary_email.in_(wanted))
                )
                for account, profile in by_email:
                    if account.primary_email is not None:
                        matches[account.primary_email].add(account.id)
                    rows[account.id] = (account, profile)

        results = []
        for principal in principals:
            account_ids = matches.get(principal, set())
            if len(account_ids) != 1:
                results.append(ResolvedPrincipal(principal=principal, ambiguous=len(account_ids) > 1))
                continue
            record = _record(*rows[next(iter(account_ids))])
            results.append(
                ResolvedPrincipal(
                    principal=principal,
                    account_id=record.account_id,
                    display_name=record.public_display_name(principal),
                    idp_display_name=record.idp_display_name,
                    display_name_is_override=record.display_name_override is not None,
                    avatar_url=record.avatar_url,
                    status=record.status,
                )
            )
        return results
