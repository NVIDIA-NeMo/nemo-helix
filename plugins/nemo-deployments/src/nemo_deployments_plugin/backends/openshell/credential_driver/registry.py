# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist the driver handles that connect OpenShell provider slots to Helix secrets."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy import URL, Boolean, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import NullPool

from .tickets import RegistrationClaims, RegistrationDenied, RegistrationTickets

OWNED_HANDLE_PREFIX = "owned-v1-"
OwnedState = Literal["pending", "active", "deleting", "deleted"]


class ProviderOwner(BaseModel):
    """The gateway's provider identity and credential slot that own a driver handle."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_min_length=1)

    workspace: str
    provider: str
    provider_id: str
    credential_key: str


class Binding(BaseModel):
    """A driver handle, its provider owner and the signed scope used for secret lookup."""

    model_config = ConfigDict(frozen=True)

    handle: str
    owner: ProviderOwner
    claims: RegistrationClaims


class RegistryModel(DeclarativeBase):
    """Gateway-local metadata; never credential values."""


class BindingRecord(RegistryModel):
    __tablename__ = "bindings"

    ticket_id: Mapped[str] = mapped_column(Text, primary_key=True)
    ticket_digest: Mapped[str] = mapped_column(Text)
    handle: Mapped[str] = mapped_column(Text, unique=True)
    # Pydantic validates persisted owner and ticket scope at the read boundary.
    owner_json: Mapped[str] = mapped_column("owner", Text)
    claims_json: Mapped[str] = mapped_column("claims", Text)
    object_id: Mapped[str] = mapped_column(Text)
    existing_handle: Mapped[str] = mapped_column(Text)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    @property
    def owner(self) -> ProviderOwner:
        return ProviderOwner.model_validate_json(self.owner_json)

    @property
    def claims(self) -> RegistrationClaims:
        return RegistrationClaims.model_validate_json(self.claims_json)


class RevokedJob(RegistryModel):
    __tablename__ = "revoked_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)


class RegistryIdentity(RegistryModel):
    __tablename__ = "registry_identity"

    singleton: Mapped[int] = mapped_column(primary_key=True)
    gateway_id: Mapped[str] = mapped_column(Text)


class OwnedCredential(BaseModel):
    """An OpenShell-owned value stored in Secrets, with its durable cleanup state."""

    model_config = ConfigDict(frozen=True)

    handle: str
    owner: ProviderOwner
    workspace: str
    name: str
    value_digest: str
    state: OwnedState
    create_uncertain: bool


class OwnedCredentialRecord(RegistryModel):
    __tablename__ = "owned_credentials"

    operation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    handle: Mapped[str] = mapped_column(Text, unique=True)
    owner_json: Mapped[str] = mapped_column(Text)
    object_id: Mapped[str] = mapped_column(Text)
    existing_handle: Mapped[str] = mapped_column(Text)
    workspace: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text, unique=True)
    value_digest: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(Text)
    create_uncertain: Mapped[bool] = mapped_column(Boolean, default=False)


def require_private_directory(directory: Path) -> None:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Driver state and socket directories must be owned by this user with mode 0700.")


class BindingRegistry:
    """Store provider-to-secret associations behind opaque handles in local SQLite.

    The credential driver registers signed requests here and looks up their scopes
    when OpenShell asks for credentials. Matching registration retries reuse a
    handle; revoked records remain stored so retries cannot restore access.
    Secret values are fetched separately by the driver and are never stored here.
    """

    def __init__(self, path: Path, tickets: RegistrationTickets, *, clock: Callable[[], float] = time.time) -> None:
        require_private_directory(path.parent)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise ValueError("Binding database must be a private regular file.")
        finally:
            os.close(descriptor)
        self.path = path
        self.tickets = tickets
        self.clock = clock
        self._owned_key: bytes | None = None
        self._engine = create_engine(
            URL.create("sqlite+pysqlite", database=str(path)),
            connect_args={"timeout": 10, "isolation_level": None},
            poolclass=NullPool,
        )
        with self._session(write=True) as session:
            RegistryModel.metadata.create_all(session.connection())
            identity = session.get(RegistryIdentity, 1)
            if identity is None:
                session.add(RegistryIdentity(singleton=1, gateway_id=tickets.gateway_id))
            elif identity.gateway_id != tickets.gateway_id:
                raise ValueError("Binding database belongs to another gateway.")

    @contextmanager
    def _session(self, *, write: bool = False) -> Generator[Session, None, None]:
        with self._engine.begin() as connection:
            # Lock before reading on writes: concurrent ticket retries must
            # not both observe an absent registration and create two handles.
            # Explicit BEGIN also avoids sqlite3's legacy SELECT/DDL autocommit.
            connection.exec_driver_sql("BEGIN IMMEDIATE" if write else "BEGIN")
            with Session(connection, join_transaction_mode="rollback_only") as session, session.begin():
                yield session

    def _binding(self, session: Session, row: BindingRecord, owner: ProviderOwner) -> Binding:
        claims = row.claims
        if (
            row.owner != owner
            or row.revoked
            or (claims.workload is None and int(self.clock()) >= claims.binding_expires_at)
            or session.get(RevokedJob, claims.job_id) is not None
        ):
            raise RegistrationDenied("Credential binding is not active for this provider.")
        return Binding(handle=row.handle, owner=owner, claims=claims)

    def register(
        self,
        owner: ProviderOwner,
        ticket: SecretStr,
        *,
        object_id: str = "",
        existing_handle: str = "",
    ) -> str:
        claims = self.tickets.verify(ticket)
        if (claims.gateway_workspace, claims.provider, claims.credential_key) != (
            owner.workspace,
            owner.provider,
            owner.credential_key,
        ):
            raise RegistrationDenied("Registration ticket does not authorize this provider.")
        digest = hashlib.sha256(ticket.get_secret_value().encode()).hexdigest()
        with self._session(write=True) as session:
            row = session.get(BindingRecord, claims.ticket_id)
            if row is not None:
                binding = self._binding(session, row, owner)
                if (row.ticket_digest, row.object_id, row.existing_handle) != (
                    digest,
                    object_id,
                    existing_handle,
                ):
                    raise RegistrationDenied("Registration ticket has already been consumed.")
                return binding.handle
            if session.get(RevokedJob, claims.job_id) is not None:
                raise RegistrationDenied("Job bindings have been revoked.")
            if existing_handle:
                previous = session.scalar(select(BindingRecord).where(BindingRecord.handle == existing_handle))
                if previous is None or previous.owner != owner:
                    raise RegistrationDenied("Existing handle does not belong to this provider.")
            handle = secrets.token_urlsafe(32)
            session.add(
                BindingRecord(
                    ticket_id=claims.ticket_id,
                    ticket_digest=digest,
                    handle=handle,
                    owner_json=owner.model_dump_json(),
                    claims_json=claims.model_dump_json(),
                    object_id=object_id,
                    existing_handle=existing_handle,
                    revoked=False,
                )
            )
            # StoreCredential stages a new handle. Do not revoke the previous
            # handle before OpenShell's provider update commits successfully.
            return handle

    def resolve(self, owner: ProviderOwner, handle: str) -> Binding:
        """Find an owned, unrevoked binding; the driver checks live workload and creator access."""
        with self._session() as session:
            row = session.scalar(select(BindingRecord).where(BindingRecord.handle == handle))
            if row is None:
                raise RegistrationDenied("Unknown credential binding.")
            return self._binding(session, row, owner)

    def delete(self, owner: ProviderOwner, handle: str) -> None:
        """Revoke the handle while retaining its record; leave the Helix secret untouched."""
        with self._session(write=True) as session:
            row = session.scalar(select(BindingRecord).where(BindingRecord.handle == handle))
            if row is None:
                return
            if row.owner != owner:
                raise RegistrationDenied("Credential binding belongs to another provider.")
            row.revoked = True

    def revoke_job(self, job_id: str) -> None:
        with self._session(write=True) as session:
            if session.get(RevokedJob, job_id) is None:
                session.add(RevokedJob(job_id=job_id))

    def owned_digest(self, value: str) -> str:
        if self._owned_key is None:
            # The database lock also serializes first-time key-file creation.
            with self._session(write=True) as session:
                path = self.path.with_suffix(self.path.suffix + ".owned-key")
                if not path.exists() and session.scalar(select(OwnedCredentialRecord.handle).limit(1)) is not None:
                    raise ValueError("Owned credential digest key is missing for the existing registry.")
                try:
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(secrets.token_bytes(32))
                        stream.flush()
                        os.fsync(stream.fileno())
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_uid != os.getuid()
                        or stat.S_IMODE(info.st_mode) & 0o077
                    ):
                        raise ValueError("Owned credential digest key must be a private regular file.")
                    key = stream.read(33)
                if len(key) != 32:
                    raise ValueError("Invalid owned credential digest key.")
                self._owned_key = key
        return hmac.new(self._owned_key, value.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _owned(row: OwnedCredentialRecord) -> OwnedCredential:
        return OwnedCredential.model_validate(
            {
                "handle": row.handle,
                "owner": ProviderOwner.model_validate_json(row.owner_json),
                "workspace": row.workspace,
                "name": row.name,
                "value_digest": row.value_digest,
                "state": row.state,
                "create_uncertain": row.create_uncertain,
            }
        )

    @staticmethod
    def _owned_row(session: Session, owner: ProviderOwner, handle: str) -> OwnedCredentialRecord | None:
        if not handle.startswith(OWNED_HANDLE_PREFIX):
            raise RegistrationDenied("Expected an owned credential handle.")
        row = session.scalar(select(OwnedCredentialRecord).where(OwnedCredentialRecord.handle == handle))
        if row is not None and ProviderOwner.model_validate_json(row.owner_json) != owner:
            raise RegistrationDenied("Owned credential belongs to another owner.")
        return row

    @staticmethod
    def _require_owned_workspace(session: Session, workspace: str) -> None:
        if (
            session.scalar(
                select(OwnedCredentialRecord.handle).where(OwnedCredentialRecord.workspace != workspace).limit(1)
            )
            is not None
        ):
            raise RegistrationDenied("Configured owned-credential workspace differs from the existing registry.")

    def require_owned_workspace(self, workspace: str) -> None:
        """Keep persisted handles and cleanup intent in their original workspace."""
        with self._session() as session:
            self._require_owned_workspace(session, workspace)

    def reserve_owned(
        self,
        owner: ProviderOwner,
        value: SecretStr,
        *,
        workspace: str,
        object_id: str,
        existing_handle: str,
    ) -> OwnedCredential:
        digest = self.owned_digest(value.get_secret_value())
        operation_id = hashlib.sha256(
            json.dumps(
                [owner.model_dump(), object_id, existing_handle if not object_id else ""], sort_keys=True
            ).encode()
        ).hexdigest()
        with self._session(write=True) as session:
            self._require_owned_workspace(session, workspace)
            row = session.get(OwnedCredentialRecord, operation_id)
            if row is not None:
                if (
                    row.workspace != workspace
                    or row.existing_handle != existing_handle
                    or not hmac.compare_digest(row.value_digest, digest)
                    or row.state not in {"pending", "active"}
                ):
                    raise RegistrationDenied("Owned credential operation has already been consumed.")
                return self._owned(row)
            if existing_handle:
                previous = self._owned_row(session, owner, existing_handle)
                if previous is None or previous.state != "active" or previous.workspace != workspace:
                    raise RegistrationDenied("Existing owned credential is not active for this owner.")
            row = OwnedCredentialRecord(
                operation_id=operation_id,
                handle=OWNED_HANDLE_PREFIX + secrets.token_urlsafe(32),
                owner_json=owner.model_dump_json(),
                object_id=object_id,
                existing_handle=existing_handle,
                workspace=workspace,
                name="credential-" + secrets.token_hex(24),
                value_digest=digest,
                state="pending",
                create_uncertain=False,
            )
            session.add(row)
            return self._owned(row)

    def get_owned(self, owner: ProviderOwner, handle: str, *, active: bool = False) -> OwnedCredential:
        with self._session() as session:
            row = self._owned_row(session, owner, handle)
            if row is None or (active and row.state != "active"):
                raise RegistrationDenied("Owned credential is not active for this owner.")
            return self._owned(row)

    def begin_owned_create(self, owner: ProviderOwner, handle: str) -> bool:
        with self._session(write=True) as session:
            row = self._owned_row(session, owner, handle)
            if row is None or row.state != "pending":
                raise RegistrationDenied("Owned credential cannot be created.")
            was_uncertain = row.create_uncertain
            row.create_uncertain = True
            return was_uncertain

    def activate_owned(self, owner: ProviderOwner, handle: str, *, retain_uncertainty: bool) -> None:
        with self._session(write=True) as session:
            row = self._owned_row(session, owner, handle)
            if row is None or row.state not in {"pending", "active"}:
                raise RegistrationDenied("Owned credential cannot be activated.")
            row.state = "active"
            if not retain_uncertainty:
                row.create_uncertain = False

    def revoke_owned(self, owner: ProviderOwner, handle: str) -> OwnedCredential | None:
        with self._session(write=True) as session:
            row = self._owned_row(session, owner, handle)
            if row is None:
                return None
            if row.state != "deleted":
                row.state = "deleting"
            return self._owned(row)

    def finish_owned_delete(self, owner: ProviderOwner, handle: str) -> None:
        with self._session(write=True) as session:
            row = self._owned_row(session, owner, handle)
            if row is None or row.state not in {"deleting", "deleted"}:
                raise RegistrationDenied("Owned credential is not pending deletion.")
            # A timed-out POST can finish after DELETE reports 404. Only these
            # uncertain creations retain cleanup intent for subsequent sweeps.
            if not row.create_uncertain:
                row.state = "deleted"

    def pending_owned(self) -> list[OwnedCredential]:
        with self._session() as session:
            return [
                self._owned(row)
                for row in session.scalars(
                    select(OwnedCredentialRecord).where(OwnedCredentialRecord.state.in_(("pending", "deleting")))
                )
            ]
