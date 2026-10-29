# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist the driver handles that connect OpenShell provider slots to Helix secrets."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy import URL, Boolean, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import NullPool

from .tickets import RegistrationDenied, RegistrationTickets, TicketClaims


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
    claims: TicketClaims


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
    def claims(self) -> TicketClaims:
        return TicketClaims.model_validate_json(self.claims_json)


class RevokedJob(RegistryModel):
    __tablename__ = "revoked_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)


class RegistryIdentity(RegistryModel):
    __tablename__ = "registry_identity"

    singleton: Mapped[int] = mapped_column(primary_key=True)
    gateway_id: Mapped[str] = mapped_column(Text)


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
