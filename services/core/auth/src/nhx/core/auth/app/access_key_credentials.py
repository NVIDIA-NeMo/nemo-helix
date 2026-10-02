# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Account-credential persistence adapter for Scoped Access Keys."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, Literal

from nemo_helix_plugin.auth.access_keys.types import AccessKeyEntityType
from nhx.common.auth.access_keys import SERVICE_ACCOUNT_PRINCIPAL_PREFIX
from nhx.core.auth.entities import AccessKeyEntity
from nhx.core.entities.app.repository import AccountCredentialRecord, AccountCredentialStore, AccountIdentityStore

ACCESS_KEY_CREDENTIAL_TYPE = "access_key"
AccessKeyCredentialStatus = Literal["ACTIVE", "REVOKED", "SUSPENDED", "ROTATING"]


class AccessKeyCredentialAdapter:
    """Map access-key lifecycle records onto the shared credential table."""

    def __init__(
        self,
        credential_store: AccountCredentialStore,
        identity_store: AccountIdentityStore,
    ) -> None:
        self.credential_store = credential_store
        self.identity_store = identity_store

    async def create(self, record: AccessKeyEntity, *, owner_account_id: str | None) -> AccessKeyEntity:
        if owner_account_id is None:
            raise RuntimeError("A stable owner account is required to create a Scoped Access Key")
        subject_account_id = owner_account_id
        if record.is_service_account():
            subject_principal = record.subject_principal or ""
            identity = await self.identity_store.resolve_or_materialize(
                issuer="nemo:service-account",
                subject=subject_principal.removeprefix(SERVICE_ACCOUNT_PRINCIPAL_PREFIX),
                subject_claim="service_account_id",
                account_type="service",
                display_name=subject_principal,
                claims_snapshot={"principal_id": subject_principal},
                linked_via="access_key_issuance",
                linked_by=owner_account_id,
            )
            subject_account_id = identity.account_id
        stored = await self.credential_store.create(
            credential_type=ACCESS_KEY_CREDENTIAL_TYPE,
            handle=record.name,
            owner_account_id=owner_account_id,
            subject_account_id=subject_account_id,
            status=AccessKeyCredentialAdapter._status(record.status),
            issued_at=record.issued_at,
            expires_at=record.expires_at,
            public_metadata=self._metadata(record),
        )
        return self._entity(stored)

    async def get(self, jti: str) -> AccessKeyEntity | None:
        record = await self.credential_store.get(ACCESS_KEY_CREDENTIAL_TYPE, jti)
        return self._entity(record) if record is not None else None

    async def get_active(self, jti: str) -> AccessKeyEntity | None:
        record = await self.credential_store.get_active(ACCESS_KEY_CREDENTIAL_TYPE, jti)
        return self._entity(record) if record is not None else None

    async def update(self, record: AccessKeyEntity) -> AccessKeyEntity:
        stored = await self.credential_store.get(ACCESS_KEY_CREDENTIAL_TYPE, record.name)
        if stored is None:
            raise KeyError(record.name)
        updated = await self.credential_store.update(
            replace(
                stored,
                status=record.status,
                expires_at=record.expires_at,
                last_used_at=record.last_used_at,
                public_metadata=self._metadata(record),
                db_version=record.db_version,
            )
        )
        return self._entity(updated)

    async def delete(self, jti: str) -> bool:
        return await self.credential_store.delete(ACCESS_KEY_CREDENTIAL_TYPE, jti)

    async def list_for_owner(
        self,
        owner_account_id: str,
        *,
        include_service_accounts: bool,
        offset: int,
        limit: int,
    ) -> tuple[list[AccessKeyEntity], bool]:
        records, has_more = await self.credential_store.list_for_owner(
            ACCESS_KEY_CREDENTIAL_TYPE,
            owner_account_id,
            include_service_subjects=include_service_accounts,
            offset=offset,
            limit=limit,
        )
        return [self._entity(record) for record in records], has_more

    @staticmethod
    def _metadata(record: AccessKeyEntity) -> dict[str, Any]:
        return {
            "jti": record.name,
            "key_name": record.key_name,
            "description": record.description,
            "owner_principal": record.principal,
            "subject_principal": record.subject_principal,
            "entity_type": record.entity_type,
            "issuer": record.issuer,
            "audiences": list(record.audiences),
            "scope": list(record.scope),
            "grace_period_expires_at": AccessKeyCredentialAdapter._serialize_datetime(record.grace_period_expires_at),
            "rotation_successor_jti": record.rotation_successor_jti,
        }

    @staticmethod
    def _entity(record: AccountCredentialRecord) -> AccessKeyEntity:
        metadata = record.public_metadata
        entity = AccessKeyEntity(
            name=str(metadata["jti"]),
            workspace="system",
            key_name=AccessKeyCredentialAdapter._optional_string(metadata.get("key_name")),
            description=AccessKeyCredentialAdapter._optional_string(metadata.get("description")),
            principal=str(metadata["owner_principal"]),
            subject_principal=AccessKeyCredentialAdapter._optional_string(metadata.get("subject_principal")),
            entity_type=AccessKeyCredentialAdapter._entity_type(metadata.get("entity_type")),
            issuer=str(metadata["issuer"]),
            audiences=AccessKeyCredentialAdapter._string_list(metadata.get("audiences")),
            scope=AccessKeyCredentialAdapter._string_list(metadata.get("scope")),
            issued_at=record.issued_at,
            expires_at=record.expires_at,
            last_used_at=record.last_used_at,
            status=AccessKeyCredentialAdapter._status(record.status),
            grace_period_expires_at=AccessKeyCredentialAdapter._parse_datetime(metadata.get("grace_period_expires_at")),
            rotation_successor_jti=AccessKeyCredentialAdapter._optional_string(metadata.get("rotation_successor_jti")),
        )
        entity._db_version = record.db_version
        return entity

    @staticmethod
    def _serialize_datetime(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        return datetime.fromisoformat(value) if isinstance(value, str) else None

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _string_list(value: object) -> list[str]:
        return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []

    @staticmethod
    def _entity_type(value: object) -> AccessKeyEntityType:
        if value == "USER":
            return "USER"
        if value == "SERVICE_ACCOUNT":
            return "SERVICE_ACCOUNT"
        raise ValueError("Stored access-key credential has an invalid entity type")

    @staticmethod
    def _status(value: object) -> AccessKeyCredentialStatus:
        if value == "ACTIVE":
            return "ACTIVE"
        if value == "REVOKED":
            return "REVOKED"
        if value == "SUSPENDED":
            return "SUSPENDED"
        if value == "ROTATING":
            return "ROTATING"
        raise ValueError("Stored access-key credential has an invalid status")
