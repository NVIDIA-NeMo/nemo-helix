# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth service entities."""

from datetime import datetime
from typing import Any, Literal, Self

from nemo_platform_plugin.auth.access_keys.types import AccessKeyEntityType
from nmp.common.auth.access_keys import SERVICE_ACCOUNT_PRINCIPAL_PREFIX
from nmp.common.auth.models import Principal
from nmp.common.entities import EntityBase
from pydantic import Field, model_validator


class RoleBindingEntity(EntityBase):
    """Role binding entity for authorization.

    Extends EntityBase which provides:
    - id: str (auto-generated UUID)
    - workspace: str (the workspace this binding grants access to)
    - created_at: datetime
    - updated_at: datetime

    The workspace field from EntityBase serves dual purpose:
    - It's the workspace where this entity is stored
    - It's also the workspace this role binding grants access to
    """

    __entity_type__ = "role_binding"

    principal: str
    role: str
    granted_by: str
    granted_at: datetime
    revoked_at: datetime | None = None


class AccessKeyEntity(EntityBase):
    """Persistent lifecycle record for a Scoped Access Key.

    The inherited ``name`` field stores the JWT ID (jti) as the stable entity key.
    ``key_name`` is the optional human-readable label supplied at creation time.
    """

    __entity_type__ = "access_key"

    key_name: str | None = None
    description: str | None = None
    # ``principal`` is the lifecycle owner. It remains the token subject for
    # user-bound keys and is the creating administrator for service-bound keys.
    principal: str
    subject_principal: str | None = None
    entity_type: AccessKeyEntityType = "USER"
    issuer: str
    audiences: list[str]
    scope: list[str] = Field(default_factory=list)
    issued_at: datetime
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    status: Literal["ACTIVE", "REVOKED", "SUSPENDED", "ROTATING"] = "ACTIVE"
    # Set when status == "ROTATING": the instant after which this rotated-out key is
    # treated as revoked. Unused for every other status.
    grace_period_expires_at: datetime | None = None
    # Set when status == "ROTATING": the jti of the successor key minted for this
    # rotation. Lets an ambiguous begin_rotation outcome be attributed to the request
    # that actually committed it, rather than to a differently-raced concurrent
    # request that happened to also see this key as ROTATING.
    rotation_successor_jti: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_revoked_at(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("entity_type") == cls.__entity_type__:
            data = dict(data)
            data.pop("entity_type")
        if isinstance(data, dict) and data.get("revoked_at") is not None:
            data = dict(data)
            data["status"] = "REVOKED"
        return data

    @property
    def is_service_account(self) -> bool:
        return self.entity_type == "SERVICE_ACCOUNT"

    @model_validator(mode="after")
    def _validate_identity_binding(self) -> Self:
        if self.is_service_account:
            if (
                self.subject_principal is None
                or self.subject_principal == SERVICE_ACCOUNT_PRINCIPAL_PREFIX
                or not self.subject_principal.startswith(SERVICE_ACCOUNT_PRINCIPAL_PREFIX)
            ):
                raise ValueError("service-account access keys require a service-account subject principal")
            if Principal(id=self.principal).is_service_identity():
                raise ValueError("service-account access keys require a human creator principal")
        elif self.subject_principal is not None:
            raise ValueError("user access keys cannot have a separate subject principal")
        return self
