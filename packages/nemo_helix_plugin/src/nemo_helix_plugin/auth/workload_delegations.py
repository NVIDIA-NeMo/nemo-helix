# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-safe workload delegation primitives for workload OBO token exchange."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar

from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.entities import SyncEntityClient
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from pydantic import BaseModel, ConfigDict, Field

SYSTEM_WORKSPACE = "system"
WORKLOAD_DELEGATION_ENTITY_TYPE = "workload_delegation"
JWT_WORKLOAD_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE = "urn:nvidia:nemo:params:oauth:token-type:docker-opaque-workload-proof"
OPAQUE_DOCKER_PROOF_PREFIX = "nhx_obo_v1"
KUBERNETES_POD_UID_REFERENCE_NAME = "authentication.kubernetes.io/pod-uid"

_OPAQUE_DOCKER_PROOF_SECRET_BYTES = 32
_OPAQUE_DOCKER_PROOF_HASH_PREFIX = "v1:sha256:"
_DELEGATION_HASH_LENGTH = 48


class WorkloadDelegationError(Exception):
    """Base exception for workload delegation failures."""


class WorkloadDelegationValidationError(WorkloadDelegationError):
    """Raised when a delegation record is malformed or unusable."""


class WorkloadDelegationConflictError(WorkloadDelegationError):
    """Raised when an active delegation already exists for the same name."""


class InvalidWorkloadProofTokenError(WorkloadDelegationError):
    """Raised when a Docker opaque workload proof token is malformed."""


@dataclass(frozen=True)
class ParsedOpaqueDockerProofToken:
    """Parsed Docker opaque workload proof token."""

    delegation_name: str
    secret: bytes


class WorkloadDelegationLookupScope(BaseModel):
    """Lookup scope for delegations associated with one workload instance.

    ``workload_instance_id`` is stored in ``data.workload_id`` for lifecycle lookups.
    ``workload_claim_id`` preserves the optional logical workload_id from reusable config.
    """

    model_config = ConfigDict(frozen=True)

    workload_workspace: str = Field(min_length=1)
    workload_kind: str | None = Field(default=None, min_length=1)
    workload_instance_id: str = Field(min_length=1)
    workload_claim_id: str | None = Field(default=None, min_length=1)


class WorkloadDelegationScope(WorkloadDelegationLookupScope):
    """Create-time scope for a delegation associated with one workload instance."""

    workload_kind: str = Field(min_length=1)


class WorkloadDelegationEntity(NemoEntity, entity_type=WORKLOAD_DELEGATION_ENTITY_TYPE):
    """Auth-owned delegation record used to mint delegated workload tokens."""

    __entity_type__: ClassVar[str] = WORKLOAD_DELEGATION_ENTITY_TYPE

    workload_subject: str
    workload_audience: str
    workload_workspace: str
    workload_kind: str | None = None
    workload_id: str | None = None
    workload_claim_id: str | None = None
    workload_generation: str | None = None
    job_id: str | None = None
    attempt_id: str | None = None
    step_id: str | None = None
    auth_context: AuthContext
    bound_reference_name: str | None = None
    bound_reference_value: str | None = None
    opaque_subject_token_hash: str | None = None
    expires_at: datetime
    revoked_at: datetime | None = None

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Return whether this delegation is expired at the supplied time."""
        effective_now = as_aware_utc(now or datetime.now(timezone.utc))
        return as_aware_utc(self.expires_at) <= effective_now

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether this delegation can still be used."""
        return self.revoked_at is None and not self.is_expired(now=now)


def docker_delegation_name(*, workload_workspace: str, job_id: str, attempt_id: str, step_id: str) -> str:
    """Return the deterministic Docker delegation entity name."""
    return _delegation_hash_name(
        "job",
        [
            _require_non_empty(workload_workspace, "workload_workspace"),
            _require_non_empty(job_id, "job_id"),
            _require_non_empty(attempt_id, "attempt_id"),
            _require_non_empty(step_id, "step_id"),
        ],
    )


def docker_workload_delegation_name(
    *,
    scope: WorkloadDelegationScope,
    workload_generation: str,
) -> str:
    """Return the deterministic Docker opaque-proof delegation name for any workload type."""
    return _delegation_hash_name(
        "docker",
        [
            scope.workload_workspace,
            scope.workload_kind,
            scope.workload_instance_id,
            _require_non_empty(workload_generation, "workload_generation"),
        ],
    )


def docker_deployment_delegation_name(
    *,
    workload_workspace: str,
    deployment_id: str,
    container_name: str,
) -> str:
    """Return the deterministic Docker delegation name for a deployment container."""
    return docker_workload_delegation_name(
        scope=WorkloadDelegationScope(
            workload_workspace=workload_workspace,
            workload_kind="deployment",
            workload_instance_id=deployment_id,
        ),
        workload_generation=container_name,
    )


def _scope_from_delegation_entity(entity: WorkloadDelegationEntity) -> WorkloadDelegationScope:
    return WorkloadDelegationScope(
        workload_workspace=entity.workload_workspace,
        workload_kind=_require_non_empty(entity.workload_kind, "workload_kind"),
        workload_instance_id=_require_non_empty(entity.workload_id, "workload_id"),
        workload_claim_id=entity.workload_claim_id,
    )


def reference_delegation_name(
    *,
    workload_audience: str,
    workload_subject: str,
    bound_reference_name: str,
    bound_reference_value: str,
) -> str:
    """Return the deterministic verified-reference delegation entity name."""
    return _delegation_hash_name(
        "ref",
        [
            _require_non_empty(workload_audience, "workload_audience"),
            _require_non_empty(workload_subject, "workload_subject"),
            _require_non_empty(bound_reference_name, "bound_reference_name"),
            _require_non_empty(bound_reference_value, "bound_reference_value"),
        ],
    )


def kubernetes_pod_uid_delegation_name(
    *,
    workload_audience: str,
    workload_subject: str,
    pod_uid: str,
) -> str:
    """Return the deterministic delegation name for a Kubernetes Pod UID-bound workload."""
    return reference_delegation_name(
        workload_audience=workload_audience,
        workload_subject=workload_subject,
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value=pod_uid,
    )


def create_opaque_docker_proof_token(delegation_name: str) -> tuple[str, str]:
    """Create an opaque Docker workload proof token and its stored secret hash."""
    secret = secrets.token_bytes(_OPAQUE_DOCKER_PROOF_SECRET_BYTES)
    token = ".".join(
        [
            OPAQUE_DOCKER_PROOF_PREFIX,
            _b64url_encode(_require_non_empty(delegation_name, "delegation_name").encode("utf-8")),
            _b64url_encode(secret),
        ]
    )
    return token, _opaque_docker_proof_token_hash(secret)


def parse_opaque_docker_proof_token(token: str) -> ParsedOpaqueDockerProofToken:
    """Parse a Docker opaque workload proof token envelope."""
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != OPAQUE_DOCKER_PROOF_PREFIX:
        raise InvalidWorkloadProofTokenError("invalid Docker opaque workload proof token envelope")

    try:
        delegation_name = _b64url_decode(parts[1]).decode("utf-8")
        secret = _b64url_decode(parts[2])
    except (UnicodeDecodeError, ValueError) as exc:
        raise InvalidWorkloadProofTokenError("invalid Docker opaque workload proof token encoding") from exc

    if not delegation_name:
        raise InvalidWorkloadProofTokenError("Docker opaque workload proof token is missing a delegation name")
    if len(secret) != _OPAQUE_DOCKER_PROOF_SECRET_BYTES:
        raise InvalidWorkloadProofTokenError("Docker opaque workload proof token secret has invalid length")

    return ParsedOpaqueDockerProofToken(delegation_name=delegation_name, secret=secret)


def verify_opaque_docker_proof_token_hash(secret: bytes, expected_hash: str) -> bool:
    """Constant-time check for a parsed Docker opaque proof token secret."""
    if not expected_hash.startswith(_OPAQUE_DOCKER_PROOF_HASH_PREFIX):
        return False
    return hmac.compare_digest(_opaque_docker_proof_token_hash(secret), expected_hash)


def subject_token_type_for_exchange(subject_token: str) -> str:
    """Return the RFC 8693 subject_token_type for a workload identity subject token."""
    if subject_token.startswith(f"{OPAQUE_DOCKER_PROOF_PREFIX}."):
        return DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE
    return JWT_WORKLOAD_SUBJECT_TOKEN_TYPE


class WorkloadDelegationStore:
    """Identity-based store wrapper for plugin-managed workload delegation records."""

    def __init__(self, entity_client: NemoEntitiesClient):
        self._entity_client = entity_client

    async def register(
        self,
        entity: WorkloadDelegationEntity,
        *,
        expected_db_version: int | None = None,
        require_opaque_subject_token_hash: bool = False,
    ) -> WorkloadDelegationEntity:
        """Create a delegation row, or replace an expired/revoked row with the same name."""
        _validate_delegation(entity, require_opaque_subject_token_hash=require_opaque_subject_token_hash)

        try:
            return await self._entity_client.add(entity)
        except NemoEntityConflictError as exc:
            existing = await self.get(entity.name)
            replacement = _replacement_after_register_conflict(
                entity,
                existing,
                expected_db_version=expected_db_version,
                conflict=exc,
            )
            return await self._entity_client.update(replacement)

    async def get(self, name: str) -> WorkloadDelegationEntity | None:
        """Fetch one delegation row by its deterministic name."""
        try:
            return await self._entity_client.get(WorkloadDelegationEntity, name, workspace=SYSTEM_WORKSPACE)
        except NemoEntityNotFoundError:
            return None

    async def update(
        self,
        entity: WorkloadDelegationEntity,
        *,
        expected_db_version: int | None = None,
    ) -> WorkloadDelegationEntity:
        """Update a delegation row using the db_version from the latest fetched row."""
        _validate_delegation(entity)
        existing = await self.get(entity.name)
        if existing is None:
            raise NemoEntityNotFoundError(f"Entity '{entity.name}' not found in workspace '{SYSTEM_WORKSPACE}'")
        replacement = _replacement_for_update(entity, existing, expected_db_version=expected_db_version)
        return await self._entity_client.update(replacement)

    async def revoke(self, name: str, *, now: datetime | None = None) -> WorkloadDelegationEntity | None:
        """Soft-revoke a delegation row by setting revoked_at."""
        existing = await self.get(name)
        if existing is None:
            return None
        return await self._entity_client.update(_mark_revoked(existing, now=now))

    async def list_by_workload(
        self,
        scope: WorkloadDelegationLookupScope,
        *,
        page_size: int = 100,
    ) -> list[WorkloadDelegationEntity]:
        """List delegation rows associated with one physical workload scope."""
        return await _list_by_workload_async(
            self._entity_client,
            scope=scope,
            page_size=page_size,
        )

    async def revoke_by_workload(
        self,
        scope: WorkloadDelegationLookupScope,
        *,
        now: datetime | None = None,
        page_size: int = 100,
    ) -> list[WorkloadDelegationEntity]:
        """Soft-revoke every delegation row associated with one physical workload scope."""
        entities = await self.list_by_workload(
            scope,
            page_size=page_size,
        )
        return [await self._entity_client.update(_mark_revoked(entity, now=now)) for entity in entities]


class SyncWorkloadDelegationStore:
    """Synchronous store wrapper for workload delegation records."""

    def __init__(self, entity_client: SyncEntityClient):
        self._entity_client = entity_client

    def register(
        self,
        entity: WorkloadDelegationEntity,
        *,
        expected_db_version: int | None = None,
        require_opaque_subject_token_hash: bool = False,
    ) -> WorkloadDelegationEntity:
        """Create a delegation row, or replace an expired/revoked row with the same name."""
        _validate_delegation(entity, require_opaque_subject_token_hash=require_opaque_subject_token_hash)

        try:
            return self._entity_client.add(entity)
        except NemoEntityConflictError as exc:
            existing = self.get(entity.name)
            replacement = _replacement_after_register_conflict(
                entity,
                existing,
                expected_db_version=expected_db_version,
                conflict=exc,
            )
            return self._entity_client.update(replacement)

    def get(self, name: str) -> WorkloadDelegationEntity | None:
        """Fetch one delegation row by its deterministic name."""
        try:
            return self._entity_client.get(WorkloadDelegationEntity, name, workspace=SYSTEM_WORKSPACE)
        except NemoEntityNotFoundError:
            return None

    def update(
        self,
        entity: WorkloadDelegationEntity,
        *,
        expected_db_version: int | None = None,
    ) -> WorkloadDelegationEntity:
        """Update a delegation row using the db_version from the latest fetched row."""
        _validate_delegation(entity)
        existing = self.get(entity.name)
        if existing is None:
            raise NemoEntityNotFoundError(f"Entity '{entity.name}' not found in workspace '{SYSTEM_WORKSPACE}'")
        replacement = _replacement_for_update(entity, existing, expected_db_version=expected_db_version)
        return self._entity_client.update(replacement)

    def revoke(self, name: str, *, now: datetime | None = None) -> WorkloadDelegationEntity | None:
        """Soft-revoke a delegation row by setting revoked_at."""
        existing = self.get(name)
        if existing is None:
            return None
        return self._entity_client.update(_mark_revoked(existing, now=now))

    def list_by_workload(
        self,
        scope: WorkloadDelegationLookupScope,
        *,
        page_size: int = 100,
    ) -> list[WorkloadDelegationEntity]:
        """List delegation rows associated with one physical workload scope."""
        return _list_by_workload_sync(
            self._entity_client,
            scope=scope,
            page_size=page_size,
        )

    def revoke_by_workload(
        self,
        scope: WorkloadDelegationLookupScope,
        *,
        now: datetime | None = None,
        page_size: int = 100,
    ) -> list[WorkloadDelegationEntity]:
        """Soft-revoke every delegation row associated with one physical workload scope."""
        entities = self.list_by_workload(
            scope,
            page_size=page_size,
        )
        return [self._entity_client.update(_mark_revoked(entity, now=now)) for entity in entities]


async def _list_by_workload_async(
    entity_client: NemoEntitiesClient,
    *,
    scope: WorkloadDelegationLookupScope,
    page_size: int,
) -> list[WorkloadDelegationEntity]:
    filter_operation = _workload_filter(scope)
    page = 1
    entities: list[WorkloadDelegationEntity] = []
    while True:
        response = await entity_client.list(
            WorkloadDelegationEntity,
            workspace=SYSTEM_WORKSPACE,
            filter_operation=filter_operation,
            page=page,
            page_size=page_size,
        )
        entities.extend(response.data)
        if response.pagination.page >= response.pagination.total_pages:
            return entities
        page += 1


def _list_by_workload_sync(
    entity_client: SyncEntityClient,
    *,
    scope: WorkloadDelegationLookupScope,
    page_size: int,
) -> list[WorkloadDelegationEntity]:
    filter_operation = _workload_filter(scope)
    page = 1
    entities: list[WorkloadDelegationEntity] = []
    while True:
        response = entity_client.list(
            WorkloadDelegationEntity,
            workspace=SYSTEM_WORKSPACE,
            filter_operation=filter_operation,
            page=page,
            page_size=page_size,
        )
        entities.extend(response.data)
        if response.pagination.page >= response.pagination.total_pages:
            return entities
        page += 1


def _copy_delegation_payload(
    target: WorkloadDelegationEntity,
    source: WorkloadDelegationEntity,
) -> WorkloadDelegationEntity:
    target.name = source.name
    target.workspace = SYSTEM_WORKSPACE
    target.project = source.project
    target.workload_subject = source.workload_subject
    target.workload_audience = source.workload_audience
    target.workload_workspace = source.workload_workspace
    target.workload_kind = source.workload_kind
    target.workload_id = source.workload_id
    target.workload_claim_id = source.workload_claim_id
    target.workload_generation = source.workload_generation
    target.job_id = source.job_id
    target.attempt_id = source.attempt_id
    target.step_id = source.step_id
    target.auth_context = source.auth_context
    target.bound_reference_name = source.bound_reference_name
    target.bound_reference_value = source.bound_reference_value
    target.opaque_subject_token_hash = source.opaque_subject_token_hash
    target.expires_at = source.expires_at
    target.revoked_at = source.revoked_at
    return target


def _replacement_after_register_conflict(
    entity: WorkloadDelegationEntity,
    existing: WorkloadDelegationEntity | None,
    *,
    expected_db_version: int | None,
    conflict: Exception,
) -> WorkloadDelegationEntity:
    if existing is None:
        raise WorkloadDelegationConflictError(f"delegation '{entity.name}' already exists") from conflict
    if existing.is_active():
        raise WorkloadDelegationConflictError(f"active delegation '{entity.name}' already exists") from conflict
    return _replacement_for_update(
        entity,
        existing,
        expected_db_version=expected_db_version,
        conflict=conflict,
        allow_revoked_replacement=True,
    )


def _replacement_for_update(
    entity: WorkloadDelegationEntity,
    existing: WorkloadDelegationEntity,
    *,
    expected_db_version: int | None,
    conflict: Exception | None = None,
    allow_revoked_replacement: bool = False,
) -> WorkloadDelegationEntity:
    if expected_db_version is not None and existing.db_version != expected_db_version:
        error = WorkloadDelegationConflictError(
            f"delegation '{entity.name}' db_version {existing.db_version} does not match "
            f"expected db_version {expected_db_version}"
        )
        if conflict is not None:
            raise error from conflict
        raise error
    if existing.revoked_at is not None and entity.revoked_at is None and not allow_revoked_replacement:
        error = WorkloadDelegationConflictError(f"revoked delegation '{entity.name}' cannot be reactivated by update")
        if conflict is not None:
            raise error from conflict
        raise error
    return _copy_delegation_payload(existing, entity)


def _validate_delegation(
    entity: WorkloadDelegationEntity,
    *,
    require_opaque_subject_token_hash: bool = False,
) -> None:
    if entity.workspace != SYSTEM_WORKSPACE:
        raise WorkloadDelegationValidationError("workload delegations must be stored in the system workspace")

    _require_non_empty(entity.name, "name")
    _require_non_empty(entity.workload_subject, "workload_subject")
    _require_non_empty(entity.workload_audience, "workload_audience")
    _require_non_empty(entity.workload_workspace, "workload_workspace")
    _validate_workload_identity_metadata(entity)

    if entity.is_expired():
        raise WorkloadDelegationValidationError("workload delegation expires_at must be in the future")

    has_reference_name = bool(entity.bound_reference_name)
    has_reference_value = bool(entity.bound_reference_value)
    if has_reference_name != has_reference_value:
        raise WorkloadDelegationValidationError(
            "bound_reference_name and bound_reference_value must be provided together"
        )

    if _is_docker_delegation(entity) and (has_reference_name or has_reference_value):
        raise WorkloadDelegationValidationError("Docker workload delegations cannot use bound references")

    if entity.opaque_subject_token_hash and (has_reference_name or has_reference_value):
        raise WorkloadDelegationValidationError("opaque Docker workload delegations cannot also use bound references")

    expected_name = _expected_delegation_name(entity)
    if expected_name is None:
        raise WorkloadDelegationValidationError(
            "workload delegations must use a Docker or verified-reference lookup key"
        )
    if entity.name != expected_name:
        raise WorkloadDelegationValidationError("workload delegation name does not match its canonical lookup key")

    if require_opaque_subject_token_hash and not entity.opaque_subject_token_hash:
        raise WorkloadDelegationValidationError("opaque Docker workload delegations require a stored token hash")


def _expected_delegation_name(entity: WorkloadDelegationEntity) -> str | None:
    if _is_docker_delegation(entity):
        return docker_delegation_name(
            workload_workspace=entity.workload_workspace,
            job_id=entity.job_id or "",
            attempt_id=entity.attempt_id or "",
            step_id=entity.step_id or "",
        )

    if _is_generic_docker_delegation(entity):
        return docker_workload_delegation_name(
            scope=_scope_from_delegation_entity(entity),
            workload_generation=_require_non_empty(entity.workload_generation, "workload_generation"),
        )

    if entity.bound_reference_name and entity.bound_reference_value:
        return reference_delegation_name(
            workload_audience=entity.workload_audience,
            workload_subject=entity.workload_subject,
            bound_reference_name=entity.bound_reference_name,
            bound_reference_value=entity.bound_reference_value,
        )

    return None


def _is_docker_delegation(entity: WorkloadDelegationEntity) -> bool:
    return (
        entity.name.startswith("job-")
        and entity.workload_subject == entity.name
        and bool(entity.job_id)
        and bool(entity.attempt_id)
        and bool(entity.step_id)
    )


def _is_generic_docker_delegation(entity: WorkloadDelegationEntity) -> bool:
    return (
        entity.name.startswith("docker-")
        and entity.workload_subject == entity.name
        and bool(entity.opaque_subject_token_hash)
    )


def _validate_workload_identity_metadata(entity: WorkloadDelegationEntity) -> None:
    generic_fields = (entity.workload_kind, entity.workload_id, entity.workload_generation)
    legacy_job_fields = (entity.job_id, entity.attempt_id, entity.step_id)

    if any(generic_fields):
        _require_non_empty(entity.workload_kind, "workload_kind")
        _require_non_empty(entity.workload_id, "workload_id")
        if entity.workload_claim_id is not None:
            _require_non_empty(entity.workload_claim_id, "workload_claim_id")
        _require_non_empty(entity.workload_generation, "workload_generation")
        return

    if all(legacy_job_fields):
        return

    raise WorkloadDelegationValidationError(
        "workload delegations must include workload_kind/workload_id/workload_generation "
        "or legacy job_id/attempt_id/step_id metadata"
    )


def _workload_filter(scope: WorkloadDelegationLookupScope) -> FilterOperation:
    operations: list[FilterOperation] = [
        ComparisonOperation(
            operator=FilterOperator.EQ,
            field="data.workload_workspace",
            value=scope.workload_workspace,
        ),
        ComparisonOperation(
            operator=FilterOperator.EQ,
            field="data.workload_id",
            value=scope.workload_instance_id,
        ),
    ]
    if scope.workload_kind is not None:
        operations.insert(
            1,
            ComparisonOperation(
                operator=FilterOperator.EQ,
                field="data.workload_kind",
                value=scope.workload_kind,
            ),
        )
    return LogicalOperation(
        operator=FilterOperator.AND,
        operations=operations,
    )


def _mark_revoked(entity: WorkloadDelegationEntity, *, now: datetime | None) -> WorkloadDelegationEntity:
    entity.revoked_at = as_aware_utc(now or datetime.now(timezone.utc))
    return entity


def _opaque_docker_proof_token_hash(secret: bytes) -> str:
    digest = hashlib.sha256(secret).digest()
    return _OPAQUE_DOCKER_PROOF_HASH_PREFIX + _b64url_encode(digest)


def _require_non_empty(value: str | None, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkloadDelegationValidationError(f"{field_name} must be a non-empty string")
    return value


def _delegation_hash_name(prefix: str, values: list[str]) -> str:
    canonical = json.dumps(values, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:_DELEGATION_HASH_LENGTH]}"


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    if not value:
        raise ValueError("empty base64url value")
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def as_aware_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime, treating naive values as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
