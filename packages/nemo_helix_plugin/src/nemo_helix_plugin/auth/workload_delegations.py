# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-safe workload delegation primitives for workload OBO token exchange."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import ClassVar

from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.entities import SyncEntityClient
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.entity_client import NemoEntitiesClient, NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

SYSTEM_WORKSPACE = "system"
WORKLOAD_DELEGATION_ENTITY_TYPE = "workload_delegation"
JWT_WORKLOAD_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE = "urn:nvidia:nemo:params:oauth:token-type:docker-opaque-workload-proof"
OPAQUE_DOCKER_PROOF_PREFIX = "nhx_obo_v1"
KUBERNETES_POD_UID_REFERENCE_NAME = "authentication.kubernetes.io/pod-uid"

_OPAQUE_DOCKER_PROOF_SECRET_BYTES = 32
_OPAQUE_DOCKER_PROOF_HASH_PREFIX = "v1:sha256:"
_DELEGATION_HASH_LENGTH = 48
_REVOCATION_DELETE_ATTEMPTS = 3
# The ``workload_kind`` the Jobs backends give a job step's delegation.
_JOB_WORKLOAD_KIND = "job"


class WorkloadDelegationError(Exception):
    """Base exception for workload delegation failures."""


class WorkloadDelegationValidationError(WorkloadDelegationError):
    """Raised when a delegation record is malformed or unusable."""


class WorkloadDelegationConflictError(WorkloadDelegationError):
    """Raised when registration or mutation conflicts with stored delegation state."""


class WorkloadDelegationAlreadyActiveError(WorkloadDelegationConflictError):
    """Raised only when registration finds an active delegation at its lookup name."""


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
    """Auth-owned delegation record used to mint delegated workload tokens.

    Docker job controllers bind ``docker_container_id`` once before startup.
    Cleanup verifies that association against Docker's actual container ID;
    token exchange continues to require the opaque secret proof.
    """

    __entity_type__: ClassVar[str] = WORKLOAD_DELEGATION_ENTITY_TYPE

    workload_subject: str
    workload_audience: str
    workload_workspace: str
    workload_kind: str | None = None
    workload_id: str | None = None
    workload_claim_id: str | None = None
    workload_generation: str | None = None
    job_id: str | None = None
    # Entity ID of the job (HelixJob.id). ``job_id`` above is the job's NAME as the Jobs backends write it; a
    # later job can reuse a name, never an ID. None for non-job workloads, for records written before this
    # field existed, and when the writer was not given the ID: "unknown", never a match. Compare it with
    # ``is_step_of_job``, not with ``==``.
    job_uid: str | None = None
    attempt_id: str | None = None
    step_id: str | None = None
    auth_context: AuthContext
    bound_reference_name: str | None = None
    bound_reference_value: str | None = None
    docker_container_id: str | None = Field(default=None, min_length=1)
    opaque_subject_token_hash: str | None = None
    expires_at: datetime
    # Existing stored markers remain authoritative; new revocations delete the record.
    revoked_at: datetime | None = None

    @field_validator("bound_reference_name", "bound_reference_value", "job_uid")
    @classmethod
    def normalize_empty_reference(cls, value: str | None) -> str | None:
        return value or None

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Return whether this delegation is expired at the supplied time."""
        effective_now = as_aware_utc(now or datetime.now(timezone.utc))
        return as_aware_utc(self.expires_at) <= effective_now

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether this delegation can still be used."""
        return self.revoked_at is None and not self.is_expired(now=now)

    def is_step_of_job(self, job_uid: str, *, now: datetime | None = None) -> bool:
        """Whether this record stands and was written for a step of the job whose entity ID is ``job_uid``.

        False when either ID is missing or empty, so a record written before ``job_uid`` existed never matches a
        caller that has no job ID either. It says the step belongs to some attempt of that job. It does not say
        who created the job or what the job runs. "Stands" means not revoked and not past ``expires_at``; it does
        not mean the step is still running.

        This is what the record states. The record is an ordinary entity in the system workspace, so the answer
        is as trustworthy as the principals that may write entities there.
        """
        return (
            self.is_active(now=now)
            and self.workload_kind == _JOB_WORKLOAD_KIND
            and bool(job_uid)
            and bool(self.job_uid)
            and self.job_uid == job_uid
        )

    def is_step_of_attempt(self, job_uid: str, attempt_id: str, *, now: datetime | None = None) -> bool:
        """``is_step_of_job``, and the step belongs to the attempt ``attempt_id`` of that job."""
        return (
            self.is_step_of_job(job_uid, now=now)
            and bool(attempt_id)
            and bool(self.attempt_id)
            and self.attempt_id == attempt_id
        )


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
    """Delegation lifecycle over the existing asynchronous entity client."""

    def __init__(self, entity_client: NemoEntitiesClient):
        self._entity_client = entity_client

    async def register(
        self, entity: WorkloadDelegationEntity, *, require_opaque_subject_token_hash: bool = False
    ) -> WorkloadDelegationEntity:
        """Create a fresh grant, deleting an inactive record if its lookup name is occupied."""
        _validate_delegation(entity, require_opaque_subject_token_hash=require_opaque_subject_token_hash)
        try:
            return await self._entity_client.add(entity)
        except NemoEntityConflictError as exc:
            existing = await self.get(entity.name)
            inactive = _inactive_after_register_conflict(entity, existing, conflict=exc)
            await self._delete(inactive, reason="revoked" if inactive.revoked_at is not None else "expired")
            try:
                saved = await self._entity_client.add(entity)
            except NemoEntityConflictError as conflict:
                raise WorkloadDelegationConflictError(f"delegation '{entity.name}' registration raced") from conflict
            _log_replacement_delegation(inactive, saved)
            return saved

    async def get(self, name: str) -> WorkloadDelegationEntity | None:
        """Fetch the current delegation at a deterministic lookup name."""
        try:
            return await self._entity_client.get(WorkloadDelegationEntity, name, workspace=SYSTEM_WORKSPACE)
        except NemoEntityNotFoundError:
            return None

    async def get_by_id(self, delegation_id: str) -> WorkloadDelegationEntity | None:
        """Fetch an exact grant without following a reusable lookup name."""
        try:
            entity = await self._entity_client.get_by_id(WorkloadDelegationEntity, delegation_id)
        except NemoEntityNotFoundError:
            return None
        return entity if entity.workspace == SYSTEM_WORKSPACE else None

    async def renew(
        self,
        delegation_id: str,
        *,
        expected_db_version: int,
        expires_at: datetime,
        opaque_subject_token_hash: str | None = None,
    ) -> WorkloadDelegationEntity:
        """Renew expiry or rotate an existing Docker proof without changing the grant binding.

        Trusted reconciliation may renew an expired record to recover a live deployment.
        Deleted records and legacy revoked records cannot be renewed.
        """
        existing = await self.get_by_id(delegation_id)
        if existing is None:
            raise WorkloadDelegationConflictError("workload delegation no longer exists")
        renewed = _renewal(
            existing,
            expected_db_version=expected_db_version,
            expires_at=expires_at,
            opaque_subject_token_hash=opaque_subject_token_hash,
        )
        try:
            return await self._entity_client.update(renewed)
        except (NemoEntityConflictError, NemoEntityNotFoundError) as exc:
            raise WorkloadDelegationConflictError("workload delegation changed during renewal") from exc

    async def bind_docker_container(self, delegation_id: str, container_id: str) -> WorkloadDelegationEntity:
        """Persist a write-once container association before Docker workload startup."""
        existing = await self.get_by_id(delegation_id)
        if existing is None:
            raise NemoEntityNotFoundError("Docker workload delegation not found")
        bound = _bind_docker_container(existing, container_id)
        if existing.docker_container_id == container_id:
            return existing
        return await self._entity_client.update(bound)

    async def revoke_for_docker_container(self, delegation_id: str, container_id: str) -> None:
        """Delete a grant only if its stored container association matches Docker's actual ID."""
        existing = await self.get_by_id(delegation_id)
        if existing is not None:
            _check_docker_container(existing, container_id)
            await self._revoke(existing)

    async def revoke(self, name: str) -> None:
        """Delete the current grant at this lookup name."""
        existing = await self.get(name)
        if existing is not None:
            await self._revoke(existing)

    async def revoke_by_id(self, delegation_id: str) -> None:
        """Delete one exact grant without selecting a replacement by name."""
        existing = await self.get_by_id(delegation_id)
        if existing is not None:
            await self._revoke(existing)

    async def list_by_workload(
        self, scope: WorkloadDelegationLookupScope, *, page_size: int = 100
    ) -> list[WorkloadDelegationEntity]:
        """List delegation records for one physical workload scope."""
        return await _list_by_workload_async(self._entity_client, scope=scope, page_size=page_size)

    async def revoke_by_workload(self, scope: WorkloadDelegationLookupScope, *, page_size: int = 100) -> None:
        """Delete the exact records selected for this workload scope."""
        failures: list[Exception] = []
        for entity in await self.list_by_workload(scope, page_size=page_size):
            try:
                await self._revoke(entity)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise WorkloadDelegationError("Some workload delegations could not be revoked") from ExceptionGroup(
                "Workload delegation revocation failures", failures
            )

    async def _revoke(self, entity: WorkloadDelegationEntity) -> None:
        for attempt in range(_REVOCATION_DELETE_ATTEMPTS):
            try:
                await self._delete(entity, reason="revoked")
                return
            except WorkloadDelegationConflictError:
                if attempt == _REVOCATION_DELETE_ATTEMPTS - 1:
                    raise
                current = await self.get_by_id(entity.id)
                if current is None:
                    return
                if current.docker_container_id != entity.docker_container_id:
                    raise WorkloadDelegationConflictError("Docker binding changed during revocation") from None
                entity = current

    async def _delete(self, entity: WorkloadDelegationEntity, *, reason: str) -> None:
        if not entity.id:
            raise WorkloadDelegationValidationError("deleting a delegation requires its persisted ID")
        try:
            await self._entity_client.delete_by_id(
                WorkloadDelegationEntity, entity.id, expected_db_version=entity.db_version
            )
        except NemoEntityNotFoundError:
            return
        except NemoEntityConflictError as exc:
            raise WorkloadDelegationConflictError("workload delegation changed before deletion") from exc
        _log_deleted_delegation(entity, reason=reason)


class SyncWorkloadDelegationStore:
    """Delegation lifecycle over the existing synchronous entity client."""

    def __init__(self, entity_client: SyncEntityClient):
        self._entity_client = entity_client

    def register(
        self, entity: WorkloadDelegationEntity, *, require_opaque_subject_token_hash: bool = False
    ) -> WorkloadDelegationEntity:
        """Create a fresh grant, deleting an inactive record if its lookup name is occupied."""
        _validate_delegation(entity, require_opaque_subject_token_hash=require_opaque_subject_token_hash)
        try:
            return self._entity_client.add(entity)
        except NemoEntityConflictError as exc:
            existing = self.get(entity.name)
            inactive = _inactive_after_register_conflict(entity, existing, conflict=exc)
            self._delete(inactive, reason="revoked" if inactive.revoked_at is not None else "expired")
            try:
                saved = self._entity_client.add(entity)
            except NemoEntityConflictError as conflict:
                raise WorkloadDelegationConflictError(f"delegation '{entity.name}' registration raced") from conflict
            _log_replacement_delegation(inactive, saved)
            return saved

    def get(self, name: str) -> WorkloadDelegationEntity | None:
        """Fetch the current delegation at a deterministic lookup name."""
        try:
            return self._entity_client.get(WorkloadDelegationEntity, name, workspace=SYSTEM_WORKSPACE)
        except NemoEntityNotFoundError:
            return None

    def get_by_id(self, delegation_id: str) -> WorkloadDelegationEntity | None:
        """Fetch an exact grant without following a reusable lookup name."""
        try:
            entity = self._entity_client.get_by_id(WorkloadDelegationEntity, delegation_id)
        except NemoEntityNotFoundError:
            return None
        return entity if entity.workspace == SYSTEM_WORKSPACE else None

    def renew(
        self,
        delegation_id: str,
        *,
        expected_db_version: int,
        expires_at: datetime,
        opaque_subject_token_hash: str | None = None,
    ) -> WorkloadDelegationEntity:
        """Renew expiry or rotate an existing Docker proof without changing the grant binding.

        Trusted reconciliation may renew an expired record to recover a live deployment.
        Deleted records and legacy revoked records cannot be renewed.
        """
        existing = self.get_by_id(delegation_id)
        if existing is None:
            raise WorkloadDelegationConflictError("workload delegation no longer exists")
        renewed = _renewal(
            existing,
            expected_db_version=expected_db_version,
            expires_at=expires_at,
            opaque_subject_token_hash=opaque_subject_token_hash,
        )
        try:
            return self._entity_client.update(renewed)
        except (NemoEntityConflictError, NemoEntityNotFoundError) as exc:
            raise WorkloadDelegationConflictError("workload delegation changed during renewal") from exc

    def bind_docker_container(self, delegation_id: str, container_id: str) -> WorkloadDelegationEntity:
        """Persist a write-once container association before Docker workload startup."""
        existing = self.get_by_id(delegation_id)
        if existing is None:
            raise NemoEntityNotFoundError("Docker workload delegation not found")
        bound = _bind_docker_container(existing, container_id)
        if existing.docker_container_id == container_id:
            return existing
        return self._entity_client.update(bound)

    def revoke_for_docker_container(self, delegation_id: str, container_id: str) -> None:
        """Delete a grant only if its stored container association matches Docker's actual ID."""
        existing = self.get_by_id(delegation_id)
        if existing is not None:
            _check_docker_container(existing, container_id)
            self._revoke(existing)

    def revoke(self, name: str) -> None:
        """Delete the current grant at this lookup name."""
        existing = self.get(name)
        if existing is not None:
            self._revoke(existing)

    def revoke_by_id(self, delegation_id: str) -> None:
        """Delete one exact grant without selecting a replacement by name."""
        existing = self.get_by_id(delegation_id)
        if existing is not None:
            self._revoke(existing)

    def list_by_workload(
        self, scope: WorkloadDelegationLookupScope, *, page_size: int = 100
    ) -> list[WorkloadDelegationEntity]:
        """List delegation records for one physical workload scope."""
        return _list_by_workload_sync(self._entity_client, scope=scope, page_size=page_size)

    def revoke_by_workload(self, scope: WorkloadDelegationLookupScope, *, page_size: int = 100) -> None:
        """Delete the exact records selected for this workload scope."""
        failures: list[Exception] = []
        for entity in self.list_by_workload(scope, page_size=page_size):
            try:
                self._revoke(entity)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise WorkloadDelegationError("Some workload delegations could not be revoked") from ExceptionGroup(
                "Workload delegation revocation failures", failures
            )

    def _revoke(self, entity: WorkloadDelegationEntity) -> None:
        for attempt in range(_REVOCATION_DELETE_ATTEMPTS):
            try:
                self._delete(entity, reason="revoked")
                return
            except WorkloadDelegationConflictError:
                if attempt == _REVOCATION_DELETE_ATTEMPTS - 1:
                    raise
                current = self.get_by_id(entity.id)
                if current is None:
                    return
                if current.docker_container_id != entity.docker_container_id:
                    raise WorkloadDelegationConflictError("Docker binding changed during revocation") from None
                entity = current

    def _delete(self, entity: WorkloadDelegationEntity, *, reason: str) -> None:
        if not entity.id:
            raise WorkloadDelegationValidationError("deleting a delegation requires its persisted ID")
        try:
            self._entity_client.delete_by_id(WorkloadDelegationEntity, entity.id, expected_db_version=entity.db_version)
        except NemoEntityNotFoundError:
            return
        except NemoEntityConflictError as exc:
            raise WorkloadDelegationConflictError("workload delegation changed before deletion") from exc
        _log_deleted_delegation(entity, reason=reason)


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


def _inactive_after_register_conflict(
    entity: WorkloadDelegationEntity,
    existing: WorkloadDelegationEntity | None,
    *,
    conflict: Exception,
) -> WorkloadDelegationEntity:
    if existing is None:
        raise WorkloadDelegationConflictError(f"delegation '{entity.name}' already exists") from conflict
    if existing.is_active():
        raise WorkloadDelegationAlreadyActiveError(f"active delegation '{entity.name}' already exists") from conflict
    if not existing.id:
        raise WorkloadDelegationValidationError("deleting an inactive delegation requires its persisted ID")
    return existing


def _log_deleted_delegation(entity: WorkloadDelegationEntity, *, reason: str) -> None:
    logger.info(
        "Deleted workload delegation",
        extra={
            "delegation_id": entity.id,
            "delegation_name": entity.name,
            "reason": reason,
        },
    )


def _log_replacement_delegation(previous: WorkloadDelegationEntity, saved: WorkloadDelegationEntity) -> None:
    logger.info(
        "Registered replacement workload delegation",
        extra={
            "previous_delegation_id": previous.id,
            "delegation_id": saved.id,
            "delegation_name": saved.name,
            "reason": "revoked" if previous.revoked_at is not None else "expired",
        },
    )


def _bind_docker_container(entity: WorkloadDelegationEntity, container_id: str) -> WorkloadDelegationEntity:
    _require_non_empty(container_id, "docker_container_id")
    if (
        not entity.is_active()
        or not entity.opaque_subject_token_hash
        or entity.bound_reference_name
        or entity.bound_reference_value
    ):
        raise WorkloadDelegationConflictError("container binding requires an active Docker delegation")
    if entity.docker_container_id is not None and entity.docker_container_id != container_id:
        raise WorkloadDelegationConflictError("Docker delegation is already bound to a different container")
    return entity.model_copy(update={"docker_container_id": container_id})


def _check_docker_container(entity: WorkloadDelegationEntity, container_id: str) -> None:
    _require_non_empty(container_id, "docker_container_id")
    if entity.docker_container_id != container_id:
        raise WorkloadDelegationConflictError("Docker container does not match the delegation binding")


def _renewal(
    existing: WorkloadDelegationEntity,
    *,
    expected_db_version: int,
    expires_at: datetime,
    opaque_subject_token_hash: str | None,
) -> WorkloadDelegationEntity:
    if existing.db_version != expected_db_version or existing.revoked_at is not None:
        raise WorkloadDelegationConflictError("workload delegation changed or was revoked")
    if as_aware_utc(expires_at) <= datetime.now(timezone.utc):
        raise WorkloadDelegationValidationError("renewed expiry must be in the future")
    if opaque_subject_token_hash is not None:
        _require_non_empty(opaque_subject_token_hash, "opaque_subject_token_hash")
        if not existing.opaque_subject_token_hash or existing.bound_reference_name or existing.bound_reference_value:
            raise WorkloadDelegationValidationError("proof rotation requires an opaque Docker delegation")
    return existing.model_copy(
        update={
            "expires_at": expires_at,
            "opaque_subject_token_hash": opaque_subject_token_hash
            if opaque_subject_token_hash is not None
            else existing.opaque_subject_token_hash,
        }
    )


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

    # A consistency rule for writers that register through a store. A direct write to the entity store skips it.
    if entity.job_uid is not None and not (
        entity.workload_kind == _JOB_WORKLOAD_KIND and entity.job_id and entity.attempt_id and entity.step_id
    ):
        raise WorkloadDelegationValidationError(
            "job_uid is only valid on a job step's delegation that also carries job_id, attempt_id and step_id"
        )

    if entity.is_expired():
        raise WorkloadDelegationValidationError("workload delegation expires_at must be in the future")
    if entity.revoked_at is not None:
        raise WorkloadDelegationValidationError("new delegations cannot carry a legacy revocation marker")

    if entity.docker_container_id is not None and not entity.opaque_subject_token_hash:
        raise WorkloadDelegationValidationError("Docker container association requires an opaque Docker proof")

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
