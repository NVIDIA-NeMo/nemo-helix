# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-safe workload identity helpers for OBO token exchange."""

from __future__ import annotations

import datetime
import io
import logging
import tarfile

from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.auth.workload_delegations import (
    KUBERNETES_POD_UID_REFERENCE_NAME,
    SYSTEM_WORKSPACE,
    WorkloadDelegationEntity,
    WorkloadDelegationScope,
    WorkloadDelegationValidationError,
    create_opaque_docker_proof_token,
    docker_delegation_name,
    docker_workload_delegation_name,
    kubernetes_pod_uid_delegation_name,
)
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from pydantic import ValidationError

logger = logging.getLogger(__name__)

WORKLOAD_IDENTITY_TOKEN_FILE_PATH = "/var/run/secrets/nemo-helix/workload/token"
WORKLOAD_IDENTITY_VOLUME_PATH = "/var/run/secrets/nemo-helix/workload"
WORKLOAD_IDENTITY_VOLUME_NAME = "nhx-workload-identity"
WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS = 300
DEFAULT_WORKLOAD_AUDIENCE = "nemo-helix"


class WorkloadIdentityConfigError(RuntimeError):
    """Raised when platform auth config cannot be resolved for workload identity."""


def workload_identity_env(
    *,
    token_file_path: str = WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
) -> dict[str, str]:
    """Return environment variables that point workload code at its subject token file."""
    return {WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR: _require_non_empty(token_file_path, "token_file_path")}


def is_workload_identity_token_exchange_enabled() -> bool:
    """Return whether controllers should inject workload identity token-file config."""
    try:
        from nhx.common.config import get_auth_config

        return bool(get_auth_config().oidc.workload_token_exchange_enabled)
    except ImportError:
        logger.debug("Could not resolve auth config for workload identity token exchange", exc_info=True)
        return False
    except (RuntimeError, ValidationError) as exc:
        message = "Could not resolve auth config for workload identity token exchange"
        logger.warning(message, exc_info=True)
        raise WorkloadIdentityConfigError(message) from exc


def get_workload_identity_token_audience() -> str:
    """Return the Kubernetes projected service-account token audience for workload identity."""
    try:
        from nhx.common.config import get_auth_config

        oidc = get_auth_config().oidc
        return oidc.workload_client_id or oidc.client_id or DEFAULT_WORKLOAD_AUDIENCE
    except ImportError:
        logger.debug("Could not resolve auth config for workload identity audience", exc_info=True)
        return DEFAULT_WORKLOAD_AUDIENCE
    except (RuntimeError, ValidationError):
        logger.warning("Could not resolve auth config for workload identity audience", exc_info=True)
        return DEFAULT_WORKLOAD_AUDIENCE


def get_workload_delegation_audience() -> str:
    """Return the RFC 8693 audience used when exchanging workload subject tokens."""
    try:
        from nhx.common.config import get_auth_config

        oidc = get_auth_config().oidc
        return oidc.workload_audience or oidc.audience or DEFAULT_WORKLOAD_AUDIENCE
    except ImportError:
        logger.debug("Could not resolve auth config for workload delegation audience", exc_info=True)
        return DEFAULT_WORKLOAD_AUDIENCE
    except (RuntimeError, ValidationError):
        logger.warning("Could not resolve auth config for workload delegation audience", exc_info=True)
        return DEFAULT_WORKLOAD_AUDIENCE


def workload_delegation_expires_at(
    *,
    ttl_seconds_active: int,
    now: datetime.datetime | None = None,
) -> datetime.datetime:
    effective_now = now or datetime.datetime.now(datetime.timezone.utc)
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=datetime.timezone.utc)
    return effective_now + datetime.timedelta(seconds=ttl_seconds_active + WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS)


def build_token_archive(token: str, *, name: str = "token.tmp") -> io.BytesIO:
    """Build a tar archive containing one read-only token file."""
    data = token.encode("utf-8")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        info.mode = 0o400
        tar.addfile(info, io.BytesIO(data))
    archive.seek(0)
    return archive


def build_docker_opaque_workload_delegation(
    *,
    scope: WorkloadDelegationScope,
    workload_audience: str,
    workload_generation: str,
    auth_context: AuthContext,
    ttl_seconds_active: int,
    now: datetime.datetime | None = None,
    job_id: str | None = None,
    attempt_id: str | None = None,
    step_id: str | None = None,
) -> tuple[WorkloadDelegationEntity, str]:
    """Build a Docker opaque-proof delegation row and the corresponding proof token."""
    _validate_optional_job_fields(job_id=job_id, attempt_id=attempt_id, step_id=step_id)
    delegation_name = _docker_delegation_name_for_workload(
        scope=scope,
        workload_generation=workload_generation,
        job_id=job_id,
        attempt_id=attempt_id,
        step_id=step_id,
    )
    proof_token, proof_token_hash = create_opaque_docker_proof_token(delegation_name)
    return (
        WorkloadDelegationEntity(
            name=delegation_name,
            workspace=SYSTEM_WORKSPACE,
            workload_subject=delegation_name,
            workload_audience=workload_audience,
            workload_workspace=scope.workload_workspace,
            workload_kind=scope.workload_kind,
            workload_id=scope.workload_instance_id,
            workload_claim_id=scope.workload_claim_id,
            workload_generation=workload_generation,
            job_id=job_id,
            attempt_id=attempt_id,
            step_id=step_id,
            auth_context=_copy_auth_context(auth_context),
            opaque_subject_token_hash=proof_token_hash,
            expires_at=workload_delegation_expires_at(ttl_seconds_active=ttl_seconds_active, now=now),
        ),
        proof_token,
    )


def build_kubernetes_pod_uid_workload_delegation(
    *,
    scope: WorkloadDelegationScope,
    workload_audience: str,
    workload_generation: str,
    namespace: str,
    service_account_name: str,
    pod_uid: str,
    auth_context: AuthContext,
    ttl_seconds_active: int,
    now: datetime.datetime | None = None,
    job_id: str | None = None,
    attempt_id: str | None = None,
    step_id: str | None = None,
) -> WorkloadDelegationEntity:
    """Build a Kubernetes Pod UID-bound delegation row."""
    _validate_optional_job_fields(job_id=job_id, attempt_id=attempt_id, step_id=step_id)
    workload_subject = kubernetes_service_account_subject(
        namespace=namespace, service_account_name=service_account_name
    )
    return WorkloadDelegationEntity(
        name=kubernetes_pod_uid_delegation_name(
            workload_audience=workload_audience,
            workload_subject=workload_subject,
            pod_uid=pod_uid,
        ),
        workspace=SYSTEM_WORKSPACE,
        workload_subject=workload_subject,
        workload_audience=workload_audience,
        workload_workspace=scope.workload_workspace,
        workload_kind=scope.workload_kind,
        workload_id=scope.workload_instance_id,
        workload_claim_id=scope.workload_claim_id,
        workload_generation=workload_generation,
        job_id=job_id,
        attempt_id=attempt_id,
        step_id=step_id,
        auth_context=_copy_auth_context(auth_context),
        bound_reference_name=KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value=pod_uid,
        expires_at=workload_delegation_expires_at(ttl_seconds_active=ttl_seconds_active, now=now),
    )


def kubernetes_service_account_subject(*, namespace: str, service_account_name: str) -> str:
    """Return the Kubernetes service account subject string used by bound tokens."""
    return (
        f"system:serviceaccount:"
        f"{_require_non_empty(namespace, 'namespace')}:"
        f"{_require_non_empty(service_account_name, 'service_account_name')}"
    )


def _docker_delegation_name_for_workload(
    *,
    scope: WorkloadDelegationScope,
    workload_generation: str,
    job_id: str | None,
    attempt_id: str | None,
    step_id: str | None,
) -> str:
    if job_id is not None and attempt_id is not None and step_id is not None:
        return docker_delegation_name(
            workload_workspace=scope.workload_workspace,
            job_id=job_id,
            attempt_id=attempt_id,
            step_id=step_id,
        )
    return docker_workload_delegation_name(
        scope=scope,
        workload_generation=workload_generation,
    )


def _validate_optional_job_fields(
    *,
    job_id: str | None,
    attempt_id: str | None,
    step_id: str | None,
) -> None:
    if not any((job_id, attempt_id, step_id)):
        return
    for field_name, value in (("job_id", job_id), ("attempt_id", attempt_id), ("step_id", step_id)):
        _require_non_empty(value, field_name)


def _copy_auth_context(auth_context: AuthContext) -> AuthContext:
    return AuthContext.model_validate(auth_context.model_dump(mode="python", exclude_none=True))


def _require_non_empty(value: str | None, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkloadDelegationValidationError(f"{field_name} must be a non-empty string")
    return value
