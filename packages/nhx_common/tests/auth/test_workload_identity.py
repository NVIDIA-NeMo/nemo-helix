# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import tarfile

import pytest
from nemo_helix_plugin.auth import AuthContext as PluginAuthContext
from nhx.common.auth import (
    KUBERNETES_POD_UID_REFERENCE_NAME,
    WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS,
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    WORKLOAD_IDENTITY_TOKEN_FILE_PATH,
    Principal,
    WorkloadDelegationScope,
    WorkloadDelegationValidationError,
    build_docker_opaque_workload_delegation,
    build_kubernetes_pod_uid_workload_delegation,
    build_token_archive,
    docker_delegation_name,
    docker_workload_delegation_name,
    kubernetes_pod_uid_delegation_name,
    kubernetes_service_account_subject,
    parse_opaque_docker_proof_token,
    verify_opaque_docker_proof_token_hash,
    workload_delegation_expires_at,
    workload_identity_env,
)
from nhx.common.entities import SYSTEM_WORKSPACE


def _auth_context() -> PluginAuthContext:
    return PluginAuthContext.from_principal(Principal(id="creator@example.com", groups=["team-a"]))


def test_workload_identity_env_points_to_default_token_file() -> None:
    assert workload_identity_env() == {WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR: WORKLOAD_IDENTITY_TOKEN_FILE_PATH}
    assert workload_identity_env(token_file_path="/custom/token") == {
        WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR: "/custom/token"
    }


def test_build_token_archive_contains_read_only_token_file() -> None:
    archive = build_token_archive("subject-token", name="token.tmp")

    with tarfile.open(fileobj=archive, mode="r") as tar:
        member = tar.getmember("token.tmp")
        extracted = tar.extractfile(member)

        assert member.mode == 0o400
        assert extracted is not None
        assert extracted.read() == b"subject-token"


def test_workload_delegation_expires_at_adds_active_ttl_and_cleanup_buffer() -> None:
    now = datetime.datetime(2026, 8, 10, 12, 0, tzinfo=datetime.timezone.utc)

    expires_at = workload_delegation_expires_at(ttl_seconds_active=900, now=now)

    assert expires_at == now + datetime.timedelta(seconds=900 + WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS)


def test_workload_delegation_expires_at_normalizes_naive_now_to_utc() -> None:
    now = datetime.datetime(2026, 8, 10, 12, 0)

    expires_at = workload_delegation_expires_at(ttl_seconds_active=60, now=now)

    assert expires_at.tzinfo is datetime.timezone.utc
    assert expires_at == now.replace(tzinfo=datetime.timezone.utc) + datetime.timedelta(
        seconds=60 + WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS
    )


def test_build_docker_opaque_workload_delegation_preserves_legacy_job_name() -> None:
    now = datetime.datetime(2026, 8, 31, 9, 30, tzinfo=datetime.timezone.utc)

    delegation, proof_token = build_docker_opaque_workload_delegation(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="job",
            workload_instance_id="job-123",
        ),
        workload_audience="nemo-helix",
        workload_generation="attempt-1/step-a",
        job_id="job-123",
        attempt_id="attempt-1",
        step_id="step-a",
        auth_context=_auth_context(),
        ttl_seconds_active=900,
        now=now,
    )

    assert delegation.name == docker_delegation_name(
        workload_workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step_id="step-a",
    )
    assert delegation.workspace == SYSTEM_WORKSPACE
    assert delegation.workload_kind == "job"
    assert delegation.workload_id == "job-123"
    assert delegation.workload_claim_id is None
    assert delegation.workload_generation == "attempt-1/step-a"
    assert delegation.workload_subject == delegation.name
    assert delegation.opaque_subject_token_hash is not None
    assert delegation.expires_at == now + datetime.timedelta(seconds=900 + WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS)
    parsed = parse_opaque_docker_proof_token(proof_token)
    assert parsed.delegation_name == delegation.name
    assert verify_opaque_docker_proof_token_hash(parsed.secret, delegation.opaque_subject_token_hash)


def test_build_docker_opaque_workload_delegation_supports_deployment_rows_without_job_fields() -> None:
    scope = WorkloadDelegationScope(
        workload_workspace="default",
        workload_kind="deployment",
        workload_instance_id="deployment-123",
        workload_claim_id="logical-deployment",
    )
    delegation, proof_token = build_docker_opaque_workload_delegation(
        scope=scope,
        workload_audience="nemo-helix",
        workload_generation="container-abc",
        auth_context=_auth_context(),
        ttl_seconds_active=900,
    )

    assert delegation.name == docker_workload_delegation_name(
        scope=scope,
        workload_generation="container-abc",
    )
    assert delegation.job_id is None
    assert delegation.attempt_id is None
    assert delegation.step_id is None
    assert delegation.workload_id == "deployment-123"
    assert delegation.workload_claim_id == "logical-deployment"
    assert parse_opaque_docker_proof_token(proof_token).delegation_name == delegation.name


def test_build_docker_opaque_workload_delegation_rejects_partial_job_metadata() -> None:
    with pytest.raises(WorkloadDelegationValidationError):
        build_docker_opaque_workload_delegation(
            scope=WorkloadDelegationScope(
                workload_workspace="default",
                workload_kind="job",
                workload_instance_id="job-123",
            ),
            workload_audience="nemo-helix",
            workload_generation="attempt-1/step-a",
            job_id="job-123",
            auth_context=_auth_context(),
            ttl_seconds_active=900,
        )


def test_build_kubernetes_pod_uid_workload_delegation_sets_verified_reference_fields() -> None:
    now = datetime.datetime(2026, 8, 31, 10, 0, tzinfo=datetime.timezone.utc)

    delegation = build_kubernetes_pod_uid_workload_delegation(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="deployment",
            workload_instance_id="deployment-123",
            workload_claim_id="logical-deployment",
        ),
        workload_audience="nemo-helix",
        workload_generation="pod-uid-123",
        namespace="deployments",
        service_account_name="runner",
        pod_uid="pod-uid-123",
        auth_context=_auth_context(),
        ttl_seconds_active=600,
        now=now,
    )

    workload_subject = kubernetes_service_account_subject(namespace="deployments", service_account_name="runner")
    assert workload_subject == "system:serviceaccount:deployments:runner"
    assert delegation.name == kubernetes_pod_uid_delegation_name(
        workload_audience="nemo-helix",
        workload_subject=workload_subject,
        pod_uid="pod-uid-123",
    )
    assert delegation.bound_reference_name == KUBERNETES_POD_UID_REFERENCE_NAME
    assert delegation.bound_reference_value == "pod-uid-123"
    assert delegation.workload_subject == workload_subject
    assert delegation.workload_id == "deployment-123"
    assert delegation.workload_claim_id == "logical-deployment"
    assert delegation.expires_at == now + datetime.timedelta(seconds=600 + WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS)
