# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import tarfile

import httpx
from nemo_platform import NeMoPlatform
from nmp.common.auth import AuthContext, Principal, WorkloadDelegationEntity, docker_delegation_name
from nmp.common.entities import SYSTEM_WORKSPACE
from nmp.core.jobs.controllers.backends.workload_tokens import (
    WORKLOAD_DELEGATION_TTL_BUFFER_SECONDS,
    build_token_archive,
    create_authenticated_workload_delegation_store,
    workload_delegation_expires_at,
)


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


def test_authenticated_workload_delegation_store_uses_sync_service_client() -> None:
    entity = _workload_delegation_entity()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return httpx.Response(
            201,
            request=request,
            json={
                "entity_type": "workload_delegation",
                "id": "delegation-id",
                "workspace": SYSTEM_WORKSPACE,
                "parent": None,
                "project": None,
                "name": body["name"],
                "data": body["data"],
                "created_at": now,
                "created_by": "service:jobs",
                "updated_at": now,
                "updated_by": "service:jobs",
                "db_version": 1,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        sdk = NeMoPlatform(
            base_url="http://platform",
            default_headers={
                "Authorization": "Bearer controller-token",
                "X-NMP-Principal-On-Behalf-Of": "alice@example.com",
                "X-NMP-Principal-On-Behalf-Of-Email": "alice@example.com",
                "X-NMP-Principal-On-Behalf-Of-Groups": "team-a",
            },
            http_client=http_client,
        )

        saved = create_authenticated_workload_delegation_store(sdk).register(entity)

    assert saved.id == "delegation-id"
    assert saved.auth_context.principal_id == "creator@example.com"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://platform/apis/entities/v2/workspaces/system/entities/workload_delegation"
    assert request.headers["Authorization"] == "Bearer controller-token"
    assert request.headers["X-NMP-Principal-Id"] == "service:jobs"
    assert request.headers["X-NMP-Actor-Aliases"] == "service:jobs"
    assert request.headers["X-NMP-Principal-On-Behalf-Of"] == ""
    assert request.headers["X-NMP-Principal-On-Behalf-Of-Email"] == ""
    assert request.headers["X-NMP-Principal-On-Behalf-Of-Groups"] == ""
    assert request.headers["X-NMP-Internal"] == "true"


def _workload_delegation_entity() -> WorkloadDelegationEntity:
    delegation_name = docker_delegation_name(
        workload_workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step_id="step-a",
    )
    return WorkloadDelegationEntity(
        name=delegation_name,
        workspace=SYSTEM_WORKSPACE,
        workload_subject=delegation_name,
        workload_audience="nemo-platform",
        workload_workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step_id="step-a",
        auth_context=AuthContext.from_principal(Principal(id="creator@example.com")),
        expires_at=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    )
