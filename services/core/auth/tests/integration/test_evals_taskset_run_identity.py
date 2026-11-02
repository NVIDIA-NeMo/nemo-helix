# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run reads preserve the authenticated user's workspace identity end to end."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from nemo_evals.api import dependencies
from nemo_evals.service import EvaluatorPluginService
from nemo_evals.taskset_runs.entities import MemberPhase, TasksetRunEntity, TasksetRunMemberEntity
from nemo_helix_ext.auth.helpers import generate_unsigned_jwt
from nemo_helix_plugin.auth import RuntimePrincipal
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nhx.common.config import AuthConfig
from nhx.core.entities.app.repository import get_async_session_maker
from nhx.core.entities.app.repository.sqlalchemy.models import DBAccountIdentity
from nhx.core.entities.service import EntitiesService
from nhx.core.files.service import FilesService
from nhx.platform_runner.plugin_adapter import NemoServiceAdapter
from nhx.testing import ClientContext, create_test_client
from sqlalchemy import select


class TestEvalsService(NemoServiceAdapter):
    __test__ = False

    def __init__(self) -> None:
        super().__init__(EvaluatorPluginService())


async def _resolved_account_id(issuer: str, subject: str) -> str:
    sessions = await get_async_session_maker()
    async with sessions() as session:
        result = await session.execute(
            select(DBAccountIdentity.account_id).where(
                DBAccountIdentity.issuer == issuer, DBAccountIdentity.subject == subject
            )
        )
        return result.scalar_one()


async def _seed_run(client: AsyncNemoClient, workspace: str) -> None:
    entities = EntityClient(AsyncEntitiesClient.from_client(client))
    now = datetime.now(timezone.utc)
    run = await entities.create(
        TasksetRunEntity(
            name="run-identity-check",
            workspace=workspace,
            taskset_ref=f"{workspace}/suite#" + "a" * 64,
            plan_ref=f"{workspace}/plan#plan.json",
            plan_digest="b" * 64,
            request_digest="c" * 64,
            accepted_at=now,
            execution_deadline_at=now + timedelta(hours=1),
            finalization_deadline_at=now + timedelta(hours=2),
        )
    )
    member = TasksetRunMemberEntity(name="task-0", workspace=workspace, task_index=0, phase=MemberPhase.REGISTERED)
    member._parent = run.id
    await entities.create(member)


def test_run_reads_preserve_email_group_and_account_grants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    issuer = "https://identity.example.test/evals"
    subject = "evals-user-subject"
    email = "evals-user@example.test"
    group = "evals-editors"
    admin = {"X-NHX-Principal-Id": "service:integration-test"}
    user = {
        "Authorization": "Bearer "
        + generate_unsigned_jwt(
            principal_id=subject,
            email=email,
            groups=[group],
            issuer=issuer,
        )
    }
    with create_test_client(
        TestEvalsService,
        EntitiesService,
        FilesService,
        client_type=ClientContext,
        auth_enabled=True,
        tmp_dir=tmp_path,
        workspaces=[],
        projects=[],
        auth_bundle_cache_seconds=0,
        auth_policy_data_refresh_interval=3600,
        access_log=True,
        service_configs={
            AuthConfig: AuthConfig(
                enabled=True,
                allow_unsigned_jwt=True,
                policy_decision_point_provider="embedded",
                policy_decision_point_base_url="http://127.0.0.1",
                propagation_poll_interval_seconds=0.05,
            )
        },
    ) as ctx:
        assert ctx.access_log is not None
        portal = ctx.test_client.portal
        assert portal is not None
        assert ctx.test_client.get("/apis/entities/v2/workspaces", headers=user).status_code == 200
        account_id = portal.call(_resolved_account_id, issuer, subject)
        factory = dependencies.get_async_nemo_client

        def provide(*, as_service: str, internal: bool, on_behalf_of: str | RuntimePrincipal) -> AsyncNemoClient:
            client = factory(as_service=as_service, internal=internal, on_behalf_of=on_behalf_of)
            # Keep production client/header construction; route its HTTP transport to the test platform.
            monkeypatch.setattr(client._http, "_transport_for_url", ctx.async_client._http._transport_for_url)
            return client

        monkeypatch.setattr(dependencies, "get_async_nemo_client", provide)
        for kind, principal in [
            ("email", email),
            ("group", group),
            ("account", account_id),
            ("subject", subject),
            ("none", ""),
        ]:
            workspace = f"identity-{kind}"
            created = ctx.test_client.post(
                "/apis/entities/v2/workspaces?wait_role_propagation=false", json={"name": workspace}, headers=admin
            )
            assert created.status_code == 201, created.text
            if principal:
                granted = ctx.test_client.post(
                    "/apis/auth/v2/iam/role-bindings?wait_role_propagation=false",
                    json={"principal": principal, "workspace": workspace, "role": "Editor"},
                    headers=admin,
                )
                assert granted.status_code == 200, granted.text
            portal.call(_seed_run, ctx.async_client.with_headers({"X-NHX-Principal-Id": "service:evals"}), workspace)
            base = f"/apis/evals/v2/workspaces/{workspace}"
            ctx.access_log.clear()
            for suffix in [
                "/tasks",
                "/agent-evaluate/runs",
                "/agent-evaluate/runs/run-identity-check",
                "/agent-evaluate/runs/run-identity-check/tasks",
            ]:
                response = ctx.test_client.get(base + suffix, headers=user)
                assert response.status_code == (200 if principal else 403), (kind, suffix, response.text)
                if principal and suffix.endswith("run-identity-check"):
                    assert response.json()["run_id"] == "run-identity-check"
                if principal and suffix.endswith("run-identity-check/tasks"):
                    assert response.json()["data"][0]["task_index"] == 0
            delegated = [
                entry
                for entry in ctx.access_log.requests
                if entry.principal_id == "service:evals" and entry.on_behalf_of == subject
            ]
            assert bool(delegated) == bool(principal)
            for entry in delegated:
                assert entry.headers["x-nhx-subject-account-id"] == account_id
                assert entry.headers["x-nhx-principal-on-behalf-of-email"] == email
                assert entry.headers["x-nhx-principal-on-behalf-of-groups"] == group
        created = ctx.test_client.post(
            "/apis/entities/v2/workspaces?wait_role_propagation=false",
            json={"name": "identity-user-created"},
            headers=user,
        )
        assert created.status_code == 201, created.text
        response = ctx.test_client.get(
            "/apis/evals/v2/workspaces/identity-user-created/agent-evaluate/runs", headers=user
        )
        assert response.status_code == 200, response.text
