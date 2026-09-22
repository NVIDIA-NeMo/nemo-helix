# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage for online stable-account identity materialization."""

import asyncio
import uuid
from collections.abc import Generator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from nemo_platform_ext.auth.helpers import generate_unsigned_jwt
from nmp.common.config import AuthConfig
from nmp.core.entities.app.repository import get_async_session_maker
from nmp.core.entities.app.repository.sqlalchemy.models import DBAccount, DBAccountIdentity
from nmp.testing.client import create_test_client
from sqlalchemy import select

AUTHENTIK_ISSUER = "https://authentik.example.test/application/o/nemo/"
IAM_ROLE_BINDINGS_PATH = "/apis/auth/v2/iam/role-bindings"
SERVICE_HEADERS = {"X-NMP-Principal-Id": "service:integration-test"}
WORKSPACES_PATH = "/apis/entities/v2/workspaces"

# Each test boots the full platform and reloads policy data on every PDP call
# (bundle_cache_seconds=0); with coverage on a loaded CI runner that runs past
# the 120s session default.
pytestmark = pytest.mark.timeout(300)


@dataclass(frozen=True)
class IdentityRow:
    account_id: str
    account_type: str
    primary_email: str | None
    issuer: str
    subject: str
    subject_claim: str
    linked_via: str


@pytest.fixture
def account_migration_client() -> Generator[TestClient]:
    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
        auth_bundle_cache_seconds=0,
        workspaces=[],
        projects=[],
        service_configs={
            AuthConfig: AuthConfig(
                enabled=True,
                allow_unsigned_jwt=True,
                policy_decision_point_provider="embedded",
                policy_decision_point_base_url="http://testserver",
                propagation_poll_interval_seconds=0.05,
            ),
        },
    ) as client:
        yield client


async def _read_identity_rows(*, issuer: str, subject: str) -> list[IdentityRow]:
    session_maker = await get_async_session_maker()
    async with session_maker() as session:
        result = await session.execute(
            select(DBAccountIdentity, DBAccount)
            .join(DBAccount, DBAccount.id == DBAccountIdentity.account_id)
            .where(DBAccountIdentity.issuer == issuer, DBAccountIdentity.subject == subject)
        )
        return [
            IdentityRow(
                account_id=account.id,
                account_type=account.type,
                primary_email=account.primary_email,
                issuer=identity.issuer,
                subject=identity.subject,
                subject_claim=identity.subject_claim,
                linked_via=identity.linked_via,
            )
            for identity, account in result.all()
        ]


def _identity_rows(*, issuer: str, subject: str) -> list[IdentityRow]:
    return asyncio.run(_read_identity_rows(issuer=issuer, subject=subject))


def _create_workspace(client: TestClient, workspace: str) -> None:
    response = client.post(
        f"{WORKSPACES_PATH}?wait_role_propagation=false",
        json={"name": workspace, "description": "Stable account migration test"},
        headers=SERVICE_HEADERS,
    )
    assert response.status_code in {200, 201}, response.text


def _grant_workspace_role(client: TestClient, *, workspace: str, principal: str, role: str) -> None:
    response = client.post(
        f"{IAM_ROLE_BINDINGS_PATH}?wait_role_propagation=false",
        json={"principal": principal, "role": role, "workspace": workspace},
        headers=SERVICE_HEADERS,
    )
    assert response.status_code in {200, 201}, response.text


def test_first_bearer_request_materializes_account_and_uses_legacy_alias_binding(
    account_migration_client: TestClient,
) -> None:
    subject = f"authentik-sub-{uuid.uuid4().hex[:8]}"
    email = f"authentik-user-{uuid.uuid4().hex[:8]}@example.com"
    workspace = f"account-migration-{uuid.uuid4().hex[:8]}"
    token = generate_unsigned_jwt(principal_id=subject, email=email, issuer=AUTHENTIK_ISSUER)

    assert _identity_rows(issuer=AUTHENTIK_ISSUER, subject=subject) == []

    _create_workspace(account_migration_client, workspace)
    _grant_workspace_role(account_migration_client, workspace=workspace, principal=email, role="Viewer")

    try:
        response = account_migration_client.get(
            f"{WORKSPACES_PATH}/{workspace}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["name"] == workspace

        rows = _identity_rows(issuer=AUTHENTIK_ISSUER, subject=subject)
        assert len(rows) == 1
        row = rows[0]
        assert row.account_id.startswith("account-")
        assert row.account_type == "user"
        assert row.primary_email == email
        assert row.subject_claim == "sub"
        assert row.linked_via == "resolver_materialization"

        repeat_response = account_migration_client.get(
            f"{WORKSPACES_PATH}/{workspace}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert repeat_response.status_code == 200, repeat_response.text
        assert _identity_rows(issuer=AUTHENTIK_ISSUER, subject=subject) == rows
    finally:
        account_migration_client.delete(f"{WORKSPACES_PATH}/{workspace}", headers=SERVICE_HEADERS)


def test_pdp_materializes_service_actor_and_delegated_subject_accounts(
    account_migration_client: TestClient,
) -> None:
    subject = f"delegated-sub-{uuid.uuid4().hex[:8]}"
    email = f"delegated-user-{uuid.uuid4().hex[:8]}@example.com"

    assert _identity_rows(issuer="nemo:service", subject="jobs") == []
    assert _identity_rows(issuer="nemo:trusted-header", subject=subject) == []

    response = account_migration_client.post(
        "/apis/auth/v2/authz/allow",
        json={
            "input": {
                "principal_id": "service:jobs",
                "on_behalf_of_principal_id": subject,
                "principal_email": email,
                "method": "GET",
                "path": WORKSPACES_PATH,
            }
        },
        headers=SERVICE_HEADERS,
    )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["caller_kind"] == "service_principal"
    assert result["actor_account_id"].startswith("account-")
    assert result["actor_aliases"] == ["service:jobs"]
    assert result["subject_account_id"].startswith("account-")
    assert result["subject_aliases"] == [subject, email]

    service_rows = _identity_rows(issuer="nemo:service", subject="jobs")
    assert len(service_rows) == 1
    assert service_rows[0].account_id == result["actor_account_id"]
    assert service_rows[0].account_type == "service"
    assert service_rows[0].subject_claim == "service"

    subject_rows = _identity_rows(issuer="nemo:trusted-header", subject=subject)
    assert len(subject_rows) == 1
    assert subject_rows[0].account_id == result["subject_account_id"]
    assert subject_rows[0].account_type == "user"
    assert subject_rows[0].primary_email == email
