# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Granting roles to a service account whose Scoped Access Keys are bound elsewhere."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from nhx.core.entities.api.v2.workspaces.endpoints import _require_bound_workspace_admin

TEAM_A = SimpleNamespace(id="workspace-aaa", name="team-a")
TEAM_B = SimpleNamespace(id="workspace-bbb", name="team-b")
SERVICE_ACCOUNT = "service-account:otel-collector"
ACCOUNT_ID = "account-123"


class _AuthClient:
    def __init__(self, *, enabled: bool = True, privileged: bool = False, roles: dict[str, str] | None = None) -> None:
        self.auth_enabled = enabled
        self.principal = SimpleNamespace(
            effective_principal=SimpleNamespace(is_privileged=lambda: privileged),
        )
        self._roles = roles or {}
        self.role_checks: list[tuple[str, str]] = []

    async def has_role(self, workspace: str, role: str) -> bool:
        self.role_checks.append((workspace, role))
        return self._roles.get(workspace) == role


class _WorkspaceRepository:
    def __init__(self, *workspaces: SimpleNamespace) -> None:
        self.workspaces = list(workspaces)

    async def get_workspace_by_id(self, *, workspace_id: str):
        return next((w for w in self.workspaces if w.id == workspace_id), None)

    async def get_workspace_by_name(self, *, name: str):
        # The last registered workspace of a name is the one that currently holds it.
        return next((w for w in reversed(self.workspaces) if w.name == name), None)


class _IdentityStore:
    def __init__(self, account_id: str | None = ACCOUNT_ID) -> None:
        self.account_id = account_id
        self.lookups = 0

    async def get_account_id(self, *, issuer: str, subject: str) -> str | None:
        self.lookups += 1
        return self.account_id


class _CredentialStore:
    def __init__(self, bound_workspace_ids: set[str] | None = None) -> None:
        self.bound = bound_workspace_ids or set()
        self.lookups = 0

    async def bound_workspace_ids_for_subject(self, credential_type: str, subject_account_id: str) -> set[str]:
        self.lookups += 1
        return set(self.bound)


async def _check(
    principal: str,
    workspace: SimpleNamespace = TEAM_B,
    *,
    auth_client: _AuthClient | None = None,
    workspace_repository: _WorkspaceRepository | None = None,
    identity_store: _IdentityStore | None = None,
    credential_store: _CredentialStore | None = None,
) -> None:
    await _require_bound_workspace_admin(
        principal,
        workspace,
        auth_client=auth_client or _AuthClient(),
        workspace_repository=workspace_repository or _WorkspaceRepository(TEAM_A, TEAM_B),
        identity_store=identity_store or _IdentityStore(),
        credential_store=credential_store or _CredentialStore({TEAM_A.id}),
    )


async def test_a_human_principal_is_never_restricted():
    identity_store, credential_store = _IdentityStore(), _CredentialStore({TEAM_A.id})

    await _check("alice@example.com", identity_store=identity_store, credential_store=credential_store)

    assert (identity_store.lookups, credential_store.lookups) == (0, 0)


async def test_an_account_that_was_never_issued_a_key_is_not_restricted():
    credential_store = _CredentialStore({TEAM_A.id})

    await _check(SERVICE_ACCOUNT, identity_store=_IdentityStore(None), credential_store=credential_store)

    assert credential_store.lookups == 0


async def test_an_unbound_service_account_is_not_restricted():
    await _check(SERVICE_ACCOUNT, credential_store=_CredentialStore(set()))


async def test_granting_roles_in_the_workspace_the_keys_are_bound_to_is_not_restricted():
    auth_client = _AuthClient()

    await _check(SERVICE_ACCOUNT, TEAM_A, auth_client=auth_client)

    assert auth_client.role_checks == []


async def test_granting_roles_elsewhere_is_refused_for_a_caller_who_doesnt_administer_the_bound_workspace():
    with pytest.raises(HTTPException) as refused:
        await _check(SERVICE_ACCOUNT, TEAM_B, auth_client=_AuthClient(roles={"team-b": "Admin"}))

    assert refused.value.status_code == 403
    assert "'team-a'" in refused.value.detail
    assert "'team-b'" in refused.value.detail


async def test_an_admin_of_the_bound_workspace_may_grant_roles_elsewhere():
    await _check(SERVICE_ACCOUNT, TEAM_B, auth_client=_AuthClient(roles={"team-a": "Admin", "team-b": "Admin"}))


async def test_a_helix_admin_may_grant_roles_elsewhere():
    await _check(SERVICE_ACCOUNT, TEAM_B, auth_client=_AuthClient(roles={"system": "HelixAdmin"}))


async def test_a_privileged_service_caller_is_not_restricted():
    auth_client = _AuthClient(privileged=True)

    await _check(SERVICE_ACCOUNT, TEAM_B, auth_client=auth_client)

    assert auth_client.role_checks == []


async def test_nothing_is_restricted_while_auth_is_disabled():
    identity_store = _IdentityStore()

    await _check(SERVICE_ACCOUNT, TEAM_B, auth_client=_AuthClient(enabled=False), identity_store=identity_store)

    assert identity_store.lookups == 0


async def test_an_account_id_principal_resolves_without_the_identity_store():
    identity_store = _IdentityStore()

    with pytest.raises(HTTPException):
        await _check(ACCOUNT_ID, TEAM_B, identity_store=identity_store)

    assert identity_store.lookups == 0


async def test_a_bound_workspace_that_no_longer_exists_keeps_the_key_helix_admin_only():
    with pytest.raises(HTTPException) as refused:
        await _check(SERVICE_ACCOUNT, TEAM_B, workspace_repository=_WorkspaceRepository(TEAM_B))

    assert refused.value.status_code == 403
    # Named by ID, since there is no name left to report.
    assert TEAM_A.id in refused.value.detail


async def test_admin_of_a_recreated_workspace_cannot_grant_the_old_workspaces_account_roles_elsewhere():
    # Defense in depth: if a stale row ever let the old ID resolve to the name 'team-a' while
    # that name belongs to a new workspace, being Admin of the new one must still grant nothing.
    recreated = SimpleNamespace(id="workspace-aaa-2", name="team-a")

    with pytest.raises(HTTPException) as refused:
        await _check(
            SERVICE_ACCOUNT,
            TEAM_B,
            auth_client=_AuthClient(roles={"team-a": "Admin"}),
            workspace_repository=_WorkspaceRepository(TEAM_A, recreated, TEAM_B),
        )

    assert refused.value.status_code == 403
