# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only a namespaced service account's own workspace may grant it roles elsewhere."""

from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.entities.api.v2.workspaces.endpoints import (
    _require_service_account_namespace_owner,
    add_workspace_member,
    create_workspace,
    update_workspace_member,
)
from nhx.core.entities.api.v2.workspaces.schemas import WorkspaceInput, WorkspaceMemberInput, WorkspaceMemberUpdate
from nhx.core.entities.app.repository import (
    SERVICE_ACCOUNT_IDENTITY_ISSUER,
    AccountIdentityStore,
    EntityRepositoryInterface,
    WorkspaceRepositoryInterface,
)

# Account IDs and the service accounts they belong to, as the identity store has them.
_SERVICE_ACCOUNT_IDS = {"account-teamb": "team-b--ci", "account-ops": "platform-ops"}
# When the workspaces were created, and when service accounts were first issued a key: after them.
_WORKSPACES_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)
_FIRST_KEYS_ISSUED_AT = datetime(2026, 2, 1, tzinfo=UTC)


def _identity_store(*, linked_at: datetime | None = _FIRST_KEYS_ISSUED_AT) -> AccountIdentityStore:
    async def get_subject(*, account_id: str, issuer: str) -> str | None:
        assert issuer == SERVICE_ACCOUNT_IDENTITY_ISSUER
        return _SERVICE_ACCOUNT_IDS.get(account_id)

    async def get_linked_at(*, issuer: str, subject: str) -> datetime | None:
        assert issuer == SERVICE_ACCOUNT_IDENTITY_ISSUER
        return linked_at

    store = AsyncMock(spec=AccountIdentityStore)
    store.get_subject.side_effect = get_subject
    store.get_linked_at.side_effect = get_linked_at
    return store


_IDENTITIES = _identity_store()


def _workspace_repository(*existing: str) -> AsyncMock:
    async def get_workspace_by_name(*, name: str) -> object | None:
        return (
            SimpleNamespace(name=name, created_at=_WORKSPACES_CREATED_AT, _deletion_stage=None)
            if name in existing
            else None
        )

    repository = AsyncMock(spec=WorkspaceRepositoryInterface)
    repository.get_workspace_by_name.side_effect = get_workspace_by_name
    return repository


# The workspaces that exist: the namespaces the tests' service accounts are named after.
_WORKSPACES = _workspace_repository("team-b", "victim", "platform-ops")


def _auth_client(
    principal: str = "mallory@example.com", *, on_behalf_of: str | None = None, auth_enabled: bool = True
) -> AuthClient:
    return AuthClient(
        principal=Principal(id=principal, on_behalf_of=on_behalf_of), config=AuthConfig(enabled=auth_enabled)
    )


@pytest.fixture
def role_checks() -> Iterator[tuple[set[tuple[str, str]], list[tuple[str, str]]]]:
    """(roles the caller holds, role checks made), with AuthClient.has_role answering from the former."""
    held: set[tuple[str, str]] = set()
    checked: list[tuple[str, str]] = []

    async def has_role(_self: AuthClient, workspace: str, role: str) -> bool:
        checked.append((workspace, role))
        return (workspace, role) in held

    with patch.object(AuthClient, "has_role", has_role):
        yield held, checked


@pytest.mark.asyncio
async def test_another_workspaces_admin_cannot_grant_a_namespaced_service_account_roles(role_checks) -> None:
    held, _ = role_checks
    held.add(("m", "Admin"))

    with pytest.raises(HTTPException) as raised:
        await _require_service_account_namespace_owner(
            _auth_client(), _IDENTITIES, _WORKSPACES, "service-account:team-b--ci", "m"
        )

    assert raised.value.status_code == 403
    assert raised.value.detail == (
        "Only a HelixAdmin or an Admin of workspace 'team-b' can grant 'service-account:team-b--ci' roles in "
        "workspace 'm'"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [("team-b", "Admin"), ("system", "HelixAdmin")], ids=["owner-admin", "helix-admin"])
async def test_the_owning_workspaces_admin_or_a_helix_admin_can_grant_it_roles_elsewhere(
    role_checks, role: tuple[str, str]
) -> None:
    held, _ = role_checks
    held.add(role)

    await _require_service_account_namespace_owner(
        _auth_client("alice@example.com"), _IDENTITIES, _WORKSPACES, "service-account:team-b--ci", "team-c"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal", "workspace"),
    [
        # In its own workspace.
        ("service-account:team-b--ci", "team-b"),
        # Not namespaced, or not a service account.
        ("service-account:platform-ops", "m"),
        ("service-account:team-b--", "m"),
        ("alice@example.com", "m"),
        ("team-b--ci", "m"),
    ],
)
async def test_other_grants_need_no_role_check(role_checks, principal: str, workspace: str) -> None:
    _, checked = role_checks

    await _require_service_account_namespace_owner(_auth_client(), _IDENTITIES, _WORKSPACES, principal, workspace)

    assert checked == []


@pytest.mark.asyncio
async def test_internal_services_and_disabled_auth_are_not_restricted(role_checks) -> None:
    _, checked = role_checks

    await _require_service_account_namespace_owner(
        _auth_client("service:auth"), _IDENTITIES, _WORKSPACES, "service-account:team-b--ci", "m"
    )
    await _require_service_account_namespace_owner(
        _auth_client(auth_enabled=False), _IDENTITIES, _WORKSPACES, "service-account:team-b--ci", "m"
    )

    assert checked == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth_client",
    [
        # A service account Mallory administers, e.g. through a Scoped Access Key for evil--bot.
        _auth_client("service-account:evil--bot"),
        # An internal service acting on that service account's behalf.
        _auth_client("service:jobs", on_behalf_of="service-account:evil--bot"),
    ],
    ids=["service-account", "service-on-behalf-of-service-account"],
)
async def test_a_service_account_caller_is_checked_like_any_other(role_checks, auth_client: AuthClient) -> None:
    held, checked = role_checks
    held.add(("evil", "Admin"))

    with pytest.raises(HTTPException) as raised:
        await _require_service_account_namespace_owner(
            auth_client, _IDENTITIES, _WORKSPACES, "service-account:victim--svc", "evil"
        )

    assert raised.value.status_code == 403
    assert checked == [("system", "HelixAdmin"), ("victim", "Admin")]


@pytest.mark.asyncio
async def test_a_binding_keyed_by_a_namespaced_service_accounts_account_id_is_checked_too(role_checks) -> None:
    # Its keys authenticate as its account ID too, so a binding keyed by it would lock the owning
    # workspace's Admins out just the same.
    held, _ = role_checks
    held.add(("m", "Admin"))

    with pytest.raises(HTTPException) as raised:
        await _require_service_account_namespace_owner(_auth_client(), _IDENTITIES, _WORKSPACES, "account-teamb", "m")

    assert raised.value.status_code == 403
    assert raised.value.detail == (
        "Only a HelixAdmin or an Admin of workspace 'team-b' can grant 'service-account:team-b--ci' roles in "
        "workspace 'm'"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("principal", ["account-ops", "account-someone"], ids=["not-namespaced", "not-a-service"])
async def test_other_account_ids_need_no_role_check(role_checks, principal: str) -> None:
    _, checked = role_checks

    await _require_service_account_namespace_owner(_auth_client(), _IDENTITIES, _WORKSPACES, principal, "m")

    assert checked == []


@pytest.mark.asyncio
async def test_a_namespace_whose_workspace_does_not_exist_has_no_owner_to_protect(role_checks) -> None:
    # e.g. an account named before this rule existed, after a workspace nobody has created.
    _, checked = role_checks

    await _require_service_account_namespace_owner(
        _auth_client(), _IDENTITIES, _WORKSPACES, "service-account:etl--loader", "m"
    )

    assert checked == []


@pytest.mark.asyncio
async def test_a_namespaced_service_account_creating_a_workspace_becomes_its_admin(role_checks) -> None:
    # It acts for itself, so the namespace rule (which stops other workspaces' Admins locking its
    # owners out) doesn't apply; accounts named with '--' before the rule existed keep working.
    _, checked = role_checks
    workspaces = _workspace_repository("team-b")
    workspaces.create_workspace.return_value = SimpleNamespace(name="elsewhere")
    entities = AsyncMock(spec=EntityRepositoryInterface)

    with patch("nhx.core.entities.api.v2.workspaces.endpoints.bindings_cache_delete", AsyncMock()):
        await create_workspace(
            WorkspaceInput(name="elsewhere"),
            workspaces,
            entities,
            _auth_client("service-account:team-b--ci"),
            wait_role_propagation=False,
        )

    assert checked == []
    binding = entities.create_entity.await_args.kwargs["data"]
    assert (binding["principal"], binding["role"]) == ("service-account:team-b--ci", "Admin")


@pytest.mark.asyncio
async def test_a_namespace_whose_workspace_is_being_deleted_has_no_owner_to_protect(role_checks) -> None:
    _, checked = role_checks
    deleting = SimpleNamespace(name="team-b", created_at=_WORKSPACES_CREATED_AT, _deletion_stage="deleting")
    workspaces = AsyncMock(spec=WorkspaceRepositoryInterface)
    workspaces.get_workspace_by_name.return_value = deleting

    await _require_service_account_namespace_owner(
        _auth_client(), _IDENTITIES, workspaces, "service-account:team-b--ci", "m"
    )

    assert checked == []


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", ["team-b", "elsewhere"], ids=["own-workspace", "elsewhere"])
async def test_an_account_older_than_its_workspace_takes_a_helix_admin_to_grant_roles(
    role_checks, workspace: str
) -> None:
    # team-b was deleted and recreated after team-b--ci was first issued a key: that key, bound to
    # the old team-b, may still be live, so the new team-b's Admins mustn't extend it anywhere.
    held, _ = role_checks
    held.update({("team-b", "Admin"), (workspace, "Admin")})
    older = _identity_store(linked_at=_WORKSPACES_CREATED_AT.replace(year=2025))

    with pytest.raises(HTTPException) as raised:
        await _require_service_account_namespace_owner(
            _auth_client(), older, _WORKSPACES, "service-account:team-b--ci", workspace
        )

    assert raised.value.status_code == 403
    assert raised.value.detail == (
        "'service-account:team-b--ci' is older than workspace 'team-b', so it belongs to an earlier workspace "
        f"of that name; only a HelixAdmin can grant it roles in workspace '{workspace}'"
    )
    held.add(("system", "HelixAdmin"))
    await _require_service_account_namespace_owner(
        _auth_client(), older, _WORKSPACES, "service-account:team-b--ci", workspace
    )


@pytest.mark.asyncio
async def test_an_account_never_issued_a_key_is_granted_roles_in_its_own_workspace_without_checks(
    role_checks,
) -> None:
    _, checked = role_checks

    await _require_service_account_namespace_owner(
        _auth_client(), _identity_store(linked_at=None), _WORKSPACES, "service-account:team-b--ci", "team-b"
    )

    assert checked == []


@pytest.mark.asyncio
async def test_a_membership_request_that_grants_no_roles_is_not_refused(role_checks) -> None:
    _, checked = role_checks
    workspaces = _workspace_repository("team-b", "other")
    entities = AsyncMock(spec=EntityRepositoryInterface)

    with patch("nhx.core.entities.api.v2.workspaces.endpoints.bindings_cache_delete", AsyncMock()):
        member = await add_workspace_member(
            "other",
            WorkspaceMemberInput(principal="service-account:team-b--ci", roles=[]),
            workspaces,
            entities,
            _auth_client(),
            _IDENTITIES,
            wait_role_propagation=False,
        )

    assert member.roles == []
    assert checked == []
    entities.create_entity.assert_not_awaited()


@pytest.mark.asyncio
async def test_adding_a_member_refuses_another_workspaces_admin_before_writing(role_checks) -> None:
    held, _ = role_checks
    held.add(("m", "Admin"))
    entities = AsyncMock(spec=EntityRepositoryInterface)

    with pytest.raises(HTTPException) as raised:
        await add_workspace_member(
            "m",
            WorkspaceMemberInput(principal="service-account:team-b--ci", roles=["Editor"]),
            _workspace_repository("team-b", "m"),
            entities,
            _auth_client(),
            _IDENTITIES,
            wait_role_propagation=False,
        )

    assert raised.value.status_code == 403
    entities.get_entity_by_name.assert_not_awaited()
    entities.create_entity.assert_not_awaited()
    entities.update_entity.assert_not_awaited()


def _bindings(*roles: str) -> AsyncMock:
    """An entity repository holding the service account's active bindings for ``roles`` in workspace 'm'."""
    entities = AsyncMock(spec=EntityRepositoryInterface)
    entities.list_entities.return_value = (
        [
            SimpleNamespace(
                id=f"binding-{role}",
                data={"principal": "service-account:team-b--ci", "role": role, "revoked_at": None},
            )
            for role in roles
        ],
        None,
    )
    return entities


@pytest.mark.asyncio
async def test_updating_a_member_refuses_another_workspaces_admin_adding_a_role_before_writing(role_checks) -> None:
    held, _ = role_checks
    held.add(("m", "Admin"))
    entities = _bindings("Viewer")

    with pytest.raises(HTTPException) as raised:
        await update_workspace_member(
            "m",
            "service-account:team-b--ci",
            WorkspaceMemberUpdate(roles=["Viewer", "Editor"]),
            _workspace_repository("team-b", "m"),
            entities,
            _auth_client(),
            _IDENTITIES,
            wait_role_propagation=False,
        )

    assert raised.value.status_code == 403
    entities.create_entity.assert_not_awaited()
    entities.update_entity.assert_not_awaited()


@pytest.mark.asyncio
async def test_updating_a_member_to_only_remove_roles_needs_no_namespace_check(role_checks) -> None:
    _, checked = role_checks
    entities = _bindings("Viewer", "Editor")

    with patch("nhx.core.entities.api.v2.workspaces.endpoints.bindings_cache_delete", AsyncMock()):
        member = await update_workspace_member(
            "m",
            "service-account:team-b--ci",
            WorkspaceMemberUpdate(roles=["Viewer"]),
            _workspace_repository("team-b", "m"),
            entities,
            _auth_client(),
            _IDENTITIES,
            wait_role_propagation=False,
        )

    assert member.roles == ["Viewer"]
    assert checked == []
    entities.update_entity.assert_awaited_once()
    assert entities.update_entity.await_args.kwargs["entity_id"] == "binding-Editor"
    entities.create_entity.assert_not_awaited()
