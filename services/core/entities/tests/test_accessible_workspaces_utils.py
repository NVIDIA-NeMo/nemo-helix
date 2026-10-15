# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for workspace access helpers in api/v2/utils.py."""

from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nhx.common.auth import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig, Configuration
from nhx.core.entities.api.v2.utils import (
    _applicable_principal_strings,
    clear_principal_bindings_cache,
    get_accessible_workspaces,
    require_workspace_access,
)
from nhx.core.entities.config import EntitiesConfig
from nhx.core.entities.entities import Entity

TEST_PRINCIPAL = "test-user@example.com"


def _binding_entity(
    workspace: str, role: str, revoked_at: str | None = None, *, principal: str = TEST_PRINCIPAL
) -> Entity:
    now = datetime.now(timezone.utc)
    return Entity(
        entity_type="role_binding",
        id=f"binding-{workspace}-{role}",
        workspace=workspace,
        name=f"{TEST_PRINCIPAL}-{workspace}-{role}",
        data={
            "principal": principal,
            "workspace": workspace,
            "role": role,
            "revoked_at": revoked_at,
        },
        created_at=now,
        updated_at=now,
        db_version=1,
    )


def test_applicable_principal_strings_id_only() -> None:
    p = Principal(id="172d75ab-0866-4d10-b3ab-c42e37bf20b4", email=None, groups=[])
    assert _applicable_principal_strings(p) == ["172d75ab-0866-4d10-b3ab-c42e37bf20b4"]


def test_applicable_principal_strings_id_and_distinct_email() -> None:
    p = Principal(
        id="172d75ab-0866-4d10-b3ab-c42e37bf20b4",
        email="user@example.com",
        groups=[],
    )
    assert _applicable_principal_strings(p) == [
        "172d75ab-0866-4d10-b3ab-c42e37bf20b4",
        "user@example.com",
    ]


def test_applicable_principal_strings_dedupes_when_id_is_email_shaped() -> None:
    """If Principal-Id is already the email, do not duplicate."""
    p = Principal(id="same@example.com", email="same@example.com", groups=[])
    assert _applicable_principal_strings(p) == ["same@example.com"]


def test_applicable_principal_strings_includes_groups() -> None:
    p = Principal(
        id="sub-1",
        email="u@example.com",
        groups=["group-a", "group-b"],
    )
    assert _applicable_principal_strings(p) == ["sub-1", "u@example.com", "group-a", "group-b"]


def test_applicable_principal_strings_group_dedupes_against_id() -> None:
    p = Principal(id="dup", email=None, groups=["dup", "other"])
    assert _applicable_principal_strings(p) == ["dup", "other"]


@pytest.mark.asyncio
async def test_get_accessible_workspaces_excludes_revoked_bindings() -> None:
    bindings_by_principal = {
        TEST_PRINCIPAL: [
            _binding_entity("workspace-active", "viewer"),
            _binding_entity("workspace-revoked", "viewer", revoked_at="2026-01-01T00:00:00Z"),
        ],
    }

    async def list_entities(**kwargs):
        queried = kwargs["filter_op"].value
        return bindings_by_principal.get(queried, []), 0

    entity_repository = AsyncMock()
    entity_repository.list_entities.side_effect = list_entities

    principal = Principal(id=TEST_PRINCIPAL, email=None, groups=[])
    auth_client = AuthClient(principal=principal, config=AuthConfig(enabled=True), http_client=None)
    token = auth_client_context.set(auth_client)
    try:
        accessible = await get_accessible_workspaces(entity_repository)
    finally:
        auth_client_context.reset(token)

    assert accessible == {"workspace-active"}


@pytest.fixture
def workspace_auth() -> Iterator[AuthClient]:
    auth_client = AuthClient(principal=Principal(id=TEST_PRINCIPAL), config=AuthConfig(enabled=True))
    token = auth_client_context.set(auth_client)
    try:
        yield auth_client
    finally:
        auth_client_context.reset(token)


@pytest.fixture
async def cached_bindings_repository() -> AsyncIterator[tuple[AsyncMock, dict[str, list[Entity]]]]:
    Configuration.set_override(EntitiesConfig(principal_bindings_cache_enabled=True))
    await clear_principal_bindings_cache()
    bindings: dict[str, list[Entity]] = {}

    async def list_entities(**kwargs):
        matches = list(bindings.get(kwargs["filter_op"].value, []))
        return matches, len(matches)

    repository = AsyncMock()
    repository.list_entities.side_effect = list_entities
    try:
        yield repository, bindings
    finally:
        await clear_principal_bindings_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal", "binding_principal"),
    [
        (Principal(id=TEST_PRINCIPAL), TEST_PRINCIPAL),
        (Principal(id="user-id", email=TEST_PRINCIPAL), TEST_PRINCIPAL),
        (Principal(id=TEST_PRINCIPAL, groups=["test-group"]), "test-group"),
        (Principal(id=TEST_PRINCIPAL), "*"),
        (Principal(id="service:agents", on_behalf_of=TEST_PRINCIPAL), TEST_PRINCIPAL),
    ],
    ids=["id", "email", "group", "wildcard", "on-behalf-of"],
)
async def test_workspace_access_refreshes_stale_denial(
    cached_bindings_repository, workspace_auth: AuthClient, principal: Principal, binding_principal: str
) -> None:
    repository, bindings = cached_bindings_repository
    workspace_auth.principal = principal
    assert await get_accessible_workspaces(repository) == set()

    # Another replica grants access without invalidating this reader's cache.
    bindings[binding_principal] = [_binding_entity("new-workspace", "Admin", principal=binding_principal)]
    assert await require_workspace_access(repository, "new-workspace") == {"new-workspace"}

    repository.list_entities.reset_mock()
    assert await require_workspace_access(repository, "new-workspace") == {"new-workspace"}
    repository.list_entities.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("revoked", [False, True], ids=["no-binding", "revoked-binding"])
async def test_workspace_access_still_denies_after_refresh(
    cached_bindings_repository, workspace_auth: AuthClient, revoked: bool
) -> None:
    repository, bindings = cached_bindings_repository
    assert await get_accessible_workspaces(repository) == set()
    if revoked:
        bindings[TEST_PRINCIPAL] = [_binding_entity("denied-workspace", "Admin", revoked_at="2026-01-01T00:00:00Z")]
    repository.list_entities.reset_mock()

    with pytest.raises(HTTPException) as denied:
        await require_workspace_access(repository, "denied-workspace", detail="Access denied", status_code=422)

    assert denied.value.status_code == 422
    assert denied.value.detail == "Access denied"
    assert repository.list_entities.await_count == 2  # One refresh of the user and wildcard bindings.


@pytest.mark.asyncio
async def test_workspace_access_does_not_retry_when_cache_disabled(workspace_auth: AuthClient) -> None:
    repository = AsyncMock()
    repository.list_entities.return_value = ([], 0)

    with pytest.raises(HTTPException) as denied:
        await require_workspace_access(repository, "denied-workspace")

    assert denied.value.status_code == 403
    assert repository.list_entities.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_enabled", [False, True], ids=["auth-disabled", "service-principal"])
async def test_workspace_access_does_not_query_bindings_for_unscoped_access(
    cached_bindings_repository, workspace_auth: AuthClient, auth_enabled: bool
) -> None:
    repository, _ = cached_bindings_repository
    workspace_auth.config.enabled = auth_enabled
    if auth_enabled:
        workspace_auth.principal = Principal(id="service:agents")

    assert await require_workspace_access(repository, "new-workspace") is None
    repository.list_entities.assert_not_awaited()
