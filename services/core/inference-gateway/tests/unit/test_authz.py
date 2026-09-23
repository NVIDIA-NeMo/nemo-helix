# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for delegated (on-behalf-of) workspace access enforcement on the proxy path."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from nmp.common.auth.dependencies import auth_client_context
from nmp.common.auth.models import Principal
from nmp.core.inference_gateway.api.authz import (
    MODEL_EXEC_PERMISSION,
    OPENAI_EXEC_PERMISSION,
    enforce_delegated_workspace_access,
    may_use_from_workspace,
)


def _auth_client(principal: Principal, *, enabled: bool = True, allowed: bool = True) -> MagicMock:
    client = MagicMock()
    client.auth_enabled = enabled
    client.principal = principal
    client.on_behalf_of_has_permissions = AsyncMock(return_value=allowed)
    return client


@pytest.fixture(autouse=True)
def _clear_auth_context() -> Iterator[None]:
    token = auth_client_context.set(None)
    try:
        yield
    finally:
        auth_client_context.reset(token)


@pytest.mark.asyncio
async def test_no_auth_context_is_noop() -> None:
    # No middleware / no context: the route gate already decided; do not raise.
    await enforce_delegated_workspace_access("ws", OPENAI_EXEC_PERMISSION)


@pytest.mark.asyncio
async def test_auth_disabled_is_noop() -> None:
    client = _auth_client(Principal(id="service:agents", on_behalf_of="user:alice"), enabled=False, allowed=False)
    auth_client_context.set(client)
    # Auth disabled short-circuits before any PDP call.
    await enforce_delegated_workspace_access("ws", OPENAI_EXEC_PERMISSION)
    client.on_behalf_of_has_permissions.assert_not_awaited()


@pytest.mark.asyncio
async def test_plain_user_is_noop() -> None:
    # A non-service principal was already gated by the route gate as itself.
    client = _auth_client(Principal(id="user:alice", email="alice@example.com"), allowed=False)
    auth_client_context.set(client)
    await enforce_delegated_workspace_access("ws", OPENAI_EXEC_PERMISSION)
    client.on_behalf_of_has_permissions.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_delegated_service_principal_is_noop() -> None:
    # A service principal that is NOT delegating keeps its internal bypass.
    client = _auth_client(Principal(id="service:agents"), allowed=False)
    auth_client_context.set(client)
    await enforce_delegated_workspace_access("ws", OPENAI_EXEC_PERMISSION)
    client.on_behalf_of_has_permissions.assert_not_awaited()


@pytest.mark.asyncio
async def test_delegated_service_principal_allowed_when_obo_user_has_permission() -> None:
    client = _auth_client(Principal(id="service:agents", on_behalf_of="user:alice"), allowed=True)
    auth_client_context.set(client)
    await enforce_delegated_workspace_access("ws", OPENAI_EXEC_PERMISSION)
    client.on_behalf_of_has_permissions.assert_awaited_once_with("ws", [OPENAI_EXEC_PERMISSION])


@pytest.mark.asyncio
async def test_delegated_service_principal_denied_when_obo_user_lacks_permission() -> None:
    client = _auth_client(Principal(id="service:agents", on_behalf_of="user:mallory"), allowed=False)
    auth_client_context.set(client)
    with pytest.raises(HTTPException) as exc:
        await enforce_delegated_workspace_access("secret-ws", OPENAI_EXEC_PERMISSION)
    assert exc.value.status_code == 403
    assert "secret-ws" in exc.value.detail
    client.on_behalf_of_has_permissions.assert_awaited_once_with("secret-ws", [OPENAI_EXEC_PERMISSION])


# --------------------------------------------------------------------------- #
# may_use_from_workspace — entities outside the request workspace             #
# --------------------------------------------------------------------------- #


def _user_client(*, allowed: bool, enabled: bool = True) -> MagicMock:
    client = _auth_client(Principal(id="user:bob", email="bob@example.com"), enabled=enabled)
    client.has_permissions = AsyncMock(return_value=allowed)
    return client


@pytest.mark.asyncio
async def test_entity_in_the_request_workspace_needs_no_extra_check() -> None:
    client = _user_client(allowed=False)
    auth_client_context.set(client)

    assert await may_use_from_workspace("team-a", "team-a", OPENAI_EXEC_PERMISSION) is True
    client.has_permissions.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_entitled_in_the_entity_workspace_may_use_it() -> None:
    client = _user_client(allowed=True)
    auth_client_context.set(client)

    assert await may_use_from_workspace("team-a", "default", OPENAI_EXEC_PERMISSION) is True
    client.has_permissions.assert_awaited_once_with("default", [OPENAI_EXEC_PERMISSION])


@pytest.mark.asyncio
async def test_user_not_entitled_in_the_entity_workspace_may_not_use_it() -> None:
    """Being allowed in the request workspace says nothing about the shared one."""
    auth_client_context.set(_user_client(allowed=False))

    assert await may_use_from_workspace("team-a", "default", OPENAI_EXEC_PERMISSION) is False


@pytest.mark.asyncio
async def test_delegated_service_principal_is_checked_as_its_user() -> None:
    client = _auth_client(Principal(id="service:agents", on_behalf_of="user:mallory"), allowed=False)
    auth_client_context.set(client)

    assert await may_use_from_workspace("team-a", "default", MODEL_EXEC_PERMISSION) is False
    client.on_behalf_of_has_permissions.assert_awaited_once_with("default", [MODEL_EXEC_PERMISSION])


@pytest.mark.asyncio
async def test_non_delegated_service_principal_keeps_its_bypass() -> None:
    auth_client_context.set(_auth_client(Principal(id="service:agents"), allowed=False))

    assert await may_use_from_workspace("team-a", "default", OPENAI_EXEC_PERMISSION) is True


@pytest.mark.asyncio
async def test_auth_disabled_allows_other_workspaces() -> None:
    auth_client_context.set(_user_client(allowed=False, enabled=False))

    assert await may_use_from_workspace("team-a", "default", OPENAI_EXEC_PERMISSION) is True
