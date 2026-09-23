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
from nmp.common.entities.utils import ParsedEntityRef
from nmp.core.inference_gateway.api.authz import (
    MODEL_EXEC_PERMISSION,
    OPENAI_EXEC_PERMISSION,
    enforce_delegated_workspace_access,
    enforce_model_ref_access,
    model_ref_workspaces,
)


def _auth_client(principal: Principal, *, enabled: bool = True, allowed: bool = True) -> MagicMock:
    client = MagicMock()
    client.auth_enabled = enabled
    client.principal = principal
    client.on_behalf_of_has_permissions = AsyncMock(return_value=allowed)
    client.has_permissions = AsyncMock(return_value=allowed)
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


@pytest.mark.asyncio
async def test_model_ref_same_workspace_is_noop() -> None:
    client = _auth_client(Principal(id="user:alice", email="alice@example.com"), allowed=False)
    auth_client_context.set(client)
    await enforce_model_ref_access("carol-ws", ParsedEntityRef(workspace="carol-ws", name="gpt"))
    client.has_permissions.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_ref_no_auth_context_is_noop() -> None:
    await enforce_model_ref_access("carol-ws", ParsedEntityRef(workspace="default", name="gpt"))


@pytest.mark.asyncio
async def test_model_ref_auth_disabled_is_noop() -> None:
    client = _auth_client(Principal(id="user:alice", email="alice@example.com"), enabled=False, allowed=False)
    auth_client_context.set(client)
    await enforce_model_ref_access("carol-ws", ParsedEntityRef(workspace="default", name="gpt"))
    client.has_permissions.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_ref_cross_workspace_allowed_when_caller_has_permission() -> None:
    client = _auth_client(Principal(id="user:alice", email="alice@example.com"), allowed=True)
    auth_client_context.set(client)
    await enforce_model_ref_access("carol-ws", ParsedEntityRef(workspace="default", name="gpt"))
    client.has_permissions.assert_awaited_once_with("default", [MODEL_EXEC_PERMISSION])


@pytest.mark.asyncio
async def test_model_ref_cross_workspace_denied_when_caller_lacks_permission() -> None:
    client = _auth_client(Principal(id="user:carol", email="carol@example.com"), allowed=False)
    auth_client_context.set(client)
    with pytest.raises(HTTPException) as exc:
        await enforce_model_ref_access("carol-ws", ParsedEntityRef(workspace="default", name="gpt"))
    assert exc.value.status_code == 403
    assert "default" in exc.value.detail
    client.has_permissions.assert_awaited_once_with("default", [MODEL_EXEC_PERMISSION])


@pytest.mark.asyncio
async def test_model_ref_delegated_service_principal_checked_as_on_behalf_of_user() -> None:
    client = _auth_client(Principal(id="service:agents", on_behalf_of="user:carol"), allowed=False)
    auth_client_context.set(client)
    with pytest.raises(HTTPException) as exc:
        await enforce_model_ref_access("carol-ws", ParsedEntityRef(workspace="secret-ws", name="gpt"))
    assert exc.value.status_code == 403
    client.on_behalf_of_has_permissions.assert_awaited_once_with("secret-ws", [MODEL_EXEC_PERMISSION])
    client.has_permissions.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_ref_lora_adapter_workspace_is_checked() -> None:
    client = _auth_client(Principal(id="user:carol", email="carol@example.com"), allowed=False)
    auth_client_context.set(client)
    with pytest.raises(HTTPException) as exc:
        await enforce_model_ref_access(
            "carol-ws", ParsedEntityRef(workspace="carol-ws", name="base&adapters/secret-ws/private-adapter")
        )
    assert exc.value.status_code == 403
    assert "secret-ws" in exc.value.detail
    client.has_permissions.assert_awaited_once_with("secret-ws", [MODEL_EXEC_PERMISSION])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gpt", ["base-ws"]),
        ("base&adapters/adapter-ws/adapter", ["base-ws", "adapter-ws"]),
        ("base&adapters/base-ws/adapter", ["base-ws"]),
        ("base&adapters/malformed", ["base-ws"]),
    ],
)
def test_model_ref_workspaces(name: str, expected: list[str]) -> None:
    assert model_ref_workspaces(ParsedEntityRef(workspace="base-ws", name=name)) == expected
