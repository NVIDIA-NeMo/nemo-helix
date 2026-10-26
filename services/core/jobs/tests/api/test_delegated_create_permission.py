# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from nhx.common.auth.models import Principal
from nhx.core.jobs.api.v2.jobs.endpoints import require_delegated_create_permission

DELEGATED = Principal(id="service:evals", on_behalf_of="alice@example.com", on_behalf_of_groups=["evals-team"])


def _auth_client(principal: Principal, *, allowed: bool = True) -> Mock:
    return Mock(principal=principal, on_behalf_of_has_permissions=AsyncMock(return_value=allowed))


async def test_delegated_create_without_permission_is_forbidden() -> None:
    auth_client = _auth_client(DELEGATED, allowed=False)

    with pytest.raises(HTTPException) as exc_info:
        await require_delegated_create_permission(auth_client, "team-ws")

    assert exc_info.value.status_code == 403
    assert "alice@example.com" in exc_info.value.detail
    auth_client.on_behalf_of_has_permissions.assert_awaited_once_with("team-ws", ["jobs.create"])


async def test_delegated_create_with_permission_is_allowed() -> None:
    auth_client = _auth_client(DELEGATED)

    await require_delegated_create_permission(auth_client, "team-ws")

    auth_client.on_behalf_of_has_permissions.assert_awaited_once_with("team-ws", ["jobs.create"])


@pytest.mark.parametrize(
    "principal",
    [
        pytest.param(Principal(id="service:evals"), id="service"),
        pytest.param(Principal(id="alice@example.com"), id="user"),
        pytest.param(Principal(id="alice@example.com", on_behalf_of="bob@example.com"), id="user-delegating"),
    ],
)
async def test_undelegated_or_user_create_is_left_to_the_route_gate(principal: Principal) -> None:
    auth_client = _auth_client(principal, allowed=False)

    await require_delegated_create_permission(auth_client, "team-ws")

    auth_client.on_behalf_of_has_permissions.assert_not_awaited()
