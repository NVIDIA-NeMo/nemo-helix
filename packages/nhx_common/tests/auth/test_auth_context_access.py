# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auth-context access decisions use only trusted request identity."""

import pytest
from nemo_helix_plugin.auth.access import AuthContextAccessRule, AuthContextResponsePolicy
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from pydantic import BaseModel


class Item(BaseModel):
    auth_context: str | None = None


@pytest.mark.parametrize(
    ("principal", "auth_enabled", "expected"),
    [
        (Principal(id="service:jobs"), True, (True, True, True)),
        (
            Principal(id="service:evals", on_behalf_of="user@example.com"),
            True,
            (True, False, True),
        ),
        (Principal(id="user@example.com"), True, (True, False, False)),
        (Principal(id="service-account:probe"), True, (True, False, False)),
        (Principal(id="service:jobs"), False, (False, True, True)),
        (Principal(id="user@example.com"), False, (False, True, True)),
        (Principal(), False, (False, True, True)),
    ],
)
def test_auth_context_access_rule(
    principal: Principal,
    auth_enabled: bool,
    expected: tuple[bool, bool, bool],
) -> None:
    auth_client = AuthClient(principal=principal, config=AuthConfig(enabled=auth_enabled))

    capabilities = AuthContextAccessRule.from_auth_client(auth_client)

    assert (capabilities.policy_enabled, capabilities.can_read, capabilities.can_write) == expected


@pytest.mark.parametrize(
    ("principal", "auth_enabled", "can_read"),
    [
        (Principal(id="service:jobs"), True, True),
        (Principal(id="service:jobs", on_behalf_of="user@example.com"), True, False),
        (Principal(id="user@example.com"), True, False),
        (Principal(), False, True),
    ],
)
def test_response_policy_uses_shared_access_rule(principal: Principal, auth_enabled: bool, can_read: bool) -> None:
    auth_client = AuthClient(principal=principal, config=AuthConfig(enabled=auth_enabled))
    policy = AuthContextResponsePolicy(dto_type=Item, field="auth_context")

    assert policy.can_read(auth_client) is can_read
