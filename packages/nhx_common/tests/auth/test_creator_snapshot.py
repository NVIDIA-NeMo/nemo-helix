# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Creator snapshot capability decisions use only trusted request identity."""

import pytest
from nhx.common.auth.client import AuthClient
from nhx.common.auth.creator_snapshot import CreatorSnapshotCapabilities
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig


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
        (Principal(id="service:jobs"), False, (False, True, True)),
        (Principal(id="user@example.com"), False, (False, True, True)),
        (Principal(), False, (False, True, True)),
    ],
)
def test_creator_snapshot_capabilities(
    principal: Principal,
    auth_enabled: bool,
    expected: tuple[bool, bool, bool],
) -> None:
    auth_client = AuthClient(principal=principal, config=AuthConfig(enabled=auth_enabled))

    capabilities = CreatorSnapshotCapabilities.from_auth_client(auth_client)

    assert (capabilities.policy_enabled, capabilities.can_read, capabilities.can_write) == expected
