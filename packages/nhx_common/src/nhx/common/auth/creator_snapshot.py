# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted request capabilities for a stored creator identity snapshot."""

from dataclasses import dataclass
from typing import Self

from nhx.common.auth.client import AuthClient


@dataclass(frozen=True)
class CreatorSnapshotCapabilities:
    """Read and write capabilities derived from the authenticated HTTP caller."""

    policy_enabled: bool
    can_read: bool
    can_write: bool

    @classmethod
    def from_auth_client(cls, auth_client: AuthClient) -> Self:
        if not auth_client.auth_enabled:
            return cls(policy_enabled=False, can_read=True, can_write=True)

        principal = auth_client.principal
        service_caller = principal.caller_kind == "service_principal"
        return cls(
            policy_enabled=True,
            can_read=service_caller and not principal.is_delegated,
            can_write=service_caller,
        )
