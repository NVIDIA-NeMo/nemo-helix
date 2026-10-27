# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted request access to persisted auth context."""

from dataclasses import dataclass
from typing import Self

from nemo_helix_plugin.api.response_policy import ResponsePolicy
from nemo_helix_plugin.auth import RuntimeAuthClient
from pydantic import BaseModel


@dataclass(frozen=True)
class AuthContextAccessRule:
    policy_enabled: bool
    can_read: bool
    can_write: bool

    @classmethod
    def from_auth_client(cls, auth_client: RuntimeAuthClient) -> Self:
        if not auth_client.auth_enabled:
            return cls(policy_enabled=False, can_read=True, can_write=True)

        principal = auth_client.principal
        service_caller = principal.caller_kind == "service_principal"
        return cls(
            policy_enabled=True,
            can_read=service_caller and not principal.is_delegated,
            can_write=service_caller,
        )


class AuthContextResponsePolicy[DTO: BaseModel](ResponsePolicy[DTO]):
    def can_read(self, auth_client: RuntimeAuthClient) -> bool:
        return AuthContextAccessRule.from_auth_client(auth_client).can_read
