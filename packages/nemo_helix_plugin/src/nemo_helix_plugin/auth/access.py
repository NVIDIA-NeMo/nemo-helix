# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate trusted PDP field decisions for generic repository/response enforcement."""

from dataclasses import dataclass
from typing import Self

from nemo_helix_plugin.api.response_policy import ResponsePolicy
from nemo_helix_plugin.auth import RuntimeAuthClient
from nemo_helix_plugin.auth.decisions import DecisionRequest, FieldDecision, FieldTarget
from pydantic import BaseModel


@dataclass(frozen=True)
class AuthContextAccessRule:
    policy_enabled: bool
    can_read: bool
    can_write: bool

    @classmethod
    def from_decision(cls, decision: FieldDecision, *, policy_enabled: bool) -> Self:
        return cls(policy_enabled=policy_enabled, can_read=decision.read, can_write=decision.write)

    @classmethod
    async def from_auth_client(cls, auth_client: RuntimeAuthClient, *, target: FieldTarget) -> Self:
        decision = await auth_client.decide(DecisionRequest(operation="entities.auth_context.access", target=target))
        if not isinstance(decision, FieldDecision):
            raise ValueError("Auth-context access requires a typed field decision")
        return cls.from_decision(decision, policy_enabled=auth_client.auth_enabled)


class AuthContextResponsePolicy[DTO: BaseModel](ResponsePolicy[DTO]):
    async def can_read(self, auth_client: RuntimeAuthClient) -> bool:
        decision = await AuthContextAccessRule.from_auth_client(
            auth_client, target=FieldTarget(workspace="-", resource_type=self.dto_type.__name__)
        )
        return decision.can_read
