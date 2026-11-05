# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python field adapter faithfully enforces a trusted PDP result."""

import pytest
from nemo_helix_plugin.auth.access import AuthContextAccessRule, AuthContextResponsePolicy
from nemo_helix_plugin.auth.decisions import (
    DecisionRequest,
    FieldDecision,
    FieldTarget,
    OperationDecision,
    ResourceTarget,
)
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from pydantic import BaseModel


class Item(BaseModel):
    auth_context: str | None = None


@pytest.mark.parametrize(("read", "write"), [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.asyncio
async def test_adapter_translates_pdp_flags_without_recalculating_identity(
    read: bool,
    write: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decide(self: AuthClient, request: DecisionRequest) -> FieldDecision:
        assert request.operation == "entities.auth_context.access"
        return FieldDecision(
            protocol_version=2,
            operation=request.operation,
            target=FieldTarget.model_validate(request.target),
            principal_context_digest="bound",
            policy_revision="revision",
            reason="policy_decision",
            decision_type="field",
            field="auth_context",
            read=read,
            write=write,
        )

    monkeypatch.setattr(AuthClient, "decide", decide)
    # Even a powerful-looking identity cannot override a denied PDP flag.
    client = AuthClient(
        principal=Principal(id="service:jobs", on_behalf_of="admin"),
        config=AuthConfig(enabled=True),
    )
    target = FieldTarget(workspace="team", resource_type="platform_job")
    access = await AuthContextAccessRule.from_auth_client(client, target=target)
    assert (access.policy_enabled, access.can_read, access.can_write) == (True, read, write)
    assert await AuthContextResponsePolicy(dto_type=Item, field="auth_context").can_read(client) is read


@pytest.mark.asyncio
async def test_adapter_rejects_wrong_decision_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    async def decide(self: AuthClient, request: DecisionRequest) -> OperationDecision:
        return OperationDecision(
            protocol_version=2,
            operation=request.operation,
            target=ResourceTarget(workspace="team", resource_type="platform_job", name="item"),
            principal_context_digest="bound",
            policy_revision="revision",
            reason="policy_decision",
            decision_type="operation",
            allowed=True,
        )

    monkeypatch.setattr(AuthClient, "decide", decide)
    client = AuthClient(principal=Principal(id="service:jobs"), config=AuthConfig(enabled=True))
    with pytest.raises(ValueError, match="typed field decision"):
        await AuthContextAccessRule.from_auth_client(
            client, target=FieldTarget(workspace="team", resource_type="platform_job")
        )


@pytest.mark.asyncio
async def test_explicit_auth_off_uses_shared_typed_bypass() -> None:
    client = AuthClient(principal=Principal(), config=AuthConfig(enabled=False))
    access = await AuthContextAccessRule.from_auth_client(
        client, target=FieldTarget(workspace="team", resource_type="platform_job")
    )
    assert (access.policy_enabled, access.can_read, access.can_write) == (False, True, True)
