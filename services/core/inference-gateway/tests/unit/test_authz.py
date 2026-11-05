# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved model/adapter workspaces use canonical operation decisions, including the path workspace."""

import httpx
import pytest
from fastapi import HTTPException
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.common.entities.utils import ParsedEntityRef
from nhx.core.inference_gateway.api.authz import enforce_model_ref_access, model_ref_workspaces
from nhx.testing.policy import PolicyHarness


@pytest.fixture
def policy() -> PolicyHarness:
    harness = PolicyHarness()
    harness.document["authz"]["roles"]["CanInfer"] = {"permissions": ["inference.gateway.model.exec"]}
    harness.document["authz"]["principals"]["alice"] = {"workspaces": {"allowed": ["CanInfer"]}}
    harness.document["authz"]["principals"]["actor-admins"] = {"workspaces": {"system": ["HelixAdmin"]}}
    harness.install(harness.document)
    return harness


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actor,subject,target,allowed",
    [
        ("alice", None, "allowed", True),
        ("alice", None, "private", False),
        ("service:agents", "alice", "allowed", True),
        ("service:agents", "alice", "private", False),
        ("service:agents", "missing", "allowed", False),
        ("service:agents", None, "private", True),
    ],
)
async def test_resolved_workspace_authority(
    policy, actor: str, subject: str | None, target: str, allowed: bool
) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(policy.respond)) as transport:
        auth = AuthClient(
            principal=Principal(id=actor, on_behalf_of=subject, groups=["actor-admins"] if subject else []),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp.invalid"),
            http_client=transport,
            service_name="inference-gateway",
        )
        token = auth_client_context.set(auth)
        try:
            if allowed:
                await enforce_model_ref_access(target, ParsedEntityRef(workspace=target, name="model"))
            else:
                with pytest.raises(HTTPException) as exc:
                    await enforce_model_ref_access(target, ParsedEntityRef(workspace=target, name="model"))
                assert exc.value.status_code == 403
        finally:
            auth_client_context.reset(token)


@pytest.mark.asyncio
async def test_adapter_target_is_checked_in_its_own_workspace(policy) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(policy.respond)) as transport:
        auth = AuthClient(
            principal=Principal(id="alice"),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp.invalid"),
            http_client=transport,
            service_name="inference-gateway",
        )
        token = auth_client_context.set(auth)
        try:
            with pytest.raises(HTTPException) as exc:
                await enforce_model_ref_access(
                    "allowed", ParsedEntityRef(workspace="allowed", name="model&adapters/private/adapter")
                )
            assert "private" in exc.value.detail
        finally:
            auth_client_context.reset(token)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model", ["base"]),
        ("model&adapters/other/adapter", ["base", "other"]),
        ("model&adapters/base/adapter", ["base"]),
        ("model&adapters/malformed", ["base"]),
    ],
)
def test_model_ref_workspaces(name: str, expected: list[str]) -> None:
    assert model_ref_workspaces(ParsedEntityRef(workspace="base", name=name)) == expected
