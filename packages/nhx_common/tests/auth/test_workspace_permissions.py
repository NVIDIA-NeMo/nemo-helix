# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch workspace permissions retain explicit authority and fail closed."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from nhx.common.auth.client import AuthClient
from nhx.common.auth.exceptions import InvalidPermissionFormatError
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from pydantic import JsonValue


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
async def test_workspace_filter_sends_one_explicit_query(delegated: bool) -> None:
    calls: list[dict[str, JsonValue]] = []

    def evaluate(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["input"])
        return httpx.Response(200, json={"result": {"allowed_workspaces": ["system"]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(evaluate)) as http_client:
        client = AuthClient(
            principal=Principal(
                id="service:auth" if delegated else "admin", on_behalf_of="admin" if delegated else None
            ),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp"),
            http_client=http_client,
        )
        requirements = {"subject": ["iam.list"]} if delegated else {"actor": ["iam.list"]}
        assert await client.filter_workspaces({"system", "private"}, **requirements) == {"system"}
    assert len(calls) == 1
    assert calls[0]["workspaces"] == ["private", "system"]
    assert "workspace" not in calls[0]
    assert calls[0]["subject_permissions" if delegated else "actor_permissions"] == ["iam.list"]
    assert ("actor_permissions" if delegated else "subject_permissions") not in calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [None, True, {}, "system", ["system", 1], ["not-requested"]])
async def test_invalid_workspace_filter_response_fails_closed(allowed: JsonValue) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"result": {"allowed_workspaces": allowed}})
        )
    ) as http_client:
        client = AuthClient(
            principal=Principal(id="service:auth", on_behalf_of="admin"),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp"),
            http_client=http_client,
        )
        with pytest.raises(HTTPException) as denied:
            await client.filter_workspaces({"system"}, subject=["iam.list"])
    assert denied.value.status_code == 500


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
async def test_workspace_filter_requires_obo_even_for_empty_subject_and_auth_off(enabled: bool, empty: bool) -> None:
    client = AuthClient(principal=Principal(id="service:auth"), config=AuthConfig(enabled=enabled))
    with pytest.raises(ValueError, match="on-behalf-of"):
        await client.filter_workspaces(set() if empty else {"system"}, subject=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_workspace_filter_validates_before_shortcuts(enabled: bool) -> None:
    client = AuthClient(principal=Principal(id="admin"), config=AuthConfig(enabled=enabled))
    with pytest.raises(ValueError, match="At least one"):
        await client.filter_workspaces(set())
    with pytest.raises(InvalidPermissionFormatError):
        await client.filter_workspaces(set(), actor=["invalid permission"])


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_workspace_filter_shortcuts_do_not_query_pdp(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    query = AsyncMock()
    monkeypatch.setattr(AuthClient, "_pdp_query", query)
    client = AuthClient(principal=Principal(id="admin"), config=AuthConfig(enabled=enabled))
    candidates = set() if enabled else {"system", "private"}
    assert await client.filter_workspaces(candidates, actor=["iam.list"]) == candidates
    query.assert_not_awaited()


@pytest.mark.asyncio
async def test_workspace_filter_http_failure_has_no_authority_fallback() -> None:
    calls = 0

    def unavailable(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as http_client:
        client = AuthClient(
            principal=Principal(id="service:auth", on_behalf_of="admin"),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp"),
            http_client=http_client,
        )
        with pytest.raises(HTTPException) as denied:
            await client.filter_workspaces({"system", "private"}, subject=["iam.list"])
    assert denied.value.status_code == 503
    assert calls == 1
