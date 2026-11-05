# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport corruption or policy outages never turn into an authorization grant."""

import json

import httpx
import pytest
from fastapi import HTTPException
from nemo_helix_plugin.auth.decisions import DecisionRequest, ResourceTarget
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.testing.policy import mock_operation_response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "outage",
        "wrong-operation",
        "wrong-target",
        "wrong-context",
        "wrong-version",
        "bad-boolean",
        "extra-allow",
        "missing-tag",
    ],
)
async def test_policy_failure_never_retries_with_other_credentials(failure: str) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        document = json.loads(request.content)["input"]
        if failure == "outage":
            return httpx.Response(503)
        payload = json.loads(mock_operation_response(document, allowed=True).content)
        result = payload["result"]
        if failure == "wrong-operation":
            result["operation"] = "another.operation"
        elif failure == "wrong-target":
            result["target"]["workspace"] = "other"
        elif failure == "wrong-context":
            result["principal_context_digest"] = "other"
        elif failure == "wrong-version":
            result["protocol_version"] = 1
        elif failure == "bad-boolean":
            result["allowed"] = "true"
        elif failure == "extra-allow":
            result["bypass"] = True
        elif failure == "missing-tag":
            result.pop("decision_type")
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = AuthClient(
            principal=Principal(id="service:jobs", on_behalf_of="alice"),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp.invalid"),
            http_client=transport,
            service_name="jobs",
        )
        with pytest.raises(HTTPException) as exc:
            await client.has_operation(
                DecisionRequest(
                    operation="example.read",
                    target=ResourceTarget(workspace="team", resource_type="example", name="item"),
                )
            )
        assert exc.value.status_code == 503
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/decision")


@pytest.mark.asyncio
async def test_named_decision_preserves_scopes_from_the_route_check() -> None:
    observed: list[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        document = json.loads(request.content)["input"]
        observed.append(document["scopes"])
        return mock_operation_response(document, allowed=True)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = AuthClient(
            principal=Principal(id="alice"),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp.invalid"),
            http_client=transport,
            service_name="jobs",
        )
        await client.authorize_request("GET", "/known", scopes=["jobs:read"])
        await client.has_operation(
            DecisionRequest(
                operation="example.read", target=ResourceTarget(workspace="team", resource_type="example", name="item")
            )
        )
    assert observed == [["jobs:read"], ["jobs:read"]]
