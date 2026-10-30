# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell binding preflight uses existing auth and Secrets APIs only."""

import json
from unittest.mock import patch

import httpx
import pytest
from nemo_deployments_plugin.backends.openshell import authorization as deployment_authorization
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.core.jobs.controllers.backends.openshell import authorization as job_authorization


@pytest.fixture(params=[job_authorization, deployment_authorization], ids=["job", "deployment"])
def authorization(request):
    return request.param


@pytest.mark.parametrize("outcome", ["allowed", "denied", "unavailable", "missing", "wrong-scope", "disabled"])
async def test_binding_preflight_checks_creator_before_metadata(outcome, authorization):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/has_permissions"):
            # Evaluate the signed creator, never the privileged HTTP caller.
            payload = json.loads(request.content)["input"]
            assert payload["principal_id"] == "user:creator"
            assert payload["workspace"] == "team"
            assert payload["permissions"] == ["secrets.read"]
            if outcome == "unavailable":
                return httpx.Response(503)
            return httpx.Response(200, json={"result": {"allowed": outcome != "denied"}})
        assert request.method == "GET"
        assert request.url.path == "/apis/secrets/v2/workspaces/team/secrets/token"
        if outcome == "missing":
            return httpx.Response(404)
        return httpx.Response(200, json={"name": "token", "workspace": "other" if outcome == "wrong-scope" else "team"})

    async with AsyncSecretsClient(
        base_url="https://helix.example",
        default_headers={"X-NHX-Principal-Id": "service:deployments"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        owns_http_client=True,
    ) as client:
        with patch.object(authorization, "platform_auth_enabled", return_value=True) as enabled:
            if outcome in {"allowed", "disabled"}:
                await authorization.authorize_secret_binding(
                    workspace="team",
                    name="token",
                    principal_id=None if outcome == "disabled" else "user:creator",
                    secrets_client=client,
                )
            else:
                expected_error = {"unavailable": NemoHTTPError, "missing": NotFoundError}.get(outcome, PermissionError)
                with pytest.raises(expected_error):
                    await authorization.authorize_secret_binding(
                        workspace="team",
                        name="token",
                        principal_id="user:creator",
                        secrets_client=client,
                    )
            if outcome == "disabled":
                enabled.assert_not_called()
    assert len(requests) == (1 if outcome in {"denied", "unavailable", "disabled"} else 2)
    assert not any(request.url.path.endswith(("/access", "/authorization")) for request in requests)


@pytest.mark.parametrize("enabled,principal", [(False, "user:creator"), (True, "service:jobs")])
async def test_enabled_binding_cannot_skip_creator_check(enabled, principal, authorization):
    transport = httpx.MockTransport(lambda request: pytest.fail("No HTTP request should be made"))
    async with AsyncSecretsClient(
        base_url="https://helix.example", http_client=httpx.AsyncClient(transport=transport), owns_http_client=True
    ) as client:
        with patch.object(authorization, "platform_auth_enabled", return_value=enabled):
            with pytest.raises(PermissionError):
                await authorization.authorize_secret_binding(
                    workspace="team", name="token", principal_id=principal, secrets_client=client
                )
