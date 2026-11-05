# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell binding preflight uses existing auth and Secrets APIs only."""

import json
from unittest.mock import patch

import httpx
import pytest
from nemo_deployments_plugin.backends.openshell import authorization as deployment_authorization
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.client.errors import NemoHTTPError, NotFoundError
from nemo_helix_plugin.openshell import authorization as shared_authorization
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.core.jobs.controllers.backends.openshell import authorization as job_authorization
from nhx.testing.policy import PolicyHarness


@pytest.fixture(params=[job_authorization, deployment_authorization], ids=["job", "deployment"])
def authorization(request):
    return request.param


@pytest.mark.parametrize("outcome", ["allowed", "denied", "unavailable", "missing", "wrong-scope", "disabled"])
async def test_binding_preflight_checks_creator_before_metadata(outcome, authorization):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path.endswith("/decision"):
            # Evaluate the signed creator, never the privileged HTTP caller.
            payload = json.loads(request.content)["input"]
            assert payload["principal_id"] in {"service:jobs", "service:deployments"}
            assert payload["operation"] == "secrets.binding.authorize"
            assert payload["target"]["workspace"] == "team"
            assert payload["facts"] == {"kind": "creator", "principal_id": "user:creator"}
            if outcome == "unavailable":
                return httpx.Response(503)
            return httpx.Response(
                200,
                json={
                    "result": {
                        "protocol_version": 2,
                        "decision_type": "operation",
                        "allowed": outcome != "denied",
                        "operation": payload["operation"],
                        "target": payload["target"],
                        "principal_context_digest": payload["principal_context_digest"],
                        "policy_revision": "test-policy",
                        "reason": "policy_decision",
                    }
                },
            )
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
        with patch.object(shared_authorization, "platform_auth_enabled", return_value=True) as enabled:
            if outcome in {"allowed", "disabled"}:
                await authorization.authorize_secret_binding(
                    workspace="team",
                    name="token",
                    creator=None if outcome == "disabled" else AuthContext(principal_id="user:creator"),
                    secrets_client=client,
                )
            else:
                expected_error = {"unavailable": NemoHTTPError, "missing": NotFoundError}.get(outcome, PermissionError)
                with pytest.raises(expected_error):
                    await authorization.authorize_secret_binding(
                        workspace="team",
                        name="token",
                        creator=AuthContext(principal_id="user:creator"),
                        secrets_client=client,
                    )
            if outcome == "disabled":
                enabled.assert_not_called()
    assert len(requests) == (1 if outcome in {"denied", "unavailable", "disabled"} else 2)
    assert not any(request.url.path.endswith(("/access", "/authorization")) for request in requests)


@pytest.mark.parametrize("enabled,principal", [(False, "user:creator"), (True, "service:jobs")])
async def test_enabled_binding_cannot_skip_creator_check(enabled, principal, authorization):
    policy = PolicyHarness()

    def respond(request):
        if not enabled:
            pytest.fail("Authentication mode mismatch must fail before a policy query")
        assert request.url.path.endswith("/decision")
        return policy.respond(request)

    transport = httpx.MockTransport(respond)
    async with AsyncSecretsClient(
        base_url="https://helix.example", http_client=httpx.AsyncClient(transport=transport), owns_http_client=True
    ) as client:
        with patch.object(shared_authorization, "platform_auth_enabled", return_value=enabled):
            with pytest.raises(PermissionError):
                await authorization.authorize_secret_binding(
                    workspace="team", name="token", creator=AuthContext(principal_id=principal), secrets_client=client
                )


@pytest.mark.parametrize(
    "membership", ["creator-subject", "creator@example.test", "creator-group", "creator-account", "creator-alias"]
)
@pytest.mark.parametrize("delegated", [False, True])
async def test_preflight_resolves_the_complete_captured_creator(membership, delegated):
    policy = PolicyHarness()
    policy.document["authz"]["roles"]["CreatorReader"] = {"permissions": ["secrets.read"]}
    policy.document["authz"]["principals"][membership] = {"workspaces": {"team": ["CreatorReader"]}}
    policy.install(policy.document)
    creator = AuthContext(
        principal_id="service:submitter" if delegated else "creator-subject",
        principal_email=None if delegated else "creator@example.test",
        principal_groups=["actor-group"] if delegated else ["creator-group"],
        principal_account_id=None if delegated else "creator-account",
        principal_authz_aliases=[] if delegated else ["creator-alias"],
        principal_on_behalf_of="creator-subject" if delegated else None,
        principal_on_behalf_of_email="creator@example.test" if delegated else None,
        principal_on_behalf_of_groups=["creator-group"] if delegated else None,
        principal_on_behalf_of_account_id="creator-account" if delegated else None,
        principal_on_behalf_of_authz_aliases=["creator-alias"] if delegated else [],
    )
    async with AsyncSecretsClient(
        base_url="https://helix.example",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(policy.respond)),
        owns_http_client=True,
    ) as client:
        await shared_authorization.require_creator_secret_access(
            workspace="team", name="token", creator=creator, actor_service="jobs", secrets_client=client
        )
        with pytest.raises(PermissionError):
            await shared_authorization.require_creator_secret_access(
                workspace="other", name="token", creator=creator, actor_service="jobs", secrets_client=client
            )
        policy.document["authz"]["principals"].pop(membership)
        policy.document["authz"]["principals"]["actor-group"] = {"workspaces": {"system": ["HelixAdmin"]}}
        policy.install(policy.document)
        with pytest.raises(PermissionError):
            await shared_authorization.require_creator_secret_access(
                workspace="team", name="token", creator=creator, actor_service="jobs", secrets_client=client
            )
