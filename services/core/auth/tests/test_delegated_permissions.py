# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actor grants cannot satisfy represented-user permissions in the shipped WASM."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
import yaml
from fastapi import HTTPException
from nemo_helix_plugin.authz import AuthzContribution, AuthzEndpointMethod
from nemo_helix_plugin.authz_merge import merge_authz_contributions
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.auth.app.embedded_pdp import OPAPolicy
from nhx.core.auth.app.embedded_pdp.policy_wasm import ensure_embedded_policy_wasm
from nhx.core.auth.app.scope_families import with_scope_families
from nhx.core.entities.api.v2.entity_access import (
    get_control_entity_list_workspaces,
    require_control_entity_access,
    require_entity_read_access,
)
from nhx.core.entities.app.repository import WorkspaceRepositoryInterface

ASSETS = Path(__file__).parents[1] / "src/nhx/core/auth/assets"


@pytest.fixture
def policy_data() -> dict[str, Any]:
    # OPA policy documents are extensible JSON data at the evaluator boundary.
    data = yaml.safe_load((ASSETS / "static-authz.yaml").read_text())
    data["authz"]["principals"] = {
        "viewer": {"workspaces": {"private": ["Viewer"]}},
        "service:admin": {"workspaces": {"private": ["Admin"]}},
        "service:platform-admin": {"workspaces": {"system": ["HelixAdmin"]}},
    }
    data["authz"]["principals"].update(
        {
            "administrator": {"workspaces": {"private": ["Admin"]}},
            "platform-admin": {"workspaces": {"system": ["HelixAdmin"]}},
            "service:storage": {"workspaces": {"private": ["ServiceEntityStore"]}},
            "workload:powerful": {"workspaces": {"system": ["HelixAdmin"], "private": ["Admin"]}},
            "service:viewer": {"workspaces": {"private": ["Viewer"]}},
        }
    )
    return with_scope_families(data)


@pytest.fixture(scope="session")
def policy_wasm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build current policy sources once, independent of checkout artifacts."""
    return ensure_embedded_policy_wasm(wasm_path=tmp_path_factory.mktemp("delegated-policy") / "policy.wasm")


@pytest.fixture
def policy(policy_data: dict[str, Any], policy_wasm: Path) -> OPAPolicy:
    policy = OPAPolicy(str(policy_wasm))
    policy.set_data(policy_data)
    return policy


@pytest.fixture
def pdp_calls() -> list[dict[str, Any]]:
    return []


@pytest_asyncio.fixture
async def pdp_client(policy: OPAPolicy, pdp_calls: list[dict[str, Any]]) -> AsyncIterator[httpx.AsyncClient]:
    def evaluate(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)["input"]
        pdp_calls.append(body)
        entrypoint = 0 if request.url.path.endswith("/allow") else 1
        result = policy.evaluate(body, entrypoint=entrypoint)
        return httpx.Response(200, json={"result": result[0]["result"]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(evaluate)) as client:
        yield client


def auth_client(pdp_client: httpx.AsyncClient, actor: str, subject: str | None = "viewer") -> AuthClient:
    return AuthClient(
        principal=Principal(id=actor, on_behalf_of=subject),
        config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp"),
        http_client=pdp_client,
    )


@pytest.mark.parametrize("actor", ["service:intake", "service:admin", "service:platform-admin", "workload:powerful"])
@pytest.mark.parametrize("subject", ["viewer", "non-member"])
@pytest.mark.parametrize("permission", ["iam.list", "workspaces.members.create"])
def test_delegated_query_cannot_use_actor_grants(policy: OPAPolicy, actor: str, subject: str, permission: str) -> None:
    result = policy.evaluate(
        {
            "principal_id": actor,
            "on_behalf_of_principal_id": subject,
            "workspace": "private",
            "permissions": [permission],
        },
        entrypoint=1,
    )
    assert result[0]["result"]["allowed"] is False


@pytest.mark.parametrize("actor", ["service:intake", "service:admin", "service:platform-admin", "workload:powerful"])
@pytest.mark.parametrize("subject", ["viewer", "non-member"])
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/apis/auth/v2/iam/role-bindings"),
        ("GET", "/apis/entities/v2/workspaces/private/members"),
        ("POST", "/apis/entities/v2/workspaces/private/members"),
    ],
)
def test_delegated_route_cannot_use_actor_grants(
    policy: OPAPolicy, actor: str, subject: str, method: str, path: str
) -> None:
    result = policy.evaluate(
        {"principal_id": actor, "on_behalf_of_principal_id": subject, "method": method, "path": path},
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is False


@pytest.mark.parametrize("actor", ["service:intake", "service:platform-admin"])
def test_delegated_private_workspace_read_needs_subject_membership(policy: OPAPolicy, actor: str) -> None:
    for subject, expected in [("viewer", True), ("non-member", False)]:
        result = policy.evaluate(
            {
                "principal_id": actor,
                "on_behalf_of_principal_id": subject,
                "method": "GET",
                "path": "/apis/entities/v2/workspaces/private",
            },
            entrypoint=0,
        )
        assert result[0]["result"]["allowed"] is expected


@pytest.mark.parametrize("source", ["id", "email", "group", "account", "alias"])
def test_authorized_subject_identity_grants(policy: OPAPolicy, policy_data: dict[str, Any], source: str) -> None:
    policy_data["authz"]["principals"]["grant"] = {"workspaces": {"private": ["Admin"]}}
    policy.set_data(policy_data)
    body = {
        "principal_id": "service:viewer",
        "on_behalf_of_principal_id": "unbound",
        "workspace": "private",
        "permissions": ["iam.create"],
    }
    match source:
        case "id":
            body["on_behalf_of_principal_id"] = "grant"
        case "email":
            body["principal_email"] = "grant"
        case "group":
            body["principal_groups"] = ["grant"]
        case "account":
            body["subject_account_id"] = "grant"
        case "alias":
            body["subject_aliases"] = ["grant"]
    assert policy.evaluate(body, entrypoint=1)[0]["result"]["allowed"] is True
    body.update(method="POST", path="/apis/entities/v2/workspaces/private/members")
    assert policy.evaluate(body, entrypoint=0)[0]["result"]["allowed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actor", "subject", "expected"),
    [
        ("service:storage", "administrator", True),
        ("service:storage", "viewer", False),
        ("service:viewer", "administrator", False),
        ("service:viewer", "viewer", False),
    ],
)
async def test_real_query_actor_storage_and_subject_iam_are_independent(
    pdp_client: httpx.AsyncClient, actor: str, subject: str, expected: bool
) -> None:
    client = auth_client(pdp_client, actor, subject)
    assert await client.check_permissions("private", actor=["entities.read"], subject=["iam.read"]) is expected
    if expected:
        await require_entity_read_access(client, "private", "role_binding")
    else:
        with pytest.raises(HTTPException, match="403"):
            await require_entity_read_access(client, "private", "role_binding")


@pytest.mark.asyncio
async def test_real_legacy_query_and_control_guard_cannot_use_actor_admin(pdp_client: httpx.AsyncClient) -> None:
    client = auth_client(pdp_client, "service:platform-admin")
    assert await client.has_permissions("private", ["iam.create"]) is False
    with pytest.raises(HTTPException) as caught:
        await require_control_entity_access(client, "private", "role_binding", "create")
    assert caught.value.status_code == 403
    assert await client.check_permissions("private", subject=["iam.create"]) is False
    assert await client.check_permissions("private", actor=[], subject=["iam.create"]) is False
    assert await client.check_permissions("private", subject=["workspaces.read"]) is True
    assert await client.check_permissions("private", actor=[], subject=["workspaces.read"]) is True


@pytest.mark.asyncio
async def test_internal_storage_read_does_not_require_user_storage_grants(pdp_client: httpx.AsyncClient) -> None:
    client = auth_client(pdp_client, "service:storage")
    await require_entity_read_access(client, "private", "operational_record")
    allowed = await client.authorize_request("GET", "/apis/entities/v2/workspaces/private/entities/operational_record")
    assert allowed.allowed is True
    assert await client.check_permissions("private", subject=["entities.read"]) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", [None, "", " "])
@pytest.mark.parametrize("enabled", [False, True])
async def test_selected_empty_subject_requires_identity_before_grant_bypass(
    pdp_client: httpx.AsyncClient, pdp_calls: list[dict[str, Any]], subject: str | None, enabled: bool
) -> None:
    client = auth_client(pdp_client, "service:intake", subject)
    client.config.enabled = enabled
    with pytest.raises(ValueError, match="on-behalf-of"):
        await client.check_permissions("private", subject=[])
    with pytest.raises(ValueError, match="on-behalf-of"):
        await client.check_permissions("private", actor=["iam.create"], subject=[])
    assert pdp_calls == []


@pytest.mark.asyncio
async def test_empty_subject_and_direct_actor_queries(pdp_client: httpx.AsyncClient) -> None:
    client = auth_client(pdp_client, "service:storage")
    assert await client.check_permissions("private", subject=[]) is True
    with pytest.raises(ValueError, match="At least one"):
        await client.check_permissions("private")
    direct = auth_client(pdp_client, "service:storage", None)
    assert await direct.check_permissions("private", actor=["entities.read"]) is True
    assert await direct.has_permissions("private", ["entities.read"]) is True


@pytest.mark.asyncio
async def test_actor_claims_do_not_leak_into_subject_grants(
    pdp_client: httpx.AsyncClient, policy: OPAPolicy, policy_data: dict[str, Any], pdp_calls: list[dict[str, Any]]
) -> None:
    policy_data["authz"]["principals"]["actor-admin-group"] = {"workspaces": {"private": ["Admin"]}}
    policy.set_data(policy_data)
    client = auth_client(pdp_client, "service:viewer")
    client.principal.groups = ["actor-admin-group"]
    client.principal.on_behalf_of_groups = ["subject-viewers"]
    client.principal.on_behalf_of_email = "viewer@example.com"
    assert await client.check_permissions("private", actor=["iam.create"]) is True
    assert await client.check_permissions("private", subject=["iam.create"]) is False
    assert pdp_calls[-1]["actor_groups"] == ["actor-admin-group"]
    assert pdp_calls[-1]["principal_groups"] == ["subject-viewers"]


@pytest.mark.parametrize(
    ("actor", "subject", "expected"),
    [
        ("service:intake", "viewer", True),
        ("service:intake", "non-member", False),
        ("service:viewer", "viewer", False),
        ("service:viewer", "platform-admin", False),
    ],
)
def test_secret_route_requires_both_distinct_permissions(
    policy: OPAPolicy, actor: str, subject: str, expected: bool
) -> None:
    result = policy.evaluate(
        {
            "principal_id": actor,
            "on_behalf_of_principal_id": subject,
            "method": "GET",
            "path": "/apis/secrets/v2/workspaces/private/secrets/example/access",
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is expected


@pytest.mark.parametrize("scopes", [["models:read"], ["entities:read"]])
def test_actor_admin_does_not_bypass_route_scopes(policy: OPAPolicy, scopes: list[str]) -> None:
    result = policy.evaluate(
        {
            "principal_id": "service:platform-admin",
            "on_behalf_of_principal_id": "administrator",
            "method": "POST",
            "path": "/apis/entities/v2/workspaces/private/members",
            "scopes": scopes,
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is False


def test_unknown_delegated_route_cannot_use_service_fallback(policy: OPAPolicy) -> None:
    result = policy.evaluate(
        {"principal_id": "service:intake", "on_behalf_of_principal_id": "viewer", "method": "GET", "path": "/unknown"},
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", ["true", "false", 1, None])
async def test_malformed_pdp_allow_values_fail_closed(allowed: str | int | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"allowed": allowed}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = auth_client(transport, "service:intake")
        assert await client.check_permissions("private", subject=["iam.create"]) is False
        assert (await client.authorize_request("POST", "/apis/entities/v2/workspaces/private/members")).allowed is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 503])
async def test_pdp_failure_never_retries_with_actor_authority(status: int) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = auth_client(transport, "service:platform-admin")
        with pytest.raises(HTTPException) as caught:
            await client.check_permissions("private", subject=["iam.create"])
        assert caught.value.status_code == 503
        assert len(calls) == 1


def test_runtime_log_upload_uses_actor_grant_without_user_log_permission(
    policy: OPAPolicy, policy_data: dict[str, Any]
) -> None:
    policy_data["authz"]["principals"]["runner"] = {"workspaces": {"private": ["JobRunner"]}}
    policy.set_data(policy_data)
    for path, methods in policy_data["authz"]["endpoints"].items():
        for method, declaration in methods.items():
            if declaration.get("direct", {}).get("actor_perms") == ["jobs.logs.create"]:
                request_path = path.replace("{workspace}", "private").replace("{name}", "logs")
                result = policy.evaluate(
                    {
                        "principal_id": "runner",
                        "on_behalf_of_principal_id": "viewer",
                        "method": method.upper(),
                        "path": request_path,
                    },
                    entrypoint=0,
                )
                assert result[0]["result"]["allowed"] is True
                return
    pytest.fail("The shipped policy must declare the runtime log-upload operation")


def test_missing_actor_requirement_is_rejected_before_evaluation(
    policy: OPAPolicy, policy_data: dict[str, Any]
) -> None:
    path = "/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access"
    del policy_data["authz"]["endpoints"][path]["get"]["delegated"]["actor_perms"]
    with pytest.raises(ValueError, match="actor_perms"):
        policy.set_data(policy_data)


@pytest.mark.parametrize("source", ["id", "account", "alias", "email", "group"])
def test_actor_identity_grants_do_not_supply_subject_permission(
    policy: OPAPolicy, policy_data: dict[str, Any], source: str
) -> None:
    policy_data["authz"]["principals"]["actor-grant"] = {"workspaces": {"system": ["HelixAdmin"]}}
    policy.set_data(policy_data)
    body: dict[str, Any] = {
        "principal_id": "actor",
        "on_behalf_of_principal_id": "viewer",
        "workspace": "private",
        "actor_permissions": ["iam.create"],
        "subject_permissions": ["iam.create"],
    }
    match source:
        case "id":
            body["principal_id"] = "actor-grant"
        case "account":
            body["actor_account_id"] = "actor-grant"
        case "alias":
            body["actor_aliases"] = ["actor-grant"]
        case "email":
            body["actor_email"] = "actor-grant"
        case "group":
            body["actor_groups"] = ["actor-grant"]
    assert policy.evaluate(body, entrypoint=1)[0]["result"]["allowed"] is False
    del body["subject_permissions"]
    assert policy.evaluate(body, entrypoint=1)[0]["result"]["allowed"] is True


@pytest.mark.parametrize("missing", ["direct", "delegated"])
def test_missing_call_type_branch_denies_without_fallback(
    policy: OPAPolicy, policy_data: dict[str, Any], missing: str
) -> None:
    path = "/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access"
    del policy_data["authz"]["endpoints"][path]["get"][missing]
    policy.set_data(policy_data)
    body = {
        "principal_id": "service:platform-admin",
        "method": "GET",
        "path": "/apis/secrets/v2/workspaces/private/secrets/example/access",
    }
    if missing == "delegated":
        body["on_behalf_of_principal_id"] = "platform-admin"
    assert policy.evaluate(body, entrypoint=0)[0]["result"]["allowed"] is False


def test_declared_actor_only_delegated_branch_needs_no_subject_grant(
    policy: OPAPolicy, policy_data: dict[str, Any]
) -> None:
    path = "/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}"
    del policy_data["authz"]["endpoints"][path]["get"]["delegated"]["subject_perms"]
    policy.set_data(policy_data)
    result = policy.evaluate(
        {
            "principal_id": "service:storage",
            "on_behalf_of_principal_id": "non-member",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces/private/entities/operational_record",
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is True


@pytest.mark.parametrize("delegated", [False, True])
def test_malformed_selected_branch_is_rejected_before_evaluation(
    policy: OPAPolicy, policy_data: dict[str, Any], delegated: bool
) -> None:
    path = "/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access"
    branch = "delegated" if delegated else "direct"
    policy_data["authz"]["endpoints"][path]["get"][branch]["actor_perms"] = None
    with pytest.raises(ValueError, match="actor_perms"):
        policy.set_data(policy_data)


def test_undeclared_method_has_no_branch_even_for_admin(policy: OPAPolicy) -> None:
    result = policy.evaluate(
        {
            "principal_id": "service:platform-admin",
            "method": "OPTIONS",
            "path": "/apis/secrets/v2/workspaces/private/secrets/example/access",
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is False


@pytest.mark.parametrize("delegated", [False, True])
def test_workspace_comes_from_the_selected_most_specific_contract(
    policy: OPAPolicy, policy_data: dict[str, Any], delegated: bool
) -> None:
    for path in (
        "/apis/example/v2/workspaces/{workspace}/items",
        "/apis/example/v2/{workspace}/{name}/items",
    ):
        policy_data["authz"]["endpoints"][path] = {
            "get": {
                "direct": {"actor_perms": ["models.read"]},
                "delegated": {"actor_perms": [], "subject_perms": ["models.read"]},
            },
        }
    body = {
        "principal_id": "service:intake" if delegated else "viewer",
        "method": "GET",
        "path": "/apis/example/v2/workspaces/private/items",
    }
    if delegated:
        body["on_behalf_of_principal_id"] = "viewer"
    policy.set_data(policy_data)
    assert policy.evaluate(body, entrypoint=0)[0]["result"]["allowed"] is True
    policy_data["authz"]["principals"]["viewer"] = {"workspaces": {"workspaces": ["Viewer"]}}
    policy.set_data(policy_data)
    assert policy.evaluate(body, entrypoint=0)[0]["result"]["allowed"] is False


@pytest.mark.parametrize("claim", ["email", "groups"])
def test_envoy_actor_claims_cannot_supply_subject_permissions(
    policy: OPAPolicy, policy_data: dict[str, Any], claim: str
) -> None:
    grant = "actor-admin@example.com" if claim == "email" else "actor-admins"
    policy_data["authz"]["principals"][grant] = {"workspaces": {"private": ["Admin"]}}
    policy.set_data(policy_data)
    result = policy.evaluate(
        {
            "attributes": {
                "request": {
                    "http": {
                        "method": "POST",
                        "path": "/apis/entities/v2/workspaces/private/members",
                        "headers": {
                            "X-NHX-Principal-Id": "service:viewer",
                            "X-NHX-Principal-On-Behalf-Of": "viewer",
                            f"X-NHX-Principal-{claim}": grant,
                        },
                    }
                }
            }
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is False


@pytest.mark.parametrize("claim", ["email", "groups"])
def test_envoy_subject_claims_supply_only_subject_permissions(
    policy: OPAPolicy, policy_data: dict[str, Any], claim: str
) -> None:
    grant = "subject-admin@example.com" if claim == "email" else "subject-admins"
    policy_data["authz"]["principals"][grant] = {"workspaces": {"private": ["Admin"]}}
    policy.set_data(policy_data)
    result = policy.evaluate(
        {
            "attributes": {
                "request": {
                    "http": {
                        "method": "POST",
                        "path": "/apis/entities/v2/workspaces/private/members",
                        "headers": {
                            "X-NHX-Principal-Id": "service:viewer",
                            "X-NHX-Principal-On-Behalf-Of": "viewer",
                            f"X-NHX-Principal-On-Behalf-Of-{claim}": grant,
                        },
                    }
                }
            }
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is True


@pytest.mark.parametrize("claim", ["email", "groups"])
def test_envoy_dual_secret_checks_keep_actor_claims(policy: OPAPolicy, policy_data: dict[str, Any], claim: str) -> None:
    grant = "actor-access@example.com" if claim == "email" else "actor-accessors"
    policy_data["authz"]["roles"]["SecretAccessor"] = {"permissions": ["secrets.access"]}
    policy_data["authz"]["principals"][grant] = {"workspaces": {"private": ["SecretAccessor"]}}
    policy.set_data(policy_data)
    result = policy.evaluate(
        {
            "attributes": {
                "request": {
                    "http": {
                        "method": "GET",
                        "path": "/apis/secrets/v2/workspaces/private/secrets/example/access",
                        "headers": {
                            "X-NHX-Principal-Id": "service:viewer",
                            "X-NHX-Principal-On-Behalf-Of": "viewer",
                            f"X-NHX-Principal-{claim}": grant,
                        },
                    }
                }
            }
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is True


@pytest.mark.parametrize("claim", ["email", "groups"])
def test_envoy_direct_caller_keeps_own_claims(policy: OPAPolicy, policy_data: dict[str, Any], claim: str) -> None:
    grant = "direct-admin@example.com" if claim == "email" else "direct-admins"
    policy_data["authz"]["principals"][grant] = {"workspaces": {"private": ["Admin"]}}
    policy.set_data(policy_data)
    result = policy.evaluate(
        {
            "attributes": {
                "request": {
                    "http": {
                        "method": "POST",
                        "path": "/apis/entities/v2/workspaces/private/members",
                        "headers": {"X-NHX-Principal-Id": "viewer", f"X-NHX-Principal-{claim}": grant},
                    }
                }
            }
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is True


@pytest.mark.parametrize("subject", ["viewer", "non-member"])
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/apis/models/v2/workspaces/-/models"),
        ("HEAD", "/apis/files/v2/workspaces/-/filesets/example/-/file.txt"),
    ],
)
def test_delegated_wildcard_reads_preserve_existing_admission(
    policy: OPAPolicy, subject: str, method: str, path: str
) -> None:
    result = policy.evaluate(
        {
            "principal_id": "service:models",
            "on_behalf_of_principal_id": subject,
            "method": method,
            "path": path,
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is True


@pytest.mark.parametrize("actor_allowed", [True, False])
@pytest.mark.parametrize("subject", ["non-member", "platform-admin"])
def test_delegated_cross_workspace_reads_keep_actor_requirements(
    policy: OPAPolicy, policy_data: dict[str, Any], actor_allowed: bool, subject: str
) -> None:
    declaration = policy_data["authz"]["endpoints"]["/apis/models/v2/workspaces/{workspace}/models"]["get"]
    declaration["delegated"] = {"actor_perms": ["entities.read"], "subject_perms": ["models.read"]}
    policy_data["authz"]["principals"]["service:restricted"] = {
        "workspaces": {"-": ["ServiceEntityStore" if actor_allowed else "Viewer"]}
    }
    policy.set_data(policy_data)
    result = policy.evaluate(
        {
            "principal_id": "service:restricted",
            "on_behalf_of_principal_id": subject,
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models",
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is actor_allowed


@pytest.mark.parametrize(
    ("method", "resource"),
    [("POST", "models"), ("PUT", "prompts/example"), ("PATCH", "models/example"), ("DELETE", "models/example")],
)
def test_delegated_cross_workspace_mutations_cannot_use_actor_admin(
    policy: OPAPolicy, method: str, resource: str
) -> None:
    result = policy.evaluate(
        {
            "principal_id": "service:platform-admin",
            "on_behalf_of_principal_id": "viewer",
            "method": method,
            "path": f"/apis/models/v2/workspaces/-/{resource}",
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is False


@pytest.mark.parametrize("restriction", ["scope", "missing-branch", "explicit-deny", "quarantine"])
def test_delegated_cross_workspace_read_keeps_mandatory_denials(
    policy: OPAPolicy, policy_data: dict[str, Any], restriction: str
) -> None:
    declaration = policy_data["authz"]["endpoints"]["/apis/models/v2/workspaces/{workspace}/models"]["get"]
    body = {
        "principal_id": "service:platform-admin",
        "on_behalf_of_principal_id": "viewer",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/-/models",
    }
    match restriction:
        case "scope":
            body["scopes"] = ["models:write"]
        case "missing-branch":
            declaration.pop("delegated", None)
            declaration["direct"] = {"actor_perms": []}
        case "explicit-deny":
            declaration["deny"] = True
        case "quarantine":
            policy_data["authz"].setdefault("config", {})["denied_plugin_prefixes"] = ["/apis/models"]
    policy.set_data(policy_data)
    assert policy.evaluate(body, entrypoint=0)[0]["result"]["allowed"] is False


def test_embedded_loader_rejects_shorthand_without_mutating_it(policy: OPAPolicy) -> None:
    declaration = {"permissions": ["models.read"]}
    with pytest.raises(ValueError, match="shorthand"):
        policy.set_data({"authz": {"endpoints": {"/api/x": {"get": declaration}}}})
    assert declaration == {"permissions": ["models.read"]}


@pytest.mark.parametrize("service_only", [True, False])
@pytest.mark.parametrize("delegated", [True, False])
def test_contributed_contract_reaches_wasm_without_loader_conversion(
    policy: OPAPolicy, policy_data: dict[str, Any], service_only: bool, delegated: bool
) -> None:
    path = "/apis/example/v2/workspaces/{workspace}/items"
    contribution = AuthzContribution(
        endpoints={
            path: {
                "get": AuthzEndpointMethod(
                    permissions=["models.read"],
                    callers=["service_principal"] if service_only else ["principal", "service_principal"],
                )
            }
        }
    )
    policy.set_data(merge_authz_contributions(policy_data, [contribution.to_dict()]))
    body = {"principal_id": "service:intake", "method": "GET", "path": path.replace("{workspace}", "private")}
    if delegated:
        body["on_behalf_of_principal_id"] = "viewer"
    assert policy.evaluate(body, entrypoint=0)[0]["result"]["allowed"] is (not service_only or not delegated)


@pytest.mark.parametrize("actor", ["service:intake", "service:platform-admin"])
@pytest.mark.parametrize(
    ("workspace", "subject", "allowed"),
    [("-", "viewer", True), ("-", "non-member", True), ("private", "viewer", True), ("private", "non-member", False)],
)
def test_delegated_projects_list_preserves_wildcard_admission_and_named_workspace_checks(
    policy: OPAPolicy, actor: str, workspace: str, subject: str, allowed: bool
) -> None:
    """Check route admission only; project-result filtering is separate enforcement."""
    result = policy.evaluate(
        {
            "principal_id": actor,
            "on_behalf_of_principal_id": subject,
            "method": "GET",
            "path": f"/apis/entities/v2/workspaces/{workspace}/projects",
        },
        entrypoint=0,
    )
    assert result[0]["result"]["allowed"] is allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("actor", ["service:intake", "service:platform-admin"])
@pytest.mark.parametrize("entity_type", ["role_binding", "access_key"])
@pytest.mark.parametrize("subject_role", ["Admin", "Viewer"])
async def test_control_list_uses_subject_iam_in_real_workspaces(
    policy: OPAPolicy,
    policy_data: dict[str, Any],
    pdp_client: httpx.AsyncClient,
    pdp_calls: list[dict[str, Any]],
    actor: str,
    entity_type: str,
    subject_role: str,
) -> None:
    policy_data["authz"]["principals"]["system-user"] = {
        "workspaces": {"system": [subject_role], "private": ["Viewer"]}
    }
    policy.set_data(policy_data)
    client = auth_client(pdp_client, actor, "system-user")
    allowed = await get_control_entity_list_workspaces(
        client, AsyncMock(spec=WorkspaceRepositoryInterface), entity_type, {"system", "private"}
    )
    assert allowed == ({"system"} if subject_role == "Admin" else set())
    assert len(pdp_calls) == 1
    assert set(pdp_calls[0]["workspaces"]) == {"system", "private"}
    assert pdp_calls[0]["subject_permissions"] == ["iam.list"]
    assert "actor_permissions" not in pdp_calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
async def test_workspace_batch_matches_scalar_permission_decisions(
    policy: OPAPolicy,
    policy_data: dict[str, Any],
    pdp_client: httpx.AsyncClient,
    pdp_calls: list[dict[str, Any]],
    delegated: bool,
) -> None:
    policy_data["authz"]["principals"]["system-user"] = {"workspaces": {"system": ["Admin"], "private": ["Viewer"]}}
    policy.set_data(policy_data)
    client = auth_client(
        pdp_client, "service:platform-admin" if delegated else "system-user", "system-user" if delegated else None
    )
    candidates = {"system", "private", "no-binding"}
    requirements = {"subject": ["iam.list"]} if delegated else {"actor": ["iam.list"]}
    scalar = {workspace for workspace in candidates if await client.check_permissions(workspace, **requirements)}
    pdp_calls.clear()
    assert await client.filter_workspaces(candidates, **requirements) == scalar == {"system"}
    assert len(pdp_calls) == 1


@pytest.mark.asyncio
async def test_large_workspace_batch_uses_one_pdp_request_with_current_wasm(
    pdp_client: httpx.AsyncClient,
    pdp_calls: list[dict[str, Any]],
) -> None:
    client = auth_client(pdp_client, "service:intake", "viewer")
    candidates = {f"workspace-{i}" for i in range(1000)} | {"private"}
    assert await client.filter_workspaces(candidates, subject=["models.read"]) == {"private"}
    assert len(pdp_calls) == 1


@pytest.mark.asyncio
async def test_workspace_batch_preserves_service_workspace_overrides(
    policy: OPAPolicy,
    policy_data: dict[str, Any],
    pdp_client: httpx.AsyncClient,
    pdp_calls: list[dict[str, Any]],
) -> None:
    policy_data["authz"]["principals"]["service:limited"] = {"workspaces": {"private": ["Viewer"]}}
    policy.set_data(policy_data)
    client = auth_client(pdp_client, "service:limited", None)
    assert await client.filter_workspaces({"private", "no-binding"}, actor=["iam.list"]) == {"no-binding"}
    assert len(pdp_calls) == 1
