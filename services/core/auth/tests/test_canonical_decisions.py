# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual production WASM: independent identity sets, scopes and mandatory boundaries."""

import httpx
import pytest
from nemo_helix_plugin.auth.decisions import (
    AccessKeyFacts,
    CollectionTarget,
    DecisionEvaluationInput,
    DecisionRequest,
    FieldDecision,
    FieldTarget,
    OperationDecision,
    RequestTarget,
    ResourceTarget,
    WorkspaceDecision,
)
from nhx.common.auth.client import AuthClient
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.testing.policy import PolicyHarness


@pytest.fixture
def policy() -> PolicyHarness:
    result = PolicyHarness()
    result.document["authz"]["principals"].update(
        {
            "alice": {"workspaces": {"team": ["Viewer"]}},
            "user-group": {"workspaces": {"team": ["Editor"]}},
            "actor-group": {"workspaces": {"system": ["HelixAdmin"]}},
            "admin": {"workspaces": {"system": ["HelixAdmin"]}},
            "service-account:bot": {"workspaces": {"system": ["HelixAdmin"]}},
        }
    )
    result.install(result.document)
    return result


def envelope(**updates) -> DecisionEvaluationInput:
    return DecisionEvaluationInput.model_validate(
        {
            "protocol_version": 2,
            "operation": "route",
            "target": RequestTarget(
                method="GET", path="/apis/secrets/v2/workspaces/team/secrets/key/access"
            ).model_dump(),
            "method": "GET",
            "path": "/apis/secrets/v2/workspaces/team/secrets/key/access",
            "principal_id": "job-step:example",
            "caller_kind": "service_principal",
            "on_behalf_of_principal_id": "alice",
            "verified_workload_delegation": True,
            "principal_context_digest": "trusted",
            "scopes": ["secrets:read"],
            **updates,
        }
    )


@pytest.mark.parametrize("actor", ["job-step:example", "deployment:example"])
@pytest.mark.parametrize(
    "subject,workspace,verified,scopes,expected",
    [
        ("alice", "team", True, ["secrets:read"], True),
        ("alice", "other", True, ["secrets:read"], False),
        ("missing", "team", True, ["secrets:read"], False),
        ("alice", "team", False, ["secrets:read"], False),
        ("alice", "team", True, ["jobs:read"], False),
        ("alice", "team", True, ["secrets:write"], False),
        ("alice", "team", True, ["platform:read"], True),
    ],
)
def test_workload_secret_contract(policy, actor, subject, workspace, verified, scopes, expected) -> None:
    path = f"/apis/secrets/v2/workspaces/{workspace}/secrets/key/access"
    query = envelope(
        principal_id=actor,
        on_behalf_of_principal_id=subject,
        verified_workload_delegation=verified,
        scopes=scopes,
        path=path,
        target=RequestTarget(method="GET", path=path).model_dump(),
    )
    result = policy.decide(query)
    assert isinstance(result, OperationDecision) and result.allowed is expected
    assert policy.evaluate("allow", query.model_dump(mode="json"))["allowed"] is expected


def test_actor_identity_never_supplies_subject_permissions(policy) -> None:
    assert not policy.decide(envelope(on_behalf_of_principal_id="missing", actor_groups=["actor-group"])).allowed
    assert policy.decide(envelope(on_behalf_of_principal_id="missing", subject_groups=["user-group"])).allowed


@pytest.mark.parametrize(
    "actor,subject,read,write",
    [
        ("service:jobs", "", True, True),
        ("service:models", "", True, False),
        ("service:jobs", "alice", False, False),
        ("service:jobs", "admin", False, False),
        ("job-step:example", "admin", False, False),
        ("admin", "", False, False),
        ("service-account:bot", "", False, False),
    ],
)
def test_field_decisions_deny_every_obo(policy, actor, subject, read, write) -> None:
    query = envelope(
        operation="entities.auth_context.access",
        target=FieldTarget(workspace="team", resource_type="platform_job").model_dump(),
        principal_id=actor,
        on_behalf_of_principal_id=subject,
        scopes=[],
        caller_kind="service_principal" if actor.startswith("service:") or subject else "principal",
    )
    decision = policy.decide(query)
    assert isinstance(decision, FieldDecision)
    assert (decision.read, decision.write) == (read, write)


@pytest.mark.parametrize(
    "actor,resource_type,subject,allowed",
    [
        ("service:jobs", "platform_job", "", True),
        ("service:builder", "builder_queued_build", "", True),
        ("service:builder", "builder_queued_build", "admin", False),
        ("service:builder", "platform_job", "", False),
        ("service:jobs", "deployments_deployment", "", False),
        ("service:agents", "deployments_deployment", "", False),
        ("service:deployments", "deployments_deployment", "", True),
        ("service:jobs", "platform_job", "admin", False),
        ("job-step:example", "platform_job", "admin", False),
    ],
)
def test_capture_is_registered_to_actual_direct_writer(policy, actor, resource_type, subject, allowed) -> None:
    query = envelope(
        operation="entities.auth_context.access",
        target=FieldTarget(workspace="team", resource_type=resource_type).model_dump(),
        principal_id=actor,
        on_behalf_of_principal_id=subject,
        scopes=[],
    )
    result = policy.decide(query)
    assert isinstance(result, FieldDecision) and result.write is allowed


@pytest.mark.parametrize(
    "principal,owner,service_bound,allowed",
    [
        ("alice", "alice", False, True),
        ("alice", "other", False, False),
        ("admin", "alice", False, False),
        ("admin", "alice", True, True),
        ("alice", "alice", True, False),
        ("service-account:bot", "service-account:bot", True, False),
        ("service-account:bot", "service-account:bot", False, False),
        ("service:auth", "service:auth", True, False),
    ],
)
def test_key_owner_and_human_admin_rules(policy, principal, owner, service_bound, allowed) -> None:
    query = envelope(
        operation="auth.access-key.lifecycle",
        target=ResourceTarget(workspace="system", resource_type="access_key", name="key").model_dump(),
        facts=AccessKeyFacts(owner=owner, service_bound=service_bound).model_dump(),
        principal_id=principal,
        on_behalf_of_principal_id="",
        caller_kind="principal",
        scopes=[],
    )
    result = policy.decide(query)
    assert isinstance(result, OperationDecision) and result.allowed is allowed


def test_empty_membership_scope_is_restricted_not_unrestricted(policy) -> None:
    query = envelope(
        operation="entities.workspaces.visible",
        target=CollectionTarget(workspace="-", resource_type="model").model_dump(),
        on_behalf_of_principal_id="missing",
        scopes=[],
    )
    decision = policy.decide(query)
    assert isinstance(decision, WorkspaceDecision)
    assert decision.access.model_dump() == {"kind": "restricted", "workspaces": []}
    policy.document["authz"]["principals"]["*"] = {"workspaces": {"shared": ["Viewer"]}}
    policy.install(policy.document)
    assert policy.decide(query).access.model_dump() == {"kind": "restricted", "workspaces": ["shared"]}


@pytest.mark.asyncio
async def test_remote_client_adapter_returns_same_compiled_decision(policy) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(policy.respond)) as transport:
        client = AuthClient(
            principal=Principal(id="service:jobs", on_behalf_of="alice"),
            config=AuthConfig(enabled=True, policy_decision_point_base_url="http://pdp.invalid"),
            http_client=transport,
            service_name="jobs",
        )
        request = DecisionRequest(
            operation="entities.auth_context.access", target=FieldTarget(workspace="team", resource_type="platform_job")
        )
        decision = await client.decide(request)
        assert isinstance(decision, FieldDecision) and not decision.read and not decision.write


def test_internal_storage_role_applies_to_the_actor_without_becoming_a_user_grant(policy) -> None:
    policy.document["authz"]["principals"]["service:restricted"] = {"workspaces": {"team": ["ServiceEntityStore"]}}
    policy.document["authz"]["principals"]["alice"]["workspaces"]["team"] = ["Viewer", "ServiceEntityStore"]
    policy.install(policy.document)
    path = "/apis/entities/v2/workspaces/team/entities/platform_job"
    query = envelope(
        principal_id="service:restricted",
        on_behalf_of_principal_id="alice",
        scopes=[],
        target=RequestTarget(method="GET", path=path).model_dump(),
        method="GET",
        path=path,
    )
    assert policy.decide(query).allowed
    # A human's role binding cannot confer internal storage provenance.
    direct = query.model_copy(
        update={"principal_id": "alice", "on_behalf_of_principal_id": "", "caller_kind": "principal"}
    )
    assert not policy.decide(direct).allowed
    # A service-kind external actor cannot claim this internal role through a group.
    external = query.model_copy(update={"principal_id": "workload:external", "actor_groups": ["service:restricted"]})
    assert not policy.decide(external).allowed


def test_job_logs_use_the_existing_actor_grant_and_no_subject_grant(policy) -> None:
    policy.document["authz"]["principals"]["runtime-group"] = {"workspaces": {"team": ["JobRunner"]}}
    policy.install(policy.document)
    path = "/apis/files/v2/workspaces/team/filesets/artifacts/otlp/v1/logs"
    query = envelope(
        on_behalf_of_principal_id="missing",
        actor_groups=["runtime-group"],
        scopes=[],
        target=RequestTarget(method="POST", path=path).model_dump(),
        method="POST",
        path=path,
    )
    assert policy.decide(query).allowed
    assert not policy.decide(query.model_copy(update={"actor_groups": [], "subject_groups": ["runtime-group"]})).allowed


@pytest.mark.parametrize("roles,allowed", [(["ServiceEntityStore"], False), (["TestIAM"], True)])
def test_all_workspace_control_queries_respect_explicit_workspace_restrictions(policy, roles, allowed) -> None:
    from nemo_helix_plugin.auth.decisions import EntityFacts

    policy.document["authz"]["roles"]["TestIAM"] = {"permissions": ["iam.list"]}
    policy.document["authz"]["principals"]["service:scoped"] = {"workspaces": {"team": roles}}
    policy.install(policy.document)
    query = envelope(
        operation="entities.control.access",
        principal_id="service:scoped",
        on_behalf_of_principal_id="",
        scopes=[],
        target=ResourceTarget(workspace="-", resource_type="role_binding", name="").model_dump(),
        facts=EntityFacts(action="list").model_dump(),
    )
    assert policy.decide(query).allowed is allowed


@pytest.mark.parametrize("claim", ["groups", "email", "account-id", "aliases"])
def test_envoy_actor_claims_cannot_supply_subject_grants(policy, claim: str) -> None:
    actor_header, subject_header = {
        "groups": ("x-nhx-principal-groups", "x-nhx-principal-on-behalf-of-groups"),
        "email": ("x-nhx-principal-email", "x-nhx-principal-on-behalf-of-email"),
        "account-id": ("x-nhx-actor-account-id", "x-nhx-subject-account-id"),
        "aliases": ("x-nhx-actor-aliases", "x-nhx-subject-aliases"),
    }[claim]
    headers = {
        "x-nhx-principal-id": "service:jobs",
        "x-nhx-principal-on-behalf-of": "missing",
        actor_header: "user-group",
    }
    query = {
        "attributes": {
            "request": {
                "http": {
                    "method": "GET",
                    "path": "/apis/secrets/v2/workspaces/team/secrets/key/access",
                    "headers": headers,
                }
            }
        }
    }
    assert not policy.evaluate("allow", query)["allowed"]
    del headers[actor_header]
    headers[subject_header] = "user-group"
    assert policy.evaluate("allow", query)["allowed"]


def test_restricted_platform_actor_cannot_gain_workload_default_capability(policy) -> None:
    policy.document["authz"]["principals"]["service:restricted"] = {"workspaces": {"team": ["ServiceEntityStore"]}}
    policy.install(policy.document)
    # Service bearer OBO tokens also have act claims. They must honor the
    # configured platform actor's grants instead of gaining workload defaults.
    assert not policy.decide(envelope(principal_id="service:restricted")).allowed
    assert policy.decide(envelope(principal_id="job-step:example")).allowed


@pytest.mark.parametrize(
    "actor,subject,roles,scopes,allowed",
    [
        ("service:deployments", "", [], ["platform:write"], True),
        ("service:deployments", "", ["Viewer"], ["platform:write"], False),
        ("service:deployments", "alice", [], ["platform:write"], False),
        ("service:deployments", "admin", [], ["platform:write"], False),
        ("job-step:example", "alice", [], ["platform:write"], False),
        ("job-step:example", "", [], ["platform:write"], False),
        ("admin", "", [], ["platform:write"], False),
        ("service:deployments", "", [], ["jobs:read"], False),
    ],
)
def test_service_only_route_requires_direct_platform_authority(policy, actor, subject, roles, scopes, allowed) -> None:
    pattern = "/apis/example/v2/workspaces/{workspace}/status"
    path = "/apis/example/v2/workspaces/team/status"
    policy.document["authz"]["endpoints"][pattern] = {
        "put": {
            "callers": ["service_principal"],
            "permissions": ["deployments.status.update"],
            "scopes": ["platform:write"],
        }
    }
    if roles:
        policy.document["authz"]["principals"][actor] = {"workspaces": {"team": roles}}
    policy.install(policy.document)
    query = envelope(
        method="PUT",
        path=path,
        target=RequestTarget(method="PUT", path=path).model_dump(),
        principal_id=actor,
        caller_kind="service_principal",
        on_behalf_of_principal_id=subject,
        scopes=scopes,
    )
    result = policy.decide(query)
    assert isinstance(result, OperationDecision) and result.allowed is allowed
    assert policy.evaluate("allow", query.model_dump(mode="json"))["allowed"] is allowed


@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("second_grant", [False, True])
def test_permission_queries_and_operations_aggregate_subject_groups(policy, delegated, second_grant) -> None:
    authz = policy.document["authz"]
    authz["roles"].update(
        {
            "First": {"permissions": ["example.first"]},
            "Second": {"permissions": ["example.second"]},
        }
    )
    authz["principals"].update(
        {
            "group-first": {"workspaces": {"team": ["First"]}},
            "group-second": {"workspaces": {"team": ["Second"]}},
        }
    )
    authz["operations"]["example.aggregate"] = {
        "owner": "example",
        "predicate": "permissions",
        "permissions": ["example.first", "example.second"],
        "callers": ["principal", "service_principal"],
        "scopes": [],
    }
    policy.install(policy.document)
    groups = ["group-first", "group-second"] if second_grant else ["group-first"]
    query = envelope(
        operation="example.aggregate",
        target=ResourceTarget(workspace="team", resource_type="example", name="item").model_dump(),
        principal_id="service:jobs" if delegated else "alice",
        caller_kind="service_principal" if delegated else "principal",
        on_behalf_of_principal_id="alice" if delegated else "",
        subject_groups=groups if delegated else [],
        actor_groups=[] if delegated else groups,
        scopes=[],
    )
    operation = policy.decide(query)
    permissions = policy.evaluate(
        "has_permissions",
        {
            **query.model_dump(mode="json"),
            "workspace": "team",
            "permissions": ["example.first", "example.second"],
        },
    )
    assert isinstance(operation, OperationDecision)
    assert operation.allowed is second_grant
    assert permissions["allowed"] is second_grant


@pytest.mark.parametrize(
    "email,owner,allowed",
    [
        ("", "", False),
        ("", "alice", True),
        ("alice@example.com", "alice@example.com", True),
        ("alice@example.com", "", False),
    ],
)
@pytest.mark.parametrize("delegated", [False, True])
def test_workspace_deletion_owner_must_be_nonempty(policy, email, owner, allowed, delegated):
    from nemo_helix_plugin.auth.decisions import LifecycleFacts

    result = policy.decide(
        envelope(
            operation="entities.workspace.deletion-status",
            target=ResourceTarget(workspace="team", resource_type="workspace", name="team").model_dump(),
            facts=LifecycleFacts(created_by=owner).model_dump(),
            principal_id="service:jobs" if delegated else "alice",
            caller_kind="service_principal" if delegated else "principal",
            on_behalf_of_principal_id="alice" if delegated else "",
            actor_email="" if delegated else email,
            subject_email=email if delegated else "",
            scopes=[],
        )
    )
    assert isinstance(result, OperationDecision) and result.allowed is allowed


@pytest.mark.parametrize(
    "subject,actor_groups,allowed",
    [("admin", [], True), ("alice", ["actor-group"], False), ("service-account:bot", [], False)],
)
def test_internal_service_can_only_represent_an_eligible_human_key_admin(policy, subject, actor_groups, allowed):
    result = policy.decide(
        envelope(
            operation="auth.access-key.lifecycle",
            target=ResourceTarget(workspace="system", resource_type="access_key", name="key").model_dump(),
            facts=AccessKeyFacts(owner="other", service_bound=True).model_dump(),
            principal_id="service:auth",
            on_behalf_of_principal_id=subject,
            actor_groups=actor_groups,
            scopes=[],
        )
    )
    assert isinstance(result, OperationDecision) and result.allowed is allowed
