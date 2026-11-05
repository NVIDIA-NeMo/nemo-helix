# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import APIRouter
from nemo_helix_plugin.auth.decisions import (
    POLICY_DECISION_ADAPTER,
    DelegatedRequirements,
    FieldDecision,
    FieldTarget,
    OperationRule,
)
from nemo_helix_plugin.authz import (
    AuthzContribution,
    AuthzEndpointMethod,
    CallerKind,
    PermissionSet,
    path_rule,
    perm,
)
from nemo_helix_plugin.authz_discovery import _derive_service_contribution, _method_from_dict
from nemo_helix_plugin.authz_format import validate_static_authz_data
from nemo_helix_plugin.authz_merge import merge_authz_contributions
from nemo_helix_plugin.service import NemoService, RouterSpec
from pydantic import ValidationError


class ExamplePermissions(PermissionSet, namespace="example.items"):
    READ = perm("Read items")
    RETRIEVE = perm("Retrieve values")


def test_delegated_policy_metadata_roundtrips() -> None:
    method = {
        "permissions": ["example.items.retrieve"],
        "callers": ["service_principal"],
        "on_behalf_of": {
            "actor_permissions": ["example.items.retrieve"],
            "permissions": ["example.items.read"],
        },
    }
    parsed = _method_from_dict(method)
    assert parsed.on_behalf_of == DelegatedRequirements(
        actor_permissions=["example.items.retrieve"], permissions=["example.items.read"]
    )
    contribution = AuthzContribution(endpoints={"/example": {"get": parsed}})
    merged = merge_authz_contributions({"authz": {}}, [contribution.to_dict()])
    validate_static_authz_data(merged)
    assert merged["authz"]["endpoints"]["/example"]["get"] == method


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"permissions": ["example.items.read"]},
        {"actor_permissions": []},
        {"actor_permissions": [], "permissions": [], "allow_all": True},
        {"actor_permissions": "example.items.retrieve", "permissions": []},
        {"actor_permissions": [], "permissions": [True]},
    ],
)
def test_invalid_delegated_metadata_is_rejected(value: dict) -> None:
    with pytest.raises(ValidationError):
        validate_static_authz_data(
            {"authz": {"endpoints": {"/example": {"get": {"permissions": [], "on_behalf_of": value}}}}}
        )


def test_operations_cannot_override_a_registered_contract() -> None:
    original = OperationRule(owner="example", permissions=["example.items.read"], callers=["principal"], scopes=[])
    modified = original.model_copy(update={"permissions": []})
    base = {"authz": {"operations": {"example.read": original.model_dump(exclude_none=True)}}}
    contribution = AuthzContribution(operations={"example.read": modified})
    with pytest.raises(ValueError, match="conflicting operation"):
        merge_authz_contributions(base, [contribution.to_dict()])


def test_operation_namespace_is_checked() -> None:
    rule = OperationRule(owner="example", permissions=[], callers=["principal"], scopes=[])
    with pytest.raises(ValueError, match="does not belong"):
        merge_authz_contributions({}, [AuthzContribution(operations={"entities.read": rule}).to_dict()])


def test_field_decision_requires_exact_booleans_and_binding() -> None:
    payload = {
        "protocol_version": 2,
        "operation": "entities.auth_context.access",
        "decision_type": "field",
        "target": FieldTarget(workspace="team", resource_type="example").model_dump(),
        "principal_context_digest": "context",
        "policy_revision": "revision",
        "reason": "policy_decision",
        "field": "auth_context",
        "read": False,
        "write": False,
    }
    assert isinstance(POLICY_DECISION_ADAPTER.validate_python(payload), FieldDecision)
    for patch in [{"read": "false"}, {"protocol_version": 1}, {"extra_allow": True}]:
        with pytest.raises(ValidationError):
            POLICY_DECISION_ADAPTER.validate_python({**payload, **patch})


def test_serialization_does_not_drop_explicit_empty_user_requirement() -> None:
    requirement = DelegatedRequirements(actor_permissions=["example.items.retrieve"], permissions=[])
    contribution = AuthzContribution(
        endpoints={"/example": {"get": AuthzEndpointMethod(permissions=[], on_behalf_of=requirement)}}
    )
    assert contribution.to_dict()["endpoints"]["/example"]["get"]["on_behalf_of"]["permissions"] == []


def test_typed_sdk_serializes_required_nested_discriminators() -> None:
    import json

    from nemo_helix_plugin.auth.decisions import CreatorFacts, DecisionEvaluationInput, ResourceTarget
    from nemo_helix_plugin.iam.endpoints import evaluate_authorization
    from nemo_helix_plugin.iam.types import AuthzRequest

    query = DecisionEvaluationInput(
        protocol_version=2,
        operation="secrets.binding.authorize",
        target=ResourceTarget(workspace="team", resource_type="secret", name="key"),
        facts=CreatorFacts(principal_id="alice"),
        principal_id="service:jobs",
        caller_kind="service_principal",
        principal_context_digest="trusted",
    )
    prepared = evaluate_authorization(entrypoint="decision", body=AuthzRequest(input=query.model_dump(mode="json")))
    assert isinstance(prepared.content, bytes)
    payload = json.loads(prepared.content)["input"]
    assert payload["protocol_version"] == 2
    assert payload["target"]["kind"] == "resource"
    assert payload["facts"]["kind"] == "creator"
    assert payload["scopes"] == []


def test_new_contributions_require_and_propagate_the_contract_version() -> None:
    rule = OperationRule(owner="example", permissions=[], callers=["principal"], scopes=[])
    contribution = AuthzContribution(operations={"example.check": rule}).to_dict()
    assert contribution["decision_contract_version"] == 2
    assert merge_authz_contributions({}, [contribution])["authz"]["decision_contract_version"] == 2
    contribution.pop("decision_contract_version")
    assert merge_authz_contributions({}, [contribution])["authz"]["decision_contract_version"] == 2
    contribution["decision_contract_version"] = 1
    with pytest.raises(ValueError, match="version 2"):
        merge_authz_contributions({}, [contribution])


def test_named_operation_cannot_reference_undeclared_permissions() -> None:
    router = APIRouter()

    @router.get("/items")
    @path_rule(callers=[CallerKind.PRINCIPAL], permissions=[ExamplePermissions.READ])
    async def items() -> list[str]:
        return []

    class Example(NemoService):
        name = "example"

        def get_routers(self) -> list[RouterSpec]:
            return [RouterSpec(router, prefix="/v2")]

        def authorization_operations(self) -> dict[str, OperationRule]:
            return {
                "example.bad": OperationRule(
                    owner="example",
                    permissions=["secrets.access"],
                    callers=["service_principal"],
                    scopes=[],
                )
            }

    contribution, errors, _ = _derive_service_contribution(Example())
    assert any("undeclared permission" in error for error in errors)
    assert not contribution.operations
    assert contribution.endpoints["/apis/example/v2/items"]["get"].deny


def test_wire_route_version_and_target_binding_are_required() -> None:
    from nemo_helix_plugin.auth.decisions import DecisionEvaluationInput, RequestTarget

    document = {
        "protocol_version": 2,
        "operation": "route",
        "target": RequestTarget(method="GET", path="/known").model_dump(),
        "method": "GET",
        "path": "/known",
        "principal_id": "alice",
        "caller_kind": "principal",
        "principal_context_digest": "trusted",
    }
    DecisionEvaluationInput.model_validate(document)
    for patch in [{"protocol_version": 1}, {"path": "/other"}, {"method": "POST"}]:
        with pytest.raises(ValidationError):
            DecisionEvaluationInput.model_validate({**document, **patch})
    document.pop("protocol_version")
    with pytest.raises(ValidationError):
        DecisionEvaluationInput.model_validate(document)


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", [False, True])
async def test_plugin_decision_client_reuses_generic_evaluation(mismatch: bool) -> None:
    import json

    import httpx
    from nemo_helix_plugin.auth.decision_client import query_decision
    from nemo_helix_plugin.auth.decisions import DecisionEvaluationInput, OperationDecision, ResourceTarget
    from nemo_helix_plugin.client.client import AsyncNemoClient

    target = ResourceTarget(workspace="team", resource_type="secret", name="key")
    query = DecisionEvaluationInput(
        protocol_version=2,
        operation="secrets.binding.authorize",
        target=target,
        principal_id="service:jobs",
        caller_kind="service_principal",
        principal_context_digest="trusted",
    )
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path == "/apis/auth/v2/authz/decision"
        assert json.loads(request.content)["input"] == query.model_dump(mode="json")
        decision = OperationDecision(
            protocol_version=2,
            operation="wrong.operation" if mismatch else query.operation,
            target=target,
            principal_context_digest=query.principal_context_digest,
            policy_revision="test",
            reason="policy_decision",
            decision_type="operation",
            allowed=True,
        )
        return httpx.Response(200, json={"result": decision.model_dump(mode="json")})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        client = AsyncNemoClient(base_url="http://platform.test", http_client=transport)
        if mismatch:
            with pytest.raises(ValueError, match="does not match"):
                await query_decision(client, query)
        else:
            result = await query_decision(client, query)
            assert isinstance(result, OperationDecision) and result.allowed
    assert len(calls) == 1
