# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real compiled-policy fixture for authorization adapter and integration tests."""

import json
import threading
from copy import deepcopy

import httpx
import pytest
import yaml
from fastapi import FastAPI
from nemo_helix_plugin.auth.decisions import (
    POLICY_DECISION_ADAPTER,
    AccessKeyFacts,
    DecisionEvaluationInput,
    OperationDecision,
    PolicyDecision,
    RequestTarget,
    ResourceTarget,
)
from nemo_helix_plugin.authz_discovery import _derive_service_contribution
from nemo_helix_plugin.authz_merge import merge_authz_contributions
from nemo_helix_plugin.service import NemoService
from nhx.common.auth.client import AuthClient
from nhx.common.auth.middleware import AuthorizationMiddleware
from nhx.common.auth.models import Principal
from nhx.common.config import AuthConfig
from nhx.core.auth.app.embedded_pdp.engine import ENTRYPOINT_MAP, OPAPolicy
from nhx.core.auth.app.embedded_pdp.policy_wasm import DEFAULT_POLICY_WASM_PATH
from nhx.core.auth.app.scope_families import with_policy_revision
from nhx.core.auth.entities import AccessKeyEntity


class PolicyHarness:
    def __init__(self) -> None:
        self.document = yaml.safe_load(DEFAULT_POLICY_WASM_PATH.with_name("static-authz.yaml").read_text())
        self.document["authz"]["policy_revision"] = "compiled-test-policy"
        self._local = threading.local()
        self.install(self.document)

    def auth_client(
        self,
        *,
        actor: str,
        subject: str | None = None,
        enabled: bool = True,
        service_name: str = "platform",
        http_client: httpx.AsyncClient | None = None,
    ) -> AuthClient:
        return AuthClient(
            principal=Principal(id=actor, on_behalf_of=subject, verified_workload_delegation=bool(subject)),
            config=AuthConfig(enabled=enabled, policy_decision_point_base_url="http://pdp.invalid"),
            http_client=http_client
            if http_client is not None
            else httpx.AsyncClient(transport=httpx.MockTransport(self.respond)),
            service_name=service_name,
        )

    def merge_service(self, service: NemoService) -> None:
        contribution, errors, _ = _derive_service_contribution(service)
        if errors:
            raise ValueError(f"Invalid test service declaration: {errors}")
        self.install(merge_authz_contributions(self.document, [contribution.to_dict()]))

    def install(self, document: dict) -> None:
        self.document = deepcopy(document)

    def evaluate(self, entrypoint: str, document: dict) -> dict:
        if not hasattr(self._local, "policy"):
            self._local.policy = OPAPolicy(str(DEFAULT_POLICY_WASM_PATH))
        policy = self._local.policy
        policy.set_data(with_policy_revision(self.document))
        values = policy.evaluate(document, ENTRYPOINT_MAP[entrypoint])
        return values[0]["result"] if values else {}

    def decide(self, document: DecisionEvaluationInput) -> PolicyDecision:
        return POLICY_DECISION_ADAPTER.validate_python(self.evaluate("decision", document.model_dump(mode="json")))

    def respond(self, request: httpx.Request) -> httpx.Response:
        entrypoint = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, json={"result": self.evaluate(entrypoint, json.loads(request.content)["input"])})

    async def authorize_key(self, principal: str, record: AccessKeyEntity) -> bool:
        result = self.decide(
            DecisionEvaluationInput(
                protocol_version=2,
                operation="auth.access-key.lifecycle",
                target=ResourceTarget(workspace=record.workspace, resource_type="access_key", name=record.name),
                facts=AccessKeyFacts(owner=record.principal, service_bound=record.is_service_account()),
                principal_id=principal,
                caller_kind="principal",
                principal_context_digest=principal,
            )
        )
        return isinstance(result, OperationDecision) and result.allowed


def mock_operation_response(document: dict, *, allowed: bool) -> httpx.Response:
    """Mock the wire result for adapter tests; use PolicyHarness to test grant resolution."""
    query = DecisionEvaluationInput.model_validate(document)
    assert isinstance(query.target, (RequestTarget, ResourceTarget))
    decision = OperationDecision(
        protocol_version=2,
        operation=query.operation,
        target=query.target,
        principal_context_digest=query.principal_context_digest,
        policy_revision="mock-policy",
        reason="policy_decision",
        decision_type="operation",
        allowed=allowed,
    )
    return httpx.Response(200, json={"result": decision.model_dump(mode="json")})


def auth_disabled_client() -> AuthClient:
    return AuthClient(principal=Principal(), config=AuthConfig(enabled=False))


def install_policy_middleware(app: FastAPI, service: NemoService, monkeypatch: pytest.MonkeyPatch) -> PolicyHarness:
    """Mount actual middleware against the service's production compiled declarations."""
    policy = PolicyHarness()
    policy.merge_service(service)

    async def evaluate(self: AuthClient, document: DecisionEvaluationInput) -> PolicyDecision:
        return policy.decide(document)

    monkeypatch.setattr(AuthClient, "_evaluate_decision", evaluate)
    monkeypatch.setattr("nhx.common.auth.middleware.get_auth_config", lambda: AuthConfig(enabled=True))
    app.add_middleware(AuthorizationMiddleware, service_name=service.name)
    return policy
