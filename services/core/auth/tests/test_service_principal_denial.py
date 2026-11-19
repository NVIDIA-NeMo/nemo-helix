# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise evaluated-principal rejection through real HTTP, policy and token paths."""

import json
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import jwt
import nhx.common.auth.middleware as middleware_module
import nhx.common.config as common_config
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.jwks import AsyncJWKSClient
from nhx.common.auth.jwt import JWTValidator
from nhx.common.auth.middleware import AuthorizationMiddleware
from nhx.common.auth.models import Principal
from nhx.common.auth.workload_tokens import issue_service_workload_access_token_async, workload_public_jwk_async
from nhx.common.config.base import OIDCConfig, OIDCWorkloadConfig, TokenSigningConfig
from nhx.core.auth.api.v2.authz import endpoints
from nhx.core.auth.app import account_resolution
from nhx.core.auth.app.embedded_pdp import engine
from nhx.core.auth.config import AuthServiceConfig
from nhx.core.entities.app.repository.account_identity import (
    AccountIdentityConflictError,
    AccountIdentityRecord,
    AccountIdentityStore,
    AccountIdentityUnavailableError,
)

Mode = Literal["no-auth", "headers", "token"]
Fault = Literal["none", "connect", "timeout", "outer401", "outer403", "database", "inactive", "conflict", "engine"]
MODEL_PATH = "/apis/models/v2/workspaces/{workspace}/models"

pytestmark = pytest.mark.asyncio


class IdentityStore(AccountIdentityStore):
    def __init__(self, fault: Fault) -> None:
        self.fault = fault

    async def resolve_or_materialize(
        self,
        *,
        issuer: str,
        subject: str,
        subject_claim: str,
        account_type: Literal["user", "service"],
        display_name: str | None = None,
        primary_email: str | None = None,
        link_key: str | None = None,
        link_key_claim: str | None = None,
        claims_snapshot: dict[str, Any] | None = None,
        linked_via: str = "resolver_materialization",
        linked_by: str | None = None,
    ) -> AccountIdentityRecord:
        if self.fault == "inactive":
            raise AccountIdentityUnavailableError("private inactive identity")
        if self.fault == "conflict":
            raise AccountIdentityConflictError("private materialization conflict")
        if self.fault == "database":
            raise RuntimeError("private database failure")
        return AccountIdentityRecord(
            account_id=f"account-{subject}",
            account_type=account_type,
            identity_id=f"identity-{subject}",
            issuer=issuer,
            subject=subject,
        )


@dataclass
class Scenario:
    caller: httpx.AsyncClient
    pdp: httpx.AsyncClient
    config: AuthServiceConfig
    handler_calls: list[str] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)


ScenarioFactory = Callable[[Mode, Fault], AbstractAsyncContextManager[Scenario]]


@pytest.fixture
def scenarios(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ScenarioFactory:
    key_path = tmp_path / "workload.pem"
    key_path.write_bytes(
        rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    policy_path = Path(__file__).parents[1] / "src/nhx/core/auth/assets/static-authz.yaml"

    @asynccontextmanager
    async def create(mode: Mode, fault: Fault) -> AsyncGenerator[Scenario]:
        cfg = AuthServiceConfig(
            enabled=mode != "no-auth",
            policy_decision_point_base_url="https://pdp.test",
            bundle_cache_seconds=5,
            token_signing=TokenSigningConfig(key_id="test", private_key_file=str(key_path)),
            oidc=OIDCConfig(
                enabled=mode == "token",
                issuer="https://external-idp.test",
                audience="nemo-helix",
                workload=OIDCWorkloadConfig(
                    client_id="test-workload",
                    audience="nemo-helix",
                    token_issuer="https://auth.test/apis/auth",
                    scope="platform:read",
                )
                if mode == "token"
                else None,
            ),
        )
        monkeypatch.setattr(common_config, "get_service_config", lambda cls: cfg)
        monkeypatch.setattr(endpoints, "get_service_config", lambda cls: cfg)
        monkeypatch.setattr(middleware_module, "get_auth_config", lambda: cfg)
        store = IdentityStore(fault)

        async def identity_store() -> AccountIdentityStore:
            return store

        monkeypatch.setattr(account_resolution, "_account_identity_store", identity_store)
        if fault == "engine":

            def broken_policy(entrypoint: str, input_data: dict[str, Any]) -> dict[str, Any]:
                raise engine.PolicyEngineError("private policy failure")

            monkeypatch.setattr(endpoints, "evaluate", broken_policy)
        if mode == "token":
            jwks = AsyncJWKSClient("https://external-idp.test/jwks")
            jwks._jwks = {"keys": [await workload_public_jwk_async(cfg)]}
            jwks._jwks_cache_time = time.time()

            async def local_jwks(self: JWTValidator) -> AsyncJWKSClient:
                return jwks

            monkeypatch.setattr(JWTValidator, "_get_jwks_client", local_jwks)
        data = yaml.safe_load(policy_path.read_text())
        data["authz"]["roles"]["TaskReader"] = {"permissions": ["models.list"]}
        data["authz"]["principals"] = {"team-readers": {"workspaces": {"allowed": ["TaskReader"]}}}
        data["authz"].setdefault("workspaces", {}).update({"allowed": {}, "blocked": {}})
        engine._reset_policy_state_for_testing()
        engine.set_policy_data(data)
        pdp_app = FastAPI()
        pdp_app.include_router(endpoints.router, prefix="/apis/auth")
        pdp_app.dependency_overrides[endpoints.get_entity_client] = lambda: store
        protected_pdp = AuthorizationMiddleware(pdp_app, service_name="auth")
        requests: list[httpx.Request] = []
        calls: list[str] = []

        async def forward(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if fault == "connect":
                raise httpx.ConnectError("private connection failure", request=request)
            if fault == "timeout":
                raise httpx.ReadTimeout("private timeout", request=request)
            if fault in ("outer401", "outer403"):
                return httpx.Response(
                    401 if fault == "outer401" else 403, json={"detail": "private service credential"}, request=request
                )
            return await httpx.ASGITransport(app=protected_pdp, raise_app_exceptions=False).handle_async_request(
                request
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(forward)) as pdp_http:
            app = FastAPI()

            @app.get(MODEL_PATH)
            async def business(workspace: str) -> dict[str, str | None]:
                calls.append(workspace)
                auth_client = auth_client_context.get()
                assert auth_client is not None
                principal = auth_client.principal
                return {"actor": principal.id, "subject": principal.on_behalf_of}

            protected = AuthorizationMiddleware(app, service_name="models", http_client=pdp_http)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=protected), base_url="https://models.test"
            ) as caller:
                yield Scenario(caller, pdp_http, cfg, calls, requests)
        engine._reset_policy_state_for_testing()

    return create


async def credentials(cfg: AuthServiceConfig, mode: Mode, actor: str = "jobs") -> dict[str, str]:
    creator = Principal(id="creator", email="creator@example.test", groups=["team-readers"])
    if mode == "token":
        token = await issue_service_workload_access_token_async(cfg, service_name=actor, on_behalf_of=creator)
        return {"Authorization": f"Bearer {token}"}
    return {
        "X-NHX-Principal-Id": f"service:{actor}",
        "X-NHX-Principal-On-Behalf-Of": creator.id,
        "X-NHX-Principal-On-Behalf-Of-Groups": ",".join(creator.groups),
    }


@pytest.mark.parametrize("mode", ["no-auth", "headers", "token"])
@pytest.mark.parametrize("workspace,actor", [("allowed", "jobs"), ("blocked", "jobs"), ("allowed", "unknown-task")])
async def test_request_decision(scenarios: ScenarioFactory, mode: Mode, workspace: str, actor: str) -> None:
    async with scenarios(mode, "none") as case:
        response = await case.caller.get(
            MODEL_PATH.format(workspace=workspace), headers=await credentials(case.config, mode, actor)
        )
        allowed = mode == "no-auth" or (workspace == "allowed" and actor == "jobs")
        assert response.status_code == (200 if allowed else 403), response.text
        assert case.handler_calls == ([workspace] if allowed else [])
        if mode == "no-auth":
            assert not case.requests
        else:
            assert len(case.requests) == 1
            sent = case.requests[0]
            if mode == "token":
                assert "authorization" in sent.headers
                assert "x-nhx-principal-id" not in sent.headers
            else:
                assert sent.headers["x-nhx-principal-id"] == "service:models"
            assert json.loads(sent.content)["input"]["on_behalf_of_principal_id"] == "creator"


@pytest.mark.parametrize("mode", ["no-auth", "headers", "token"])
@pytest.mark.parametrize(
    "fault,status",
    [
        ("connect", 503),
        ("timeout", 504),
        ("outer401", 502),
        ("outer403", 502),
        ("database", 502),
        ("inactive", 502),
        ("conflict", 502),
        ("engine", 502),
    ],
)
async def test_dependency_errors_stay_errors(scenarios: ScenarioFactory, mode: Mode, fault: Fault, status: int) -> None:
    async with scenarios(mode, fault) as case:
        response = await case.caller.get(
            MODEL_PATH.format(workspace="allowed"), headers=await credentials(case.config, mode)
        )
        assert response.status_code == (200 if mode == "no-auth" else status), response.text
        assert "private" not in response.text
        assert case.handler_calls == (["allowed"] if mode == "no-auth" else [])


@pytest.mark.parametrize("entrypoint", ["allow", "has_permissions", "has_role"])
async def test_only_allow_returns_a_negative_decision(scenarios: ScenarioFactory, entrypoint: str) -> None:
    async with scenarios("headers", "none") as case:
        response = await case.pdp.post(
            case.config.get_pdp_url(entrypoint),
            headers={"X-NHX-Principal-Id": "service:models"},
            json={"input": {"principal_id": "service:unknown-task"}},
        )
        assert response.status_code == (200 if entrypoint == "allow" else 403)
        if entrypoint == "allow":
            assert response.json() == {
                "result": {
                    "allowed": False,
                    "reason": "service_principal_not_allowed",
                    "headers": {"X-NHX-Authorized": "false"},
                }
            }
            assert "unknown-task" not in response.text


@pytest.mark.parametrize("failure", ["signature", "expired", "audience", "issuer"])
async def test_bad_signed_credentials_stay_unauthenticated(scenarios: ScenarioFactory, failure: str) -> None:
    async with scenarios("token", "none") as case:
        headers = await credentials(case.config, "token")
        token = headers["Authorization"].removeprefix("Bearer ")
        claims = jwt.decode(token, options={"verify_signature": False})
        if failure == "expired":
            claims.update(iat=0, nbf=0, exp=1)
        elif failure == "audience":
            claims["aud"] = "wrong"
        elif failure == "issuer":
            claims["iss"] = "https://wrong.test"
        assert case.config.token_signing.private_key_file is not None
        key = Path(case.config.token_signing.private_key_file).read_bytes()
        if failure == "signature":
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        bad_token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test"})
        response = await case.caller.get(
            MODEL_PATH.format(workspace="allowed"), headers={"Authorization": f"Bearer {bad_token}"}
        )
        assert response.status_code == 401
        assert not case.requests and not case.handler_calls


async def test_token_mode_rejects_trusted_headers(scenarios: ScenarioFactory) -> None:
    async with scenarios("token", "none") as case:
        headers = await credentials(case.config, "token")
        headers["X-NHX-Principal-Id"] = "service:jobs"
        response = await case.caller.get(MODEL_PATH.format(workspace="allowed"), headers=headers)
        assert response.status_code == 400
        assert not case.requests and not case.handler_calls


async def test_no_auth_does_not_require_a_working_pdp(scenarios: ScenarioFactory) -> None:
    async with scenarios("no-auth", "connect") as case:
        response = await case.caller.get(
            MODEL_PATH.format(workspace="allowed"), headers={"Authorization": "Bearer invalid"}
        )
        assert response.status_code == 200
        assert not case.requests
