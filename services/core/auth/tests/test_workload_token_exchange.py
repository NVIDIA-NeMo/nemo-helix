# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import nhx.common.auth.signing_keys as signing_keys_mod
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm
from nemo_helix_plugin.auth import AuthContext as PluginAuthContext
from nhx.common.auth import Principal
from nhx.common.auth.signing_keys import RSASigningKeyCache
from nhx.common.auth.workload_delegations import (
    DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE,
    WorkloadDelegationEntity,
    WorkloadDelegationScope,
    create_opaque_docker_proof_token,
    docker_delegation_name,
    docker_workload_delegation_name,
    kubernetes_pod_uid_delegation_name,
    reference_delegation_name,
)
from nhx.common.config import AuthConfig, Configuration
from nhx.common.config.base import AccessKeyConfig, OIDCConfig, TokenSigningConfig
from nhx.common.entities import SYSTEM_WORKSPACE, EntityNotFoundError
from nhx.core.auth.api.v2 import workload_token_exchange as exchange
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _clear_config_overrides():
    yield
    Configuration.clear_overrides()


@pytest.fixture
def exchange_service() -> exchange.WorkloadTokenExchangeService:
    return exchange.WorkloadTokenExchangeService()


class _FakeEntityClient:
    def __init__(self) -> None:
        self.entities: dict[str, WorkloadDelegationEntity] = {}
        self.get_calls: list[str] = []

    async def get(
        self,
        entity_type: type[WorkloadDelegationEntity],
        name: str,
        *,
        workspace: str,
    ) -> WorkloadDelegationEntity:
        self.get_calls.append(name)
        assert entity_type is WorkloadDelegationEntity
        assert workspace == SYSTEM_WORKSPACE
        try:
            return self.entities[name]
        except KeyError as exc:
            raise EntityNotFoundError(f"Entity '{name}' not found") from exc


@pytest.fixture
def entity_client() -> _FakeEntityClient:
    return _FakeEntityClient()


@pytest.fixture
def workload_signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _private_key_pem(private_key: rsa.RSAPrivateKey) -> str:
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def _openapi(client: TestClient) -> dict[str, Any]:
    app = client.app
    assert isinstance(app, FastAPI)
    return app.openapi()


def _exchange_form(
    subject_token: str,
    *,
    subject_token_type: str = exchange.JWT_TOKEN_TYPE,
    audience: str | None = None,
    scope: str | None = None,
) -> dict[str, str]:
    data = {
        "grant_type": exchange.TOKEN_EXCHANGE_GRANT_TYPE,
        "client_id": "nemo-helix-workload",
        "subject_token": subject_token,
        "subject_token_type": subject_token_type,
    }
    if audience is not None:
        data["audience"] = audience
    if scope is not None:
        data["scope"] = scope
    return data


def _post_form_items(client: TestClient, items: list[tuple[str, str]]) -> Any:
    return client.post(
        "/token",
        content=urlencode(items),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )


def _decode_access_token(
    token: str,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    *,
    audience: str = "nemo-helix",
) -> dict[str, Any]:
    signing_key = exchange_service.workload_signing_key(exchange_config)
    return exchange.jwt.decode(token, signing_key.public_key, algorithms=["RS256"], audience=audience)


def _docker_delegation_name() -> str:
    return docker_delegation_name(
        workload_workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step_id="step-a",
    )


def _delegation_entity(**overrides: Any) -> WorkloadDelegationEntity:
    delegation_name = _docker_delegation_name()
    entity = WorkloadDelegationEntity(
        name=delegation_name,
        workspace=SYSTEM_WORKSPACE,
        workload_subject=delegation_name,
        workload_audience="nemo-helix",
        workload_workspace="default",
        job_id="job-123",
        attempt_id="attempt-1",
        step_id="step-a",
        auth_context=PluginAuthContext.from_principal(
            Principal(id="creator@example.com", email="creator@example.com", groups=["workspace-editors"])
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    return entity.model_copy(update=overrides)


def _deployment_docker_delegation_name() -> str:
    return docker_workload_delegation_name(
        scope=WorkloadDelegationScope(
            workload_workspace="default",
            workload_kind="deployment",
            workload_instance_id="deployment-123",
        ),
        workload_generation="container-abc",
    )


def _deployment_delegation_entity(**overrides: Any) -> WorkloadDelegationEntity:
    delegation_name = _deployment_docker_delegation_name()
    entity = WorkloadDelegationEntity(
        name=delegation_name,
        workspace=SYSTEM_WORKSPACE,
        workload_subject=delegation_name,
        workload_audience="nemo-helix",
        workload_workspace="default",
        workload_kind="deployment",
        workload_id="deployment-123",
        workload_generation="container-abc",
        auth_context=PluginAuthContext.from_principal(
            Principal(id="creator@example.com", email="creator@example.com", groups=["workspace-editors"])
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    return entity.model_copy(update=overrides)


def _reference_delegation_entity(**overrides: Any) -> WorkloadDelegationEntity:
    values = {
        "name": reference_delegation_name(
            workload_audience="nemo-helix",
            workload_subject="system:serviceaccount:nemo-runs:job-runner",
            bound_reference_name=exchange.KUBERNETES_POD_UID_REFERENCE_NAME,
            bound_reference_value="pod-uid-123",
        ),
        "workload_subject": "system:serviceaccount:nemo-runs:job-runner",
        "bound_reference_name": exchange.KUBERNETES_POD_UID_REFERENCE_NAME,
        "bound_reference_value": "pod-uid-123",
    }
    values.update(overrides)
    return _delegation_entity(**values)


def _deployment_reference_delegation_entity(**overrides: Any) -> WorkloadDelegationEntity:
    workload_subject = "system:serviceaccount:nemo-deployments:deployment-runner"
    pod_uid = "deployment-pod-uid-123"
    entity = WorkloadDelegationEntity(
        name=kubernetes_pod_uid_delegation_name(
            workload_audience="nemo-helix",
            workload_subject=workload_subject,
            pod_uid=pod_uid,
        ),
        workspace=SYSTEM_WORKSPACE,
        workload_subject=workload_subject,
        workload_audience="nemo-helix",
        workload_workspace="default",
        workload_kind="deployment",
        workload_id="deployment-123",
        workload_generation=pod_uid,
        auth_context=PluginAuthContext.from_principal(
            Principal(id="creator@example.com", email="creator@example.com", groups=["workspace-editors"])
        ),
        bound_reference_name=exchange.KUBERNETES_POD_UID_REFERENCE_NAME,
        bound_reference_value=pod_uid,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    return entity.model_copy(update=overrides)


def _pod_uid_bound_reference(value: str = "pod-uid-123") -> exchange.VerifiedWorkloadReference:
    return exchange.VerifiedWorkloadReference(name=exchange.KUBERNETES_POD_UID_REFERENCE_NAME, value=value)


def _decoded_subject_token(
    claims: dict[str, Any],
    *,
    bound_reference: exchange.VerifiedWorkloadReference | None = None,
) -> exchange.DecodedSubjectToken:
    return exchange.DecodedSubjectToken(claims=claims, bound_reference=bound_reference)


@pytest.fixture
def exchange_config(workload_signing_key: rsa.RSAPrivateKey, tmp_path) -> AuthConfig:
    private_key_file = tmp_path / "workload-token-private-key.pem"
    private_key_file.write_text(_private_key_pem(workload_signing_key), encoding="utf-8")
    return AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.example.com/application/o/nemo-cli/",
            additional_issuers=["https://idp.example.com/application/o/nemo/"],
            client_id="nemo-helix-cli",
            workload_token_exchange_enabled=True,
            workload_client_id="nemo-helix-workload",
            workload_audience="nemo-helix",
            workload_scope="openid email groups",
            workload_subject_jwks_uri="https://idp.example.com/application/o/nemo-workload/jwks/",
            workload_subject_issuers=["https://idp.example.com/application/o/nemo-workload/"],
            workload_token_private_key_file=str(private_key_file),
        ),
    )


@pytest.fixture
def client(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
) -> TestClient:
    Configuration.set_override(exchange_config)
    app = FastAPI()
    app.dependency_overrides[exchange.get_workload_token_exchange_service] = lambda: exchange_service
    app.dependency_overrides[exchange.get_entity_client] = lambda: entity_client
    app.include_router(exchange.router)
    return TestClient(app, raise_server_exceptions=False)


def _jwks_test_client(config: AuthConfig, exchange_service: exchange.WorkloadTokenExchangeService) -> TestClient:
    Configuration.set_override(config)
    app = FastAPI()
    app.dependency_overrides[exchange.get_workload_token_exchange_service] = lambda: exchange_service
    app.include_router(exchange.router)
    return TestClient(app, raise_server_exceptions=False)


def test_jwks_publishes_workload_exchange_signing_key(client: TestClient) -> None:
    response = client.get("/jwks")

    assert response.status_code == 200
    keys = response.json()["keys"]
    assert keys[0]["kid"] == "nemo-helix-signing"
    assert keys[0]["use"] == "sig"
    assert keys[0]["alg"] == "RS256"


def test_jwks_returns_empty_key_set_when_workload_exchange_and_access_keys_are_disabled(
    exchange_service: exchange.WorkloadTokenExchangeService,
) -> None:
    config = AuthConfig(enabled=True)
    client = _jwks_test_client(config, exchange_service)

    response = client.get("/jwks")

    assert response.status_code == 200
    assert response.json() == {"keys": []}


def test_jwks_returns_only_access_key_signing_key_when_workload_exchange_is_disabled(
    tmp_path: Path,
    exchange_service: exchange.WorkloadTokenExchangeService,
) -> None:
    access_key_private_key_file = tmp_path / "access-key-private.pem"
    access_key_private_key_file.write_text(
        _private_key_pem(rsa.generate_private_key(public_exponent=65537, key_size=2048)),
        encoding="utf-8",
    )
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            key_id="access-key-signing",
            private_key_file=str(access_key_private_key_file),
        ),
        access_keys=AccessKeyConfig(enabled=True),
    )
    client = _jwks_test_client(config, exchange_service)

    response = client.get("/jwks")

    assert response.status_code == 200
    assert [key["kid"] for key in response.json()["keys"]] == ["access-key-signing"]


def test_jwks_includes_distinct_workload_and_access_key_signing_keys(
    tmp_path: Path,
    exchange_service: exchange.WorkloadTokenExchangeService,
) -> None:
    workload_private_key_file = tmp_path / "workload-private.pem"
    access_key_private_key_file = tmp_path / "access-key-private.pem"
    workload_private_key_file.write_text(
        _private_key_pem(rsa.generate_private_key(public_exponent=65537, key_size=2048)),
        encoding="utf-8",
    )
    access_key_private_key_file.write_text(
        _private_key_pem(rsa.generate_private_key(public_exponent=65537, key_size=2048)),
        encoding="utf-8",
    )
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            key_id="access-key-signing",
            private_key_file=str(access_key_private_key_file),
        ),
        oidc=OIDCConfig(
            enabled=True,
            workload_token_exchange_enabled=True,
            workload_token_key_id="workload-signing",
            workload_token_private_key_file=str(workload_private_key_file),
        ),
        access_keys=AccessKeyConfig(enabled=True),
    )
    client = _jwks_test_client(config, exchange_service)

    response = client.get("/jwks")

    assert response.status_code == 200
    assert [key["kid"] for key in response.json()["keys"]] == ["workload-signing", "access-key-signing"]


def test_jwks_deduplicates_shared_workload_and_access_key_signing_key(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
) -> None:
    config = exchange_config.model_copy(
        update={
            "token_signing": exchange_config.token_signing.model_copy(
                update={"private_key_file": exchange_config.oidc.workload_token_private_key_file}
            ),
            "access_keys": AccessKeyConfig(enabled=True),
        }
    )
    client = _jwks_test_client(config, exchange_service)

    response = client.get("/jwks")

    assert response.status_code == 200
    assert [key["kid"] for key in response.json()["keys"]] == ["nemo-helix-signing"]


def test_jwks_openapi_documents_jwks_response(client: TestClient) -> None:
    openapi = _openapi(client)
    operation = openapi["paths"]["/jwks"]["get"]

    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/JsonWebKeySetResponse"
    }
    jwks_schema = openapi["components"]["schemas"]["JsonWebKeySetResponse"]
    assert jwks_schema["required"] == ["keys"]
    assert jwks_schema["properties"]["keys"]["type"] == "array"
    assert jwks_schema["properties"]["keys"]["items"] == {"$ref": "#/components/schemas/JsonWebKey"}
    assert openapi["components"]["schemas"]["JsonWebKey"] == {
        "additionalProperties": True,
        "properties": {},
        "type": "object",
        "title": "JsonWebKey",
        "description": "JSON Web Key object.",
    }


def test_token_exchange_openapi_documents_form_request_and_token_response(client: TestClient) -> None:
    openapi = _openapi(client)
    operation = openapi["paths"]["/token"]["post"]

    request_schema = operation["requestBody"]["content"]["application/x-www-form-urlencoded"]["schema"]
    if "$ref" in request_schema:
        request_schema = openapi["components"]["schemas"][request_schema["$ref"].rsplit("/", 1)[-1]]

    assert {
        "grant_type",
        "client_id",
        "subject_token",
        "subject_token_type",
        "requested_token_type",
        "audience",
        "resource",
        "actor_token",
        "actor_token_type",
        "scope",
    } <= set(request_schema["properties"])
    assert DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE in request_schema["properties"]["subject_token_type"]["enum"]
    for unsupported_property in ("resource", "actor_token", "actor_token_type"):
        assert request_schema["properties"][unsupported_property] == {
            "type": "string",
            "description": "Unsupported for the jobs workload token exchange profile.",
            "not": {},
        }
    assert set(operation["responses"]) == {"200", "400", "401"}
    assert operation["responses"]["200"]["description"] == "Successful Response"
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/WorkloadTokenExchangeResponse"
    }
    assert operation["responses"]["400"]["description"] == "RFC 8693 token exchange error"
    assert operation["responses"]["400"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/WorkloadTokenExchangeErrorResponse"
    }
    assert operation["responses"]["401"]["description"] == "OAuth 2.0 invalid_client error"
    assert operation["responses"]["401"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/WorkloadTokenExchangeErrorResponse"
    }
    error_schema = openapi["components"]["schemas"]["WorkloadTokenExchangeErrorResponse"]
    assert error_schema["properties"]["error"]["type"] == "string"
    error_description = error_schema["properties"]["error"]["description"]
    for error_code in ("invalid_client", "invalid_request", "invalid_scope", "invalid_target"):
        assert error_code in error_description
    response_schema = openapi["components"]["schemas"]["WorkloadTokenExchangeResponse"]
    assert {
        "access_token",
        "issued_token_type",
        "token_type",
        "expires_in",
        "scope",
    } == set(response_schema["properties"])
    assert response_schema["required"] == [
        "access_token",
        "issued_token_type",
        "token_type",
        "expires_in",
    ]


def test_token_exchange_rejects_subject_token_without_subject(
    client: TestClient,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token({"email": "svc@example.com"})

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post(
        "/token",
        data={
            "grant_type": exchange.TOKEN_EXCHANGE_GRANT_TYPE,
            "client_id": "nemo-helix-workload",
            "subject_token": "subject-token",
            "subject_token_type": exchange.JWT_TOKEN_TYPE,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_request",
        "error_description": "Could not validate subject token",
    }


def test_token_exchange_rejects_disallowed_audience_before_subject_validation(
    client: TestClient,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        raise AssertionError("disallowed audience should be rejected before subject token validation")

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post(
        "/token",
        data={
            "grant_type": exchange.TOKEN_EXCHANGE_GRANT_TYPE,
            "client_id": "nemo-helix-workload",
            "subject_token": "subject-token",
            "subject_token_type": exchange.JWT_TOKEN_TYPE,
            "audience": "unexpected-audience",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_target",
        "error_description": "Requested audience is not allowed",
    }


@pytest.mark.parametrize(
    "actor_fields",
    [
        {"actor_token": "actor-token"},
        {"actor_token_type": exchange.JWT_TOKEN_TYPE},
        {"actor_token": "actor-token", "actor_token_type": exchange.JWT_TOKEN_TYPE},
    ],
)
def test_token_exchange_rejects_actor_token_parameters(
    client: TestClient,
    actor_fields: dict[str, str],
) -> None:
    response = client.post(
        "/token",
        data={**_exchange_form("subject-token"), **actor_fields},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_request",
        "error_description": "actor_token is not supported for this workload token exchange profile",
    }


@pytest.mark.parametrize(
    ("field", "values", "expected_response"),
    [
        (
            "actor_token",
            [""],
            {
                "error": "invalid_request",
                "error_description": "actor_token is not supported for this workload token exchange profile",
            },
        ),
        (
            "actor_token",
            ["   "],
            {
                "error": "invalid_request",
                "error_description": "actor_token is not supported for this workload token exchange profile",
            },
        ),
        (
            "actor_token",
            ["", "   "],
            {
                "error": "invalid_request",
                "error_description": "actor_token is not supported for this workload token exchange profile",
            },
        ),
        (
            "actor_token_type",
            [""],
            {
                "error": "invalid_request",
                "error_description": "actor_token is not supported for this workload token exchange profile",
            },
        ),
        (
            "actor_token_type",
            ["   "],
            {
                "error": "invalid_request",
                "error_description": "actor_token is not supported for this workload token exchange profile",
            },
        ),
        (
            "actor_token_type",
            ["", "   "],
            {
                "error": "invalid_request",
                "error_description": "actor_token is not supported for this workload token exchange profile",
            },
        ),
        (
            "resource",
            [""],
            {
                "error": "invalid_target",
                "error_description": "resource is not supported for workload token exchange",
            },
        ),
        (
            "resource",
            ["   "],
            {
                "error": "invalid_target",
                "error_description": "resource is not supported for workload token exchange",
            },
        ),
        (
            "resource",
            ["", "   "],
            {
                "error": "invalid_target",
                "error_description": "resource is not supported for workload token exchange",
            },
        ),
    ],
)
def test_token_exchange_rejects_unsupported_fields_by_presence(
    client: TestClient,
    field: str,
    values: list[str],
    expected_response: dict[str, str],
) -> None:
    response = _post_form_items(
        client,
        [
            *_exchange_form("subject-token").items(),
            *((field, value) for value in values),
        ],
    )

    assert response.status_code == 400
    assert response.json() == expected_response


def test_token_exchange_rejects_resource_target_before_subject_validation(client: TestClient) -> None:
    response = client.post(
        "/token",
        data={**_exchange_form("subject-token"), "resource": "https://nhx.example.com/apis/secrets"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_target",
        "error_description": "resource is not supported for workload token exchange",
    }


def test_token_exchange_rejects_multiple_audiences_before_subject_validation(client: TestClient) -> None:
    response = _post_form_items(
        client,
        [
            *_exchange_form("subject-token").items(),
            ("audience", "nemo-helix"),
            ("audience", "extra-audience"),
        ],
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": "invalid_target",
        "error_description": "Only one audience is supported for workload token exchange",
    }


def test_token_exchange_accepts_single_allowed_audience(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange_config.oidc.workload_allowed_audiences.append("extra-audience")

    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token({"sub": "workload-subject"})

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post("/token", data=_exchange_form("subject-token", audience="extra-audience"))

    assert response.status_code == 200
    claims = _decode_access_token(
        response.json()["access_token"],
        exchange_config,
        exchange_service,
        audience="extra-audience",
    )
    assert claims["aud"] == "extra-audience"


def test_token_exchange_mints_access_token_signed_by_configured_key(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        captured["subject_audience"] = audience
        return _decoded_subject_token(
            {
                "sub": "workload-subject",
                "email": "svc@example.com",
                "groups": ["svc-group", "system:serviceaccounts"],
            }
        )

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post(
        "/token",
        data={
            "grant_type": exchange.TOKEN_EXCHANGE_GRANT_TYPE,
            "client_id": "nemo-helix-workload",
            "subject_token": "subject-token",
            "subject_token_type": exchange.JWT_TOKEN_TYPE,
        },
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["expires_in"] == exchange_config.oidc.workload_token_ttl_seconds
    access_token = response_body["access_token"]
    signing_key = exchange_service.workload_signing_key(exchange_config)
    assert exchange.jwt.get_unverified_header(access_token)["kid"] == signing_key.kid
    claims = exchange.jwt.decode(access_token, signing_key.public_key, algorithms=["RS256"], audience="nemo-helix")
    assert captured["subject_audience"] == "nemo-helix-workload"
    assert claims["sub"] == "workload-subject"
    assert claims["email"] == "svc@example.com"
    assert claims["groups"] == "svc-group,system:serviceaccounts"


def test_token_exchange_ignores_non_string_subject_group_values(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token(
            {
                "sub": "workload-subject",
                "groups": ["svc-group", 42, " system:serviceaccounts "],
            }
        )

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post(
        "/token",
        data={
            "grant_type": exchange.TOKEN_EXCHANGE_GRANT_TYPE,
            "client_id": "nemo-helix-workload",
            "subject_token": "subject-token",
            "subject_token_type": exchange.JWT_TOKEN_TYPE,
        },
    )

    assert response.status_code == 200
    claims = _decode_access_token(response.json()["access_token"], exchange_config, exchange_service)
    assert claims["groups"] == "svc-group,system:serviceaccounts"


def test_token_exchange_intersects_requested_scope_with_configured_workload_scope(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token({"sub": "workload-subject"})

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post(
        "/token",
        data=_exchange_form("subject-token", scope="openid admin:write email"),
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["scope"] == "openid email"
    claims = _decode_access_token(response_body["access_token"], exchange_config, exchange_service)
    assert claims["scope"] == "openid email"


def test_token_exchange_omits_scope_when_requested_scope_is_not_allowed(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token({"sub": "workload-subject"})

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post(
        "/token",
        data=_exchange_form("subject-token", scope="admin:write"),
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["scope"] is None
    claims = _decode_access_token(response_body["access_token"], exchange_config, exchange_service)
    assert "scope" not in claims


def test_opaque_docker_proof_token_with_missing_row_returns_invalid_request(client: TestClient) -> None:
    subject_token, _token_hash = create_opaque_docker_proof_token(_docker_delegation_name())

    response = client.post(
        "/token",
        data=_exchange_form(subject_token, subject_token_type=DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE),
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_opaque_docker_proof_token_mints_delegated_access_token(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
) -> None:
    subject_token, token_hash = create_opaque_docker_proof_token(_docker_delegation_name())
    delegation = _delegation_entity(opaque_subject_token_hash=token_hash)
    entity_client.entities[delegation.name] = delegation

    response = client.post(
        "/token",
        data=_exchange_form(subject_token, subject_token_type=DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE),
    )

    assert response.status_code == 200
    claims = _decode_access_token(response.json()["access_token"], exchange_config, exchange_service)
    assert claims["sub"] == "creator@example.com"
    assert claims["act"] == {"sub": delegation.name}


def test_opaque_docker_proof_token_mints_delegated_access_token_for_deployment_row(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
) -> None:
    subject_token, token_hash = create_opaque_docker_proof_token(_deployment_docker_delegation_name())
    delegation = _deployment_delegation_entity(opaque_subject_token_hash=token_hash)
    entity_client.entities[delegation.name] = delegation

    response = client.post(
        "/token",
        data=_exchange_form(subject_token, subject_token_type=DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE),
    )

    assert response.status_code == 200
    claims = _decode_access_token(response.json()["access_token"], exchange_config, exchange_service)
    assert claims["sub"] == "creator@example.com"
    assert claims["act"] == {"sub": delegation.name}


def test_opaque_docker_proof_token_requires_private_subject_token_type(
    client: TestClient,
    entity_client: _FakeEntityClient,
) -> None:
    subject_token, token_hash = create_opaque_docker_proof_token(_docker_delegation_name())
    delegation = _delegation_entity(opaque_subject_token_hash=token_hash)
    entity_client.entities[delegation.name] = delegation

    response = client.post("/token", data=_exchange_form(subject_token, subject_token_type=exchange.JWT_TOKEN_TYPE))

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert entity_client.get_calls == []


def test_delegated_access_token_expiry_is_capped_by_delegation_expiry(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
) -> None:
    exchange_config.oidc.workload_token_ttl_seconds = 600
    subject_token, token_hash = create_opaque_docker_proof_token(_docker_delegation_name())
    delegation_expires_at = datetime.now(timezone.utc) + timedelta(seconds=120)
    delegation = _delegation_entity(
        opaque_subject_token_hash=token_hash,
        expires_at=delegation_expires_at,
    )
    entity_client.entities[delegation.name] = delegation

    response = client.post(
        "/token",
        data=_exchange_form(subject_token, subject_token_type=DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE),
    )

    assert response.status_code == 200
    response_body = response.json()
    assert 0 < response_body["expires_in"] <= 120
    claims = _decode_access_token(response_body["access_token"], exchange_config, exchange_service)
    assert claims["exp"] <= int(delegation_expires_at.timestamp())


def test_delegated_access_token_expiry_keeps_configured_ttl_when_delegation_expires_later(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
) -> None:
    exchange_config.oidc.workload_token_ttl_seconds = 120
    subject_token, token_hash = create_opaque_docker_proof_token(_docker_delegation_name())
    delegation = _delegation_entity(
        opaque_subject_token_hash=token_hash,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    entity_client.entities[delegation.name] = delegation

    response = client.post(
        "/token",
        data=_exchange_form(subject_token, subject_token_type=DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE),
    )

    assert response.status_code == 200
    response_body = response.json()
    assert 0 < response_body["expires_in"] <= 120
    claims = _decode_access_token(response_body["access_token"], exchange_config, exchange_service)
    assert claims["exp"] - claims["iat"] <= 120


@pytest.mark.parametrize(
    "delegation_overrides",
    [
        {"opaque_subject_token_hash": "v1:sha256:wrong"},
        {"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)},
        {"revoked_at": datetime.now(timezone.utc)},
        {
            "bound_reference_name": exchange.KUBERNETES_POD_UID_REFERENCE_NAME,
            "bound_reference_value": "pod-uid-123",
        },
    ],
)
def test_opaque_docker_proof_token_rejects_ineligible_rows(
    client: TestClient,
    entity_client: _FakeEntityClient,
    delegation_overrides: dict[str, Any],
) -> None:
    subject_token, token_hash = create_opaque_docker_proof_token(_docker_delegation_name())
    entity_overrides = {"opaque_subject_token_hash": token_hash, **delegation_overrides}
    delegation = _delegation_entity(**entity_overrides)
    entity_client.entities[delegation.name] = delegation

    response = client.post(
        "/token",
        data=_exchange_form(subject_token, subject_token_type=DOCKER_OPAQUE_WORKLOAD_PROOF_TOKEN_TYPE),
    )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_kubernetes_tokenreview_without_pod_uid_keeps_workload_only_exchange(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token(
            {
                "sub": "system:serviceaccount:nemo-runs:job-runner",
                "groups": ["system:serviceaccounts"],
            }
        )

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post("/token", data=_exchange_form("kubernetes-token"))

    assert response.status_code == 200
    claims = _decode_access_token(response.json()["access_token"], exchange_config, exchange_service)
    assert entity_client.get_calls == []
    assert claims["sub"] == "system:serviceaccount:nemo-runs:job-runner"
    assert claims["groups"] == "system:serviceaccounts"
    assert "act" not in claims


def test_kubernetes_reference_row_mints_delegated_access_token(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegation = _reference_delegation_entity()
    entity_client.entities[delegation.name] = delegation

    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token(
            {
                "sub": "system:serviceaccount:nemo-runs:job-runner",
                "groups": ["system:serviceaccounts"],
            },
            bound_reference=_pod_uid_bound_reference(),
        )

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post("/token", data=_exchange_form("kubernetes-token"))

    assert response.status_code == 200
    claims = _decode_access_token(response.json()["access_token"], exchange_config, exchange_service)
    assert entity_client.get_calls == [delegation.name]
    assert claims["sub"] == "creator@example.com"
    assert claims["act"] == {
        "sub": "system:serviceaccount:nemo-runs:job-runner",
        "groups": "system:serviceaccounts",
    }


def test_kubernetes_reference_row_mints_delegated_access_token_for_deployment_row(
    client: TestClient,
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    entity_client: _FakeEntityClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegation = _deployment_reference_delegation_entity()
    entity_client.entities[delegation.name] = delegation

    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token(
            {
                "sub": "system:serviceaccount:nemo-deployments:deployment-runner",
                "groups": ["system:serviceaccounts"],
            },
            bound_reference=_pod_uid_bound_reference("deployment-pod-uid-123"),
        )

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = client.post("/token", data=_exchange_form("kubernetes-token"))

    assert response.status_code == 200
    claims = _decode_access_token(response.json()["access_token"], exchange_config, exchange_service)
    assert entity_client.get_calls == [delegation.name]
    assert claims["sub"] == "creator@example.com"
    assert claims["act"] == {
        "sub": "system:serviceaccount:nemo-deployments:deployment-runner",
        "groups": "system:serviceaccounts",
    }


def test_kubernetes_reference_lookup_retries_missing_row_without_waiting(
    exchange_config: AuthConfig,
    entity_client: _FakeEntityClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    exchange_service = exchange.WorkloadTokenExchangeService(
        delegation_lookup_retry_timeout_seconds=0.3,
        delegation_lookup_retry_interval_seconds=0.1,
        sleep=sleep,
    )
    app = FastAPI()
    Configuration.set_override(exchange_config)
    app.dependency_overrides[exchange.get_workload_token_exchange_service] = lambda: exchange_service
    app.dependency_overrides[exchange.get_entity_client] = lambda: entity_client
    app.include_router(exchange.router)
    retry_client = TestClient(app, raise_server_exceptions=False)

    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token(
            {"sub": "system:serviceaccount:nemo-runs:job-runner"},
            bound_reference=_pod_uid_bound_reference(),
        )

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = retry_client.post("/token", data=_exchange_form("kubernetes-token"))

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert len(entity_client.get_calls) == 4
    assert sleeps == [0.1, 0.1, 0.1]


def test_kubernetes_reference_lookup_default_retry_budget_uses_five_seconds(
    exchange_config: AuthConfig,
    entity_client: _FakeEntityClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    exchange_service = exchange.WorkloadTokenExchangeService(sleep=sleep)
    app = FastAPI()
    Configuration.set_override(exchange_config)
    app.dependency_overrides[exchange.get_workload_token_exchange_service] = lambda: exchange_service
    app.dependency_overrides[exchange.get_entity_client] = lambda: entity_client
    app.include_router(exchange.router)
    retry_client = TestClient(app, raise_server_exceptions=False)

    async def decode_subject_token(
        config: AuthConfig, subject_token: str, audience: str
    ) -> exchange.DecodedSubjectToken:
        return _decoded_subject_token(
            {"sub": "system:serviceaccount:nemo-runs:job-runner"},
            bound_reference=_pod_uid_bound_reference(),
        )

    monkeypatch.setattr(exchange_service, "decode_subject_token", decode_subject_token)

    response = retry_client.post("/token", data=_exchange_form("kubernetes-token"))

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert len(entity_client.get_calls) == 51
    assert sleeps == [0.1] * 50


def test_validated_audience_accepts_configured_allowlist(exchange_config: AuthConfig) -> None:
    exchange_config.oidc.workload_allowed_audiences.append("extra-audience")

    assert exchange._validated_audience(exchange_config, "extra-audience") == "extra-audience"


def test_workload_signing_key_uses_shared_token_signing_when_workload_override_unset(
    workload_signing_key: rsa.RSAPrivateKey,
    tmp_path,
) -> None:
    private_key_file = tmp_path / "platform-token-private-key.pem"
    private_key_file.write_text(_private_key_pem(workload_signing_key), encoding="utf-8")
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="https://nhx.example.com/apis/auth",
            key_id="nemo-helix-signing",
            private_key_file=str(private_key_file),
        ),
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.example.com/application/o/nemo-cli/",
            client_id="nemo-helix-cli",
            workload_token_exchange_enabled=True,
        ),
    )

    signing_key = exchange.WorkloadTokenExchangeService().workload_signing_key(config)

    assert signing_key.kid == "nemo-helix-signing"


def test_workload_exchange_requires_resolved_token_signing_key_id() -> None:
    with pytest.raises(ValidationError, match="workload_token_key_id or auth.token_signing.key_id"):
        AuthConfig(
            enabled=True,
            token_signing=TokenSigningConfig(key_id=""),
            oidc=OIDCConfig(
                enabled=True,
                workload_token_exchange_enabled=True,
            ),
        )


def test_workload_exchange_accepts_workload_key_id_when_shared_key_id_unset() -> None:
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(key_id=""),
        oidc=OIDCConfig(
            enabled=True,
            workload_token_exchange_enabled=True,
            workload_token_key_id="workload-signing",
        ),
    )

    assert config.oidc.workload_token_key_id == "workload-signing"


def test_workload_exchange_requires_distinct_key_id_for_distinct_access_key_jwks_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="workload_token_key_id must be distinct"):
        AuthConfig(
            enabled=True,
            token_signing=TokenSigningConfig(
                key_id="nemo-helix-signing",
                private_key_file=str(tmp_path / "access-key.pem"),
            ),
            oidc=OIDCConfig(
                enabled=True,
                workload_token_exchange_enabled=True,
                workload_token_private_key_file=str(tmp_path / "workload-token.pem"),
            ),
            access_keys=AccessKeyConfig(enabled=True),
        )


def test_workload_signing_key_specific_override_wins_over_shared_token_signing(
    workload_signing_key: rsa.RSAPrivateKey,
    tmp_path,
) -> None:
    shared_private_key_file = tmp_path / "platform-token-private-key.pem"
    workload_private_key_file = tmp_path / "workload-token-private-key.pem"
    shared_private_key_file.write_text(_private_key_pem(workload_signing_key), encoding="utf-8")
    workload_private_key_file.write_text(_private_key_pem(workload_signing_key), encoding="utf-8")
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="https://nhx.example.com/apis/auth",
            key_id="nemo-helix-signing",
            private_key_file=str(shared_private_key_file),
        ),
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.example.com/application/o/nemo-cli/",
            client_id="nemo-helix-cli",
            workload_token_exchange_enabled=True,
            workload_token_key_id="nemo-workload-exchange",
            workload_token_private_key_file=str(workload_private_key_file),
        ),
    )

    signing_key = exchange.WorkloadTokenExchangeService().workload_signing_key(config)

    assert signing_key.kid == "nemo-workload-exchange"


def test_workload_signing_key_reuses_cached_private_key_file(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = signing_keys_mod._load_rsa_signing_key_async
    load_count = 0

    async def counted_load(**kwargs: Any) -> signing_keys_mod.RSASigningKey:
        nonlocal load_count
        load_count += 1
        return await original_load(**kwargs)

    monkeypatch.setattr(signing_keys_mod, "_load_rsa_signing_key_async", counted_load)

    signing_key = exchange_service.workload_signing_key(exchange_config)
    public_jwk = exchange_service.public_jwk(exchange_config)
    signing_key_again = exchange_service.workload_signing_key(exchange_config)

    assert signing_key_again is signing_key
    assert public_jwk["kid"] == signing_key.kid
    assert load_count == 1


class _AsyncOnlySigningKeyCache(RSASigningKeyCache):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def public_jwk_from_file(
        self,
        *,
        kid: str,
        private_key_file: str | None,
        missing_private_key_message: str,
        invalid_private_key_message: str,
    ) -> dict[str, Any]:
        raise AssertionError("sync public JWK path was called")

    async def public_jwk_from_file_async(
        self,
        *,
        kid: str,
        private_key_file: str | None,
        missing_private_key_message: str,
        invalid_private_key_message: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "kid": kid,
                "private_key_file": private_key_file,
                "missing_private_key_message": missing_private_key_message,
                "invalid_private_key_message": invalid_private_key_message,
            }
        )
        return {"kid": kid, "use": "sig", "alg": "RS256"}


def test_auth_jwks_response_uses_async_workload_public_jwk_path(exchange_config: AuthConfig) -> None:
    signing_key_cache = _AsyncOnlySigningKeyCache()
    exchange_service = exchange.WorkloadTokenExchangeService(signing_key_cache=signing_key_cache)

    response = asyncio.run(exchange.auth_jwks_response(exchange_config, exchange_service))

    assert [key.model_dump() for key in response.keys] == [{"kid": "nemo-helix-signing", "use": "sig", "alg": "RS256"}]
    assert signing_key_cache.calls == [
        {
            "kid": "nemo-helix-signing",
            "private_key_file": exchange_config.oidc.workload_token_private_key_file,
            "missing_private_key_message": "auth.token_signing.private_key_file must be configured for workload token exchange",
            "invalid_private_key_message": "workload token private key must be an RSA private key",
        }
    ]


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _signed_subject_token(
    config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    *,
    audience: str,
    issuer: str | None = None,
    private_key: rsa.RSAPrivateKey | None = None,
    key_id: str | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    token_issuer = issuer or config.oidc.workload_subject_issuers[0]
    signing_key = exchange_service.workload_signing_key(config)
    claims = {
        "iss": token_issuer,
        "sub": "authentik-user",
        "aud": audience,
        "exp": int(exchange.time.time()) + 300,
    }
    if extra_claims:
        claims.update(extra_claims)
    return exchange.jwt.encode(
        claims,
        private_key or signing_key.private_key,
        algorithm="RS256",
        headers={"kid": key_id or signing_key.kid},
    )


def _public_jwk_for_key(private_key: rsa.RSAPrivateKey, *, key_id: str) -> dict[str, Any]:
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk.update({"kid": key_id, "use": "sig", "alg": "RS256"})
    return jwk


def _mock_subject_jwks_client(
    config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {"request_count": 0}
    jwks = {"keys": [exchange_service.public_jwk(config)]}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            captured["request_count"] += 1
            captured["url"] = url
            return _FakeResponse(jwks)

    monkeypatch.setattr(exchange.httpx, "AsyncClient", FakeAsyncClient)
    return captured


def test_jwt_subject_token_decoder_fetches_configured_jwks(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _mock_subject_jwks_client(exchange_config, exchange_service, monkeypatch)
    subject_token = _signed_subject_token(exchange_config, exchange_service, audience="nemo-helix-workload")

    claims = asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))

    assert claims["sub"] == "authentik-user"
    assert captured == {
        "request_count": 1,
        "timeout": 30.0,
        "url": "https://idp.example.com/application/o/nemo-workload/jwks/",
    }


def test_jwt_subject_token_decoder_does_not_trust_private_bound_reference_claim_names(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_subject_jwks_client(exchange_config, exchange_service, monkeypatch)
    subject_token = _signed_subject_token(
        exchange_config,
        exchange_service,
        audience="nemo-helix-workload",
        extra_claims={
            "_nhx_bound_reference_name": exchange.KUBERNETES_POD_UID_REFERENCE_NAME,
            "_nhx_bound_reference_value": "pod-uid-123",
            "_nhx_bound_reference_trusted_source": "kubernetes",
        },
    )

    decoded = asyncio.run(exchange_service.decode_subject_token(exchange_config, subject_token, "nemo-helix"))

    assert decoded.claims["sub"] == "authentik-user"
    assert decoded.bound_reference is None


def test_jwt_subject_token_decoder_caches_configured_jwks(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _mock_subject_jwks_client(exchange_config, exchange_service, monkeypatch)
    subject_token = _signed_subject_token(exchange_config, exchange_service, audience="nemo-helix-workload")

    asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))
    asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))

    assert captured["request_count"] == 1


def test_jwt_subject_token_decoder_does_not_cache_invalid_jwks(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jwks_responses = [
        {},
        {"keys": [exchange_service.public_jwk(exchange_config)]},
    ]
    captured: dict[str, Any] = {"request_count": 0}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            captured["request_count"] += 1
            captured["url"] = url
            return _FakeResponse(jwks_responses.pop(0))

    monkeypatch.setattr(exchange.httpx, "AsyncClient", FakeAsyncClient)
    subject_token = _signed_subject_token(exchange_config, exchange_service, audience="nemo-helix-workload")

    with pytest.raises(exchange.jwt.InvalidTokenError, match="valid JWKS"):
        asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))

    claims = asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))

    assert claims["sub"] == "authentik-user"
    assert captured["request_count"] == 2


def test_jwt_subject_token_decoder_refreshes_cached_jwks_on_unknown_kid(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rotated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject_token = _signed_subject_token(
        exchange_config,
        exchange_service,
        audience="nemo-helix-workload",
        private_key=rotated_key,
        key_id="rotated-key",
    )
    jwks_responses = [
        {"keys": [exchange_service.public_jwk(exchange_config)]},
        {"keys": [_public_jwk_for_key(rotated_key, key_id="rotated-key")]},
    ]
    captured: dict[str, Any] = {"request_count": 0}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            captured["request_count"] += 1
            captured["url"] = url
            return _FakeResponse(jwks_responses.pop(0))

    monkeypatch.setattr(exchange.httpx, "AsyncClient", FakeAsyncClient)

    claims = asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))

    assert claims["sub"] == "authentik-user"
    assert captured["request_count"] == 2


def test_jwt_subject_token_decoder_rejects_wrong_audience(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_subject_jwks_client(exchange_config, exchange_service, monkeypatch)
    subject_token = _signed_subject_token(exchange_config, exchange_service, audience="some-other-client")

    with pytest.raises(exchange.jwt.InvalidAudienceError):
        asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))


def test_jwt_subject_token_decoder_rejects_unexpected_issuer(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_subject_jwks_client(exchange_config, exchange_service, monkeypatch)
    subject_token = _signed_subject_token(
        exchange_config,
        exchange_service,
        audience="nemo-helix-workload",
        issuer="https://idp.example.com/application/o/other/",
    )

    with pytest.raises(exchange.jwt.InvalidIssuerError):
        asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))


def test_jwt_subject_token_decoder_requires_explicit_workload_subject_issuers(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_subject_jwks_client(exchange_config, exchange_service, monkeypatch)
    exchange_config.oidc.workload_subject_issuers = []
    subject_token = _signed_subject_token(
        exchange_config,
        exchange_service,
        audience="nemo-helix-workload",
        issuer=exchange_config.oidc.additional_issuers[0],
    )

    with pytest.raises(exchange.jwt.InvalidIssuerError):
        asyncio.run(exchange_service.decode_jwt_subject_token(exchange_config, subject_token))


def test_subject_token_decoder_reports_all_validation_failures(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject_token = _signed_subject_token(exchange_config, exchange_service, audience="nemo-helix-workload")
    exchange_config.oidc.workload_subject_jwks_uri = None
    exchange_config.oidc.workload_kubernetes_token_review_enabled = True
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)

    with pytest.raises(exchange.jwt.InvalidTokenError) as exc_info:
        asyncio.run(exchange_service.decode_subject_token(exchange_config, subject_token, "nemo-helix"))

    assert str(exc_info.value) == (
        "JWT subject token: JWT subject token validation is disabled; "
        "Kubernetes TokenReview subject token: Kubernetes service environment is unavailable"
    )


def test_subject_token_decoder_does_not_mask_unexpected_decoder_errors(
    exchange_config: AuthConfig,
    exchange_service: exchange.WorkloadTokenExchangeService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject_jwt_subject_token(config: AuthConfig, subject_token: str) -> dict[str, Any]:
        raise exchange.jwt.InvalidTokenError("JWT rejected")

    async def broken_kubernetes_subject_token(
        config: AuthConfig,
        subject_token: str,
        audience: str,
    ) -> exchange.DecodedSubjectToken:
        raise RuntimeError("Kubernetes decoder broke")

    monkeypatch.setattr(exchange_service, "decode_jwt_subject_token", reject_jwt_subject_token)
    monkeypatch.setattr(exchange, "_decode_kubernetes_subject_token", broken_kubernetes_subject_token)

    with pytest.raises(RuntimeError, match="Kubernetes decoder broke"):
        asyncio.run(exchange_service.decode_subject_token(exchange_config, "subject-token", "nemo-helix"))


def test_kubernetes_subject_token_decoder_posts_token_review_with_async_client(
    exchange_config: AuthConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    exchange_config.oidc.workload_kubernetes_token_review_enabled = True
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    monkeypatch.setattr(exchange, "_kubernetes_reviewer_credentials", lambda: ("reviewer-token", "/tmp/ca.crt"))

    class FakeAsyncClient:
        def __init__(self, *, timeout: float, verify: str) -> None:
            captured["timeout"] = timeout
            captured["verify"] = verify

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            captured["url"] = url
            captured["json"] = kwargs["json"]
            captured["headers"] = kwargs["headers"]
            return _FakeResponse(
                {
                    "status": {
                        "authenticated": True,
                        "user": {
                            "username": "system:serviceaccount:default:nemo",
                            "groups": ["system:serviceaccounts"],
                        },
                    }
                }
            )

    monkeypatch.setattr(exchange.httpx, "AsyncClient", FakeAsyncClient)

    decoded = asyncio.run(
        exchange._decode_kubernetes_subject_token(exchange_config, "kubernetes-subject-token", "nemo-helix")
    )

    assert decoded == exchange.DecodedSubjectToken(
        claims={
            "sub": "system:serviceaccount:default:nemo",
            "groups": ["system:serviceaccounts"],
        }
    )
    assert captured == {
        "timeout": 10.0,
        "verify": "/tmp/ca.crt",
        "url": "https://kubernetes.default.svc:443/apis/authentication.k8s.io/v1/tokenreviews",
        "json": {
            "apiVersion": "authentication.k8s.io/v1",
            "kind": "TokenReview",
            "spec": {
                "token": "kubernetes-subject-token",
                "audiences": ["nemo-helix"],
            },
        },
        "headers": {
            "Authorization": "Bearer reviewer-token",
            "Content-Type": "application/json",
        },
    }


def test_kubernetes_subject_token_decoder_preserves_single_pod_uid_reference(
    exchange_config: AuthConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange_config.oidc.workload_kubernetes_token_review_enabled = True
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    monkeypatch.setattr(exchange, "_kubernetes_reviewer_credentials", lambda: ("reviewer-token", "/tmp/ca.crt"))

    class FakeAsyncClient:
        def __init__(self, *, timeout: float, verify: str) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse(
                {
                    "status": {
                        "authenticated": True,
                        "user": {
                            "username": "system:serviceaccount:nemo-runs:job-runner",
                            "groups": ["system:serviceaccounts"],
                            "extra": {
                                exchange.KUBERNETES_POD_UID_REFERENCE_NAME: ["pod-uid-123"],
                            },
                        },
                    }
                }
            )

    monkeypatch.setattr(exchange.httpx, "AsyncClient", FakeAsyncClient)

    decoded = asyncio.run(
        exchange._decode_kubernetes_subject_token(exchange_config, "kubernetes-subject-token", "nemo-helix")
    )

    assert decoded.claims == {
        "sub": "system:serviceaccount:nemo-runs:job-runner",
        "groups": ["system:serviceaccounts"],
    }
    assert decoded.bound_reference == _pod_uid_bound_reference()


@pytest.mark.parametrize(
    "pod_uids",
    [
        [],
        ["pod-uid-1", "pod-uid-2"],
    ],
)
def test_kubernetes_subject_token_decoder_rejects_ambiguous_pod_uid_reference(
    exchange_config: AuthConfig,
    monkeypatch: pytest.MonkeyPatch,
    pod_uids: list[str],
) -> None:
    exchange_config.oidc.workload_kubernetes_token_review_enabled = True
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    monkeypatch.setattr(exchange, "_kubernetes_reviewer_credentials", lambda: ("reviewer-token", "/tmp/ca.crt"))

    class FakeAsyncClient:
        def __init__(self, *, timeout: float, verify: str) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse(
                {
                    "status": {
                        "authenticated": True,
                        "user": {
                            "username": "system:serviceaccount:nemo-runs:job-runner",
                            "extra": {
                                exchange.KUBERNETES_POD_UID_REFERENCE_NAME: pod_uids,
                            },
                        },
                    }
                }
            )

    monkeypatch.setattr(exchange.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(exchange._InvalidGrantError):
        asyncio.run(
            exchange._decode_kubernetes_subject_token(exchange_config, "kubernetes-subject-token", "nemo-helix")
        )
