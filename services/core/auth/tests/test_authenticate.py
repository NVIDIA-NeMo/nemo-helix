# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, patch

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.auth.jwt import JWTValidator
from nhx.common.auth.token_claims import ActorClaims, TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken
from nhx.common.config import AuthConfig
from nhx.common.config.base import AccessKeyConfig, OIDCConfig, TokenSigningConfig
from nhx.core.auth.api.v2.authenticate import router
from nhx.core.auth.api.v2.workload_token_exchange import (
    WorkloadTokenExchangeService,
    get_workload_token_exchange_service,
)
from nhx.core.auth.app.access_keys import get_access_key_registry


def _private_key_pem() -> bytes:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _assert_no_principal_response_headers(response) -> None:
    for header_name in (
        "X-NHX-Principal-Id",
        "X-NHX-Principal-Email",
        "X-NHX-Principal-Groups",
        "X-NHX-Principal-On-Behalf-Of",
        "X-NHX-Principal-On-Behalf-Of-Email",
        "X-NHX-Principal-On-Behalf-Of-Groups",
        "X-NHX-Scopes",
    ):
        assert header_name not in response.headers


class AlwaysActiveAccessKeyRegistry:
    async def is_active(self, jti: str, principal: str, **kwargs) -> bool:
        return True


class RevokedAccessKeyRegistry:
    async def is_active(self, jti: str, principal: str, **kwargs) -> bool:
        return False


class ClaimAwareAccessKeyRegistry:
    def __init__(self) -> None:
        self.claims = None

    async def is_active(self, jti: str, principal: str, **kwargs) -> bool:
        self.claims = kwargs.get("claims")
        return True


@contextmanager
def _test_client(
    config: AuthConfig,
    *,
    workload_token_exchange_service: WorkloadTokenExchangeService | None = None,
) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_access_key_registry] = lambda: AlwaysActiveAccessKeyRegistry()
    if workload_token_exchange_service is not None:
        app.dependency_overrides[get_workload_token_exchange_service] = lambda: workload_token_exchange_service
    with patch("nhx.core.auth.api.v2.authenticate.get_auth_config", return_value=config):
        yield TestClient(app)


def _auth_config_with_workload_exchange(
    tmp_path,
    *,
    oidc_issuer: str = "https://sso.example.com",
    additional_issuers: list[str] | None = None,
    oidc_jwks_uri: str | None = "https://sso.example.com/jwks",
    token_issuer: str | None = "http://testserver/apis/auth",
) -> AuthConfig:
    private_key_file = tmp_path / "private.pem"
    private_key_file.write_bytes(_private_key_pem())
    return AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer=token_issuer,
            key_id="test-workload",
            private_key_file=str(private_key_file),
        ),
        oidc=OIDCConfig(
            enabled=True,
            issuer=oidc_issuer,
            additional_issuers=additional_issuers or [],
            jwks_uri=oidc_jwks_uri,
            workload_token_exchange_enabled=True,
            workload_audience="nemo-helix",
        ),
    )


def test_authenticate_access_key_returns_principal_json(tmp_path):
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="test-access-key",
            private_key_file=str(tmp_path / "private.pem"),
        ),
        access_keys=AccessKeyConfig(enabled=True),
    )
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    claims = TokenClaims(
        subject="alice@example.com",
        email="alice@example.com",
        groups=["team-ml"],
        scopes=["models:read"],
        raw_claims={"jti": "ak_example", "nhx_token_type": "access_key"},
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind="access_key")
    with (
        _test_client(config) as client,
        patch(
            "nhx.core.auth.api.v2.authenticate.resolve_bearer_token",
            new=AsyncMock(return_value=resolved),
        ) as resolver,
    ):
        response = client.post(
            "/authenticate",
            headers={"Authorization": "Bearer signed.jwt.token"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "principal": "alice@example.com",
        "email": "alice@example.com",
        "groups": ["team-ml"],
        "scopes": ["models:read"],
        "jti": "ak_example",
        "token_kind": "access_key",
        "on_behalf_of": None,
        "on_behalf_of_email": None,
        "on_behalf_of_groups": [],
    }
    _assert_no_principal_response_headers(response)
    resolver_call = resolver.await_args
    assert resolver_call is not None
    assert resolver_call.args[:2] == (config, "signed.jwt.token")
    assert len(resolver_call.kwargs["extra_resolvers"]) == 2


def test_authenticate_passes_access_key_claims_for_legacy_record_backfill(tmp_path):
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")),
        access_keys=AccessKeyConfig(enabled=True),
    )
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    claims = TokenClaims(
        subject="alice@example.com",
        email=None,
        groups=[],
        scopes=[],
        raw_claims={
            "iss": "http://testserver/apis/auth",
            "aud": ["nemo-helix-access-key"],
            "sub": "alice@example.com",
            "iat": 1_785_280_000,
            "nbf": 1_785_280_000,
            "jti": "ak_legacy",
            "nhx_token_type": "access_key",
            "nhx_access_key": {"version": 1, "name": "legacy"},
        },
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind="access_key")
    registry = ClaimAwareAccessKeyRegistry()
    with (
        _test_client(config) as client,
        patch("nhx.core.auth.api.v2.authenticate.resolve_bearer_token", new=AsyncMock(return_value=resolved)),
    ):
        client.app.dependency_overrides[get_access_key_registry] = lambda: registry
        response = client.get("/authenticate", headers={"Authorization": "Bearer signed.jwt.token"})

    assert response.status_code == 200
    assert registry.claims is claims


def test_authenticate_rejects_revoked_access_key(tmp_path):
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")),
        access_keys=AccessKeyConfig(enabled=True),
    )
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    claims = TokenClaims(
        subject="alice@example.com",
        email=None,
        groups=[],
        scopes=[],
        raw_claims={"jti": "ak_revoked", "nhx_token_type": "access_key"},
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind="access_key")
    with (
        _test_client(config) as client,
        patch("nhx.core.auth.api.v2.authenticate.resolve_bearer_token", new=AsyncMock(return_value=resolved)),
    ):
        client.app.dependency_overrides[get_access_key_registry] = lambda: RevokedAccessKeyRegistry()
        response = client.get("/authenticate", headers={"Authorization": "Bearer signed.jwt.token"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid bearer token"


def test_ext_authz_accepts_original_request_methods_and_returns_principal_headers(tmp_path):
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")),
        access_keys=AccessKeyConfig(enabled=True),
    )
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    claims = TokenClaims(
        subject="writer@example.com",
        email=None,
        groups=[],
        scopes=["models:write"],
        raw_claims={"jti": "ak_writer", "nhx_token_type": "access_key"},
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind="access_key")
    with (
        _test_client(config) as client,
        patch(
            "nhx.core.auth.api.v2.authenticate.resolve_bearer_token",
            new=AsyncMock(return_value=resolved),
        ) as resolver,
    ):
        response = client.delete(
            "/ext-authz/apis/entities/v2/workspaces/default",
            headers={"Authorization": "Bearer signed.jwt.token"},
        )

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["X-NHX-Principal-Id"] == "writer@example.com"
    assert response.headers["X-NHX-Scopes"] == "models:write"
    resolver.assert_awaited_once()


def test_authenticate_oidc_access_token_with_actor_returns_direct_principal_json(tmp_path):
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")),
        oidc=OIDCConfig(enabled=True, issuer="https://sso.example.com", client_id="nemo-helix-cli"),
    )
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    claims = TokenClaims(
        subject="user:alice",
        email="alice@example.test",
        groups=["researchers"],
        scopes=["models.read"],
        raw_claims={
            "sub": "user:alice",
            "email": "alice@example.test",
            "groups": "researchers",
            "act": {"sub": "service:jobs", "groups": "system:serviceaccounts"},
        },
        actor=ActorClaims(subject="service:jobs", groups=["system:serviceaccounts"]),
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind="oidc_access_token")

    with (
        _test_client(config) as client,
        patch(
            "nhx.core.auth.api.v2.authenticate.resolve_bearer_token",
            new=AsyncMock(return_value=resolved),
        ),
    ):
        response = client.get("/authenticate", headers={"Authorization": "Bearer oidc.token"})

    assert response.status_code == 200
    assert response.json() == {
        "principal": "user:alice",
        "email": "alice@example.test",
        "groups": ["researchers"],
        "scopes": ["models.read"],
        "jti": None,
        "token_kind": "oidc_access_token",
        "on_behalf_of": None,
        "on_behalf_of_email": None,
        "on_behalf_of_groups": [],
    }
    _assert_no_principal_response_headers(response)


def test_authenticate_rejects_unresolved_bearer_token(tmp_path):
    config = AuthConfig(enabled=True, token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")))
    (tmp_path / "private.pem").write_bytes(_private_key_pem())

    with (
        _test_client(config) as client,
        patch("nhx.core.auth.api.v2.authenticate.resolve_bearer_token", new=AsyncMock(return_value=None)),
    ):
        response = client.get("/authenticate", headers={"Authorization": "Bearer invalid.token"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid bearer token"


def test_authenticate_workload_access_token_returns_principal_json(tmp_path):
    private_key_file = tmp_path / "private.pem"
    private_key_file.write_bytes(_private_key_pem())
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="test-workload",
            private_key_file=str(private_key_file),
        ),
        oidc=OIDCConfig(
            workload_token_exchange_enabled=True,
            workload_audience="nemo-helix",
        ),
    )
    signing_key = WorkloadTokenExchangeService().workload_signing_key(config)
    now = datetime.now(tz=UTC)
    token = jwt.encode(
        {
            "iss": "http://testserver/apis/auth",
            "sub": "system:serviceaccount:nemo:job",
            "aud": "nemo-helix",
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(minutes=5),
            "scope": "openid email groups",
            "groups": "team-ml,team-ai",
        },
        signing_key.private_key,
        algorithm="RS256",
        headers={"kid": signing_key.kid},
    )
    with _test_client(config) as client:
        response = client.get(
            "/authenticate",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "principal": "system:serviceaccount:nemo:job",
        "email": None,
        "groups": ["team-ml", "team-ai"],
        "scopes": ["openid", "email", "groups"],
        "jti": None,
        "token_kind": "workload_access_token",
        "on_behalf_of": None,
        "on_behalf_of_email": None,
        "on_behalf_of_groups": [],
    }
    _assert_no_principal_response_headers(response)


def test_authenticate_delegated_workload_access_token_returns_resolved_principal_json(tmp_path):
    private_key_file = tmp_path / "private.pem"
    private_key_file.write_bytes(_private_key_pem())
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="test-workload",
            private_key_file=str(private_key_file),
        ),
        oidc=OIDCConfig(
            workload_token_exchange_enabled=True,
            workload_audience="nemo-helix",
        ),
    )
    signing_key = WorkloadTokenExchangeService().workload_signing_key(config)
    now = datetime.now(tz=UTC)
    token = jwt.encode(
        {
            "iss": "http://testserver/apis/auth",
            "sub": "submitter@example.com",
            "email": "submitter@example.com",
            "aud": "nemo-helix",
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(minutes=5),
            "scope": "openid email groups",
            "groups": "workspace-editors",
            "act": {
                "sub": "system:serviceaccount:nemo:job",
                "groups": ["system:serviceaccounts", "nemo-jobs"],
            },
        },
        signing_key.private_key,
        algorithm="RS256",
        headers={"kid": signing_key.kid},
    )
    with _test_client(config) as client:
        response = client.get(
            "/authenticate",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["token_kind"] == "workload_access_token"
    assert response.json() == {
        "principal": "system:serviceaccount:nemo:job",
        "email": None,
        "groups": ["system:serviceaccounts", "nemo-jobs"],
        "scopes": ["openid", "email", "groups"],
        "jti": None,
        "token_kind": "workload_access_token",
        "on_behalf_of": "submitter@example.com",
        "on_behalf_of_email": "submitter@example.com",
        "on_behalf_of_groups": ["workspace-editors"],
    }
    _assert_no_principal_response_headers(response)


def test_ext_authz_delegated_workload_access_token_returns_obo_principal_headers(tmp_path):
    private_key_file = tmp_path / "private.pem"
    private_key_file.write_bytes(_private_key_pem())
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="test-workload",
            private_key_file=str(private_key_file),
        ),
        oidc=OIDCConfig(
            workload_token_exchange_enabled=True,
            workload_audience="nemo-helix",
        ),
    )
    signing_key = WorkloadTokenExchangeService().workload_signing_key(config)
    now = datetime.now(tz=UTC)
    token = jwt.encode(
        {
            "iss": "http://testserver/apis/auth",
            "sub": "submitter@example.com",
            "email": "submitter@example.com",
            "aud": "nemo-helix",
            "iat": now,
            "nbf": now,
            "exp": now + timedelta(minutes=5),
            "scope": "openid email groups",
            "groups": "workspace-editors",
            "act": {
                "sub": "system:serviceaccount:nemo:job",
                "groups": ["system:serviceaccounts", "nemo-jobs"],
            },
        },
        signing_key.private_key,
        algorithm="RS256",
        headers={"kid": signing_key.kid},
    )
    with _test_client(config) as client:
        response = client.get(
            "/ext-authz/apis/entities/v2/workspaces/default",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["X-NHX-Principal-Id"] == "system:serviceaccount:nemo:job"
    assert response.headers["X-NHX-Principal-Groups"] == "system:serviceaccounts,nemo-jobs"
    assert response.headers["X-NHX-Principal-On-Behalf-Of"] == "submitter@example.com"
    assert response.headers["X-NHX-Principal-On-Behalf-Of-Email"] == "submitter@example.com"
    assert response.headers["X-NHX-Principal-On-Behalf-Of-Groups"] == "workspace-editors"
    assert response.headers["X-NHX-Scopes"] == "openid email groups"


def test_authenticate_workload_subject_token_uses_resolver_callback(tmp_path):
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")),
        oidc=OIDCConfig(
            issuer="https://sso.example.com/application/o/nemo-cli/",
            client_id="nemo-helix-cli",
            workload_token_exchange_enabled=True,
            workload_client_id="nemo-helix-workload",
            workload_subject_jwks_uri="https://sso.example.com/application/o/nemo-workload/jwks/",
            workload_subject_issuers=["https://sso.example.com/application/o/nemo-workload/"],
        ),
    )
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    subject_claims = {
        "sub": "svc-nemo",
        "email": "svc-nemo@example.com",
        "groups": "nemo-workloads",
        "scope": "openid email groups",
    }
    exchange_service = WorkloadTokenExchangeService()

    async def resolve_via_subject_callback(config_arg, token_arg, **kwargs):
        assert config_arg == config
        assert token_arg == "workload.subject.token"
        extra_resolvers = kwargs["extra_resolvers"]
        assert len(extra_resolvers) == 2
        return await extra_resolvers[1](token_arg)

    with (
        _test_client(config, workload_token_exchange_service=exchange_service) as client,
        patch(
            "nhx.core.auth.api.v2.authenticate.resolve_bearer_token",
            new=AsyncMock(side_effect=resolve_via_subject_callback),
        ),
        patch.object(exchange_service, "decode_jwt_subject_token", new=AsyncMock(return_value=subject_claims)),
    ):
        response = client.get("/authenticate", headers={"Authorization": "Bearer workload.subject.token"})

    assert response.status_code == 200
    assert response.json()["principal"] == "svc-nemo"
    assert response.json()["token_kind"] == "workload_subject_token"
    assert response.json()["on_behalf_of"] is None
    assert response.json()["on_behalf_of_email"] is None
    assert response.json()["on_behalf_of_groups"] == []
    _assert_no_principal_response_headers(response)


def test_authenticate_invalid_workload_access_token_returns_401(tmp_path):
    private_key_file = tmp_path / "private.pem"
    private_key_file.write_bytes(_private_key_pem())
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="test-workload",
            private_key_file=str(private_key_file),
        ),
        oidc=OIDCConfig(
            workload_token_exchange_enabled=True,
            workload_audience="nemo-helix",
        ),
    )
    with _test_client(config) as client:
        response = client.get(
            "/authenticate",
            headers={"Authorization": "Bearer not-a-jwt"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid bearer token"


def test_authenticate_workload_access_token_surfaces_signing_key_misconfiguration(caplog):
    config = AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            workload_token_exchange_enabled=True,
            workload_audience="nemo-helix",
        ),
    )
    with _test_client(config) as client, caplog.at_level(logging.ERROR, logger="nhx.core.auth.api.v2.authenticate"):
        response = client.get(
            "/authenticate",
            headers={"Authorization": "Bearer signed.jwt.token"},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Workload token authentication is misconfigured"
    assert "Failed to load workload access token signing key" in caplog.text


def test_authenticate_fails_closed_when_oidc_issuer_matches_workload_token_issuer(tmp_path):
    config = _auth_config_with_workload_exchange(
        tmp_path,
        oidc_issuer="http://testserver/apis/auth",
        token_issuer="http://testserver/apis/auth",
    )

    with _test_client(config) as client:
        response = client.get("/authenticate", headers={"Authorization": "Bearer token"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Authentication token issuers are misconfigured"


def test_authenticate_fails_closed_when_additional_oidc_issuer_matches_workload_token_issuer(tmp_path):
    config = _auth_config_with_workload_exchange(
        tmp_path,
        additional_issuers=["http://testserver/apis/auth"],
        token_issuer="http://testserver/apis/auth",
    )

    with _test_client(config) as client:
        response = client.get("/authenticate", headers={"Authorization": "Bearer token"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Authentication token issuers are misconfigured"


def test_authenticate_fails_closed_when_oidc_jwks_uri_matches_workload_jwks_uri(tmp_path):
    config = _auth_config_with_workload_exchange(
        tmp_path,
        oidc_jwks_uri="http://testserver/apis/auth/jwks",
    )

    with _test_client(config) as client:
        response = client.get("/authenticate", headers={"Authorization": "Bearer token"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Authentication token issuers are misconfigured"


def test_authenticate_fails_closed_when_discovered_oidc_jwks_uri_matches_workload_jwks_uri(tmp_path):
    config = _auth_config_with_workload_exchange(
        tmp_path,
        oidc_jwks_uri=None,
    )

    with (
        _test_client(config) as client,
        patch.object(
            JWTValidator,
            "_discover_oidc_config",
            new=AsyncMock(return_value={"jwks_uri": "http://testserver/apis/auth/jwks"}),
        ),
    ):
        response = client.get("/authenticate", headers={"Authorization": "Bearer token"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Authentication token issuers are misconfigured"


def test_authenticate_fails_closed_when_oidc_jwks_contains_workload_public_key_material(tmp_path):
    config = _auth_config_with_workload_exchange(tmp_path)
    service = WorkloadTokenExchangeService()
    mirrored_jwk = {**service.public_jwk(config), "kid": "idp-key"}

    with (
        _test_client(config, workload_token_exchange_service=service) as client,
        patch.object(JWTValidator, "jwks", new=AsyncMock(return_value={"keys": [mirrored_jwk]})),
    ):
        response = client.get("/authenticate", headers={"Authorization": "Bearer token"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Authentication token issuers are misconfigured"


def test_authenticate_allows_same_kid_when_oidc_jwks_key_material_differs(tmp_path):
    config = _auth_config_with_workload_exchange(tmp_path)
    other_private_key_file = tmp_path / "other-private.pem"
    other_private_key_file.write_bytes(_private_key_pem())
    other_config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="https://other.example.test",
            key_id="test-workload",
            private_key_file=str(other_private_key_file),
        ),
    )
    service = WorkloadTokenExchangeService()
    idp_jwk = service.public_jwk(other_config)
    claims = TokenClaims(
        subject="user:alice",
        email="alice@example.test",
        groups=["researchers"],
        scopes=[],
        raw_claims={},
    )
    resolved = ResolvedBearerToken(claims=claims, token_kind="oidc_access_token")

    with (
        _test_client(config, workload_token_exchange_service=service) as client,
        patch.object(JWTValidator, "jwks", new=AsyncMock(return_value={"keys": [idp_jwk]})),
        patch(
            "nhx.core.auth.api.v2.authenticate.resolve_bearer_token",
            new=AsyncMock(return_value=resolved),
        ),
    ):
        response = client.get("/authenticate", headers={"Authorization": "Bearer token"})

    assert response.status_code == 200
    assert response.json()["principal"] == "user:alice"
    assert response.json()["on_behalf_of"] is None
    _assert_no_principal_response_headers(response)


def test_authenticate_openapi_hides_ext_authz_and_documents_obo_fields(tmp_path):
    config = AuthConfig(enabled=True, token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")))
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    with _test_client(config) as client:
        openapi = cast(FastAPI, client.app).openapi()

    assert "/ext-authz" not in openapi["paths"]
    assert "/ext-authz/{original_path}" not in openapi["paths"]
    for method in ("get", "post"):
        responses = openapi["paths"]["/authenticate"][method]["responses"]
        assert responses["200"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AuthenticateResponse"
        }
        assert responses["401"]["description"] == "Missing or invalid bearer token"
        assert responses["401"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AuthenticateErrorResponse"
        }
        assert responses["500"]["description"] == "Bearer token authentication is misconfigured"
        assert responses["500"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AuthenticateErrorResponse"
        }

    authenticate_schema = openapi["components"]["schemas"]["AuthenticateResponse"]
    assert authenticate_schema["properties"]["email"]["nullable"] is True
    assert authenticate_schema["properties"]["jti"]["nullable"] is True
    assert authenticate_schema["properties"]["on_behalf_of"]["nullable"] is True
    assert authenticate_schema["properties"]["on_behalf_of_email"]["nullable"] is True
    assert authenticate_schema["properties"]["on_behalf_of_groups"]["items"]["type"] == "string"
    assert authenticate_schema["properties"]["token_kind"]["enum"] == [
        "access_key",
        "oidc_access_token",
        "workload_access_token",
        "workload_subject_token",
    ]


def test_authenticate_rejects_missing_bearer_token(tmp_path):
    config = AuthConfig(enabled=True, token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")))
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    with _test_client(config) as client:
        response = client.post("/authenticate")

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing bearer token"


def test_authenticate_rejects_malformed_bearer_token(tmp_path):
    config = AuthConfig(enabled=True, token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")))
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    with _test_client(config) as client:
        response = client.post("/authenticate", headers={"Authorization": "Bearer token extra"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid bearer token"


def test_authenticate_prefixed_callout_route_is_removed(tmp_path):
    config = AuthConfig(enabled=True, token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")))
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    with _test_client(config) as client:
        response = client.get(
            "/authenticate/apis/entities/v2/workspaces/default",
            headers={"Authorization": "Bearer token"},
        )

    assert response.status_code == 404


def test_authenticate_non_json_callout_methods_are_removed(tmp_path):
    config = AuthConfig(enabled=True, token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")))
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    with _test_client(config) as client:
        response = client.put("/authenticate", headers={"Authorization": "Bearer token"})

    assert response.status_code == 405


def test_access_key_specific_authenticate_route_is_removed(tmp_path):
    config = AuthConfig(enabled=True, token_signing=TokenSigningConfig(private_key_file=str(tmp_path / "private.pem")))
    (tmp_path / "private.pem").write_bytes(_private_key_pem())
    with _test_client(config) as client:
        response = client.post("/v2/access-keys/authenticate")

    assert response.status_code == 404
