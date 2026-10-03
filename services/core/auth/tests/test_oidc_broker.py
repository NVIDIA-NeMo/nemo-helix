# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nhx.common.auth.token_claims import TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken
from nhx.common.config import Configuration
from nhx.common.config.base import AuthConfig, OIDCConfig
from nhx.core.auth.oidc_broker.crypto import client_secret_basic_header, pkce_challenge
from nhx.core.auth.oidc_broker.routes import get_credential_store, get_identity_store, get_login_store
from nhx.core.auth.oidc_broker.store import DBOIDCLoginTransaction, LoginRecord, OidcLoginStore
from nhx.core.entities.app.repository import AccountCredentialRecord
from nhx.core.entities.app.repository.account_identity import AccountIdentityRecord
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


class MemoryLoginStore:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, str]]] = {}

    async def put(self, kind: str, record_id: str, payload: dict[str, str], ttl_seconds: int) -> None:
        self.rows[record_id] = (kind, dict(payload))

    async def get(self, kind: str, record_id: str) -> LoginRecord | None:
        row = self.rows.get(record_id)
        if row is None or row[0] != kind:
            return None
        return LoginRecord(kind=kind, payload=row[1], expires_at=datetime.now(timezone.utc))

    async def pop(self, kind: str, record_id: str) -> LoginRecord | None:
        record = await self.get(kind, record_id)
        if record is not None:
            del self.rows[record_id]
        return record


class MemoryCredentialStore:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], AccountCredentialRecord] = {}

    async def create(self, **kwargs) -> AccountCredentialRecord:
        handle = kwargs.pop("handle")
        credential_type = kwargs.pop("credential_type")
        subject_account_id = kwargs.pop("subject_account_id", None) or kwargs["owner_account_id"]
        record = AccountCredentialRecord(
            id=f"credential-{len(self.rows) + 1}",
            credential_type=credential_type,
            lookup_hash=f"hash:{handle}",
            subject_account_id=subject_account_id,
            status=kwargs.pop("status", "ACTIVE"),
            issued_at=kwargs.pop("issued_at", None) or datetime.now(timezone.utc),
            last_used_at=None,
            revoked_at=None,
            revoked_by=None,
            db_version=1,
            **kwargs,
        )
        self.rows[(credential_type, handle)] = record
        return record

    async def get_active(self, credential_type: str, handle: str) -> AccountCredentialRecord | None:
        record = self.rows.get((credential_type, handle))
        if record is None or record.status != "ACTIVE":
            return None
        return record

    async def consume_active(self, credential_type: str, handle: str) -> AccountCredentialRecord | None:
        record = await self.get_active(credential_type, handle)
        if record is not None:
            del self.rows[(credential_type, handle)]
        return record

    async def update(self, record: AccountCredentialRecord) -> AccountCredentialRecord:
        for key, current in self.rows.items():
            if current.id == record.id:
                updated = replace(record, db_version=record.db_version + 1)
                self.rows[key] = updated
                return updated
        raise AssertionError("credential not found")

    async def revoke(self, credential_type: str, handle: str) -> bool:
        record = self.rows.get((credential_type, handle))
        if record is None or record.status == "REVOKED":
            return False
        self.rows[(credential_type, handle)] = replace(record, status="REVOKED")
        return True


class MemoryIdentityStore:
    async def resolve_or_materialize(self, **kwargs) -> AccountIdentityRecord:
        return AccountIdentityRecord(
            account_id="account-1",
            account_type="user",
            identity_id="identity-1",
            issuer=kwargs["issuer"],
            subject=kwargs["subject"],
        )


def test_client_secret_basic_form_encodes_then_base64() -> None:
    import base64

    header = client_secret_basic_header("client id", "secret/value")
    encoded = header.removeprefix("Basic ")
    decoded = base64.b64decode(encoded).decode()
    assert decoded == "client+id:secret%2Fvalue"


def test_pkce_s256_is_stable() -> None:
    challenge = pkce_challenge("verifier")
    assert challenge
    assert "=" not in challenge
    assert challenge == pkce_challenge("verifier")


@pytest.mark.asyncio
async def test_login_store_pop_is_single_use_and_kind_scoped() -> None:
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(DBOIDCLoginTransaction.__table__.create)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    store = OidcLoginStore(session_maker, "encryption-key")
    try:
        await store.put("web_txn", "state", {"verifier": "secret"}, ttl_seconds=60)

        assert await store.pop("cli_txn", "state") is None
        record = await store.pop("web_txn", "state")
        assert record is not None
        assert record.payload == {"verifier": "secret"}
        assert await store.pop("web_txn", "state") is None
    finally:
        await engine.dispose()


@pytest.fixture
def broker_app(monkeypatch: pytest.MonkeyPatch):
    transactions = MemoryLoginStore()
    credentials = MemoryCredentialStore()
    identities = MemoryIdentityStore()
    config = AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.example.com",
            client_id="nemo-helix-user",
            token_endpoint_auth_method="client_secret_basic",
            client_secret_env_var="NHX_OIDC_CLIENT_SECRET",
            session_encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY",
            authorization_endpoint="https://idp.example.com/authorize",
            token_endpoint="https://idp.example.com/token",
            public_client_id="nemo-helix-cli",
        ),
    )
    Configuration.set_override(config)
    monkeypatch.setenv("NHX_OIDC_CLIENT_SECRET", "user-login-secret")
    monkeypatch.setenv("NHX_AUTH_SESSION_ENCRYPTION_KEY", "session-key")

    async def resolved_token(*args, **kwargs) -> ResolvedBearerToken:
        return ResolvedBearerToken(
            claims=TokenClaims(
                subject="user-1",
                email="user@example.com",
                groups=["nemo-users"],
                scopes=["openid"],
                raw_claims={"sub": "user-1", "email": "user@example.com", "name": "User One"},
            ),
            token_kind="oidc_access_token",
        )

    monkeypatch.setattr("nhx.core.auth.oidc_broker.routes.resolve_bearer_token", resolved_token)
    app = FastAPI()
    from nhx.core.auth.oidc_broker.routes import router

    app.dependency_overrides[get_login_store] = lambda: transactions
    app.dependency_overrides[get_credential_store] = lambda: credentials
    app.dependency_overrides[get_identity_store] = lambda: identities
    app.include_router(router, prefix="/apis/auth")
    yield TestClient(app, base_url="https://nemo.example.com"), transactions, credentials
    Configuration.clear_overrides()


def test_login_uses_basic_auth_and_sets_account_bound_cookie(broker_app) -> None:
    client, _transactions, credentials = broker_app
    with respx.mock:
        respx.post("https://idp.example.com/token").mock(
            return_value=httpx.Response(
                200,
                json={
                    "access_token": "provider-access",
                    "refresh_token": "provider-refresh",
                    "expires_in": 900,
                },
            )
        )
        start = client.get("/apis/auth/v2/login", follow_redirects=False)
        assert start.status_code == 302
        query = parse_qs(urlparse(start.headers["location"]).query)
        assert query["client_id"] == ["nemo-helix-user"]
        assert query["code_challenge_method"] == ["S256"]
        assert "client_secret" not in query
        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": query["state"][0]},
            follow_redirects=False,
        )
        assert callback.status_code == 302
        session_handle = callback.cookies["nhx_session"]
        session_record = credentials.rows[("web_session", session_handle)]
        assert session_record.owner_account_id == "account-1"
        assert session_record.account_identity_id == "identity-1"
        assert "provider-refresh" not in str(session_record.public_metadata)
        assert "provider-refresh" not in session_record.encrypted_payload

        session = client.get("/apis/auth/v2/session")
        assert session.json() == {
            "id": "user-1",
            "email": "user@example.com",
            "groups": ["nemo-users"],
            "account_id": "account-1",
            "authz_aliases": ["user-1", "user@example.com"],
        }
        assert "provider-access" not in session.text
        credentials.rows[("web_session", session_handle)] = replace(
            session_record,
            public_metadata={**session_record.public_metadata, "principal_id": None},
        )
        assert client.get("/apis/auth/v2/session").status_code == 401
        token_call = respx.calls.last
        assert token_call.request.headers["authorization"].startswith("Basic ")


def test_invalid_provider_token_response_is_sanitized(broker_app) -> None:
    client, _transactions, _credentials = broker_app
    with respx.mock:
        respx.post("https://idp.example.com/token").mock(return_value=httpx.Response(200, content=b"not-json"))
        start = client.get("/apis/auth/v2/login", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]

        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )

    assert callback.status_code == 502
    assert callback.json() == {"detail": "Identity provider token response was invalid"}


def test_provider_token_transport_failure_is_sanitized(broker_app) -> None:
    client, _transactions, _credentials = broker_app
    with respx.mock:
        respx.post("https://idp.example.com/token").mock(
            side_effect=httpx.ConnectError("provider details must not reach the client")
        )
        start = client.get("/apis/auth/v2/login", follow_redirects=False)
        state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]

        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )

    assert callback.status_code == 502
    assert callback.json() == {"detail": "Identity provider token request failed"}
    assert "provider details" not in callback.text


@pytest.mark.parametrize("return_to", ["https://evil.example.com", "//evil.example.com", r"/\evil.example.com"])
def test_web_login_rejects_non_relative_return_target(broker_app, return_to: str) -> None:
    client, _transactions, _credentials = broker_app

    response = client.get("/apis/auth/v2/login", params={"return_to": return_to})

    assert response.status_code == 400


@pytest.mark.parametrize(
    "redirect_uri",
    [
        "https://127.0.0.1:54321/callback",
        "http://example.com:54321/callback",
        "http://127.0.0.1/callback",
        "http://user@127.0.0.1:54321/callback",
        "http://127.0.0.1:54321/callback#fragment",
    ],
)
def test_cli_login_rejects_invalid_loopback_redirect(broker_app, redirect_uri: str) -> None:
    client, _transactions, _credentials = broker_app

    response = client.post("/apis/auth/v2/cli/login", json={"redirect_uri": redirect_uri})

    assert response.status_code == 400


def test_cli_receives_opaque_refresh_handle_and_refresh_uses_provider_token(broker_app) -> None:
    client, transactions, credentials = broker_app
    with respx.mock:
        token_route = respx.post("https://idp.example.com/token").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "access_token": "provider-access-1",
                        "refresh_token": "provider-refresh-1",
                        "expires_in": 900,
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "access_token": "provider-access-2",
                        "refresh_token": "provider-refresh-2",
                        "expires_in": 900,
                    },
                ),
            ]
        )
        start = client.post("/apis/auth/v2/cli/login", json={"redirect_uri": "http://127.0.0.1:54321/callback"})
        transaction_id = start.json()["transaction_id"]
        assert transaction_id in transactions.rows
        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": transaction_id},
            follow_redirects=False,
        )
        one_time_code = parse_qs(urlparse(callback.headers["location"]).query)["code"][0]
        exchanged = client.post(
            "/apis/auth/v2/cli/token",
            json={"grant_type": "authorization_code", "code": one_time_code},
        )
        assert exchanged.status_code == 200
        refresh_handle = exchanged.json()["refresh_token"]
        assert refresh_handle != "provider-refresh-1"
        assert ("cli_login_code", one_time_code) not in credentials.rows
        assert ("cli_refresh", refresh_handle) in credentials.rows

        refreshed = client.post(
            "/apis/auth/v2/cli/token",
            json={"grant_type": "refresh_token", "refresh_token": refresh_handle},
        )
        assert refreshed.json()["access_token"] == "provider-access-2"
        assert refreshed.json()["refresh_token"] == refresh_handle
        assert token_route.calls[-1].request.content == b"grant_type=refresh_token&refresh_token=provider-refresh-1"


def test_discovery_advertises_only_configured_clients() -> None:
    from nhx.core.auth.api.v2.discovery.endpoints import get_auth_discovery

    config = AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.example.com",
            client_id="nemo-helix-user",
            token_endpoint_auth_method="client_secret_basic",
            client_secret_env_var="NHX_OIDC_CLIENT_SECRET",
            session_encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY",
            authorization_endpoint="https://idp.example.com/authorize",
            token_endpoint="https://idp.example.com/token",
            public_client_id="nemo-helix-cli",
        ),
    )
    Configuration.set_override(config)
    try:
        import asyncio

        result = asyncio.run(get_auth_discovery())
    finally:
        Configuration.clear_overrides()
    assert result.oidc is not None
    dumped = result.oidc.model_dump()
    assert dumped["client_id"] == "nemo-helix-user"
    assert [client["name"] for client in dumped["clients"]] == ["platform", "public"]
    assert dumped["clients"][0]["default"] is True
    assert "client_secret" not in dumped
    assert "NHX_OIDC_CLIENT_SECRET" not in str(dumped)
