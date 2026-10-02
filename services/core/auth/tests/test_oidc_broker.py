# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response as Httpx2Response
from nemo_helix_plugin.config import Runtime
from nhx.common.auth.token_claims import TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken
from nhx.common.config import Configuration, HelixConfig
from nhx.common.config.base import (
    AuthConfig,
    OIDCConfidentialClientConfig,
    OIDCConfig,
    OIDCPublicClientConfig,
    OIDCServerSessionsConfig,
)
from nhx.core.auth.oidc_broker.crypto import client_secret_basic_header, pkce_challenge
from nhx.core.auth.oidc_broker.routes import get_credential_store, get_identity_store, get_login_store
from nhx.core.auth.oidc_broker.store import DBAuthLoginTransaction, LoginRecord, OidcLoginStore
from nhx.core.entities.app.repository import AccountCredentialRecord, AccountCredentialType
from nhx.core.entities.app.repository.account_identity import AccountIdentityRecord
from nhx.core.entities.app.repository.sqlalchemy.base import Base
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

_CLI_CODE_VERIFIER = "v" * 48
_CLI_CODE_CHALLENGE = pkce_challenge(_CLI_CODE_VERIFIER)
_CLI_STATE = "cli-state-1234567890"


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
        self.rows: dict[tuple[AccountCredentialType, str], AccountCredentialRecord] = {}

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

    async def get_active(self, credential_type: AccountCredentialType, handle: str) -> AccountCredentialRecord | None:
        record = self.rows.get((credential_type, handle))
        if record is None or record.status != "ACTIVE":
            return None
        return record

    async def consume_active(
        self, credential_type: AccountCredentialType, handle: str
    ) -> AccountCredentialRecord | None:
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

    async def rotate_active(
        self,
        record: AccountCredentialRecord,
        *,
        new_handle: str,
        expires_at: datetime | None,
        public_metadata: dict[str, object],
        encrypted_payload: str,
    ) -> AccountCredentialRecord | None:
        for key, current in list(self.rows.items()):
            if current.id == record.id and current.status == "ACTIVE":
                del self.rows[key]
                return await self.create(
                    credential_type=record.credential_type,
                    handle=new_handle,
                    owner_account_id=record.owner_account_id,
                    subject_account_id=record.subject_account_id,
                    account_identity_id=record.account_identity_id,
                    expires_at=expires_at,
                    public_metadata=public_metadata,
                    encrypted_payload=encrypted_payload,
                )
        return None

    async def revoke(self, credential_type: AccountCredentialType, handle: str) -> bool:
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


def _confidential_auth_config(
    *,
    authorization_endpoint: str = "https://idp.example.com/authorize",
    bearer_token_source: Literal["access_token", "id_token"] = "access_token",
) -> AuthConfig:
    return AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.example.com",
            confidential_client=OIDCConfidentialClientConfig(
                client_id="nemo-helix-user",
                client_secret_env_var="NHX_OIDC_CLIENT_SECRET",
                login_redirect_uri="https://nemo.example.com/apis/auth/v2/login/callback",
                authorization_endpoint=authorization_endpoint,
                token_endpoint="https://idp.example.com/token",
                bearer_token_source=bearer_token_source,
            ),
            server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
        ),
    )


def _public_broker_auth_config() -> AuthConfig:
    return AuthConfig(
        enabled=True,
        oidc=OIDCConfig(
            enabled=True,
            issuer="https://idp.example.com",
            public_client=OIDCPublicClientConfig(
                client_id="nemo-helix-public",
                server_side_sessions=True,
                authorization_endpoint="https://idp.example.com/authorize",
                token_endpoint="https://idp.example.com/token",
            ),
            server_sessions=OIDCServerSessionsConfig(encryption_key_env_var="NHX_AUTH_SESSION_ENCRYPTION_KEY"),
        ),
    )


@pytest.mark.asyncio
async def test_login_store_pop_is_single_use_and_kind_scoped() -> None:
    engine, _session_maker, store = await _new_login_store()
    try:
        await store.put("token_authorization_txn", "state", {"verifier": "secret"}, ttl_seconds=60)

        assert await store.pop("other_txn", "state") is None
        record = await store.pop("token_authorization_txn", "state")
        assert record is not None
        assert record.payload == {"verifier": "secret"}
        assert await store.pop("token_authorization_txn", "state") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_login_store_pop_rejects_expired_transaction_and_deletes_row() -> None:
    engine, session_maker, store = await _new_login_store()
    try:
        await store.put("token_authorization_txn", "state", {"verifier": "secret"}, ttl_seconds=-1)

        assert await store.pop("token_authorization_txn", "state") is None
        assert await _login_transaction_kinds(session_maker) == set()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_login_store_get_rejects_expired_transaction() -> None:
    engine, _session_maker, store = await _new_login_store()
    try:
        await store.put("token_authorization_txn", "state", {"verifier": "secret"}, ttl_seconds=-1)

        assert await store.get("token_authorization_txn", "state") is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_login_store_put_purges_abandoned_expired_transactions() -> None:
    engine, session_maker, store = await _new_login_store()
    try:
        await store.put("expired_txn", "expired-state", {"verifier": "expired"}, ttl_seconds=-1)
        await store.put("valid_txn", "valid-state", {"verifier": "valid"}, ttl_seconds=60)
        await store.put("next_txn", "next-state", {"verifier": "next"}, ttl_seconds=60)

        assert await _login_transaction_kinds(session_maker) == {"valid_txn", "next_txn"}
    finally:
        await engine.dispose()


def test_login_transaction_expiry_has_index() -> None:
    assert "idx_auth_login_transactions_expires" in {index.name for index in DBAuthLoginTransaction.__table__.indexes}


async def _new_login_store() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession], OidcLoginStore]:
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, session_maker, OidcLoginStore(session_maker, "encryption-key")


async def _login_transaction_kinds(session_maker: async_sessionmaker[AsyncSession]) -> set[str]:
    async with session_maker() as session:
        result = await session.scalars(select(DBAuthLoginTransaction.kind))
        return set(result.all())


@pytest.fixture
def broker_app(monkeypatch: pytest.MonkeyPatch):
    transactions = MemoryLoginStore()
    credentials = MemoryCredentialStore()
    identities = MemoryIdentityStore()
    Configuration.set_override(_confidential_auth_config())
    Configuration.set_override(
        HelixConfig(
            runtime=Runtime.NONE,
            base_url="http://nemo-api:8080",
            advertised_base_url="https://nemo.example.com",
        )
    )
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


def _start_cli_authorization(client: TestClient) -> tuple[str, Httpx2Response]:
    start = client.post(
        "/apis/auth/v2/authorize",
        json={
            "redirect_uri": "http://127.0.0.1:54321/callback",
            "code_challenge": _CLI_CODE_CHALLENGE,
            "state": _CLI_STATE,
        },
    )
    transaction_id = start.json()["transaction_id"]
    redirect = client.get(start.json()["authorization_url"], follow_redirects=False)
    return transaction_id, redirect


def _start_web_session_login(
    client: TestClient,
    *,
    oidc_client: str = "confidential",
    return_to: str = "/studio/workspaces/example",
) -> tuple[str, Httpx2Response]:
    redirect = client.get(
        "/apis/auth/v2/login",
        params={"client": oidc_client, "return_to": return_to},
        follow_redirects=False,
    )
    state = parse_qs(urlparse(redirect.headers["location"]).query)["state"][0]
    return state, redirect


def test_web_session_login_sets_account_bound_cookie_and_supports_logout(broker_app) -> None:
    client, _transactions, credentials = broker_app
    with respx.mock:
        token_route = respx.post("https://idp.example.com/token").mock(
            return_value=httpx.Response(
                200,
                json={"access_token": "provider-access", "refresh_token": "provider-refresh"},
            )
        )
        state, redirect = _start_web_session_login(client)
        provider_query = parse_qs(urlparse(redirect.headers["location"]).query)
        assert provider_query["client_id"] == ["nemo-helix-user"]
        assert provider_query["redirect_uri"] == ["https://nemo.example.com/apis/auth/v2/login/callback"]

        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )

    assert callback.status_code == 302
    assert callback.headers["location"] == "/studio/workspaces/example"
    assert token_route.calls[0].request.headers["authorization"] == client_secret_basic_header(
        "nemo-helix-user", "user-login-secret"
    )
    session_handle = callback.cookies["nhx_session"]
    session_record = credentials.rows[("web_session", session_handle)]
    assert session_record.owner_account_id == "account-1"
    assert session_record.account_identity_id == "identity-1"
    assert session_record.public_metadata["oidc_client"] == "confidential"
    assert session_record.encrypted_payload is not None
    assert "provider-access" not in session_record.encrypted_payload
    assert "provider-refresh" not in session_record.encrypted_payload

    session = client.get("/apis/auth/v2/session")
    assert session.status_code == 200
    assert session.json() == {
        "id": "user-1",
        "email": "user@example.com",
        "groups": ["nemo-users"],
        "account_id": "account-1",
        "authz_aliases": ["user-1", "user@example.com"],
    }
    assert "provider-access" not in session.text

    missing_csrf = client.post("/apis/auth/v2/logout")
    assert missing_csrf.status_code == 403
    assert client.get("/apis/auth/v2/session").status_code == 200

    logout = client.post("/apis/auth/v2/logout", headers={"X-Source": "NeMo Studio"})
    assert logout.status_code == 204
    assert credentials.rows[("web_session", session_handle)].status == "REVOKED"
    assert client.get("/apis/auth/v2/session").status_code == 401


def test_public_web_session_login_exchanges_without_client_secret(broker_app) -> None:
    client, _transactions, credentials = broker_app
    Configuration.set_override(_public_broker_auth_config())
    with respx.mock:
        token_route = respx.post("https://idp.example.com/token").mock(
            return_value=httpx.Response(200, json={"access_token": "provider-access"})
        )
        state, redirect = _start_web_session_login(client, oidc_client="public")
        provider_query = parse_qs(urlparse(redirect.headers["location"]).query)
        assert provider_query["client_id"] == ["nemo-helix-public"]

        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )

    assert callback.status_code == 302
    provider_request = token_route.calls[0].request
    assert "authorization" not in provider_request.headers
    assert parse_qs(provider_request.content.decode())["client_id"] == ["nemo-helix-public"]
    session_handle = callback.cookies["nhx_session"]
    assert credentials.rows[("web_session", session_handle)].public_metadata["oidc_client"] == "public"


@pytest.mark.parametrize(
    "return_to",
    ["https://attacker.example.com", "//attacker.example.com", r"/\\attacker.example.com"],
)
def test_web_session_login_rejects_non_relative_return_target(broker_app, return_to: str) -> None:
    client, _transactions, _credentials = broker_app

    response = client.get("/apis/auth/v2/login", params={"return_to": return_to}, follow_redirects=False)

    assert response.status_code == 400
    assert response.json() == {"detail": "return_to must be a relative path"}


def test_cli_authorization_url_preserves_provider_query_params(broker_app) -> None:
    client, _transactions, _credentials = broker_app
    Configuration.set_override(
        _confidential_auth_config(authorization_endpoint="https://idp.example.com/authorize?audience=nemo")
    )

    _transaction_id, response = _start_cli_authorization(client)

    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "idp.example.com"
    assert parsed.path == "/authorize"
    assert query["audience"] == ["nemo"]
    assert query["client_id"] == ["nemo-helix-user"]
    assert query["redirect_uri"] == ["https://nemo.example.com/apis/auth/v2/login/callback"]


def test_invalid_provider_token_response_is_sanitized(broker_app) -> None:
    client, _transactions, _credentials = broker_app
    with respx.mock:
        respx.post("https://idp.example.com/token").mock(return_value=httpx.Response(200, content=b"not-json"))
        state, _redirect = _start_cli_authorization(client)

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
        state, _redirect = _start_cli_authorization(client)

        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )

    assert callback.status_code == 502
    assert callback.json() == {"detail": "Identity provider token request failed"}
    assert "provider details" not in callback.text


def test_provider_token_error_response_is_sanitized(broker_app) -> None:
    client, _transactions, _credentials = broker_app
    with respx.mock:
        respx.post("https://idp.example.com/token").mock(
            return_value=httpx.Response(400, json={"error": "invalid_grant", "error_description": "provider details"})
        )
        state, _redirect = _start_cli_authorization(client)

        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )

    assert callback.status_code == 502
    assert callback.json() == {"detail": "Identity provider token request failed"}
    assert "provider details" not in callback.text


def test_provider_callback_error_redirects_to_cli_loopback(broker_app) -> None:
    client, _transactions, _credentials = broker_app
    state, _redirect = _start_cli_authorization(client)

    callback = client.get(
        "/apis/auth/v2/login/callback",
        params={"error": "access_denied", "error_description": "User cancelled", "state": state},
        follow_redirects=False,
    )

    assert callback.status_code == 302
    parsed = urlparse(callback.headers["location"])
    assert parsed.scheme == "http"
    assert parsed.netloc == "127.0.0.1:54321"
    assert parsed.path == "/callback"
    query = parse_qs(parsed.query)
    assert query["error"] == ["access_denied"]
    assert query["error_description"] == ["User cancelled"]
    assert query["state"] == [_CLI_STATE]


def test_provider_callback_without_code_or_error_is_rejected(broker_app) -> None:
    client, _transactions, _credentials = broker_app
    state, _redirect = _start_cli_authorization(client)

    callback = client.get(
        "/apis/auth/v2/login/callback",
        params={"state": state},
        follow_redirects=False,
    )

    assert callback.status_code == 400
    assert callback.json() == {"detail": "Login callback did not include an authorization code"}


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
def test_token_authorization_rejects_invalid_loopback_redirect(broker_app, redirect_uri: str) -> None:
    client, _transactions, _credentials = broker_app

    response = client.post(
        "/apis/auth/v2/authorize",
        json={
            "redirect_uri": redirect_uri,
            "code_challenge": _CLI_CODE_CHALLENGE,
            "state": _CLI_STATE,
        },
    )

    assert response.status_code == 400


def test_token_endpoint_returns_opaque_refresh_handle_and_refresh_uses_provider_token(broker_app) -> None:
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
        start = client.post(
            "/apis/auth/v2/authorize",
            json={
                "redirect_uri": "http://127.0.0.1:54321/callback",
                "code_challenge": _CLI_CODE_CHALLENGE,
                "state": _CLI_STATE,
            },
        )
        transaction_id = start.json()["transaction_id"]
        assert transaction_id in transactions.rows
        assert start.json()["authorization_url"] == f"https://nemo.example.com/apis/auth/v2/authorize/{transaction_id}"
        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": transaction_id},
            follow_redirects=False,
        )
        assert token_route.calls[0].request.headers["authorization"].startswith("Basic ")
        assert b"client_secret" not in token_route.calls[0].request.content
        one_time_code = parse_qs(urlparse(callback.headers["location"]).query)["code"][0]
        pending = credentials.rows[("token_authorization_code", one_time_code)]
        assert pending.encrypted_payload is not None
        assert "provider-refresh-1" not in pending.encrypted_payload
        assert parse_qs(urlparse(callback.headers["location"]).query)["state"] == [_CLI_STATE]
        exchanged = client.post(
            "/apis/auth/v2/token",
            json={
                "grant_type": "authorization_code",
                "code": one_time_code,
                "code_verifier": _CLI_CODE_VERIFIER,
            },
        )
        assert exchanged.status_code == 200
        refresh_handle = exchanged.json()["refresh_token"]
        assert refresh_handle != "provider-refresh-1"
        assert ("token_authorization_code", one_time_code) not in credentials.rows
        assert ("broker_refresh", refresh_handle) in credentials.rows

        refreshed = client.post(
            "/apis/auth/v2/token",
            json={"grant_type": "refresh_token", "refresh_token": refresh_handle},
        )
        assert refreshed.json()["access_token"] == "provider-access-2"
        rotated_refresh_handle = refreshed.json()["refresh_token"]
        assert rotated_refresh_handle != refresh_handle
        assert ("broker_refresh", refresh_handle) not in credentials.rows
        assert ("broker_refresh", rotated_refresh_handle) in credentials.rows
        assert token_route.calls[-1].request.content == b"grant_type=refresh_token&refresh_token=provider-refresh-1"

        reused = client.post(
            "/apis/auth/v2/token",
            json={"grant_type": "refresh_token", "refresh_token": refresh_handle},
        )
        assert reused.status_code == 400
        assert reused.json() == {"detail": "Refresh token was not found"}


def test_confidential_broker_returns_configured_id_token_bearer(broker_app) -> None:
    client, transactions, _credentials = broker_app
    Configuration.set_override(_confidential_auth_config(bearer_token_source="id_token"))
    with respx.mock:
        respx.post("https://idp.example.com/token").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "access_token": "provider-access",
                        "id_token": "provider-id",
                        "refresh_token": "provider-refresh",
                        "expires_in": 900,
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "id_token": "provider-id-2",
                        "refresh_token": "provider-refresh-2",
                        "expires_in": 900,
                    },
                ),
            ]
        )
        start = client.post(
            "/apis/auth/v2/authorize",
            json={
                "redirect_uri": "http://127.0.0.1:54321/callback",
                "code_challenge": _CLI_CODE_CHALLENGE,
                "state": _CLI_STATE,
            },
        )
        transaction_id = start.json()["transaction_id"]
        assert transaction_id in transactions.rows
        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": transaction_id},
            follow_redirects=False,
        )
        one_time_code = parse_qs(urlparse(callback.headers["location"]).query)["code"][0]

        exchanged = client.post(
            "/apis/auth/v2/token",
            json={
                "grant_type": "authorization_code",
                "code": one_time_code,
                "code_verifier": _CLI_CODE_VERIFIER,
            },
        )

        assert exchanged.status_code == 200
        assert exchanged.json()["access_token"] == "provider-id"
        refreshed = client.post(
            "/apis/auth/v2/token",
            json={"grant_type": "refresh_token", "refresh_token": exchanged.json()["refresh_token"]},
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"] == "provider-id-2"


def test_token_endpoint_rejects_wrong_cli_code_verifier(broker_app) -> None:
    client, _transactions, credentials = broker_app
    with respx.mock:
        respx.post("https://idp.example.com/token").mock(
            return_value=httpx.Response(
                200,
                json={"access_token": "provider-access", "refresh_token": "provider-refresh"},
            )
        )
        state, _redirect = _start_cli_authorization(client)
        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": state},
            follow_redirects=False,
        )
        one_time_code = parse_qs(urlparse(callback.headers["location"]).query)["code"][0]

    response = client.post(
        "/apis/auth/v2/token",
        json={
            "grant_type": "authorization_code",
            "code": one_time_code,
            "code_verifier": "wrong-verifier" * 4,
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Code verifier did not match"}
    assert ("token_authorization_code", one_time_code) in credentials.rows

    accepted = client.post(
        "/apis/auth/v2/token",
        json={
            "grant_type": "authorization_code",
            "code": one_time_code,
            "code_verifier": _CLI_CODE_VERIFIER,
        },
    )
    assert accepted.status_code == 200


def test_public_broker_exchanges_without_client_secret_basic(broker_app) -> None:
    client, _transactions, credentials = broker_app
    Configuration.set_override(_public_broker_auth_config())
    with respx.mock:
        token_route = respx.post("https://idp.example.com/token").mock(
            return_value=httpx.Response(
                200,
                json={"access_token": "provider-access", "refresh_token": "provider-refresh"},
            )
        )
        start = client.post(
            "/apis/auth/v2/authorize?client=public",
            json={
                "redirect_uri": "http://127.0.0.1:54321/callback",
                "code_challenge": _CLI_CODE_CHALLENGE,
                "state": _CLI_STATE,
            },
        )
        transaction_id = start.json()["transaction_id"]
        redirect = client.get(start.json()["authorization_url"], follow_redirects=False)
        provider_query = parse_qs(urlparse(redirect.headers["location"]).query)
        assert provider_query["client_id"] == ["nemo-helix-public"]
        assert provider_query["redirect_uri"] == ["https://nemo.example.com/apis/auth/v2/login/callback"]

        callback = client.get(
            "/apis/auth/v2/login/callback",
            params={"code": "auth-code", "state": transaction_id},
            follow_redirects=False,
        )

    provider_request = token_route.calls[0].request
    assert "authorization" not in provider_request.headers
    provider_form = parse_qs(provider_request.content.decode())
    assert provider_form["client_id"] == ["nemo-helix-public"]
    assert provider_form["code_verifier"]
    one_time_code = parse_qs(urlparse(callback.headers["location"]).query)["code"][0]
    pending = credentials.rows[("token_authorization_code", one_time_code)]
    assert pending.encrypted_payload is not None
    assert "provider-refresh" not in pending.encrypted_payload
    exchanged = client.post(
        "/apis/auth/v2/token?client=public",
        json={
            "grant_type": "authorization_code",
            "code": one_time_code,
            "code_verifier": _CLI_CODE_VERIFIER,
        },
    )
    assert exchanged.status_code == 200
    assert exchanged.json()["access_token"] == "provider-access"
    refresh_handle = exchanged.json()["refresh_token"]

    with respx.mock:
        refresh_route = respx.post("https://idp.example.com/token").mock(
            return_value=httpx.Response(200, json={"access_token": "provider-access-2"})
        )
        refreshed = client.post(
            "/apis/auth/v2/token?client=public",
            json={"grant_type": "refresh_token", "refresh_token": refresh_handle},
        )

    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"] == "provider-access-2"
    assert "authorization" not in refresh_route.calls[0].request.headers
    assert parse_qs(refresh_route.calls[0].request.content.decode()) == {
        "grant_type": ["refresh_token"],
        "refresh_token": ["provider-refresh"],
        "client_id": ["nemo-helix-public"],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"grant_type": "authorization_code"},
        {
            "grant_type": "authorization_code",
            "code": "code",
            "code_verifier": "not-ascii-\N{LATIN SMALL LETTER E WITH ACUTE}" * 5,
        },
        {"grant_type": "refresh_token"},
        {"grant_type": "unsupported", "code": "code", "code_verifier": _CLI_CODE_VERIFIER},
        {
            "grant_type": "authorization_code",
            "code": "code",
            "code_verifier": _CLI_CODE_VERIFIER,
            "refresh_token": "refresh",
        },
    ],
)
def test_token_endpoint_rejects_invalid_grant_shapes(broker_app, payload: dict[str, str]) -> None:
    client, _transactions, _credentials = broker_app

    response = client.post("/apis/auth/v2/token", json=payload)

    assert response.status_code == 422


def test_discovery_advertises_only_configured_clients() -> None:
    from nhx.core.auth.api.v2.discovery.endpoints import get_auth_discovery

    Configuration.set_override(_confidential_auth_config())
    Configuration.set_override(
        HelixConfig(
            runtime=Runtime.NONE,
            base_url="http://nemo-api:8080",
            advertised_base_url="https://nemo.example.com",
        )
    )
    try:
        import asyncio

        result = asyncio.run(get_auth_discovery())
    finally:
        Configuration.clear_overrides()
    assert result.oidc is not None
    dumped = result.oidc.model_dump()
    assert [client["name"] for client in dumped["clients"]] == ["confidential"]
    assert [client["client_authentication"] for client in dumped["clients"]] == ["client_secret_basic"]
    assert dumped["clients"][0]["default"] is True
    assert dumped["clients"][0]["authorization_start_endpoint"] == "https://nemo.example.com/apis/auth/v2/authorize"
    assert dumped["clients"][0]["broker_token_endpoint"] == "https://nemo.example.com/apis/auth/v2/token"
    assert "client_id" not in dumped
    assert "client_secret" not in dumped
    assert "NHX_OIDC_CLIENT_SECRET" not in str(dumped)


def test_broker_redirects_advertise_302_location_responses(broker_app) -> None:
    client, _transactions, _credentials = broker_app

    paths = client.app.openapi()["paths"]

    for path in ("/apis/auth/v2/authorize/{transaction_id}", "/apis/auth/v2/login/callback"):
        responses = paths[path]["get"]["responses"]
        assert "200" not in responses
        response = responses["302"]
        assert response["headers"]["Location"]["schema"] == {"type": "string"}
