# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the credentials router.

Uses an in-process FastAPI TestClient with a mocked psycopg connection
(via dependency_overrides on the mounted /v1 sub-app). End-to-end coverage
against a real Postgres lives in tests/integration/.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api import credential_verification as verification
from scaled_evals.api import crypto
from scaled_evals.api.credential_verification import (
    CredentialVerificationFailed,
    CredentialVerificationResult,
    CredentialVerificationUnavailable,
)
from scaled_evals.api.db import get_conn
from scaled_evals.api.routers import credentials as credentials_router
from scaled_evals.api.settings import settings


def _empty_db() -> Iterator[MagicMock]:
    """Fake connection whose queries return nothing — empty list, no row."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    yield conn


@pytest.fixture(autouse=True)
def _override_db():
    v1.dependency_overrides[get_conn] = _empty_db
    yield
    v1.dependency_overrides.pop(get_conn, None)


# ---------- list ----------------------------------------------------------


def test_list_returns_envelope_shape() -> None:
    response = client.get("/v1/credentials")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}


def test_list_rejects_unknown_provider_filter() -> None:
    response = client.get("/v1/credentials", params={"provider": "bogus"})
    assert response.status_code == 422


# ---------- get / patch / rotate / delete / verify: 404 when missing ------


def test_get_returns_404_when_missing() -> None:
    response = client.get("/v1/credentials/cred_does_not_exist")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_patch_returns_404_when_missing() -> None:
    response = client.patch("/v1/credentials/cred_missing", json={"name": "new"})
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_rotate_returns_404_when_missing() -> None:
    response = client.post("/v1/credentials/cred_missing/rotate", json={"key": "sk-new"})
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_rotate_returns_409_when_active_evaluation_references_credential() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {"id": "ev_active"}

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.post("/v1/credentials/cred_in_use/rotate", json={"key": "sk-new"})

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "credential_in_use"
    calls = cur.execute.call_args_list
    assert "FOR UPDATE" in calls[0].args[0]
    assert "status NOT IN" in calls[1].args[0]


def test_delete_returns_404_when_missing() -> None:
    response = client.delete("/v1/credentials/cred_missing")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_delete_returns_409_when_active_evaluation_references_credential() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {"id": "ev_active"}

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen

    response = client.delete("/v1/credentials/cred_in_use")

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "credential_in_use"


def test_delete_allows_credential_when_no_active_evaluation_references_it() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [
        {"id": "cred_done"},
        None,
        {"id": "cred_done"},
    ]

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen

    response = client.delete("/v1/credentials/cred_done")

    assert response.status_code == 200
    assert response.json() == {"id": "cred_done", "deleted": True}


def test_verify_returns_404_when_missing() -> None:
    response = client.post("/v1/credentials/cred_missing/verify")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_verify_probes_provider_with_decrypted_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {
        "id": "cred_live",
        "provider": "openai",
        "payload_kind": "key",
        "encrypted_payload": crypto.encrypt("sk-live"),
        "fingerprint": "sha256:live",
    }

    def _gen() -> Iterator[MagicMock]:
        yield conn

    observed = {}

    def _verify(row: dict) -> CredentialVerificationResult:
        observed.update(row)
        return CredentialVerificationResult(True, "provider accepted credential")

    monkeypatch.setattr(credentials_router, "verify_stored_credential", _verify)
    v1.dependency_overrides[get_conn] = _gen

    response = client.post("/v1/credentials/cred_live/verify")

    assert response.status_code == 200
    assert response.json() == {
        "id": "cred_live",
        "verified": True,
        "reason": "provider accepted credential",
    }
    assert observed["provider"] == "openai"
    assert observed["payload_kind"] == "key"
    assert "encrypted_payload" in observed
    assert "sk-live" not in response.text


def test_verify_returns_422_when_provider_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {
        "id": "cred_bad",
        "provider": "nvidia",
        "payload_kind": "key",
        "encrypted_payload": crypto.encrypt("nvapi-bad"),
        "fingerprint": "sha256:bad",
    }

    def _gen() -> Iterator[MagicMock]:
        yield conn

    def _verify(row: dict) -> CredentialVerificationResult:
        raise CredentialVerificationFailed("provider rejected credential", status_code=401)

    monkeypatch.setattr(credentials_router, "verify_stored_credential", _verify)
    v1.dependency_overrides[get_conn] = _gen

    response = client.post("/v1/credentials/cred_bad/verify")

    assert response.status_code == 422
    body = response.json()["detail"]["error"]
    assert body["code"] == "credential_verify_failed"
    assert body["details"] == {"provider": "nvidia", "status_code": 401}
    assert "nvapi-bad" not in response.text


def test_verify_returns_503_when_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {
        "id": "cred_unavailable",
        "provider": "openai",
        "payload_kind": "key",
        "encrypted_payload": crypto.encrypt("sk-unavailable"),
        "fingerprint": "sha256:unavailable",
    }

    def _gen() -> Iterator[MagicMock]:
        yield conn

    def _verify(row: dict) -> CredentialVerificationResult:
        raise CredentialVerificationUnavailable("openai verification endpoint unavailable")

    monkeypatch.setattr(credentials_router, "verify_stored_credential", _verify)
    v1.dependency_overrides[get_conn] = _gen

    response = client.post("/v1/credentials/cred_unavailable/verify")

    assert response.status_code == 503
    body = response.json()["detail"]["error"]
    assert body["code"] == "credential_verify_unavailable"
    assert body["details"] == {"provider": "openai"}
    assert "sk-unavailable" not in response.text


def test_verify_returns_inconclusive_for_nhx_yaml() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {
        "id": "cred_nhx",
        "provider": "nhx",
        "payload_kind": "yaml",
        "encrypted_payload": crypto.encrypt("workspace: ws-1\n"),
        "fingerprint": "sha256:nhx",
    }

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen

    response = client.post("/v1/credentials/cred_nhx/verify")

    assert response.status_code == 200
    assert response.json() == {
        "id": "cred_nhx",
        "verified": None,
        "reason": "provider verification unsupported for nhx",
    }
    assert "workspace: ws-1" not in response.text


def test_verify_provider_credential_uses_openai_bearer_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "credential_verify_openai_models_url",
        "https://provider.test/v1/models",
    )
    observed = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    with httpx.Client(transport=httpx.MockTransport(_handler)) as upstream:
        result = verification.verify_provider_credential(
            "openai",
            "key",
            "sk-test",
            client=upstream,
        )

    assert result == CredentialVerificationResult(True, "provider accepted credential")
    assert observed == {
        "url": "https://provider.test/v1/models",
        "authorization": "Bearer sk-test",
    }


def test_verify_stored_credential_decrypts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def _verify(
        provider: str,
        payload_kind: str,
        payload: str,
        *,
        client: httpx.Client | None = None,
    ) -> CredentialVerificationResult:
        observed.update(
            provider=provider,
            payload_kind=payload_kind,
            payload=payload,
            client=client,
        )
        return CredentialVerificationResult(True, "provider accepted credential")

    monkeypatch.setattr(verification, "verify_provider_credential", _verify)
    upstream = MagicMock(spec=httpx.Client)

    result = verification.verify_stored_credential(
        {
            "provider": "anthropic",
            "payload_kind": "key",
            "encrypted_payload": crypto.encrypt("sk-stored"),
        },
        client=upstream,
    )

    assert result.verified is True
    assert observed == {
        "provider": "anthropic",
        "payload_kind": "key",
        "payload": "sk-stored",
        "client": upstream,
    }


def test_verify_provider_credential_raises_on_rejected_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "credential_verify_nvidia_models_url",
        "https://provider.test/v1/models",
    )

    transport = httpx.MockTransport(lambda request: httpx.Response(401))
    with (
        httpx.Client(transport=transport) as upstream,
        pytest.raises(CredentialVerificationFailed) as exc_info,
    ):
        verification.verify_provider_credential(
            "nvidia",
            "key",
            "nvapi-bad",
            client=upstream,
        )

    assert str(exc_info.value) == "provider rejected credential"
    assert exc_info.value.status_code == 401


# ---------- create: request-body validation (no DB hit) -------------------
#
# Pydantic validation runs before dependency resolution, so these don't
# reach the INSERT — they're guarded at the boundary.


def test_create_rejects_empty_name() -> None:
    response = client.post(
        "/v1/credentials",
        json={"name": "", "provider": "openai", "key": "sk-x"},
    )
    assert response.status_code == 422


def test_create_rejects_missing_payload() -> None:
    response = client.post(
        "/v1/credentials",
        json={"name": "k", "provider": "openai"},
    )
    assert response.status_code == 422


def test_create_rejects_both_payloads() -> None:
    response = client.post(
        "/v1/credentials",
        json={
            "name": "k",
            "provider": "openai",
            "key": "sk-x",
            "yaml": "a: 1",
        },
    )
    assert response.status_code == 422


def test_create_rejects_unknown_provider() -> None:
    response = client.post(
        "/v1/credentials",
        json={"name": "k", "provider": "cohere", "key": "sk-x"},
    )
    assert response.status_code == 422


def test_create_422_never_echoes_submitted_secret() -> None:
    """A validation 422 must not leak the submitted secret (key/yaml).

    FastAPI's default handler echoes the offending value in each error's
    ``input``/``ctx``; on a secrets endpoint that leaks the plaintext key.
    We send a request carrying a secret but missing the required ``name`` so
    validation fails, then assert the secret is absent from the response while
    the structural error shape (type/loc/msg) is preserved.
    """
    secret = "sk-super-secret-do-not-leak"
    response = client.post(
        "/v1/credentials",
        json={"provider": "openai", "key": secret},
    )

    assert response.status_code == 422
    assert secret not in response.text

    errors = response.json()["detail"]
    assert errors, "expected at least one validation error entry"
    for err in errors:
        assert {"type", "loc", "msg"} <= err.keys()
        assert "input" not in err
        assert "ctx" not in err
