# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_platform_plugin.auth.access_keys.issuer import AccessKeyOperationNotImplementedError
from nemo_platform_plugin.auth.access_keys.types import AccessKeyCreateResponse
from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_platform_plugin.workspaces.types import CreateWorkspaceMemberRequest
from nmp.common.auth.access_keys import AccessKeyValidationError
from nmp.common.auth.client import AuthClient
from nmp.common.auth.dependencies import auth_client_context
from nmp.common.auth.models import Principal
from nmp.common.config import AuthConfig
from nmp.common.config.base import AccessKeyConfig, TokenSigningConfig
from nmp.common.entities import EntityConflictError
from nmp.core.auth.api.v2.access_keys.endpoints import get_access_key_issuer, get_workspaces_client, router
from nmp.core.auth.app.access_keys import AccessKeyNotFoundError, AccessKeyStateConflictError, get_access_key_registry


class InMemoryAccessKeyRegistry:
    def __init__(self):
        self.keys = {}
        self.owners = {}
        self.revoked = set()
        self.suspended = set()
        # jti -> grace_period_expires_at for keys rotated out via begin_rotation.
        self.rotating = {}
        # jti -> the successor key jti recorded by whichever begin_rotation call
        # actually committed the ACTIVE -> ROTATING transition.
        self.rotating_successor = {}
        self.begin_rotation_error = None
        self.begin_rotation_pre_write_error = None
        self.begin_rotation_commits_before_error = False
        self.mark_rotating_before_begin_rotation_error = False
        # Lets a test simulate a *different*, concurrently-racing request's successor
        # having won the transition, rather than this request's own successor_jti.
        self.begin_rotation_successor_override = None
        # Lets a test simulate get_status reconciling to a status that already moved
        # on past ROTATING (e.g. the grace period elapsed, or a manual revoke landed)
        # by the time reconciliation runs, independent of self.rotating's own deadline.
        self.get_status_status_override = None
        self.add_error = None
        self.add_commits_before_error = False
        self.discarded = set()

    async def add(self, key, *, owner_principal=None):
        if self.add_error is not None and not self.add_commits_before_error:
            raise self.add_error
        self.keys[key.jti] = key
        self.owners[key.jti] = owner_principal or key.principal
        if self.add_error is not None:
            raise self.add_error

    def _status(self, jti, key):
        if jti in self.revoked:
            return "REVOKED"
        if key.expires_at is not None and key.expires_at <= datetime.now(tz=UTC) - timedelta(seconds=30):
            return "EXPIRED"
        if jti in self.rotating:
            return "REVOKED" if datetime.now(tz=UTC) >= self.rotating[jti] else "ROTATING"
        if jti in self.suspended:
            return "SUSPENDED"
        return key.status

    async def _may_manage(self, jti, principal, *, admin_override=None):
        key = self.keys.get(jti)
        if key is None:
            return None
        if key.entity_type == "SERVICE_ACCOUNT":
            if admin_override is not None and await admin_override():
                return key
        elif self.owners[jti] == principal:
            return key
        return None

    async def list_for_principal(self, principal, *, page, page_size, include_service_accounts=False):
        from nemo_platform_plugin.auth.access_keys.types import AccessKeyListResponse, AccessKeyMetadataResponse

        # Sort newest-first then by jti to match the real registry's `sort="-issued_at"`.
        owned = sorted(
            [
                (jti, key)
                for jti, key in self.keys.items()
                if self.owners[jti] == principal
                and key.entity_type != "SERVICE_ACCOUNT"
                or (include_service_accounts and key.entity_type == "SERVICE_ACCOUNT")
            ],
            key=lambda item: (-item[1].created_at.timestamp(), item[0]),
        )
        start = (page - 1) * page_size
        selected = owned[start : start + page_size]
        return AccessKeyListResponse(
            data=[
                AccessKeyMetadataResponse.model_validate(
                    key.model_dump(exclude={"token", "token_type"})
                    | {
                        "status": self._status(jti, key),
                        "grace_period_expires_at": (
                            self.rotating.get(jti) if self._status(jti, key) == "ROTATING" else None
                        ),
                    }
                )
                for jti, key in selected
            ],
            has_more=start + page_size < len(owned),
        )

    async def revoke(self, jti, principal, *, admin_override=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        revoked = jti not in self.revoked
        self.revoked.add(jti)
        self.suspended.discard(jti)
        return revoked

    async def get_for_rotation(self, jti, principal, *, admin_override=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyValidationError(
                f"rotates references a Scoped Access Key that does not exist or is not owned by the caller: {jti}"
            )
        if key.entity_type == "SERVICE_ACCOUNT":
            raise AccessKeyValidationError(
                "rotates only supports personal Scoped Access Keys without a service account"
            )
        return key

    async def suspend(self, jti, principal, *, admin_override=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        if jti in self.revoked:
            raise AccessKeyStateConflictError(f"Revoked Scoped Access Key {jti} cannot be suspended")
        if key.expires_at is not None and key.expires_at <= datetime.now(tz=UTC):
            return False, "EXPIRED"
        changed = jti not in self.suspended
        self.suspended.add(jti)
        return changed, self._status(jti, key)

    async def unsuspend(self, jti, principal, *, admin_override=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        if jti in self.revoked:
            raise AccessKeyStateConflictError(f"Revoked Scoped Access Key {jti} cannot be unsuspended")
        if key.expires_at is not None and key.expires_at <= datetime.now(tz=UTC):
            return False, "EXPIRED"
        changed = jti in self.suspended
        self.suspended.discard(jti)
        return changed, self._status(jti, key)

    async def is_active(self, jti, principal, **kwargs):
        key = self.keys.get(jti)
        return key is not None and key.principal == principal and self._status(jti, key) in ("ACTIVE", "ROTATING")

    def _ensure_rotatable(self, jti, key):
        if jti in self.revoked or jti in self.suspended or jti in self.rotating:
            raise AccessKeyStateConflictError(f"Scoped Access Key {jti} must be ACTIVE to rotate")
        if key.expires_at is not None and key.expires_at <= datetime.now(tz=UTC):
            raise AccessKeyStateConflictError(f"Expired Scoped Access Key {jti} cannot be rotated")

    def _as_entity_view(self, jti, key):
        # AccessKeyRegistry.get_rotatable returns an AccessKeyEntity-shaped record
        # (key_name/issued_at/subject_principal, not name/created_at/principal-as-subject).
        # This fake stores raw AccessKeyCreateResponse objects, so translate field names
        # for rotate_async's state reads, which rely on the entity shape.
        owner = self.owners[jti]
        return SimpleNamespace(
            key_name=key.name,
            description=key.description,
            principal=owner,
            subject_principal=key.principal if key.principal != owner else None,
            entity_type=key.entity_type,
            scope=key.scope or [],
            issued_at=key.created_at,
            expires_at=key.expires_at,
            status=self._status(jti, key),
            grace_period_expires_at=self.rotating.get(jti),
            rotation_successor_jti=self.rotating_successor.get(jti),
        )

    async def get_rotatable(self, jti, principal, *, admin_override=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        self._ensure_rotatable(jti, key)
        return self._as_entity_view(jti, key)

    async def get_status(self, jti, principal, *, admin_override=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        view = self._as_entity_view(jti, key)
        if self.get_status_status_override is not None:
            view = SimpleNamespace(**{**vars(view), "status": self.get_status_status_override})
        return view

    async def discard_unreturned(self, jti):
        self.keys.pop(jti, None)
        self.owners.pop(jti, None)
        self.revoked.discard(jti)
        self.suspended.discard(jti)
        self.rotating.pop(jti, None)
        self.rotating_successor.pop(jti, None)
        self.discarded.add(jti)

    async def begin_rotation(self, jti, principal, *, grace_period_seconds, successor_jti, admin_override=None):
        recorded_successor_jti = self.begin_rotation_successor_override or successor_jti
        if self.begin_rotation_pre_write_error is not None:
            error = self.begin_rotation_pre_write_error
            error.__dict__["write_attempted"] = False
            if self.mark_rotating_before_begin_rotation_error:
                self.rotating[jti] = datetime.now(tz=UTC) + timedelta(seconds=grace_period_seconds)
                self.rotating_successor[jti] = recorded_successor_jti
            raise error
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        self._ensure_rotatable(jti, key)
        if self.begin_rotation_error is not None:
            if self.mark_rotating_before_begin_rotation_error:
                self.rotating[jti] = datetime.now(tz=UTC) + timedelta(seconds=grace_period_seconds)
                self.rotating_successor[jti] = recorded_successor_jti
            if self.begin_rotation_commits_before_error:
                self.rotating[jti] = datetime.now(tz=UTC) + timedelta(seconds=grace_period_seconds)
                self.rotating_successor[jti] = recorded_successor_jti
            raise self.begin_rotation_error
        self.rotating[jti] = datetime.now(tz=UTC) + timedelta(seconds=grace_period_seconds)
        self.rotating_successor[jti] = successor_jti
        return self._as_entity_view(jti, key)


@pytest.fixture
def access_key_workspaces_client():
    return AsyncMock(spec=AsyncWorkspacesClient)


@pytest.fixture
def client(tmp_path, access_key_workspaces_client):
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="test-access-key",
            private_key_file=str(tmp_path / "private.pem"),
        ),
        access_keys=AccessKeyConfig(
            enabled=True,
            audience="nemo-platform-access-key",
        ),
    )

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    (tmp_path / "private.pem").write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    app = FastAPI()
    app.include_router(router)
    registry = InMemoryAccessKeyRegistry()
    app.dependency_overrides[get_access_key_registry] = lambda: registry
    app.dependency_overrides[get_workspaces_client] = lambda: access_key_workspaces_client

    token = auth_client_context.set(
        AuthClient(
            principal=Principal(id="alice@example.com", email="alice@example.com", groups=["team-ml"]),
            config=config,
        )
    )
    with patch("nmp.core.auth.api.v2.access_keys.endpoints.get_auth_config", return_value=config):
        yield TestClient(app)
    auth_client_context.reset(token)


@pytest.fixture
def disabled_client(access_key_workspaces_client):
    config = AuthConfig(enabled=True, access_keys=AccessKeyConfig())
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_access_key_registry] = lambda: InMemoryAccessKeyRegistry()
    app.dependency_overrides[get_workspaces_client] = lambda: access_key_workspaces_client
    token = auth_client_context.set(
        AuthClient(
            principal=Principal(id="alice@example.com", email="alice@example.com", groups=["team-ml"]),
            config=config,
        )
    )
    with patch("nmp.core.auth.api.v2.access_keys.endpoints.get_auth_config", return_value=config):
        yield TestClient(app)
    auth_client_context.reset(token)


def test_create_access_key_returns_token_for_current_principal(client):
    response = client.post(
        "/v2/access-keys",
        json={
            "name": "ci-intake",
            "description": "CI intake automation",
            "expires_in_seconds": 3600,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["jti"].startswith("ak_")
    assert body["name"] == "ci-intake"
    assert body["description"] == "CI intake automation"
    assert body["token_type"] == "Bearer"
    assert body["principal"] == "alice@example.com"
    assert body["scope"] == []
    assert body["expires_at"] is not None
    assert body["token"].count(".") == 2


def test_create_access_key_rotates_prior_owned_key(client):
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    response = client.post(
        "/v2/access-keys",
        json={"name": "replacement", "rotates": prior["jti"]},
    )

    assert response.status_code == 200
    listed = client.get("/v2/access-keys").json()["data"]
    assert next(key for key in listed if key["jti"] == prior["jti"])["status"] == "REVOKED"


def test_create_access_key_rejects_rotation_of_missing_key(client):
    missing_jti = "ak_00000000000000000000000000000000"

    response = client.post(
        "/v2/access-keys",
        json={"name": "replacement", "rotates": missing_jti},
    )

    assert response.status_code == 400
    listed = client.get("/v2/access-keys").json()["data"]
    assert [key["name"] for key in listed] == []


def test_create_access_key_rejects_rotation_of_service_bound_key(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        service_key = client.post(
            "/v2/access-keys",
            json={"name": "service-key", "service_account_id": "otel-collector"},
        ).json()

        response = client.post(
            "/v2/access-keys",
            json={"name": "replacement", "rotates": service_key["jti"]},
        )

    assert response.status_code == 400
    assert "service account" in response.json()["detail"]


def test_create_access_key_rotation_revoke_failure_fails_the_request_and_compensates(client, caplog, monkeypatch):
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    # Simulate an unexpected failure revoking the rotation target specifically (e.g. a
    # concurrent delete), while leaving revocation of any other jti (the compensating
    # self-revoke of the new key) working normally.
    from nmp.core.auth.app.access_keys import PersistentAccessKeyIssuer

    original_revoke_async = PersistentAccessKeyIssuer.revoke_async

    async def _boom_for_target_only(self, jti):
        if jti == prior["jti"]:
            raise RuntimeError("entity storage unavailable")
        return await original_revoke_async(self, jti)

    monkeypatch.setattr(
        "nmp.core.auth.app.access_keys.PersistentAccessKeyIssuer.revoke_async",
        _boom_for_target_only,
    )

    with caplog.at_level(logging.WARNING):
        # The request fails instead of silently returning a key while the predecessor stays live.
        with pytest.raises(RuntimeError, match="entity storage unavailable"):
            client.post(
                "/v2/access-keys",
                json={"name": "replacement", "rotates": prior["jti"]},
            )

    record = next(
        record
        for record in caplog.records
        if getattr(record, "audit_event", None) == "access_key.rotation_revoke_failed"
    )
    assert record.rotated_from_jti == prior["jti"]

    listed = client.get("/v2/access-keys").json()["data"]
    prior_listed = next(key for key in listed if key["jti"] == prior["jti"])
    assert prior_listed["status"] == "ACTIVE"
    # The half-created replacement was compensated away (revoked) rather than left dangling.
    replacement_listed = next(key for key in listed if key["name"] == "replacement")
    assert replacement_listed["status"] == "REVOKED"


def test_create_access_key_rotation_revoke_failure_logs_when_compensation_also_fails(client, caplog, monkeypatch):
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    async def _boom(self, jti):
        raise RuntimeError("entity storage unavailable")

    monkeypatch.setattr(
        "nmp.core.auth.app.access_keys.PersistentAccessKeyIssuer.revoke_async",
        _boom,
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="entity storage unavailable"):
            client.post(
                "/v2/access-keys",
                json={"name": "replacement", "rotates": prior["jti"]},
            )

    assert any(
        getattr(record, "audit_event", None) == "access_key.compensating_revoke_failed" for record in caplog.records
    )


def test_create_access_key_rejects_rotation_combined_with_service_account(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

        response = client.post(
            "/v2/access-keys",
            json={
                "name": "replacement",
                "rotates": prior["jti"],
                "service_account_id": "otel-collector",
            },
        )

    assert response.status_code == 400
    assert "rotates cannot be combined with service_account_id" in response.json()["detail"]
    listed = client.get("/v2/access-keys").json()["data"]
    assert [key["name"] for key in listed] == ["prior"]


def test_create_access_key_grants_workspace_membership(client, access_key_workspaces_client):
    response = client.post(
        "/v2/access-keys",
        json={
            "name": "workspace-key",
            "workspaces": [{"workspace": "team-a", "roles": ["Viewer"]}],
        },
    )

    assert response.status_code == 200
    access_key_workspaces_client.create_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        body=CreateWorkspaceMemberRequest(
            principal="alice@example.com",
            roles=["Viewer"],
        ),
    )


def test_create_access_key_workspace_grant_failure_fails_the_request_and_revokes_the_key(
    client,
    access_key_workspaces_client,
    caplog,
):
    access_key_workspaces_client.create_workspace_member.side_effect = RuntimeError("workspace not found")

    with caplog.at_level(logging.WARNING):
        # The request fails instead of returning a 200 for a key that lacks the access it was
        # supposed to have -- --workspace is meant to replace a separate, failure-visible call.
        with pytest.raises(RuntimeError, match="workspace not found"):
            client.post(
                "/v2/access-keys",
                json={"name": "workspace-key", "workspaces": [{"workspace": "missing"}]},
            )

    record = next(
        record
        for record in caplog.records
        if getattr(record, "audit_event", None) == "access_key.workspace_grant_failed"
    )
    assert record.workspace == "missing"
    assert record.roles == ["Editor"]

    listed = client.get("/v2/access-keys").json()["data"]
    workspace_key_listed = next(key for key in listed if key["name"] == "workspace-key")
    assert workspace_key_listed["status"] == "REVOKED"


def test_access_key_issuer_uses_effective_principal_for_delegated_requests():
    auth_client = AuthClient(
        principal=Principal(
            id="service:evaluator",
            on_behalf_of="admin@example.com",
            on_behalf_of_email="admin@example.com",
        ),
        config=AuthConfig(enabled=True),
    )

    issuer = get_access_key_issuer(
        auth_client=auth_client,
        registry=InMemoryAccessKeyRegistry(),
        workspaces_client=AsyncMock(),
    )

    assert issuer.principal == "admin@example.com"


def test_create_service_access_key_requires_platform_admin(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=False) as has_role:
        response = client.post(
            "/v2/access-keys",
            json={"name": "otel", "service_account_id": "otel-collector"},
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Only PlatformAdmin can create service-bound Scoped Access Keys"
    has_role.assert_awaited_once_with("system", "PlatformAdmin")


def test_service_account_principal_cannot_create_or_manage_service_bound_keys_even_with_platform_admin_role(client):
    service_account_token = auth_client_context.set(
        AuthClient(
            principal=Principal(id="service-account:ci-bot"),
            config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
        )
    )
    try:
        with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True) as has_role:
            response = client.post(
                "/v2/access-keys",
                json={"name": "otel", "service_account_id": "otel-collector"},
            )
    finally:
        auth_client_context.reset(service_account_token)

    assert response.status_code == 403
    assert response.json()["detail"] == "Only PlatformAdmin can create service-bound Scoped Access Keys"
    # The service-account identity check short-circuits before any PDP role lookup.
    has_role.assert_not_awaited()


def test_privileged_service_principal_cannot_create_service_bound_keys_even_with_platform_admin_role(client):
    service_token = auth_client_context.set(
        AuthClient(
            principal=Principal(id="service:auth"),
            config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
        )
    )
    try:
        with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True) as has_role:
            response = client.post(
                "/v2/access-keys",
                json={"name": "otel", "service_account_id": "otel-collector"},
            )
    finally:
        auth_client_context.reset(service_token)

    assert response.status_code == 403
    assert response.json()["detail"] == "Only PlatformAdmin can create service-bound Scoped Access Keys"
    has_role.assert_not_awaited()


def test_create_service_access_key_is_denied_when_auth_is_disabled(client):
    with (
        patch.object(AuthClient, "auth_enabled", new_callable=PropertyMock, return_value=False),
        patch.object(AuthClient, "has_role", new_callable=AsyncMock) as has_role,
    ):
        response = client.post(
            "/v2/access-keys",
            json={"service_account_id": "otel-collector"},
        )

    assert response.status_code == 403
    has_role.assert_not_awaited()


def test_platform_admin_creates_and_manages_service_access_key(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post(
            "/v2/access-keys",
            json={"name": "otel", "service_account_id": "otel-collector"},
        )

        assert response.status_code == 200
        created = response.json()
        assert created["principal"] == "service-account:otel-collector"
        assert created["entity_type"] == "SERVICE_ACCOUNT"

        listed = client.get("/v2/access-keys")
        assert listed.status_code == 200
        assert listed.json()["data"][0]["principal"] == "service-account:otel-collector"
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 200


def test_demoted_platform_admin_lists_personal_key_but_not_service_key(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        personal = client.post("/v2/access-keys", json={"name": "personal"}).json()
        service = client.post(
            "/v2/access-keys",
            json={"name": "otel", "service_account_id": "otel-collector"},
        ).json()

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=False):
        response = client.get("/v2/access-keys")

    assert response.status_code == 200
    listed_jtis = {key["jti"] for key in response.json()["data"]}
    assert personal["jti"] in listed_jtis
    assert service["jti"] not in listed_jtis


def test_service_key_creator_cannot_revoke_after_losing_platform_admin(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        created = client.post(
            "/v2/access-keys",
            json={"name": "otel", "service_account_id": "otel-collector"},
        ).json()

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=False):
        revoked = client.delete(f"/v2/access-keys/{created['jti']}")

    assert revoked.status_code == 404


def test_platform_admin_can_revoke_service_key_created_by_a_different_admin(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        created = client.post(
            "/v2/access-keys",
            json={"name": "otel", "service_account_id": "otel-collector"},
        ).json()

        # A second PlatformAdmin, distinct from the creator, manages the same registry.
        other_admin_token = auth_client_context.set(
            AuthClient(
                principal=Principal(id="bob@example.com", email="bob@example.com"),
                config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
            )
        )
        try:
            # Listing stays scoped to the caller's own keys (see PersistentAccessKeyIssuer.list_async);
            # a PlatformAdmin manages another admin's service-bound key by its jti directly.
            revoked = client.delete(f"/v2/access-keys/{created['jti']}")
        finally:
            auth_client_context.reset(other_admin_token)

    assert revoked.status_code == 200
    assert revoked.json()["revoked"] is True


def test_platform_admin_rotates_service_key_and_preserves_binding_and_expiration(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "otel-collector",
                "expires_in_seconds": 3600,
            },
        ).json()

        other_admin_token = auth_client_context.set(
            AuthClient(
                principal=Principal(id="bob@example.com", email="bob@example.com"),
                config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
            )
        )
        try:
            rotated = client.post(f"/v2/access-keys/{created['jti']}/rotate")
        finally:
            auth_client_context.reset(other_admin_token)

    assert rotated.status_code == 200
    new_key = rotated.json()["new_key"]
    assert new_key["principal"] == "service-account:otel-collector"
    assert new_key["entity_type"] == "SERVICE_ACCOUNT"

    registry = client.app.dependency_overrides[get_access_key_registry]()
    persisted_successor = registry.keys[new_key["jti"]]
    assert persisted_successor.principal == "service-account:otel-collector"
    assert persisted_successor.entity_type == "SERVICE_ACCOUNT"
    old_key = registry.keys[created["jti"]]
    assert old_key.expires_at - old_key.created_at == timedelta(seconds=3600)
    assert persisted_successor.expires_at - persisted_successor.created_at == timedelta(seconds=3600)


def test_service_key_creator_cannot_rotate_after_losing_platform_admin(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        created = client.post(
            "/v2/access-keys",
            json={"name": "otel", "service_account_id": "otel-collector"},
        ).json()

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=False):
        rotated = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert rotated.status_code == 404


def test_platform_admin_cannot_revoke_another_admins_personal_access_key(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        created = client.post("/v2/access-keys", json={"name": "personal"}).json()

        other_admin_token = auth_client_context.set(
            AuthClient(
                principal=Principal(id="bob@example.com", email="bob@example.com"),
                config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
            )
        )
        try:
            revoked = client.delete(f"/v2/access-keys/{created['jti']}")
        finally:
            auth_client_context.reset(other_admin_token)

    assert revoked.status_code == 404


def test_create_and_revoke_emit_actor_aware_audit_logs(client, caplog):
    with caplog.at_level(logging.INFO, logger="nmp.core.auth.app.access_keys"):
        created = client.post(
            "/v2/access-keys",
            json={"name": "ci-intake", "description": "CI intake automation"},
        ).json()
        response = client.delete(f"/v2/access-keys/{created['jti']}")
        repeat_response = client.delete(f"/v2/access-keys/{created['jti']}")

    assert response.status_code == 200
    assert repeat_response.status_code == 200
    events = {record.audit_event: record for record in caplog.records if hasattr(record, "audit_event")}
    assert events["access_key.created"].actor_principal == "alice@example.com"
    assert events["access_key.created"].access_key_jti == created["jti"]
    assert events["access_key.revoked"].actor_principal == "alice@example.com"
    assert events["access_key.revoked"].access_key_jti == created["jti"]
    assert not events["access_key.revoked"].access_key_already_revoked
    assert events["access_key.revoke_noop"].actor_principal == "alice@example.com"
    assert events["access_key.revoke_noop"].access_key_jti == created["jti"]
    assert events["access_key.revoke_noop"].access_key_already_revoked
    assert created["token"] not in caplog.text
    assert "CI intake automation" not in caplog.text


def test_suspend_and_unsuspend_emit_actor_aware_audit_logs(client, caplog):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    jti = created["jti"]

    with caplog.at_level(logging.INFO, logger="nmp.core.auth.app.access_keys"):
        client.post(f"/v2/access-keys/{jti}/suspend")
        client.post(f"/v2/access-keys/{jti}/suspend")
        client.post(f"/v2/access-keys/{jti}/unsuspend")
        client.post(f"/v2/access-keys/{jti}/unsuspend")

    events = {record.audit_event: record for record in caplog.records if hasattr(record, "audit_event")}
    assert set(events) >= {
        "access_key.suspended",
        "access_key.suspend_noop",
        "access_key.unsuspended",
        "access_key.unsuspend_noop",
    }
    for event in events.values():
        assert event.actor_principal == "alice@example.com"
        assert event.access_key_jti == jti
    assert events["access_key.suspended"].access_key_state_changed
    assert not events["access_key.suspend_noop"].access_key_state_changed
    assert events["access_key.unsuspended"].access_key_state_changed
    assert not events["access_key.unsuspend_noop"].access_key_state_changed


@pytest.mark.asyncio
async def test_in_memory_access_key_registry_reports_expired_status() -> None:
    registry = InMemoryAccessKeyRegistry()
    await registry.add(
        AccessKeyCreateResponse(
            jti="ak_expired",
            name="expired-key",
            token="signed.jwt.token",
            token_type="Bearer",
            principal="alice@example.com",
            created_at=datetime.now(tz=UTC) - timedelta(hours=2),
            expires_at=datetime.now(tz=UTC) - timedelta(hours=1),
            description=None,
            status="ACTIVE",
            issuer="http://testserver/apis/auth",
            audiences=["nemo-platform-access-key"],
        )
    )

    result = await registry.list_for_principal("alice@example.com", page=1, page_size=100)

    assert result.data[0].status == "EXPIRED"
    assert not await registry.is_active("ak_expired", "alice@example.com")


def test_create_access_key_allows_unnamed_tokens(client):
    response = client.post("/v2/access-keys", json={"expires_in_seconds": 3600})

    assert response.status_code == 200
    body = response.json()
    assert body["jti"].startswith("ak_")
    assert body["name"] is None
    assert body["token"].count(".") == 2


def test_rotate_discards_successor_when_begin_rotation_fails(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.begin_rotation_error = AccessKeyStateConflictError("simulated rotation race")

    response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 409
    assert response.json()["detail"] == "simulated rotation race"
    assert len(registry.discarded) == 1
    successor_jti = next(iter(registry.discarded))
    assert successor_jti not in registry.keys


def test_rotate_discards_successor_when_concurrent_rotation_wins(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.mark_rotating_before_begin_rotation_error = True
    registry.begin_rotation_error = AccessKeyStateConflictError("simulated concurrent rotation")

    response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 409
    assert len(registry.discarded) == 1
    successor_jti = next(iter(registry.discarded))
    assert successor_jti not in registry.keys
    assert created["jti"] in registry.rotating


def test_rotate_keeps_successor_when_begin_rotation_committed_before_raising(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.begin_rotation_commits_before_error = True
    registry.begin_rotation_error = RuntimeError("simulated post-commit transport failure")

    response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 200
    body = response.json()
    assert body["previous_status"] == "ROTATING"
    assert body["grace_period_expires_at"] is not None
    assert body["new_key"]["jti"] in registry.keys
    assert registry.discarded == set()
    listing = {entry["jti"]: entry for entry in client.get("/v2/access-keys").json()["data"]}
    assert listing[created["jti"]]["grace_period_expires_at"] == body["grace_period_expires_at"]


def test_rotate_discards_successor_when_ambiguous_failure_reconciles_to_a_different_concurrent_winner(client):
    # An ambiguous begin_rotation failure that reconciles to ROTATING must only be
    # treated as *this* request's success if it's paired with *this* request's own
    # successor. Here the reconciled record is ROTATING but for a different
    # (concurrently-raced) successor, so this request's own successor must be
    # discarded rather than misattributed as the one that committed.
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.begin_rotation_commits_before_error = True
    registry.begin_rotation_successor_override = "ak_" + "9" * 32
    registry.begin_rotation_error = RuntimeError("simulated post-commit transport failure")

    with pytest.raises(RuntimeError, match="simulated post-commit transport failure"):
        client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert len(registry.discarded) == 1
    successor_jti = next(iter(registry.discarded))
    assert successor_jti not in registry.keys
    assert registry.rotating_successor[created["jti"]] == "ak_" + "9" * 32


def test_rotate_reports_success_when_reconciliation_finds_own_successor_already_finalized(client):
    # An ambiguous begin_rotation failure that reconciles to a *matching* successor
    # confirms this request's write committed, even if the record has since moved
    # on past ROTATING (e.g. reconciliation itself was delayed past the grace
    # deadline, or a manual revoke landed first) -- report the real status rather
    # than discarding a successor this request is confirmed to own.
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.begin_rotation_commits_before_error = True
    registry.begin_rotation_error = RuntimeError("simulated post-commit transport failure")
    registry.get_status_status_override = "REVOKED"

    response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 200
    body = response.json()
    assert body["previous_status"] == "REVOKED"
    assert body["grace_period_seconds"] == 0
    assert body["grace_period_expires_at"] is None
    assert body["new_key"]["jti"] in registry.keys
    assert registry.discarded == set()


def test_rotate_discards_persisted_successor_when_add_raises_after_commit(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.add_commits_before_error = True
    registry.add_error = RuntimeError("simulated post-commit persistence failure")

    with pytest.raises(RuntimeError, match="simulated post-commit persistence failure"):
        client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert len(registry.discarded) == 1
    successor_jti = next(iter(registry.discarded))
    assert successor_jti not in registry.keys
    assert created["jti"] in registry.keys


def test_rotate_discards_successor_for_pre_write_read_failure_even_if_old_key_is_rotating(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    registry = client.app.dependency_overrides[get_access_key_registry]()
    error = RuntimeError("simulated transport failure during pre-write read")
    registry.begin_rotation_pre_write_error = error
    registry.mark_rotating_before_begin_rotation_error = True

    with pytest.raises(RuntimeError, match="simulated transport failure during pre-write read") as exc_info:
        client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert exc_info.value is error
    assert created["jti"] in registry.rotating
    assert len(registry.discarded) == 1
    successor_jti = next(iter(registry.discarded))
    assert successor_jti not in registry.keys


def test_create_access_key_defaults_expiration_when_omitted(client):
    response = client.post("/v2/access-keys", json={"name": "ci-intake"})

    assert response.status_code == 200
    body = response.json()
    assert body["jti"].startswith("ak_")
    assert body["name"] == "ci-intake"
    assert body["expires_at"] is not None
    assert body["token"].count(".") == 2


def test_create_access_key_rejects_explicit_null_expiration_when_max_configured(client):
    response = client.post(
        "/v2/access-keys",
        json={"name": "bad-request", "expires_in_seconds": None},
    )

    assert response.status_code == 400
    assert "expires_in_seconds=null requires auth.access_keys.max_expires_in_seconds" in response.json()["detail"]


def test_create_access_key_accepts_optional_expiration(client):
    response = client.post("/v2/access-keys", json={"name": "short-lived", "expires_in_seconds": 60})

    assert response.status_code == 200
    assert response.json()["expires_at"] is not None


def test_create_access_key_is_disabled_by_default(disabled_client):
    response = disabled_client.post("/v2/access-keys", json={})

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Scoped Access Keys are not enabled"
    assert body["code"] == "access_keys_disabled"


def test_create_service_access_key_is_disabled_before_role_check(disabled_client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock) as has_role:
        response = disabled_client.post(
            "/v2/access-keys",
            json={"service_account_id": "otel-collector"},
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Scoped Access Keys are not enabled",
        "code": "access_keys_disabled",
    }
    has_role.assert_not_awaited()


def test_create_access_key_is_explicitly_not_implemented(client):
    class NotImplementedIssuer:
        async def create_async(self, request):
            raise AccessKeyOperationNotImplementedError("Scoped Access Key creation is not implemented")

    client.app.dependency_overrides[get_access_key_issuer] = lambda: NotImplementedIssuer()
    try:
        response = client.post("/v2/access-keys", json={"name": "ci-intake"})
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 501
    assert response.json()["detail"] == "Scoped Access Key creation is not implemented"


def test_create_access_key_does_not_misclassify_unexpected_runtime_error(client):
    class BrokenIssuer:
        async def create_async(self, request):
            raise RuntimeError("entity storage unavailable")

    client.app.dependency_overrides[get_access_key_issuer] = lambda: BrokenIssuer()
    try:
        with pytest.raises(RuntimeError, match="entity storage unavailable"):
            client.post("/v2/access-keys", json={})
    finally:
        client.app.dependency_overrides.clear()


def test_list_access_keys_is_disabled_by_default(disabled_client):
    response = disabled_client.get("/v2/access-keys")

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Scoped Access Keys are not enabled"
    assert body["code"] == "access_keys_disabled"


def test_revoke_access_key_is_disabled_by_default(disabled_client):
    response = disabled_client.delete("/v2/access-keys/ak_" + "a" * 32)

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Scoped Access Keys are not enabled"
    assert body["code"] == "access_keys_disabled"


@pytest.mark.parametrize("action", ["suspend", "unsuspend"])
def test_suspension_actions_are_disabled_by_default(disabled_client, action):
    response = disabled_client.post(f"/v2/access-keys/ak_{'a' * 32}/{action}")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Scoped Access Keys are not enabled",
        "code": "access_keys_disabled",
    }


def test_access_key_specific_jwks_route_does_not_accept_get(client):
    response = client.get("/v2/access-keys/jwks")

    assert response.status_code == 405
    assert "DELETE" in response.headers["allow"]


def test_access_key_specific_jwks_route_is_not_in_openapi(client):
    assert "/v2/access-keys/jwks" not in client.app.openapi()["paths"]


def test_access_key_lifecycle_openapi_documents_error_responses(client):
    openapi = client.app.openapi()

    create_responses = openapi["paths"]["/v2/access-keys"]["post"]["responses"]
    assert create_responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyCreateResponse"
    }
    assert create_responses["400"]["description"] == "Scoped Access Key creation error"
    assert create_responses["400"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert create_responses["403"]["description"] == "Service-bound Scoped Access Keys require PlatformAdmin"
    assert create_responses["403"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert create_responses["404"]["description"] == "Scoped Access Keys are not enabled"
    assert create_responses["404"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert create_responses["409"]["description"] == "Concurrent access-key update conflict"
    assert create_responses["409"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert create_responses["501"]["description"] == "Not Implemented"
    assert create_responses["501"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyNotImplementedErrorResponse"
    }
    request_schema = openapi["components"]["schemas"]["AccessKeyCreateRequest"]
    assert request_schema["properties"]["name"]["nullable"] is True
    assert request_schema["properties"]["description"]["nullable"] is True
    assert request_schema["properties"]["expires_in_seconds"]["nullable"] is True
    metadata_schema = openapi["components"]["schemas"]["AccessKeyMetadataResponse"]
    assert metadata_schema["properties"]["name"]["nullable"] is True
    assert metadata_schema["properties"]["description"]["nullable"] is True
    assert metadata_schema["properties"]["audiences"]["uniqueItems"] is True
    assert metadata_schema["properties"]["expires_at"]["nullable"] is True
    assert metadata_schema["properties"]["grace_period_expires_at"]["nullable"] is True
    assert metadata_schema["properties"]["last_used_at"]["nullable"] is True
    create_response_schema = openapi["components"]["schemas"]["AccessKeyCreateResponse"]
    assert create_response_schema["properties"]["audiences"]["uniqueItems"] is True
    list_schema = openapi["components"]["schemas"]["AccessKeyListResponse"]
    assert list_schema["properties"]["has_more"]["default"] is False
    revoke_schema = openapi["components"]["schemas"]["AccessKeyRevokeResponse"]
    assert set(revoke_schema["required"]) == {"jti", "revoked"}
    assert openapi["components"]["schemas"]["AccessKeyMetadataResponse"]["properties"]["status"]["enum"] == [
        "ACTIVE",
        "EXPIRED",
        "REVOKED",
        "SUSPENDED",
        "ROTATING",
    ]
    status_change_schema = openapi["components"]["schemas"]["AccessKeyStatusChangeResponse"]
    assert set(status_change_schema["required"]) == {"jti", "status", "changed"}
    assert status_change_schema["properties"]["status"]["enum"] == ["ACTIVE", "EXPIRED", "SUSPENDED"]
    error_code_schema = openapi["components"]["schemas"]["AccessKeyErrorResponse"]["properties"]["code"]
    assert error_code_schema["nullable"] is True
    assert error_code_schema["anyOf"][0]["const"] == "access_keys_disabled"
    assert error_code_schema["description"] == (
        "Set to access_keys_disabled when the Scoped Access Key feature is disabled."
    )

    list_operation = openapi["paths"]["/v2/access-keys"]["get"]
    list_parameters = {parameter["name"]: parameter for parameter in list_operation["parameters"]}
    assert list_parameters["page"]["schema"] == {
        "type": "integer",
        "minimum": 1,
        "default": 1,
        "title": "Page",
        "description": "Page number to retrieve.",
    }
    assert list_parameters["page_size"]["schema"] == {
        "type": "integer",
        "maximum": 100,
        "minimum": 1,
        "default": 100,
        "title": "Page Size",
        "description": "Number of keys to retrieve per page.",
    }
    list_responses = list_operation["responses"]
    assert list_responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyListResponse"
    }
    assert list_responses["404"]["description"] == "Scoped Access Keys are not enabled"
    assert list_responses["404"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert list_responses["501"]["description"] == "Not Implemented"
    assert list_responses["501"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyNotImplementedErrorResponse"
    }

    for action in ["suspend", "unsuspend"]:
        responses = openapi["paths"][f"/v2/access-keys/{{jti}}/{action}"]["post"]["responses"]
        assert responses["200"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AccessKeyStatusChangeResponse"
        }
        assert responses["404"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AccessKeyErrorResponse"
        }
        assert responses["409"]["description"] == "Invalid or concurrent access-key state transition"
        assert responses["409"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AccessKeyErrorResponse"
        }

    rotate_responses = openapi["paths"]["/v2/access-keys/{jti}/rotate"]["post"]["responses"]
    assert rotate_responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyRotateResponse"
    }
    assert rotate_responses["400"]["description"] == "Scoped Access Key rotation error"
    assert rotate_responses["409"]["description"] == "Invalid or concurrent access-key state transition"
    rotate_schema = openapi["components"]["schemas"]["AccessKeyRotateResponse"]
    assert rotate_schema["properties"]["grace_period_expires_at"]["nullable"] is True
    assert set(rotate_schema["required"]) == {
        "new_key",
        "previous_jti",
        "previous_status",
        "grace_period_seconds",
    }

    revoke_operation = openapi["paths"]["/v2/access-keys/{jti}"]["delete"]
    jti_parameter = next(parameter for parameter in revoke_operation["parameters"] if parameter["name"] == "jti")
    assert jti_parameter["schema"]["pattern"] == "^ak_[0-9a-f]{32}$"
    assert jti_parameter["description"] == "Stable JWT ID of the Scoped Access Key for the lifecycle operation."
    revoke_responses = revoke_operation["responses"]
    assert revoke_responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyRevokeResponse"
    }
    assert revoke_responses["404"]["description"] == "Scoped Access Keys are not enabled or the key was not found"
    assert revoke_responses["404"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert revoke_responses["409"]["description"] == "Concurrent access-key update conflict"
    assert revoke_responses["409"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert revoke_responses["501"]["description"] == "Not Implemented"
    assert revoke_responses["501"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyNotImplementedErrorResponse"
    }


def test_list_access_keys_returns_current_principals_persisted_keys(client):
    created = client.post(
        "/v2/access-keys",
        json={"name": "ci-intake", "description": "CI intake automation"},
    ).json()

    response = client.get("/v2/access-keys")

    assert response.status_code == 200
    assert response.json()["data"] == [
        {
            "jti": created["jti"],
            "name": "ci-intake",
            "principal": "alice@example.com",
            "entity_type": "USER",
            "created_at": created["created_at"],
            "expires_at": created["expires_at"],
            "grace_period_expires_at": None,
            "last_used_at": None,
            "description": "CI intake automation",
            "status": "ACTIVE",
            "issuer": "http://testserver/apis/auth",
            "audiences": ["nemo-platform-access-key"],
            "scope": [],
        }
    ]
    assert response.json()["has_more"] is False


def test_list_access_keys_supports_pagination(client):
    first = client.post("/v2/access-keys", json={"name": "first"}).json()
    second = client.post("/v2/access-keys", json={"name": "second"}).json()

    first_page = client.get("/v2/access-keys", params={"page": 1, "page_size": 1})
    second_page = client.get("/v2/access-keys", params={"page": 2, "page_size": 1})

    assert first_page.status_code == 200
    assert len(first_page.json()["data"]) == 1
    assert first_page.json()["has_more"] is True
    assert second_page.status_code == 200
    assert len(second_page.json()["data"]) == 1
    assert second_page.json()["has_more"] is False
    # Both keys appear exactly once across the two pages (order depends on issued_at).
    all_jtis = {
        key["jti"] for page_data in [first_page.json()["data"], second_page.json()["data"]] for key in page_data
    }
    assert all_jtis == {first["jti"], second["jti"]}


def test_revoke_access_key_marks_key_revoked_in_listing(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()

    response = client.delete(f"/v2/access-keys/{created['jti']}")

    assert response.status_code == 200
    assert response.json() == {"jti": created["jti"], "revoked": True}
    listed = client.get("/v2/access-keys").json()["data"]
    assert len(listed) == 1
    assert listed[0]["jti"] == created["jti"]
    assert listed[0]["status"] == "REVOKED"

    repeat_response = client.delete(f"/v2/access-keys/{created['jti']}")

    assert repeat_response.status_code == 200
    assert repeat_response.json() == {"jti": created["jti"], "revoked": False}


def test_revoke_access_key_returns_not_found_for_unknown_key(client):
    unknown_jti = "ak_" + "0" * 32
    response = client.delete(f"/v2/access-keys/{unknown_jti}")

    assert response.status_code == 404
    assert response.json()["detail"] == f"Scoped Access Key {unknown_jti} was not found"


def test_revoke_access_key_rejects_malformed_jti(client):
    response = client.delete("/v2/access-keys/ak_tooshort")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert detail[0]["type"] == "string_pattern_mismatch"


def test_suspend_and_unsuspend_access_key(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    jti = created["jti"]

    suspended = client.post(f"/v2/access-keys/{jti}/suspend")
    repeated_suspend = client.post(f"/v2/access-keys/{jti}/suspend")

    assert suspended.status_code == 200
    assert suspended.json() == {"jti": jti, "status": "SUSPENDED", "changed": True}
    assert repeated_suspend.json() == {"jti": jti, "status": "SUSPENDED", "changed": False}
    assert client.get("/v2/access-keys").json()["data"][0]["status"] == "SUSPENDED"

    unsuspended = client.post(f"/v2/access-keys/{jti}/unsuspend")
    repeated_unsuspend = client.post(f"/v2/access-keys/{jti}/unsuspend")

    assert unsuspended.status_code == 200
    assert unsuspended.json() == {"jti": jti, "status": "ACTIVE", "changed": True}
    assert repeated_unsuspend.json() == {"jti": jti, "status": "ACTIVE", "changed": False}
    assert client.get("/v2/access-keys").json()["data"][0]["status"] == "ACTIVE"


def test_unsuspend_reports_expired_status_for_key_that_expired_while_suspended(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake", "expires_in_seconds": 3600}).json()
    jti = created["jti"]
    assert client.post(f"/v2/access-keys/{jti}/suspend").status_code == 200

    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.keys[jti].expires_at = datetime.now(tz=UTC) - timedelta(hours=1)

    unsuspended = client.post(f"/v2/access-keys/{jti}/unsuspend")

    assert unsuspended.status_code == 200
    assert unsuspended.json() == {"jti": jti, "status": "EXPIRED", "changed": False}
    assert client.get("/v2/access-keys").json()["data"][0]["status"] == "EXPIRED"


@pytest.mark.parametrize("action", ["suspend", "unsuspend"])
def test_suspension_transition_reports_expired_at_expiration_boundary(client, action):
    created = client.post("/v2/access-keys", json={"name": "ci-intake", "expires_in_seconds": 3600}).json()
    jti = created["jti"]
    if action == "unsuspend":
        assert client.post(f"/v2/access-keys/{jti}/suspend").status_code == 200

    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.keys[jti].expires_at = datetime.now(tz=UTC)

    response = client.post(f"/v2/access-keys/{jti}/{action}")

    assert response.status_code == 200
    assert response.json() == {"jti": jti, "status": "EXPIRED", "changed": False}


@pytest.mark.parametrize("action", ["suspend", "unsuspend"])
def test_revoked_access_key_cannot_be_suspended_or_unsuspended(client, action):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]
    assert client.delete(f"/v2/access-keys/{jti}").status_code == 200

    response = client.post(f"/v2/access-keys/{jti}/{action}")

    assert response.status_code == 409
    assert response.json()["detail"] == f"Revoked Scoped Access Key {jti} cannot be {action}ed"


@pytest.mark.parametrize("action", ["suspend", "unsuspend"])
def test_suspension_action_returns_not_found_for_unknown_key(client, action):
    unknown_jti = "ak_" + "0" * 32

    response = client.post(f"/v2/access-keys/{unknown_jti}/{action}")

    assert response.status_code == 404
    assert response.json()["detail"] == f"Scoped Access Key {unknown_jti} was not found"


def test_rotate_access_key_mints_successor_and_starts_grace_period(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake", "description": "CI intake automation"}).json()
    jti = created["jti"]

    response = client.post(f"/v2/access-keys/{jti}/rotate")

    assert response.status_code == 200
    body = response.json()
    assert body["previous_jti"] == jti
    assert body["previous_status"] == "ROTATING"
    assert body["grace_period_seconds"] > 0
    assert body["grace_period_expires_at"] is not None
    new_key = body["new_key"]
    assert new_key["jti"] != jti
    assert new_key["jti"].startswith("ak_")
    assert new_key["token"].count(".") == 2
    assert new_key["name"] == "ci-intake"
    assert new_key["description"] == "CI intake automation"

    listing = {entry["jti"]: entry for entry in client.get("/v2/access-keys").json()["data"]}
    assert listing[jti]["status"] == "ROTATING"
    assert listing[jti]["grace_period_expires_at"] == body["grace_period_expires_at"]
    assert listing[new_key["jti"]]["status"] == "ACTIVE"
    assert listing[new_key["jti"]]["grace_period_expires_at"] is None


def test_rotate_access_key_preserves_scope(client):
    created = client.post("/v2/access-keys", json={"name": "intake-worker", "scope": ["intake", "entities"]}).json()
    jti = created["jti"]

    response = client.post(f"/v2/access-keys/{jti}/rotate")

    assert response.status_code == 200
    new_key = response.json()["new_key"]
    assert new_key["scope"] == ["intake", "entities"]

    listing = {entry["jti"]: entry for entry in client.get("/v2/access-keys").json()["data"]}
    assert listing[new_key["jti"]]["scope"] == ["intake", "entities"]


def test_rotate_access_key_keeps_unscoped_successor_unscoped(client):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    jti = created["jti"]

    response = client.post(f"/v2/access-keys/{jti}/rotate")

    assert response.status_code == 200
    new_key = response.json()["new_key"]
    assert new_key["scope"] == []


def test_rotate_access_key_accepts_explicit_grace_period_seconds(client):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]

    response = client.post(f"/v2/access-keys/{jti}/rotate", json={"grace_period_seconds": 3600})

    assert response.status_code == 200
    body = response.json()
    assert body["grace_period_seconds"] <= 3600
    assert body["grace_period_seconds"] > 0


def test_rotate_access_key_rejects_grace_period_seconds_above_configured_max(client):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]
    max_grace_period_seconds = AccessKeyConfig().max_rotation_grace_period_seconds

    response = client.post(
        f"/v2/access-keys/{jti}/rotate",
        json={"grace_period_seconds": max_grace_period_seconds + 1},
    )

    assert response.status_code == 400
    assert "max_rotation_grace_period_seconds" in response.json()["detail"]


def test_rotate_discards_successor_when_begin_rotation_exhausts_conflicts_and_concurrent_winner_is_visible(
    client,
):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.mark_rotating_before_begin_rotation_error = True
    registry.begin_rotation_error = EntityConflictError("entity version changed")

    response = client.post(f"/v2/access-keys/{jti}/rotate")

    assert response.status_code == 409
    assert response.json()["detail"] == "Concurrent update conflict; retry."
    assert len(registry.discarded) == 1
    discarded_jti = next(iter(registry.discarded))
    assert discarded_jti not in registry.keys
    assert jti in registry.rotating


def test_rotated_out_key_stays_active_for_auth_during_grace_period(client):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]

    client.post(f"/v2/access-keys/{jti}/rotate")

    registry = client.app.dependency_overrides[get_access_key_registry]()
    assert asyncio.run(registry.is_active(jti, "alice@example.com"))


def test_rotated_out_key_becomes_revoked_after_grace_period_elapses(client):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]

    client.post(f"/v2/access-keys/{jti}/rotate")
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.rotating[jti] = datetime.now(tz=UTC) - timedelta(seconds=1)

    assert not asyncio.run(registry.is_active(jti, "alice@example.com"))
    listing = {entry["jti"]: entry for entry in client.get("/v2/access-keys").json()["data"]}
    assert listing[jti]["status"] == "REVOKED"


def test_rotated_out_key_can_still_be_manually_revoked(client):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]
    client.post(f"/v2/access-keys/{jti}/rotate")

    response = client.delete(f"/v2/access-keys/{jti}")

    assert response.status_code == 200
    assert response.json() == {"jti": jti, "revoked": True}


def test_already_rotating_access_key_cannot_be_rotated_again(client):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]
    client.post(f"/v2/access-keys/{jti}/rotate")

    response = client.post(f"/v2/access-keys/{jti}/rotate")

    assert response.status_code == 409
    assert f"Scoped Access Key {jti} must be ACTIVE to rotate" in response.json()["detail"]


@pytest.mark.parametrize("action", ["suspend", "revoke"])
def test_non_active_access_key_cannot_be_rotated(client, action):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]
    method = client.delete if action == "revoke" else client.post
    path = f"/v2/access-keys/{jti}" if action == "revoke" else f"/v2/access-keys/{jti}/{action}"
    assert method(path).status_code == 200

    response = client.post(f"/v2/access-keys/{jti}/rotate")

    assert response.status_code == 409


def test_rotate_access_key_returns_not_found_for_unknown_key(client):
    unknown_jti = "ak_" + "0" * 32

    response = client.post(f"/v2/access-keys/{unknown_jti}/rotate")

    assert response.status_code == 404
    assert response.json()["detail"] == f"Scoped Access Key {unknown_jti} was not found"


def test_rotate_access_key_discards_successor_when_begin_rotation_reports_not_found(client):
    created = client.post("/v2/access-keys", json={}).json()
    jti = created["jti"]
    registry = client.app.dependency_overrides[get_access_key_registry]()
    registry.begin_rotation_error = AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")

    response = client.post(f"/v2/access-keys/{jti}/rotate")

    assert response.status_code == 404
    assert response.json()["detail"] == f"Scoped Access Key {jti} was not found"
    assert len(registry.discarded) == 1
    assert registry.discarded.isdisjoint(registry.keys)
    assert set(registry.keys) == {jti}


def test_rotate_access_key_is_disabled_by_default(disabled_client):
    response = disabled_client.post(f"/v2/access-keys/ak_{'a' * 32}/rotate")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Scoped Access Keys are not enabled",
        "code": "access_keys_disabled",
    }


def test_rotate_and_new_key_emit_actor_aware_audit_logs(client, caplog):
    created = client.post("/v2/access-keys", json={"name": "ci-intake"}).json()
    jti = created["jti"]

    with caplog.at_level(logging.INFO, logger="nmp.core.auth.app.access_keys"):
        response = client.post(f"/v2/access-keys/{jti}/rotate")

    new_jti = response.json()["new_key"]["jti"]
    events = {record.audit_event: record for record in caplog.records if hasattr(record, "audit_event")}
    assert events["access_key.rotated"].actor_principal == "alice@example.com"
    assert events["access_key.rotated"].access_key_jti == jti
    assert events["access_key.rotated"].access_key_new_jti == new_jti
