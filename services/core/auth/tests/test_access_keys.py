# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, call, patch

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from nemo_helix_plugin.auth.access_keys.issuer import AccessKeyOperationNotImplementedError
from nemo_helix_plugin.auth.access_keys.types import AccessKeyCreateResponse
from nemo_helix_plugin.client.errors import NemoTransportError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, UpdateWorkspaceMemberRequest
from nhx.common.api.common import PaginatedResult, PaginationData
from nhx.common.auth.access_keys import AccessKeyValidationError
from nhx.common.auth.client import AuthClient
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.models import Principal
from nhx.common.auth.token_claims import TokenClaims
from nhx.common.auth.token_resolver import ResolvedBearerToken
from nhx.common.config import AuthConfig
from nhx.common.config.base import AccessKeyConfig, TokenSigningConfig
from nhx.common.entities import EntityConflictError, EntityStoreError
from nhx.core.auth.api.v2.access_keys.endpoints import get_access_key_issuer, get_workspaces_client, router
from nhx.core.auth.app.access_key_credentials import AccessKeyCredentialAdapter
from nhx.core.auth.app.access_keys import (
    AccessKeyNotAdministeredError,
    AccessKeyNotFoundError,
    AccessKeyRegistry,
    AccessKeyStateConflictError,
    LiveServiceAccountKey,
    ServiceAccountAccess,
    get_access_key_registry,
)
from nhx.core.auth.entities import AccessKeyEntity
from nhx.core.entities.app.repository import AccountIdentityUnavailableError

# When the fake workspaces were created, unless a test says otherwise.
WORKSPACES_CREATED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class InMemoryAccessKeyRegistry:
    def __init__(self):
        self.keys = {}
        self.owners = {}
        self.bound = {}
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
        # principal -> workspaces it holds an active role binding in, independent of any key.
        self.bindings = {}
        # Service-account principals whose account identity is disabled.
        self.disabled_service_accounts = set()
        # Runs once right after the next add() persists, to simulate a concurrent request
        # landing between the pre-insert confinement check and the insert.
        self.after_add = None
        # Workspaces the caller holds Admin role bindings in, whatever the PDP says (it's still
        # asked per key), and the (principals, role) of each role-binding query.
        self.admin_binding_workspaces = {"team-a", "team-b", "team-c", "ci", "prod"}
        self.role_binding_queries = []
        # Service-account principal -> when it was first issued a key, if a test says.
        self.linked_at = {}
        # Workspace -> {principal: roles} granted through the workspaces client (see the client fixture).
        self.granted_members = {}

    async def add(
        self,
        key,
        *,
        owner_principal=None,
        owner_account_id=None,
        bound_workspace_ids=(),
    ):
        if self.add_error is not None and not self.add_commits_before_error:
            raise self.add_error
        self.keys[key.jti] = key
        self.owners[key.jti] = owner_principal or key.principal
        self.bound[key.jti] = list(bound_workspace_ids)
        if self.after_add is not None:
            after_add, self.after_add = self.after_add, None
            after_add(self)
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

    def _ids(self, jti):
        return self.bound.get(jti, [])

    async def _may_manage(self, jti, principal, *, admin_override=None):
        key = self.keys.get(jti)
        if key is None:
            return None
        if key.entity_type == "SERVICE_ACCOUNT":
            if admin_override is not None:
                if await admin_override(self._ids(jti)):
                    return key
                await AccessKeyRegistry.reject_partial_admin(jti, self._ids(jti), admin_override)
        elif self.owners[jti] == principal:
            return key
        return None

    async def list_for_principal(
        self,
        principal,
        *,
        page,
        page_size,
        include_service_accounts=False,
        owner_account_id=None,
        admin_override=None,
        service_key_workspace_ids=(),
    ):
        from nemo_helix_plugin.auth.access_keys.types import AccessKeyListResponse, AccessKeyMetadataResponse

        # Sort newest-first then by jti to match the real registry's `sort="-issued_at"`.
        visible = []
        for jti, key in self.keys.items():
            if key.entity_type == "SERVICE_ACCOUNT":
                fetched = self.owners[jti] == principal or set(self._ids(jti)) & set(service_key_workspace_ids)
                if include_service_accounts or (
                    fetched and admin_override is not None and await admin_override(self._ids(jti))
                ):
                    visible.append((jti, key))
            elif self.owners[jti] == principal:
                visible.append((jti, key))
        owned = sorted(visible, key=lambda item: (-item[1].created_at.timestamp(), item[0]))
        start = (page - 1) * page_size
        selected = owned[start : start + page_size]
        return AccessKeyListResponse(
            data=[
                AccessKeyMetadataResponse.model_validate(
                    key.model_dump(exclude={"token", "token_type"})
                    | {
                        "status": self._status(jti, key),
                        "bound_workspaces": self._ids(jti),
                        "bound_workspace_ids": self._ids(jti),
                        "grace_period_expires_at": (
                            self.rotating.get(jti) if self._status(jti, key) == "ROTATING" else None
                        ),
                    }
                )
                for jti, key in selected
            ],
            has_more=start + page_size < len(owned),
        )

    async def service_account_access(self, service_account_principal):
        if service_account_principal in self.disabled_service_accounts:
            raise AccountIdentityUnavailableError("disabled")
        # Roles from the memberships granted through the workspaces client, and from `bindings`.
        roles = {workspace: frozenset({"Viewer"}) for workspace in self.bindings.get(service_account_principal, ())}
        for workspace, members in self.granted_members.items():
            if service_account_principal in members:
                roles[workspace] = roles.get(workspace, frozenset()) | frozenset(members[service_account_principal])
        return ServiceAccountAccess(
            roles=roles,
            principals=(service_account_principal,),
            live_keys=tuple(
                LiveServiceAccountKey(jti=jti, bound_workspace_ids=frozenset(self._ids(jti)))
                for jti, key in self.keys.items()
                if key.principal == service_account_principal and self._status(jti, key) not in ("REVOKED", "EXPIRED")
            ),
            linked_at=self.linked_at.get(service_account_principal),
        )

    async def wildcard_roles(self, workspaces):
        return {workspace: frozenset(self.granted_members.get(workspace, {}).get("*", ())) for workspace in workspaces}

    async def is_service_account_disabled(self, service_account_principal):
        return service_account_principal in self.disabled_service_accounts

    async def workspaces_with_role(self, principals, role):
        self.role_binding_queries.append((list(principals), role))
        return set(self.admin_binding_workspaces) if role == "Admin" else set()

    async def revoke(self, jti, principal, *, admin_override=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        revoked = jti not in self.revoked
        self.revoked.add(jti)
        self.suspended.discard(jti)
        return revoked

    async def get_for_rotation(self, jti, principal, *, admin_override=None):
        # Mirrors AccessKeyRegistry.get_for_rotation, which treats a partly administered key as unknown.
        try:
            key = await self._may_manage(jti, principal, admin_override=admin_override)
        except AccessKeyNotAdministeredError:
            key = None
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

    async def unsuspend(self, jti, principal, *, admin_override=None, before_unsuspend=None):
        key = await self._may_manage(jti, principal, admin_override=admin_override)
        if key is None:
            raise AccessKeyNotFoundError(f"Scoped Access Key {jti} was not found")
        if jti in self.revoked:
            raise AccessKeyStateConflictError(f"Revoked Scoped Access Key {jti} cannot be unsuspended")
        if key.expires_at is not None and key.expires_at <= datetime.now(tz=UTC):
            return False, "EXPIRED"
        changed = jti in self.suspended
        if changed and before_unsuspend is not None:
            await before_unsuspend(self._as_entity_view(jti, key))
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
            bound_workspace_ids=self._ids(jti),
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


def _members_response(members: list) -> SimpleNamespace:
    # Matches the real client: awaiting list_workspace_members() returns a NemoResponse
    # whose sync .data() unwraps to a WorkspaceMemberListResponse, which has a plain
    # `data: list[WorkspaceMember]` attribute.
    return SimpleNamespace(data=lambda: SimpleNamespace(data=members))


@pytest.fixture
def access_key_workspaces_client():
    client = AsyncMock(spec=AsyncWorkspacesClient)
    # An unconfigured AsyncMock returns more AsyncMocks for any attribute or call, so
    # `.data()` on the default return value would itself be an unawaited coroutine rather
    # than a callable returning a list. Default to "no prior members" so tests that don't
    # care about prior state don't need to configure this every time.
    client.list_workspace_members.return_value = _members_response([])
    # Memberships granted through create_workspace_member, per workspace. Listing a workspace's
    # members returns them once there are any, and the configured return_value until then.
    client.granted_members = {}

    async def create_workspace_member(*, workspace, body):
        client.granted_members.setdefault(workspace, {})[body.principal] = list(body.roles)

    async def list_workspace_members(*, workspace):
        granted = client.granted_members.get(workspace)
        if not granted:
            return client.list_workspace_members.return_value
        return _members_response([SimpleNamespace(principal=p, roles=roles) for p, roles in granted.items()])

    client.create_workspace_member.side_effect = create_workspace_member
    client.list_workspace_members.side_effect = list_workspace_members
    # Each workspace has its own ID, "ws-id-<name>", unless a test replaces it (as when a workspace
    # is deleted and recreated under the same name) in `workspace_ids`; and was created at
    # WORKSPACES_CREATED_AT, unless a test sets another time in `created_at`.
    client.workspace_ids = {}
    client.created_at = {}

    # Workspaces the caller can see when listing them, e.g. to find the ones they administer.
    client.visible_workspaces = ["team-a", "team-b", "team-c", "ci", "prod"]

    def current_id(name):
        return client.workspace_ids.get(name, f"ws-id-{name}")

    def name_for_id(workspace_id):
        names = {current_id(name): name for name in [*client.visible_workspaces, *client.workspace_ids]}
        if workspace_id in names:
            return names[workspace_id]
        name = workspace_id.removeprefix("ws-id-")
        # Any other workspace exists under its default ID, unless it was recreated with a new one.
        return name if workspace_id.startswith("ws-id-") and name not in client.workspace_ids else None

    async def get_workspace(*, name):
        workspace_id = current_id(name)
        created_at = client.created_at.get(name, WORKSPACES_CREATED_AT)
        return SimpleNamespace(data=lambda: SimpleNamespace(id=workspace_id, name=name, created_at=created_at))

    async def list_workspaces(*, query_params=None):
        workspace_filter = json.loads((query_params or {}).get("filter") or "null")
        if workspace_filter is not None and "name" in workspace_filter:
            names = workspace_filter["name"]["$in"]
            workspaces = [SimpleNamespace(id=current_id(name), name=name) for name in names]
        elif workspace_filter is not None:
            ids = workspace_filter["id"]
            ids = ids["$in"] if isinstance(ids, dict) else [ids]
            workspaces = [
                SimpleNamespace(id=workspace_id, name=name_for_id(workspace_id))
                for workspace_id in ids
                if name_for_id(workspace_id) is not None
            ]
        else:
            workspaces = [SimpleNamespace(id=current_id(name), name=name) for name in client.visible_workspaces]

        async def items():
            for workspace in workspaces:
                yield workspace

        return SimpleNamespace(items=items)

    client.get_workspace.side_effect = get_workspace
    client.list_workspaces.side_effect = list_workspaces
    return client


@pytest.fixture
def access_key_registry():
    return InMemoryAccessKeyRegistry()


@pytest.fixture
def client(tmp_path, access_key_workspaces_client, access_key_registry):
    access_key_registry.granted_members = access_key_workspaces_client.granted_members
    config = AuthConfig(
        enabled=True,
        token_signing=TokenSigningConfig(
            issuer="http://testserver/apis/auth",
            key_id="test-access-key",
            private_key_file=str(tmp_path / "private.pem"),
        ),
        access_keys=AccessKeyConfig(
            enabled=True,
            audience="nemo-helix-access-key",
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
    app.dependency_overrides[get_access_key_registry] = lambda: access_key_registry
    app.dependency_overrides[get_workspaces_client] = lambda: access_key_workspaces_client

    token = auth_client_context.set(
        AuthClient(
            principal=Principal(id="alice@example.com", email="alice@example.com", groups=["team-ml"]),
            config=config,
        )
    )
    with patch("nhx.core.auth.api.v2.access_keys.endpoints.get_auth_config", return_value=config):
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
    with patch("nhx.core.auth.api.v2.access_keys.endpoints.get_auth_config", return_value=config):
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


def test_create_access_key_with_rotates_inherits_predecessor_scope_when_scope_omitted(client):
    prior = client.post("/v2/access-keys", json={"name": "prior", "scope": ["intake", "entities"]}).json()

    response = client.post(
        "/v2/access-keys",
        json={"name": "replacement", "rotates": prior["jti"]},
    )

    assert response.status_code == 200
    assert response.json()["scope"] == ["intake", "entities"]


def test_create_access_key_with_rotates_rejects_explicit_scope_that_differs_from_predecessor(client):
    prior = client.post("/v2/access-keys", json={"name": "prior", "scope": ["intake"]}).json()

    response = client.post(
        "/v2/access-keys",
        json={"name": "replacement", "rotates": prior["jti"], "scope": ["entities"]},
    )

    assert response.status_code == 400
    assert "rotates requires the replacement scope to match the predecessor's scope" in response.json()["detail"]
    listed = client.get("/v2/access-keys").json()["data"]
    assert next(key for key in listed if key["jti"] == prior["jti"])["status"] == "ACTIVE"


def test_create_access_key_with_rotates_rejects_explicit_scope_when_predecessor_unscoped(client):
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    response = client.post(
        "/v2/access-keys",
        json={"name": "replacement", "rotates": prior["jti"], "scope": ["entities"]},
    )

    assert response.status_code == 400
    assert "rotates requires the replacement scope to match the predecessor's scope" in response.json()["detail"]


def test_create_access_key_with_rotates_allows_explicit_scope_matching_predecessor(client):
    prior = client.post("/v2/access-keys", json={"name": "prior", "scope": ["intake", "entities"]}).json()

    response = client.post(
        "/v2/access-keys",
        # Order shouldn't matter: the check compares scope as a set.
        json={"name": "replacement", "rotates": prior["jti"], "scope": ["entities", "intake"]},
    )

    assert response.status_code == 200
    assert response.json()["scope"] == ["entities", "intake"]


def test_create_access_key_with_rotates_keeps_unscoped_predecessor_unscoped(client):
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    response = client.post(
        "/v2/access-keys",
        json={"name": "replacement", "rotates": prior["jti"]},
    )

    assert response.status_code == 200
    assert response.json()["scope"] == []


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

    # Simulate an unexpected failure revoking just the rotation target (e.g. a concurrent
    # delete), while revoking any other jti (the compensating self-revoke of the new key)
    # keeps working normally.
    from nhx.core.auth.app.access_keys import PersistentAccessKeyIssuer

    original_revoke_async = PersistentAccessKeyIssuer.revoke_async

    async def _boom_for_target_only(self, jti):
        if jti == prior["jti"]:
            raise RuntimeError("entity storage unavailable")
        return await original_revoke_async(self, jti)

    monkeypatch.setattr(
        "nhx.core.auth.app.access_keys.PersistentAccessKeyIssuer.revoke_async",
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
    # The half-created replacement was revoked rather than left dangling.
    replacement_listed = next(key for key in listed if key["name"] == "replacement")
    assert replacement_listed["status"] == "REVOKED"


def test_create_access_key_rotation_revoke_failure_rolls_back_granted_workspace_memberships(
    client, access_key_workspaces_client, caplog, monkeypatch
):
    # Workspace grant succeeds, but revoking the predecessor then fails, so the grant on the
    # now-revoked replacement must be undone too.
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    from nhx.core.auth.app.access_keys import PersistentAccessKeyIssuer

    original_revoke_async = PersistentAccessKeyIssuer.revoke_async

    async def _boom_for_target_only(self, jti):
        if jti == prior["jti"]:
            raise RuntimeError("entity storage unavailable")
        return await original_revoke_async(self, jti)

    monkeypatch.setattr(
        "nhx.core.auth.app.access_keys.PersistentAccessKeyIssuer.revoke_async",
        _boom_for_target_only,
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="entity storage unavailable"):
            client.post(
                "/v2/access-keys",
                json={
                    "name": "replacement",
                    "rotates": prior["jti"],
                    "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
                },
            )

    access_key_workspaces_client.create_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        body=CreateWorkspaceMemberRequest(principal="alice@example.com", roles=["Editor"]),
    )
    access_key_workspaces_client.delete_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        principal_id="alice@example.com",
    )

    listed = client.get("/v2/access-keys").json()["data"]
    prior_listed = next(key for key in listed if key["jti"] == prior["jti"])
    assert prior_listed["status"] == "ACTIVE"
    replacement_listed = next(key for key in listed if key["name"] == "replacement")
    assert replacement_listed["status"] == "REVOKED"


def test_create_access_key_rotation_revoke_failure_logs_when_compensation_also_fails(client, caplog, monkeypatch):
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    async def _boom(self, jti):
        raise RuntimeError("entity storage unavailable")

    monkeypatch.setattr(
        "nhx.core.auth.app.access_keys.PersistentAccessKeyIssuer.revoke_async",
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
        # Fails instead of returning a 200 for a key missing its requested access.
        with pytest.raises(RuntimeError, match="workspace not found"):
            client.post(
                "/v2/access-keys",
                json={"name": "workspace-key", "workspaces": [{"workspace": "missing", "roles": ["Editor"]}]},
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


def test_create_access_key_partial_workspace_grant_failure_rolls_back_the_successful_grants(
    client,
    access_key_workspaces_client,
    caplog,
):
    # First grant (team-a) succeeds; second (team-b) reports a failure. Rollback treats team-b
    # as uncertain, since the mutation may have committed before the error, and undoes both
    # attempts in reverse order.
    access_key_workspaces_client.create_workspace_member.side_effect = [None, RuntimeError("workspace not found")]

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="workspace not found"):
            client.post(
                "/v2/access-keys",
                json={
                    "name": "workspace-key",
                    "workspaces": [
                        {"workspace": "team-a", "roles": ["Editor"]},
                        {"workspace": "team-b", "roles": ["Editor"]},
                    ],
                },
            )

    assert access_key_workspaces_client.delete_workspace_member.await_args_list == [
        call(workspace="team-b", principal_id="alice@example.com"),
        call(workspace="team-a", principal_id="alice@example.com"),
    ]
    access_key_workspaces_client.update_workspace_member.assert_not_awaited()

    listed = client.get("/v2/access-keys").json()["data"]
    workspace_key_listed = next(key for key in listed if key["name"] == "workspace-key")
    assert workspace_key_listed["status"] == "REVOKED"


def test_create_access_key_partial_workspace_grant_failure_restores_prior_roles(
    client,
    access_key_workspaces_client,
    caplog,
):
    # team-a already has alice as a Viewer before this request; the Editor grant to team-a
    # succeeds but team-b then fails. Rollback must restore team-a's prior Viewer role instead
    # of deleting the pre-existing membership outright. The uncertain team-b grant had no
    # prior membership, so it gets deleted first.
    async def list_members(*, workspace):
        members = [SimpleNamespace(principal="alice@example.com", roles=["Viewer"])] if workspace == "team-a" else []
        return _members_response(members)

    access_key_workspaces_client.list_workspace_members.side_effect = list_members
    access_key_workspaces_client.create_workspace_member.side_effect = [None, RuntimeError("workspace not found")]

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="workspace not found"):
            client.post(
                "/v2/access-keys",
                json={
                    "name": "workspace-key",
                    "workspaces": [
                        {"workspace": "team-a", "roles": ["Editor"]},
                        {"workspace": "team-b", "roles": ["Editor"]},
                    ],
                },
            )

    access_key_workspaces_client.update_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        principal_id="alice@example.com",
        body=UpdateWorkspaceMemberRequest(roles=["Viewer"]),
    )
    access_key_workspaces_client.delete_workspace_member.assert_awaited_once_with(
        workspace="team-b",
        principal_id="alice@example.com",
    )


def test_create_access_key_apply_then_error_rolls_back_all_attempts_in_reverse_order(
    client,
    access_key_workspaces_client,
    caplog,
):
    workspace_roles = {"team-b": ["Viewer"]}

    async def list_members(*, workspace):
        roles = workspace_roles.get(workspace)
        members = [] if roles is None else [SimpleNamespace(principal="alice@example.com", roles=roles)]
        return _members_response(members)

    async def create_member(*, workspace, body):
        if workspace in workspace_roles:
            workspace_roles[workspace][:] = body.roles
        else:
            workspace_roles[workspace] = list(body.roles)
        if workspace == "team-b":
            # Model an ambiguous transport failure after the membership mutation committed.
            raise RuntimeError("response lost after workspace mutation")

    rollback_order: list[tuple[str, str]] = []

    async def update_member(*, workspace, principal_id, body):
        workspace_roles[workspace] = list(body.roles)
        rollback_order.append(("update", workspace))

    async def delete_member(*, workspace, principal_id):
        workspace_roles.pop(workspace, None)
        rollback_order.append(("delete", workspace))

    access_key_workspaces_client.list_workspace_members.side_effect = list_members
    access_key_workspaces_client.create_workspace_member.side_effect = create_member
    access_key_workspaces_client.update_workspace_member.side_effect = update_member
    access_key_workspaces_client.delete_workspace_member.side_effect = delete_member

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="response lost after workspace mutation"):
            client.post(
                "/v2/access-keys",
                json={
                    "name": "workspace-key",
                    "workspaces": [
                        {"workspace": "team-a", "roles": ["Editor"]},
                        {"workspace": "team-b", "roles": ["Editor"]},
                    ],
                },
            )

    access_key_workspaces_client.update_workspace_member.assert_awaited_once_with(
        workspace="team-b",
        principal_id="alice@example.com",
        body=UpdateWorkspaceMemberRequest(roles=["Viewer"]),
    )
    access_key_workspaces_client.delete_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        principal_id="alice@example.com",
    )
    assert rollback_order == [("update", "team-b"), ("delete", "team-a")]
    assert workspace_roles == {"team-b": ["Viewer"]}

    listed = client.get("/v2/access-keys").json()["data"]
    workspace_key_listed = next(key for key in listed if key["name"] == "workspace-key")
    assert workspace_key_listed["status"] == "REVOKED"


def test_create_access_key_grant_aborts_without_mutating_when_prior_role_read_fails(
    client,
    access_key_workspaces_client,
    caplog,
):
    # If team-a's prior membership can't be read, granting anyway and later deleting it on
    # rollback could destroy real pre-existing access. Don't touch this workspace at all.
    access_key_workspaces_client.list_workspace_members.side_effect = RuntimeError("workspace unreachable")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="workspace unreachable"):
            client.post(
                "/v2/access-keys",
                json={"name": "workspace-key", "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}]},
            )

    access_key_workspaces_client.create_workspace_member.assert_not_awaited()
    access_key_workspaces_client.delete_workspace_member.assert_not_awaited()
    access_key_workspaces_client.update_workspace_member.assert_not_awaited()

    listed = client.get("/v2/access-keys").json()["data"]
    workspace_key_listed = next(key for key in listed if key["name"] == "workspace-key")
    assert workspace_key_listed["status"] == "REVOKED"


def test_create_access_key_grant_rolls_back_earlier_grant_when_later_prior_role_read_fails(
    client,
    access_key_workspaces_client,
    caplog,
):
    # team-a's grant succeeds first; team-b's prior-role read then fails. team-b must never be
    # mutated, and team-a's just-applied grant (which had no prior membership) must be unwound.
    async def list_members(*, workspace):
        if workspace == "team-b":
            raise RuntimeError("workspace unreachable")
        return _members_response([])

    access_key_workspaces_client.list_workspace_members.side_effect = list_members

    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="workspace unreachable"):
            client.post(
                "/v2/access-keys",
                json={
                    "name": "workspace-key",
                    "workspaces": [
                        {"workspace": "team-a", "roles": ["Editor"]},
                        {"workspace": "team-b", "roles": ["Editor"]},
                    ],
                },
            )

    access_key_workspaces_client.create_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        body=CreateWorkspaceMemberRequest(principal="alice@example.com", roles=["Editor"]),
    )
    access_key_workspaces_client.delete_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        principal_id="alice@example.com",
    )

    listed = client.get("/v2/access-keys").json()["data"]
    workspace_key_listed = next(key for key in listed if key["name"] == "workspace-key")
    assert workspace_key_listed["status"] == "REVOKED"


def _scoped_caller_auth_client(services: list[str] | None) -> AuthClient:
    """An AuthClient for a caller authenticated via a Scoped Access Key with `services` scope.

    `services=None` simulates an unscoped access key (no platform scope claims at all).
    """
    scopes = [f"{service}:{verb}" for service in (services or []) for verb in ("read", "write")]
    claims = TokenClaims(
        subject="alice@example.com",
        email="alice@example.com",
        groups=["team-ml"],
        scopes=scopes,
        raw_claims={},
    )
    return AuthClient(
        principal=Principal(id="alice@example.com", email="alice@example.com", groups=["team-ml"]),
        config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
        resolved_bearer_token=ResolvedBearerToken(claims=claims, token_kind="access_key"),
    )


def test_scope_restricted_access_key_cannot_create_unscoped_replacement(client):
    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        response = client.post("/v2/access-keys", json={"name": "escalated"})
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 400
    listed = client.get("/v2/access-keys").json()["data"]
    assert [key["name"] for key in listed] == []


def test_auth_only_scope_restricted_access_key_cannot_grant_workspace(
    client,
    access_key_workspaces_client,
):
    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "workspace-key",
                "scope": ["auth"],
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 400
    access_key_workspaces_client.create_workspace_member.assert_not_awaited()
    access_key_workspaces_client.update_workspace_member.assert_not_awaited()
    access_key_workspaces_client.delete_workspace_member.assert_not_awaited()
    listed = client.get("/v2/access-keys").json()["data"]
    assert [key["name"] for key in listed] == []


@pytest.mark.parametrize("workspace_service", ["entities", "platform"])
def test_scope_restricted_access_key_with_workspace_service_can_grant_workspace(
    client,
    access_key_workspaces_client,
    workspace_service,
):
    token = auth_client_context.set(_scoped_caller_auth_client(["auth", workspace_service]))
    try:
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "workspace-key",
                "scope": ["auth"],
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 200
    access_key_workspaces_client.create_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        body=CreateWorkspaceMemberRequest(principal="alice@example.com", roles=["Editor"]),
    )


def test_scope_restricted_access_key_cannot_create_key_scoped_to_other_service(client):
    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        response = client.post("/v2/access-keys", json={"name": "escalated", "scope": ["intake"]})
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 400
    listed = client.get("/v2/access-keys").json()["data"]
    assert [key["name"] for key in listed] == []


def test_scope_restricted_access_key_can_create_key_scoped_to_subset_of_its_own_scope(client):
    token = auth_client_context.set(_scoped_caller_auth_client(["auth", "intake"]))
    try:
        response = client.post("/v2/access-keys", json={"name": "narrower", "scope": ["auth"]})
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 200
    assert response.json()["scope"] == ["auth"]


def test_unscoped_access_key_can_create_unscoped_replacement(client):
    token = auth_client_context.set(_scoped_caller_auth_client(None))
    try:
        response = client.post("/v2/access-keys", json={"name": "unrestricted"})
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 200
    assert response.json()["scope"] == []


def test_oidc_caller_scopes_do_not_restrict_access_key_creation(client):
    claims = TokenClaims(
        subject="alice@example.com",
        email="alice@example.com",
        groups=["team-ml"],
        scopes=["auth:read", "auth:write"],
        raw_claims={},
    )
    auth_client = AuthClient(
        principal=Principal(id="alice@example.com", email="alice@example.com", groups=["team-ml"]),
        config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
        resolved_bearer_token=ResolvedBearerToken(claims=claims, token_kind="oidc_access_token"),
    )
    token = auth_client_context.set(auth_client)
    try:
        response = client.post("/v2/access-keys", json={"name": "from-oidc-session"})
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 200
    assert response.json()["scope"] == []


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
        registry=AccessKeyRegistry(entity_client=AsyncMock()),
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
    assert (
        response.json()["detail"]
        == "Service-bound Scoped Access Keys require HelixAdmin or Admin of every bound workspace"
    )
    has_role.assert_awaited_once_with("system", "HelixAdmin")


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
    assert (
        response.json()["detail"]
        == "Service-bound Scoped Access Keys require HelixAdmin or Admin of every bound workspace"
    )
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
    assert (
        response.json()["detail"]
        == "Service-bound Scoped Access Keys require HelixAdmin or Admin of every bound workspace"
    )
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


def _has_role_for(*granted: tuple[str, str]):
    async def has_role(self, workspace_id, role):
        return (workspace_id, role) in granted

    return has_role


def test_workspace_admin_creates_and_manages_workspace_bound_service_key(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )
        assert response.status_code == 200
        created = response.json()
        assert created["principal"] == "service-account:team-a--otel-collector"
        assert created["bound_workspaces"] == ["team-a"]

        listed = client.get("/v2/access-keys")
        assert [key["bound_workspaces"] for key in listed.json()["data"]] == [["team-a"]]
        rotated = client.post(f"/v2/access-keys/{created['jti']}/rotate")
        assert rotated.status_code == 200
        assert rotated.json()["new_key"]["bound_workspaces"] == ["team-a"]
        assert client.delete(f"/v2/access-keys/{rotated.json()['new_key']['jti']}").status_code == 200


def test_recreated_workspace_admin_cannot_manage_key_bound_to_the_old_workspace(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()

    # team-a is deleted and recreated: same name, new ID.
    access_key_workspaces_client.workspace_ids["team-a"] = "recreated-ws-id"
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.get("/v2/access-keys").json()["data"] == []
        assert client.post(f"/v2/access-keys/{created['jti']}/rotate").status_code == 404
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 404

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 200


def test_recreated_workspace_admin_cannot_take_over_service_account_with_a_key_from_the_old_workspace(
    client, access_key_workspaces_client, access_key_registry
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        old = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()

    # team-a is deleted and recreated. Deleting a workspace doesn't revoke its keys, so the old
    # key is still live and would gain the new workspace's membership with the account.
    access_key_workspaces_client.workspace_ids["team-a"] = "recreated-ws-id"
    access_key_workspaces_client.create_workspace_member.reset_mock()
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 403
    assert "earlier workspace of the same name" in response.json()["detail"]
    access_key_workspaces_client.create_workspace_member.assert_not_awaited()
    assert list(access_key_registry.keys) == [old["jti"]]
    assert old["jti"] not in access_key_registry.revoked

    # A HelixAdmin is not restricted, and can clean up or reissue the key.
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.delete(f"/v2/access-keys/{old['jti']}").status_code == 200


def test_workspace_replaced_during_role_check_cannot_manage_key_bound_to_the_old_workspace(
    client, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()

    # The ID matches before the role check, then team-a is deleted and recreated while it runs.
    async def has_role_after_recreation(_self, workspace, role):
        access_key_workspaces_client.workspace_ids["team-a"] = "recreated-ws-id"
        return (workspace, role) == ("team-a", "Admin")

    with patch.object(AuthClient, "has_role", has_role_after_recreation):
        assert client.post(f"/v2/access-keys/{created['jti']}/rotate").status_code == 404


def _transport_error() -> NemoTransportError:
    return NemoTransportError(httpx.ConnectError("workspaces service unreachable"))


def _workspace_not_found() -> NotFoundError:
    return NotFoundError(httpx.Response(404, json={"detail": "Workspace 'team-a' not found"}))


def test_workspace_bound_key_cannot_be_created_when_the_workspace_does_not_exist(
    client, access_key_workspaces_client, access_key_registry
):
    access_key_workspaces_client.get_workspace.side_effect = _workspace_not_found()
    body = {
        "name": "otel",
        "service_account_id": "team-a--otel-collector",
        "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
    }
    # A workspace Admin can't be an Admin of a workspace they can't resolve.
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=body).status_code == 403
    # A HelixAdmin gets told why.
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post("/v2/access-keys", json=body)

    assert response.status_code == 400
    assert "could not be resolved" in response.json()["detail"]
    assert access_key_registry.keys == {}


def test_workspace_bound_key_is_not_created_when_the_admin_check_fails_against_the_resolved_workspace_id(
    client, access_key_workspaces_client, access_key_registry
):
    async def has_role(_self, workspace, role):
        # team-a is deleted and recreated while its Admin role is being checked.
        if (workspace, role) != ("team-a", "Admin"):
            return False
        access_key_workspaces_client.workspace_ids["team-a"] = "recreated-ws-id"
        return True

    with patch.object(AuthClient, "has_role", has_role):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 403
    assert access_key_registry.keys == {}


def test_workspace_bound_key_creation_reports_a_transient_workspace_lookup_failure_as_503(
    client, access_key_workspaces_client, access_key_registry
):
    access_key_workspaces_client.get_workspace.side_effect = _transport_error()
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 503
    assert access_key_registry.keys == {}


def test_managing_a_bound_key_reports_a_transient_workspace_lookup_failure_as_503_not_404(
    client, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()

        access_key_workspaces_client.get_workspace.side_effect = _transport_error()
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 503
        assert client.post(f"/v2/access-keys/{created['jti']}/rotate").status_code == 503

        # A workspace that is really gone is a definitive answer: the key is not manageable.
        access_key_workspaces_client.get_workspace.side_effect = _workspace_not_found()
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 404


def test_workspace_admin_cannot_create_service_key_without_workspace_or_for_another_workspace(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        no_workspace = client.post("/v2/access-keys", json={"name": "otel", "service_account_id": "otel-collector"})
        other_workspace = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "otel-collector",
                "workspaces": [{"workspace": "team-b", "roles": ["Editor"]}],
            },
        )

    assert no_workspace.status_code == 403
    assert other_workspace.status_code == 403


def test_workspace_admin_cannot_manage_service_key_bound_to_another_workspace(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-b", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-b--otel-collector",
                "workspaces": [{"workspace": "team-b", "roles": ["Editor"]}],
            },
        ).json()

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.get("/v2/access-keys").json()["data"] == []
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 404


def test_workspace_admin_suspends_and_unsuspends_workspace_bound_service_key(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()

        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200
        assert [key["status"] for key in client.get("/v2/access-keys").json()["data"]] == ["SUSPENDED"]
        assert client.post(f"/v2/access-keys/{created['jti']}/unsuspend").status_code == 200
        assert [key["status"] for key in client.get("/v2/access-keys").json()["data"]] == ["ACTIVE"]


def test_workspace_admin_cannot_suspend_or_unsuspend_service_key_bound_to_another_workspace(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-b", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-b--otel-collector",
                "workspaces": [{"workspace": "team-b", "roles": ["Editor"]}],
            },
        ).json()

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 404
        assert client.post(f"/v2/access-keys/{created['jti']}/unsuspend").status_code == 404


def test_workspace_bound_service_key_still_applies_an_explicit_grant_for_an_existing_member(
    client, access_key_workspaces_client
):
    access_key_workspaces_client.list_workspace_members.return_value = _members_response(
        [SimpleNamespace(principal="service-account:team-a--otel-collector", roles=["Viewer"])]
    )
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel-2",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 200
    access_key_workspaces_client.create_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        body=CreateWorkspaceMemberRequest(principal="service-account:team-a--otel-collector", roles=["Editor"]),
    )


def test_rotating_workspace_bound_service_key_leaves_workspace_memberships_untouched(
    client, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Viewer"]}],
            },
        ).json()
        access_key_workspaces_client.reset_mock()

        rotated = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert rotated.status_code == 200
    # Same service-account principal, so its Viewer role carries over; rotation must not
    # re-grant any role (or make any other membership change).
    access_key_workspaces_client.create_workspace_member.assert_not_awaited()
    access_key_workspaces_client.update_workspace_member.assert_not_awaited()
    access_key_workspaces_client.delete_workspace_member.assert_not_awaited()


def test_workspace_bound_service_key_honours_an_explicit_role_for_its_workspace(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [{"workspace": "team-a", "roles": ["Viewer"]}],
            },
        )

    assert response.status_code == 200
    access_key_workspaces_client.create_workspace_member.assert_awaited_once_with(
        workspace="team-a",
        body=CreateWorkspaceMemberRequest(principal="service-account:team-a--otel-collector", roles=["Viewer"]),
    )


def test_workspace_bound_service_key_rejects_a_grant_outside_its_workspace(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel-collector",
                "workspaces": [
                    {"workspace": "team-a", "roles": ["Editor"]},
                    {"workspace": "team-b", "roles": ["Viewer"]},
                ],
            },
        )

    # Granting a role in team-b binds the key to team-b too, which takes Admin of team-b.
    assert response.status_code == 403
    access_key_workspaces_client.create_workspace_member.assert_not_awaited()


def test_workspace_admin_cannot_take_over_service_account_provisioned_outside_the_workspace(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        provisioned = client.post(
            "/v2/access-keys",
            json={
                "name": "runner",
                "service_account_id": "ci--runner",
                "workspaces": [{"workspace": "prod", "roles": ["Editor"]}],
            },
        )
    assert provisioned.status_code == 200

    with patch.object(AuthClient, "has_role", _has_role_for(("ci", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "runner",
                "service_account_id": "ci--runner",
                "workspaces": [{"workspace": "ci", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 403
    assert "outside workspace(s) 'ci'" in response.json()["detail"]


def test_workspace_admin_cannot_rotate_key_whose_service_account_gained_a_key_outside_the_workspace(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        other = {"name": "other", "service_account_id": "team-a--otel", "acknowledge_workspace_bound_keys": True}
        assert client.post("/v2/access-keys", json=other).status_code == 200

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 403


def test_workspace_admin_cannot_take_over_service_account_with_a_role_binding_outside_the_workspace(
    client, access_key_registry
):
    access_key_registry.bindings["service-account:team-a--otel"] = {"team-b"}

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 403
    assert "keys or role bindings outside workspace(s) 'team-a'" in response.json()["detail"]
    assert access_key_registry.keys == {}


def test_workspace_admin_cannot_rotate_key_whose_service_account_gained_a_role_binding_outside_the_workspace(
    client, access_key_registry
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()
        access_key_registry.bindings["service-account:team-a--otel"] = {"team-a", "team-b"}
        response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 403
    assert list(access_key_registry.keys) == [created["jti"]]


def test_create_is_rolled_back_when_outside_access_appears_between_the_check_and_the_insert(
    client, access_key_registry
):
    access_key_registry.after_add = lambda registry: registry.bindings.update(
        {"service-account:team-a--otel": {"prod"}}
    )

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 403
    assert "keys or role bindings outside workspace(s) 'team-a'" in response.json()["detail"]
    # The never-returned key is discarded, not revoked: a revoked key would still count against the
    # account in later confinement checks.
    assert access_key_registry.keys == {}
    assert len(access_key_registry.discarded) == 1
    assert access_key_registry.revoked == set()


def test_create_is_rolled_back_when_a_key_from_an_earlier_workspace_appears_between_the_check_and_the_insert(
    client, access_key_registry
):
    def add_key_bound_to_the_earlier_workspace(registry):
        stale = next(iter(registry.keys.values())).model_copy(update={"jti": "ak_stale"})
        registry.keys["ak_stale"] = stale
        registry.owners["ak_stale"] = "someone-else"
        registry.bound["ak_stale"] = ["ws-before-recreation"]

    access_key_registry.after_add = add_key_bound_to_the_earlier_workspace

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 403
    assert "outside workspace(s) 'team-a'" in response.json()["detail"]
    assert list(access_key_registry.keys) == ["ak_stale"]
    assert len(access_key_registry.discarded) == 1
    assert access_key_registry.revoked == set()


def test_rotation_discards_successor_when_outside_access_appears_between_the_check_and_the_insert(
    client, access_key_registry
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post(
            "/v2/access-keys",
            json={
                "name": "otel",
                "service_account_id": "team-a--otel",
                "workspaces": [{"workspace": "team-a", "roles": ["Editor"]}],
            },
        ).json()
        access_key_registry.after_add = lambda registry: registry.bindings.update(
            {"service-account:team-a--otel": {"prod"}}
        )
        response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 403
    assert len(access_key_registry.discarded) == 1
    assert list(access_key_registry.keys) == [created["jti"]]
    assert created["jti"] not in access_key_registry.revoked


def test_helix_admin_is_not_rolled_back_when_outside_access_appears_after_the_insert(client, access_key_registry):
    access_key_registry.after_add = lambda registry: registry.bindings.update({"service-account:ci--runner": {"prod"}})

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "runner",
                "service_account_id": "ci--runner",
                "workspaces": [{"workspace": "ci", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 200


def test_listing_falls_back_to_own_keys_when_the_admin_role_check_fails(client):
    personal = client.post("/v2/access-keys", json={"name": "personal"}).json()

    with patch.object(
        AuthClient, "has_role", new_callable=AsyncMock, side_effect=HTTPException(status_code=503, detail="PDP down")
    ):
        response = client.get("/v2/access-keys")

    assert response.status_code == 200
    assert [key["jti"] for key in response.json()["data"]] == [personal["jti"]]


def test_listing_hides_only_service_keys_whose_workspace_check_fails(client):
    personal = client.post("/v2/access-keys", json={"name": "personal"}).json()
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))):
        for name, workspace in (("a1", "team-a"), ("a2", "team-a"), ("b1", "team-b")):
            created = client.post(
                "/v2/access-keys",
                json={
                    "name": name,
                    "service_account_id": f"{workspace}--{name}",
                    "workspaces": [{"workspace": workspace, "roles": ["Editor"]}],
                },
            )
            assert created.status_code == 200
        team_b_key = next(
            key for key in client.get("/v2/access-keys").json()["data"] if key["bound_workspaces"] == ["team-b"]
        )

    pdp_calls: list[tuple[str, str]] = []

    async def has_role(_self, workspace, role):
        pdp_calls.append((workspace, role))
        if (workspace, role) == ("team-a", "Admin"):
            raise HTTPException(status_code=503, detail="PDP unavailable")
        return (workspace, role) == ("team-b", "Admin")

    with patch.object(AuthClient, "has_role", has_role):
        response = client.get("/v2/access-keys")

    # The failed team-a check hides team-a's keys but not the personal key or team-b's.
    assert response.status_code == 200
    assert {key["jti"] for key in response.json()["data"]} == {personal["jti"], team_b_key["jti"]}
    # And it is not retried for each of team-a's keys.
    assert pdp_calls.count(("team-a", "Admin")) == 1


def test_listing_does_not_mask_a_failure_listing_the_keys_themselves(client, access_key_registry):
    access_key_registry.list_for_principal = AsyncMock(
        side_effect=HTTPException(status_code=502, detail="entity service unavailable")
    )

    with patch.object(AuthClient, "has_role", _has_role_for()):
        response = client.get("/v2/access-keys")

    assert response.status_code == 502
    access_key_registry.list_for_principal.assert_awaited_once()


def test_helix_admin_can_create_bound_key_for_service_account_with_keys_elsewhere(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.post("/v2/access-keys", json={"name": "a", "service_account_id": "ci--runner"}).status_code == 200
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "b",
                "service_account_id": "ci--runner",
                "workspaces": [{"workspace": "ci", "roles": ["Editor"]}],
            },
        )

    assert response.status_code == 200


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

        # A second HelixAdmin, distinct from the creator, manages the same registry.
        other_admin_token = auth_client_context.set(
            AuthClient(
                principal=Principal(id="bob@example.com", email="bob@example.com"),
                config=AuthConfig(enabled=True, access_keys=AccessKeyConfig(enabled=True)),
            )
        )
        try:
            # A HelixAdmin can discover and manage another administrator's service-bound key.
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
    with caplog.at_level(logging.INFO, logger="nhx.core.auth.app.access_keys"):
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

    with caplog.at_level(logging.INFO, logger="nhx.core.auth.app.access_keys"):
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
            audiences=["nemo-helix-access-key"],
        )
    )

    result = await registry.list_for_principal("alice@example.com", page=1, page_size=100)

    assert result.data[0].status == "EXPIRED"
    assert not await registry.is_active("ak_expired", "alice@example.com")


@pytest.mark.asyncio
async def test_mixed_store_listing_does_not_drop_fetched_rows() -> None:
    issued_at = datetime.now(tz=UTC)
    legacy = AccessKeyEntity(
        name="ak_legacy",
        workspace="system",
        key_name="legacy",
        principal="alice@example.com",
        issuer="https://nemo.example.com/apis/auth",
        audiences=["nemo-helix-access-key"],
        issued_at=issued_at - timedelta(seconds=1),
    )
    credential = AccessKeyEntity(
        name="ak_credential",
        workspace="system",
        key_name="credential",
        principal="alice@example.com",
        issuer="https://nemo.example.com/apis/auth",
        audiences=["nemo-helix-access-key"],
        issued_at=issued_at,
    )

    class StubCredentialAdapter(AccessKeyCredentialAdapter):
        def __init__(self, record: AccessKeyEntity) -> None:
            self.record = record

        async def list_for_owner(
            self,
            owner_account_id: str,
            *,
            include_service_accounts: bool,
            service_bound_workspace_ids=(),
            offset: int,
            limit: int,
        ) -> tuple[list[AccessKeyEntity], bool]:
            return [self.record], False

    entity_client = AsyncMock()
    entity_client.list.return_value = PaginatedResult(
        data=[legacy],
        pagination=PaginationData(
            page=1,
            page_size=1,
            current_page_size=1,
            total_pages=1,
            total_results=1,
        ),
    )
    registry = AccessKeyRegistry(
        entity_client=entity_client,
        credential_adapter=StubCredentialAdapter(credential),
    )

    result = await registry.list_for_principal(
        "alice@example.com",
        page=1,
        page_size=1,
        owner_account_id="account-1",
    )

    assert [record.jti for record in result.data] == ["ak_credential", "ak_legacy"]
    assert result.has_more is False


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
    assert (
        create_responses["403"]["description"]
        == "Service-bound Scoped Access Keys require HelixAdmin, or Admin of every bound workspace for a service "
        "account named `<workspace>--<name>` after one of them, with no access outside them"
    )
    assert create_responses["403"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert create_responses["404"]["description"] == "Scoped Access Keys are not enabled"
    assert create_responses["404"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AccessKeyErrorResponse"
    }
    assert create_responses["409"]["description"] == (
        "Concurrent access-key update conflict, a bound workspace changed while the key was being "
        "created (retry), or (code workspace_bound_keys_exist) the service account has workspace-bound "
        "keys and the request needs acknowledge_workspace_bound_keys"
    )
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
    assert error_code_schema["anyOf"][0]["enum"] == [
        "access_keys_disabled",
        "workspace_bound_keys_exist",
        "bound_workspaces_not_administered",
    ]
    assert error_code_schema["description"] == (
        "Set to access_keys_disabled when the Scoped Access Key feature is disabled, to "
        "workspace_bound_keys_exist when creating the key needs acknowledge_workspace_bound_keys, or to "
        "bound_workspaces_not_administered when the caller is Admin of some of a service-bound key's "
        "bound workspaces but not all of them."
    )
    paths = openapi["paths"]
    for path, method in (
        ("/v2/access-keys/{jti}", "delete"),
        ("/v2/access-keys/{jti}/suspend", "post"),
    ):
        forbidden = paths[path][method]["responses"]["403"]
        assert forbidden["description"] == (
            "The key wasn't changed: (code bound_workspaces_not_administered) the caller is Admin of some of the "
            "service-bound key's bound workspaces but not all of them, which the detail names"
        )
        assert forbidden["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/AccessKeyErrorResponse"
        }
    for path in ("/v2/access-keys/{jti}/unsuspend", "/v2/access-keys/{jti}/rotate"):
        assert "(code bound_workspaces_not_administered)" in paths[path]["post"]["responses"]["403"]["description"]

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
            "bound_workspaces": [],
            "bound_workspace_ids": [],
            "created_at": created["created_at"],
            "expires_at": created["expires_at"],
            "grace_period_expires_at": None,
            "last_used_at": None,
            "description": "CI intake automation",
            "status": "ACTIVE",
            "issuer": "http://testserver/apis/auth",
            "audiences": ["nemo-helix-access-key"],
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


def test_scope_restricted_access_key_cannot_rotate_an_unscoped_owned_key(client):
    prior = client.post("/v2/access-keys", json={"name": "prior"}).json()

    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        response = client.post(f"/v2/access-keys/{prior['jti']}/rotate")
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 400
    listed = client.get("/v2/access-keys").json()["data"]
    assert next(key for key in listed if key["jti"] == prior["jti"])["status"] == "ACTIVE"


def test_scope_restricted_access_key_cannot_rotate_a_differently_scoped_owned_key(client):
    prior = client.post("/v2/access-keys", json={"name": "prior", "scope": ["intake"]}).json()

    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        response = client.post(f"/v2/access-keys/{prior['jti']}/rotate")
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 400
    listed = client.get("/v2/access-keys").json()["data"]
    assert next(key for key in listed if key["jti"] == prior["jti"])["status"] == "ACTIVE"


def test_scope_restricted_access_key_can_rotate_a_key_within_its_own_scope(client):
    prior = client.post("/v2/access-keys", json={"name": "prior", "scope": ["auth"]}).json()

    token = auth_client_context.set(_scoped_caller_auth_client(["auth", "intake"]))
    try:
        response = client.post(f"/v2/access-keys/{prior['jti']}/rotate")
    finally:
        auth_client_context.reset(token)

    assert response.status_code == 200
    assert response.json()["new_key"]["scope"] == ["auth"]


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
    assert max_grace_period_seconds is not None

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

    with caplog.at_level(logging.INFO, logger="nhx.core.auth.app.access_keys"):
        response = client.post(f"/v2/access-keys/{jti}/rotate")

    new_jti = response.json()["new_key"]["jti"]
    events = {record.audit_event: record for record in caplog.records if hasattr(record, "audit_event")}
    assert events["access_key.rotated"].actor_principal == "alice@example.com"
    assert events["access_key.rotated"].access_key_jti == jti
    assert events["access_key.rotated"].access_key_new_jti == new_jti


def _bound_body(*workspaces: str, name: str = "otel", account: str = "team-a--otel", roles=("Editor",)) -> dict:
    return {
        "name": name,
        "service_account_id": account,
        "workspaces": [{"workspace": workspace, "roles": list(roles)} for workspace in workspaces],
    }


def test_workspace_admin_must_namespace_the_service_account_under_a_bound_workspace(client, access_key_registry):
    both = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))
    with patch.object(AuthClient, "has_role", both):
        for account in (
            "platform-ops",
            "team-a--",
            "team-c--ops",
            "team-ab--ops",
            "team-a/ops",
            "team-a-ops",
            "team-a--ops/x",
        ):
            response = client.post("/v2/access-keys", json=_bound_body("team-a", account=account))
            assert response.status_code == 403, account
            assert "namespaced under a bound workspace" in response.json()["detail"]
        assert access_key_registry.keys == {}

        created = client.post("/v2/access-keys", json=_bound_body("team-a", account="team-a--ops"))
        assert created.status_code == 200
        assert client.post(f"/v2/access-keys/{created.json()['jti']}/rotate").status_code == 200
        # A key bound to several workspaces may be namespaced under any of them.
        multi = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b", account="team-b--shared"))
        assert multi.status_code == 200


def test_helix_admin_key_for_an_account_not_named_after_a_workspace_grants_roles_but_stays_unbound(
    client, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post("/v2/access-keys", json=_bound_body("team-a", account="platform-ops"))
    assert response.status_code == 200
    # Binding it would hand a platform-owned credential to team-a's Admins.
    assert response.json()["bound_workspaces"] == []
    assert response.json()["bound_workspace_ids"] == []
    assert access_key_workspaces_client.create_workspace_member.await_args_list == [
        call(
            workspace="team-a",
            body=CreateWorkspaceMemberRequest(principal="service-account:platform-ops", roles=["Editor"]),
        )
    ]

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.get("/v2/access-keys").json()["data"] == []
        for path in ("rotate", "suspend"):
            assert client.post(f"/v2/access-keys/{response.json()['jti']}/{path}").status_code == 404
        assert client.delete(f"/v2/access-keys/{response.json()['jti']}").status_code == 404


def test_helix_admin_key_for_an_account_named_after_a_workspace_is_bound_to_it(client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        created = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b", account="team-b--ops")).json()
    assert created["bound_workspaces"] == ["team-a", "team-b"]

    # Bound to both, so it's managed by Admins of both, like a key a workspace Admin creates.
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))):
        rotated = client.post(f"/v2/access-keys/{created['jti']}/rotate")
        assert rotated.status_code == 200
        assert rotated.json()["new_key"]["bound_workspace_ids"] == ["ws-id-team-a", "ws-id-team-b"]


def test_workspace_bound_key_grants_editor_when_a_grant_names_no_roles(
    client, access_key_workspaces_client, access_key_registry
):
    body = _bound_body("team-a", "team-b")
    # Roles given but all blank are a mistake, not a request for the default.
    for blank in ([{"workspace": "team-a", "roles": []}], [{"workspace": "team-a", "roles": [" "]}]):
        assert client.post("/v2/access-keys", json={**body, "workspaces": blank}).status_code == 422
    access_key_workspaces_client.create_workspace_member.assert_not_awaited()
    assert access_key_registry.keys == {}

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json={**body, "workspaces": [{"workspace": "team-a"}]})

    assert created.status_code == 200
    assert access_key_workspaces_client.create_workspace_member.await_args_list == [
        call(
            workspace="team-a",
            body=CreateWorkspaceMemberRequest(principal="service-account:team-a--otel", roles=["Editor"]),
        )
    ]


def test_workspace_bound_key_rejects_duplicate_bound_workspaces(client):
    body = _bound_body("team-a")
    duplicated = [{"workspace": "team-a", "roles": ["Editor"]}, {"workspace": "team-a", "roles": ["Viewer"]}]
    assert client.post("/v2/access-keys", json={**body, "workspaces": duplicated}).status_code == 422


def test_multi_workspace_key_needs_admin_of_every_bound_workspace_to_create(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b"))
    assert response.status_code == 403
    assert access_key_registry.keys == {}


def test_multi_workspace_key_is_granted_each_workspace_and_managed_by_admins_of_all(
    client, access_key_workspaces_client
):
    both = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))
    with patch.object(AuthClient, "has_role", both):
        response = client.post(
            "/v2/access-keys",
            json={
                **_bound_body("team-a", "team-b"),
                "workspaces": [
                    {"workspace": "team-a", "roles": ["Editor"]},
                    {"workspace": "team-b", "roles": ["Viewer"]},
                ],
            },
        )
        assert response.status_code == 200
        created = response.json()
        assert created["bound_workspaces"] == ["team-a", "team-b"]
        assert created["bound_workspace_ids"] == ["ws-id-team-a", "ws-id-team-b"]
        assert [key["bound_workspaces"] for key in client.get("/v2/access-keys").json()["data"]] == [
            ["team-a", "team-b"]
        ]
    assert access_key_workspaces_client.create_workspace_member.await_args_list == [
        call(
            workspace="team-a",
            body=CreateWorkspaceMemberRequest(principal="service-account:team-a--otel", roles=["Editor"]),
        ),
        call(
            workspace="team-b",
            body=CreateWorkspaceMemberRequest(principal="service-account:team-a--otel", roles=["Viewer"]),
        ),
    ]

    # An Admin of only one of the workspaces can't see it, and is told which workspaces stop them managing it.
    for only, other in ((("team-a", "Admin"), "team-b"), (("team-b", "Admin"), "team-a")):
        with patch.object(AuthClient, "has_role", _has_role_for(only)):
            assert client.get("/v2/access-keys").json()["data"] == []
            for action, response in (
                ("suspend", client.post(f"/v2/access-keys/{created['jti']}/suspend")),
                ("unsuspend", client.post(f"/v2/access-keys/{created['jti']}/unsuspend")),
                ("rotate", client.post(f"/v2/access-keys/{created['jti']}/rotate")),
                ("revoke", client.delete(f"/v2/access-keys/{created['jti']}")),
            ):
                assert response.status_code == 403, action
                assert response.json() == {
                    "detail": (
                        f"Scoped Access Key {created['jti']} is also bound to workspace(s) '{other}', which you "
                        "aren't Admin of; only a HelixAdmin, or an Admin of every workspace it's bound to, can "
                        f"{action} it"
                    ),
                    "code": "bound_workspaces_not_administered",
                }
    # Nor did any of that change it.
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert [key["status"] for key in client.get("/v2/access-keys").json()["data"]] == ["ACTIVE"]
    # An Admin of neither workspace isn't told the key exists.
    with patch.object(AuthClient, "has_role", _has_role_for(("team-c", "Admin"))):
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 404
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 404

    with patch.object(AuthClient, "has_role", both):
        rotated = client.post(f"/v2/access-keys/{created['jti']}/rotate")
        assert rotated.status_code == 200
        assert rotated.json()["new_key"]["bound_workspaces"] == ["team-a", "team-b"]
        assert rotated.json()["new_key"]["bound_workspace_ids"] == ["ws-id-team-a", "ws-id-team-b"]


def test_multi_workspace_key_is_not_managed_once_any_bound_workspace_is_recreated(client, access_key_workspaces_client):
    both = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))
    with patch.object(AuthClient, "has_role", both):
        created = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b")).json()

    access_key_workspaces_client.workspace_ids["team-b"] = "recreated-ws-id"
    with patch.object(AuthClient, "has_role", both):
        assert client.get("/v2/access-keys").json()["data"] == []
        # Still Admin of team-a, so told why. The old team-b is gone, so it's only counted: it isn't
        # the team-b they administer now, and its ID isn't theirs to see.
        revoked = client.delete(f"/v2/access-keys/{created['jti']}")
        assert revoked.status_code == 403
        assert revoked.json()["detail"] == (
            f"Scoped Access Key {created['jti']} is also bound to 1 workspace(s) you can't see or that no longer "
            "exist; only a HelixAdmin, or an Admin of every workspace it's bound to, can revoke it"
        )
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        # The old team-b is gone, so a HelixAdmin sees it by ID.
        listed = client.get("/v2/access-keys").json()["data"]
        assert [key["bound_workspaces"] for key in listed] == [["team-a", "ws-id-team-b"]]
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 200


def test_workspace_admins_see_and_manage_keys_other_admins_bound_to_their_workspace(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()

    alice = auth_client_context.get()
    assert alice is not None
    other_admin = auth_client_context.set(
        AuthClient(principal=Principal(id="bob@example.com", email="bob@example.com"), config=alice.config)
    )
    try:
        with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
            assert [key["jti"] for key in client.get("/v2/access-keys").json()["data"]] == [created["jti"]]
            assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200
        # Not an Admin of team-a: neither listed nor manageable.
        with patch.object(AuthClient, "has_role", _has_role_for(("team-b", "Admin"))):
            assert client.get("/v2/access-keys").json()["data"] == []
            assert client.post(f"/v2/access-keys/{created['jti']}/unsuspend").status_code == 404
    finally:
        auth_client_context.reset(other_admin)


def test_listing_degrades_to_personal_keys_when_workspaces_cannot_be_read(client, access_key_workspaces_client):
    personal = client.post("/v2/access-keys", json={"name": "personal"}).json()
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        client.post("/v2/access-keys", json=_bound_body("team-a"))
        access_key_workspaces_client.list_workspaces.side_effect = _transport_error()
        listed = client.get("/v2/access-keys")

    # Listing doesn't fail; the bound key can't be admin-checked, so it's left out until a retry.
    assert listed.status_code == 200
    assert [key["jti"] for key in listed.json()["data"]] == [personal["jti"]]


def test_workspace_admin_can_widen_a_service_account_to_more_workspaces_they_administer(client):
    both = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))
    with patch.object(AuthClient, "has_role", both):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
        # The account's existing key stays within team-a and team-b, so it can be used for both, once
        # the caller confirms that the team-a key also gets the team-b access.
        widened = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b", name="b"))
        assert widened.status_code == 409
        assert widened.json()["code"] == "workspace_bound_keys_exist"
        acknowledged = {**_bound_body("team-a", "team-b", name="b"), "acknowledge_workspace_bound_keys": True}
        assert client.post("/v2/access-keys", json=acknowledged).status_code == 200


def test_workspace_admin_cannot_narrow_a_service_account_that_has_a_key_in_another_workspace(client):
    both = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))
    with patch.object(AuthClient, "has_role", both):
        assert client.post("/v2/access-keys", json=_bound_body("team-a", "team-b")).status_code == 200
        response = client.post("/v2/access-keys", json=_bound_body("team-a", name="a-only"))

    assert response.status_code == 403
    assert "outside workspace(s) 'team-a'" in response.json()["detail"]


def test_concurrent_admins_racing_for_an_account_do_not_lock_each_other_out(client, access_key_registry):
    def other_workspace_inserts_its_key(registry):
        mine = next(iter(registry.keys.values()))
        registry.keys["ak_racer"] = mine.model_copy(update={"jti": "ak_racer"})
        registry.owners["ak_racer"] = "bob@example.com"
        registry.bound["ak_racer"] = ["ws-id-team-b"]

    access_key_registry.after_add = other_workspace_inserts_its_key
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 403

        # The racing request loses the same way, and is discarded too.
        access_key_registry.keys.pop("ak_racer")
        access_key_registry.bound.pop("ak_racer")
        # Nothing is left behind to count against the account, so a retry succeeds.
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
    assert access_key_registry.revoked == set()


def test_bound_key_is_withdrawn_when_its_workspace_is_recreated_while_granting(
    client, access_key_workspaces_client, access_key_registry
):
    async def recreate_while_granting(*, workspace, body):
        access_key_workspaces_client.workspace_ids["team-a"] = "recreated-ws-id"

    access_key_workspaces_client.create_workspace_member.side_effect = recreate_while_granting
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post("/v2/access-keys", json=_bound_body("team-a"))

    # A retryable conflict, not a permissions problem.
    assert response.status_code == 409
    assert "changed while the key was being created" in response.json()["detail"]
    # The membership granted in the recreated workspace is rolled back, and the key is discarded.
    access_key_workspaces_client.delete_workspace_member.assert_awaited_once_with(
        workspace="team-a", principal_id="service-account:team-a--otel"
    )
    assert access_key_registry.keys == {}
    assert len(access_key_registry.discarded) == 1


def test_helix_admin_must_acknowledge_widening_a_service_accounts_workspace_bound_keys(
    client, access_key_registry, access_key_workspaces_client, caplog
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
    access_key_workspaces_client.create_workspace_member.reset_mock()
    elsewhere = _bound_body("team-b", name="platform")

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        refused = client.post("/v2/access-keys", json=elsewhere)
        assert refused.status_code == 409
        assert refused.json()["code"] == "workspace_bound_keys_exist"
        # Resending can't change the answer, so clients that retry 409s mustn't.
        assert refused.headers["x-should-retry"] == "false"
        # The caller supplied the account, and can see every workspace its keys are bound to.
        assert refused.json()["detail"].startswith(
            "Service account 'team-a--otel' already has live keys bound to workspace(s) 'team-a'"
        )
        assert refused.json()["detail"].endswith("Set acknowledge_workspace_bound_keys to confirm.")
        # Nothing was minted or granted for the refused request.
        assert len(access_key_registry.keys) == 1
        access_key_workspaces_client.create_workspace_member.assert_not_awaited()

        with caplog.at_level(logging.WARNING, logger="nhx.core.auth.app.access_keys"):
            accepted = client.post("/v2/access-keys", json={**elsewhere, "acknowledge_workspace_bound_keys": True})

    assert accepted.status_code == 200
    events = {record.audit_event: record for record in caplog.records if hasattr(record, "audit_event")}
    assert events["access_key.workspace_bound_keys_widened"].workspaces == ["team-a"]
    assert events["access_key.workspace_bound_keys_widened"].workspace_ids == ["ws-id-team-a"]
    assert events["access_key.workspace_bound_keys_widened"].service_account_id == "team-a--otel"


@pytest.mark.parametrize(
    "body",
    [
        # Grants nothing, so the bound key gains nothing.
        {"name": "platform", "service_account_id": "team-a--otel"},
        # Binds to, and grants roles only inside, the bound key's workspace.
        _bound_body("team-a", name="again"),
    ],
    ids=["no-grants", "bound-inside"],
)
def test_helix_admin_needs_no_acknowledgement_for_a_key_whose_grants_stay_within_the_bound_keys(client, body):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.post("/v2/access-keys", json=body).status_code == 200


def test_revoked_workspace_bound_keys_need_no_acknowledgement(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 200
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post(
            "/v2/access-keys",
            json={
                "name": "platform",
                "service_account_id": "team-a--otel",
                "workspaces": [{"workspace": "team-b", "roles": ["Editor"]}],
            },
        )
    assert response.status_code == 200


def test_validation_errors_take_precedence_over_the_widening_acknowledgement(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
    # A caller restricted to the auth scope can't grant memberships, whatever it confirms.
    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
            response = client.post(
                "/v2/access-keys",
                json={
                    "name": "platform",
                    "service_account_id": "team-a--otel",
                    "workspaces": [{"workspace": "team-b", "roles": ["Editor"]}],
                },
            )
    finally:
        auth_client_context.reset(token)
    # The caller isn't asked to confirm a request that would fail anyway.
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("has_role", "body"),
    [
        (_has_role_for(("system", "HelixAdmin")), {"name": "platform", "service_account_id": "team-a--otel"}),
        (_has_role_for(("team-a", "Admin")), _bound_body("team-a")),
    ],
    ids=["helix-admin", "workspace-admin"],
)
def test_creating_a_key_for_a_disabled_service_account_is_rejected_up_front(
    client, access_key_registry, access_key_workspaces_client, has_role, body
):
    access_key_registry.disabled_service_accounts.add("service-account:team-a--otel")

    with patch.object(AuthClient, "has_role", has_role):
        response = client.post("/v2/access-keys", json={**body, "acknowledge_workspace_bound_keys": False})

    # A clear 400, not a confirmation prompt (409) for a key that couldn't be saved anyway.
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Service account 'team-a--otel' is disabled; enable it before creating keys for it"
    )
    assert access_key_registry.keys == {}
    access_key_workspaces_client.create_workspace_member.assert_not_awaited()


def test_listing_checks_each_bound_workspace_once_however_many_keys_share_it(client, access_key_workspaces_client):
    every = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"), ("team-c", "Admin"))
    with patch.object(AuthClient, "has_role", every):
        for name, workspaces in (("a", ("team-a",)), ("ab", ("team-a", "team-b")), ("ac", ("team-a", "team-c"))):
            body = {**_bound_body(*workspaces, name=name, account=f"team-a--svc-{name}")}
            assert client.post("/v2/access-keys", json=body).status_code == 200
    access_key_workspaces_client.get_workspace.reset_mock()
    role_checks: list[tuple[str, str]] = []

    async def has_role(self, workspace_id, role):
        role_checks.append((workspace_id, role))
        return (workspace_id, role) in {("team-a", "Admin"), ("team-b", "Admin"), ("team-c", "Admin")}

    with patch.object(AuthClient, "has_role", has_role):
        listed = client.get("/v2/access-keys")

    assert listed.status_code == 200
    assert sorted(key["name"] for key in listed.json()["data"]) == ["a", "ab", "ac"]
    # Each workspace a listed key is bound to is checked once, however many keys share it: one role
    # check each, plus the HelixAdmin check, and one name-to-ID confirmation per workspace that passed
    # its role check. Administered workspaces no key is bound to (ci, prod) cost nothing.
    assert sorted(role_checks) == [
        ("system", "HelixAdmin"),
        ("team-a", "Admin"),
        ("team-b", "Admin"),
        ("team-c", "Admin"),
    ]
    assert sorted(call.kwargs["name"] for call in access_key_workspaces_client.get_workspace.await_args_list) == [
        "team-a",
        "team-b",
        "team-c",
    ]


def _name_lookups(workspaces_client) -> list[str]:
    """IDs looked up by name (list_workspaces with an ID filter), in call order."""
    filters = [
        json.loads(call.kwargs["query_params"]["filter"])
        for call in workspaces_client.list_workspaces.await_args_list
        if (call.kwargs.get("query_params") or {}).get("filter")
    ]
    looked_up = []
    for workspace_filter in filters:
        ids = workspace_filter.get("id")
        looked_up.extend(ids["$in"] if isinstance(ids, dict) else [ids] if ids else [])
    return looked_up


def test_listing_reuses_the_workspace_names_it_listed_for_the_admin_checks(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
        access_key_workspaces_client.list_workspaces.reset_mock()

        listed = client.get("/v2/access-keys")

    assert listed.status_code == 200
    assert [key["bound_workspaces"] for key in listed.json()["data"]] == [["team-a"]]
    # Each visible workspace's name came from listing them; none is looked up again by ID.
    assert _name_lookups(access_key_workspaces_client) == []


def test_rotation_reuses_the_name_the_admin_check_read_instead_of_failing_on_a_second_lookup(
    client, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        list_workspaces = access_key_workspaces_client.list_workspaces.side_effect
        name_lookups = 0

        async def first_name_lookup_only(*, query_params=None):
            nonlocal name_lookups
            if (query_params or {}).get("filter"):
                name_lookups += 1
                if name_lookups > 1:
                    raise _transport_error()
            return await list_workspaces(query_params=query_params)

        access_key_workspaces_client.list_workspaces.side_effect = first_name_lookup_only
        rotated = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    # The confinement check uses the name the Admin check already read, so a later lookup
    # failure can't turn the bound workspace's own role binding into "access outside" it.
    assert rotated.status_code == 200
    assert rotated.json()["new_key"]["bound_workspaces"] == ["team-a"]
    assert name_lookups == 1


def test_rotating_a_key_for_a_disabled_service_account_is_rejected_up_front(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        access_key_registry.disabled_service_accounts.add("service-account:team-a--otel")
        keys_before = set(access_key_registry.keys)

        response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    # A clear 400, not the confinement check's misleading "access outside the workspace" 403.
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Service account 'team-a--otel' is disabled; enable it before rotating its keys"
    )
    assert set(access_key_registry.keys) == keys_before


def test_granting_a_service_account_a_new_role_in_a_bound_keys_workspace_needs_acknowledgement(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a", roles=("Viewer",))).status_code == 200
        # The same role again gives the first key nothing new.
        again = _bound_body("team-a", name="again", roles=("Viewer",))
        assert client.post("/v2/access-keys", json=again).status_code == 200

        # A higher role would reach the first key too, which was meant to be Viewer-only.
        editor = _bound_body("team-a", name="editor", roles=("Editor",))
        escalated = client.post("/v2/access-keys", json=editor)
        assert escalated.status_code == 409
        assert escalated.json()["code"] == "workspace_bound_keys_exist"
        assert "'team-a'" in escalated.json()["detail"]
        assert (
            client.post("/v2/access-keys", json={**editor, "acknowledge_workspace_bound_keys": True}).status_code == 200
        )
        # Now that it holds Editor, Viewer (which Editor includes) gives the first key nothing new.
        viewer = _bound_body("team-a", name="viewer", roles=("Viewer",))
        assert client.post("/v2/access-keys", json=viewer).status_code == 200


def _racing_key_bound_to(workspace_id: str):
    """An after_add hook that lands another live key for the same account, bound to ``workspace_id``."""

    def insert(registry):
        mine = next(key for jti, key in registry.keys.items() if jti != "ak_racer")
        # Issued first, but written only after this key passed its checks.
        registry.keys["ak_racer"] = mine.model_copy(
            update={"jti": "ak_racer", "created_at": mine.created_at - timedelta(seconds=1)}
        )
        registry.owners["ak_racer"] = "bob@example.com"
        registry.bound["ak_racer"] = [workspace_id]

    return insert


def test_a_key_that_widens_a_concurrently_created_bound_key_is_discarded_unless_acknowledged(
    client, access_key_registry, access_key_workspaces_client, caplog
):
    access_key_registry.after_add = _racing_key_bound_to("ws-id-team-b")
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        # Nothing to widen when checked, but the racing team-b key landed before this key's record did.
        response = client.post("/v2/access-keys", json=_bound_body("team-a"))

        assert response.status_code == 409
        assert response.json()["code"] == "workspace_bound_keys_exist"
        assert "'team-b'" in response.json()["detail"]
        assert set(access_key_registry.keys) == {"ak_racer"}
        assert len(access_key_registry.discarded) == 1
        access_key_workspaces_client.create_workspace_member.assert_not_awaited()

        access_key_registry.keys.pop("ak_racer")
        access_key_registry.after_add = _racing_key_bound_to("ws-id-team-b")
        with caplog.at_level(logging.WARNING, logger="nhx.core.auth.app.access_keys"):
            acknowledged = client.post(
                "/v2/access-keys", json={**_bound_body("team-a"), "acknowledge_workspace_bound_keys": True}
            )

    assert acknowledged.status_code == 200
    events = {record.audit_event: record for record in caplog.records if hasattr(record, "audit_event")}
    assert events["access_key.workspace_bound_keys_widened"].workspace_ids == ["ws-id-team-b"]


def test_of_two_keys_created_at_once_each_is_asked_to_confirm(client, access_key_registry):
    def insert_later_racer(registry):
        mine = next(key for jti, key in registry.keys.items() if jti != "ak_racer")
        registry.keys["ak_racer"] = mine.model_copy(
            update={"jti": "ak_racer", "created_at": mine.created_at + timedelta(seconds=1)}
        )
        registry.owners["ak_racer"] = "bob@example.com"
        registry.bound["ak_racer"] = ["ws-id-team-b"]

    access_key_registry.after_add = insert_later_racer
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post("/v2/access-keys", json=_bound_body("team-a"))

    # Issued later, the racer still counts: its own request's check may already have run, so if
    # this one let it pass, neither would be confirmed.
    assert response.status_code == 409
    assert set(access_key_registry.keys) == {"ak_racer"}


def test_a_key_whose_post_write_checks_fail_is_discarded(client, access_key_registry):
    read_access = access_key_registry.service_account_access
    reads = 0

    async def fails_after_the_write(principal):
        nonlocal reads
        reads += 1
        if reads > 1:
            raise HTTPException(status_code=503, detail="Role bindings could not be read; retry")
        return await read_access(principal)

    access_key_registry.after_add = _racing_key_bound_to("ws-id-team-a")
    access_key_registry.service_account_access = fails_after_the_write
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post("/v2/access-keys", json=_bound_body("team-a"))

    assert response.status_code == 503
    # The unreturned key isn't left behind to count against the account.
    assert set(access_key_registry.keys) == {"ak_racer"}
    assert len(access_key_registry.discarded) == 1
    assert access_key_registry.revoked == set()


def test_listing_resolves_only_the_workspaces_the_caller_holds_admin_bindings_in(
    client, access_key_registry, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        access_key_registry.admin_binding_workspaces = {"team-a"}
        access_key_registry.role_binding_queries.clear()
        access_key_workspaces_client.list_workspaces.reset_mock()

        listed = client.get("/v2/access-keys")

    assert [key["jti"] for key in listed.json()["data"]] == [created["jti"]]
    # One role-binding query, for every identifier the caller's bindings may be keyed by.
    assert access_key_registry.role_binding_queries == [(["alice@example.com", "team-ml"], "Admin")]
    # One lookup for the administered workspace's ID, not a walk over every workspace the caller sees.
    assert [
        json.loads(call.kwargs["query_params"]["filter"])
        for call in access_key_workspaces_client.list_workspaces.await_args_list
    ] == [{"name": {"$in": ["team-a"]}}]


def test_listing_for_a_caller_who_is_admin_nowhere_reads_no_workspaces_or_other_admins_keys(
    client, access_key_registry, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        client.post("/v2/access-keys", json=_bound_body("team-a"))
    personal = client.post("/v2/access-keys", json={"name": "personal"}).json()
    access_key_registry.owners = dict.fromkeys(access_key_registry.owners, "bob@example.com") | {
        personal["jti"]: "alice@example.com"
    }
    access_key_registry.admin_binding_workspaces = set()
    access_key_workspaces_client.list_workspaces.reset_mock()

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=False) as has_role:
        listed = client.get("/v2/access-keys")

    assert [key["jti"] for key in listed.json()["data"]] == [personal["jti"]]
    access_key_workspaces_client.list_workspaces.assert_not_awaited()
    has_role.assert_awaited_once_with("system", "HelixAdmin")


def test_listing_resolves_administered_workspaces_in_batches(client, access_key_registry, access_key_workspaces_client):
    access_key_registry.admin_binding_workspaces = {f"team-{index:03}" for index in range(120)}

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=False):
        assert client.get("/v2/access-keys").status_code == 200

    batches = [
        json.loads(call.kwargs["query_params"]["filter"])["name"]["$in"]
        for call in access_key_workspaces_client.list_workspaces.await_args_list
    ]
    assert [len(batch) for batch in batches] == [50, 50, 20]
    assert sorted(name for batch in batches for name in batch) == sorted(access_key_registry.admin_binding_workspaces)


def test_admin_checks_for_several_workspaces_run_concurrently(client):
    started: set[str] = set()
    both_started: asyncio.Event | None = None

    async def has_role(_self, workspace, role):
        nonlocal both_started
        if workspace == "system":
            return False
        if both_started is None:
            both_started = asyncio.Event()
        started.add(workspace)
        if started >= {"team-a", "team-b"}:
            both_started.set()
        # Checked one after another, the first check would never see the second one start.
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return True

    with patch.object(AuthClient, "has_role", has_role):
        response = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b"))

    assert response.status_code == 200


def test_a_failed_admin_check_for_one_of_several_workspaces_is_reported_as_is(client, access_key_registry):
    async def has_role(_self, workspace, role):
        if workspace == "team-b":
            raise HTTPException(status_code=503, detail="Role check service unavailable")
        return workspace == "team-a"

    with patch.object(AuthClient, "has_role", has_role):
        response = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b"))

    assert response.status_code == 503
    assert response.json()["detail"] == "Role check service unavailable"
    assert access_key_registry.keys == {}


def test_workspace_admin_cannot_unsuspend_a_key_whose_account_gained_access_elsewhere(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
    # A HelixAdmin gives the account access elsewhere, and suspends its team-a key.
    access_key_registry.bindings["service-account:team-a--otel"] = ["team-c"]
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        refused = client.post(f"/v2/access-keys/{created['jti']}/unsuspend")
        # Suspending and revoking only take access away, so they're still allowed.
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200

    assert refused.status_code == 403
    assert "only a HelixAdmin can create, rotate or unsuspend keys for it" in refused.json()["detail"]
    assert created["jti"] in access_key_registry.suspended
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.post(f"/v2/access-keys/{created['jti']}/unsuspend").status_code == 200


def test_workspace_admin_can_unsuspend_a_confined_key_but_not_one_for_a_disabled_account(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200
        assert client.post(f"/v2/access-keys/{created['jti']}/unsuspend").status_code == 200
        access_key_registry.bindings["service-account:team-a--otel"] = ["team-c"]
        # A key that is already active isn't changed, so there's nothing to refuse.
        unchanged = client.post(f"/v2/access-keys/{created['jti']}/unsuspend")
        assert unchanged.status_code == 200
        assert unchanged.json()["changed"] is False
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200
        assert client.post(f"/v2/access-keys/{created['jti']}/unsuspend").status_code == 403

        access_key_registry.bindings.clear()
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200
        access_key_registry.disabled_service_accounts.add("service-account:team-a--otel")
        disabled = client.post(f"/v2/access-keys/{created['jti']}/unsuspend")

    assert disabled.status_code == 400
    assert disabled.json()["detail"] == (
        "Service account 'team-a--otel' is disabled; enable it before unsuspending its keys"
    )


def test_unsuspend_is_rechecked_after_the_write_and_suspended_again_if_it_fails(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200
        unsuspend = access_key_registry.unsuspend

        async def unsuspend_after_a_grant_elsewhere(*args, **kwargs):
            # A role elsewhere lands between the check before the write and the write itself.
            access_key_registry.bindings["service-account:team-a--otel"] = ["team-c"]
            return await unsuspend(*args, **kwargs)

        access_key_registry.unsuspend = unsuspend_after_a_grant_elsewhere
        response = client.post(f"/v2/access-keys/{created['jti']}/unsuspend")

    assert response.status_code == 403
    assert "only a HelixAdmin can create, rotate or unsuspend keys for it" in response.json()["detail"]
    assert created["jti"] in access_key_registry.suspended


def test_unsuspending_a_revoked_bound_key_reports_the_conflict_before_confinement(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        assert client.delete(f"/v2/access-keys/{created['jti']}").status_code == 200
        access_key_registry.bindings["service-account:team-a--otel"] = ["team-c"]
        response = client.post(f"/v2/access-keys/{created['jti']}/unsuspend")

    assert response.status_code == 409
    assert response.json()["detail"] == f"Revoked Scoped Access Key {created['jti']} cannot be unsuspended"


def test_rotates_naming_a_partly_administered_service_key_is_a_bad_request(client):
    both = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))
    with patch.object(AuthClient, "has_role", both):
        created = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b")).json()

    # rotates never accepts a service-bound key, so more Admin roles wouldn't help: no 403.
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post("/v2/access-keys", json={"name": "personal", "rotates": created["jti"]})

    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"rotates references a Scoped Access Key that does not exist or is not owned by the caller: {created['jti']}"
    )


def test_rotation_revokes_a_successor_it_could_not_discard(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        add = access_key_registry.add

        async def add_then_grant_elsewhere(key, **kwargs):
            await add(key, **kwargs)
            # Fails the confinement check that runs once the successor's record is written.
            access_key_registry.bindings["service-account:team-a--otel"] = ["team-c"]

        async def discard_fails(jti):
            raise EntityStoreError("store unavailable")

        access_key_registry.add = add_then_grant_elsewhere
        access_key_registry.discard_unreturned = discard_fails
        response = client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert response.status_code == 403
    successors = set(access_key_registry.keys) - {created["jti"]}
    assert len(successors) == 1
    # Never returned, and it couldn't be deleted, so it's revoked rather than left live.
    assert successors <= access_key_registry.revoked


def test_bound_key_is_withdrawn_when_its_workspace_cannot_be_rechecked_after_granting(
    client, access_key_workspaces_client, access_key_registry
):
    record_member = access_key_workspaces_client.create_workspace_member.side_effect
    get_workspace = access_key_workspaces_client.get_workspace.side_effect
    granted = False

    async def grant(*, workspace, body):
        nonlocal granted
        granted = True
        await record_member(workspace=workspace, body=body)

    async def unreadable_once_granted(*, name):
        if granted:
            raise _transport_error()
        return await get_workspace(name=name)

    access_key_workspaces_client.create_workspace_member.side_effect = grant
    access_key_workspaces_client.get_workspace.side_effect = unreadable_once_granted
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post("/v2/access-keys", json=_bound_body("team-a"))

    assert response.status_code == 503
    # The membership is rolled back and the never-returned key discarded.
    access_key_workspaces_client.delete_workspace_member.assert_awaited_once_with(
        workspace="team-a", principal_id="service-account:team-a--otel"
    )
    assert access_key_registry.keys == {}
    assert len(access_key_registry.discarded) == 1


def test_rotation_discards_the_successor_when_its_post_insert_check_fails(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
    check = access_key_registry.service_account_access
    checks = 0

    async def fails_after_insert(*args, **kwargs):
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RuntimeError("entity store unavailable")
        return await check(*args, **kwargs)

    access_key_registry.service_account_access = fails_after_insert
    with (
        patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))),
        pytest.raises(RuntimeError, match="entity store unavailable"),
    ):
        client.post(f"/v2/access-keys/{created['jti']}/rotate")

    assert set(access_key_registry.keys) == {created["jti"]}
    assert len(access_key_registry.discarded) == 1
    assert created["jti"] not in access_key_registry.rotating


def test_listing_degrades_to_the_callers_own_keys_when_admin_bindings_cannot_be_read(client, access_key_registry):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        bound = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
    personal = client.post("/v2/access-keys", json={"name": "personal"}).json()
    access_key_registry.workspaces_with_role = AsyncMock(side_effect=EntityStoreError("entity store unavailable"))

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        listed = client.get("/v2/access-keys")

    assert listed.status_code == 200
    assert sorted(key["jti"] for key in listed.json()["data"]) == sorted([bound["jti"], personal["jti"]])


def test_helix_admin_listing_reads_the_bound_workspace_names_in_one_batch(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        for name, workspace in (("a", "team-a"), ("b", "team-b"), ("c", "team-c")):
            body = _bound_body(workspace, name=name, account=f"{workspace}--svc")
            assert client.post("/v2/access-keys", json=body).status_code == 200
        access_key_workspaces_client.list_workspaces.reset_mock()

        listed = client.get("/v2/access-keys")

    assert sorted(key["bound_workspaces"][0] for key in listed.json()["data"]) == ["team-a", "team-b", "team-c"]
    assert access_key_workspaces_client.list_workspaces.await_count == 1
    assert sorted(_name_lookups(access_key_workspaces_client)) == ["ws-id-team-a", "ws-id-team-b", "ws-id-team-c"]


def test_helix_admin_listing_shows_bound_workspace_ids_when_names_cannot_be_read(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
        access_key_workspaces_client.list_workspaces.side_effect = _transport_error()

        listed = client.get("/v2/access-keys")

    assert listed.status_code == 200
    assert [key["bound_workspaces"] for key in listed.json()["data"]] == [["ws-id-team-a"]]


def test_workspace_admin_learns_nothing_about_an_account_outside_their_namespace(client, access_key_registry):
    access_key_registry.disabled_service_accounts.add("service-account:platform-ops")

    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post("/v2/access-keys", json=_bound_body("team-a", account="platform-ops"))

    # The namespace 403, not the 400 that would reveal the account is disabled.
    assert response.status_code == 403
    assert "namespaced under a bound workspace" in response.json()["detail"]


def test_an_invalid_expiry_is_reported_before_asking_to_confirm_widening(client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"), ("team-b", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
        response = client.post(
            "/v2/access-keys", json={**_bound_body("team-a", "team-b", name="b"), "expires_in_seconds": 10**9}
        )

    assert response.status_code == 400
    assert "max_expires_in_seconds" in response.json()["detail"]


def test_an_acknowledged_key_is_created_even_if_its_widening_cannot_be_audited(client, access_key_registry, caplog):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
    access_key_registry.service_account_access = AsyncMock(side_effect=RuntimeError("entity store down"))
    elsewhere = {**_bound_body("team-b", name="platform"), "acknowledge_workspace_bound_keys": True}

    with (
        patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True),
        caplog.at_level(logging.WARNING, logger="nhx.core.auth.app.access_keys"),
    ):
        response = client.post("/v2/access-keys", json=elsewhere)

    assert response.status_code == 200
    events = [record for record in caplog.records if hasattr(record, "audit_event")]
    widened = [record for record in events if record.audit_event == "access_key.workspace_bound_keys_widened"]
    assert [record.workspace_ids for record in widened] == [None]


def test_create_discards_a_service_bound_record_written_despite_a_failed_add(client, access_key_registry):
    access_key_registry.add_commits_before_error = True
    access_key_registry.add_error = RuntimeError("simulated post-commit persistence failure")

    with (
        patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))),
        pytest.raises(RuntimeError, match="simulated post-commit persistence failure"),
    ):
        client.post("/v2/access-keys", json=_bound_body("team-a"))

    # The never-returned key can't linger as a live bound key for the account.
    assert access_key_registry.keys == {}
    assert len(access_key_registry.discarded) == 1


def test_a_revoked_key_no_longer_keeps_workspace_admins_from_an_account(client):
    # A HelixAdmin creates an unbound key for the account by mistake, then revokes it.
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        mistake = client.post("/v2/access-keys", json={"name": "oops", "service_account_id": "team-a--otel"}).json()
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 403
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        assert client.delete(f"/v2/access-keys/{mistake['jti']}").status_code == 200

    # The revoked key can't authenticate, so it no longer counts as access outside team-a.
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200


def test_admin_checks_run_at_most_a_few_workspaces_at_once(client):
    workspaces = [f"team-{letter}" for letter in "abcdefghijkl"]
    running = 0
    most_running = 0

    async def has_role(_self, workspace, role):
        nonlocal running, most_running
        if workspace == "system":
            return False
        running += 1
        most_running = max(most_running, running)
        await asyncio.sleep(0.01)
        running -= 1
        return True

    with patch.object(AuthClient, "has_role", has_role):
        response = client.post("/v2/access-keys", json=_bound_body(*workspaces))

    assert response.status_code == 200
    assert 1 < most_running <= 8


def test_workspaces_the_caller_cannot_see_are_counted_not_named(client, access_key_workspaces_client):
    every = _has_role_for(("team-a", "Admin"), ("team-b", "Admin"), ("secret", "Admin"))
    with patch.object(AuthClient, "has_role", every):
        created = client.post("/v2/access-keys", json=_bound_body("team-a", "team-b", "secret")).json()

    # The caller administers team-a, can see team-b, and can't see the private workspace at all.
    hidden_name = access_key_workspaces_client.list_workspaces.side_effect

    async def list_workspaces(*, query_params):
        if "ws-id-secret" in query_params["filter"]:
            return await hidden_name(query_params={**query_params, "filter": json.dumps({"id": "no-such-id"})})
        return await hidden_name(query_params=query_params)

    access_key_workspaces_client.list_workspaces.side_effect = list_workspaces
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        response = client.post(f"/v2/access-keys/{created['jti']}/suspend")

    assert response.status_code == 403
    assert response.json()["detail"] == (
        f"Scoped Access Key {created['jti']} is also bound to workspace(s) 'team-b', which you aren't Admin of, "
        "and 1 workspace(s) you can't see or that no longer exist; only a HelixAdmin, or an Admin of every "
        "workspace it's bound to, can suspend it"
    )
    assert "secret" not in response.text


def test_a_scoped_key_that_cannot_read_workspaces_is_told_so_when_naming_workspaces(
    client, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()

    # The workspaces service refuses lookups a key without entities or platform scope makes.
    refusal = PermissionDeniedError(httpx.Response(403, json={"detail": "Insufficient scope"}))
    access_key_workspaces_client.get_workspace.side_effect = refusal
    access_key_workspaces_client.list_workspaces.side_effect = refusal
    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
            refused = client.post("/v2/access-keys", json=_bound_body("team-a", name="another"))
            suspended = client.post(f"/v2/access-keys/{created['jti']}/suspend")
    finally:
        auth_client_context.reset(token)

    # Not the "require Admin" its refused workspace lookups would otherwise read as.
    assert refused.status_code == 400
    assert "must include entities or platform in its own scope" in refused.json()["detail"]
    # Managing an existing key names no workspace, and its refused lookups read as not
    # administered: a 404, so the key's existence isn't revealed either.
    assert suspended.status_code == 404


def test_helix_admin_with_a_scoped_key_that_cannot_read_workspaces_is_told_so(client):
    token = auth_client_context.set(_scoped_caller_auth_client(["auth"]))
    try:
        with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
            refused = client.post("/v2/access-keys", json=_bound_body("team-a"))
    finally:
        auth_client_context.reset(token)

    assert refused.status_code == 400
    assert "must include entities or platform in its own scope" in refused.json()["detail"]


def test_workspace_admin_cannot_use_an_account_first_issued_a_key_before_their_workspace_existed(
    client, access_key_registry
):
    # team-a was deleted and recreated since team-a--otel's first key: it belongs to the old team-a.
    access_key_registry.linked_at["service-account:team-a--otel"] = WORKSPACES_CREATED_AT - timedelta(days=1)
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        refused = client.post("/v2/access-keys", json=_bound_body("team-a"))

    assert refused.status_code == 403
    assert refused.json()["detail"] == (
        "Service account 'team-a--otel' was first issued a key before workspace 'team-a' existed, so it belongs "
        "to an earlier workspace of that name; only a HelixAdmin can create, rotate or unsuspend keys for it"
    )
    assert access_key_registry.keys == {}

    access_key_registry.linked_at["service-account:team-a--otel"] = WORKSPACES_CREATED_AT + timedelta(days=1)
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200


def test_granting_roles_the_account_already_holds_needs_no_acknowledgement(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
    # The account already holds Viewer in team-b, so the team-a key has it too.
    access_key_workspaces_client.granted_members["team-b"] = {"service-account:team-a--otel": ["Viewer"]}

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post("/v2/access-keys", json=_bound_body("team-b", name="b", roles=("Viewer",)))
        assert response.status_code == 200
        # A role it doesn't hold there yet would reach the team-a key.
        editor = client.post("/v2/access-keys", json=_bound_body("team-b", name="b-editor", roles=("Editor",)))
        assert editor.status_code == 409


def test_listing_keeps_the_admin_workspaces_read_before_a_lookup_failed(
    client, access_key_registry, access_key_workspaces_client
):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
    # Enough Admin workspaces for two lookups; the second fails.
    access_key_registry.admin_binding_workspaces = {"team-a", *(f"zz-{index:02}" for index in range(60))}
    list_workspaces = access_key_workspaces_client.list_workspaces.side_effect
    lookups = 0

    async def second_lookup_fails(*, query_params=None):
        nonlocal lookups
        lookups += 1
        if lookups == 2:
            raise _transport_error()
        return await list_workspaces(query_params=query_params)

    access_key_workspaces_client.list_workspaces.side_effect = second_lookup_fails
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        listed = client.get("/v2/access-keys")

    assert listed.status_code == 200
    assert [key["jti"] for key in listed.json()["data"]] == [created["jti"]]


def test_helix_admin_can_unsuspend_a_key_whose_service_account_is_disabled(client, access_key_registry):
    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        created = client.post("/v2/access-keys", json=_bound_body("team-a")).json()
        assert client.post(f"/v2/access-keys/{created['jti']}/suspend").status_code == 200
        access_key_registry.disabled_service_accounts.add("service-account:team-a--otel")

        # e.g. unsuspended first, then the account enabled again.
        assert client.post(f"/v2/access-keys/{created['jti']}/unsuspend").status_code == 200


def test_a_role_everyone_holds_counts_as_held(client, access_key_workspaces_client):
    with patch.object(AuthClient, "has_role", _has_role_for(("team-a", "Admin"))):
        assert client.post("/v2/access-keys", json=_bound_body("team-a")).status_code == 200
    # team-b is public: everyone, the account included, is a Viewer there.
    access_key_workspaces_client.granted_members["team-b"] = {"*": ["Viewer"]}

    with patch.object(AuthClient, "has_role", new_callable=AsyncMock, return_value=True):
        response = client.post("/v2/access-keys", json=_bound_body("team-b", name="b", roles=("Viewer",)))

    assert response.status_code == 200
