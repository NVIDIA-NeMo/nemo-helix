# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private storage bootstrap and durable cleanup lifecycle for the credential driver."""

import asyncio
import io
import json
import tarfile
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_deployments_plugin.backends.openshell.credential_driver import server
from nemo_deployments_plugin.backends.openshell.credential_driver.owned import OwnedCredentials
from nemo_deployments_plugin.backends.openshell.credential_driver.workspace import (
    OWNED_SECRETS_WORKSPACE,
    OwnedWorkspaceError,
    ensure_owned_workspace,
)
from nemo_helix_plugin.auth.types import AuthenticateResponse, ResolvedTokenKind
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.workspaces.types import WorkspaceMember

SERVICE = "service:openshell-credentials"
TIMESTAMP = "2026-11-06T00:00:00Z"


@dataclass
class WorkspaceAPI:
    exists: bool = True
    identity: AuthenticateResponse = field(
        default_factory=lambda: AuthenticateResponse(principal=SERVICE, token_kind="workload_access_token")
    )
    members: list[WorkspaceMember] = field(default_factory=list)
    policy: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    calls: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path == "/apis/auth/authenticate":
            return httpx.Response(200, json=self.identity.model_dump())
        if path == "/apis/auth/v2/iam/opa-bundle.tar.gz":
            payload = json.dumps(
                {"authz": {"principals": {key: {"workspaces": value} for key, value in self.policy.items()}}}
            ).encode()
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                entry = tarfile.TarInfo("data.json")
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))
            return httpx.Response(200, content=buffer.getvalue(), headers={"Content-Type": "application/gzip"})
        if path == "/apis/auth/v2/authz/has_permissions":
            body = json.loads(request.content)["input"]
            assert body["principal_id"] == SERVICE
            assert body["workspace"] == OWNED_SECRETS_WORKSPACE
            assert "secrets.access" in body["permissions"]
            roles = [role for member in self.members if member.principal == SERVICE for role in member.roles]
            return httpx.Response(200, json={"result": {"allowed": not roles or "ServiceSystem" in roles}})
        if path == "/apis/auth/v2/iam/role-bindings":
            assert request.method == "POST"
            body = json.loads(request.content)
            assert body == {"principal": SERVICE, "workspace": OWNED_SECRETS_WORKSPACE, "role": "ServiceSystem"}
            self.members.append(WorkspaceMember(principal=SERVICE, roles=["ServiceSystem"]))
            return httpx.Response(
                201,
                json={
                    **body,
                    "name": "binding",
                    "id": "binding-id",
                    "granted_by": SERVICE,
                    "granted_at": TIMESTAMP,
                    "revoked_at": None,
                },
            )
        if path.endswith("/members"):
            assert request.method == "GET"
            return httpx.Response(200, json={"data": [member.model_dump() for member in self.members]})
        status = 200
        if request.method == "POST":
            assert path == "/apis/entities/v2/workspaces"
            self.exists = True
            self.members.append(WorkspaceMember(principal=SERVICE, roles=["Admin"]))
            status = 201
        else:
            assert path == f"/apis/entities/v2/workspaces/{OWNED_SECRETS_WORKSPACE}"
            if not self.exists:
                return httpx.Response(404, json={"detail": "Not found"})
        return httpx.Response(
            status,
            json={
                "id": "workspace-id",
                "name": OWNED_SECRETS_WORKSPACE,
                "created_by": SERVICE,
                "created_at": TIMESTAMP,
                "updated_at": TIMESTAMP,
            },
        )

    def client(self) -> AsyncSecretsClient:
        return AsyncSecretsClient(
            base_url="https://helix.example",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handle)),
            owns_http_client=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("principal", "token_kind", "on_behalf_of"),
    [
        ("user@example.com", "oidc_access_token", None),
        ("service-account:agent", "access_key", None),
        (SERVICE, "workload_subject_token", None),
        (SERVICE, "workload_access_token", "user@example.com"),
    ],
)
async def test_bootstrap_requires_authenticated_direct_service(
    principal: str, token_kind: ResolvedTokenKind, on_behalf_of: str | None
) -> None:
    api = WorkspaceAPI(
        identity=AuthenticateResponse(principal=principal, token_kind=token_kind, on_behalf_of=on_behalf_of)
    )
    async with api.client() as client:
        with pytest.raises(OwnedWorkspaceError, match="direct internal service"):
            await ensure_owned_workspace(client, auth_enabled=True)
    assert [request.url.path for request in api.calls] == ["/apis/auth/authenticate"]


@pytest.mark.asyncio
@pytest.mark.parametrize("principal", ["*", "user@example.com", "service-account:agent"])
@pytest.mark.parametrize("static", [True, False])
async def test_rejects_ordinary_grants_without_rewriting_them(principal: str, static: bool) -> None:
    api = WorkspaceAPI()
    if static:
        api.policy[principal] = {OWNED_SECRETS_WORKSPACE: ["Editor"]}
    else:
        api.members.append(WorkspaceMember(principal=principal, roles=["Editor"]))
    async with api.client() as client:
        with pytest.raises(OwnedWorkspaceError, match="ordinary principals"):
            await ensure_owned_workspace(client, auth_enabled=True)
    assert all(request.method == "GET" for request in api.calls)


@pytest.mark.asyncio
async def test_existing_workspace_preserves_service_and_platform_admin_grants() -> None:
    api = WorkspaceAPI(
        members=[
            WorkspaceMember(principal=SERVICE, roles=["ServiceSystem"]),
            WorkspaceMember(principal="operator@example.com", roles=["Admin"]),
        ],
        policy={"operator@example.com": {"system": ["HelixAdmin"], OWNED_SECRETS_WORKSPACE: ["Admin"]}},
    )
    async with api.client() as client:
        assert await ensure_owned_workspace(client, auth_enabled=True) == OWNED_SECRETS_WORKSPACE
    assert not any(request.url.path.endswith("/role-bindings") for request in api.calls)


@pytest.mark.asyncio
async def test_new_workspace_adds_service_access_lost_by_creator_admin_binding() -> None:
    api = WorkspaceAPI(exists=False)
    async with api.client() as client:
        await ensure_owned_workspace(client, auth_enabled=True)
    assert [member.roles for member in api.members] == [["Admin"], ["ServiceSystem"]]
    assert not any(request.method in {"DELETE", "PUT", "PATCH"} for request in api.calls)


@pytest.mark.asyncio
async def test_existing_restricted_service_is_not_silently_elevated() -> None:
    api = WorkspaceAPI(members=[WorkspaceMember(principal=SERVICE, roles=["Admin"])])
    async with api.client() as client:
        with pytest.raises(OwnedWorkspaceError, match="lacks owned-credential"):
            await ensure_owned_workspace(client, auth_enabled=True)
    assert not any(request.url.path.endswith("/role-bindings") for request in api.calls)


@pytest.mark.asyncio
async def test_auth_disabled_bootstrap_does_not_depend_on_iam() -> None:
    api = WorkspaceAPI(exists=False)
    async with api.client() as client:
        await ensure_owned_workspace(client, auth_enabled=False)
    assert all(request.url.path.startswith("/apis/entities/") for request in api.calls)


@pytest.mark.asyncio
async def test_cleanup_retries_redacts_failures_and_stops_with_server(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    interval = asyncio.Event()
    attempted = asyncio.Event()
    recovered = asyncio.Event()
    attempts = 0

    async def reconcile() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            attempted.set()
            raise RuntimeError("private credential material")
        recovered.set()

    async def next_interval(seconds: float) -> None:
        assert seconds == 30
        await interval.wait()
        interval.clear()

    owned = AsyncMock(spec=OwnedCredentials)
    owned.reconcile.side_effect = reconcile
    monkeypatch.setattr(server.asyncio, "sleep", next_interval)
    async with server.owned_credential_recovery(owned):
        await asyncio.wait_for(attempted.wait(), timeout=1)
        assert attempts == 1
        interval.set()
        await asyncio.wait_for(recovered.wait(), timeout=1)
        assert attempts == 2
    assert "retained deletion intent" in caplog.text
    assert "private credential material" not in caplog.text
    assert not any(task.get_name() == "owned-credential-cleanup" for task in asyncio.all_tasks())


@pytest.mark.asyncio
async def test_pending_cleanup_does_not_block_startup_and_is_cancelled_at_shutdown() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def reconcile() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    owned = AsyncMock(spec=OwnedCredentials)
    owned.reconcile.side_effect = reconcile
    async with asyncio.timeout(1), server.owned_credential_recovery(owned):
        await started.wait()
    assert cancelled.is_set()
