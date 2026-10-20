# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration and wire-contract tests, without a running Helix or gateway."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from nemo_deployments_plugin.backends.openshell.credential_driver.registry import BindingRegistry, ProviderOwner
from nemo_deployments_plugin.backends.openshell.credential_driver.tickets import (
    BindingScope,
    RegistrationDenied,
    RegistrationTickets,
)
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from pydantic import SecretStr
from sqlalchemy.exc import OperationalError

KEY = b"test-only-registration-key-32-bytes"


class Harness:
    def __init__(self, directory: Path) -> None:
        self.now = 1_800_000_000
        self.clock = lambda: self.now
        self.tickets = RegistrationTickets(KEY, "gateway-a", clock=self.clock)
        self.registry = BindingRegistry(directory / "bindings.db", self.tickets, clock=self.clock)
        self.scope = BindingScope(
            gateway_id="gateway-a",
            gateway_workspace="openshell-team",
            provider="job-provider",
            credential_key="API_TOKEN",
            secret_workspace="helix-team",
            secret_name="service-token",
            job_id="job-1",
        )
        self.owner = ProviderOwner(
            workspace="openshell-team", provider="job-provider", provider_id="provider-uuid", credential_key="API_TOKEN"
        )
        self.calls: list[httpx.Request] = []
        self.secret = "real-secret-never-persist-this"
        self.status = 200
        self.client = AsyncSecretsClient(
            base_url="https://helix.example",
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(self.http),
            ),
            owns_http_client=True,
        )

    def http(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": self.secret})
        assert "/workspaces/helix-team/secrets/service-token" in request.url.path
        payload = {"name": "service-token", "workspace": "helix-team"}
        if request.url.path.endswith("/access"):
            payload["value"] = self.secret
        return httpx.Response(200, json=payload)

    async def issue(self) -> SecretStr:
        return await self.tickets.issue(self.scope, requester_secrets=self.client)


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.mark.asyncio
async def test_register_restart_retry_and_delete_preserve_user_secret(harness: Harness) -> None:
    ticket = await harness.issue()
    handle = harness.registry.register(harness.owner, ticket)
    assert harness.registry.register(harness.owner, ticket) == handle
    assert all(not request.url.path.endswith("/access") for request in harness.calls)
    db_contents = harness.registry.path.read_bytes()
    assert ticket.get_secret_value().encode() not in db_contents
    assert harness.secret.encode() not in db_contents
    restarted = BindingRegistry(harness.registry.path, harness.tickets, clock=harness.clock)
    assert restarted.resolve(harness.owner, handle).claims.secret_name == "service-token"
    restarted.delete(harness.owner, handle)
    restarted.delete(harness.owner, handle)
    with pytest.raises(RegistrationDenied):
        restarted.register(harness.owner, ticket)
    with pytest.raises(RegistrationDenied):
        restarted.resolve(harness.owner, handle)
    assert all(request.method == "GET" and not request.url.path.endswith("/access") for request in harness.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["provider", "provider_id", "workspace", "credential_key"])
async def test_cross_provider_registration_resolution_and_deletion_denied(harness: Harness, field: str) -> None:
    ticket = await harness.issue()
    handle = harness.registry.register(harness.owner, ticket)
    other = harness.owner.model_copy(update={field: "other"})
    with pytest.raises(RegistrationDenied):
        harness.registry.register(other, ticket)
    with pytest.raises(RegistrationDenied):
        harness.registry.resolve(other, handle)
    with pytest.raises(RegistrationDenied):
        harness.registry.delete(other, handle)
    assert harness.registry.resolve(harness.owner, handle)


@pytest.mark.asyncio
async def test_ticket_tampering_expiry_and_gateway_audience(harness: Harness) -> None:
    ticket = await harness.issue()
    raw = ticket.get_secret_value()
    for invalid in (SecretStr("raw-api-key"), SecretStr(raw[:-1] + ("0" if raw[-1] != "0" else "1"))):
        with pytest.raises(RegistrationDenied):
            harness.registry.register(harness.owner, invalid)
    other_gateway = RegistrationTickets(KEY, "gateway-b", clock=harness.clock)
    with pytest.raises(RegistrationDenied):
        other_gateway.verify(ticket)
    with pytest.raises(ValueError, match="another gateway"):
        BindingRegistry(harness.registry.path, other_gateway)
    harness.now += 301
    with pytest.raises(RegistrationDenied):
        harness.registry.register(harness.owner, ticket)


@pytest.mark.asyncio
async def test_issuer_checks_requester_permission_before_signing(harness: Harness) -> None:
    harness.status = 403
    with pytest.raises(Exception) as error:
        await harness.issue()
    assert getattr(error.value, "status_code", None) == 403
    assert len(harness.calls) == 1
    assert not harness.calls[0].url.path.endswith("/access")


@pytest.mark.asyncio
async def test_revoke_job_blocks_pending_and_registered_tickets(harness: Harness) -> None:
    ticket = await harness.issue()
    pending = await harness.issue()
    handle = harness.registry.register(harness.owner, ticket)
    harness.registry.revoke_job("job-1")
    with pytest.raises(RegistrationDenied):
        harness.registry.resolve(harness.owner, handle)
    with pytest.raises(RegistrationDenied):
        harness.registry.register(harness.owner, pending)


@pytest.mark.asyncio
async def test_staged_update_does_not_revoke_old_handle_until_gateway_commit(harness: Harness) -> None:
    old = harness.registry.register(harness.owner, await harness.issue())
    ticket = await harness.issue()
    new = harness.registry.register(harness.owner, ticket, existing_handle=old, object_id="write-2")
    assert new != old
    assert harness.registry.resolve(harness.owner, old)
    assert harness.registry.resolve(harness.owner, new)
    assert harness.registry.register(harness.owner, ticket, existing_handle=old, object_id="write-2") == new
    # Gateway rolls back the staged write: old still works.
    harness.registry.delete(harness.owner, new)
    assert harness.registry.resolve(harness.owner, old)
    harness.now += 3600
    with pytest.raises(RegistrationDenied):
        harness.registry.resolve(harness.owner, old)


@pytest.mark.asyncio
async def test_concurrent_registration_retries_produce_one_handle(harness: Harness) -> None:
    ticket = await harness.issue()
    with ThreadPoolExecutor(max_workers=4) as pool:
        handles = list(pool.map(lambda _: harness.registry.register(harness.owner, ticket), range(8)))
    assert len(set(handles)) == 1


@pytest.mark.asyncio
async def test_concurrent_registry_startup_and_registration(harness: Harness, tmp_path: Path) -> None:
    ticket = await harness.issue()
    path = tmp_path / "concurrent.db"

    def register(_: int) -> str:
        registry = BindingRegistry(path, harness.tickets, clock=harness.clock)
        return registry.register(harness.owner, ticket)

    with ThreadPoolExecutor(max_workers=4) as pool:
        handles = list(pool.map(register, range(8)))
    assert len(set(handles)) == 1


@pytest.mark.parametrize("size", [0, 64 * 1024, 64 * 1024 + 1])
def test_credential_file_size_boundary(tmp_path: Path, size: int) -> None:
    pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver.server import read_private_file

    path = tmp_path / "credential"
    path.touch(mode=0o600)
    path.write_bytes(b"x" * size)
    if size == 64 * 1024:
        assert len(read_private_file(path)) == size
    else:
        with pytest.raises(ValueError, match="file size"):
            read_private_file(path)


@pytest.mark.asyncio
async def test_existing_handle_cannot_be_borrowed_from_another_provider(harness: Harness) -> None:
    old = harness.registry.register(harness.owner, await harness.issue())
    other_scope = harness.scope.model_copy(update={"provider": "other-provider"})
    other = harness.owner.model_copy(update={"provider": "other-provider", "provider_id": "other-uuid"})
    ticket = await harness.tickets.issue(other_scope, requester_secrets=harness.client)
    with pytest.raises(RegistrationDenied):
        harness.registry.register(other, ticket, existing_handle=old)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gateway",
    [
        {},
        {"protocol_version": {"major": 2}, "supported_capabilities": ["openshell.credentials.contract"]},
        {"protocol_version": {"major": 1}, "supported_capabilities": []},
        {
            "protocol_version": {"major": 1},
            "supported_capabilities": ["openshell.credentials.contract"],
            "required_capabilities": ["unsupported-feature"],
        },
    ],
)
async def test_protocol_negotiation_rejects_unmet_contract(harness: Harness, gateway: dict) -> None:
    pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    driver = CredentialDriver(harness.registry, harness.client, clock=harness.clock)
    with pytest.raises(RegistrationDenied):
        await driver._dispatch("GetCapabilities", {"gateway": gateway})


@pytest.mark.asyncio
async def test_invalid_batch_does_not_read_any_secrets(harness: Harness) -> None:
    pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    handle = harness.registry.register(harness.owner, await harness.issue())
    harness.calls.clear()
    lookup = {**harness.owner.model_dump(), "handle": {"driver": "nemo", "handle": handle}, "request_id": "one"}
    driver = CredentialDriver(harness.registry, harness.client, clock=harness.clock)
    with pytest.raises(RegistrationDenied):
        await driver._dispatch(
            "ResolveCredentials",
            {
                "credentials": [
                    lookup,
                    {
                        **lookup,
                        "provider_id": "other",
                        "request_id": "two",
                    },
                ]
            },
        )
    assert not harness.calls


@pytest.fixture
def socket_directory() -> Iterator[Path]:
    # AF_UNIX paths must fit the host limit, including on macOS.
    with tempfile.TemporaryDirectory(prefix="nemo-cred-", dir="/tmp") as name:
        yield Path(name)


@pytest.mark.asyncio
async def test_real_grpc_registration_resolution_and_delete(harness: Harness, socket_directory: Path) -> None:
    grpc = pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver import wire
    from nemo_deployments_plugin.backends.openshell.credential_driver.server import running_server
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CAPABILITY, CredentialDriver

    driver = CredentialDriver(harness.registry, harness.client, clock=harness.clock)
    socket = socket_directory / "driver.sock"
    async with running_server(driver, socket):
        async with grpc.aio.insecure_channel(f"unix:{socket}") as channel:

            async def call(method: str, data: dict) -> dict:
                request_name, response_name = wire.METHODS[method]
                rpc = channel.unary_unary(
                    f"/{wire.SERVICE}/{method}",
                    request_serializer=wire.message_type(request_name).SerializeToString,
                    response_deserializer=wire.message_type(response_name).FromString,
                )
                return wire.data(await rpc(wire.message(request_name, data), timeout=5))

            caps = await call(
                "GetCapabilities",
                {
                    "gateway": {
                        "protocol_version": {"major": 1},
                        "supported_capabilities": [CAPABILITY],
                        "required_capabilities": [CAPABILITY],
                    }
                },
            )
            assert caps["supports_expires_at"] is True
            owner = harness.owner.model_dump()
            ticket = await harness.issue()
            stored = await call("StoreCredential", {**owner, "value": ticket.get_secret_value()})
            handle = stored["handle"]
            lookup = {**owner, "handle": handle, "request_id": "read-1"}
            result = await call("ResolveCredentials", {"credentials": [lookup]})
            assert result["credentials"][0]["value"] == harness.secret
            assert result["credentials"][0]["request_id"] == "read-1"
            assert datetime.fromisoformat(result["credentials"][0]["expiration_time"]).timestamp() == harness.now + 30
            # Rotation is visible on the next driver resolution; no value cache.
            harness.secret = "rotated-secret"
            result = await call("ResolveCredentials", {"credentials": [lookup]})
            assert result["credentials"][0]["value"] == "rotated-secret"
            for status in (403, 404, 503):
                harness.status = status
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await call("ResolveCredentials", {"credentials": [lookup]})
                expected = grpc.StatusCode.UNAVAILABLE if status == 503 else grpc.StatusCode.PERMISSION_DENIED
                assert error.value.code() == expected
                assert harness.secret not in error.value.details()
            harness.status = 200
            with patch.object(
                harness.registry, "resolve", side_effect=OperationalError("SELECT", {}, Exception(harness.secret))
            ):
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await call("ResolveCredentials", {"credentials": [lookup]})
                assert error.value.code() == grpc.StatusCode.UNAVAILABLE
                assert harness.secret not in error.value.details()
            await call("DeleteCredential", {**owner, "handle": handle})
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call("ResolveCredentials", {"credentials": [lookup]})
            assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED
            assert harness.secret not in error.value.details()
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call("ListCredentials", {})
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
    assert not socket.exists()


@pytest.mark.asyncio
async def test_driver_descriptor_can_coexist_with_pinned_sdk(harness: Harness) -> None:
    dm = pytest.importorskip("openshell._proto.datamodel_pb2")
    pb = pytest.importorskip("openshell._proto.openshell_pb2")
    from nemo_deployments_plugin.backends.openshell.credential_driver import wire
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    ticket = await harness.issue()
    # The normal CreateProvider request carries a ticket in credentials, never
    # a caller-supplied credential_handle. Simulate only the gateway's documented
    # StoreCredential handoff; this is not a live gateway integration test.
    create = pb.CreateProviderRequest(
        workspace_scope=dm.WorkspaceSelector(workspace=harness.scope.gateway_workspace),
        provider=dm.Provider(
            metadata=dm.ObjectMeta(name=harness.scope.provider),
            type="service-read",
            credentials={harness.scope.credential_key: ticket.get_secret_value()},
        ),
    )
    assert not create.provider.credential_handles
    driver = CredentialDriver(harness.registry, harness.client, clock=harness.clock)
    response = await driver._dispatch(
        "StoreCredential",
        {
            "workspace": create.workspace_scope.workspace,
            "provider": create.provider.metadata.name,
            "provider_id": harness.owner.provider_id,
            "credential_key": "API_TOKEN",
            "value": create.provider.credentials["API_TOKEN"],
        },
    )
    result = wire.data(response)
    # The returned handle is also wire-compatible with the installed SDK.
    sdk_handle = dm.CredentialHandle.FromString(getattr(response, "handle").SerializeToString())
    assert sdk_handle.handle == result["handle"]["handle"]
    assert sdk_handle.driver == "nemo"


@pytest.mark.asyncio
async def test_revoke_during_secret_read_wins(harness: Harness) -> None:
    pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    handle = harness.registry.register(harness.owner, await harness.issue())
    original = harness.client.access_secret

    async def access(**kwargs):
        response = await original(**kwargs)
        harness.registry.revoke_job("job-1")
        return response

    harness.client.access_secret = AsyncMock(side_effect=access)
    driver = CredentialDriver(harness.registry, harness.client, clock=harness.clock)
    with pytest.raises(RegistrationDenied):
        await driver._dispatch(
            "ResolveCredentials",
            {
                "credentials": [
                    {
                        **harness.owner.model_dump(),
                        "handle": {"driver": "nemo", "handle": handle},
                        "request_id": "read",
                    }
                ]
            },
        )
