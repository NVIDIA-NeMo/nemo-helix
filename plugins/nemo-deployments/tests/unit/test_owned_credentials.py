# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recovery and ownership boundaries for OpenShell's internal credential storage."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from nemo_deployments_plugin.backends.openshell.credential_driver import server, wire
from nemo_deployments_plugin.backends.openshell.credential_driver.owned import OwnedCredentials
from nemo_deployments_plugin.backends.openshell.credential_driver.registry import (
    OWNED_HANDLE_PREFIX,
    BindingRegistry,
    OwnedState,
    ProviderOwner,
)
from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver
from nemo_deployments_plugin.backends.openshell.credential_driver.tickets import (
    BindingScope,
    RegistrationDenied,
    RegistrationTickets,
)
from nemo_helix_plugin.client.errors import NemoHTTPError, NemoTransportError
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from pydantic import SecretStr

WORKSPACE = "gateway-private"
VALUE = SecretStr("synthetic-internal-key-not-for-production")
USER_SECRET = ("users", "provider-token")
type CreateTimeout = Literal["before_commit", "after_commit"]


class Harness:
    """Real registry and typed client with controllable Secrets responses."""

    def __init__(self, directory: Path) -> None:
        self.tickets = RegistrationTickets(b"test-only-registration-key-32-bytes", "gateway")
        self.registry = BindingRegistry(directory / "bindings.db", self.tickets)
        self.owner = ProviderOwner(
            workspace="gateway-workspace",
            provider="sandbox-ssh-sandbox-id",
            provider_id="sandbox-id",
            credential_key="SSH_HOST_PRIVATE_KEY",
        )
        self.values: dict[tuple[str, str], str] = {USER_SECRET: "synthetic-provider-token"}
        self.calls: list[tuple[str, str]] = []
        self.create_timeout: CreateTimeout | None = None
        self.failed_deletes: set[str] = set()
        self.create_started = asyncio.Event()
        self.create_release = asyncio.Event()
        self.create_release.set()
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()
        self.read_release.set()
        self.client = AsyncSecretsClient(
            base_url="https://helix.example",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.http)),
            owns_http_client=True,
        )
        self.owned = OwnedCredentials(self.registry, self.client, WORKSPACE)

    def restart(self) -> OwnedCredentials:
        return OwnedCredentials(BindingRegistry(self.registry.path, self.tickets), self.client, WORKSPACE)

    async def http(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        match request.method, path.strip("/").split("/"):
            case "POST", ["apis", "secrets", "v2", "workspaces", workspace, "secrets"]:
                return await self._create_secret(request, workspace)
            case "DELETE", ["apis", "secrets", "v2", "workspaces", workspace, "secrets", name]:
                return self._delete_secret(workspace, name)
            case "GET", ["apis", "secrets", "v2", "workspaces", workspace, "secrets", name, "access"]:
                return await self._access_secret(workspace, name)
            case _:
                raise RuntimeError(f"Unexpected test request: {request.method} {path}")

    async def _create_secret(self, request: httpx.Request, workspace: str) -> httpx.Response:
        secret = HelixSecretCreateRequest.model_validate_json(request.content)
        self.create_started.set()
        await self.create_release.wait()
        if self.create_timeout == "before_commit":
            raise httpx.ReadTimeout("synthetic lost response", request=request)
        if (workspace, secret.name) in self.values:
            return httpx.Response(409, json={"detail": "Already exists"})
        self.values[workspace, secret.name] = secret.value.get_secret_value()
        if self.create_timeout == "after_commit":
            raise httpx.ReadTimeout("synthetic lost response", request=request)
        return httpx.Response(201, json={"name": secret.name, "workspace": workspace})

    def _delete_secret(self, workspace: str, name: str) -> httpx.Response:
        if name in self.failed_deletes:
            return httpx.Response(503, json={"detail": "synthetic outage"})
        if self.values.pop((workspace, name), None) is None:
            return httpx.Response(404, json={"detail": "Not found"})
        return httpx.Response(204)

    async def _access_secret(self, workspace: str, name: str) -> httpx.Response:
        value = self.values.get((workspace, name))
        if value is None:
            return httpx.Response(404, json={"detail": "Not found"})
        self.read_started.set()
        await self.read_release.wait()
        return httpx.Response(200, json={"name": name, "workspace": workspace, "value": value})


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[Harness]:
    harness = Harness(tmp_path)
    yield harness
    await harness.client.aclose()


@pytest.mark.asyncio
async def test_retry_and_restart_reuse_handle_without_persisting_value(harness: Harness) -> None:
    handle = await harness.owned.store(harness.owner, VALUE, object_id="operation")
    assert handle.startswith(OWNED_HANDLE_PREFIX)
    restarted = harness.restart()
    assert await restarted.store(harness.owner, VALUE, object_id="operation") == handle
    assert await restarted.resolve(harness.owner, handle) == VALUE.get_secret_value()
    assert sum(method == "POST" for method, _ in harness.calls) == 1
    assert VALUE.get_secret_value().encode() not in harness.registry.path.read_bytes()
    with pytest.raises(RegistrationDenied):
        await restarted.store(harness.owner, SecretStr("different-value"), object_id="operation")
    await restarted.delete(harness.owner, handle)
    before = len(harness.calls)
    await harness.restart().reconcile()
    assert len(harness.calls) == before  # Confirmed creations need no permanent delete sweep.
    with pytest.raises(RegistrationDenied):
        await restarted.store(harness.owner, VALUE, object_id="operation")
    assert harness.values == {USER_SECRET: "synthetic-provider-token"}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "active", "deleting", "deleted"])
async def test_startup_rejects_workspace_change_for_persisted_handles(harness: Harness, state: OwnedState) -> None:
    binding = harness.registry.reserve_owned(
        harness.owner, VALUE, workspace=WORKSPACE, object_id="operation", existing_handle=""
    )
    if state == "active":
        harness.registry.activate_owned(harness.owner, binding.handle, retain_uncertainty=False)
    elif state in {"deleting", "deleted"}:
        harness.registry.revoke_owned(harness.owner, binding.handle)
        if state == "deleted":
            harness.registry.finish_owned_delete(harness.owner, binding.handle)
    restarted = BindingRegistry(harness.registry.path, harness.tickets)
    restarted.require_owned_workspace(WORKSPACE)
    key_path = harness.registry.path.parent / "registration.key"
    key_path.touch(mode=0o600)
    key_path.write_bytes(b"test-only-registration-key-32-bytes")
    args = Namespace(
        helix_url="https://helix.example",
        gateway_id="gateway",
        registration_key_file=key_path,
        state_path=harness.registry.path,
        owned_secrets_workspace="openshell-internal",
        auth_disabled=True,
        cache_lifetime_seconds=30,
        socket_path=harness.registry.path.parent / "driver.sock",
    )
    with (
        patch.object(server, "AsyncSecretsClient", side_effect=AssertionError("Startup contacted Helix")),
        pytest.raises(RegistrationDenied, match="workspace differs"),
    ):
        await server.serve(args)
    assert restarted.get_owned(harness.owner, binding.handle).state == state


@pytest.mark.asyncio
async def test_first_store_prevents_another_driver_selecting_a_different_workspace(harness: Harness) -> None:
    second = BindingRegistry(harness.registry.path, harness.tickets)
    harness.registry.require_owned_workspace(WORKSPACE)
    second.require_owned_workspace("other-private")
    handle = await harness.owned.store(harness.owner, VALUE, object_id="first")
    before = len(harness.calls)
    other = OwnedCredentials(second, harness.client, "other-private")
    with pytest.raises(RegistrationDenied, match="workspace differs"):
        await other.store(harness.owner, VALUE, object_id="second")
    assert len(harness.calls) == before
    assert await harness.owned.resolve(harness.owner, handle) == VALUE.get_secret_value()


@pytest.mark.asyncio
async def test_restore_without_digest_key_does_not_replace_it(harness: Harness) -> None:
    handle = await harness.owned.store(harness.owner, VALUE, object_id="operation")
    key_path = harness.registry.path.with_suffix(".db.owned-key")
    key_path.unlink()
    with pytest.raises(ValueError, match="key is missing"):
        await harness.restart().resolve(harness.owner, handle)
    assert not key_path.exists()


@pytest.mark.asyncio
async def test_replacement_keeps_previous_handle_until_gateway_deletes_it(harness: Harness) -> None:
    old = await harness.owned.store(harness.owner, VALUE)
    new = await harness.owned.store(harness.owner, SecretStr("replacement"), existing_handle=old)
    assert new != old
    assert await harness.owned.resolve(harness.owner, old) == VALUE.get_secret_value()
    assert await harness.owned.resolve(harness.owner, new) == "replacement"
    other = harness.owner.model_copy(update={"provider_id": "other-sandbox"})
    before = len(harness.calls)
    with pytest.raises(RegistrationDenied):
        await harness.owned.delete(other, old)
    with pytest.raises(RegistrationDenied):
        await harness.owned.store(other, VALUE, existing_handle=old)
    with pytest.raises(RegistrationDenied):
        await harness.owned.store(harness.owner, VALUE, existing_handle="user-reference-handle")
    assert len(harness.calls) == before
    await harness.owned.delete(harness.owner, old)
    assert await harness.owned.resolve(harness.owner, new) == "replacement"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_phase", ["before_commit", "after_commit"])
async def test_ambiguous_create_recovers_after_restart(harness: Harness, timeout_phase: CreateTimeout) -> None:
    harness.create_timeout = timeout_phase
    with pytest.raises(NemoTransportError):
        await harness.owned.store(harness.owner, VALUE, object_id="operation")
    pending = harness.registry.pending_owned()[0]
    assert pending.state == "pending" and pending.create_uncertain
    with pytest.raises(RegistrationDenied):
        harness.owned.validate(harness.owner, pending.handle)
    restarted = harness.restart()
    await restarted.reconcile()
    assert sum(method == "POST" for method, _ in harness.calls) == 1
    harness.create_timeout = None
    assert await restarted.store(harness.owner, VALUE, object_id="operation") == pending.handle
    assert await restarted.resolve(harness.owner, pending.handle) == VALUE.get_secret_value()
    assert harness.registry.get_owned(harness.owner, pending.handle).create_uncertain


@pytest.mark.asyncio
async def test_conflicting_secret_is_not_adopted_or_overwritten(harness: Harness) -> None:
    pending = harness.registry.reserve_owned(
        harness.owner, VALUE, workspace=WORKSPACE, object_id="operation", existing_handle=""
    )
    harness.values[WORKSPACE, pending.name] = "unrelated-existing-value"
    with pytest.raises(RegistrationDenied, match="does not match"):
        await harness.owned.store(harness.owner, VALUE, object_id="operation")
    with pytest.raises(RegistrationDenied):
        harness.owned.validate(harness.owner, pending.handle)
    assert harness.values[WORKSPACE, pending.name] == "unrelated-existing-value"


@pytest.mark.asyncio
async def test_failed_delete_stays_revoked_and_does_not_starve_other_cleanup(harness: Harness) -> None:
    handles = [await harness.owned.store(harness.owner, VALUE, object_id=str(index)) for index in range(2)]
    first = harness.registry.get_owned(harness.owner, handles[0])
    harness.failed_deletes.add(first.name)
    with pytest.raises(NemoHTTPError):
        await harness.owned.delete(harness.owner, handles[0])
    harness.registry.revoke_owned(harness.owner, handles[1])
    with pytest.raises(RegistrationDenied):
        harness.owned.validate(harness.owner, handles[0])
    restarted = harness.restart()
    with pytest.raises(RuntimeError, match="1 records"):
        await restarted.reconcile()
    assert harness.registry.get_owned(harness.owner, handles[1]).state == "deleted"
    harness.failed_deletes.clear()
    # Simulate a prior successful deletion whose response was lost.
    del harness.values[WORKSPACE, first.name]
    await restarted.reconcile()
    assert not harness.registry.pending_owned()
    assert harness.values == {USER_SECRET: "synthetic-provider-token"}


@pytest.mark.asyncio
async def test_delayed_create_after_delete_404_is_cleaned_by_recovery(harness: Harness) -> None:
    harness.create_timeout = "before_commit"
    with pytest.raises(NemoTransportError):
        await harness.owned.store(harness.owner, VALUE, object_id="operation")
    pending = harness.registry.pending_owned()[0]
    await harness.owned.delete(harness.owner, pending.handle)
    # The server finishes the timed-out POST after the DELETE saw no secret.
    harness.values[WORKSPACE, pending.name] = VALUE.get_secret_value()
    await harness.restart().reconcile()
    assert harness.values == {USER_SECRET: "synthetic-provider-token"}
    assert harness.registry.get_owned(harness.owner, pending.handle).state == "deleting"
    with pytest.raises(RegistrationDenied):
        await harness.owned.store(harness.owner, VALUE, object_id="operation")


@pytest.mark.asyncio
async def test_concurrent_store_retries_share_one_remote_create(harness: Harness) -> None:
    harness.create_release.clear()
    first = asyncio.create_task(harness.owned.store(harness.owner, VALUE, object_id="operation"))
    await asyncio.wait_for(harness.create_started.wait(), 2)
    second = asyncio.create_task(harness.restart().store(harness.owner, VALUE, object_id="operation"))
    harness.create_release.set()
    left, right = await asyncio.gather(first, second)
    assert left == right
    assert sum(method == "POST" for method, _ in harness.calls) == 1


@pytest.mark.asyncio
async def test_delete_waits_for_inflight_create_across_driver_instances(harness: Harness) -> None:
    harness.create_release.clear()
    store = asyncio.create_task(harness.owned.store(harness.owner, VALUE, object_id="operation"))
    await asyncio.wait_for(harness.create_started.wait(), 2)
    pending = harness.registry.pending_owned()[0]
    delete = asyncio.create_task(harness.restart().delete(harness.owner, pending.handle))
    harness.create_release.set()
    await asyncio.gather(store, delete)
    assert harness.registry.get_owned(harness.owner, pending.handle).state == "deleted"
    assert harness.values == {USER_SECRET: "synthetic-provider-token"}


@pytest.mark.asyncio
async def test_revocation_during_remote_read_wins(harness: Harness) -> None:
    handle = await harness.owned.store(harness.owner, VALUE, object_id="operation")
    harness.read_release.clear()
    resolve = asyncio.create_task(harness.owned.resolve(harness.owner, handle))
    await asyncio.wait_for(harness.read_started.wait(), 2)
    await harness.owned.delete(harness.owner, handle)
    harness.read_release.set()
    with pytest.raises(RegistrationDenied):
        await resolve


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefix", "credential_key"),
    [("sandbox-ssh-", "SSH_HOST_PRIVATE_KEY"), ("sandbox-instance-", "SHARED_INSTANCE_LAUNCH_MATERIAL")],
)
async def test_mixed_resolution_keeps_reference_expiry_and_owned_lifecycle(
    harness: Harness, prefix: str, credential_key: str
) -> None:
    owner = harness.owner.model_copy(
        update={"provider": prefix + harness.owner.provider_id, "credential_key": credential_key}
    )
    driver = CredentialDriver(harness.registry, harness.client, owned=harness.owned, auth_enabled=False)
    stored = wire.data(
        await driver._dispatch(
            "StoreCredential", {**owner.model_dump(), "object_id": "internal", "value": VALUE.get_secret_value()}
        )
    )
    internal_handle = stored["handle"]
    reference_owner = owner.model_copy(update={"provider": "provider", "credential_key": "API_TOKEN"})
    ticket = harness.tickets.sign(
        BindingScope(
            gateway_id="gateway",
            gateway_workspace=reference_owner.workspace,
            provider=reference_owner.provider,
            credential_key=reference_owner.credential_key,
            secret_workspace=USER_SECRET[0],
            secret_name=USER_SECRET[1],
            job_id="job-id",
        )
    )
    reference_handle = {
        "driver": "nemo",
        "handle": harness.registry.register(reference_owner, ticket),
    }
    request = {
        "credentials": [
            {**reference_owner.model_dump(), "request_id": "reference", "handle": reference_handle},
            {**owner.model_dump(), "request_id": "internal", "handle": internal_handle},
        ]
    }
    resolved = wire.data(await driver._dispatch("ResolveCredentials", request))["credentials"]
    assert resolved[0]["value"] == "synthetic-provider-token"
    assert resolved[0]["expiration_time"]
    assert resolved[1]["value"] == VALUE.get_secret_value()
    assert "expiration_time" not in resolved[1]
    for target, source, value in (
        (owner, reference_handle, VALUE.get_secret_value()),
        (reference_owner, internal_handle, ticket.get_secret_value()),
    ):
        with pytest.raises(RegistrationDenied):
            await driver._dispatch(
                "StoreCredential",
                {**target.model_dump(), "object_id": "overwrite", "value": value, "existing_handle": source},
            )
    await driver._dispatch("DeleteCredential", {**reference_owner.model_dump(), "handle": reference_handle})
    await driver._dispatch("DeleteCredential", {**owner.model_dump(), "handle": internal_handle})
    assert harness.values == {USER_SECRET: "synthetic-provider-token"}


@pytest.mark.asyncio
async def test_cleanup_retries_redacts_failures_and_stops_with_server(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    interval = asyncio.Event()
    attempted = asyncio.Event()
    recovered = asyncio.Event()
    attempts = 0
    delays: list[float] = []

    async def reconcile() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            attempted.set()
            raise RuntimeError("private credential material")
        recovered.set()

    async def next_interval(seconds: float) -> None:
        delays.append(seconds)
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
    assert delays and all(seconds == 30 for seconds in delays)
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
