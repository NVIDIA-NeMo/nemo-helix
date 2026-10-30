# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration and wire-contract tests, without a running Helix or gateway."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock, patch

import httpx
import jwt
import pytest
import time_machine
from nemo_deployments_plugin.backends.openshell.credential_driver.registry import BindingRegistry, ProviderOwner
from nemo_deployments_plugin.backends.openshell.credential_driver.server import read_private_file, running_server
from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver
from nemo_deployments_plugin.backends.openshell.credential_driver.sidecar import (
    CredentialInitializationError,
    check_ready,
    prepare_credentials,
)
from nemo_deployments_plugin.backends.openshell.credential_driver.sidecar import (
    main as sidecar_main,
)
from nemo_deployments_plugin.backends.openshell.credential_driver.tickets import (
    BindingScope,
    RegistrationClaims,
    RegistrationDenied,
    RegistrationTickets,
)
from nemo_helix_plugin.openshell.types import GatewayWorkload
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from pydantic import SecretStr
from sqlalchemy.exc import OperationalError

KEY = b"test-only-registration-key-32-bytes"


class Harness:
    def __init__(self, directory: Path, traveller: time_machine.Traveller) -> None:
        self.traveller = traveller
        self.tickets = RegistrationTickets(KEY, "gateway-a")
        self.registry = BindingRegistry(directory / "bindings.db", self.tickets)
        self.scope = BindingScope(
            gateway_id="gateway-a",
            gateway_workspace="openshell-team",
            provider="job-provider",
            credential_key="API_TOKEN",
            secret_workspace="helix-team",
            secret_name="service-token",
            job_id="job-1",
            principal_id="user-1",
        )
        self.owner = ProviderOwner(
            workspace="openshell-team", provider="job-provider", provider_id="provider-uuid", credential_key="API_TOKEN"
        )
        self.calls: list[httpx.Request] = []
        self.secret = "real-secret-never-persist-this"
        self.status = 200
        self.job_status = "active"
        self.step_status = "active"
        self.attempt = "attempt-1"
        self.step_id = "step-1"
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
        if request.url.path == "/apis/auth/v2/authz/has_permissions":
            assert json.loads(request.content)["input"] == {
                "principal_id": self.scope.principal_id,
                "workspace": "helix-team",
                "permissions": ["secrets.read"],
            }
            return httpx.Response(200, json={"result": {"allowed": True}})
        if "/apis/jobs/" in request.url.path:
            if request.url.path.endswith("/steps/run"):
                return httpx.Response(
                    200,
                    json={
                        "id": self.step_id,
                        "entity_id": self.step_id,
                        "parent": self.attempt,
                        "attempt_id": self.attempt,
                        "name": "run",
                        "workspace": "helix-team",
                        "status": self.step_status,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "id": "job-id",
                    "attempt_id": self.attempt,
                    "name": "job-1",
                    "workspace": "helix-team",
                    "source": "test",
                    "platform_spec": {
                        "steps": [{"name": "run", "executor": {"provider": "subprocess", "command": ["true"]}}]
                    },
                    "fileset": "results",
                    "status": self.job_status,
                },
            )
        assert "/workspaces/helix-team/secrets/service-token" in request.url.path
        payload = {"id": "secret-id", "name": "service-token", "workspace": "helix-team"}
        if request.url.path.endswith("/access"):
            assert "X-NHX-Principal-On-Behalf-Of" not in request.headers
            payload["value"] = self.secret
        return httpx.Response(200, json=payload)

    async def issue(self) -> SecretStr:
        return self.tickets.sign(self.scope)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    with time_machine.travel(1_800_000_000, tick=False) as traveller:
        yield Harness(tmp_path, traveller)


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_enabled", [True, False])
async def test_driver_rejects_mismatched_auth_mode_before_secret_access(harness: Harness, auth_enabled: bool) -> None:
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    harness.scope = harness.scope.model_copy(update={"principal_id": None if auth_enabled else "user-1"})
    ticket = await harness.issue()
    driver = CredentialDriver(harness.registry, harness.client, auth_enabled=auth_enabled)
    harness.calls.clear()
    with pytest.raises(RegistrationDenied, match="authentication mode"):
        await driver._dispatch("StoreCredential", {**harness.owner.model_dump(), "value": ticket.get_secret_value()})
    # A persisted binding from before an operator changes modes is denied too.
    handle = harness.registry.register(harness.owner, ticket)
    with pytest.raises(RegistrationDenied, match="authentication mode"):
        await driver._dispatch(
            "ResolveCredentials",
            {
                "credentials": [
                    {**harness.owner.model_dump(), "request_id": "read", "handle": {"driver": "nemo", "handle": handle}}
                ]
            },
        )
    assert not harness.calls


def test_auth_disabled_sidecar_needs_only_signing_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "registration.key").write_bytes(KEY)
    destination = tmp_path / "private"
    prepare_credentials(source, destination, auth_disabled=True)
    assert (destination / "registration.key").read_bytes() == KEY
    assert not (destination / "helix.token").exists()
    with pytest.raises(CredentialInitializationError, match="helix.token"):
        prepare_credentials(source, destination)  # Missing token never selects disabled mode implicitly.
    (source / "helix.token").write_bytes(b"staged-service-token")
    prepare_credentials(source, destination)
    (source / "helix.token").unlink()
    (source / "registration.key").write_bytes(b"different-signing-key-32-bytes-long")
    with pytest.raises(CredentialInitializationError, match="explicit migration"):
        prepare_credentials(source, destination, auth_disabled=True)
    assert (destination / "helix.token").read_bytes() == b"staged-service-token"
    (source / "registration.key").write_bytes(KEY)
    prepare_credentials(source, destination, auth_disabled=True)
    assert not (destination / "helix.token").exists()
    assert (destination / "registration.key").read_bytes() == KEY


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("missing-token", "Cannot open mounted helix.token"),
        ("short-key", "signing key of at least 32 bytes"),
        ("key-mismatch", "explicit migration is required"),
        ("invalid-utf8", "nonempty UTF-8 token"),
    ],
)
def test_sidecar_init_reports_safe_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str, expected: str
) -> None:
    source, destination = tmp_path / "source", tmp_path / "private"
    source.mkdir()
    key = source / "registration.key"
    token = source / "helix.token"
    key.write_bytes(KEY)
    token.write_bytes(b"sensitive-service-token")
    if failure == "missing-token":
        token.unlink()
    elif failure == "short-key":
        key.write_bytes(b"sensitive-short-key")
    elif failure == "key-mismatch":
        prepare_credentials(source, destination)
        key.write_bytes(b"sensitive-different-key-32-bytes-long")
    else:
        token.write_bytes(b"sensitive-service-token\xff")
    with patch("sys.argv", ["sidecar", "init", "--source", str(source), "--destination", str(destination)]):
        with pytest.raises(SystemExit) as error:
            sidecar_main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert expected in output.err
    assert "sensitive" not in output.err
    assert KEY.decode() not in output.err
    assert not output.out


@pytest.mark.parametrize("command", ["init", "healthcheck"])
def test_sidecar_unexpected_errors_do_not_disclose_credentials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    module = "nemo_deployments_plugin.backends.openshell.credential_driver.sidecar"
    if command == "init":
        argv = ["sidecar", "init", "--source", str(tmp_path), "--destination", str(tmp_path)]
        function = "prepare_credentials"
    else:
        argv = ["sidecar", "healthcheck", "--socket-path", str(tmp_path / "driver.sock")]
        function = "check_ready"
    with patch("sys.argv", argv), patch(f"{module}.{function}", side_effect=ValueError("sensitive-token")):
        with pytest.raises(SystemExit) as error:
            sidecar_main()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.err == "Credential-driver initialization or readiness check failed.\n"
    assert not output.out


@pytest.mark.asyncio
async def test_register_restart_retry_and_delete_preserve_user_secret(harness: Harness) -> None:
    ticket = await harness.issue()
    handle = harness.registry.register(harness.owner, ticket)
    assert harness.registry.register(harness.owner, ticket) == handle
    assert all(not request.url.path.endswith("/access") for request in harness.calls)
    db_contents = harness.registry.path.read_bytes()
    assert ticket.get_secret_value().encode() not in db_contents
    assert harness.secret.encode() not in db_contents
    restarted = BindingRegistry(harness.registry.path, harness.tickets)
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
    header, payload, signature = raw.split(".")
    tampered = f"{header}.{payload}.{'A' if signature[0] != 'A' else 'B'}{signature[1:]}"
    for invalid in (SecretStr("raw-api-key"), SecretStr(tampered)):
        with pytest.raises(RegistrationDenied):
            harness.registry.register(harness.owner, invalid)
    other_gateway = RegistrationTickets(KEY, "gateway-b")
    with pytest.raises(RegistrationDenied):
        other_gateway.verify(ticket)
    with pytest.raises(ValueError, match="another gateway"):
        BindingRegistry(harness.registry.path, other_gateway)
    harness.traveller.shift(299)
    harness.tickets.verify(ticket)
    harness.traveller.shift(1)
    with pytest.raises(RegistrationDenied):
        harness.registry.register(harness.owner, ticket)


@pytest.mark.asyncio
async def test_registration_jwt_interoperates_without_changing_stored_claims(harness: Harness) -> None:
    token = await harness.issue()
    decoded = jwt.decode_complete(token.get_secret_value(), KEY, algorithms=["HS256"], audience="gateway-a")
    assert decoded["header"]["typ"] == "nemo-registration+jwt"
    payload = decoded["payload"]
    assert payload["aud"] == "gateway-a"
    assert payload["iat"] == int(time.time())
    assert payload["exp"] == int(time.time()) + 300
    assert payload["jti"]
    claims = harness.tickets.verify(token)
    stored = claims.model_dump_json()
    assert {"gateway_id", "ticket_id", "issued_at", "expires_at"} <= json.loads(stored).keys()
    assert RegistrationClaims.model_validate_json(stored) == claims
    handle = harness.registry.register(harness.owner, token)
    restarted = BindingRegistry(harness.registry.path, harness.tickets)
    assert restarted.resolve(harness.owner, handle).claims == claims


@pytest.mark.parametrize("missing", ["aud", "jti", "iat", "exp"])
def test_registration_requires_standard_claims(harness: Harness, missing: str) -> None:
    claims = {
        **harness.scope.model_dump(mode="json", by_alias=True),
        "jti": "registration-id",
        "iat": int(time.time()),
        "exp": int(time.time()) + 300,
        "binding_expires_at": int(time.time()) + 3600,
    }
    del claims[missing]
    token = jwt.encode(claims, KEY, algorithm="HS256", headers={"typ": "nemo-registration+jwt"})
    with pytest.raises(RegistrationDenied):
        harness.registry.register(harness.owner, SecretStr(token))


@pytest.mark.parametrize(
    ("algorithm", "key", "token_type", "issued", "expires", "binding_expires"),
    [
        ("HS384", KEY, "nemo-registration+jwt", 0, 300, 3600),
        ("none", "", "nemo-registration+jwt", 0, 300, 3600),
        ("HS256", b"another-synthetic-key-32-bytes!!!!", "nemo-registration+jwt", 0, 300, 3600),
        ("HS256", KEY, "JWT", 0, 300, 3600),
        ("HS256", KEY, "nemo-registration+jwt", 1, 300, 3600),
        ("HS256", KEY, "nemo-registration+jwt", -300, 0, 3300),
        ("HS256", KEY, "nemo-registration+jwt", 0, 301, 3600),
        ("HS256", KEY, "nemo-registration+jwt", 0, 300, 3601),
        ("HS256", KEY, "nemo-registration+jwt", 0, 300, 299),
    ],
)
def test_registration_rejects_other_token_types_and_invalid_lifetimes(
    harness: Harness,
    algorithm: str,
    key: bytes | str,
    token_type: str,
    issued: int,
    expires: int,
    binding_expires: int,
) -> None:
    claims = {
        **harness.scope.model_dump(mode="json", by_alias=True),
        "jti": "registration-id",
        "iat": int(time.time()) + issued,
        "exp": int(time.time()) + expires,
        "binding_expires_at": int(time.time()) + binding_expires,
    }
    token = jwt.encode(claims, key, algorithm=algorithm, headers={"typ": token_type})
    with pytest.raises(RegistrationDenied):
        harness.registry.register(harness.owner, SecretStr(token))


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
    harness.traveller.shift(3600)
    with pytest.raises(RegistrationDenied):
        harness.registry.resolve(harness.owner, old)


@pytest.mark.asyncio
async def test_concurrent_registry_startup_and_registration(harness: Harness, tmp_path: Path) -> None:
    ticket = await harness.issue()
    path = tmp_path / "concurrent.db"

    def register(_: int) -> str:
        registry = BindingRegistry(path, harness.tickets)
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
    ticket = harness.tickets.sign(other_scope)
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

    driver = CredentialDriver(harness.registry, harness.client)
    with pytest.raises(RegistrationDenied):
        await driver._dispatch("GetCapabilities", {"gateway": gateway})


@pytest.mark.asyncio
async def test_invalid_batch_does_not_read_any_secrets(harness: Harness) -> None:
    pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    handle = harness.registry.register(harness.owner, await harness.issue())
    harness.calls.clear()
    lookup = {**harness.owner.model_dump(), "handle": {"driver": "nemo", "handle": handle}, "request_id": "one"}
    driver = CredentialDriver(harness.registry, harness.client)
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
async def test_sidecar_recovers_stale_socket_and_rejects_second_driver(
    harness: Harness, socket_directory: Path
) -> None:
    path = socket_directory / "driver.sock"
    # SIGKILL leaves the same filesystem artifact: a socket with no listener.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
        stale.bind(str(path))
    driver = CredentialDriver(harness.registry, harness.client, clock=harness.clock)
    async with running_server(driver, path):
        await check_ready(path)
        with pytest.raises(ValueError, match="Another credential driver"):
            async with running_server(driver, path):
                pytest.fail("A second driver acquired the socket")
        await check_ready(path)
    assert not path.exists()
    assert not harness.calls  # Readiness does not read secrets or require Helix.


@pytest.mark.parametrize("kind", ["file", "symlink", "listener"])
def test_sidecar_does_not_remove_other_socket_owners(socket_directory: Path, kind: str) -> None:
    from nemo_deployments_plugin.backends.openshell.credential_driver.server import socket_lease

    path = socket_directory / "driver.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        if kind == "file":
            path.write_text("leave me alone")
        elif kind == "symlink":
            path.symlink_to(socket_directory / "missing")
        else:
            listener.bind(str(path))
            listener.listen(1)
        inode = path.lstat().st_ino
        with pytest.raises(ValueError):
            with socket_lease(path):
                pytest.fail("Existing path was accepted")
        assert path.lstat().st_ino == inode


def test_sidecar_stages_private_files_and_rotates_only_token(tmp_path: Path) -> None:
    source, destination = tmp_path / "mount", tmp_path / "private"
    source.mkdir()
    (source / "registration.key").write_bytes(KEY)
    (source / "helix.token").write_text("token-one")
    prepare_credentials(source, destination)
    assert destination.stat().st_mode & 0o777 == 0o700
    assert read_private_file(destination / "registration.key") == KEY
    assert read_private_file(destination / "helix.token") == b"token-one"
    inode = (destination / "helix.token").stat().st_ino
    (source / "helix.token").write_text("token-two")
    prepare_credentials(source, destination)
    assert read_private_file(destination / "helix.token") == b"token-two"
    assert (destination / "helix.token").stat().st_ino != inode
    (source / "registration.key").write_bytes(b"replacement-signing-key-32-bytes!")
    (source / "helix.token").write_text("token-three")
    with pytest.raises(ValueError, match="explicit migration"):
        prepare_credentials(source, destination)
    assert read_private_file(destination / "registration.key") == KEY
    assert read_private_file(destination / "helix.token") == b"token-two"


def test_sidecar_syncs_rotated_and_removed_token_before_success(tmp_path: Path) -> None:
    source, destination = tmp_path / "mount", tmp_path / "private"
    source.mkdir()
    (source / "registration.key").write_bytes(KEY)
    token = source / "helix.token"
    token.write_bytes(b"old-token")
    prepare_credentials(source, destination)
    token.write_bytes(b"new-token")
    original_fsync = os.fsync
    synced_tokens: list[bytes | None] = []

    def sync_and_observe(descriptor: int) -> None:
        original_fsync(descriptor)
        info = os.fstat(descriptor)
        if stat.S_ISDIR(info.st_mode):
            assert info.st_ino == destination.stat().st_ino
            staged = destination / "helix.token"
            synced_tokens.append(staged.read_bytes() if staged.exists() else None)

    module = "nemo_deployments_plugin.backends.openshell.credential_driver.sidecar"
    with patch(f"{module}.os.fsync", side_effect=sync_and_observe):
        prepare_credentials(source, destination)
        assert synced_tokens[-1] == b"new-token"
        prepare_credentials(source, destination, auth_disabled=True)
        assert synced_tokens[-1] is None

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("Directory sync failed")
        original_fsync(descriptor)

    with patch(f"{module}.os.fsync", side_effect=fail_directory_sync):
        with pytest.raises(OSError, match="Directory sync failed"):
            prepare_credentials(source, destination)


@pytest.mark.parametrize("invalid", ["symlink", "empty", "oversize", "short-key"])
def test_sidecar_rejects_invalid_mounted_credentials(tmp_path: Path, invalid: str) -> None:
    source, destination = tmp_path / "mount", tmp_path / "private"
    source.mkdir()
    (source / "registration.key").write_bytes(KEY if invalid != "short-key" else b"short")
    token = source / "helix.token"
    if invalid == "symlink":
        token.symlink_to(source / "registration.key")
    else:
        token.write_bytes(b"x" * (65537 if invalid == "oversize" else 0) if invalid != "short-key" else b"token")
    with pytest.raises((OSError, ValueError)):
        prepare_credentials(source, destination)
    assert not destination.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_enabled", [True, False])
async def test_real_grpc_registration_resolution_and_delete(
    harness: Harness, socket_directory: Path, auth_enabled: bool
) -> None:
    grpc = pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver import wire
    from nemo_deployments_plugin.backends.openshell.credential_driver.server import running_server
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CAPABILITY, CredentialDriver

    if not auth_enabled:
        harness.scope = harness.scope.model_copy(update={"principal_id": None})
    driver = CredentialDriver(harness.registry, harness.client, auth_enabled=auth_enabled)
    socket = socket_directory / "driver.sock"
    async with running_server(driver, socket):
        async with grpc.aio.insecure_channel(f"unix:{socket}") as channel:

            async def call(method: str, data: dict) -> dict:
                from openshell._proto import credential_driver_pb2_grpc

                request_name, _ = wire.METHODS[method]
                rpc = getattr(credential_driver_pb2_grpc.CredentialDriverStub(channel), method)
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
            entered = Event()
            register = harness.registry.register

            def blocked_registration(*args, **kwargs):
                entered.set()
                return register(*args, **kwargs)

            # A writer waiting for SQLite must not prevent other RPCs from running.
            with patch.object(harness.registry, "register", side_effect=blocked_registration):
                try:
                    with harness.registry._session(write=True):
                        storing = asyncio.create_task(
                            call("StoreCredential", {**owner, "value": ticket.get_secret_value()})
                        )
                        assert await asyncio.to_thread(entered.wait, 2)
                        await call(
                            "GetCapabilities",
                            {"gateway": {"protocol_version": {"major": 1}, "supported_capabilities": [CAPABILITY]}},
                        )
                        assert not storing.done()
                finally:
                    stored = await storing
            handle = stored["handle"]
            lookup = {**owner, "handle": handle, "request_id": "read-1"}
            result = await call("ResolveCredentials", {"credentials": [lookup]})
            assert result["credentials"][0]["value"] == harness.secret
            assert result["credentials"][0]["request_id"] == "read-1"
            assert (
                datetime.fromisoformat(result["credentials"][0]["expiration_time"]).timestamp()
                == int(time.time()) + 300
            )
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
@pytest.mark.parametrize("change", ["revoke", "slow-read"])
async def test_revoke_or_expiry_during_secret_read_wins(harness: Harness, change: str) -> None:
    pytest.importorskip("grpc")
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    handle = harness.registry.register(harness.owner, await harness.issue())
    original = harness.client.access_secret

    async def access(**kwargs):
        response = await original(**kwargs)
        if change == "revoke":
            harness.registry.revoke_job("job-1")
        else:
            harness.traveller.shift(31)
        return response

    harness.client.access_secret = AsyncMock(side_effect=access)
    driver = CredentialDriver(harness.registry, harness.client, cache_lifetime=30)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("lifetime", [1, 3600])
async def test_configured_credential_lifetime_is_capped_by_binding(harness: Harness, lifetime: int) -> None:
    from nemo_deployments_plugin.backends.openshell.credential_driver import wire
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    handle = harness.registry.register(harness.owner, await harness.issue())
    claims = harness.registry.resolve(harness.owner, handle).claims
    harness.traveller.move_to(claims.binding_expires_at - 100)
    driver = CredentialDriver(harness.registry, harness.client, cache_lifetime=lifetime)
    response = await driver._dispatch(
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
    expires = datetime.fromisoformat(wire.data(response)["credentials"][0]["expiration_time"]).timestamp()
    assert expires == min(int(time.time()) + lifetime, claims.binding_expires_at)


@pytest.mark.parametrize("lifetime", [0, -1, 3601])
def test_invalid_credential_lifetime(harness: Harness, lifetime: int) -> None:
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    with pytest.raises(ValueError, match="cache lifetime"):
        CredentialDriver(harness.registry, harness.client, cache_lifetime=lifetime)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["completed", "cancelling", "paused", "step-error", "new-attempt", "new-step", "active"]
)
async def test_job_binding_tracks_live_execution_identity(harness: Harness, change: str) -> None:
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    harness.scope = harness.scope.model_copy(
        update={
            "workload": GatewayWorkload(
                kind="job",
                workspace="helix-team",
                name="job-1",
                id="step-1",
                attempt_id="attempt-1",
                step_name="run",
            )
        }
    )
    handle = harness.registry.register(harness.owner, await harness.issue())
    harness.calls.clear()
    harness.traveller.shift(7200)  # A live workload is not limited to a one-hour registration lease.
    if change in {"completed", "cancelling", "paused"}:
        harness.job_status = change
    elif change == "step-error":
        harness.step_status = "error"
    elif change == "new-attempt":
        harness.attempt = "attempt-2"
    elif change == "new-step":
        harness.step_id = "step-2"
    driver = CredentialDriver(harness.registry, harness.client)
    request = {
        "credentials": [
            {**harness.owner.model_dump(), "handle": {"driver": "nemo", "handle": handle}, "request_id": "read"}
        ]
    }
    if change == "active":
        from nemo_deployments_plugin.backends.openshell.credential_driver import wire

        response = wire.data(await driver._dispatch("ResolveCredentials", request))
        assert response["credentials"][0]["value"] == harness.secret
        assert len([r for r in harness.calls if "/apis/jobs/" in r.url.path]) == 4
    else:
        with pytest.raises(RegistrationDenied):
            await driver._dispatch("ResolveCredentials", request)
        assert not any(r.url.path.endswith("/access") for r in harness.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["deleted", "recreated", "stopping", "failed", "active"])
async def test_deployment_binding_tracks_live_entity(harness: Harness, change: str) -> None:
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver
    from nemo_deployments_plugin.entities import Deployment
    from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

    driver = CredentialDriver(harness.registry, harness.client)
    workload = GatewayWorkload(kind="deployment", workspace="helix-team", name="agent", id="dep-1")
    entity = Deployment(
        name="agent",
        workspace="helix-team",
        deployment_config="config",
        status="FAILED" if change == "failed" else "READY",
        desired_state="STOPPED" if change == "stopping" else "READY",
    )
    entity._id = "dep-2" if change == "recreated" else "dep-1"
    with patch("nemo_deployments_plugin.backends.openshell.credential_driver.service.NemoEntitiesClient") as client:
        client.return_value.get = AsyncMock(return_value=entity)
        if change == "deleted":
            client.return_value.get.side_effect = NemoEntityNotFoundError("gone")
        if change == "active":
            await driver._require_active_workload(workload)
        else:
            with pytest.raises(RegistrationDenied):
                await driver._require_active_workload(workload)


@pytest.mark.asyncio
async def test_standalone_expiry_after_final_registry_check_denies_response(harness: Harness) -> None:
    from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver

    handle = harness.registry.register(harness.owner, await harness.issue())
    claims = harness.registry.resolve(harness.owner, handle).claims
    harness.traveller.move_to(claims.binding_expires_at - 1)
    driver = CredentialDriver(harness.registry, harness.client)
    checks = 0

    async def advance(workload: GatewayWorkload | None) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            harness.traveller.shift(2)

    with patch.object(driver, "_require_active_workload", side_effect=advance):
        with pytest.raises(RegistrationDenied, match="cache lifetime"):
            await driver._dispatch(
                "ResolveCredentials",
                {
                    "credentials": [
                        {
                            **harness.owner.model_dump(),
                            "request_id": "read",
                            "handle": {"driver": "nemo", "handle": handle},
                        }
                    ]
                },
            )
