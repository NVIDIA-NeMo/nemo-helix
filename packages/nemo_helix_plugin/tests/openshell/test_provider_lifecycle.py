# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise controller registration against the real driver and wire messages."""

# ruff: noqa: E402 -- Optional OpenShell dependencies must be checked before driver imports.

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google.protobuf.message import Message

grpc = pytest.importorskip("grpc")
pytest.importorskip("openshell")

from nemo_deployments_plugin.backends.openshell import authorization as deployment_authorization
from nemo_deployments_plugin.backends.openshell import provisioning as deployment_provisioning
from nemo_deployments_plugin.backends.openshell.credential_driver import wire
from nemo_deployments_plugin.backends.openshell.credential_driver.registry import BindingRegistry, ProviderOwner
from nemo_deployments_plugin.backends.openshell.credential_driver.service import CredentialDriver
from nemo_deployments_plugin.backends.openshell.policy import (
    SandboxFilesystem,
    build_sandbox_policy,
    generate_policy_dict,
)
from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.client.errors import NotFoundError, PermissionDeniedError
from nemo_helix_plugin.openshell.isolation import restricted_platform_policy, validate_provider_environment
from nemo_helix_plugin.openshell.provider_client import ProviderGatewayClient
from nemo_helix_plugin.openshell.registration import RegistrationDenied, RegistrationTickets
from nemo_helix_plugin.openshell.types import (
    GatewayCredentialConfig,
    GatewayProviderBinding,
    GatewayWorkload,
    ProviderAccessRevoked,
)
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.core.jobs.controllers.backends.openshell import authorization as job_authorization
from nhx.core.jobs.controllers.backends.openshell import provisioning as job_provisioning
from openshell._proto import datamodel_pb2 as dm
from openshell._proto import openshell_pb2 as pb


class Missing(grpc.RpcError):
    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.NOT_FOUND


class Conflict(grpc.RpcError):
    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.ABORTED


class Gateway:
    def __init__(self, driver: CredentialDriver) -> None:
        self.driver = driver
        self.providers: dict[str, dm.Provider] = {}
        self.attached: set[str] = set()
        self.phase = pb.SANDBOX_PHASE_READY
        self.detached = False
        self.detached_providers: set[str] = set()
        self.sandbox_version = 1
        self.updates: list[pb.UpdateProviderRequest] = []
        self.profile = pb.ProviderProfile(
            endpoints=[{"host": "service.example"}],
            credentials=[{"env_vars": ["API_TOKEN"]}],
        )

    def channel(
        self, call: Callable[[str, Message], Coroutine[object, object, Message]] | None = None
    ) -> grpc.aio.Channel:
        """In-process async channel for the SDK stub and simulated gateway."""
        dispatch = call or self.call

        def unary_unary(
            path: str,
            *,
            request_serializer: Callable[[Message], bytes],
            response_deserializer: Callable[[bytes], Message],
            _registered_method: bool,
        ) -> Callable[..., Coroutine[object, object, Message]]:
            method = path.rsplit("/", 1)[-1]

            async def invoke(request: Message, *, timeout: float) -> Message:
                assert timeout == 30.0
                decoded_request = type(request).FromString(request_serializer(request))
                response = await dispatch(method, decoded_request)
                return response_deserializer(response.SerializeToString())

            return invoke

        channel = MagicMock(spec=grpc.aio.Channel)
        channel.unary_unary.side_effect = unary_unary
        return channel

    async def call(self, method: str, request: Message) -> Message:
        if method == "GetProviderProfile":
            return pb.ProviderProfileResponse(profile=self.profile)
        if method == "GetProvider":
            assert isinstance(request, pb.GetProviderRequest)
            if request.name not in self.providers:
                raise Missing()
            # The gateway redacts values and hides handles, but preserves the
            # resource version that an update must carry back.
            provider = dm.Provider()
            provider.CopyFrom(self.providers[request.name])
            for key in provider.credential_handles:
                provider.credentials[key] = "********"
            provider.credential_handles.clear()
            return pb.ProviderResponse(provider=provider)
        if method in ("CreateProvider", "UpdateProvider"):
            assert isinstance(request, (pb.CreateProviderRequest, pb.UpdateProviderRequest))
            provider = dm.Provider()
            provider.CopyFrom(request.provider)
            assert all(len(value) <= 63 for value in provider.metadata.labels.values())
            previous = self.providers.get(provider.metadata.name)
            if method == "UpdateProvider":
                assert isinstance(request, pb.UpdateProviderRequest)
                assert previous is not None
                assert provider.metadata.id == previous.metadata.id
                if provider.metadata.resource_version != previous.metadata.resource_version:
                    raise Conflict()
                assert not provider.credential_handles
                # Renewal must not depend on metadata updates being applied.
                provider.metadata.CopyFrom(previous.metadata)
            else:
                provider.metadata.id = "provider-" + provider.metadata.name
            provider.metadata.resource_version += 1
            updated_keys = set(provider.credentials)
            if previous:
                for key, handle in previous.credential_handles.items():
                    provider.credential_handles[key].CopyFrom(handle)
            for key, ticket in provider.credentials.items():
                response = await self.driver._dispatch(
                    "StoreCredential",
                    {
                        "workspace": provider.metadata.workspace,
                        "provider": provider.metadata.name,
                        "provider_id": provider.metadata.id,
                        "credential_key": key,
                        "value": ticket,
                        **(
                            {"existing_handle": {"handle": previous.credential_handles[key].handle}} if previous else {}
                        ),
                    },
                )
                provider.credential_handles[key].CopyFrom(dm.CredentialHandle(**wire.data(response)["handle"]))
            provider.credentials.clear()
            if isinstance(request, pb.UpdateProviderRequest):
                # Credential storage yields; compare against the committed winner.
                if (
                    request.provider.metadata.resource_version
                    != self.providers[provider.metadata.name].metadata.resource_version
                ):
                    raise Conflict()
                self.updates.append(request)
            self.providers[provider.metadata.name] = provider
            if previous:
                for key in updated_keys:
                    handle = previous.credential_handles[key]
                    self.driver.registry.delete(
                        ProviderOwner(
                            workspace=provider.metadata.workspace,
                            provider=provider.metadata.name,
                            provider_id=provider.metadata.id,
                            credential_key=key,
                        ),
                        handle.handle,
                    )
            return pb.ProviderResponse(provider=provider)
        if method == "ListProviders":
            return pb.ListProvidersResponse(providers=list(self.providers.values()), next_page_token="")
        if method == "GetSandbox":
            assert isinstance(request, pb.GetSandboxRequest)
            if request.name in self.attached:
                return pb.SandboxResponse(
                    sandbox=pb.Sandbox(
                        status=pb.SandboxStatus(phase=self.phase),
                        metadata=dm.ObjectMeta(resource_version=self.sandbox_version),
                        spec=pb.SandboxSpec(
                            providers=[] if self.detached else list(set(self.providers) - self.detached_providers)
                        ),
                    )
                )
            raise Missing()
        if method == "DetachSandboxProvider":
            assert isinstance(request, pb.DetachSandboxProviderRequest)
            assert request.sandbox in self.attached
            if request.expected_resource_version != self.sandbox_version:
                raise Conflict()
            self.detached_providers.add(request.provider)
            self.sandbox_version += 1
            return pb.DetachSandboxProviderResponse()
        if method == "DeleteProvider":
            assert isinstance(request, pb.DeleteProviderRequest)
            provider = self.providers.pop(request.name, None)
            if provider is None:
                assert request.allow_missing
                return pb.DeleteProviderResponse()
            for key, handle in provider.credential_handles.items():
                self.driver.registry.delete(
                    ProviderOwner(
                        workspace=provider.metadata.workspace,
                        provider=provider.metadata.name,
                        provider_id=provider.metadata.id,
                        credential_key=key,
                    ),
                    handle.handle,
                )
            return pb.DeleteProviderResponse()
        raise AssertionError(method)


@pytest.fixture(params=["job", "deployment"])
async def setup(tmp_path: Path, monkeypatch, request):
    kind = request.param
    authorization = job_authorization if kind == "job" else deployment_authorization
    provisioning = job_provisioning if kind == "job" else deployment_provisioning
    provisioner_type = (
        job_provisioning.JobProviderProvisioner
        if kind == "job"
        else deployment_provisioning.DeploymentProviderProvisioner
    )
    monkeypatch.setattr(authorization, "platform_auth_enabled", lambda: True)
    requests = []

    def http(request):
        requests.append(request)
        assert not request.url.path.endswith("/access"), "Provisioning must never fetch a secret value"
        if request.url.path.endswith("/has_permissions"):
            return httpx.Response(200, json={"result": {"allowed": True}})
        assert request.url.path.endswith("/secrets/token")
        return httpx.Response(200, json={"id": "token-id", "name": "token", "workspace": "team"})

    async with AsyncSecretsClient(
        base_url="https://helix.example",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(http)),
        owns_http_client=True,
    ) as client:
        tickets = RegistrationTickets(b"k" * 32, "gateway")
        registry = BindingRegistry(tmp_path / "bindings.db", tickets)
        gateway = Gateway(CredentialDriver(registry, client))
        config = GatewayCredentialConfig(
            gateway_id="gateway",
            registration_key_file=tmp_path / "key",
            credential_refresh_verified=True,
            profiles={"service-read": {"workspaces": {"team"}, "credential_keys": {"API_TOKEN"}}},
        )
        provisioner = provisioner_type(config, "gateway-team", gateway.channel(), tickets=tickets)
        kwargs = dict(
            workload_workspace="team",
            workload_id="sandbox-1",
            workload=GatewayWorkload(
                kind=kind,
                workspace="team",
                name="agent",
                id="workload-1",
                attempt_id="attempt-1" if kind == "job" else None,
                step_name="run" if kind == "job" else None,
            ),
            auth_context=AuthContext(principal_id="user-1"),
            secrets_client=client,
            platform_host="helix.example",
        )
        yield SimpleNamespace(
            provisioner=provisioner,
            gateway=gateway,
            registry=registry,
            requests=requests,
            kwargs=kwargs,
            error_type=provisioning.ProviderProvisioningError,
        )


def binding() -> dict[str, GatewayProviderBinding]:
    return {"service": GatewayProviderBinding(profile="service-read", credentials={"API_TOKEN": "team/token"})}


async def test_auth_disabled_provisioning_renewal_and_cleanup(setup):
    setup.gateway.driver.auth_enabled = False
    setup.provisioner.config.renewal_mode = "helix"
    setup.kwargs.update(auth_context=None, auth_enabled=False)
    names = await setup.provisioner.prepare(binding(), **setup.kwargs)
    provider = setup.gateway.providers[names[0]]
    owner = ProviderOwner(
        workspace="gateway-team", provider=names[0], provider_id=provider.metadata.id, credential_key="API_TOKEN"
    )
    handle = provider.credential_handles["API_TOKEN"].handle
    assert setup.registry.resolve(owner, handle).claims.principal_id is None
    setup.gateway.attached.add("sandbox-1")
    setup.provisioner.clock = lambda: 4000000000
    assert await setup.provisioner.prepare(binding(), **setup.kwargs, renew=True) == names
    renewed = setup.gateway.providers[names[0]].credential_handles["API_TOKEN"].handle
    assert renewed != handle
    assert setup.registry.resolve(owner, renewed).claims.principal_id is None
    assert all("X-NHX-Principal-On-Behalf-Of" not in request.headers for request in setup.requests)
    await setup.provisioner.cleanup(workload_workspace="team", workload_id="sandbox-1")
    assert not setup.gateway.providers


async def test_prepare_retry_cleanup_stores_only_reference_metadata(setup):
    names = await setup.provisioner.prepare(binding(), **setup.kwargs)
    assert await setup.provisioner.prepare(binding(), **setup.kwargs) == names
    assert len(setup.requests) == 2
    provider = setup.gateway.providers[names[0]]
    assert not provider.credentials
    owner = ProviderOwner(
        workspace="gateway-team", provider=names[0], provider_id=provider.metadata.id, credential_key="API_TOKEN"
    )
    handle = provider.credential_handles["API_TOKEN"].handle
    grant = setup.registry.resolve(owner, handle)
    assert grant.claims.principal_id == "user-1"
    assert grant.claims.workload == setup.kwargs["workload"]
    assert grant.claims.secret_name == "token"
    await setup.provisioner.cleanup(workload_workspace="other", workload_id="sandbox-1")
    assert names[0] in setup.gateway.providers
    await setup.provisioner.cleanup(workload_workspace="team", workload_id="sandbox-1")
    assert not setup.gateway.providers
    with pytest.raises(RegistrationDenied):
        setup.registry.resolve(owner, handle)


@pytest.mark.parametrize("change", ["unverified", "workspace", "slot", "platform", "principal", "profile"])
async def test_reject_unsupported_bindings_before_registration(setup, change):
    bindings = binding()
    if change == "unverified":
        setup.provisioner.config.credential_refresh_verified = False
    elif change == "workspace":
        setup.kwargs["workload_workspace"] = "other"
    elif change == "slot":
        bindings["service"] = GatewayProviderBinding(profile="service-read", credentials={"OTHER": "team/token"})
    elif change == "platform":
        setup.gateway.profile.endpoints[0].host = "helix.example"
    elif change == "principal":
        setup.kwargs["auth_context"] = None
    else:
        bindings["service"] = GatewayProviderBinding(profile="unknown", credentials={"API_TOKEN": "team/token"})
    with pytest.raises(setup.error_type):
        await setup.provisioner.prepare(bindings, **setup.kwargs)
    assert not setup.gateway.providers
    assert not setup.requests


async def test_recreated_workload_gets_distinct_provider_and_cannot_borrow_existing(setup):
    original = await setup.provisioner.prepare(binding(), **setup.kwargs)
    setup.kwargs["workload"] = setup.kwargs["workload"].model_copy(update={"id": "deployment-2"})
    recreated = await setup.provisioner.prepare(binding(), **setup.kwargs)
    assert original != recreated
    setup.gateway.providers[recreated[0]].metadata.labels["nemo-helix/provider-binding"] = "different-owner"
    with pytest.raises(setup.error_type, match="does not belong"):
        await setup.provisioner.prepare(binding(), **setup.kwargs)


async def test_orphan_sweep_preserves_attached_and_recent_providers(setup):
    now = [1000.0]
    setup.provisioner.clock = lambda: now[0]
    names = await setup.provisioner.prepare(binding(), **setup.kwargs)
    now[0] += 599
    await setup.provisioner.cleanup_unused()
    assert names[0] in setup.gateway.providers
    now[0] += 1
    setup.gateway.attached.add("sandbox-1")
    await setup.provisioner.cleanup_unused()
    assert names[0] in setup.gateway.providers
    setup.gateway.attached.clear()
    await setup.provisioner.cleanup_unused()
    assert not setup.gateway.providers


async def test_renewal_survives_controller_restart_without_fetching_values(setup):
    now = [1000.0]
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: now[0]
    names = await provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    original = setup.gateway.providers[names[0]]
    for advance, updates in [(9, 0), (1, 1), (2, 1), (7, 1), (1, 2), (10, 3)]:
        now[0] += advance
        # A fresh controller reads durable timing and ownership from the gateway.
        restarted = type(provisioner)(
            provisioner.config,
            provisioner.workspace,
            setup.gateway.channel(),
            tickets=provisioner.tickets,
            clock=lambda: now[0],
        )
        assert await restarted.prepare(binding(), **setup.kwargs, renew=True) == names
        assert len(setup.gateway.updates) == updates
    current = setup.gateway.providers[names[0]]
    assert current.metadata.id == original.metadata.id
    assert current.credential_handles["API_TOKEN"] != original.credential_handles["API_TOKEN"]
    assert len(setup.requests) == 8  # Creator permission and metadata, never /access.
    owner = ProviderOwner(
        workspace=current.metadata.workspace,
        provider=names[0],
        provider_id=current.metadata.id,
        credential_key="API_TOKEN",
    )
    claims = setup.registry.resolve(owner, current.credential_handles["API_TOKEN"].handle).claims
    assert claims.workload == setup.kwargs["workload"]
    assert claims.principal_id == "user-1"
    with pytest.raises(RegistrationDenied):
        setup.registry.resolve(owner, original.credential_handles["API_TOKEN"].handle)


async def test_concurrent_renewal_uses_gateway_version_and_retry_observes_winner(setup):
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: 1000
    await provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    barrier = asyncio.Barrier(2)

    async def concurrent_call(method: str, request: Message) -> Message:
        response = await setup.gateway.call(method, request)
        if method == "GetProvider":
            assert isinstance(request, pb.GetProviderRequest)
            await barrier.wait()
        return response

    provisioner.clock = lambda: 1010
    provisioner.client = ProviderGatewayClient(setup.gateway.channel(concurrent_call))
    results = await asyncio.gather(
        provisioner.prepare(binding(), **setup.kwargs, renew=True),
        provisioner.prepare(binding(), **setup.kwargs, renew=True),
        return_exceptions=True,
    )
    assert sum(isinstance(result, Conflict) for result in results) == 1
    assert len(setup.gateway.updates) == 1
    provisioner.client = ProviderGatewayClient(setup.gateway.channel())
    await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    assert len(setup.gateway.updates) == 1


async def test_renewal_prefers_config_timestamp_and_preserves_unrelated_config(setup):
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: 1000
    names = await provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    provider = setup.gateway.providers[names[0]]
    marker = "nemo-helix/renewed"
    provider.metadata.labels[marker] = provider.config.pop(marker)
    provider.config["service-option"] = "preserved"
    provisioner.clock = lambda: 1009
    await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    assert not setup.gateway.updates
    provisioner.clock = lambda: 1010
    await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    updated = setup.gateway.providers[names[0]]
    assert updated.config[marker] == "1010"
    assert updated.config["service-option"] == "preserved"
    assert updated.metadata.labels[marker] == "1000"  # Gateway leaves labels untouched.
    restarted = type(provisioner)(
        provisioner.config,
        provisioner.workspace,
        setup.gateway.channel(),
        tickets=provisioner.tickets,
        clock=lambda: 1012,
    )
    await restarted.prepare(binding(), **setup.kwargs, renew=True)
    assert len(setup.gateway.updates) == 1


@pytest.mark.parametrize("condition", ["detached", "missing", "owner", "profile", "update"])
async def test_failed_renewal_never_recreates_or_advances_committed_state(setup, condition):
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: 1000
    names = await provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    provider = setup.gateway.providers[names[0]]
    original_handle = provider.credential_handles["API_TOKEN"].handle
    provisioner.clock = lambda: 1010
    if condition == "detached":
        setup.gateway.detached = True
    elif condition == "missing":
        del setup.gateway.providers[names[0]]
    elif condition == "owner":
        provider.metadata.labels["nemo-helix/provider-owner"] = "other"
    elif condition == "profile":
        provisioner.config.profiles.clear()
    elif condition == "update":

        async def fail_update(method: str, request: Message) -> Message:
            if method == "UpdateProvider":
                assert isinstance(request, pb.UpdateProviderRequest)
                raise grpc.RpcError("transient gateway failure")
            return await setup.gateway.call(method, request)

        provisioner.client = ProviderGatewayClient(setup.gateway.channel(fail_update))
    with pytest.raises((setup.error_type, ProviderAccessRevoked, grpc.RpcError)):
        await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    assert provider.config["nemo-helix/renewed"] == "1000"
    assert provider.credential_handles["API_TOKEN"].handle == original_handle
    assert not setup.gateway.updates
    if condition == "missing":
        assert not setup.gateway.providers
    if condition == "update":
        provisioner.client = ProviderGatewayClient(setup.gateway.channel())
        await provisioner.prepare(binding(), **setup.kwargs, renew=True)
        assert len(setup.gateway.updates) == 1


@pytest.mark.parametrize("denial", [PermissionError, PermissionDeniedError, NotFoundError])
@pytest.mark.parametrize("auth_enabled", [True, False])
async def test_denied_provider_is_detached_deleted_and_never_renewed(setup, denial, auth_enabled):
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: 1000
    setup.kwargs.update(auth_enabled=auth_enabled)
    setup.gateway.driver.auth_enabled = auth_enabled
    names = await provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    provisioner.clock = lambda: 1010
    error = (
        PermissionError()
        if denial is PermissionError
        else denial(httpx.Response(403 if denial is PermissionDeniedError else 404))
    )
    original_issue = provisioner._issue_ticket
    provisioner._issue_ticket = AsyncMock(side_effect=error)
    with pytest.raises(ProviderAccessRevoked):
        await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    assert set(names) == setup.gateway.detached_providers
    assert not setup.gateway.providers
    assert not setup.gateway.updates[0].provider.credentials  # Intent update never sends redactions or values.
    provisioner._issue_ticket = original_issue
    with pytest.raises(ProviderAccessRevoked):
        await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    assert not setup.gateway.providers


@pytest.mark.parametrize("failure", ["UpdateProvider", "DetachSandboxProvider", "DeleteProvider"])
async def test_revocation_retries_after_restart(setup, failure):
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: 1000
    names = await provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    provisioner.clock = lambda: 1010
    provisioner._issue_ticket = AsyncMock(side_effect=PermissionError)

    async def failing_call(method: str, request: Message) -> Message:
        if method == failure:
            raise Conflict()
        return await setup.gateway.call(method, request)

    provisioner.client = ProviderGatewayClient(setup.gateway.channel(failing_call))
    with pytest.raises(Conflict):
        await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    assert names[0] in setup.gateway.providers
    # After intent was committed, restoring permission cannot cancel cleanup.
    if failure != "UpdateProvider":
        provisioner._issue_ticket = AsyncMock(side_effect=AssertionError("must not authorize a revoked provider"))
    restarted = type(provisioner)(
        provisioner.config,
        provisioner.workspace,
        setup.gateway.channel(),
        tickets=provisioner.tickets,
        clock=lambda: 1011,
    )
    restarted._issue_ticket = provisioner._issue_ticket
    with pytest.raises(ProviderAccessRevoked):
        await restarted.prepare(binding(), **setup.kwargs, renew=True)
    assert not setup.gateway.providers
    assert set(names) == setup.gateway.detached_providers


async def test_secret_outage_does_not_remove_provider_and_recovers(setup):
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: 1000
    names = await provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    provisioner.clock = lambda: 1010
    original_issue = provisioner._issue_ticket
    provisioner._issue_ticket = AsyncMock(side_effect=TimeoutError)
    with pytest.raises(TimeoutError):
        await provisioner.prepare(binding(), **setup.kwargs, renew=True)
    assert names[0] in setup.gateway.providers
    assert not setup.gateway.detached_providers
    assert not setup.gateway.updates
    provisioner._issue_ticket = original_issue
    assert await provisioner.prepare(binding(), **setup.kwargs, renew=True) == names


async def test_denied_provider_does_not_prevent_other_provider_renewal(setup):
    provisioner = setup.provisioner
    provisioner.config.renewal_mode = "helix"
    provisioner.clock = lambda: 1000
    provisioner.config.profiles["second"] = provisioner.config.profiles["service-read"].model_copy(
        update={"credential_keys": {"OTHER_TOKEN"}}
    )
    setup.gateway.profile.credentials[0].env_vars.append("OTHER_TOKEN")
    bindings = {
        **binding(),
        "second": GatewayProviderBinding(profile="second", credentials={"OTHER_TOKEN": "team/token"}),
    }
    names = await provisioner.prepare(bindings, **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    original_issue = provisioner._issue_ticket

    async def issue(scope, **kwargs):
        if scope.credential_key == "OTHER_TOKEN":
            raise PermissionError
        return await original_issue(scope, **kwargs)

    provisioner._issue_ticket = issue
    provisioner.clock = lambda: 1010
    with pytest.raises(ProviderAccessRevoked):
        await provisioner.prepare(bindings, **setup.kwargs, renew=True)
    # The denied binding sorts first. Processing must continue to the other one.
    assert names[0] in setup.gateway.detached_providers
    assert names[1] in setup.gateway.providers
    assert setup.gateway.providers[names[1]].config["nemo-helix/renewed"] == "1010"


async def test_orphan_sweep_finishes_pending_revocation_without_waiting_for_grace(setup):
    names = await setup.provisioner.prepare(binding(), **setup.kwargs)
    setup.gateway.attached.add("sandbox-1")
    setup.gateway.providers[names[0]].config["nemo-helix/revoked"] = "1"
    await setup.provisioner.cleanup_unused()
    assert not setup.gateway.providers
    assert set(names) == setup.gateway.detached_providers


@pytest.mark.parametrize(
    "mode,phase",
    [("gateway", pb.SANDBOX_PHASE_READY), ("helix", pb.SANDBOX_PHASE_COMPLETED), ("helix", pb.SANDBOX_PHASE_STOPPING)],
)
async def test_renewal_skips_disabled_or_inactive_sandboxes(setup, mode, phase):
    setup.provisioner.config.renewal_mode = mode
    setup.gateway.phase = phase
    setup.gateway.attached.add("sandbox-1")
    assert await setup.provisioner.prepare(binding(), **setup.kwargs, renew=True) == []
    assert not setup.gateway.providers
    assert not setup.requests


def test_policy_allows_workload_operations_without_broad_platform_access():
    policy = build_sandbox_policy(generate_policy_dict(filesystem=SandboxFilesystem(), egress=None))
    restricted = restricted_platform_policy(
        policy,
        host="helix.example",
        port=443,
        workspace="team",
        binaries=["/usr/bin/python"],
        job="job-1",
        step="run",
        fileset="results",
    )
    assert set(restricted.network_policies) == {"nemo_workload"}
    endpoint = restricted.network_policies["nemo_workload"].endpoints[0]
    allowed = {(rule.allow.method, rule.allow.path) for rule in endpoint.rules}
    assert ("GET", "/apis/jobs/v2/workspaces/team/jobs/job-1/steps/run") in allowed
    assert ("POST", "/apis/inference-gateway/v2/workspaces/team/openai/-/v1/chat/completions") in allowed
    assert all("/secrets/" not in path and "/entities/" not in path and "/auth/" not in path for _, path in allowed)
    assert ("POST", "/apis/jobs/v2/workspaces/team/jobs") not in allowed
    assert restricted.landlock.compatibility == "hard_requirement"


@pytest.mark.parametrize("secret_env,env_names", [({"ALIAS": "token"}, set()), ({}, {"API_TOKEN"})])
def test_reject_provider_secret_env_aliases(secret_env, env_names):
    with pytest.raises(ValueError):
        validate_provider_environment(binding(), workspace="team", secret_env=secret_env, env_names=env_names)


@pytest.mark.parametrize("winner", ["active", "other-profile-workspace", "revoked"])
async def test_concurrent_creation_checks_winner_before_attachment(setup, winner: str) -> None:
    class AlreadyExists(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.ALREADY_EXISTS

    async def concurrent_call(method: str, request: Message) -> Message:
        response = await setup.gateway.call(method, request)
        if method == "CreateProvider":
            assert isinstance(request, pb.CreateProviderRequest)
            provider = setup.gateway.providers[request.provider.metadata.name]
            if winner == "other-profile-workspace":
                provider.profile_workspace = "another-team"
            elif winner == "revoked":
                provider.config["nemo-helix/revoked"] = "1"
            raise AlreadyExists()
        return response

    setup.provisioner.client = ProviderGatewayClient(setup.gateway.channel(concurrent_call))
    if winner == "active":
        assert await setup.provisioner.prepare(binding(), **setup.kwargs) == list(setup.gateway.providers)
    elif winner == "other-profile-workspace":
        with pytest.raises(setup.error_type, match="another binding"):
            await setup.provisioner.prepare(binding(), **setup.kwargs)
        assert setup.gateway.providers  # Never delete another binding.
    else:
        with pytest.raises(ProviderAccessRevoked):
            await setup.provisioner.prepare(binding(), **setup.kwargs)
        assert not setup.gateway.providers


async def test_backend_rejects_other_workload_kind_before_registration(setup):
    kind = "job" if setup.kwargs["workload"].kind == "deployment" else "deployment"
    setup.kwargs["workload"] = setup.kwargs["workload"].model_copy(update={"kind": kind})
    with pytest.raises(setup.error_type, match="Expected a"):
        await setup.provisioner.prepare(binding(), **setup.kwargs)
    assert not setup.gateway.providers
    assert not setup.requests
